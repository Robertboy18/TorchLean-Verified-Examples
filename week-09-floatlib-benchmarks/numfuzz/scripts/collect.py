"""Collect the Lean results of the runs into CSV files and Markdown tables.

Usage: python3 -I scripts/collect.py WORK RESULTS

Reads `WORK/inventory.json`, the plans (`plan.json`, `plan-lifted.json`, `plan-closure-first.json`,
`plan-closure-rest.json`, `plan-loops.json`), the result of every case under `WORK/run` (the one
whose key matches the current plan and runner file) and the published numbers transcribed in
`gen.py`. Writes `RESULTS/cases.csv`, `RESULTS/programs.csv`, `RESULTS/summary.json` and
`RESULTS/tables.md`. Nothing is measured or proved here: a bound is reported only when the Lean
case that states it is `proved`; the published columns are transcriptions.
"""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen
import prove

PLANS = [("plan.json", "Proofs.lean"), ("plan-lifted.json", "Proofs.lean"),
         ("plan-closure-first.json", "ProofsNF.lean"), ("plan-closure-rest.json", "ProofsNF.lean"),
         ("plan-loops.json", "ProofsNF.lean"), ("plan-branches.json", "ProofsNF.lean"),
         ("plan-direct.json", "Proofs.lean")]
U = Fraction(1, 2 ** 52)


def failure_class(row):
    """A short class of a refusal, from Lean's message."""
    status, reason = row["status"], row.get("reason", "")
    if status == "proved":
        return ""
    if status == "timeout":
        return "wall-clock limit"
    if status == "missing":
        return "not run"
    if row.get("case", {}).get("method") == "relative":
        # A rung of the ladder: the proposed number is too small when the format constants turn a
        # side goal into a false numeral inequality (`interval` then has no inequality to read).
        if "interval expects" in reason or "linarith failed" in reason:
            return "proposed bound too small for this rung"
        if "interval could not certify" in reason:
            return "interval too wide on the box"
    if "source expansion exceeded its traversal limit" in reason:
        return "traversal limit"
    if "maximum number of heartbeats" in reason:
        return "heartbeat limit"
    if "maximum recursion depth" in reason:
        return "recursion limit"
    if "no numerical rule for" in reason:
        head = reason.split("no numerical rule for", 1)[1][:600]
        if "ofRat" in head[:40]:
            return "no rule: directed constant"
        if re.search(r"\.sqrt\s+Model\.IEEERoundingMode\.up|Model\.sqrt", head):
            return "no rule: directed square root"
        if re.search(r"\.div\b|Model\.div", head):
            return "no rule: directed division"
        if "ofRat" in head[:40]:
            return "no rule: directed constant"
        return "size: a missing rule reported for a large term"
    if "no proved arithmetic rule matches" in reason:
        return "size: no arithmetic rule matched in a large term"
    if "has no backward-error form" in reason:
        if "has no rule for it" in reason:
            return "no backward rule: directed constant, division or square root"
        if "takes no perturbation" in reason or "has an absolute error" in reason:
            return "no backward-error form: the program has none on its inputs"
        if "starts from the computed value" in reason:
            return "loop from a computed value"
        return "no backward-error form"
    if "has no closed forward bound" in reason:
        if "has no rule for it" in reason:
            return "no closed rule: directed constant, division or square root"
        return "no closed forward bound"
    if "interval needs a rational lower bound" in reason or "interval expects" in reason:
        return "numeric closure: interval cannot read the bound"
    if "interval could not certify" in reason:
        return "numeric closure: interval too wide"
    if "linarith failed" in reason:
        return "numeric closure: allowance exceeds the share"
    if "roundoff expects an absolute-error bound" in reason:
        return "numeric closure: shape of the bound"
    if status == "runner_error":
        return "process error"
    return "other"


COMPUTED = ("ofModel", ".fma", ".add ", ".mul ", ".sqrt", ".div ", "ExecFloat.Binary.add",
            "ExecFloat.Binary.mul", "ExecFloat.Binary.fma", "ExecFloat.Binary.sqrt", "ExecFloat.Binary.div")


ROUNDING = ("unitRoundoffIn", "unitRoundoffAt", "halfUlpAt")
UNDERFLOW = ("underflowErrorIn", "underflowErrorAt")


def statement_form(row):
    """What a proved closed or backward theorem states, read off its text: perturbations of the
    inputs (an existential statement); a bound whose right-hand side holds only inputs; a bound
    whose right-hand side holds computed values (an operation applied to something) only in its
    underflow term; or a bound that multiplies a unit roundoff or half an ulp by computed values.
    A computed value is counted with the format constant that precedes it most closely."""
    if row is None or row["status"] != "proved":
        return ""
    statement = row.get("statement", "")
    if "∃" in statement:
        return "perturbation"
    conclusion = statement.rsplit("→", 1)[-1]
    right = conclusion.split("≤", 1)[-1] if "≤" in conclusion else ""
    places = sorted({m.start() for token in COMPUTED for m in re.finditer(re.escape(token), right)})
    if not places:
        return "inputs"
    for place in places:
        rounding = max((right.rfind(name, 0, place) for name in ROUNDING), default=-1)
        underflow = max((right.rfind(name, 0, place) for name in UNDERFLOW), default=-1)
        if rounding >= underflow:
            return "computed values"
    return "inputs, underflow at computed values"


def sci(value, digits=3):
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits - 1}e}"


def sci_up(value, digits=5):
    """`value` (an exact fraction, or its text) rounded up to `digits` significant decimal digits,
    so that the printed number is still a bound that Lean proved."""
    import math
    value = Fraction(value)
    if value <= 0:
        return "0"
    exponent = math.floor(math.log10(float(value)))
    count = math.ceil(value / Fraction(10) ** (exponent - digits + 1))
    while count >= 10 ** digits:        # the estimate of the exponent was one too small
        exponent += 1
        count = math.ceil(value / Fraction(10) ** (exponent - digits + 1))
    while count < 10 ** (digits - 1):   # or one too large
        exponent -= 1
        count = math.ceil(value / Fraction(10) ** (exponent - digits + 1))
    text = str(count)
    return f"{text[0]}.{text[1:]}e{exponent:+03d}"


def load(work):
    """Every case of every plan with its current result (or status `missing`)."""
    cases = []
    for plan_name, runner_name in PLANS:
        path = work / plan_name
        if not path.exists():
            continue
        runner = HERE / runner_name
        runner_hash = hashlib.sha256(runner.read_bytes()).hexdigest()
        for case in json.loads(path.read_text()):
            key = prove.case_key(case, runner_hash)
            result = work / "run" / f"case-{key}.result.json"
            row = json.loads(result.read_text()) if result.exists() else {
                "id": case["id"], "status": "missing", "reason": "", "elapsed_ns": "",
                "statement": "", "axioms": []}
            row["case"] = case
            row["plan"] = plan_name
            row["runner"] = runner_name
            cases.append(row)
    return cases


def seconds(row):
    value = row.get("elapsed_ns")
    return round(value / 1e9, 3) if isinstance(value, (int, float)) else ""


def main():
    work, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((work / "inventory.json").read_text())
    sizes = json.loads((work / "loops-sizes.json").read_text())
    rows = load(work)
    by_id = {row["id"]: row for row in rows}

    # ------------------------------------------------------------------ cases.csv
    with (out / "cases.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "program", "route", "variant", "method", "limits_lifted", "status",
                         "failure_class", "lean_seconds", "process_seconds", "peak_rss_mib",
                         "started_utc", "axioms", "plan", "runner", "bound", "lean_message"])
        for row in rows:
            case = row["case"]
            writer.writerow([row["id"], case.get("program", ""), case.get("route", ""),
                             case.get("variant", ""), case.get("method", ""),
                             "yes" if case.get("lifted") else "", row["status"], failure_class(row),
                             seconds(row), row.get("process_seconds", ""),
                             round(row.get("peak_rss_kib", 0) / 1024) if row.get("peak_rss_kib") else "",
                             row.get("utc", ""), " ".join(row.get("axioms", [])), row["plan"],
                             row["runner"], case.get("bound_float", ""),
                             " ".join(row.get("reason", "").split())[:400]])

    def best(program, route, variant, method):
        """The case of a program on a route, at the default limits or with them lifted: the
        proved one if any (the default one first), else the last one tried."""
        ident = f"{route}:{variant}:{method}:{program}"
        tried = [by_id[i] for i in (ident, ident + ":lifted") if i in by_id]
        for row in tried:
            if row["status"] == "proved":
                return row
        return tried[-1] if tried else None

    def cell(row):
        if row is None:
            return "", "", "", ""
        lifted = "*" if row["case"].get("lifted") else ""
        wall = row.get("process_seconds", "")
        if row["status"] == "proved":
            return "proved" + lifted, seconds(row), "", wall
        return "no" + lifted, seconds(row), failure_class(row), wall

    # ------------------------------------------------------------------ programs.csv
    table = []
    for item in inventory:
        stem = item["file"][:-3]
        record = {"program": stem, "class": item["class"], "status": item["status"],
                  "reason": item.get("reason", ""), "operations": item.get("operations", ""),
                  "roundings": item.get("roundings", ""), "inputs": item.get("theorem_inputs", ""),
                  "outputs": item.get("outputs", ""),
                  "formats": " ".join(f"binary{b}" for b in item.get("formats", [])),
                  "division": item.get("has_div", ""), "sqrt": item.get("has_sqrt", ""),
                  "conditional": item.get("has_if", ""),
                  "fptaylor_file": item.get("fptaylor_file", ""),
                  "fptaylor_same_program": item.get("fptaylor_same_program", ""),
                  "gappa_file": item.get("gappa_file", ""),
                  "notes": " | ".join(item.get("notes", []))}
        published = {}
        for number, paper in ((3, gen.TABLE3), (4, gen.TABLE4), (5, gen.TABLE5)):
            if stem in paper:
                record["paper_table"], record["paper_row"] = number, list(paper).index(stem) + 1
        if stem in gen.TABLE3:
            ops, nf, ft, ga, tnf, tft, tga = gen.TABLE3[stem]
            published = {"paper_ops": ops, "numfuzz_bound": nf, "fptaylor_bound": ft, "gappa_bound": ga,
                         "numfuzz_seconds": tnf, "fptaylor_seconds": tft, "gappa_seconds": tga}
        elif stem in gen.TABLE4:
            ops, nf, std, tnf, tsat = gen.TABLE4[stem]
            published = {"paper_ops": ops, "numfuzz_bound": nf, "literature_bound": std,
                         "numfuzz_seconds": tnf, "satire_seconds": tsat}
        elif stem in gen.TABLE5:
            nf, tms = gen.TABLE5[stem]
            published = {"numfuzz_bound": nf, "numfuzz_seconds": str(float(tms) / 1000)}
        for key in ("paper_ops", "numfuzz_bound", "fptaylor_bound", "gappa_bound", "literature_bound",
                    "numfuzz_seconds", "fptaylor_seconds", "gappa_seconds", "satire_seconds"):
            record[key] = published.get(key, "")
        if item["status"] == "read":
            for route, variant, methods in (("B", "pos", ("forward",)), ("B", "box", ("forward",)),
                                            ("A", "up", ("forward", "closed", "backward")),
                                            ("A", "rn", ("forward", "closed", "backward"))):
                for method in methods:
                    status, time, why, wall = cell(best(stem, route, variant, method))
                    name = f"{route}_{variant}_{method}"
                    record[name], record[name + "_seconds"], record[name + "_why"] = status, time, why
                    record[name + "_process_seconds"] = wall
            # The direct spelling of a directed operation (`direct.py`), and what the closed and
            # backward theorems of either spelling state.
            for method in ("forward", "closed", "backward"):
                status, time, why, wall = cell(best(stem, "D", "up", method))
                name = f"D_up_{method}"
                record[name], record[name + "_seconds"], record[name + "_why"] = status, time, why
            for route, variant in (("A", "up"), ("A", "rn"), ("D", "up")):
                for method in ("closed", "backward"):
                    record[f"{route}_{variant}_{method}_form"] = statement_form(best(stem, route, variant, method))
            # FloatLib's own numeric closure of its FPBench pipeline: an absolute error bound on the
            # box, at the default limits only (the bound is the pipeline's proposal, checked by Lean).
            for variant in ("up", "rn"):
                for method in ("forward", "closed"):
                    row = by_id.get(f"A:{variant}:{method}:{stem}:numeric")
                    name = f"A_{variant}_{method}_absolute_bound"
                    record[name] = ""
                    if row is not None and row["status"] == "proved":
                        record[name] = str(row["case"].get("proposed_bound", ""))
            # The relative bound on the box: the lowest rung that Lean accepted.
            record["box_relative_bound"] = record["box_relative_rung"] = ""
            record["box_relative_seconds"] = record["box_relative_exact"] = ""
            record["box_relative_process_seconds"] = ""
            tried = [by_id[f"B:box:relative:{stem}:r{rung}"] for rung in range(0, 14)
                     if f"B:box:relative:{stem}:r{rung}" in by_id]
            record["box_relative_tried"] = len(tried)
            record["box_relative_why"] = ""
            for row in tried:
                if row["status"] == "proved":
                    record["box_relative_bound"] = sci_up(row["case"]["bound"], 5)
                    record["box_relative_exact"] = row["case"]["bound"]
                    record["box_relative_rung"] = row["case"]["rung"]
                    record["box_relative_seconds"] = seconds(row)
                    record["box_relative_process_seconds"] = row.get("process_seconds", "")
                    break
            else:
                if tried:
                    record["box_relative_why"] = failure_class(tried[-1])
            # The second box of a program with a conditional (`branches.py`): [1e-9, 1e-6].
            low = by_id.get(f"B:low:forward:{stem}")
            record["low_forward"] = record["low_forward_seconds"] = ""
            record["low_relative_bound"] = record["low_relative_rung"] = record["low_relative_seconds"] = ""
            if low is not None:
                record["low_forward"] = "proved" if low["status"] == "proved" else "no"
                record["low_forward_seconds"] = seconds(low)
                for rung in range(0, 14):
                    row = by_id.get(f"B:low:relative:{stem}:r{rung}")
                    if row is not None and row["status"] == "proved":
                        record["low_relative_bound"] = sci_up(row["case"]["bound"], 5)
                        record["low_relative_rung"] = row["case"]["rung"]
                        record["low_relative_seconds"] = seconds(row)
                        break
            record["apriori_script"] = sci(item.get("apriori_script"), 4)
            loop = sizes.get(stem)
            record["loop"] = record["loop_exponent"] = record["loop_bound"] = ""
            record["loop_link"] = ""
            if loop:
                weight = by_id.get(f"loop:weight:{loop['exponent']}")
                relative = by_id.get(f"loop:{loop['loop']}:relative")
                if weight and relative and weight["status"] == relative["status"] == "proved":
                    record["loop"] = loop["loop"]
                    record["loop_exponent"] = loop["exponent"]
                    record["loop_bound"] = sci_up(loop["weight"], 5)
                link = by_id.get(f"loop:link:{stem}")
                record["loop_link"] = (link["status"] if link else
                                       ("same operations, other order" if loop["loop"] == "dot" else ""))
        table.append(record)
    fields = []
    for record in table:
        for key in record:
            if key not in fields:
                fields.append(key)
    with (out / "programs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(table)

    # ------------------------------------------------------------------ summary.json
    summary = {"files": len(inventory)}
    for status in ("read", "not_a_program", "no_rounding"):
        summary[status] = sum(item["status"] == status for item in inventory)
    counts = {}
    for row in rows:
        case = row["case"]
        group = ":".join(str(case.get(k, "")) for k in ("route", "variant", "method"))
        if case.get("lifted"):
            group += ":lifted"
        entry = counts.setdefault(group, {"cases": 0, "proved": 0})
        entry["cases"] += 1
        entry["proved"] += row["status"] == "proved"
    summary["cases"] = counts
    programs = [record for record in table if record["status"] == "read"]
    summary["programs"] = len(programs)
    for name in ("B_pos_forward", "B_box_forward", "A_up_forward", "A_up_closed", "A_up_backward",
                 "A_rn_forward", "A_rn_closed", "A_rn_backward"):
        summary[name] = sum(record.get(name, "").startswith("proved") for record in programs)
    for name in ("D_up_forward", "D_up_closed", "D_up_backward"):
        summary[name] = sum(record.get(name, "").startswith("proved") for record in programs)
    summary["D_up_programs"] = sum(bool(record.get("D_up_forward")) for record in programs)
    forms = {}
    for route, variant in (("A", "up"), ("A", "rn"), ("D", "up")):
        for method in ("closed", "backward"):
            name = f"{route}_{variant}_{method}_form"
            count = {}
            for record in programs:
                if record.get(name):
                    count[record[name]] = count.get(record[name], 0) + 1
            forms[name] = count
    summary["statement_forms"] = forms
    summary["box_relative"] = sum(bool(record.get("box_relative_bound")) for record in programs)
    summary["box_relative_rung0"] = sum(record.get("box_relative_rung") == 0 for record in programs)
    summary["low_box_programs"] = sum(bool(record.get("low_forward")) for record in programs)
    summary["low_box_forward"] = sum(record.get("low_forward") == "proved" for record in programs)
    summary["low_box_relative"] = sum(bool(record.get("low_relative_bound")) for record in programs)
    summary["low_box_relative_rung0"] = sum(record.get("low_relative_rung") == 0 for record in programs)
    summary["loop_rows"] = sum(bool(record.get("loop_bound")) for record in programs)
    classes = {}
    for row in rows:
        if row["status"] != "proved":
            key = failure_class(row)
            classes[key] = classes.get(key, 0) + 1
    summary["failure_classes"] = dict(sorted(classes.items(), key=lambda kv: -kv[1]))
    proved = [row for row in rows if row["status"] == "proved"]
    summary["axioms"] = sorted({axiom for row in proved for axiom in row.get("axioms", [])})
    summary["lean_seconds_total"] = round(sum(seconds(row) or 0 for row in rows), 1)
    summary["process_seconds_total"] = round(sum(row.get("process_seconds", 0) or 0 for row in rows), 1)
    summary["peak_rss_mib_max"] = round(max((row.get("peak_rss_kib", 0) or 0) for row in rows) / 1024)
    stamps = sorted(row["utc"] for row in rows if row.get("utc"))
    summary["first_case_utc"], summary["last_case_utc"] = (stamps[0], stamps[-1]) if stamps else ("", "")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    # ------------------------------------------------------------------ tables.md
    index = {record["program"]: record for record in table}
    lines = []

    def ratio(mine, theirs):
        try:
            return f"{float(mine) / float(theirs):.2f}"
        except (ValueError, ZeroDivisionError):
            return ""

    def mark(record, name):
        value = record.get(name, "")
        if value.startswith("proved"):
            return "yes" + ("*" if value.endswith("*") else "")
        return "no" + ("*" if value.endswith("*") else "") if value else ""

    def routes(record):
        return " ".join(mark(record, f"A_{variant}_{method}") or "-"
                        for variant in ("up", "rn") for method in ("forward", "closed", "backward"))

    lines.append("### Table 1. The 17 small benchmarks (the paper's Table 3)\n")
    lines.append("| program | ops | FloatLib, box | NumFuzz | FPTaylor | Gappa | FloatLib / NumFuzz | "
                 "FloatLib / best other | forward (s) | forward and relative (s) | NumFuzz (s) | FPTaylor (s) | "
                 "Gappa (s) | IEEE route: up f c b, nearest f c b |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for stem in gen.TABLE3:
        r = index[stem]
        others = [float(v) for v in (r["numfuzz_bound"], r["fptaylor_bound"], r["gappa_bound"])
                  if v not in ("", "fail")]
        mine = r.get("box_relative_bound", "")
        lines.append(f"| {stem} | {r['paper_ops']} | {mine or 'no'} | {r['numfuzz_bound']} | "
                     f"{r['fptaylor_bound']} | {r['gappa_bound']} | {ratio(mine, r['numfuzz_bound'])} | "
                     f"{ratio(mine, min(others)) if mine else ''} | {r.get('B_box_forward_seconds', '')} | "
                     f"{r.get('box_relative_seconds', '')} | {r['numfuzz_seconds']} | "
                     f"{r['fptaylor_seconds'] or '-'} | {r['gappa_seconds']} | {routes(r)} |")
    lines.append("")
    lines.append("### Table 2. The 9 large benchmarks (the paper's Table 4)\n")
    lines.append("| program | ops | straight line: reals, positive | reals, box | IEEE up f c b | "
                 "IEEE nearest f c b | FloatLib, box | FloatLib, loop form | NumFuzz | literature | "
                 "loop = program | straight line (s) | NumFuzz (s) | Satire (s) |")
    lines.append("| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    for stem in gen.TABLE4:
        r = index[stem]
        up = " ".join(mark(r, f"A_up_{m}") or "-" for m in ("forward", "closed", "backward"))
        rn = " ".join(mark(r, f"A_rn_{m}") or "-" for m in ("forward", "closed", "backward"))
        lines.append(f"| {stem.split('/')[-1]} | {r['paper_ops']} | {mark(r, 'B_pos_forward')} | "
                     f"{mark(r, 'B_box_forward')} | {up} | {rn} | {r.get('box_relative_bound') or '-'} | "
                     f"{r.get('loop_bound') or '-'} | {r['numfuzz_bound']} | {r['literature_bound'] or '-'} | "
                     f"{r.get('loop_link') or '-'} | {r.get('B_pos_forward_seconds', '')} | "
                     f"{r['numfuzz_seconds']} | {r['satire_seconds'] or '-'} |")
    lines.append("")
    lines.append("### Table 3. The 4 conditional benchmarks (the paper's Table 5)\n")
    lines.append("| program | FloatLib, box [0.1, 1000] | FloatLib, box [1e-9, 1e-6] | NumFuzz | "
                 "FloatLib / NumFuzz | reals, positive | reals, box | forward (s) | forward and relative (s) | "
                 "second box: forward and relative (s) | NumFuzz (s) | IEEE route: up f c b, nearest f c b |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for stem in gen.TABLE5:
        r = index[stem]
        mine = r.get("box_relative_bound", "")
        lines.append(f"| {stem.split('/')[-1]} | {mine or 'no'} | {r.get('low_relative_bound') or 'no'} | "
                     f"{r['numfuzz_bound']} | "
                     f"{ratio(mine, r['numfuzz_bound'])} | {mark(r, 'B_pos_forward')} | "
                     f"{mark(r, 'B_box_forward')} | {r.get('B_box_forward_seconds', '')} | "
                     f"{r.get('box_relative_seconds', '')} | {r.get('low_relative_seconds', '')} | "
                     f"{r['numfuzz_seconds']} | {routes(r)} |")
    lines.append("")
    lines.append("### Table 4. Loop forms at the sizes of the benchmarks\n")
    lines.append("| program | loop | weight | FloatLib (Lean) | NumFuzz | FPTaylor | Gappa | literature | "
                 "loop = program (`rfl`) |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for stem, loop in sizes.items():
        r = index.get(stem)
        if r is None or not r.get("loop_bound"):
            continue
        lines.append(f"| {stem} | "
                     f"{loop['loop']} | (1+u)^{loop['exponent']} - 1 | {r['loop_bound']} | "
                     f"{r['numfuzz_bound'] or '-'} | {r['fptaylor_bound'] or '-'} | {r['gappa_bound'] or '-'} | "
                     f"{r['literature_bound'] or '-'} | {r['loop_link'] or '-'} |")
    lines.append("")
    lines.append("### Table 6. Every program of the artifact\n")
    lines.append("| program | ops | formats | reals: positive | reals: box | forward (s) | "
                 "relative bound on the box | rung | forward and relative (s) | "
                 "IEEE up f c b | IEEE nearest f c b | why not (first refusal) |")
    lines.append("| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |")
    for r in table:
        if r["status"] != "read":
            continue
        up = " ".join(mark(r, f"A_up_{m}") or "-" for m in ("forward", "closed", "backward"))
        rn = " ".join(mark(r, f"A_rn_{m}") or "-" for m in ("forward", "closed", "backward"))
        why = next((r[k] for k in ("B_pos_forward_why", "B_box_forward_why", "box_relative_why",
                                    "A_up_forward_why", "A_up_closed_why", "A_up_backward_why",
                                    "A_rn_forward_why", "A_rn_closed_why", "A_rn_backward_why")
                    if r.get(k)), "")
        lines.append(f"| {r['program']} | {r['operations']} | {r['formats'].replace('binary', '')} | "
                     f"{mark(r, 'B_pos_forward')} | {mark(r, 'B_box_forward')} | "
                     f"{r.get('B_pos_forward_seconds', '')} | "
                     f"{r.get('box_relative_bound') or '-'} | {r.get('box_relative_rung')} | "
                     f"{r.get('box_relative_seconds') or '-'} | {up} | {rn} | {why} |")
    lines.append("")
    lines.append("### Table 5. Closed and backward analysis in `toPositive`, by the spelling of the operation\n")
    lines.append("| program | ops | pipeline spelling f c b | its closed theorem bounds by | its backward theorem "
                 "states | direct spelling f c b | its closed theorem bounds by | its backward theorem states | "
                 "direct: forward, closed, backward (s) |")
    lines.append("| --- | ---: | --- | --- | --- | --- | --- | --- | --- |")
    wording = {"perturbation": "perturbations of the inputs", "inputs": "the inputs",
               "inputs, underflow at computed values": "the inputs (underflow term at computed values)",
               "computed values": "computed values", "": "-"}
    backward_wording = {"perturbation": "perturbations of the inputs", "inputs": "a forward bound in the inputs",
                        "inputs, underflow at computed values": "a forward bound in the inputs",
                        "computed values": "a forward bound at computed values", "": "-"}
    for r in table:
        if r["status"] != "read" or not r.get("D_up_forward"):
            continue
        pipeline = " ".join(mark(r, f"A_up_{m}") or "-" for m in ("forward", "closed", "backward"))
        direct = " ".join(mark(r, f"D_up_{m}") or "-" for m in ("forward", "closed", "backward"))
        times = ", ".join(str(r.get(f"D_up_{m}_seconds", "")) for m in ("forward", "closed", "backward"))
        lines.append(f"| {r['program']} | {r['operations']} | {pipeline} | {wording[r['A_up_closed_form']]} | "
                     f"{backward_wording[r['A_up_backward_form']]} | {direct} | {wording[r['D_up_closed_form']]} | "
                     f"{backward_wording[r['D_up_backward_form']]} | {times} |")
    lines.append("")
    (out / "tables.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
