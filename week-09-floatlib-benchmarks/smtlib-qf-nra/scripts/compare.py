"""Join the selection, the Lean results, the SMT-COMP 2025 per-instance results and the solver runs made here.

usage: python3 -I compare.py --candidates <candidates.jsonl> --scan <scan.tsv> --smtcomp <results-sq-2025.json.gz>
           --lean-root <dir with lean/ and lean-unbounded/> --main <run dir> --rest <run dir>
           [--local <solver run dir>] --out <out dir> [key=value ...]

Each Lean run dir holds `results.jsonl` and `logs/` of `run_lean.py`: `--main` for the selection, `--rest` for
the candidates outside it.  Writes <out dir>/per-instance.csv (one row per candidate benchmark),
<out dir>/tables.md (the tables of the report) and <out dir>/run-facts.csv (the key=value pairs and the load of
the machine during the runs).  Nothing is measured here: the Lean outcomes and times are those that `run_lean.py`
recorded, the solver answers and times are the published ones or those that `run_solvers.py` recorded.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import os
import re
import statistics

SOLVERS = ["Z3-alpha", "cvc5", "Yices2", "SMT-RAT"]
OTHER = ["z3siri", "SMTInterpol", "Z3-alpha-base", "z3siri-base"]
LOCAL = ["z3", "cvc5", "yices2"]
FAMILY = {
    "20161105-Sturm-MBO": "Sturm-MBO", "20161105-Sturm-MGC": "Sturm-MGC",
    "20170501-Heizmann-UltimateInvariantSynthesis": "Heizmann", "20180501-Economics-Mulligan": "Mulligan",
    "2019-ezsmt": "ezsmt", "20200911-Pine": "Pine", "20211101-Geogebra": "Geogebra", "20220314-Uncu": "Uncu",
    "20240407-pPDA-Chiari-Pontiggia-Winkler": "pPDA", "LassoRanker": "LassoRanker",
    "UltimateAutomizer": "UltimateAutomizer", "hong": "hong", "hycomp": "hycomp", "kissing": "kissing",
    "zankl": "zankl",
}
KINDS = ["proved", "tactic", "heartbeats", "recursion-statement", "recursion-tactic", "timeout", "memory", "other"]
KIND_TEXT = {
    "proved": "proved", "tactic": "refused by the tactic", "heartbeats": "stopped at maxHeartbeats",
    "recursion-statement": "statement stopped at maxRecDepth", "recursion-tactic": "tactic stopped at maxRecDepth",
    "timeout": "timed out", "memory": "out of memory", "other": "other",
}
ERROR_AT = re.compile(r"\.lean:(\d+):(\d+): error")


def kind_of(result: dict, run_dir: str, lean_path: str) -> str:
    """One of KINDS for a row of results.jsonl.  A recursion-depth error is put in the statement when its
    position is before the line of `interval`, and in the tactic when it is on that line."""
    if result["outcome"] in ("proved", "timeout", "memory"):
        return result["outcome"]
    head = result["detail"].split(" | ")[0]
    if head == "tactic":
        return "tactic"
    if head == "Lean limit: maxHeartbeats":
        return "heartbeats"
    if head == "Lean limit: maxRecDepth":
        log = open(os.path.join(run_dir, "logs", result["lean"][:-5] + ".log"), encoding="utf-8",
                   errors="replace").read()
        m = ERROR_AT.search(log)
        tactic_line = None
        with open(lean_path, encoding="utf-8") as f:
            for number, line in enumerate(f, start=1):
                if line.strip() == "interval":
                    tactic_line = number
        if m and tactic_line and int(m.group(1)) < tactic_line:
            return "recursion-statement"
        return "recursion-tactic"
    return "other"


def med(xs, digits=1):
    return f"{statistics.median(xs):.{digits}f}" if xs else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True); ap.add_argument("--scan", required=True)
    ap.add_argument("--smtcomp", required=True); ap.add_argument("--lean-root", required=True)
    ap.add_argument("--main", required=True); ap.add_argument("--rest", required=True)
    ap.add_argument("--local", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--logs", default="", help="the directory of the job logs, for the table of the jobs")
    ap.add_argument("--trace", default="", help="how-closed.csv of trace_summary.py, for the table of the steps")
    ap.add_argument("facts", nargs="*")
    a = ap.parse_args()
    facts = dict(arg.split("=", 1) for arg in a.facts)
    rows = [json.loads(line) for line in open(a.candidates)]
    lean = {}
    for selected, run in ((True, a.main), (False, a.rest)):
        path = os.path.join(run, "results.jsonl")
        if os.path.exists(path):
            for line in open(path):
                r = json.loads(line)
                lean[(selected, r["lean"])] = (r, run)
    comp = collections.defaultdict(dict)
    comp_files = set()
    for r in json.load(gzip.open(a.smtcomp, "rt"))["results"]:
        f = r["file"]
        if f["logic"] == "QF_NRA":
            name = "/".join(f["family"]) + "/" + f["name"]
            comp[name][r["solver"]] = r
            comp_files.add(name)
    local = collections.defaultdict(dict)
    if a.local and os.path.exists(os.path.join(a.local, "results.jsonl")):
        for line in open(os.path.join(a.local, "results.jsonl")):
            r = json.loads(line)
            local[r["path"]][r["solver"]] = r

    table = []
    for r in sorted(rows, key=lambda r: r["path"]):
        selected = bool(r.get("selected"))
        row = {"path": r["path"], "family": FAMILY.get(r["family"], r["family"]), "tier": r["tier"],
               "status": r.get("status", ""), "selected": selected, "verdict": r.get("verdict", ""),
               "translated": r["outcome"] == "translated", "not_translated_reason": r.get("reason", ""),
               "reals": r.get("reals", ""), "bools": r.get("bools", ""), "nodes": r.get("nodes", ""),
               "lean_file": r.get("lean", ""), "kind": "", "outcome": "", "detail": "", "wall_s": "",
               "cpu_s": "", "maxrss_kib": "", "utc": "", "load1": ""}
        got = lean.get((selected, r.get("lean")))
        if got:
            res, run = got
            lean_path = os.path.join(a.lean_root, r["lean_dir"], r["lean"])
            row.update(kind=kind_of(res, run, lean_path), outcome=res["outcome"], detail=res["detail"],
                       wall_s=res["wall_s"], cpu_s=res["cpu_s"], maxrss_kib=res["maxrss_kib"], utc=res["utc"],
                       load1=res["load1"])
        solved = comp.get(r["path"], {})
        row["in_smtcomp_2025"] = bool(solved)
        for s in SOLVERS + OTHER:
            row[s] = solved[s]["result"] if s in solved else ""
            row[s + "_wall_s"] = round(solved[s]["wallclock_time"], 3) if s in solved else ""
            row[s + "_cpu_s"] = round(solved[s]["cpu_time"], 3) if s in solved else ""
        for s in LOCAL:
            got = local.get(r["path"], {}).get(s)
            row["here_" + s] = got["result"] if got else ""
            row["here_" + s + "_wall_s"] = got["wall_s"] if got else ""
            row["here_" + s + "_load1"] = got["load1"] if got else ""
        table.append(row)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "per-instance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    lines = []
    emit = lines.append

    # ------------------------------------------------------------------ the benchmark set and the selection
    scan = collections.defaultdict(collections.Counter)
    for r in csv.DictReader(open(a.scan), delimiter="\t"):
        if r["family"] != "meti-tarski":
            scan[FAMILY.get(r["family"], r["family"])][r["status"]] += 1
    by_family = collections.defaultdict(list)
    for r in table:
        by_family[r["family"]].append(r)
    emit("### T1. The benchmarks outside MetiTarski and the selection\n")
    emit("| family | benchmarks | `:status unsat` | `unknown`, unsat for a 2025 solver | candidates | translated | "
         "selected: every variable bounded | of these in SMT-COMP 2025 |")
    emit("|---|---:|---:|---:|---:|---:|---:|---:|")
    tot = collections.Counter()
    for fam in sorted(scan, key=str.lower):
        g = by_family.get(fam, [])
        cells = [sum(scan[fam].values()), scan[fam]["unsat"], sum(r["tier"] == "solvers" for r in g), len(g),
                 sum(r["translated"] for r in g), sum(r["selected"] and r["translated"] for r in g),
                 sum(r["selected"] and r["translated"] and r["in_smtcomp_2025"] for r in g)]
        for i, c in enumerate(cells):
            tot[i] += c
        emit(f"| {fam} | " + " | ".join(f"{c:,}" for c in cells) + " |")
    emit("| all | " + " | ".join(f"{tot[i]:,}" for i in range(7)) + " |")
    emit("")

    # ------------------------------------------------------------------------------ results per family
    def results_table(part, title, per_tier):
        emit(f"### {title}\n")
        emit("| family | translated | proved | refused | refused: by the tactic | refused: at `maxHeartbeats` | "
             "refused: statement at `maxRecDepth` | refused: tactic at `maxRecDepth` | timed out (60 s) | not run | "
             "median (max) wall of the proofs (s) |")
        emit("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        groups = collections.defaultdict(list)
        for r in part:
            groups[(r["family"], r["tier"]) if per_tier else (r["family"],)].append(r)

        def line(label, g):
            c = collections.Counter(r["kind"] or "not run" for r in g)
            walls = [r["wall_s"] for r in g if r["kind"] == "proved"]
            extra = c["memory"] + c["other"]
            note = f" (+{extra} memory or other)" if extra else ""
            refused = c["tactic"] + c["heartbeats"] + c["recursion-statement"] + c["recursion-tactic"]
            emit(f"| {label} | {len(g):,} | {c['proved']:,} | {refused:,} | {c['tactic']:,} | {c['heartbeats']:,} | "
                 f"{c['recursion-statement']:,} | {c['recursion-tactic']:,} | {c['timeout']:,}{note} | "
                 f"{c['not run']:,} | {med(walls)}" + (f" ({max(walls):.1f})" if walls else "") + " |")

        for key in sorted(groups, key=lambda k: tuple(s.lower() for s in k)):
            label = key[0] if not per_tier else key[0] + (" (status unknown)" if key[1] == "solvers" else "")
            line(label, groups[key])
        line("all", part)
        emit("")

    sel = [r for r in table if r["selected"] and r["translated"]]
    rest = [r for r in table if not r["selected"] and r["translated"]]
    results_table(sel, "T2. The selection: one call of `interval` per goal", True)
    emit("| how the selection analysis bounds the variables | goals | proved |")
    emit("|---|---:|---:|")
    for verdict, text in (("bounded", "propagation gives every real variable a finite interval"),
                          ("bounded (propagation finds no point)", "propagation alone finds the constraints "
                           "infeasible")):
        g = [r for r in sel if r["verdict"] == verdict]
        emit(f"| {text} | {len(g)} | {sum(r['kind'] == 'proved' for r in g)} |")
    emit("")

    # ------------------------------------------------------------------------------ against SMT-COMP 2025
    def against(part, title):
        both = [r for r in part if r["in_smtcomp_2025"] and r["kind"]]
        emit(f"### {title}\n")
        emit(f"{len(both):,} instances (of {len(part):,} goals) were run by SMT-COMP 2025 and measured here.\n")
        emit("| family | instances | FloatLib `interval` proved | " +
             " | ".join(f"{s} unsat" for s in SOLVERS) + " | proved, and unsat for none of the four | "
             "not proved, and unsat for all four |")
        emit("|---|---:|---:|" + "---:|" * (len(SOLVERS) + 2))
        groups = collections.defaultdict(list)
        for r in both:
            groups[r["family"]].append(r)
        for fam in sorted(groups, key=str.lower) + ["all"]:
            g = both if fam == "all" else groups[fam]
            cells = [sum(r[s] == "unsat" for r in g) for s in SOLVERS]
            alone = sum(r["kind"] == "proved" and all(r[s] != "unsat" for s in SOLVERS) for r in g)
            missed = sum(r["kind"] != "proved" and all(r[s] == "unsat" for s in SOLVERS) for r in g)
            emit(f"| {fam} | {len(g):,} | {sum(r['kind'] == 'proved' for r in g):,} | " +
                 " | ".join(f"{c:,}" for c in cells) + f" | {alone:,} | {missed:,} |")
        emit("")
        emit("| on these instances | FloatLib `interval` | " + " | ".join(SOLVERS + OTHER[:2]) + " |")
        emit("|---|---:|" + "---:|" * (len(SOLVERS) + 2))
        every = SOLVERS + OTHER[:2]
        emit("| proved / answered unsat | " + f"{sum(r['kind'] == 'proved' for r in both):,} | " +
             " | ".join(f"{sum(r[s] == 'unsat' for r in both):,}" for s in every) + " |")
        emit("| answered sat | 0 | " + " | ".join(f"{sum(r[s] == 'sat' for r in both):,}" for s in every) + " |")
        emit("| median wall time of these (s) | " + med([r["wall_s"] for r in both if r["kind"] == "proved"]) +
             " | " + " | ".join(med([r[s + "_wall_s"] for r in both if r[s] == "unsat"]) for s in every) + " |")
        emit("| proved by FloatLib, not answered unsat by the solver | | " +
             " | ".join(f"{sum(r['kind'] == 'proved' and r[s] != 'unsat' for r in both):,}" for s in every) + " |")
        emit("| answered unsat by the solver, not proved by FloatLib | | " +
             " | ".join(f"{sum(r['kind'] != 'proved' and r[s] == 'unsat' for r in both):,}" for s in every) + " |")
        emit("")

    against(sel, "T3. The selection against SMT-COMP 2025 (published per-instance results)")

    # ------------------------------------------------------------------------------ solvers run here
    name = {"z3": "Z3 " + facts.get("z3", ""), "cvc5": "cvc5 " + facts.get("cvc5", ""),
            "yices2": "Yices " + facts.get("yices", "")}
    name = {key: value.strip() for key, value in name.items()}
    here = [r for r in sel if all(r["here_" + s] for s in LOCAL)]
    if here:
        emit("### T4. The selection on this machine: `interval` and three solvers, timed the same way\n")
        emit(f"{len(here)} instances, one process per instance, 60 s of wall time each.\n")
        emit("| family | instances | FloatLib `interval` proved | " +
             " | ".join(f"{name[s]} unsat" for s in LOCAL) + " |")
        emit("|---|---:|---:|" + "---:|" * len(LOCAL))
        groups = collections.defaultdict(list)
        for r in here:
            groups[r["family"]].append(r)
        for fam in sorted(groups, key=str.lower) + ["all"]:
            g = here if fam == "all" else groups[fam]
            emit(f"| {fam} | {len(g)} | {sum(r['kind'] == 'proved' for r in g)} | " +
                 " | ".join(str(sum(r["here_" + s] == "unsat" for r in g)) for s in LOCAL) + " |")
        emit("")
        emit(f"| on the {len(here)} instances | FloatLib `interval` | " + " | ".join(name[s] for s in LOCAL) + " |")
        emit("|---|---:|" + "---:|" * len(LOCAL))
        emit("| proved / answered unsat | " + str(sum(r["kind"] == "proved" for r in here)) + " | " +
             " | ".join(str(sum(r["here_" + s] == "unsat" for r in here)) for s in LOCAL) + " |")
        emit(f"| refused (Lean reported an error within 60 s) | "
             f"{sum(r['kind'] not in ('proved', 'timeout') for r in here)} | " + " | ".join("" for s in LOCAL) + " |")
        for label, value in (("timeout at 60 s", "timeout"), ("unknown", "unknown"), ("sat", "sat"),
                             ("error", "error")):
            emit(f"| {label} | {sum(r['kind'] == 'timeout' for r in here) if value == 'timeout' else ''} | " +
                 " | ".join(str(sum(r["here_" + s] == value for r in here)) for s in LOCAL) + " |")
        times = [[r["wall_s"] for r in here if r["kind"] == "proved"]] + \
                [[r["here_" + s + "_wall_s"] for r in here if r["here_" + s] == "unsat"] for s in LOCAL]
        emit("| median wall time of these (s) | " + " | ".join(med(t, 2) for t in times) + " |")
        emit("| largest wall time of these (s) | " + " | ".join(f"{max(t):.2f}" if t else "n/a" for t in times) + " |")
        emit("| proved by FloatLib, not answered unsat by the solver | | " +
             " | ".join(str(sum(r["kind"] == "proved" and r["here_" + s] != "unsat" for r in here)) for s in LOCAL)
             + " |")
        emit("| answered unsat by the solver, not proved by FloatLib | | " +
             " | ".join(str(sum(r["kind"] != "proved" and r["here_" + s] == "unsat" for r in here)) for s in LOCAL)
             + " |")
        emit("")

    results_table(rest, "T5. For information: the translated candidates outside the selection", False)
    against(rest, "T6. For information: the other candidates against SMT-COMP 2025")
    against(sel + rest, "T7. All translated candidates against SMT-COMP 2025")

    emit("### T8. The selected instances, one by one\n")
    emit("Times are wall seconds of the whole process on this machine; `-` is no answer within 60 s. The last "
         "four columns are the published SMT-COMP 2025 answers (empty: the competition did not run the "
         "instance; `-`: no answer within 1,200 s).\n")
    emit("| benchmark | reals | Bools | nodes | `interval` | s | " + " | s | ".join(name[s] for s in LOCAL) +
         " | s | Z3-alpha | cvc5 | Yices2 | SMT-RAT |")
    emit("|---|---:|---:|---:|---|---:|---|---:|---|---:|---|---:|---|---|---|---|")
    short = {"proved": "proved", "tactic": "refused", "heartbeats": "maxHeartbeats",
             "recursion-statement": "maxRecDepth (statement)", "recursion-tactic": "maxRecDepth (tactic)",
             "timeout": "timeout", "memory": "memory", "other": "other", "": "not run"}
    for r in sel:
        cells = [f"`{r['path']}`", str(r["reals"]), str(r["bools"]), f"{r['nodes']:,}", short[r["kind"]],
                 f"{r['wall_s']:.1f}" if r["kind"] else ""]
        for s in LOCAL:
            answer = r["here_" + s]
            cells += [answer if answer in ("sat", "unsat", "unknown") else ("-" if answer else ""),
                      f"{r['here_' + s + '_wall_s']:.2f}" if answer in ("sat", "unsat", "unknown") else ""]
        for s in SOLVERS:
            cells.append("" if not r["in_smtcomp_2025"] else
                         (f"{r[s]} ({r[s + '_wall_s']:.1f} s)" if r[s] in ("sat", "unsat") else "-"))
        emit("| " + " | ".join(cells) + " |")
    emit("")

    emit("### T9. hycomp by the depth of the unrolling (the number after `global_` in the file name)\n")
    emit("| depth | goals run | variables (median) | proved | refused by the tactic | at `maxHeartbeats` | "
         "statement at `maxRecDepth` | other |")
    emit("|---:|---:|---:|---:|---:|---:|---:|---:|")
    depth = collections.defaultdict(list)
    for r in sel + rest:
        m = re.search(r"_global_(\d+)\.smt2$", r["path"])
        if r["family"] == "hycomp" and r["kind"] and m:
            depth[int(m.group(1))].append(r)
    for d in sorted(depth):
        g = depth[d]
        c = collections.Counter(r["kind"] for r in g)
        other = len(g) - c["proved"] - c["tactic"] - c["heartbeats"] - c["recursion-statement"]
        emit(f"| {d} | {len(g)} | {statistics.median(r['reals'] + r['bools'] for r in g):.0f} | {c['proved']} | "
             f"{c['tactic']} | {c['heartbeats']} | {c['recursion-statement']} | {other} |")
    emit("")

    emit("### T11. All goals by their number of variables (real and Boolean)\n")
    emit("| variables | goals run | proved | of them in the selection: run | proved |")
    emit("|---|---:|---:|---:|---:|")
    for lo, hi in ((0, 4), (5, 9), (10, 14), (15, 20), (21, 49), (50, 140), (141, None)):
        g = [r for r in sel + rest if r["kind"] and lo <= r["reals"] + r["bools"] and
             (hi is None or r["reals"] + r["bools"] <= hi)]
        label = f"{lo} to {hi}" if hi is not None else f"{lo} and more"
        emit(f"| {label} | {len(g):,} | {sum(r['kind'] == 'proved' for r in g):,} | "
             f"{sum(r['selected'] for r in g):,} | {sum(r['selected'] and r['kind'] == 'proved' for r in g):,} |")
    g = [r for r in sel + rest if r["kind"]]
    emit(f"| all | {len(g):,} | {sum(r['kind'] == 'proved' for r in g):,} | {sum(r['selected'] for r in g):,} | "
         f"{sum(r['selected'] and r['kind'] == 'proved' for r in g):,} |")
    emit("")

    if a.trace and os.path.exists(a.trace):
        emit("### T12. How the route closed each proved goal (the trace run, a diagnostic)\n")
        how = {r["lean"]: r for r in csv.DictReader(open(a.trace))}
        order = ["the ranges of the atoms alone, without a box", "a certificate of one box",
                 "certificates of more than one box", "linarith or nlinarith, without a box",
                 "a definite form, without a box", "no closing line in the trace", "not proved in the trace run"]
        count = collections.defaultdict(lambda: collections.Counter())
        fams = collections.defaultdict(lambda: collections.Counter())
        for r in sel + rest:
            if r["kind"] != "proved":
                continue
            got = how.get(r["lean_file"])
            key = got["how"] if got and got["outcome"] == "proved" else "not proved in the trace run"
            count[key]["selection" if r["selected"] else "outside"] += 1
            fams[key][r["family"]] += 1
        emit("| the goal is closed by | proofs in the selection | proofs outside it | all | families |")
        emit("|---|---:|---:|---:|---|")
        for key in order:
            if count[key]:
                emit(f"| {key} | {count[key]['selection']} | {count[key]['outside']} | "
                     f"{count[key]['selection'] + count[key]['outside']} | "
                     + ", ".join(f"{f} {n}" for f, n in sorted(fams[key].items(), key=lambda kv: (-kv[1], kv[0].lower())))
                     + " |")
        emit(f"| all | {sum(c['selection'] for c in count.values())} | {sum(c['outside'] for c in count.values())} | "
             f"{sum(c['selection'] + c['outside'] for c in count.values())} | |")
        emit("")

    # ------------------------------------------------------------------------------ the jobs
    if a.logs:
        emit("### T10. The jobs, in the order of their launch\n")
        emit("| job | launched (UTC) | elapsed | peak memory of one process (GiB) | exit | command |")
        emit("|---|---|---:|---:|---:|---|")
        jobs = []
        for entry in os.listdir(a.logs):
            if not entry.endswith(".launch") or ".part" in entry:        # a `.part` file is a copy of a log
                continue
            base = os.path.join(a.logs, entry[:-7])
            launch = open(base + ".launch").read().strip()
            when, _, tail = launch.partition(" ")
            command = tail.split("cmd=", 1)[1] if "cmd=" in tail else tail
            elapsed = rss = ""
            if os.path.exists(base + ".time"):
                text = open(base + ".time").read()
                m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text)
                elapsed = m.group(1).split(".")[0] if m else ""
                m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
                rss = f"{int(m.group(1)) / 1048576:.2f}" if m else ""
            code = open(base + ".exit").read().strip() if os.path.exists(base + ".exit") else ""
            program = command.split(" ")
            short = " ".join(os.path.basename(w) if w.startswith("/") or w.startswith("scripts/") else w
                             for w in program[:4])
            jobs.append((when, entry[:-7], elapsed, rss, code, short))
        for when, job, elapsed, rss, code, short in sorted(jobs):
            emit(f"| `{job}` | {when[11:19]} | {elapsed} | {rss} | {code} | `{short} ...` |")
        emit("")

    # ------------------------------------------------------------------------------ the machine during the runs
    def load(part, name):
        measured = [r for r in part if r["kind"]]
        if not measured:
            return
        loads = [r["load1"] for r in measured]
        facts[name + "_goals"] = len(measured)
        facts[name + "_first_utc"] = min(r["utc"] for r in measured)
        facts[name + "_last_utc"] = max(r["utc"] for r in measured)
        facts[name + "_load1_min"] = f"{min(loads):.0f}"
        facts[name + "_load1_median"] = f"{statistics.median(loads):.0f}"
        facts[name + "_load1_max"] = f"{max(loads):.0f}"
        facts[name + "_wall_sum_s"] = f"{sum(r['wall_s'] for r in measured):.0f}"
        facts[name + "_cpu_sum_s"] = f"{sum(r['cpu_s'] for r in measured):.0f}"
        facts[name + "_rss_max_gib"] = f"{max(r['maxrss_kib'] for r in measured) / 1048576:.2f}"

    load(sel, "selection")
    load(rest, "rest")
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "run-facts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for key, value in facts.items():
            w.writerow([key, value])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
