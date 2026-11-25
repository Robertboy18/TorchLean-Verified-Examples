#!/usr/bin/env python3
"""summarize.py: tables from the run records (work/runs/*.json) and work/goals.json.

Writes results/goals.csv (one row per goal), results/cases.csv (one row per harness case),
results/suites.csv (one row per suite), results/tables.md (the tables of the report) and prints a
short digest.  Nothing is measured here.

For one goal and one ladder the outcome is the first rung that proved it (run 0); its time is the
median, over the repetitions of that rung (runs 1 and 2, made with both tools interleaved; run 0
if a goal has none), of the in-process time of the `theorem` command (elaboration, the tactic and
the kernel check, `Elab.async false`), and the wall time is the median process time minus the
tool's median import-only time.
"""
import csv
import glob
import json
import os
import pathlib
import re
import statistics
import sys
from collections import defaultdict

SUITE = pathlib.Path(__file__).resolve().parents[1]
WORK = pathlib.Path(os.environ.get("WORK", SUITE / "work")).resolve()
RUNS = WORK / "runs"
LADDERS = {
    "lck": ("lc", ["kernel", "kernel-limits", "kernel-strong"]),
    "lcn": ("lc", ["native", "native-limits", "native-strong"]),
    "fl": ("fl", ["default", "limits", "strong"]),
}
RUNG = {"kernel": "plain", "native": "plain", "default": "plain", "kernel-limits": "limits",
        "native-limits": "limits", "limits": "limits", "kernel-strong": "strong",
        "native-strong": "strong", "strong": "strong"}
SUITE_ORDER = ["evaluation", "numerical-refinement", "ad", "integration", "krawczyk", "algebra",
               "dyadic-subdivision", "showcase", "trust-point", "trust-subdiv",
               "trust-optimization", "trust-finsum", "trust-integration", "examples"]
SUITE_TITLE = {
    "evaluation": "Evaluation", "numerical-refinement": "NumericalRefinement",
    "ad": "DomainAwareAD", "integration": "Integration", "krawczyk": "Krawczyk",
    "algebra": "AlgebraicBezout", "dyadic-subdivision": "DyadicSubdivision",
    "showcase": "bench-showcase", "trust-point": "bench-trust point",
    "trust-subdiv": "bench-trust subdiv", "trust-optimization": "bench-trust optimization",
    "trust-finsum": "bench-trust finsum", "trust-integration": "bench-trust integration",
    "examples": "examples and tests",
}


def load():
    goals = json.load(open(WORK / "goals.json"))
    examples = WORK / "examples.json"
    if examples.exists():
        goals += json.load(open(examples))
    records = defaultdict(list)
    for path in glob.glob(str(RUNS / "*__r*.json")):
        r = json.load(open(path))
        records[(r["goal"], r["tool"], r["config"])].append(r)
    baseline = {}
    for tool in ("lc", "fl"):
        walls = []
        for path in sorted(glob.glob(str(RUNS / f"baseline__{tool}-t*.json"))):
            walls += json.load(open(path))["walls_s"]
        baseline[tool] = statistics.median(walls) if walls else None
    return goals, records, baseline


def outcome(goal, records, baseline, ladder):
    """(status, rung, time_s, net_wall_s, repetitions, message) for one ladder of one goal."""
    tool, configs = LADDERS[ladder]
    accept = ("proved",) if ladder != "lcn" else ("proved", "proved-extra-axioms")
    last = None
    for config in configs:
        runs = records.get((goal["id"], tool, config))
        if not runs:
            continue
        first = sorted(runs, key=lambda r: r["rep"])[0]
        last = first
        if first["status"] in accept:
            good = [r for r in runs if r["status"] == first["status"]]
            # the repetitions (rep >= 1) ran with both tools interleaved; use them when they exist
            good = [r for r in good if r["rep"] >= 1] or good
            timed = statistics.median(r["timed_s"] for r in good if r["timed_s"] is not None)
            wall = statistics.median(r["wall_s"] for r in good)
            net = None if baseline[tool] is None else max(wall - baseline[tool], 0.0)
            return first["status"], RUNG[config], timed, net, len(good), ""
    if last is None:
        return "not-run", "", None, None, 0, ""
    return last["status"], RUNG[last["config"]], None, None, 1, last["message"]


def reason(message):
    """One line that says why a run failed."""
    text = " ".join(message.split())
    text = re.sub(r"FILE:\d+:\d+: error(\([A-Za-z.]+\))?: ", "", text)
    for key, label in [
        ("interval needs a rational", "an atom has no bound"),
        ("interval found no real constraint", "not an inequality"),
        ("internal proof-construction error", "LeanCert internal proof-construction error"),
        ("is not provable by", "LeanCert: not provable by this tactic"),
        ("enclosure rejected its", "a partial function rejected the box"),
        ("could not certify", "enclosure too wide"),
        ("maximum recursion depth", "maximum recursion depth"),
        ("deterministic) timeout", "heartbeat limit"),
        ("no strategy proved it", "no LeanCert strategy proved it"),
        ("interval could not certify", "enclosure too wide"),
        ("interval expects a real or nonnegative-real inequality", "not an inequality"),
        ("Stack overflow", "stack overflow"),
        ("Could not parse", "goal shape not recognized"),
        ("unsupported", "unsupported expression"),
    ]:
        if key in text:
            return label
    return text[:90]


def fmt(seconds):
    if seconds is None:
        return ""
    if seconds < 0.995:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 99.5:
        return f"{seconds:.1f} s"
    return f"{seconds:.0f} s"


def main():
    goals, records, baseline = load()
    rows = []
    for goal in goals:
        row = {"goal": goal["id"], "suite": goal["suite"], "shape": goal["shape"],
               "in_scope": int(goal["in_scope"]), "lc_only": int(goal["lc_only"]),
               "cases": len(goal["cases"]), "statement_chars": len(goal["statement"])}
        for ladder in LADDERS:
            if ladder == "fl" and goal["lc_only"]:
                status, rung, timed, net, reps, message = "not-stated", "", None, None, 0, ""
            else:
                status, rung, timed, net, reps, message = outcome(goal, records, baseline, ladder)
            row.update({f"{ladder}_status": status, f"{ladder}_rung": rung,
                        f"{ladder}_time_s": "" if timed is None else f"{timed:.4f}",
                        f"{ladder}_net_wall_s": "" if net is None else f"{net:.3f}",
                        f"{ladder}_reps": reps,
                        f"{ladder}_why": reason(message) if message else ""})
        rows.append(row)
    out = SUITE / "results"
    # A harvested statement stays only if LeanCert's own call, as written in its file, proves it in
    # a fresh file on the port; the others depend on their file (local definitions, options).
    dropped = [row for row in rows if row["suite"] == "examples" and
               not row["lcn_status"].startswith("proved") and row["lcn_status"] != "not-run"]
    rows = [row for row in rows if row not in dropped]
    with open(out / "examples-dropped.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["goal", "source", "lcn_status", "why", "fl_status"])
        source = {g["id"]: g["cases"][0] for g in goals}
        for row in dropped:
            writer.writerow([row["goal"], source[row["goal"]], row["lcn_status"], row["lcn_why"],
                             row["fl_status"]])
    print("examples dropped (LeanCert's own call does not prove them in a fresh file):", len(dropped))
    with open(out / "goals.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # one row per harness case
    by_goal = {row["goal"]: row for row in rows}
    cases = defaultdict(list)
    for goal in goals:
        if goal["id"] not in by_goal:
            continue
        for case in goal["cases"]:
            cases[(goal["suite"], case)].append(goal["id"])
    with open(out / "cases.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["suite", "case", "goals", "in_scope_goals", "lck_proved", "lcn_proved",
                         "fl_proved_of_in_scope"])
        for (suite, case), ids in cases.items():
            scoped = [g for g in ids if by_goal[g]["in_scope"]]
            writer.writerow([suite, case, len(ids), len(scoped),
                             sum(by_goal[g]["lck_status"] == "proved" for g in ids),
                             sum(by_goal[g]["lcn_status"].startswith("proved") for g in ids),
                             sum(by_goal[g]["fl_status"] == "proved" for g in scoped)])

    # one row per suite
    lines = []
    suite_rows = []
    for suite in SUITE_ORDER:
        mine = [row for row in rows if row["suite"] == suite]
        if not mine:
            continue
        scoped = [row for row in mine if row["in_scope"]]

        def count(group, ladder, rung=None):
            return sum(1 for row in group if row[f"{ladder}_status"] == "proved" and
                       (rung is None or row[f"{ladder}_rung"] == rung))

        def native(group):
            return sum(1 for row in group if row["lcn_status"].startswith("proved"))

        both = [row for row in scoped if row["lck_status"] == "proved" and row["fl_status"] == "proved"]
        lc_times = [float(row["lck_time_s"]) for row in both]
        fl_times = [float(row["fl_time_s"]) for row in both]
        ratios = [a / b for a, b in zip(lc_times, fl_times) if b > 0]
        suite_rows.append({
            "suite": suite, "title": SUITE_TITLE[suite],
            "cases": len({case for row in mine for case in
                          next(g for g in goals if g["id"] == row["goal"])["cases"]}),
            "goals": len(mine), "in_scope": len(scoped),
            "lck_all": count(mine, "lck"), "lcn_all": native(mine),
            "lck_scope": count(scoped, "lck"), "lck_scope_plain": count(scoped, "lck", "plain"),
            "lcn_scope": native(scoped),
            "fl_scope": count(scoped, "fl"), "fl_scope_plain": count(scoped, "fl", "plain"),
            "both": len(both),
            "lc_only": sum(1 for row in scoped if row["lck_status"] == "proved" and row["fl_status"] != "proved"),
            "fl_only": sum(1 for row in scoped if row["fl_status"] == "proved" and row["lck_status"] != "proved"),
            "neither": sum(1 for row in scoped if row["fl_status"] != "proved" and row["lck_status"] != "proved"),
            "lck_median_s": statistics.median(lc_times) if lc_times else None,
            "fl_median_s": statistics.median(fl_times) if fl_times else None,
            "lck_sum_s": sum(lc_times) if lc_times else None,
            "fl_sum_s": sum(fl_times) if fl_times else None,
            "ratio_median": statistics.median(ratios) if ratios else None,
            "ratio_geomean": statistics.geometric_mean(ratios) if ratios else None,
        })
    with open(out / "suites.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(suite_rows[0]))
        writer.writeheader()
        writer.writerows(suite_rows)

    lines.append("| Suite | Cases | Goals | In scope | LeanCert kernel (plain) | LeanCert native | "
                 "`interval` (plain) | both | only LeanCert | only `interval` | neither | "
                 "median LeanCert | median `interval` | median ratio |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in suite_rows:
        ratio = "" if s["ratio_median"] is None else f"{s['ratio_median']:.1f}x"
        lines.append(
            f"| {s['title']} | {s['cases']} | {s['goals']} | {s['in_scope']} | "
            f"{s['lck_scope']} ({s['lck_scope_plain']}) | {s['lcn_scope']} | "
            f"{s['fl_scope']} ({s['fl_scope_plain']}) | {s['both']} | {s['lc_only']} | "
            f"{s['fl_only']} | {s['neither']} | {fmt(s['lck_median_s'])} | {fmt(s['fl_median_s'])} | {ratio} |")
    lines.append("")
    lines.append("| Goal | LeanCert kernel | time | LeanCert native | time | `interval` | time |")
    lines.append("| --- | --- | ---: | --- | ---: | --- | ---: |")

    def cell(row, ladder):
        status = row[f"{ladder}_status"]
        if status == "proved":
            label = {"plain": "proved", "limits": "proved, raised limits",
                     "strong": "proved, larger options"}[row[f"{ladder}_rung"]]
        elif status == "proved-extra-axioms" and ladder == "lck":
            label = "only with a compiler axiom"
        elif status == "proved-extra-axioms":
            label = {"plain": "proved, compiler axiom", "limits": "proved, compiler axiom, raised limits",
                     "strong": "proved, compiler axiom, larger options"}[row[f"{ladder}_rung"]]
        elif status == "not-stated":
            label = "not stated"
        elif status == "not-run":
            label = "not run"
        elif status == "timeout":
            label = "no: 300 s"
        else:
            label = "no: " + row[f"{ladder}_why"]
        time = row[f"{ladder}_time_s"]
        return label, (fmt(float(time)) if time else "")

    statement = {g["id"]: g["statement"] for g in goals}
    example_lines = ["| Goal | Statement | LeanCert kernel | time | LeanCert as written | time | `interval` | time |",
                     "| --- | --- | --- | ---: | --- | ---: | --- | ---: |"]
    for suite in SUITE_ORDER:
        for row in rows:
            if row["suite"] != suite:
                continue
            a, at = cell(row, "lck")
            if suite == "examples":
                b, bt = cell(row, "lcn")
                c, ct = cell(row, "fl")
                if not row["in_scope"] and row["fl_status"] != "proved":
                    c = "outside: " + row["shape"]
                text = statement[row["goal"]].replace("|", "\\|")
                example_lines.append(f"| `{row['goal']}` | `{text}` | {a} | {at} | {b} | {bt} | {c} | {ct} |")
                continue
            b, bt = cell(row, "lcn")
            c, ct = cell(row, "fl")
            if not row["in_scope"]:
                c = "outside: " + row["shape"] if row["fl_status"] != "proved" else c
            lines.append(f"| `{row['goal']}` | {a} | {at} | {b} | {bt} | {c} | {ct} |")
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "examples-table.md").write_text("\n".join(example_lines) + "\n", encoding="utf-8")

    print("baseline import-only wall (median):", baseline)
    for s in suite_rows:
        print(f"{s['title']:26s} cases={s['cases']:3d} goals={s['goals']:3d} scope={s['in_scope']:3d} "
              f"LCk={s['lck_scope']:3d}({s['lck_scope_plain']}) LCn={s['lcn_scope']:3d} "
              f"FL={s['fl_scope']:3d}({s['fl_scope_plain']}) both={s['both']} lc_only={s['lc_only']} "
              f"fl_only={s['fl_only']} neither={s['neither']} "
              f"med LC={fmt(s['lck_median_s'])} FL={fmt(s['fl_median_s'])} "
              f"ratio={'' if s['ratio_median'] is None else round(s['ratio_median'], 1)} "
              f"| all goals: LCk={s['lck_all']} LCn={s['lcn_all']}")
    # digest: the numbers the prose quotes
    groups = {
        "seven suites": ["evaluation", "numerical-refinement", "ad", "integration", "krawczyk",
                         "algebra", "dyadic-subdivision"],
        "script benchmarks": ["showcase", "trust-point", "trust-subdiv", "trust-optimization",
                              "trust-finsum"],
        "examples and tests": ["examples"],
    }
    groups["all"] = [suite for names in list(groups.values()) for suite in names]
    digest = {}
    for name, suites in groups.items():
        mine = [row for row in rows if row["suite"] in suites]
        scoped = [row for row in mine if row["in_scope"]]
        both = [row for row in scoped if row["lck_status"] == "proved" and row["fl_status"] == "proved"]
        lc = [float(row["lck_time_s"]) for row in both]
        fl = [float(row["fl_time_s"]) for row in both]
        ratios = [a / b for a, b in zip(lc, fl)]
        native_both = [row for row in scoped if row["lcn_status"].startswith("proved") and row["fl_status"] == "proved"]
        nratios = [float(row["lcn_time_s"]) / float(row["fl_time_s"]) for row in native_both]
        digest[name] = {
            "goals": len(mine), "in_scope": len(scoped),
            "outside": len(mine) - len(scoped),
            "fl_proved": sum(row["fl_status"] == "proved" for row in scoped),
            "fl_plain": sum(row["fl_status"] == "proved" and row["fl_rung"] == "plain" for row in scoped),
            "fl_limits": sum(row["fl_status"] == "proved" and row["fl_rung"] == "limits" for row in scoped),
            "fl_strong": sum(row["fl_status"] == "proved" and row["fl_rung"] == "strong" for row in scoped),
            "lck_proved": sum(row["lck_status"] == "proved" for row in scoped),
            "lck_plain": sum(row["lck_status"] == "proved" and row["lck_rung"] == "plain" for row in scoped),
            "lck_limits": sum(row["lck_status"] == "proved" and row["lck_rung"] == "limits" for row in scoped),
            "lck_strong": sum(row["lck_status"] == "proved" and row["lck_rung"] == "strong" for row in scoped),
            "lcn_proved": sum(row["lcn_status"].startswith("proved") for row in scoped),
            "lcn_std_axioms": sum(row["lcn_status"] == "proved" for row in scoped),
            "both": len(both),
            "fl_only": sum(row["fl_status"] == "proved" and row["lck_status"] != "proved" for row in scoped),
            "lck_only": sum(row["fl_status"] != "proved" and row["lck_status"] == "proved" for row in scoped),
            "neither": sum(row["fl_status"] != "proved" and row["lck_status"] != "proved" for row in scoped),
            "fl_not_lcn": sum(row["fl_status"] == "proved" and not row["lcn_status"].startswith("proved") for row in scoped),
            "lcn_not_fl": sum(row["fl_status"] != "proved" and row["lcn_status"].startswith("proved") for row in scoped),
            "lck_sum_s": round(sum(lc), 1), "fl_sum_s": round(sum(fl), 1),
            "lck_median_s": round(statistics.median(lc), 4) if lc else None,
            "fl_median_s": round(statistics.median(fl), 4) if fl else None,
            "ratio_median": round(statistics.median(ratios), 2) if ratios else None,
            "ratio_geomean": round(statistics.geometric_mean(ratios), 2) if ratios else None,
            "fl_faster": sum(a > b for a, b in zip(lc, fl)),
            "lck_faster": sum(a < b for a, b in zip(lc, fl)),
            "native_ratio_median": round(statistics.median(nratios), 2) if nratios else None,
            "native_fl_faster": sum(r > 1 for r in nratios), "native_pairs": len(nratios),
            "outside_lck_proved": sum(row["lck_status"] == "proved" for row in mine if not row["in_scope"]),
            "outside_lcn_proved": sum(row["lcn_status"].startswith("proved") for row in mine if not row["in_scope"]),
        }
    digest["baseline_wall_s"] = baseline
    digest["examples_dropped"] = len(dropped)
    json.dump(digest, open(out / "digest.json", "w"), indent=1)
    for name in groups:
        print(name, json.dumps(digest[name]))
    pending = sum(1 for row in rows if "not-run" in (row["lck_status"], row["lcn_status"], row["fl_status"]))
    print("goals with a ladder not run yet:", pending)


if __name__ == "__main__":
    sys.exit(main())
