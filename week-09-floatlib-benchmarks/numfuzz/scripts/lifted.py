"""Second pass: the cases that stopped at a resource limit, with the limits lifted.

Usage: python3 -I scripts/lifted.py WORK

A case of `WORK/plan.json` that Lean refused at the default `maxHeartbeats` (200000) or
`maxRecDepth` (512), or that names a missing rule although the program has 40 or more operations
(smaller instances of the same program are proved), is run again as `<id>:lifted` with
`set_option maxHeartbeats 0 in set_option maxRecDepth 20000 in` before its commands. The wall-clock
limit of the process is then the only budget. Writes `WORK/plan-lifted.json`.
"""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path

LIMITS = ("maximum number of heartbeats", "maximum recursion depth", "(deterministic) timeout")
PREFIX = "set_option maxHeartbeats 0 in set_option maxRecDepth 20000 in "


def main():
    work = Path(sys.argv[1]).resolve()
    plan = json.loads((work / "plan.json").read_text())
    inventory = {row["file"][:-3]: row for row in json.loads((work / "inventory.json").read_text())}
    results = {}
    for path in (work / "run").glob("case-*.result.json"):
        row = json.loads(path.read_text())
        results[row["id"]] = row
    cases = []
    for case in plan:
        row = results.get(case["id"])
        if row is None or row["status"] == "proved" or case["method"].endswith(":numeric"):
            continue
        reason = row.get("reason", "")
        operations = inventory.get(case["program"], {}).get("operations", 0)
        budget = any(text in reason for text in LIMITS) or row["status"] == "timeout"
        large = operations >= 40 and reason.startswith("roundoff: no ")
        if not (budget or large):
            continue
        lifted = dict(case)
        lifted["id"] = case["id"] + ":lifted"
        lifted["commands"] = [PREFIX + command for command in case["commands"]]
        lifted["lifted"] = True
        cases.append(lifted)
    (work / "plan-lifted.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    print(f"{len(cases)} lifted cases")
    for case in cases:
        print("  ", case["id"])


if __name__ == "__main__":
    main()
