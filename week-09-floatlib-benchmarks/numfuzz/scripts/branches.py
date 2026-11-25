"""A second box for the programs with a conditional: the plan of Lean cases.

Usage: python3 -I scripts/branches.py <numfuzz examples dir> WORK

On the box [1/10, 1000] a guard that compares an input with a small constant takes one branch only
(`x > 1e-5` in the artifact's squareRoot3). For every program with a conditional this writes, into
`WORK/plan-branches.json`, the forward theorem on the box [1e-9, 1e-6] ("low") and the relative
bound on it for every rung of `closure.py`, built exactly as the cases of the first box are: the
term and the exact expression are those of `WORK/plan.json`, only the hypotheses on the inputs and
the proposed lower bound of the exact value change. The rule is by class (a conditional), not by
program.
"""
import sys
sys.dont_write_bytecode = True
import json
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import closure
import emit
import gen
import nfz

LOW = (Fraction(1, 10 ** 9), Fraction(1, 10 ** 6))


def main():
    examples, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    root = examples / "NumFuzz"
    plan = {case["id"]: case for case in json.loads((work / "plan.json").read_text())}
    inventory = json.loads((work / "inventory.json").read_text())
    cases = []
    for row in inventory:
        if row["status"] != "read" or not row.get("has_if"):
            continue
        stem = row["file"][:-3]
        first = plan[f"B:box:forward:{stem}"]
        name, inputs, outputs, notes, evaluator = nfz.load(root / row["file"], root)
        output = outputs[0]
        # The same expression as the one of the plan, or nothing is written for this program.
        if emit.lean_real_term(output) != first["term"]:
            raise SystemExit(f"{stem}: the program read now is not the one of the plan")
        used = {node[1] for node in emit.walk([output]) if node[0] == "var"}
        names = [n for n, _ in inputs if n in used]
        if names != first["inputs"]:
            raise SystemExit(f"{stem}: other inputs than in the plan")
        box = {n: LOW for n in names}
        hypotheses = " ".join(f"(hl_{n} : ({LOW[0].numerator} / {LOW[0].denominator} : ℝ) ≤ {n}) "
                              f"(hu_{n} : {n} ≤ ({LOW[1].numerator} / {LOW[1].denominator} : ℝ))"
                              for n in names)
        declaration = first["declaration"][:-len("_box")] + "_low"
        base = dict(first)
        base.update(id=f"B:low:forward:{stem}", declaration=declaration, variant="low",
                    hypotheses=hypotheses,
                    commands=[f"roundoff mixed theorem {declaration} := "
                              f"fun {first['binders']} {hypotheses} => {first['term']}"],
                    exact_lower=float(gen.exact_range(output, box)[0]),
                    box={n: [str(LOW[0]), str(LOW[1])] for n in names})
        cases.append(base)
        for rung in range(0, 9):
            relative = closure.case(base, rung)
            relative["id"] = f"B:low:relative:{stem}:r{rung}"
            relative["variant"] = "low"
            cases.append(relative)
    (work / "plan-branches.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    print(f"{len(cases)} cases -> {work / 'plan-branches.json'}")
    for case in cases:
        if case["method"] == "forward":
            print("  ", case["id"], "exact_lower", case["exact_lower"])


if __name__ == "__main__":
    main()
