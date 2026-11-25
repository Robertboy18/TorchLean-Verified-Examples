"""Cases that turn a proved route-B theorem into a numerical relative bound on the input box.

Usage: python3 -I scripts/closure.py WORK first|rest [--only SUBSTRING]

For every proved case `B:box:forward:<program>` of `WORK/run` it writes, into
`WORK/plan-closure-first.json` (rung 0) or `WORK/plan-closure-rest.json` (the later rungs of
`LADDER`, for the programs that Lean refused on rung 0), one case whose second command is

    theorem <name>_rel <inputs> <box hypotheses> : |computed - exact| ≤ B * |exact|

proved from the generated theorem. The relative part is first bounded by an expression without
the weights of sums (`nf_mono` of `ProofsNF.lean`: a sum of positive terms has at most the larger
relative error of its terms; roundings and products are monotone), and that expression by `B1`;
for a program of additions, multiplications and fused multiply-adds it is a number, compared
exactly. What `nf_mono` does not recognize (the rules of division and square root, conditionals)
is left to FloatLib's `interval` on the box, after the equalities of `NF` have put each weight
`|P| / |P + Q|` in the form `1 / (1 + Q / P)`. The absolute part is bounded by `δ` and
`m ≤ |exact|` is checked with `interval`, and `B = B1 + δ / m`.

`B1` is proposed by this script, never trusted: the worst case of the textbook propagation
(`apriori` of `gen.py`) times the factor of the rung. Lean accepts the theorem or refuses it. The
bound reported for a program is the one of the lowest rung that Lean accepted.
"""
import sys
sys.dont_write_bytecode = True
import json
import math
from fractions import Fraction
from pathlib import Path

LADDER = [1, 2, 3, 4, 5, 6, 7, 8]     # the rungs tried after rung 0
LADDER_AFTER_TIMEOUT = [3, 6, 8]
FACTORS = [Fraction(1000001, 1000000), Fraction(1001, 1000), Fraction(101, 100), Fraction(11, 10),
           Fraction(3, 2), Fraction(2), Fraction(4), Fraction(10), Fraction(100), Fraction(10 ** 4),
           Fraction(10 ** 8), Fraction(10 ** 16), Fraction(10 ** 32), Fraction(10 ** 64)]
# The share of the absolute part in the bound, by rung: B = B1 * (1 + share).
SHARES = [Fraction(1, 2 ** 30), Fraction(1, 2 ** 20), Fraction(1, 2 ** 10), Fraction(1, 2 ** 6),
          Fraction(1, 4), Fraction(1, 2), Fraction(1), Fraction(1), Fraction(1), Fraction(1),
          Fraction(1), Fraction(1), Fraction(1), Fraction(1)]


def round_down(value: Fraction, digits: int = 3) -> Fraction:
    """`value` rounded down to `digits` significant decimal digits, as an exact fraction."""
    exponent = math.floor(math.log10(float(value))) - digits + 1
    scale = Fraction(10) ** exponent
    return Fraction(math.floor(value / scale)) * scale


def real(value: Fraction) -> str:
    return f"({value.numerator} / {value.denominator} : ℝ)"


def round_up(value: Fraction, digits: int = 7) -> Fraction:
    """`value` rounded up to `digits` significant decimal digits, as an exact fraction."""
    if value == 0:
        return Fraction(0)
    exponent = math.floor(math.log10(float(value))) - digits + 1
    scale = Fraction(10) ** exponent
    return Fraction(math.ceil(value / scale)) * scale


def load_results(work: Path) -> dict:
    rows = {}
    for path in (work / "run").glob("case-*.result.json"):
        row = json.loads(path.read_text())
        rows[row["id"]] = row
    return rows


def case(base: dict, rung: int) -> dict:
    estimate = Fraction(base["apriori"])
    b1 = round_up(estimate * FACTORS[rung])
    # A lower bound of the exact value on the box, proposed from interval arithmetic by gen.py.
    lower = Fraction(base["exact_lower"])
    m = round_down(lower * Fraction(999, 1000)) if lower > 0 else Fraction(1, 2 ** 40)
    delta = b1 * m * SHARES[rung]
    bound = b1 * (1 + SHARES[rung])
    name = base["declaration"]
    inputs = base["inputs"]
    positive = "\n".join(f"  have h_{n} : (0 : ℝ) < {n} := by linarith" for n in inputs)
    arguments = " ".join(inputs) + " " + " ".join(f"hl_{n} hu_{n}" for n in inputs)
    refine = (f"refine NF.close (B1 := {real(b1)}) (δ := {real(delta)}) (m := {real(m)}) "
              f"?_ ?_ ?_ (by norm_num) (by norm_num) (by norm_num)")
    theorem = f"""set_option maxRecDepth 20000 in
set_option maxHeartbeats 4000000 in
theorem {name}_rel {base['binders']} {base['hypotheses']} :
    |{base['term']} - {base['exact']}| ≤ {real(bound)} * |{base['exact']}| := by
{positive}
  have hbound := by
    first
    | exact ({name} {arguments}).bound
    | exact ({name} {arguments} (by
        repeat' apply And.intro
        all_goals nf_goal)).bound
  refine le_trans hbound ?_
  clear hbound
  have hu16 := Model.unitRoundoffIn_pos FloatFormat.binary16 Model.IEEERoundingMode.up
  have hu32 := Model.unitRoundoffIn_pos FloatFormat.binary32 Model.IEEERoundingMode.up
  have hu64 := Model.unitRoundoffIn_pos FloatFormat.binary64 Model.IEEERoundingMode.up
  {refine}
  · first
    | (apply le_trans
       · nf_mono
       · nf_goal)
    | nf_goal
  · nf_goal
  · nf_goal"""
    return {"id": f"B:box:relative:{base['program']}:r{rung}", "declaration": f"{name}_rel",
            "commands": [*base["commands"], theorem], "program": base["program"], "route": "B",
            "variant": "box", "method": "relative", "rung": rung, "factor": str(FACTORS[rung]),
            "b1": str(b1), "bound": str(bound), "bound_float": float(bound), "m": str(m),
            "apriori": float(estimate)}


def main():
    work = Path(sys.argv[1]).resolve()
    which = sys.argv[2]                 # "first": rung 0; "rest": every later rung of LADDER
    only = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--only" else ""
    plan = json.loads((work / "plan.json").read_text())
    extra = work / "plan-lifted.json"
    if extra.exists():
        plan += json.loads(extra.read_text())
    results = load_results(work)
    cases, seen = [], set()
    for base in plan:
        if base.get("route") != "B" or base.get("variant") != "box" or base.get("method") != "forward":
            continue
        if only and only not in base["id"]:
            continue
        if results.get(base["id"], {}).get("status") != "proved":
            continue
        program = base["program"]
        if program in seen:
            continue          # a lifted copy of the same program
        seen.add(program)
        if which == "first":
            cases.append(case(base, 0))
            continue
        first = results.get(f"B:box:relative:{program}:r0", {}).get("status")
        if first == "proved":
            continue
        # A program whose rung 0 ran out of time gets three of the later rungs, the others all.
        rungs = LADDER_AFTER_TIMEOUT if first == "timeout" else LADDER
        cases.extend(case(base, rung) for rung in rungs)
    target = work / f"plan-closure-{which}.json"
    target.write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    print(f"{len(cases)} cases -> {target}")


if __name__ == "__main__":
    main()
