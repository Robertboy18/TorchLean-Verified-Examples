"""Loop forms of the artifact's long sums, Horner forms and dot products: the plan of Lean cases.

Usage: python3 -I scripts/loops.py WORK

Writes `WORK/plan-loops.json`. Each program is one Lean definition with a symbolic length
(`nfSum`, `nfHorner`, `nfDotFrom`); `roundoff` proves its forward, closed and backward theorems
once, for every length, and this suite's corollaries (proved here from the closed theorem, with the
lemmas of `ProofsNF.lean`) put the closed bound in the textbook form: one weight
`(1 + u)^k - 1` times the exact result at nonnegative inputs. `roundoff closed` does not follow a
loop that starts from a computed value, so the dot product is the loop from an input value, and
the corollary puts the first product in its place with the closed theorem of one product. The cases `weight` check the
decimal value of that weight at the sizes of the benchmarks, and the cases `link` check with `rfl`
that the loop at a benchmark's size is the straight-line program that FloatLib's FPBench pipeline
makes of the artifact's file (the term is read from `WORK/plan.json`).

The decimal bounds are proposed by this script (exact rational arithmetic, rounded up to five
significant digits); Lean accepts or refuses them.
"""
import sys
sys.dont_write_bytecode = True
import json
import math
import re
from fractions import Fraction
from pathlib import Path

U = "Model.unitRoundoffIn FloatFormat.binary64 Model.IEEERoundingMode.up"
ETA = "Model.underflowErrorIn FloatFormat.binary64 Model.IEEERoundingMode.up"
R = "ExecFloat.Binary.toReal"
FIN = "ExecFloat.Binary.isFinite"
MAX = "(Model.posMaxFinite FloatFormat.binary64).toReal"
UP = "Model.IEEERoundingMode.up"

DEF_SUM = f"""/-- Serial sum of a list after a first value, each addition rounded toward +∞. -/
def nfSum (x0 : Binary64) (xs : List Binary64) : Binary64 :=
  xs.foldl (fun s a => ExecFloat.Binary.add s a (mode := .up)) x0"""
ARG_SUM = "fun (x0 : Binary64) (xs : List Binary64) => ExecFloat.Binary.toReal (nfSum x0 xs)"
HYP_SUM = f"""(hfin : List.AllSteps (fun s a => ExecFloat.Binary.add s a {UP})
      (fun s a =>
        |{R} s + {R} a| ≤ {MAX} ∧
          {FIN} s = true ∧ {FIN} a = true)
      x0 xs)"""
REL_SUM = f"""/-- Serial sum of nonnegative values: relative error at most `(1 + u)^n - 1` for `n` additions. -/
theorem nfSum_rel (x0 : Binary64) (xs : List Binary64)
    (hx0 : 0 ≤ {R} x0) (hxs : ∀ a ∈ xs, 0 ≤ {R} a)
    {HYP_SUM} :
    |{R} (nfSum x0 xs) - List.foldl (fun s a => s + {R} a) ({R} x0) xs| ≤
      ((1 + {U}) ^ xs.length - 1) * List.foldl (fun s a => s + {R} a) ({R} x0) xs := by
  refine le_trans (nfSum_closed x0 xs hfin) ?_
  have hu : (1 : ℝ) ≤ 1 + {U} := by
    linarith [Model.unitRoundoffIn_pos FloatFormat.binary64 {UP}]
  rw [NF.foldl_add (fun a : Binary64 => {R} a) xs,
    NF.sum_map_eq_sum_fin (fun a : Binary64 => {R} a) xs, mul_add, Finset.mul_sum,
    abs_of_nonneg hx0]
  refine add_le_add (by rw [mul_comm]) (Finset.sum_le_sum fun k _ => ?_)
  have := k.isLt
  rw [abs_of_nonneg (hxs _ (List.getElem_mem _)), mul_comm]
  exact mul_le_mul_of_nonneg_right
    (sub_le_sub_right (pow_le_pow_right₀ hu (by omega)) 1) (hxs _ (List.getElem_mem _))"""

DEF_HORNER = f"""/-- Horner's rule with one fused multiply-add per coefficient, rounded toward +∞. -/
def nfHorner (lead : Binary64) (coefficients : List Binary64) (x : Binary64) : Binary64 :=
  coefficients.foldl (fun value c => ExecFloat.Binary.fma value x c (mode := .up)) lead"""
ARG_HORNER = ("fun (lead : Binary64) (coefficients : List Binary64) (x : Binary64) => "
              "ExecFloat.Binary.toReal (nfHorner lead coefficients x)")
HYP_HORNER = f"""(hfin : List.AllSteps (fun value c => ExecFloat.Binary.fma value x c {UP})
      (fun value c =>
        |{R} value * {R} x + {R} c| ≤ {MAX} ∧
          {FIN} value = true ∧ {FIN} x = true ∧ {FIN} c = true)
      lead coefficients)"""
MAG_HORNER = f"""/-- Horner's rule: error at most `(1 + u)^n - 1` times the polynomial of the magnitudes, plus
the underflow allowance of the `n` fused operations. -/
theorem nfHorner_magnitudes (lead : Binary64) (coefficients : List Binary64) (x : Binary64)
    {HYP_HORNER} :
    |{R} (nfHorner lead coefficients x) -
        List.foldl (fun value c => value * {R} x + {R} c) ({R} lead) coefficients| ≤
      ((1 + {U}) ^ coefficients.length - 1) *
          List.foldl (fun value c => value * |{R} x| + |{R} c|) |{R} lead| coefficients +
        {ETA} * ∑ k ∈ Finset.range coefficients.length, (|{R} x| * (1 + {U})) ^ k := by
  refine le_trans (nfHorner_closed lead coefficients x hfin) (add_le_add ?_ le_rfl)
  have hu : (1 : ℝ) ≤ 1 + {U} := by
    linarith [Model.unitRoundoffIn_pos FloatFormat.binary64 {UP}]
  have key := NF.foldl_horner_le |{R} x|
    (fun k : Fin coefficients.length => |{R} coefficients[(k : Nat)]|)
    (fun k : Fin coefficients.length => (1 + {U}) ^ (coefficients.length - 1 - (k : Nat) + 1) - 1)
    ((1 + {U}) ^ coefficients.length - 1)
    (abs_nonneg _) (fun k => abs_nonneg _)
    (fun k => sub_le_sub_right (pow_le_pow_right₀ hu (by have := k.isLt; omega)) 1)
    (List.finRange coefficients.length) _ _
    (le_refl (|{R} lead| * ((1 + {U}) ^ coefficients.length - 1)))
  refine le_trans key (le_of_eq ?_)
  rw [NF.foldl_horner_scale]
  congr 1
  conv_rhs => rw [← List.map_getElem_finRange coefficients]
  rw [List.foldl_map]"""
REL_HORNER = f"""/-- Horner's rule at nonnegative coefficients and a nonnegative argument: relative error at
most `(1 + u)^n - 1`, plus the underflow allowance. -/
theorem nfHorner_rel (lead : Binary64) (coefficients : List Binary64) (x : Binary64)
    (hx : 0 ≤ {R} x) (hlead : 0 ≤ {R} lead) (hc : ∀ c ∈ coefficients, 0 ≤ {R} c)
    {HYP_HORNER} :
    |{R} (nfHorner lead coefficients x) -
        List.foldl (fun value c => value * {R} x + {R} c) ({R} lead) coefficients| ≤
      ((1 + {U}) ^ coefficients.length - 1) *
          List.foldl (fun value c => value * {R} x + {R} c) ({R} lead) coefficients +
        {ETA} * ∑ k ∈ Finset.range coefficients.length, ({R} x * (1 + {U})) ^ k := by
  have h := nfHorner_magnitudes lead coefficients x hfin
  rw [abs_of_nonneg hx, abs_of_nonneg hlead] at h
  rw [List.foldl_ext (fun value c => value * {R} x + |{R} c|)
    (fun value c => value * {R} x + {R} c) _
    (fun value c hmem => by rw [abs_of_nonneg (hc c hmem)])] at h
  exact h"""

DEF_DOT = f"""/-- Dot product after a first value: each product is added to the sum so far, every operation
rounded toward +∞. With the first product as the first value this is the dot product of `n + 1`
pairs, summed from the last pair to the first as the artifact's fold does. -/
def nfDotFrom {{n : Nat}} (s0 : Binary64) (x y : Fin n → Binary64) : Binary64 :=
  Fin.foldl n (fun s j =>
    ExecFloat.Binary.add (ExecFloat.Binary.mul (x j) (y j) (mode := .up)) s (mode := .up)) s0"""
ARG_DOT = ("fun {n : Nat} (s0 : Binary64) (x y : Fin n → Binary64) => "
           "ExecFloat.Binary.toReal (nfDotFrom s0 x y)")
MAX_DOT = "Model.toReal (Model.posMaxFinite FloatFormat.binary64)"
MUL_CLOSED = ("roundoff closed theorem nfMul_closed := fun (a b : Binary64) => "
              "ExecFloat.Binary.toReal (ExecFloat.Binary.mul a b (mode := .up))")
FOLD_DOT = """/-- A fold that adds to `f j` the value so far is the start plus the sum of the `f j`. -/
theorem nf_foldl_add_left {n : Nat} (f : Fin n → ℝ) (a : ℝ) :
    (List.finRange n).foldl (fun s j => f j + s) a = a + ∑ j, f j := by
  have h : (fun (s : ℝ) (j : Fin n) => f j + s) = fun s j => s + f j := by
    funext s j; ring
  rw [h, NF.foldl_add_fin]"""
FIRST_DOT = """/-- The first product of a dot product: its rounding counts once more than the additions. -/
theorem nf_dot_first {u η F P p S V A T : ℝ} {n : ℕ} (hu : 0 ≤ u)
    (h1 : |F - (P + S)| ≤ |P| * ((1 + u) ^ n - 1) + V + T)
    (h2 : |P - p| ≤ |p| * u + η)
    (h3 : V ≤ ((1 + u) ^ (n + 1) - 1) * A) :
    |F - (p + S)| ≤ ((1 + u) ^ (n + 1) - 1) * (|p| + A) + (η * (1 + u) ^ n + T) := by
  have hW : 0 ≤ (1 + u) ^ n - 1 := sub_nonneg.mpr (one_le_pow₀ (by linarith))
  have hP : |P| ≤ (1 + u) * |p| + η := by
    have h := abs_sub_abs_le_abs_sub P p
    linarith
  have h4 : |P| * ((1 + u) ^ n - 1) ≤ ((1 + u) * |p| + η) * ((1 + u) ^ n - 1) :=
    mul_le_mul_of_nonneg_right hP hW
  have h5 : |F - (p + S)| ≤ |F - (P + S)| + |P - p| := by
    have h : F - (p + S) = (F - (P + S)) + (P - p) := by ring
    rw [h]
    exact abs_add_le _ _
  calc |F - (p + S)| ≤ |F - (P + S)| + |P - p| := h5
    _ ≤ (|P| * ((1 + u) ^ n - 1) + V + T) + (|p| * u + η) := add_le_add h1 h2
    _ ≤ (((1 + u) * |p| + η) * ((1 + u) ^ n - 1) + ((1 + u) ^ (n + 1) - 1) * A + T) +
          (|p| * u + η) := by linarith
    _ = ((1 + u) ^ (n + 1) - 1) * (|p| + A) + (η * (1 + u) ^ n + T) := by ring"""
HYP_DOT = f"""(hab : |{R} a * {R} b| ≤ {MAX_DOT} ∧ {FIN} a = true ∧ {FIN} b = true)
    (hfin : List.AllSteps
      (fun s j => ExecFloat.Binary.add (ExecFloat.Binary.mul (x j) (y j) {UP}) s {UP})
      (fun s j =>
        |{R} (ExecFloat.Binary.mul (x j) (y j) {UP}) + {R} s| ≤ {MAX_DOT} ∧
          {FIN} (ExecFloat.Binary.mul (x j) (y j) {UP}) = true ∧
            {FIN} s = true ∧
              |{R} (x j) * {R} (y j)| ≤ {MAX_DOT} ∧
                {FIN} (x j) = true ∧ {FIN} (y j) = true)
      (ExecFloat.Binary.mul a b {UP}) (List.finRange n))"""
TAIL_DOT = (f"({ETA} * (1 + {U}) ^ n +\n          {ETA} * (1 + {U}) * "
            f"∑ k ∈ Finset.range n, (1 + {U}) ^ k)")
MAG_DOT = f"""/-- Dot product of `n + 1` pairs, the sum starting from the first product: error at most
`(1 + u)^(n + 1) - 1` times `|a| |b| + |x|ᵀ|y|`, plus the underflow allowance of the products. -/
theorem nfDot_magnitudes {{n : Nat}} (a b : Binary64) (x y : Fin n → Binary64)
    {HYP_DOT} :
    |{R} (nfDotFrom (ExecFloat.Binary.mul a b {UP}) x y) -
        ({R} a * {R} b + ∑ j, {R} (x j) * {R} (y j))| ≤
      ((1 + {U}) ^ (n + 1) - 1) * (|{R} a * {R} b| + ∑ j, |{R} (x j)| * |{R} (y j)|) +
        {TAIL_DOT} := by
  have h1 := nfDotFrom_closed (ExecFloat.Binary.mul a b {UP}) x y hfin
  rw [nf_foldl_add_left (fun j => {R} (x j) * {R} (y j))] at h1
  have h2 : |{R} (ExecFloat.Binary.mul a b {UP}) - {R} a * {R} b| ≤
      |{R} a * {R} b| * {U} + {ETA} := by
    have h := nfMul_closed a b hab
    rw [abs_mul]
    linarith [h, mul_right_comm |{R} a| ({U}) |{R} b|]
  have hu : (1 : ℝ) ≤ 1 + {U} := by
    linarith [Model.unitRoundoffIn_pos FloatFormat.binary64 {UP}]
  refine nf_dot_first (Model.unitRoundoffIn_pos FloatFormat.binary64 {UP}).le h1 h2 ?_
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun j _ => ?_
  have := j.isLt
  have hW : (1 + {U}) ^ (n - 1 - (j : Nat) + 2) - 1 ≤ (1 + {U}) ^ (n + 1) - 1 :=
    sub_le_sub_right (pow_le_pow_right₀ hu (by omega)) 1
  calc _ = ((1 + {U}) ^ (n - 1 - (j : Nat) + 2) - 1) * (|{R} (x j)| * |{R} (y j)|) := by ring
    _ ≤ _ := mul_le_mul_of_nonneg_right hW (by positivity)"""
REL_DOT = f"""/-- Dot product of `n + 1` pairs of nonnegative entries: relative error at most
`(1 + u)^(n + 1) - 1`, plus the underflow allowance. -/
theorem nfDot_rel {{n : Nat}} (a b : Binary64) (x y : Fin n → Binary64)
    (ha : 0 ≤ {R} a) (hb : 0 ≤ {R} b) (hx : ∀ j, 0 ≤ {R} (x j)) (hy : ∀ j, 0 ≤ {R} (y j))
    {HYP_DOT} :
    |{R} (nfDotFrom (ExecFloat.Binary.mul a b {UP}) x y) -
        ({R} a * {R} b + ∑ j, {R} (x j) * {R} (y j))| ≤
      ((1 + {U}) ^ (n + 1) - 1) * ({R} a * {R} b + ∑ j, {R} (x j) * {R} (y j)) +
        {TAIL_DOT} := by
  have h := nfDot_magnitudes a b x y hab hfin
  have hs : ∑ j, |{R} (x j)| * |{R} (y j)| = ∑ j, {R} (x j) * {R} (y j) :=
    Finset.sum_congr rfl (fun j _ => by rw [abs_of_nonneg (hx j), abs_of_nonneg (hy j)])
  rw [hs, abs_of_nonneg (mul_nonneg ha hb)] at h
  exact h"""
EXTRA = {"sum": [], "horner": [], "dot": [MUL_CLOSED, FOLD_DOT, FIRST_DOT]}

# program -> (loop, exponent of the weight, how the loop is the program)
SIZES = {
    "test02_sum8": ("sum", 7), "large_benchmarks/serial_sum": ("sum", 1023),
    "Horner2": ("horner", 2), "Horner5": ("horner", 5), "Horner10": ("horner", 10),
    "Horner20": ("horner", 20), "large_benchmarks/Horner10": ("horner", 10),
    "large_benchmarks/Horner20": ("horner", 20), "large_benchmarks/Horner50": ("horner", 50),
    "large_benchmarks/Horner75": ("horner", 75), "large_benchmarks/Horner100": ("horner", 100),
    "large_benchmarks/dotprods/dotprod4": ("dot", 4), "large_benchmarks/dotprods/dotprod8": ("dot", 8),
    "large_benchmarks/dotprods/dotprod16": ("dot", 16),
    "large_benchmarks/dotprods/dotprod64": ("dot", 64),
    "large_benchmarks/dotprods/dotprod128": ("dot", 128),
    "large_benchmarks/mat_mul_4": ("dot", 4), "large_benchmarks/mat_mul_8": ("dot", 8),
    "large_benchmarks/mat_mul_16": ("dot", 16), "large_benchmarks/mat_mul_64": ("dot", 64),
    "large_benchmarks/mat_mul_128": ("dot", 128),
}


def weight(exponent: int, digits: int = 5) -> Fraction:
    """`(1 + 2^-52)^exponent - 1` rounded up to `digits` significant decimal digits."""
    value = (1 + Fraction(1, 2 ** 52)) ** exponent - 1
    power = math.floor(math.log10(float(value))) - digits + 1
    scale = Fraction(10) ** power
    return Fraction(math.ceil(value / scale)) * scale


def weight_case(exponent: int) -> dict:
    bound = weight(exponent)
    name = f"nf_weight_{exponent}"
    theorem = (f"set_option exponentiation.threshold 4096 in\n"
               f"theorem {name} : (1 + {U}) ^ {exponent} - 1 ≤ "
               f"({bound.numerator} / {bound.denominator} : ℝ) := by\n"
               f"  rw [NF.u64_eq]; norm_num")
    return {"id": f"loop:weight:{exponent}", "declaration": name, "commands": [theorem],
            "route": "loop", "method": "weight", "exponent": exponent,
            "bound": str(bound), "bound_float": float(bound)}


def link_case(program, kind, plan):
    base = plan.get(f"A:up:forward:{program}")
    if base is None:
        return None
    command = base["commands"][0]
    match = re.match(r"roundoff mixed theorem \S+ := fun (.*?) => ExecFloat\.Binary\.toReal \((.*)\)$",
                     command, re.S)
    if match is None:
        return None
    binders, body = match.group(1), match.group(2)
    names = re.findall(r"\((x\d+) : Binary64\)", binders)
    if kind == "sum":
        loop = f"nfSum {names[0]} [{', '.join(names[1:])}]"
        definition = DEF_SUM
    elif kind == "dot":
        # The artifact's fold adds from the last pair to the first: p0 + (p1 + (... + p_last)).
        half = len(names) // 2
        left, right = names[:half], names[half:]
        loop = (f"nfDotFrom (ExecFloat.Binary.mul {left[-1]} {right[-1]} (mode := .up)) "
                f"![{', '.join(reversed(left[:-1]))}] ![{', '.join(reversed(right[:-1]))}]")
        definition = DEF_DOT
    else:
        coefficients = names[:-1]
        loop = f"nfHorner {coefficients[-1]} [{', '.join(reversed(coefficients[:-1]))}] {names[-1]}"
        definition = DEF_HORNER
    name = "nf_link_" + re.sub(r"[^A-Za-z0-9_]", "_", program)
    theorem = (f"set_option maxRecDepth 1000000 in\nset_option maxHeartbeats 0 in\n"
               f"theorem {name} {binders} :\n    {loop} = ({body}) := rfl")
    return {"id": f"loop:link:{program}", "declaration": name, "commands": [definition, theorem],
            "route": "loop", "method": "link", "program": program, "loop": kind}


def main():
    work = Path(sys.argv[1]).resolve()
    plan = {case["id"]: case for case in json.loads((work / "plan.json").read_text())}
    cases = []
    for kind, definition, argument, magnitudes, relative in (
            ("sum", DEF_SUM, ARG_SUM, None, REL_SUM),
            ("horner", DEF_HORNER, ARG_HORNER, MAG_HORNER, REL_HORNER),
            ("dot", DEF_DOT, ARG_DOT, MAG_DOT, REL_DOT)):
        name = {"sum": "nfSum", "horner": "nfHorner", "dot": "nfDotFrom"}[kind]
        final = {"sum": "nfSum", "horner": "nfHorner", "dot": "nfDot"}[kind]
        for method, command in (("forward", "roundoff mixed theorem"),
                                ("closed", "roundoff closed theorem"),
                                ("backward", "roundoff (backward := true) theorem")):
            cases.append({"id": f"loop:{kind}:{method}", "declaration": f"{name}_{method}",
                          "commands": [definition, f"{command} {name}_{method} := {argument}"],
                          "route": "loop", "method": method, "loop": kind})
        closed = [f"roundoff closed theorem {name}_closed := {argument}"] + EXTRA[kind]
        if magnitudes:
            cases.append({"id": f"loop:{kind}:magnitudes", "declaration": f"{final}_magnitudes",
                          "commands": [definition] + closed + [magnitudes],
                          "route": "loop", "method": "magnitudes", "loop": kind})
        cases.append({"id": f"loop:{kind}:relative", "declaration": f"{final}_rel",
                      "commands": [definition] + closed + ([magnitudes] if magnitudes else []) + [relative],
                      "route": "loop", "method": "relative", "loop": kind})
    for exponent in sorted({exponent for _, exponent in SIZES.values()}):
        cases.append(weight_case(exponent))
    for program, (kind, _) in SIZES.items():
        case = link_case(program, kind, plan)
        if case is not None:
            cases.append(case)
    (work / "plan-loops.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    (work / "loops-sizes.json").write_text(json.dumps(
        {program: {"loop": kind, "exponent": exponent, "weight": str(weight(exponent)),
                   "weight_float": float(weight(exponent))}
         for program, (kind, exponent) in SIZES.items()}, indent=1))
    print(f"{len(cases)} loop cases")
    for case in cases:
        print("  ", case["id"], sum(len(c) for c in case["commands"]))


if __name__ == "__main__":
    main()
