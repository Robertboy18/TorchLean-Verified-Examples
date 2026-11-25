#!/usr/bin/env python3
"""gen_goals.py HARNESS_JSONL... OUT_JSON

State every case of LeanCert's seven benchmark suites (and its two script-level benchmark
surfaces) as a Lean goal.  The harness rows (JSONL written by `leancert-bench --format jsonl`)
supply the enclosure each case computes; the goal says that this enclosure holds.

Translation rules (fixed before any tactic was run, the same for every case):

* An evaluator case computes an enclosure [lo, hi] of an expression e over a box B.  It becomes
  two goals, `forall x in B, e x <= hi` and `forall x in B, lo <= e x`, with lo and hi the exact
  rationals the case reports.  The Lean term mirrors the benchmark's syntax tree node for node
  (`mkPower` is repeated multiplication, not `x ^ n`).
* Cases of one workload that report the same enclosure share the goal.
* Goals whose shape is not a real inequality (integrals, root existence and uniqueness, exact
  polynomial identities, list constructions, derivatives) are recorded with `in_scope = false`.
"""
import json
import sys
from fractions import Fraction

sys.set_int_max_str_digits(0)


def lit(q: Fraction) -> str:
    """A rational as a real literal."""
    n, d = q.numerator, q.denominator
    if d == 1:
        return f"({n} : ℝ)"
    return f"({n} / {d} : ℝ)"


def power(n: int) -> str:
    return "1" if n == 0 else f"x * ({power(n - 1)})" if n > 1 else "x * 1"


def polynomial(n: int) -> str:
    # .add (polynomial (n-1)) (power n), base .const 1: a left-nested sum
    terms = ["1"] + [power(k) for k in range(1, n + 1)]
    return " + ".join(terms)


def nested_sin(n: int) -> str:
    return "x" if n == 0 else f"Real.sin ({nested_sin(n - 1)})" if n > 1 else "Real.sin x"


def affine_chain(n: int) -> str:
    e = "x"
    for _ in range(n):
        e = f"(1000003 / 1000033 : ℝ) * ({e}) + 1 / 1000037" if e != "x" else \
            "(1000003 / 1000033 : ℝ) * x + 1 / 1000037"
    return e


def repeated_sum(n: int) -> str:
    e = "0"
    for _ in range(n):
        e = f"x + ({e})" if e != "0" else "x + 0"
    return e


def cancellation(n: int) -> str:
    s = repeated_sum(n)
    return f"({s}) + -({s})"


UNIT = "Set.Icc (-1 : ℝ) 1"
NARROW = "Set.Icc (-1 / 16 : ℝ) (1 / 16)"

EVAL_WORKLOADS = {
    "polynomial_10": (polynomial(10), UNIT),
    "nested_sin_3": (nested_sin(3), UNIT),
    "nested_sin_3_narrow": (nested_sin(3), NARROW),
    "x_minus_x": ("x + -x", UNIT),
    "polynomial_75": (polynomial(75), UNIT),
    "nested_sin_16": (nested_sin(16), UNIT),
    "nested_sin_4_narrow": (nested_sin(4), NARROW),
    "rational_affine_chain_160": (affine_chain(160), UNIT),
    "repeated_cancellation_128": (cancellation(128), UNIT),
    "polynomial_150": (polynomial(150), UNIT),
    "polynomial_300": (polynomial(300), UNIT),
    "nested_sin_32": (nested_sin(32), UNIT),
    "nested_sin_64": (nested_sin(64), UNIT),
    "rational_affine_chain_640": (affine_chain(640), UNIT),
    "repeated_cancellation_2048": (cancellation(2048), UNIT),
}

LI2 = ("Real.log (1 + -(x * x)) * (Real.log (1 + x) * Real.log (1 + -x))⁻¹")


def rows_of(paths):
    rows = {}
    for path in paths:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[r["case"]] = r
    return rows


def enclosure(row):
    iv = row["outcome"]["interval"]
    if iv is None:
        return None
    return Fraction(iv["lo"]["exact"]), Fraction(iv["hi"]["exact"])


def main():
    *harness, out = sys.argv[1:]
    rows = rows_of(harness)
    goals = []

    def add(gid, suite, cases, statement, in_scope=True, shape="bound", lc_tactic="leancert",
            lc_only=False, note="", lc_header="", fl_header=""):
        goals.append({
            "id": gid, "suite": suite, "cases": cases, "statement": statement,
            "in_scope": in_scope, "shape": shape, "lc_tactic": lc_tactic,
            "lc_only": lc_only, "note": note, "lc_header": lc_header, "fl_header": fl_header,
        })

    # ---- Suite 1: Evaluation -------------------------------------------------------------
    groups = {}
    for case, row in rows.items():
        if not case.startswith("eval."):
            continue
        workload = case.split(".")[1]
        enc = enclosure(row)
        groups.setdefault((workload, enc), []).append(case)
    for (workload, enc), cases in groups.items():
        expr, box = EVAL_WORKLOADS[workload]
        lo, hi = enc
        backends = sorted({rows[c]["backend_used"] for c in cases})
        tag = "+".join(backends)
        add(f"eval.{workload}.{tag}.upper", "evaluation", cases,
            f"∀ x ∈ {box}, {expr} ≤ {lit(hi)}")
        add(f"eval.{workload}.{tag}.lower", "evaluation", cases,
            f"∀ x ∈ {box}, {lit(lo)} ≤ {expr}")

    # ---- Suite 7: Numerical refinement ---------------------------------------------------
    tight = "(2718281828459045235360287472 / 1000000000000000000000000000 : ℝ)"
    add("refine.loose", "numerical-refinement",
        ["numerical_refinement.loose.fixed_-80_taylor_30",
         "numerical_refinement.loose.adaptive_first_stage"],
        "∀ x ∈ Set.Icc (0 : ℝ) 1, Real.exp x ≤ (3 : ℝ)")
    add("refine.tight", "numerical-refinement",
        ["numerical_refinement.tight.fixed_-80_rejected",
         "numerical_refinement.tight.fixed_-117_taylor_30",
         "numerical_refinement.tight.adaptive_final_stage"],
        f"∀ x ∈ Set.Icc (0 : ℝ) 1, Real.exp x ≤ {tight}")

    # ---- Suite 4: Domain-aware AD (derivative enclosures) --------------------------------
    ad_fun = {
        "inv_log_nested": ("fun t : ℝ => Real.log ((t + 1)⁻¹)", "-((x + 1)⁻¹)"),
    }
    # deepNested 60 = (...((x + 2) * (1/3) + 1/7) * (1/3) + 1/7 ...), 60 times
    deep = "t + 2"
    for _ in range(60):
        deep = f"({deep}) * (1 / 3 : ℝ) + 1 / 7"
    deep_x = deep.replace("t + 2", "x + 2")
    ad_fun["deep_inv_log"] = (f"fun t : ℝ => Real.log (({deep})⁻¹)",
                              f"-((1 / 3 : ℝ) ^ 60 * ({deep_x})⁻¹)")
    ad_groups = {}
    for case, row in rows.items():
        if not case.startswith("ad."):
            continue
        enc = enclosure(row)
        if enc is None:
            continue
        ad_groups.setdefault((case.split(".")[1], enc), []).append(case)
    for (name, enc), cases in ad_groups.items():
        fun, closed = ad_fun[name]
        lo, hi = enc
        tag = "+".join(sorted({rows[c]["backend_used"] for c in cases}))
        box = "Set.Icc (1 : ℝ) 2"
        add(f"ad.{name}.{tag}.deriv.upper", "ad", cases,
            f"∀ x ∈ {box}, deriv ({fun}) x ≤ {lit(hi)}", in_scope=False, shape="derivative")
        add(f"ad.{name}.{tag}.deriv.lower", "ad", cases,
            f"∀ x ∈ {box}, {lit(lo)} ≤ deriv ({fun}) x", in_scope=False, shape="derivative")
        add(f"ad.{name}.{tag}.closed.upper", "ad", cases,
            f"∀ x ∈ {box}, {closed} ≤ {lit(hi)}", shape="bound",
            note="derivative written in closed form by hand")
        add(f"ad.{name}.{tag}.closed.lower", "ad", cases,
            f"∀ x ∈ {box}, {lit(lo)} ≤ {closed}", shape="bound",
            note="derivative written in closed form by hand")
    add("ad.inv_crosses_zero", "ad", ["ad.inv_crosses_zero.rejected"],
        "∀ x ∈ Set.Icc (-1 : ℝ) 1, deriv (fun t : ℝ => t⁻¹) x ≤ (0 : ℝ)", in_scope=False,
        shape="derivative", note="the case expects a domain rejection; no bound exists")

    # ---- Suite 2: Integration -------------------------------------------------------------
    int_groups = {}
    for case, row in rows.items():
        if not case.startswith("integration."):
            continue
        enc = enclosure(row)
        if enc is None:
            continue
        domain = ("(1 / 1024 : ℝ)..(1023 / 1024)" if ("dyadic_level" in case or "power_two" in case)
                  else "(1 / 1000 : ℝ)..(999 / 1000)")
        int_groups.setdefault((domain, enc), []).append(case)
    for index, ((domain, enc), cases) in enumerate(int_groups.items()):
        lo, hi = enc
        name = cases[0].split(".")[1] + "." + cases[0].split(".")[-1]
        # endpoints have up to 10^6 digits: round outward to 12 significant digits
        def outward(q, up):
            from math import floor, ceil, log10
            e = floor(log10(abs(float(q)))) - 11
            s = Fraction(10) ** e
            k = ceil(q / s) if up else floor(q / s)
            return k * s
        add(f"integration.{name}.upper", "integration", cases,
            f"(∫ x in {domain}, {LI2}) ≤ {lit(outward(hi, True))}", in_scope=False,
            shape="integral", note="bound = the case's endpoint rounded outward to 12 digits")
        add(f"integration.{name}.lower", "integration", cases,
            f"{lit(outward(lo, False))} ≤ (∫ x in {domain}, {LI2})", in_scope=False,
            shape="integral", note="bound = the case's endpoint rounded outward to 12 digits")

    # The same twelve cases at the level the harness measures: the checked partition returns an
    # enclosure and a flag that every cell passed the domain validation.  No LeanCert tactic takes
    # this proposition; the kernel route is `decide +kernel`, the native one `native_decide`.
    int_header = ("import LeanCert.Benchmark.Suites.Integration\n"
                  "open LeanCert.Core LeanCert.Engine LeanCert.Validity.IntegrationDyadic "
                  "LeanCert.Benchmark.Integration\n")
    cfg = "{ precision := -53, taylorDepth := 18 }"
    checker = {}
    for cells in (100, 500, 1000, 2300):
        checker[f"integration.li2_partition_{cells}.checked.dyadic"] = (
            f"(integratePartitionDyadicChecked li2Integrand li2Box {cells} (by norm_num) {cfg}).2 = true")
    for depth in (7, 9):
        checker[f"integration.li2_uniform_power_two_{depth}.checked.dyadic"] = (
            f"(integratePartitionDyadicChecked li2Integrand li2DyadicBox.toIntervalRat (2 ^ {depth}) "
            f"(by norm_num) {cfg}).2 = true")
        checker[f"integration.li2_dyadic_level_{depth}.checked.dyadic"] = (
            f"(integrateDyadicLevelChecked li2Integrand li2DyadicBox {depth} {cfg}).2 = true")
    for backend, depth10 in (("rational", 10), ("dyadic", 18)):
        for cells in (100, 500):
            checker[f"integration.li2_partition_{cells}.public.{backend}"] = (
                f"(match LeanCert.integrateUniform li2Integrand li2Box {cells} "
                f"{{ backend := .{backend}, taylorDepth := {depth10} }} with | .ok _ => true | .error _ => false) = true")
    for case, statement in checker.items():
        add("integration.checker." + case.split(".", 1)[1], "integration", [case], statement,
            in_scope=False, shape="checker", lc_tactic="decide +kernel", lc_only=True,
            lc_header=int_header,
            note="the Boolean the harness computes; not a statement about the integral")

    # ---- Suite 3: Krawczyk ----------------------------------------------------------------
    kr_header = ("import LeanCert.Examples.Krawczyk\n"
                 "open LeanCert.Core LeanCert.Engine LeanCert.Validity "
                 "LeanCert.Examples.Krawczyk LeanCert.Tactic\n")
    add("krawczyk.coupled_quadratic_2x2.auto", "krawczyk",
        ["krawczyk.coupled_quadratic_2x2.interval_jacobian",
         "krawczyk.coupled_quadratic_2x2.matrix_norm",
         "krawczyk.coupled_quadratic_2x2.automatic_candidate"],
        "∃! p, FinBoxMem p box ∧ SystemZero system p", in_scope=False, shape="root",
        lc_tactic="system_unique_root", lc_only=True, lc_header=kr_header)
    add("krawczyk.coupled_quadratic_2x2.cert", "krawczyk",
        ["krawczyk.coupled_quadratic_2x2.checked"],
        "∃! p, FinBoxMem p box ∧ SystemZero system p", in_scope=False, shape="root",
        lc_tactic="system_unique_root using certificate", lc_only=True, lc_header=kr_header)
    add("krawczyk.exp_minus_one_1x1.cert", "krawczyk", ["krawczyk.exp_minus_one_1x1.checked"],
        "∃! p, FinBoxMem p expBox ∧ SystemZero expSystem p", in_scope=False, shape="root",
        lc_tactic="system_unique_root using expCertificate", lc_only=True, lc_header=kr_header)
    shifted = (kr_header +
               "def shiftedExpSystem : Fin 1 → LeanCert.Core.Expr :=\n"
               "  ![.add (.exp (.var 0)) (.const (-1 / 8))]\n"
               "def shiftedExpBox : Fin 1 → LeanCert.Core.IntervalRat :=\n"
               "  ![⟨-57 / 20, -33 / 20, by norm_num⟩]\n")
    add("krawczyk.shifted_exp_1x1.auto", "krawczyk", ["krawczyk.shifted_exp_1x1.refined_candidate"],
        "∃! p, FinBoxMem p shiftedExpBox ∧ SystemZero shiftedExpSystem p", in_scope=False,
        shape="root", lc_tactic="system_unique_root", lc_only=True, lc_header=shifted)
    # the same three roots in ordinary Mathlib syntax (one unknown only)
    add("krawczyk.exp_minus_one_1x1.natural", "krawczyk", ["krawczyk.exp_minus_one_1x1.checked"],
        "∃! x, x ∈ Set.Icc (-1 / 10 : ℝ) (1 / 10) ∧ Real.exp x - 1 = 0", in_scope=False,
        shape="root")
    add("krawczyk.shifted_exp_1x1.natural", "krawczyk",
        ["krawczyk.shifted_exp_1x1.refined_candidate"],
        "∃! x, x ∈ Set.Icc (-57 / 20 : ℝ) (-33 / 20) ∧ Real.exp x - 1 / 8 = 0", in_scope=False,
        shape="root")

    # ---- Suite 5: Algebraic Bezout --------------------------------------------------------
    for degree in (3, 64, 256):
        header = ("import LeanCert.Validity.Algebra\n"
                  "open LeanCert.Engine LeanCert.Validity.Algebra\n"
                  f"def P : QPoly := ⟨(((-2 : ℚ) :: List.replicate ({degree} - 1) 0) ++ [1]).toArray⟩\n"
                  f"def C : BezoutCert where\n  A := QPoly.constant ({degree} : ℚ)\n"
                  f"  B := ⟨#[0, -1]⟩\n  c := -(2 * ({degree} : ℚ))\n")
        add(f"algebra.bezout.degree_{degree}", "algebra", [f"algebra.bezout.degree_{degree}"],
            "bezoutCheck P C = true", in_scope=False, shape="identity",
            lc_tactic="decide +kernel", lc_only=True, lc_header=header,
            note="LeanCert's suite checks an exact polynomial identity; no LeanCert tactic "
                 "takes it, so the kernel route is `decide +kernel` and the native one "
                 "`native_decide`")

    # ---- Suite 6: Dyadic subdivision ------------------------------------------------------
    kinds = {
        "complete_tree": "leavesAtDepth benchUnit {d}",
        "direct_cells": "directLeavesAtDepth benchUnit {d}",
        "prepared_cells": "preparedLeavesAtDepth benchUnit {d}",
        "sequential_cells": "sequentialLeavesAtDepth benchUnit {d}",
    }
    dy_header = ("import LeanCert.Benchmark.Suites.DyadicSubdivision\n"
                 "open LeanCert.Core LeanCert.Benchmark.DyadicSubdivision\n"
                 "def benchUnit : IntervalDyadic :=\n"
                 "  ⟨LeanCert.Core.Dyadic.ofInt 0, LeanCert.Core.Dyadic.ofInt 1,\n"
                 "    by norm_num [LeanCert.Core.Dyadic.toRat_ofInt]⟩\n")
    for depth in (4, 8, 12):
        for kind, term in kinds.items():
            leaves = term.format(d=depth)
            add(f"dyadic.{kind}.depth_{depth}", "dyadic-subdivision",
                [f"dyadic_subdivision.{kind}.depth_{depth}"],
                f"({leaves}).length = 2 ^ {depth} ∧ (({leaves}).foldl (fun total cell => "
                f"total.add cell.width) (0 : LeanCert.Core.Dyadic)).toRat = "
                f"benchUnit.width.toRat",
                in_scope=False, shape="structure", lc_tactic="decide +kernel", lc_only=True,
                lc_header=dy_header)
        add(f"dyadic.addresses_only.depth_{depth}", "dyadic-subdivision",
            [f"dyadic_subdivision.addresses_only.depth_{depth}"],
            f"(List.ofFn fun index : Fin (2 ^ {depth}) => DyadicCell.mk {depth} index).length = "
            f"2 ^ {depth}", in_scope=False, shape="structure", lc_tactic="decide +kernel",
            lc_only=True, lc_header=dy_header)
        add(f"dyadic.frontier_check.depth_{depth}", "dyadic-subdivision",
            [f"dyadic_subdivision.frontier_check.depth_{depth}"],
            f"(DyadicFrontier.check (pathsAtDepth {depth})).isSome = true", in_scope=False,
            shape="structure", lc_tactic="decide +kernel", lc_only=True, lc_header=dy_header)

    # ---- Script-level surfaces: bench-showcase and bench-trust -----------------------------
    add("showcase.point_log", "showcase", ["point_log"], "Real.log 2 < 7 / 10")
    add("showcase.quantified_nonlinear", "showcase", ["quantified_nonlinear"],
        "∀ x ∈ Set.Icc (0 : ℝ) 1, Real.exp x * Real.cos x ≤ 3")
    add("showcase.multivariate_box", "showcase", ["multivariate_box"],
        "∀ x ∈ Set.Icc (0 : ℝ) 1, ∀ y ∈ Set.Icc (0 : ℝ) 1, x + y ≤ (2 : ℚ)")
    add("showcase.unique_root", "showcase", ["unique_root"],
        "∃! x, x ∈ Set.Icc (1 : ℝ) 2 ∧ x ^ 2 - 2 = 0", in_scope=False, shape="root")
    add("showcase.exact_integral", "showcase", ["exact_integral"],
        "(∫ x in (0 : ℝ)..1, x ^ 2) = 1 / 3", in_scope=False, shape="integral")
    import math
    for i, k in enumerate(range(2, 12)):
        num = int(math.log(k) * 1000) + 12
        add(f"trust.point.log_{k}", "trust-point", [f"point[{i}]"],
            f"Real.log {k} < {num}/1000", lc_tactic="interval_decide")
    for n in (16, 64):
        for k in range(n):
            add(f"trust.subdiv_{n}.cell_{k}", f"trust-subdiv", [f"subdiv[{n}][{k}]"],
                f"∀ x ∈ Set.Icc ({k} / {n} : ℝ) ({k + 1} / {n}), x * x + Real.sin x ≤ 2")
    add("trust.optimization", "trust-optimization", ["optimization"],
        "∀ x ∈ Set.Icc (-1 : ℝ) 1, x * x ≤ 1")
    add("trust.integration", "trust-integration", ["integration"],
        "(4597 / 10000 : ℝ) ≤ (∫ x in (0 : ℝ)..1, Real.sin x) ∧ "
        "(∫ x in (0 : ℝ)..1, Real.sin x) ≤ (4597 / 10000 : ℝ)", in_scope=False, shape="integral",
        note="the script checks that 4597/10000 lies in the computed enclosure, which is not a "
             "theorem about the integral; listed for completeness, not run")
    for n in (100, 1000, 10000):
        add(f"trust.finsum_{n}", "trust-finsum", [f"finsum[{n}]"],
            f"∑ k ∈ Finset.Icc (1 : ℕ) {n}, (k : ℝ) ≤ {n * (n + 1) // 2}", lc_tactic="finsum_bound")

    json.dump(goals, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    by = {}
    for g in goals:
        by.setdefault(g["suite"], []).append(g)
    for suite, gs in by.items():
        print(f"{suite:22s} goals={len(gs):3d} in_scope={sum(g['in_scope'] for g in gs):3d} "
              f"max_statement_chars={max(len(g['statement']) for g in gs)}")
    print("total", len(goals))


if __name__ == "__main__":
    main()
