/-
Copyright (c) 2026 FloatLib. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: FloatLib Team
-/
module

import FloatLib.Tactic.Interval
import FloatLib.Tactic.Roundoff
import FloatLib.Floats.Formats.BinaryInterchange.Configured
import FloatLib.Floats.Formats.BinaryInterchange.Configured.CorrectlyRounded
import FloatLib.Floats.Formats.BinaryInterchange.Complex.Core
public meta import Lean.Util.CollectAxioms
public meta import Lean.Elab.Command
public meta import Lean.Elab.Tactic

/-!
# Sequential proof benchmark runner

`FLOATLIB_SUITE_PLAN` names a JSON array of cases with `id`, `declaration`, and `commands` fields.
Each case elaborates in a fresh copy of the imported environment. Success requires a theorem
without any axiom other than `propext`, `Classical.choice`, and `Quot.sound`. The complete statement,
axioms, and elapsed elaboration time go to `FLOATLIB_SUITE_OUTPUT` as JSON lines. An error is data:
it cannot produce a successful row or contaminate the next case.

The Python driver hashes the input plan and every source dependency before running this file.
This is a benchmark of elaboration, not a persistent library declaration or a trusted data import.

The runner is a `module`, so each case elaborates as it does in a module of the library: with
the exported declarations of its imports. A plain file also loads their private ones, which
takes about twice the memory, and it can unfold Mathlib's private definitions of the arithmetic
and the order of the reals, so a unification of two real numerals recurses until Lean's limit.
-/

open Lean Elab Command
open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics

/-- Normalize concrete format constants using existing library equalities. -/
elab "suite_format" : tactic => do
  -- Binary64's subnormal scale needs exponents through 1075.
  Lean.Elab.Tactic.evalTactic (← `(tactic|
    set_option exponentiation.threshold 2048 in
    norm_num [Model.toReal_posMaxFinite, Model.unitRoundoffAt_eq, Model.underflowErrorAt_eq,
      Model.pow2, Model.bpow, FloatLib.Floats.Formats.Flocq.bpow,
      FloatLib.Numerics.binaryRadix, FloatLib.Numerics.Radix.toReal,
      FloatFormat.binary32, FloatFormat.binary64,
      FloatFormat.maxFiniteFracField, FloatFormat.fracMaskNat,
      FloatFormat.maxNormalExponent, FloatFormat.maxFiniteExpField,
      FloatFormat.Encoding.maxFiniteExponent,
      FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent,
      FloatFormat.Encoding.defaultBias, FloatFormat.ieeeBias]))

/-- Discharge a range premise through a checked, conservative format bound. -/
elab "suite_range" : tactic => do
  Lean.Elab.Tactic.evalTactic (← `(tactic|
    first
    | assumption
    | (have hmax : (2 : ℝ) ^ 100 ≤
          Model.toReal (Model.posMaxFinite FloatFormat.binary32) := by
         rw [Model.toReal_posMaxFinite]
         suite_format
       apply le_trans (b := (2 : ℝ) ^ 100) ?_ hmax
       norm_num <;> interval)
    | (have hmax : (2 : ℝ) ^ 1000 ≤
          Model.toReal (Model.posMaxFinite FloatFormat.binary64) := by
         rw [Model.toReal_posMaxFinite]
         suite_format
       apply le_trans (b := (2 : ℝ) ^ 1000) ?_ hmax
       suite_format <;> interval)
    | (suite_format <;> interval)))

/-! ## Additions of this benchmark (not part of FloatLib)

Everything above and the runner below are FloatLib's `benchmarks/suites/Proofs.lean`, unchanged.
The additions are equalities and inequalities about real numbers and lists, and three tactic
macros that only combine FloatLib's and Mathlib's tactics:

* the weights of a sum of positive terms in a form that interval arithmetic bounds without
  overestimation, and the step from a mixed bound to a relative one (`NF.close`);
* folds: a fold of additions as a sum, and the two steps that turn the closed bound of Horner's
  rule into one weight times the polynomial of the magnitudes;
* the monotonicity of the relative part of a mixed bound in the relative errors of its operands
  (`NF.round_le`, `NF.prod_le`, `NF.mean_le_max`, ...), used by `nf_mono`;
* the format constants of the directed rounding attributes as numerals (`nf_format`). -/

namespace NF

/-- The weight of the first of two positive terms: `|P| / |P + Q| = 1 / (1 + Q / P)`. -/
theorem ratio_left {P Q c : ℝ} (hP : 0 < P) (hQ : 0 < Q) :
    c * |P| / |P + Q| = c * (1 / (1 + Q / P)) := by
  rw [abs_of_pos hP, abs_of_pos (add_pos hP hQ)]
  field_simp

/-- The weight of the second of two positive terms. -/
theorem ratio_right {P Q c : ℝ} (hP : 0 < P) (hQ : 0 < Q) :
    c * |Q| / |P + Q| = c * (1 / (1 + P / Q)) := by
  rw [abs_of_pos hQ, abs_of_pos (add_pos hP hQ)]
  field_simp
  ring

/-- Two positive terms with the same relative error keep it. -/
theorem ratio_same {P Q c : ℝ} (hP : 0 < P) (hQ : 0 < Q) :
    (c * |P| + c * |Q|) / |P + Q| = c := by
  rw [abs_of_pos hP, abs_of_pos hQ, abs_of_pos (add_pos hP hQ)]
  field_simp

/-- The weighted mean of two relative errors, with one occurrence of the weight. -/
theorem ratio_both {P Q c d : ℝ} (hP : 0 < P) (hQ : 0 < Q) :
    (c * |P| + d * |Q|) / |P + Q| = d + (c - d) * (1 / (1 + Q / P)) := by
  rw [abs_of_pos hP, abs_of_pos hQ, abs_of_pos (add_pos hP hQ)]
  field_simp
  ring

/-- The factor of the square-root rule, `√R / (√R + √(d R - s))`, without `R` in two places. -/
theorem sqrt_ratio {R c d s : ℝ} (hR : 0 < R) :
    c * √R / (√R + √(d * R - s)) = c * (1 / (1 + √(d - s / R))) := by
  have hroot : 0 < √R := Real.sqrt_pos.mpr hR
  have h : d * R - s = R * (d - s / R) := by field_simp
  have h1 : (1 + √(d - s / R)) ≠ 0 := by positivity
  rw [h, Real.sqrt_mul hR.le]
  field_simp

/-- From a mixed bound to a relative one: `R ≤ B1`, `A ≤ δ`, `m ≤ |r|` and `B1 + δ / m ≤ B`. -/
theorem close {R A r B1 δ m B : ℝ} (hR : R ≤ B1) (hA : A ≤ δ) (hm : m ≤ |r|) (hm0 : 0 < m)
    (hδ : 0 ≤ δ) (hB : B1 + δ / m ≤ B) : R * |r| + A ≤ B * |r| := by
  have h1 : R * |r| ≤ B1 * |r| := mul_le_mul_of_nonneg_right hR (abs_nonneg r)
  have h3 : (B1 + δ / m) * |r| ≤ B * |r| := mul_le_mul_of_nonneg_right hB (abs_nonneg r)
  have h2 : A ≤ δ / m * |r| :=
    calc A ≤ δ := hA
      _ = δ / m * m := by field_simp
      _ ≤ δ / m * |r| := mul_le_mul_of_nonneg_left hm (div_nonneg hδ hm0.le)
  nlinarith

/-- A rounding grows with the relative error of what it rounds. -/
theorem round_le {u r r' : ℝ} (hu : 0 ≤ u) (h : r ≤ r') : (1 + u) * r + u ≤ (1 + u) * r' + u := by
  have h1 : (1 + u) * r ≤ (1 + u) * r' := mul_le_mul_of_nonneg_left h (by linarith)
  linarith

/-- The relative error of a product grows with those of its factors. -/
theorem prod_le {r s r' s' : ℝ} (hr : 0 ≤ r) (hs : 0 ≤ s) (h1 : r ≤ r') (h2 : s ≤ s') :
    r + s + r * s ≤ r' + s' + r' * s' := by
  have h3 : r * s ≤ r' * s' := mul_le_mul h1 h2 hs (le_trans hr h1)
  linarith

/-- A sum of two positive terms has at most the larger of their relative errors. -/
theorem mean_le_max {P Q c d c' d' : ℝ} (hP : 0 < P) (hQ : 0 < Q) (hc : c ≤ c') (hd : d ≤ d') :
    (c * |P| + d * |Q|) / |P + Q| ≤ max c' d' := by
  rw [abs_of_pos hP, abs_of_pos hQ, abs_of_pos (add_pos hP hQ), div_le_iff₀ (add_pos hP hQ)]
  have h1 : c * P ≤ max c' d' * P :=
    mul_le_mul_of_nonneg_right (le_trans hc (le_max_left _ _)) hP.le
  have h2 : d * Q ≤ max c' d' * Q :=
    mul_le_mul_of_nonneg_right (le_trans hd (le_max_right _ _)) hQ.le
  linarith

/-- A sum of an inexact positive term and an exact one has at most the relative error of the
inexact term. -/
theorem mean_left_le {P Q c c' : ℝ} (hP : 0 < P) (hQ : 0 < Q) (hc0 : 0 ≤ c) (hc : c ≤ c') :
    c * |P| / |P + Q| ≤ c' := by
  rw [abs_of_pos hP, abs_of_pos (add_pos hP hQ), div_le_iff₀ (add_pos hP hQ)]
  have h1 : c * P ≤ c * (P + Q) := mul_le_mul_of_nonneg_left (by linarith) hc0
  have h2 : c * (P + Q) ≤ c' * (P + Q) := mul_le_mul_of_nonneg_right hc (add_pos hP hQ).le
  linarith

/-- The same with the inexact term second. -/
theorem mean_right_le {P Q d d' : ℝ} (hP : 0 < P) (hQ : 0 < Q) (hd0 : 0 ≤ d) (hd : d ≤ d') :
    d * |Q| / |P + Q| ≤ d' := by
  rw [abs_of_pos hQ, abs_of_pos (add_pos hP hQ), div_le_iff₀ (add_pos hP hQ)]
  have h1 : d * Q ≤ d * (P + Q) := mul_le_mul_of_nonneg_left (by linarith) hd0
  have h2 : d * (P + Q) ≤ d' * (P + Q) := mul_le_mul_of_nonneg_right hd (add_pos hP hQ).le
  linarith

/-- A left fold that adds `f j` is the start plus the sum of the `f j`. -/
theorem foldl_add {α : Type} (f : α → ℝ) (l : List α) (a : ℝ) :
    l.foldl (fun s j => s + f j) a = a + (l.map f).sum := by
  induction l generalizing a with
  | nil => simp
  | cons b l ih => simp [ih, add_assoc]

/-- The same over all indices below `n`. -/
theorem foldl_add_fin {n : Nat} (f : Fin n → ℝ) (a : ℝ) :
    (List.finRange n).foldl (fun s j => s + f j) a = a + ∑ j, f j := by
  rw [foldl_add, Fin.sum_univ_def]

/-- The sum of `f` over a list is the sum over its positions. -/
theorem sum_map_eq_sum_fin {α : Type} (f : α → ℝ) (l : List α) :
    (l.map f).sum = ∑ k : Fin l.length, f l[(k : Nat)] := by
  rw [Fin.sum_univ_def]
  conv_lhs => rw [← List.map_getElem_finRange l]
  rw [List.map_map]
  rfl

/-- The error fold of Horner's rule grows with its start and with its weights. -/
theorem foldl_horner_le {ι : Type} (a : ℝ) (c w : ι → ℝ) (W : ℝ) (ha : 0 ≤ a)
    (hc : ∀ k, 0 ≤ c k) (hw : ∀ k, w k ≤ W) (l : List ι) (e e' : ℝ) (he : e ≤ e') :
    l.foldl (fun e k => e * a + c k * w k) e ≤ l.foldl (fun e k => e * a + c k * W) e' := by
  induction l generalizing e e' with
  | nil => simpa using he
  | cons k l ih =>
    simp only [List.foldl_cons]
    exact ih _ _ (add_le_add (mul_le_mul_of_nonneg_right he ha)
      (mul_le_mul_of_nonneg_left (hw k) (hc k)))

/-- With one weight `W`, the error fold is `W` times the Horner fold of the magnitudes. -/
theorem foldl_horner_scale {ι : Type} (a : ℝ) (c : ι → ℝ) (W : ℝ) (l : List ι) (v : ℝ) :
    l.foldl (fun e k => e * a + c k * W) (v * W) = W * l.foldl (fun v k => v * a + c k) v := by
  induction l generalizing v with
  | nil => simp [mul_comm]
  | cons k l ih =>
    simp only [List.foldl_cons]
    rw [← ih]
    congr 1
    ring

/-- The unit roundoff of binary64 rounded toward +∞ is `2^-52`. -/
theorem u64_eq :
    Model.unitRoundoffIn FloatFormat.binary64 Model.IEEERoundingMode.up = 1 / 2 ^ 52 := by
  norm_num [Model.unitRoundoffIn, Model.IEEERoundingMode.halfUlps, Model.unitRoundoffAt_eq,
    Model.pow2, Model.bpow, FloatLib.Floats.Formats.Flocq.bpow, FloatLib.Numerics.binaryRadix,
    FloatLib.Numerics.Radix.toReal, FloatFormat.binary64]

end NF

/-- The format constants of binary16, binary32 and binary64 in every rounding attribute, as
numerals. -/
macro "nf_format" : tactic => `(tactic|
  (set_option exponentiation.threshold 2048 in
    norm_num [max_def, Model.unitRoundoffIn, Model.underflowErrorIn,
      Model.IEEERoundingMode.halfUlps,
      Model.toReal_posMaxFinite, Model.unitRoundoffAt_eq, Model.underflowErrorAt_eq,
      Model.pow2, Model.bpow, FloatLib.Floats.Formats.Flocq.bpow,
      FloatLib.Numerics.binaryRadix, FloatLib.Numerics.Radix.toReal,
      FloatFormat.binary16, FloatFormat.binary32, FloatFormat.binary64,
      FloatFormat.maxFiniteFracField, FloatFormat.fracMaskNat,
      FloatFormat.maxNormalExponent, FloatFormat.maxFiniteExpField,
      FloatFormat.Encoding.maxFiniteExponent,
      FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent,
      FloatFormat.Encoding.defaultBias, FloatFormat.ieeeBias]))

/-- An upper bound of the relative part of a mixed bound that no longer holds the weights of sums
of positive terms: a goal `R ≤ ?B` is closed by walking `R` through its roundings
(`(1 + u) * r + u`), products (`r + s + r * s`) and sums (`(c * |P| + d * |Q|) / |P + Q|`, at most
the larger of `c` and `d`), which assigns `?B`. A part of another shape stays as it is. -/
syntax "nf_mono" : tactic
macro_rules
  | `(tactic| nf_mono) => `(tactic|
    first
    | (apply NF.round_le (Model.unitRoundoffIn_pos _ _).le
       · nf_mono)
    | (apply NF.prod_le (by positivity) (by positivity)
       · nf_mono
       · nf_mono)
    | (apply NF.mean_le_max (by positivity) (by positivity)
       · nf_mono
       · nf_mono)
    | (apply NF.mean_left_le (by positivity) (by positivity) (by positivity)
       · nf_mono)
    | (apply NF.mean_right_le (by positivity) (by positivity) (by positivity)
       · nf_mono)
    | exact le_refl _)

/-- FloatLib's `interval`, then again with 200 fractional bits: bounds near 1e-16 leave the
default grid only a few bits. -/
macro "nf_interval" : tactic => `(tactic| first | interval | interval (precision := 200))

/-- One side goal of the step from a mixed bound to a relative one: rewrite the weights of sums,
take the branches of a conditional apart, turn the format constants into numerals, and leave the
inequality to `interval`. Before that, a divisor that must not vanish goes to `positivity` and a
branch that the hypotheses exclude to `linarith`; both fail at once on any other goal. -/
macro "nf_goal" : tactic => `(tactic|
  ((try simp (disch := positivity) only [NF.ratio_same, NF.ratio_both, NF.ratio_left,
      NF.ratio_right, NF.sqrt_ratio]) <;>
   (try split_ifs) <;> nf_format <;> (try split_ifs) <;>
   (first | positivity | (refine ne_of_gt ?_; positivity)
          | (exfalso; simp_all <;> linarith) | (exfalso; norm_num at * <;> linarith)
          | nf_interval)))

private meta def field (json : Json) (key : String) : CommandElabM String :=
  match json.getObjValAs? String key with
  | .ok value => pure value
  | .error error => throwError "{error}"

-- A case must finish elaborating before its diagnostics are inspected or its state is restored.
set_option Elab.async false in
run_cmd do
  let some planPath ← IO.getEnv "FLOATLIB_SUITE_PLAN"
    | throwError "FLOATLIB_SUITE_PLAN is required"
  let some outputPath ← IO.getEnv "FLOATLIB_SUITE_OUTPUT"
    | throwError "FLOATLIB_SUITE_OUTPUT is required"
  let .ok plan := Json.parse (← IO.FS.readFile planPath)
    | throwError "Invalid plan JSON"
  let .ok cases := plan.getArr?
    | throwError "The plan must be an array"
  let output ← IO.FS.Handle.mk outputPath .append
  for item in cases do
    let id ← field item "id"
    if let some activePath ← IO.getEnv "FLOATLIB_SUITE_ACTIVE" then
      IO.FS.writeFile activePath id
    let declaration ← field item "declaration"
    let .ok commands := item.getObjValAs? (Array String) "commands"
      | throwError "Missing commands for {id}"
    let saved : Command.State ← get
    let start ← IO.monoNanosNow
    let mut status := "proved"
    let mut reason := ""
    let mut statement := ""
    let mut dependencies : Array String := #[]
    try
      for text in commands do
        let .ok command := Parser.runParserCategory (← getEnv) `command text
          | throwError "Invalid command: {text}"
        elabCommand command
      let messages := (← get).messages.toList.filter (fun msg => msg.severity == .error)
      unless messages.isEmpty do
        let errors ← messages.mapM (fun msg => msg.data.toString)
        throwError "{String.intercalate "\n" errors}"
      -- A declaration of a module is private to it unless its command says otherwise.
      let stated := declaration.toName
      let environment ← getEnv
      let name := if environment.contains stated then stated else mkPrivateName environment stated
      let some (.thmInfo info) := (← getEnv).find? name
        | throwError "Expected theorem {declaration}"
      let axioms ← collectAxioms name
      for dependency in axioms do
        unless [``propext, ``Classical.choice, ``Quot.sound].contains dependency do
          throwError "Unexpected axiom: {dependency}"
      dependencies := axioms.map Name.toString
      statement ← liftTermElabM do
        withOptions (fun options =>
            options.set `pp.maxDepth (100000 : Nat) |>.set `pp.maxSteps (10000000 : Nat)) do
          return (← Meta.ppExpr info.type).pretty 120
    catch error =>
      status := "refused"
      reason ← error.toMessageData.toString
    let elapsed := (← IO.monoNanosNow) - start
    set saved
    let row := Json.mkObj [
      ("id", toJson id), ("status", toJson status), ("reason", toJson reason),
      ("elapsed_ns", toJson elapsed), ("statement", toJson statement),
      ("axioms", toJson dependencies)]
    output.putStrLn row.compress
    output.flush
