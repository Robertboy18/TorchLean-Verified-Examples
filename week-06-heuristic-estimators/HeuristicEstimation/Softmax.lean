/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import HeuristicEstimation.Basic
public import Mathlib.Analysis.Calculus.MeanValue
public import NN.Proofs.Autograd.FDeriv.SoftmaxSpec

/-!
# Perturbing an attention row

A score approximation matters through the softmax-weighted values, not just through its
coordinate error. The main bound is independent of the number of keys: changing each logit
by at most `ε` changes a weighted output by at most `2 * V * ε`, when the values are bounded
by `V`. The proof follows TorchLean's actual stable softmax through its existing derivative
theorem. It does not assume that the two output vectors are already close.
-/

@[expose] public section

namespace HeuristicEstimation.Softmax

open TorchLean
open Proofs.Autograd
open scoped BigOperators

/-- One coordinate of an attention output, using the stable tensor softmax. -/
def output {α : Type} [Storage α] [Context α] {n : Nat}
    (logits values : Tensor α [n]) : α :=
  Tensor.sum (Activation.softmaxSpec 0 logits * values)

private theorem output_eq_dot {n : Nat} (logits values : Tensor ℝ [n]) :
    output logits values = dotCLM (getScalarE values) (softmaxVec (getScalarE logits)) := by
  change Tensor.sumSpec (Tensor.mulSpec (Activation.softmaxSpec 0 logits) values) = _
  rw [Spec.sum_spec_vec, dotCLM_apply]
  apply Finset.sum_congr rfl
  intro i _
  rw [Spec.getScalar_mul_spec]
  have h := congrArg (fun v : Vec n => v i) (getScalarE_softmaxVecSpec logits)
  rw [Proofs.softmaxSpec_zero_vec]
  change _ * values.getScalar i = values.getScalar i * _
  rw [← h]
  rw [getScalarE_ofLp, mul_comm]

private theorem weighted_jvp_le {n : Nat} (logits direction values : Vec (n + 1))
    (V ε : ℝ) (hv : ∀ i, |values i| ≤ V) (hd : ∀ i, |direction i| ≤ ε) :
    |dotCLM values (softmaxJvp logits direction)| ≤ 2 * V * ε := by
  classical
  let p := softmaxVec logits
  let center := dotCLM p direction
  have hp : ∀ i, 0 < p i := fun i => by
    simp only [p, softmaxVec, vecOfFun_apply]
    exact div_pos (Real.exp_pos _) (sumExp_pos _)
  have hsum : ∑ i, p i = 1 := by
    simp only [p, softmaxVec, vecOfFun_apply]
    rw [← Finset.sum_div]
    exact div_self (sumExp_ne_zero _)
  have hcenter : |center| ≤ ε := by
    dsimp only [center]
    rw [dotCLM_apply]
    calc
      _ ≤ ∑ i, |p i * direction i| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ i, p i * |direction i| := by
        simp_rw [abs_mul, abs_of_pos (hp _)]
      _ ≤ ∑ i, p i * ε :=
        Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hd i) (hp i).le
      _ = ε := by rw [← Finset.sum_mul, hsum, one_mul]
  have hentry (i : Fin (n + 1)) :
      |values i * (p i * (direction i - center))| ≤ V * p i * (2 * ε) := by
    have hdiff : |direction i - center| ≤ 2 * ε := by
      calc
        _ ≤ |direction i| + |center| := abs_sub _ _
        _ ≤ ε + ε := add_le_add (hd i) hcenter
        _ = _ := by ring
    rw [abs_mul, abs_mul, abs_of_pos (hp i)]
    calc
      _ ≤ V * (p i * |direction i - center|) :=
        mul_le_mul_of_nonneg_right (hv i) (mul_nonneg (hp i).le (abs_nonneg _))
      _ ≤ V * (p i * (2 * ε)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hdiff (hp i).le)
          ((abs_nonneg _).trans (hv i))
      _ = _ := by ring
  rw [dotCLM_apply]
  change |∑ i, values i * (p i * (direction i - center))| ≤ _
  calc
    _ ≤ ∑ i, |values i * (p i * (direction i - center))| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, V * p i * (2 * ε) := Finset.sum_le_sum fun i _ => hentry i
    _ = 2 * V * ε := by rw [← Finset.sum_mul, ← Finset.mul_sum, hsum]; ring

/-- A uniform logit error gives a dimension-free bound on the weighted attention output.
The values need not be positive, and no independence or distributional premise is used. -/
theorem output_sub_le {n : Nat} (logits approximate values : Tensor ℝ [n + 1])
    (V ε : ℝ) (hv : ∀ i, |values.getScalar i| ≤ V)
    (hlogits : ∀ i, |approximate.getScalar i - logits.getScalar i| ≤ ε) :
    |output approximate values - output logits values| ≤ 2 * V * ε := by
  let x := getScalarE logits
  let y := getScalarE approximate
  let v := getScalarE values
  let path : ℝ → Vec (n + 1) := AffineMap.lineMap x y
  let f := fun t : ℝ => dotCLM v (softmaxVec (path t))
  let derivative := fun t : ℝ => dotCLM v (softmaxJvp (path t) (y - x))
  have hf (t : ℝ) : HasDerivAt f (derivative t) t := by
    have hp : HasDerivAt path (y - x) t := AffineMap.hasDerivAt_lineMap
    have hs := (hasFDerivAt_softmaxVec (path t)).comp_hasDerivAt t hp
    have ho := (dotCLM v).hasFDerivAt.comp_hasDerivAt t hs
    convert! ho using 1
    exact congrArg (dotCLM v) (softmaxJvp_eq_deriv _ _)
  have hbound (t : ℝ) : ‖derivative t‖ ≤ 2 * V * ε := by
    rw [Real.norm_eq_abs]
    exact weighted_jvp_le (path t) (y - x) v V ε hv hlogits
  have h := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
    (f := f) (f' := derivative) (s := Set.Icc (0 : ℝ) 1)
    (fun t _ => (hf t).hasDerivWithinAt) (fun t _ => hbound t)
    (convex_Icc 0 1) (by simp : (0 : ℝ) ∈ Set.Icc (0 : ℝ) 1)
    (by simp : (1 : ℝ) ∈ Set.Icc (0 : ℝ) 1)
  simpa [f, path, x, y, v, Real.norm_eq_abs, output_eq_dot] using h

/-- Adding a common constant to all scores leaves the attention output unchanged. -/
theorem output_shift {n : Nat} (logits values : Tensor ℝ [n + 1]) (shift : ℝ) :
    output (Tensor.ofFn fun i => logits.getScalar i + shift) values =
      output logits values := by
  have hs : Activation.softmaxSpec 0 (Tensor.ofFn fun i => logits.getScalar i + shift) =
      Activation.softmaxSpec 0 logits := by
    rw [Proofs.softmaxSpec_zero_vec, Proofs.softmaxSpec_zero_vec]
    apply Tensor.ext_vector
    intro i
    rw [Proofs.getScalar_softmaxVecSpec_eq_exp_div,
      Proofs.getScalar_softmaxVecSpec_eq_exp_div]
    simp only [Tensor.getScalar_ofFn]
    simp_rw [Real.exp_add]
    rw [← Finset.sum_mul]
    exact mul_div_mul_right _ _ (Real.exp_ne_zero shift)
  simp only [output, hs]

/-- Only score differences between keys matter. A certificate may remove any common logit
shift before bounding its residual, including the midpoint of the minimum and maximum errors. -/
theorem output_sub_le_centered {n : Nat} (logits approximate values : Tensor ℝ [n + 1])
    (V ε shift : ℝ) (hv : ∀ i, |values.getScalar i| ≤ V)
    (hlogits : ∀ i, |approximate.getScalar i - logits.getScalar i - shift| ≤ ε) :
    |output approximate values - output logits values| ≤ 2 * V * ε := by
  rw [← output_shift logits values shift]
  apply output_sub_le _ approximate values V ε hv
  intro i
  simpa only [Tensor.getScalar_ofFn, sub_add_eq_sub_sub] using hlogits i

end HeuristicEstimation.Softmax
