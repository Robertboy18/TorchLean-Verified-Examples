/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import HeuristicEstimation.Basic

/-!
# Moments through affine tensor maps

Means and cross-covariances pass through affine tensor maps by mathlib's integration and
covariance rules. Two maps may read the same random input, whose coordinates may depend on each
other in any way. Mean and variance alone don't fix a fourth moment
(`moments_do_not_determine_fourth`).
-/

@[expose] public section

namespace HeuristicEstimation.Moments

open TorchLean MeasureTheory ProbabilityTheory
open scoped BigOperators

/-- An affine layer commutes with expectation whenever the input coordinates are integrable. -/
theorem affine_mean {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {inputs outputs : Nat} (weights : Tensor ℝ [outputs, inputs])
    (bias : Tensor ℝ [outputs]) (values : Ω → Tensor ℝ [inputs])
    (h : ∀ j, Integrable (fun ω => (values ω).getScalar j) μ) (i : Fin outputs) :
    (∫ ω, Tensor.getScalar (Tensor.matvec weights (values ω) + bias) i ∂μ) =
      Tensor.getScalar (Tensor.matvec weights
        (Tensor.ofFn fun j => ∫ ω, (values ω).getScalar j ∂μ) + bias) i := by
  change (∫ ω, (Tensor.addSpec (Tensor.matvec weights (values ω)) bias).getScalar i ∂μ) =
    (Tensor.addSpec (Tensor.matvec weights (Tensor.ofFn fun j =>
      ∫ ω, (values ω).getScalar j ∂μ)) bias).getScalar i
  simp only [Spec.getScalar_add_spec, Tensor.matvec, Spec.getScalar_mat_vec_mul_spec,
    Tensor.getScalar_ofFn]
  have hint (j : Fin inputs) := (h j).const_mul (Spec.get2 weights i j)
  rw [integral_add (f := fun ω => ∑ j, Spec.get2 weights i j * (values ω).getScalar j)
    (g := fun _ => bias.getScalar i) (integrable_finsetSum _ fun j _ => hint j)
    (integrable_const _)]
  rw [integral_finsetSum Finset.univ (fun j _ => hint j)]
  simp only [integral_const_mul, integral_const, probReal_univ, one_smul]

/-- The covariance between outputs of two affine maps of one input is `L Σ Rᵀ`, where `Σ` is the
input covariance. Biases drop out, and taking both maps equal gives `W Σ Wᵀ`. -/
theorem affine_covariance {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {inputs leftSize rightSize : Nat}
    (left : Tensor ℝ [leftSize, inputs]) (right : Tensor ℝ [rightSize, inputs])
    (leftBias : Tensor ℝ [leftSize]) (rightBias : Tensor ℝ [rightSize])
    (values : Ω → Tensor ℝ [inputs])
    (h : ∀ k, MemLp (fun ω => (values ω).getScalar k) 2 μ)
    (i : Fin leftSize) (j : Fin rightSize) :
    covariance (fun ω => Tensor.getScalar (Tensor.matvec left (values ω) + leftBias) i)
      (fun ω => Tensor.getScalar (Tensor.matvec right (values ω) + rightBias) j) μ =
        ∑ k : Fin inputs, ∑ l : Fin inputs,
          left[(i, k)] * covariance (fun ω => (values ω).getScalar k)
            (fun ω => (values ω).getScalar l) μ * right[(j, l)] := by
  change covariance
    (fun ω => (Tensor.addSpec (Tensor.matvec left (values ω)) leftBias).getScalar i)
    (fun ω => (Tensor.addSpec (Tensor.matvec right (values ω)) rightBias).getScalar j) μ = _
  simp only [Spec.getScalar_add_spec, Tensor.matvec, Spec.getScalar_mat_vec_mul_spec]
  have hleft (k : Fin inputs) := (h k).const_mul (Spec.get2 left i k)
  have hright (k : Fin inputs) := (h k).const_mul (Spec.get2 right j k)
  rw [covariance_add_const_left
    ((memLp_finsetSum Finset.univ (fun k _ => hleft k)).integrable (by norm_num))]
  rw [covariance_add_const_right
    ((memLp_finsetSum Finset.univ (fun k _ => hright k)).integrable (by norm_num))]
  rw [covariance_fun_sum_fun_sum hleft hright]
  simp only [covariance_const_mul_left, covariance_const_mul_right]
  apply Finset.sum_congr rfl
  intro k _
  apply Finset.sum_congr rfl
  intro l _
  change _ = Spec.get2 left i k * _ * Spec.get2 right j l
  ring

/-- Two equally weighted four-point distributions share mean `0` and second moment `1`, but
their fourth moments are `1` and `2`. -/
theorem moments_do_not_determine_fourth :
    ∃ a b : Tensor ℝ [4],
      Tensor.mean a = 0 ∧ Tensor.mean b = 0 ∧
      Tensor.mean (Tensor.square a) = 1 ∧ Tensor.mean (Tensor.square b) = 1 ∧
      Tensor.mean (Tensor.square (Tensor.square a)) = 1 ∧
      Tensor.mean (Tensor.square (Tensor.square b)) = 2 := by
  refine ⟨signs, Tensor.ofFn (fun i =>
    if i.val = 0 then -Real.sqrt 2 else if i.val = 3 then Real.sqrt 2 else 0), ?_⟩
  norm_num [Tensor.mean, Tensor.meanSpec, Spec.sum_spec_vec, Tensor.getScalar_eq_apply,
    Tensor.square, Tensor.squareSpec, Tensor.map, signs, Tensor.ofFn, Fin.sum_univ_succ,
    Tensor.meanDenominator, Shape.size, Real.mul_self_sqrt (by norm_num : (0 : ℝ) ≤ 2)]

end HeuristicEstimation.Moments
