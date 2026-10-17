/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import Mathlib.Probability.Moments.Covariance
public import NN.Proofs.Tensor.Basic.LinearAlgebra
public import NN.Tensor

/-!
# Shared inputs and expectation identities

The four equally weighted rows enumerate two independent signs. A shared input in
`response` creates a correlation that mean-only propagation discards. The tensor calculation
is scalar-polymorphic; the proofs use exact real arithmetic and mathlib's probability API.

The final result describes the error in replacing a partial calculation before a later
multiplication. It is a conditional error identity, not a general efficient estimator.
-/

@[expose] public section

namespace HeuristicEstimation

open TorchLean MeasureTheory ProbabilityTheory
open scoped BigOperators

/-- Reading a matrix constructed from coordinates recovers its defining entry. -/
theorem matrix_entry {α : Type} [Storage α] {m n : Nat}
    (f : Fin m → Fin n → α) (i : Fin m) (j : Fin n) :
    (Tensor.matrix f)[(i, j)] = f i j := by
  change (Tensor.unstack (Tensor.matrix f) i).getScalar j = _
  simp [Tensor.matrix]

/-- Multiply one input by a mixture of itself and a second input, entry by entry. -/
def response {α : Type} [Storage α] [One α] [Mul α] [Add α] [Sub α]
    {shape : Shape} (shared : α) (x z : Tensor α shape) : Tensor α shape :=
  x * (x.scale shared + z.scale (1 - shared))

/-- Each output coordinate uses the same two input coordinates; no independence is assumed. -/
theorem response_apply {α : Type} [Storage α] [One α] [Mul α] [Add α] [Sub α]
    {shape : Shape} (shared : α) (x z : Tensor α shape) (i : shape.Coord) :
    response shared x z i = x i * (x i * shared + z i * (1 - shared)) := by
  change Tensor.mulSpec x (Tensor.addSpec (x.scale shared) (z.scale (1 - shared))) i = _
  simp [Tensor.mulSpec, Tensor.addSpec, Tensor.map2Spec, Tensor.scaleSpec, Tensor.map]

/-- The first coordinate of the four equally likely sign pairs. -/
def signs {α : Type} [Storage α] [One α] [Neg α] : Tensor α [4] :=
  Tensor.ofFn fun i => if i.val < 2 then -1 else 1

/-- The second coordinate; together with `signs` this enumerates the product distribution. -/
def independentSigns {α : Type} [Storage α] [One α] [Neg α] : Tensor α [4] :=
  Tensor.ofFn fun i => if i.val % 2 = 0 then -1 else 1

/-- The average response is the shared-input coefficient, for every real coefficient. -/
theorem response_mean (shared : ℝ) :
    Tensor.mean (response shared (signs : Tensor ℝ [4]) independentSigns) = shared := by
  rw [Tensor.mean, Tensor.meanSpec, Spec.sum_spec_vec]
  simp [Tensor.getScalar_eq_apply, response_apply, signs, independentSigns, Tensor.ofFn,
    Fin.sum_univ_succ, Tensor.meanDenominator, Shape.size]
  ring

/-- The average squared deviation gives the exact population variance of this finite task. -/
theorem response_variance (shared : ℝ) :
    Tensor.mean (Tensor.square
      (response shared (signs : Tensor ℝ [4]) independentSigns - Tensor.full [4] shared)) =
      (1 - shared) ^ 2 := by
  change Tensor.mean (Tensor.map (fun x => x * x)
    (Tensor.subSpec (response shared signs independentSigns) (Tensor.full [4] shared))) = _
  rw [Tensor.mean, Tensor.meanSpec, Spec.sum_spec_vec]
  simp [Tensor.getScalar_eq_apply, Tensor.map, Tensor.subSpec, Tensor.map2Spec,
    response_apply, signs, independentSigns, Tensor.ofFn, Fin.sum_univ_succ,
    Tensor.meanDenominator, Shape.size]
  ring

/-- Retaining covariance recovers the exact product expectation. Square integrability supplies
the integrability hypotheses; no independence or Gaussian assumption is used. -/
theorem product_mean {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {x y : Ω → ℝ} (hx : MemLp x 2 μ) (hy : MemLp y 2 μ) :
    (∫ ω, x ω * y ω ∂μ) =
      (∫ ω, x ω ∂μ) * (∫ ω, y ω ∂μ) + covariance x y μ := by
  rw [covariance_eq_sub hx hy]
  simp only [Pi.mul_apply]
  ring

/-- A surrogate with a known mean can be corrected by the mean of its residual. This is the
expectation identity behind control variates, not a claim about their cost or variance. -/
theorem residual_correction {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    {target surrogate : Ω → ℝ} (ht : Integrable target μ) (hs : Integrable surrogate μ) :
    (∫ ω, surrogate ω ∂μ) + (∫ ω, target ω - surrogate ω ∂μ) =
      ∫ ω, target ω ∂μ := by
  rw [integral_sub ht hs]
  ring

/-- A later multiplier makes the discarded residual relevant through its correlation with
that multiplier. This identity isolates the quantity a projection rule must control. -/
theorem projection_error {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    {tail head sketch : Ω → ℝ}
    (hh : Integrable (fun ω => tail ω * head ω) μ)
    (hs : Integrable (fun ω => tail ω * sketch ω) μ) :
    (∫ ω, tail ω * head ω ∂μ) - (∫ ω, tail ω * sketch ω ∂μ) =
      ∫ ω, tail ω * (head ω - sketch ω) ∂μ := by
  simp_rw [mul_sub]
  exact (integral_sub hh hs).symm

end HeuristicEstimation
