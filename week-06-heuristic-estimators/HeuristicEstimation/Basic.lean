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

The four equally weighted positions of `signs` and `independentSigns` enumerate two independent
signs. Feeding one of them into both factors of `response` creates a correlation that mean-only
propagation discards. Definitions work over any scalar type, and the probability results are
stated over ℝ.

`projection_error` measures what is lost when a partial result is replaced before a later
multiplication.
-/

@[expose] public section

namespace HeuristicEstimation

open TorchLean MeasureTheory ProbabilityTheory
open scoped BigOperators

/-- Entry `(i, j)` of `Tensor.matrix f` is `f i j`. -/
theorem matrix_entry {α : Type} [Storage α] {m n : Nat}
    (f : Fin m → Fin n → α) (i : Fin m) (j : Fin n) :
    (Tensor.matrix f)[(i, j)] = f i j := by
  change (Tensor.unstack (Tensor.matrix f) i).getScalar j = _
  simp [Tensor.matrix]

/-- Multiply one input by a mixture of itself and a second input, entry by entry. -/
def response {α : Type} [Storage α] [One α] [Mul α] [Add α] [Sub α]
    {shape : Shape} (shared : α) (x z : Tensor α shape) : Tensor α shape :=
  x * (x.scale shared + z.scale (1 - shared))

/-- Coordinatewise form of `response`. -/
theorem response_apply {α : Type} [Storage α] [One α] [Mul α] [Add α] [Sub α]
    {shape : Shape} (shared : α) (x z : Tensor α shape) (i : shape.Coord) :
    response shared x z i = x i * (x i * shared + z i * (1 - shared)) := by
  change Tensor.mulSpec x (Tensor.addSpec (x.scale shared) (z.scale (1 - shared))) i = _
  simp [Tensor.mulSpec, Tensor.addSpec, Tensor.map2Spec, Tensor.scaleSpec, Tensor.map]

/-- First coordinates of the four equally likely sign pairs. -/
def signs {α : Type} [Storage α] [One α] [Neg α] : Tensor α [4] :=
  Tensor.ofFn fun i => if i.val < 2 then -1 else 1

/-- Second coordinates of the same pairs. With `signs`, the four positions enumerate the product
of two independent fair signs. -/
def independentSigns {α : Type} [Storage α] [One α] [Neg α] : Tensor α [4] :=
  Tensor.ofFn fun i => if i.val % 2 = 0 then -1 else 1

/-- Averaged over the four pairs, the response is `shared`, for every real `shared`. -/
theorem response_mean (shared : ℝ) :
    Tensor.mean (response shared (signs : Tensor ℝ [4]) independentSigns) = shared := by
  rw [Tensor.mean, Tensor.meanSpec, Spec.sum_spec_vec]
  simp [Tensor.getScalar_eq_apply, response_apply, signs, independentSigns, Tensor.ofFn,
    Fin.sum_univ_succ, Tensor.meanDenominator, Shape.size]
  ring

/-- Over the four pairs, the response has population variance `(1 - shared) ^ 2`. -/
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

/-- The mean of a product is the product of the means plus the covariance. Both factors need
finite second moments, and nothing is assumed about how they depend on each other. -/
theorem product_mean {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {x y : Ω → ℝ} (hx : MemLp x 2 μ) (hy : MemLp y 2 μ) :
    (∫ ω, x ω * y ω ∂μ) =
      (∫ ω, x ω ∂μ) * (∫ ω, y ω ∂μ) + covariance x y μ := by
  rw [covariance_eq_sub hx hy]
  simp only [Pi.mul_apply]
  ring

/-- A surrogate's mean plus the mean of its residual is the target's mean. Control variates rest
on this identity. -/
theorem residual_correction {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    {target surrogate : Ω → ℝ} (ht : Integrable target μ) (hs : Integrable surrogate μ) :
    (∫ ω, surrogate ω ∂μ) + (∫ ω, target ω - surrogate ω ∂μ) =
      ∫ ω, target ω ∂μ := by
  rw [integral_sub ht hs]
  ring

/-- Replacing `head` by `sketch` before multiplying by `tail` changes the mean by
`E[tail * (head - sketch)]`. A projection rule has to control this correlation between the
later factor and the discarded part. -/
theorem projection_error {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    {tail head sketch : Ω → ℝ}
    (hh : Integrable (fun ω => tail ω * head ω) μ)
    (hs : Integrable (fun ω => tail ω * sketch ω) μ) :
    (∫ ω, tail ω * head ω ∂μ) - (∫ ω, tail ω * sketch ω ∂μ) =
      ∫ ω, tail ω * (head ω - sketch ω) ∂μ := by
  simp_rw [mul_sub]
  exact (integral_sub hh hs).symm

end HeuristicEstimation
