/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import HeuristicEstimation.Basic
public import NN.Proofs.Autograd.Core.SemiringCorrectness

/-!
# Expected error at a ReLU boundary

Replacing a conditionally centered perturbation by zero removes its linear contribution to a
ReLU's expected output. What remains is nonnegative, and it vanishes unless the center lies
within the perturbation radius of zero. In `mean_error_le` the perturbation's law may depend on
the retained state, but its mean must vanish for each retained state separately.

The sensitivity may depend on the retained state too. It stays inside the weighted sum, so its
correlation with the boundary event is kept.
-/

@[expose] public section

namespace HeuristicEstimation.Boundary

open TorchLean
open Proofs.Autograd.Algebra (reluCorrect)
open scoped BigOperators

/-- What remains of `relu (center + perturbation)` after linearizing ReLU at `center`. The
slope at zero is taken to be zero. -/
noncomputable def remainder (center perturbation : ℝ) : ℝ :=
  Activation.Math.reluSpec (center + perturbation) - Activation.Math.reluSpec center -
    if 0 < center then perturbation else 0

/-- The remainder is nonnegative: ReLU lies above its linearization at every center, zero
included. -/
theorem remainder_nonneg (center perturbation : ℝ) : 0 ≤ remainder center perturbation := by
  unfold remainder
  simp only [Activation.Math.reluSpec_eq_max]
  by_cases hcenter : 0 < center
  · rw [ite_eq_left hcenter, max_eq_left hcenter.le]
    have h := le_max_left (center + perturbation) 0
    linarith
  · rw [ite_eq_right hcenter, max_eq_right (le_of_not_gt hcenter)]
    have h := le_max_right (center + perturbation) 0
    linarith

/-- A perturbation of size at most `radius` leaves a remainder of at most
`max 0 (radius - |center|)`, the distance it can carry the input past zero. A center at least
`radius` from zero leaves none. -/
theorem remainder_le (center perturbation radius : ℝ) (hperturbation : |perturbation| ≤ radius) :
    remainder center perturbation ≤ max 0 (radius - |center|) := by
  have hbounds := abs_le.mp hperturbation
  unfold remainder
  simp only [Activation.Math.reluSpec_eq_max]
  by_cases hcenter : 0 < center
  · rw [ite_eq_left hcenter, max_eq_left hcenter.le, abs_of_pos hcenter]
    by_cases hpositive : 0 ≤ center + perturbation
    · rw [max_eq_left hpositive]
      linarith [le_max_left 0 (radius - center)]
    · rw [max_eq_right (le_of_not_ge hpositive)]
      linarith [le_max_right 0 (radius - center)]
  · rw [ite_eq_right hcenter, max_eq_right (le_of_not_gt hcenter),
      abs_of_nonpos (le_of_not_gt hcenter)]
    by_cases hpositive : 0 ≤ center + perturbation
    · rw [max_eq_left hpositive]
      linarith [le_max_right 0 (radius - -center)]
    · rw [max_eq_right (le_of_not_ge hpositive)]
      linarith [le_max_left 0 (radius - -center)]

/-- Coarser form of `remainder_le`: at most `radius` when `|center| ≤ radius`, else zero. -/
theorem remainder_le_boundary (center perturbation radius : ℝ)
    (hperturbation : |perturbation| ≤ radius) :
    remainder center perturbation ≤ if |center| ≤ radius then radius else 0 := by
  apply (remainder_le center perturbation radius hperturbation).trans
  by_cases h : |center| ≤ radius
  · rw [ite_eq_left h]
    exact max_le ((abs_nonneg perturbation).trans hperturbation)
      (sub_le_self _ (abs_nonneg _))
  · rw [ite_eq_right h]
    exact max_le le_rfl (sub_nonpos.mpr (le_of_not_ge h))

/-- Subtracting TorchLean's ReLU JVP from the change in output leaves `remainder` in every
coordinate, so `remainder` and the JVP use the same slope at zero. -/
theorem relu_remainder_eq {shape : Shape} (actual reference : Tensor ℝ shape) :
    Tensor.subSpec
        (Tensor.subSpec (reluCorrect.op.forward actual) (reluCorrect.op.forward reference))
        (reluCorrect.jvp reference (Tensor.subSpec actual reference)) =
      Tensor.map2Spec remainder reference (Tensor.subSpec actual reference) := by
  apply Tensor.Internal.Rep.ext
  intro coordinate
  have hshift : reference coordinate + (actual coordinate - reference coordinate) =
      actual coordinate := by ring
  simp only [reluCorrect, Spec.reluOp, Activation.reluDerivSpec, Tensor.subSpec,
    Tensor.mulSpec, Tensor.map2Spec, Tensor.map, Tensor.Internal.Rep.zipWith_apply,
    Tensor.Internal.Rep.map_apply, remainder, Activation.Math.reluDerivSpec]
  by_cases h : 0 < reference coordinate <;> simp [h, hshift]

/-- Paired with a cotangent `seed`, the ReLU remainder is at most
`∑ |seed| * max 0 (radius - |reference|)` in absolute value, for any tensor shape. -/
theorem relu_error_le {shape : Shape} (actual reference radius seed : Tensor ℝ shape)
    (hperturbation : ∀ coordinate : shape.Coord,
      |actual coordinate - reference coordinate| ≤ radius coordinate) :
    |Proofs.TensorAlgebra.dot
      (Tensor.subSpec
        (Tensor.subSpec (reluCorrect.op.forward actual) (reluCorrect.op.forward reference))
        (reluCorrect.jvp reference (Tensor.subSpec actual reference))) seed| ≤
      ∑ coordinate : shape.Coord, |seed coordinate| *
        max 0 (radius coordinate - |reference coordinate|) := by
  classical
  rw [relu_remainder_eq, ← Spec.dot_eq_tensorAlgebra_dot, Spec.dot,
    Spec.sum_spec_eq_coord_sum]
  simp only [Tensor.mulSpec, Tensor.subSpec, Tensor.map2Spec,
    Tensor.Internal.Rep.zipWith_apply]
  calc
    _ ≤ ∑ coordinate : shape.Coord,
        |remainder (reference coordinate) (actual coordinate - reference coordinate) *
          seed coordinate| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ coordinate : shape.Coord, |seed coordinate| *
        remainder (reference coordinate) (actual coordinate - reference coordinate) := by
      apply Finset.sum_congr rfl
      intro coordinate _
      rw [abs_mul, abs_of_nonneg (remainder_nonneg _ _), mul_comm]
    _ ≤ _ := Finset.sum_le_sum fun coordinate _ => mul_le_mul_of_nonneg_left
      (remainder_le _ _ _ (hperturbation coordinate)) (abs_nonneg _)

/-- If the perturbation is centered under `law` for every retained state, the linear term
cancels, and the sensitivity-weighted mean change in ReLU output is at most
`∑ wᵢ * |sᵢ| * max 0 (rᵢ - |cᵢ|)` in absolute value. Retained weights need not sum to one.
When they do, both sides are expectations under the finite conditional law. -/
theorem mean_error_le {retained discarded : Nat}
    (weights center radius sensitivity : Tensor ℝ [retained])
    (law perturbation : Tensor ℝ [retained, discarded])
    (hweights : ∀ i : Fin retained, 0 ≤ weights.getScalar i)
    (hlaw : ∀ (i : Fin retained) (j : Fin discarded), 0 ≤ law[(i, j)])
    (hmass : ∀ i : Fin retained, ∑ j : Fin discarded, law[(i, j)] = 1)
    (hmean : ∀ i : Fin retained,
      ∑ j : Fin discarded, law[(i, j)] * perturbation[(i, j)] = 0)
    (hperturbation : ∀ (i : Fin retained) (j : Fin discarded),
      |perturbation[(i, j)]| ≤ radius.getScalar i) :
    |∑ i : Fin retained, weights.getScalar i * sensitivity.getScalar i *
      ((∑ j : Fin discarded, law[(i, j)] *
        Activation.Math.reluSpec (center.getScalar i + perturbation[(i, j)])) -
          Activation.Math.reluSpec (center.getScalar i))| ≤
      ∑ i : Fin retained, weights.getScalar i * |sensitivity.getScalar i| *
        max 0 (radius.getScalar i - |center.getScalar i|) := by
  classical
  let residual := fun i : Fin retained => ∑ j : Fin discarded,
    law[(i, j)] * remainder (center.getScalar i) perturbation[(i, j)]
  let budget := fun i : Fin retained =>
    max 0 (radius.getScalar i - |center.getScalar i|)
  have heq (i : Fin retained) :
      (∑ j : Fin discarded, law[(i, j)] *
        Activation.Math.reluSpec (center.getScalar i + perturbation[(i, j)])) -
          Activation.Math.reluSpec (center.getScalar i) = residual i := by
    dsimp [residual]
    by_cases h : 0 < center.getScalar i <;>
      simp [remainder, h, mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hmass, hmean]
  have hnonneg (i : Fin retained) : 0 ≤ residual i :=
    Finset.sum_nonneg fun j _ => mul_nonneg (hlaw i j) (remainder_nonneg _ _)
  have hbound (i : Fin retained) : residual i ≤ budget i := by
    calc
      _ ≤ ∑ j : Fin discarded, law[(i, j)] * budget i :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left
          (remainder_le _ _ _ (hperturbation i j)) (hlaw i j)
      _ = _ := by rw [← Finset.sum_mul, hmass, one_mul]
  simp_rw [heq]
  calc
    _ ≤ ∑ i : Fin retained,
        |weights.getScalar i * sensitivity.getScalar i * residual i| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i : Fin retained, weights.getScalar i * |sensitivity.getScalar i| * residual i := by
      apply Finset.sum_congr rfl
      intro i _
      rw [abs_mul, abs_mul, abs_of_nonneg (hweights i), abs_of_nonneg (hnonneg i)]
    _ ≤ _ := Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hbound i)
      (mul_nonneg (hweights i) (abs_nonneg _))

end HeuristicEstimation.Boundary
