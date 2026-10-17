/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import HeuristicEstimation.Sketch
public import FloatLib.Numerics.Quantization.Deterministic.Rational

/-!
# Finite-precision sketch coefficients

FloatLib supplies executable nearest-even integer rounding. Scaling its proved half-unit
error gives a coefficient encoder for any positive rational grid spacing. This is a grid
quantizer, not a claim that finite-exponent IEEE operations are exact.
-/

@[expose] public section

namespace HeuristicEstimation.Encoding

open TorchLean Sketch
open scoped BigOperators

/-- Store a coefficient on a rational grid using FloatLib's nearest-even rounding. -/
def round (spacing value : Rat) : Rat :=
  spacing * (FloatLib.Numerics.roundRatEven (value / spacing) : Rat)

/-- Every coefficient is within half the chosen spacing, including ties. -/
theorem round_error {spacing : Rat} (hspacing : 0 < spacing) (value : Rat) :
    |round spacing value - value| ≤ spacing / 2 := by
  have h := mul_le_mul_of_nonneg_left
    (FloatLib.Numerics.roundRatEven_error_le_half (value / spacing)) hspacing.le
  calc
    _ = |spacing * ((FloatLib.Numerics.roundRatEven (value / spacing) : Rat) -
        value / spacing)| := by
      congr 1
      dsimp [round]
      field_simp
    _ = spacing * |(FloatLib.Numerics.roundRatEven (value / spacing) : Rat) -
        value / spacing| := by rw [abs_mul, abs_of_pos hspacing]
    _ ≤ _ := by simpa only [div_eq_mul_inv, one_mul] using h

/-- Encoding all coefficients adds at most a half-spacing squared per coordinate. -/
theorem energy_round_error {d : Nat} {spacing : Rat} (hspacing : 0 < spacing)
    (values : Tensor Rat [d]) :
    energy (Tensor.ofFn fun i => round spacing (values.getScalar i) - values.getScalar i) ≤
      (d : Rat) * (spacing / 2) ^ 2 := by
  calc
    _ ≤ ∑ _i : Fin d, (spacing / 2) ^ 2 := by
      apply Finset.sum_le_sum
      intro i _
      simp only [Tensor.getScalar_ofFn]
      have h := (sq_le_sq₀ (abs_nonneg _) (by positivity : 0 ≤ spacing / 2)).mpr
        (round_error hspacing (values.getScalar i))
      simpa only [sq_abs] using h
    _ = _ := by simp

/-- Projection error and coefficient-rounding error combine before applying the same
downstream certificate. This does not assume that their cross-term vanishes. -/
theorem energy_add_le {α : Type} [Storage α] [CommRing α] [LinearOrder α]
    [IsStrictOrderedRing α] {d : Nat} (residual rounding : Tensor α [d]) :
    energy (Tensor.ofFn fun i => residual.getScalar i + rounding.getScalar i) ≤
      2 * energy residual + 2 * energy rounding := by
  simp only [energy, Tensor.getScalar_ofFn, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro i _
  nlinarith [sq_nonneg (residual.getScalar i - rounding.getScalar i)]

/-- The same Gram certificate bounds a residual after coefficient encoding. The two error
sources are combined without assuming independence or orthogonality. -/
theorem encoded_error_le {contexts d : Nat} (weights : Tensor Rat [contexts])
    (tails : Tensor Rat [contexts, d]) (bound : Rat) (hbound : 0 ≤ bound)
    (hcheck : check (gram weights tails) bound = true) {spacing : Rat}
    (hspacing : 0 < spacing) (projection coefficients : Tensor Rat [d]) :
    error weights tails (Tensor.ofFn fun i => projection.getScalar i +
      (round spacing (coefficients.getScalar i) - coefficients.getScalar i)) ≤
        2 * bound * energy projection + 2 * bound * (d : Rat) * (spacing / 2) ^ 2 := by
  let rounding : Tensor Rat [d] := Tensor.ofFn fun i =>
    round spacing (coefficients.getScalar i) - coefficients.getScalar i
  have hadd := energy_add_le projection rounding
  have hround := energy_round_error hspacing coefficients
  calc
    _ ≤ bound * energy (Tensor.ofFn fun i => projection.getScalar i + rounding.getScalar i) := by
      simpa only [rounding, Tensor.getScalar_ofFn] using
        error_le_of_check weights tails bound hcheck
          (Tensor.ofFn fun i => projection.getScalar i + rounding.getScalar i)
    _ ≤ bound * (2 * energy projection + 2 * energy rounding) :=
      mul_le_mul_of_nonneg_left hadd hbound
    _ ≤ _ := by
      have h := mul_le_mul_of_nonneg_left hround (by positivity : 0 ≤ 2 * bound)
      dsimp [rounding] at *
      nlinarith

end HeuristicEstimation.Encoding
