/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Nemotron.Cache.Error
public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.Error
public import FloatLib.Numerics.Quantization.Deterministic.Rational

/-!
# Reusing FloatLib rounding

Tensor rounding on FloatLib's real-valued IEEE grid, with FloatLib's proved half-ULP bound.
This grid has no upper exponent limit, so relating it to executable finite bits needs FloatLib's
finiteness refinement hypotheses. It rounds to nearest even, while the released Ultra cache
rounds stochastically (`Nemotron.Cache.Stochastic`).
-/

@[expose] public section

namespace Nemotron.Cache

open Spec TorchLean KimiK3
open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

/-- Round every coordinate to nearest even on FloatLib's real-valued grid for `fmt`. -/
noncomputable def roundTensor {shape : Shape} (fmt : FloatFormat)
    (x : Tensor ℝ shape) : Tensor ℝ shape :=
  Tensor.map (Model.roundAt fmt) x

/-- If every coordinate's half-ULP is at most `δ`, rounding is `Within δ`. An absolute
bound covers subnormals and needs no relative error bound near zero. -/
theorem roundTensor_within {shape : Shape} (fmt : FloatFormat)
    (x : Tensor ℝ shape) (δ : ℝ)
    (hδ : ∀ coordinate, Model.epsilonAt fmt (x coordinate) ≤ δ) :
    Within δ (roundTensor fmt x) x := by
  intro coordinate
  simpa [roundTensor, Tensor.map] using
    (Model.abs_roundAt_sub_le fmt (x coordinate)).trans (hδ coordinate)

namespace Counterexample

/-- FloatLib's nearest-even rounding to an integer, read back as a rational. -/
def quantize (x : ℚ) : ℚ := (FloatLib.Numerics.roundRatEven x : ℚ)

/-- Exact rational addition, so storage is the only place that rounds. -/
def addStep (state input : ℚ) : ℚ × Unit := (state + input, ())

/-- Integer rounding is within `1/2` everywhere. -/
theorem quantize_error (x : ℚ) : |quantize x - x| ≤ (1 : ℚ) / 2 :=
  FloatLib.Numerics.roundRatEven_error_le_half x

/-- Four nearest-even writes can end closer to the exact sum than two, through cancellation.
So "fewer sequential quantization steps" does not always mean less error. This is a fixed-grid
example, not a counterexample to the report's measured results. -/
theorem fewer_writes_can_increase_error :
    |quantizedChunks addStep quantize 0
        [[-9 / 10], [-9 / 10], [-1 / 2], [-1 / 10]] - (-12 / 5)| <
      |quantizedChunks addStep quantize 0
        [[-9 / 10, -9 / 10], [-1 / 2, -1 / 10]] - (-12 / 5)| := by
  decide +kernel

/-- Two half-unit updates written separately both round to the even value `0`. One write of
their sum keeps it. -/
theorem repeated_ties_lose_updates :
    quantizedChunks addStep quantize 0 [[1 / 2], [1 / 2]] = 0 ∧
      quantizedChunks addStep quantize 0 [[1 / 2, 1 / 2]] = 1 := by
  decide +kernel

end Counterexample

end Nemotron.Cache
