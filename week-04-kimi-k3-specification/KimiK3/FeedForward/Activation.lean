/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.Common
public import NN.Spec.Layers.Activation
public import Mathlib.Analysis.SpecialFunctions.Sigmoid
import NN.Proofs.Gradients.Activation

/-!
# SiTU-GLU

SiTU-GLU soft-caps its gate and up branches independently (Eq. 12 and Appendix B of the Kimi K3
report). The real-valued bound reuses Mathlib's sigmoid and tanh inequalities.

Reference: https://arxiv.org/abs/2607.24653, Section 2.3.2 and Appendix B.
-/

@[expose] public section

namespace KimiK3

open Spec TorchLean
open Tensor

namespace SiTU

/-- Smoothly cap a scalar at magnitude `cap` using `cap * tanh (x / cap)`. -/
def softCap {α : Type} [Context α] (cap x : α) : α :=
  cap * MathFunctions.tanh (x / cap)

/--
SiTU-GLU on one coordinate. The sigmoid sees the uncapped gate preactivation, and the linear
factors of both GLU branches are soft-capped.
-/
def scalar {α : Type} [Context α] (gateCap upCap gate up : α) : α :=
  (softCap gateCap gate * Activation.Math.sigmoidSpec gate) * softCap upCap up

/-- SiTU-GLU applied coordinatewise to gate and up projections. -/
def vector {α : Type} [Storage α] [Context α] {n : Nat} (gateCap upCap : α)
    (gate up : Tensor α [n]) : Tensor α [n] :=
  Tensor.stackLeading (fun i => Tensor.scalar (scalar gateCap upCap (Tensor.getScalar gate i)
    (Tensor.getScalar up i)))

/-- TorchLean's generic sigmoid agrees with Mathlib's `Real.sigmoid`. -/
private theorem sigmoidSpec_real (x : ℝ) : Activation.Math.sigmoidSpec x = Real.sigmoid x := by
  simpa [Real.sigmoid] using Proofs.sigmoid_eq_inv_exp x

/-- At scalar type `ℝ`, the generic `MathFunctions.tanh` is definitionally `Real.tanh`. -/
private theorem tanh_real (x : ℝ) : MathFunctions.tanh x = Real.tanh x := by
  rfl

/--
With nonnegative caps, SiTU-GLU is bounded by `gateCap * upCap` for all real preactivations, the
uniform bound stated in the paper.
-/
theorem abs_scalar_le_mul_caps {gateCap upCap : ℝ}
    (hGate : 0 ≤ gateCap) (hUp : 0 ≤ upCap) (gate up : ℝ) :
    |scalar gateCap upCap gate up| ≤ gateCap * upCap := by
  have hTanhGate : |Real.tanh (gate / gateCap)| ≤ 1 := (Real.abs_tanh_lt_one _).le
  have hTanhUp : |Real.tanh (up / upCap)| ≤ 1 := (Real.abs_tanh_lt_one _).le
  have hSigmoid : |Real.sigmoid gate| ≤ 1 := by
    rw [abs_of_nonneg (Real.sigmoid_nonneg gate)]
    exact Real.sigmoid_le_one gate
  have hSoftGate : |softCap gateCap gate| ≤ gateCap := by
    rw [softCap, tanh_real, abs_mul, abs_of_nonneg hGate]
    exact mul_le_of_le_one_right hGate hTanhGate
  have hSoftUp : |softCap upCap up| ≤ upCap := by
    rw [softCap, tanh_real, abs_mul, abs_of_nonneg hUp]
    exact mul_le_of_le_one_right hUp hTanhUp
  rw [scalar, sigmoidSpec_real, abs_mul, abs_mul]
  calc
    |softCap gateCap gate| * |Real.sigmoid gate| * |softCap upCap up|
        ≤ gateCap * 1 * upCap := by gcongr
    _ = gateCap * upCap := by ring

/-- With Kimi K3's caps `4` and `25`, every coordinate is bounded by `100`. -/
theorem paper_caps_bound (gate up : ℝ) : |scalar 4 25 gate up| ≤ 100 := by
  have h := abs_scalar_le_mul_caps (gateCap := (4 : ℝ)) (upCap := (25 : ℝ))
    (by positivity) (by positivity) gate up
  norm_num at h ⊢
  exact h

/-- SiTU-GLU written with elementary tensor operations, as the executable graph computes it, equals
`vector`.

The graph multiplies by the reciprocal of each cap instead of dividing. The identity holds even for
a zero cap, because division in a field is total and `x / c = x * c⁻¹`.
-/
theorem expanded_eq_vector {n : Nat} (gateCap upCap : ℝ)
    (gate up : Tensor ℝ [n]) :
    Tensor.mulSpec
      (Tensor.mulSpec
        (Tensor.mulSpec
          (Activation.tanhSpec (Tensor.mulSpec gate (Tensor.full [n] gateCap⁻¹)))
          (Tensor.full [n] gateCap))
        (Activation.sigmoidSpec gate))
      (Tensor.mulSpec
        (Activation.tanhSpec (Tensor.mulSpec up (Tensor.full [n] upCap⁻¹)))
        (Tensor.full [n] upCap)) =
      vector gateCap upCap gate up := by
  apply Tensor.ext_vector
  intro index
  simp [Tensor.mulSpec, Activation.tanhSpec, Activation.sigmoidSpec,
    Activation.Math.tanhSpec, vector, Tensor.stackLeading, scalar, softCap, div_eq_mul_inv]
  ring

end SiTU

end KimiK3
