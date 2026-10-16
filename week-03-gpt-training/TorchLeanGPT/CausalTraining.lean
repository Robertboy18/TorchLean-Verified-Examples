/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.Model
public import NN.Proofs.Models.Attention.CausalMask
public import Mathlib.Algebra.BigOperators.Fin

/-!
# Weighted rows and causal attention

Hard causal attention gives every strict-future position zero forward weight and zero backward
score gradient, and rows with zero weight make no direct contribution to a weighted objective. Both
facts are stated here, for the hard-mask attention specification and for weighted sums of row
losses.

The next-token and target-mask theorems sit in `Run.lean`, next to the two batch builders the
trainer calls (`causal_window_target_is_next_token`, `causal_record_target_is_next_token`, and
`causal_window_mask_is_next_target`).
-/

@[expose] public section

open TorchLean
open scoped BigOperators

namespace TorchLeanGPT

open NN.Proofs.Models.Attention
open Spec

/--
Two row-wise quantities that agree wherever the weight is nonzero have the same weighted sum.

For assistant-only training this says that zero-weight rows make no direct contribution to the
objective. It applies to losses, or to derivative values proved equal elsewhere, and does not
differentiate the runtime loss.
-/
theorem weighted_rows_eq_of_eq_on_support
    {n : Nat} (weights left right : Fin n → ℝ)
    (hEqual : ∀ i, weights i ≠ 0 → left i = right i) :
    ∑ i, weights i * left i = ∑ i, weights i * right i := by
  apply Finset.sum_congr rfl
  intro i _
  by_cases hWeight : weights i = 0
  · simp [hWeight]
  · rw [hEqual i hWeight]

/--
A coordinate with zero softmax weight has zero score derivative under TorchLean's softmax backward
equation.

This local fact carries hard masking from the forward pass over to the backward pass.
-/
private theorem softmax_backward_zero_at_zero_weight
    {n : Nat}
    (weights dWeights : Tensor ℝ (.dim n .scalar))
    (j : Fin n)
    (hWeight : Tensor.getScalar weights j = 0) :
    Tensor.getScalar
      (Spec.softmaxBackwardFromWeightsSpec weights dWeights) j = 0 := by
  simp only [Spec.softmaxBackwardFromWeightsSpec, Tensor.mulSpec,
    Tensor.getScalar_map2Spec, hWeight, zero_mul]

/-- Row-wise version for matrices: a zero weight coordinate has zero score derivative. -/
private theorem softmax_backward_get2_zero_at_zero_weight
    {m n : Nat}
    (weights dWeights : Tensor ℝ (.dim m (.dim n .scalar)))
    (i : Fin m) (j : Fin n)
    (hWeight : Spec.get2 weights i j = 0) :
    Spec.get2
      (Spec.softmaxBackwardFromWeightsSpec weights dWeights) i j = 0 := by
  simpa only [Spec.get2_eq_getScalar_get, Spec.softmaxBackwardFromWeightsSpec,
    Spec.get_dim] using
    softmax_backward_zero_at_zero_weight (weights.unstack i) (dWeights.unstack i) j hWeight

/--
Strict-future attention coordinates carry neither forward weight nor backward score gradient.

Scores and incoming weight gradients are arbitrary, so the zeros come from the causal hard mask and
not from the values of any trained checkpoint.
-/
theorem causal_attention_blocks_future_forward_and_backward
    {context : Nat}
    (scores dWeights : Tensor ℝ (.dim context (.dim context .scalar)))
    (i j : Fin context)
    (future : i.val < j.val) :
    let weights :=
      Spec.hardMaskedSoftmaxSpec scores (Spec.causalMask context)
    Spec.get2 weights i j = 0 ∧
      Spec.get2
        (Spec.softmaxBackwardFromWeightsSpec weights dWeights) i j = 0 := by
  dsimp
  have hForward :=
    hardMaskedSoftmaxSpec_causal_future_zero scores i j future
  exact ⟨hForward,
    softmax_backward_get2_zero_at_zero_weight
      (Spec.hardMaskedSoftmaxSpec scores (Spec.causalMask context))
      dWeights i j hForward⟩

end TorchLeanGPT
