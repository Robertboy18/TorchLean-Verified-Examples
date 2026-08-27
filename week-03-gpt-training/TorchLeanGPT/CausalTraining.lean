/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.Model
public import NN.Proofs.Models.Attention.CausalMask
public import Mathlib.Algebra.BigOperators.Fin

/-!
# Next-token training and causal attention

The training rows pair each input token with its successor. Hard causal attention then gives every
strict-future position exactly zero forward weight and zero backward score gradient. The results
below state these two properties for the data and attention definitions used by the runner.
-/

@[expose] public section

open TorchLean
open scoped BigOperators

namespace TorchLeanGPT

open NN.Proofs.Models.Attention
open Spec

/-- Input and next-token rows for one fixed-width causal-language-model window. -/
def causalLmTokenIdRows (context : Nat) (window : List Nat) (padId : Nat) :
    List Nat × List Nat :=
  ((List.range context).map fun position => window.getD position padId,
    (List.range context).map fun position => window.getD (position + 1) padId)

/-- Target-mask entries aligned with the next-token rows of one window. -/
def causalLmTargetMaskRow
    (context : Nat) (targetMask : Array Bool) (offset : Nat) : Array Bool :=
  (Array.range context).map fun position => targetMask.getD (offset + position + 1) false

/-- At every in-range training position, the target is the following corpus token. -/
theorem causal_window_target_is_next_token
    (context : Nat) (window : List Nat) (padId : Nat)
    (position : Nat) (hPosition : position < context) :
    (causalLmTokenIdRows context window padId).2.getD position padId =
      window.getD (position + 1) padId := by
  simp [causalLmTokenIdRows, hPosition]

/--
The instruction-tuning mask follows the prediction target rather than the input token.

For a window beginning at `offset`, row `position` predicts corpus token
`offset + position + 1`, so that is also the mask entry which decides whether the row contributes
to the loss.
-/
theorem causal_window_mask_is_next_target
    (context : Nat) (targetMask : Array Bool) (offset position : Nat)
    (hPosition : position < context) :
    (causalLmTargetMaskRow context targetMask offset).getD position false =
      targetMask.getD (offset + position + 1) false := by
  simp [causalLmTargetMaskRow, hPosition]

/--
Two row-wise quantities with the same values on the nonzero-weight support have the same weighted
sum.

The theorem applies to losses, or to derivative values established separately. It does not itself
differentiate the runtime loss implementation. For assistant-only training it states the algebraic
fact that zero-weight target rows make no direct contribution to the weighted objective.
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
A zero softmax weight also has zero score derivative under TorchLean's softmax backward equation.

This is the local fact that turns hard masking from a forward-only statement into a backward
information-flow statement.
-/
private theorem softmax_backward_zero_at_zero_weight
    {n : Nat}
    (weights dWeights : Spec.Tensor ℝ (.dim n .scalar))
    (j : Fin n)
    (hWeight : Spec.Tensor.getScalar weights j = 0) :
    Spec.Tensor.getScalar
      (Spec.softmaxBackwardFromWeightsSpec weights dWeights) j = 0 := by
  cases weights with
  | dim weightValues =>
      cases dWeights with
      | dim dWeightValues =>
          cases hW : weightValues j with
          | scalar w =>
              cases hD : dWeightValues j with
              | scalar dw =>
                  have hw : w = 0 := by
                    change Spec.Tensor.item
                      (Spec.get (Spec.Tensor.dim weightValues) j) = 0 at hWeight
                    rw [Spec.get_dim, hW] at hWeight
                    exact hWeight
                  simp [Spec.softmaxBackwardFromWeightsSpec, Spec.Tensor.getScalar,
                    Spec.Tensor.mulSpec, Spec.Tensor.subSpec, Spec.Tensor.map2Spec,
                    Spec.replicate, Spec.get, hW, hD, hw]

/-- The row-wise softmax backward equation preserves a zero matrix coordinate. -/
private theorem softmax_backward_get2_zero_at_zero_weight
    {m n : Nat}
    (weights dWeights : Spec.Tensor ℝ (.dim m (.dim n .scalar)))
    (i : Fin m) (j : Fin n)
    (hWeight : Spec.get2 weights i j = 0) :
    Spec.get2
      (Spec.softmaxBackwardFromWeightsSpec weights dWeights) i j = 0 := by
  cases weights with
  | dim weightRows =>
      cases dWeights with
      | dim dWeightRows =>
          have hRowWeight : Spec.Tensor.getScalar (weightRows i) j = 0 := by
            rw [Spec.get2_eq_getScalar_get, Spec.get_dim] at hWeight
            exact hWeight
          rw [Spec.get2_eq_getScalar_get]
          simp only [Spec.softmaxBackwardFromWeightsSpec, Spec.get_dim]
          simpa only [Spec.softmaxBackwardFromWeightsSpec] using
            softmax_backward_zero_at_zero_weight
              (weightRows i) (dWeightRows i) j hRowWeight

/--
Strict-future attention coordinates carry neither forward weight nor backward score gradient.

The theorem quantifies over arbitrary score values and arbitrary incoming weight gradients. Its
conclusion therefore comes from the causal hard mask, not from a numerical coincidence in one
trained checkpoint.
-/
theorem causal_attention_blocks_future_forward_and_backward
    {context : Nat}
    (scores dWeights : Spec.Tensor ℝ (.dim context (.dim context .scalar)))
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
