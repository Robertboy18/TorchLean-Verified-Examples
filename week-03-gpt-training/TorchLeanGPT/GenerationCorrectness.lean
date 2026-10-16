/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import NN.MLTheory.Proofs.Verification.Robustness.LipschitzCertified

/-!
# Numerically stable autoregressive generation

A one-step logit bound matters only if it composes through autoregressive decoding, where every
chosen token becomes part of the next model input. This file proves that composition: if an
approximate implementation stays within half of the ideal winner's logit margin at every history
that greedy decoding reaches, greedy generation produces the same complete continuation.

Here `ideal` is the real-valued model being specified, and `approximate` may be cached CUDA
decoding, another backend, or a rounded model. How the error bound is obtained does not matter
here. A formally checked certificate would discharge the hypotheses, while a runtime comparison
(what the Week 3 executables do) is evidence.

The Week 3 generators subtract a repeat penalty before choosing a token (`--repeat-penalty`,
default 1.05 per occurrence among the last 64 ids). That penalty depends only on the history, so
the theorems cover penalized decoding when `ideal` and `approximate` are the penalized score
functions. Both paths subtract the same amounts, which leaves their difference unchanged and only
moves the winner and its margin.
-/

@[expose] public section

open _root_.Spec TorchLean

namespace TorchLeanGPT
namespace Generation

open NN.MLTheory.Proofs.Verification.Robustness
open NN.MLTheory.Robustness.Spec

/-- A next-token model assigns one real logit to every vocabulary item after a token history. -/
abbrev LogitModel (vocab : Nat) :=
  List Nat → Tensor ℝ (.dim vocab .scalar)

/-- Greedy autoregressive generation, returning only the newly generated suffix. -/
noncomputable def greedy (model : LogitModel vocab) : Nat → List Nat → List Nat
  | 0, _history => []
  | steps + 1, history =>
      let token := argmaxClassifier (model history)
      token :: greedy model steps (history ++ [token])

/--
A numerical certificate that preserves the greedy choice at one token history.

`winner` leads every other ideal logit by at least `margin`, and the approximate logits lie within
`error` of the ideal ones in the infinity norm. Each logit moves by at most `error`, so a gap of
`margin` shrinks by at most `2 * error`, and `2 * error < margin` keeps it positive.
-/
def StepCertified
    (ideal approximate : LogitModel vocab) (history : List Nat) : Prop :=
  ∃ winner : Fin vocab, ∃ margin error : ℝ,
    0 < margin ∧
    HasLogitMargin (ideal history) winner margin ∧
    2 * error < margin ∧
    tensorDistance (tensorLinfNorm (α := ℝ)) (ideal history) (approximate history) ≤ error

/-- The one-step certificate preserves the next greedy token. -/
theorem argmax_eq_of_step_certified
    (ideal approximate : LogitModel vocab) (history : List Nat)
    (h : StepCertified ideal approximate history) :
    argmaxClassifier (approximate history) = argmaxClassifier (ideal history) := by
  rcases h with ⟨winner, margin, error, hMarginPositive, hMargin, hErrorSmall, hError⟩
  have hApprox := argmaxClassifier_eq_of_linf_distance_lt_half_margin
    (hmargin := hMargin) (hδ := hErrorSmall) (hdist := hError)
  have hIdeal := argmaxClassifier_eq_of_hasLogitMargin
    (hm := hMarginPositive) (hmargin := hMargin)
  exact hApprox.trans hIdeal.symm

/--
Per-step certificates along the ideal greedy path.

Only reached histories need a certificate. Once the current tokens agree, both models move to the
same next history, where the rest of the certificate applies.
-/
noncomputable def RolloutCertified
    (ideal approximate : LogitModel vocab) : Nat → List Nat → Prop
  | 0, _history => True
  | steps + 1, history =>
      StepCertified ideal approximate history ∧
      RolloutCertified ideal approximate steps
        (history ++ [argmaxClassifier (ideal history)])

/-- Pointwise agreement of greedy choices gives agreement of the complete generated suffix. -/
theorem greedy_eq_of_argmax_eq
    (ideal approximate : LogitModel vocab)
    (hargmax : ∀ history, argmaxClassifier (approximate history) =
      argmaxClassifier (ideal history)) :
    ∀ steps history, greedy approximate steps history = greedy ideal steps history := by
  intro steps
  induction steps with
  | zero =>
      intro history
      rfl
  | succ steps ih =>
      intro history
      simp only [greedy]
      rw [hargmax history]
      exact congrArg (List.cons (argmaxClassifier (ideal history)))
        (ih (history ++ [argmaxClassifier (ideal history)]))

/-- Certificates on the reached prefixes preserve the entire greedy continuation. -/
theorem greedy_eq_of_rollout_certified
    (ideal approximate : LogitModel vocab) :
    ∀ steps history,
      RolloutCertified ideal approximate steps history →
      greedy approximate steps history = greedy ideal steps history := by
  intro steps
  induction steps with
  | zero =>
      intro history _hCertified
      rfl
  | succ steps ih =>
      intro history hCertified
      have hChoice := argmax_eq_of_step_certified ideal approximate history hCertified.1
      simp only [greedy]
      rw [hChoice]
      exact congrArg (List.cons (argmaxClassifier (ideal history))) (ih _ hCertified.2)

/--
Logit error below half of the ideal winning margin at every history preserves an entire greedy
generation.

Its hypotheses range over all histories, reached or not, while `greedy_eq_of_rollout_certified`
needs them only along the ideal greedy path.
-/
theorem greedy_eq_of_linf_error_lt_half_margin
    (ideal approximate : LogitModel vocab)
    (winner : List Nat → Fin vocab)
    (margin error : List Nat → ℝ)
    (hmarginPositive : ∀ history, 0 < margin history)
    (hmargin : ∀ history,
      HasLogitMargin (ideal history) (winner history) (margin history))
    (herrorSmall : ∀ history, 2 * error history < margin history)
    (herror : ∀ history,
      tensorDistance (tensorLinfNorm (α := ℝ)) (ideal history) (approximate history) ≤
        error history) :
    ∀ steps history, greedy approximate steps history = greedy ideal steps history := by
  apply greedy_eq_of_argmax_eq
  intro history
  have hApprox := argmaxClassifier_eq_of_linf_distance_lt_half_margin
    (hmargin := hmargin history)
    (hδ := herrorSmall history)
    (hdist := herror history)
  have hIdeal := argmaxClassifier_eq_of_hasLogitMargin
    (hm := hmarginPositive history)
    (hmargin := hmargin history)
  exact hApprox.trans hIdeal.symm

end Generation
end TorchLeanGPT
