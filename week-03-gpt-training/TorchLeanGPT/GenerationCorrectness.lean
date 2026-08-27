/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import NN.MLTheory.Proofs.Verification.Robustness.LipschitzCertified

/-!
# Numerically stable autoregressive generation

A one-step logit bound is useful only if it composes through autoregressive decoding: every chosen
token becomes part of the next model input. This file proves that composition. If an approximate
implementation remains within half of the ideal winner's logit margin at every history, greedy
generation produces the same complete continuation.

Here `ideal` is the real-valued model whose behavior is being specified, while `approximate` may be
cached CUDA decoding, another backend, or a rounded model. The result is deliberately independent
of how an error bound is obtained. A runtime comparison supplies evidence; a formally checked
certificate supplies a theorem hypothesis. The current Week 3 executable performs the former and
does not claim the latter.
-/

@[expose] public section

open _root_.Spec

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
A numerical certificate sufficient to preserve the greedy choice at one token history.

`winner` has margin at least `margin` in the ideal logits, and the approximate logits lie within
`error` in the infinity norm. The strict inequality `2 * error < margin` prevents two independent
coordinate errors from reversing the winner and any competitor.
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

Only reached histories are certified. This is the useful adaptive statement: after equality of the
current token is proved, both models move to the same next history, where the remaining certificate
is applied.
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
Uniform logit error below half of the ideal winning margin preserves an entire greedy generation.

The hypotheses are checked at every token history because generation is adaptive: a token chosen at
one step becomes part of the next input. The conclusion says that numerical approximation cannot
change any generated token while all certified margins remain open.
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
