/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Model
public import NN.Spec.Layers.Loss

/-!
# Masked next-token pretraining

Average the next-token loss over a nonempty set of supervised positions. Text targets can be
selected even when the input includes visual embeddings or padding. This file also states a
logit-gradient formula and proves its values on supervised and masked rows.

Reference: Kimi K3 technical report, Section 2.5 and Sections 3--4, pp. 10--14.
https://arxiv.org/abs/2607.24653.
-/

@[expose] public section

namespace KimiK3

open Spec TorchLean
open Tensor
open scoped BigOperators

namespace Pretraining

/-- Negative log-likelihood of one target token, evaluated from unnormalized vocabulary logits. -/
noncomputable def tokenLoss {vocabSize : Nat}
    (logits : Tensor ℝ [vocabSize]) (target : Fin vocabSize) : ℝ :=
  -Tensor.getScalar (Activation.logSoftmaxVecSpec logits) target

/-- Mean next-token loss over the positions selected by `supervised`.

The model may receive both text and visual embeddings, but its targets are vocabulary tokens.
Visual input positions, padding, and input-only protocol tokens are left out of `supervised`, so
their entries in `targets` never reach the loss.
-/
noncomputable def nextTokenLoss {sequenceLength vocabSize : Nat}
    (logits : Tensor ℝ [sequenceLength, vocabSize])
    (targets : Fin sequenceLength → Fin vocabSize)
    (supervised : Finset (Fin sequenceLength)) (_hSupervised : supervised.Nonempty) : ℝ :=
  (∑ position ∈ supervised, tokenLoss (Spec.get logits position) (targets position)) /
    supervised.card

/-- Closed-form logit-gradient formula for the masked next-token objective.

Rows omitted from `supervised` have zero gradient. A supervised row uses the standard
`softmax(logits) - oneHot(target)` derivative, divided by the number of supervised positions.

No theorem here proves that this is the derivative or VJP of `nextTokenLoss`.
-/
noncomputable def logitGradientFormula {sequenceLength vocabSize : Nat}
    (logits : Tensor ℝ [sequenceLength, vocabSize])
    (targets : Fin sequenceLength → Fin vocabSize)
    (supervised : Finset (Fin sequenceLength)) (_hSupervised : supervised.Nonempty) :
    Tensor ℝ [sequenceLength, vocabSize] :=
  Tensor.stackLeading fun position =>
    if position ∈ supervised then
      scaleSpec
        (crossEntropyLogitsDerivSpec 0 (Spec.get logits position)
          (TorchLean.Tensor.oneHot vocabSize (targets position)))
        (1 / supervised.card)
    else
      Tensor.zeros [vocabSize]

/-- On every supervised position, the formula uses TorchLean's stable
cross-entropy-on-logits derivative, with the outer mean reduction applied. -/
theorem logitGradientFormula_at_supervised {sequenceLength vocabSize : Nat}
    (logits : Tensor ℝ [sequenceLength, vocabSize])
    (targets : Fin sequenceLength → Fin vocabSize)
    (supervised : Finset (Fin sequenceLength)) (hSupervised : supervised.Nonempty)
    (position : Fin sequenceLength) (hPosition : position ∈ supervised) :
    Spec.get (logitGradientFormula logits targets supervised hSupervised) position =
      scaleSpec
        (crossEntropyLogitsDerivSpec 0 (Spec.get logits position)
          (TorchLean.Tensor.oneHot vocabSize (targets position)))
        (1 / supervised.card) := by
  simp only [logitGradientFormula, Tensor.stackLeading, Spec.get_dim]
  rw [ite_eq_left hPosition]

/-- At every masked position the formula is zero. -/
theorem logitGradientFormula_at_masked {sequenceLength vocabSize : Nat}
    (logits : Tensor ℝ [sequenceLength, vocabSize])
    (targets : Fin sequenceLength → Fin vocabSize)
    (supervised : Finset (Fin sequenceLength)) (hSupervised : supervised.Nonempty)
    (position : Fin sequenceLength) (hPosition : position ∉ supervised) :
    Spec.get (logitGradientFormula logits targets supervised hSupervised) position =
      Tensor.zeros [vocabSize] := by
  simp only [logitGradientFormula, Tensor.stackLeading, Spec.get_dim]
  rw [ite_eq_right hPosition]

end Pretraining

namespace LanguageModel

/-- Evaluate the masked next-token objective on the logits of the complete language backbone, so
the pretraining loss belongs to a concrete model. -/
noncomputable def pretrainingLoss {cfg : TextConfig} {decayRank sequenceLength : Nat}
    (model : LanguageModel ℝ cfg decayRank)
    (embeddings : Tensor ℝ [sequenceLength, cfg.hiddenDim])
    (targets : Fin sequenceLength → Fin cfg.vocabSize)
    (supervised : Finset (Fin sequenceLength)) (hSupervised : supervised.Nonempty) : ℝ :=
  Pretraining.nextTokenLoss (model.logits embeddings) targets supervised hSupervised

end LanguageModel

end KimiK3
