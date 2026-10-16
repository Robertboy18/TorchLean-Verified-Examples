/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Model
public import Mathlib.Data.Fin.VecNotation
public import NN.Proofs.Analysis.Softmax
public import NN.Proofs.Tensor.Basic.Algebra
public import NN.Tensor
public import Mathlib.Geometry.Convex.ConvexSpace.Defs
public import Mathlib.Data.EReal.Basic
import Mathlib.Algebra.Order.Group.MinMax
import NN.Proofs.Tensor.Basic.LinearAlgebra

/-!
# EAGLE-3 draft model and speculative sampling

Read three depth features from the target model, fuse them with a token embedding, and run
the draft transition. A separate finite-distribution calculation proves that accepted
proposal mass plus corrected rejection mass equals the target distribution. This probability
identity does not verify a deployed sampler's random-number generator or cache restoration.

Reference: Kimi K3 technical report, Section 2.5 and Sections 3--4, pp. 10--14.
https://arxiv.org/abs/2607.24653.
-/

@[expose] public section

namespace KimiK3

open Spec TorchLean
open Tensor
open scoped BigOperators

namespace Draft

namespace Features

/-- Read the first, fourth, and final AttnRes block outputs named by the K3 report. The hypothesis
rules out applying the EAGLE-3 head before four block outputs exist. -/
def ofBlockState {hiddenDim : Nat} (state : AttnRes.BlockState ℝ hiddenDim)
    (hFour : 4 ≤ state.blockOutputs.length) : Tensor ℝ [3, hiddenDim] :=
  let outputs := state.blockOutputs
  have hFourOutputs : 4 ≤ outputs.length := by simpa [outputs] using hFour
  have hNonempty : outputs ≠ [] := List.ne_nil_of_length_pos (by omega)
  Tensor.stackLeading ![outputs.get ⟨0, by omega⟩,
    outputs.get ⟨3, by omega⟩, outputs.getLast hNonempty]

end Features

namespace LanguageModel

/-- Extract the EAGLE-3 feature triplet for one token after a target-model forward pass.
Features come from that token's retained AttnRes state. -/
noncomputable def eagleFeaturesAt {cfg : TextConfig} {decayRank sequenceLength : Nat}
    (model : LanguageModel ℝ cfg decayRank)
    (embeddings : Tensor ℝ [sequenceLength, cfg.hiddenDim])
    (position : Fin sequenceLength)
    (hFour : 4 ≤ (model.forwardDepthStates embeddings position).blockOutputs.length) :
    Tensor ℝ [3, cfg.hiddenDim] :=
  Features.ofBlockState (model.forwardDepthStates embeddings position) hFour

end LanguageModel

namespace FeatureFusion

variable {hiddenDim : Nat}

/-- Multiplying a real vector by an all-zero square matrix gives the zero vector. -/
@[simp]
private theorem vecMatMul_fill_zero (values : Tensor ℝ [hiddenDim]) :
    vecMatMulSpec values (Tensor.zeros [hiddenDim, hiddenDim]) =
      Tensor.zeros [hiddenDim] := by
  apply Tensor.ext_vector
  intro column
  rw [getScalar_vec_mat_mul_spec]
  simp

/-- Multiplying a real row vector by the identity matrix returns that vector. -/
@[simp]
private theorem vecMatMul_identity (values : Tensor ℝ [hiddenDim]) :
    vecMatMulSpec values (identityTensorSpec hiddenDim) = values := by
  let result := vecMatMulSpec values (identityTensorSpec hiddenDim)
  have coordinates : Tensor.getScalar result = Tensor.getScalar values := by
    funext column
    simp [result, getScalar_vec_mat_mul_spec, get2_identityTensorSpec]
  exact Tensor.ext_vector (fun column => congrFun coordinates column)

variable {α : Type} [Storage α] [Context α]

/-- Concatenate-and-project written as a sum of three matrix products. -/
def forward (fusion : Tensor α [3, hiddenDim, hiddenDim]) (features : Tensor α [3, hiddenDim]) :
    Tensor α [hiddenDim] :=
  addSpec
    (addSpec
      (vecMatMulSpec (Spec.get features 0) (Spec.get fusion 0))
      (vecMatMulSpec (Spec.get features 1) (Spec.get fusion 1)))
    (vecMatMulSpec (Spec.get features 2) (Spec.get fusion 2))

/-- With the report's `[0 0 I]` initialization, fusion returns the high-level feature. -/
theorem initial_fusion_eq_high (features : Tensor ℝ [3, hiddenDim])
    (fusion : Tensor ℝ [3, hiddenDim, hiddenDim])
    (hLow : (Spec.get fusion 0) = Tensor.zeros [hiddenDim, hiddenDim])
    (hMiddle : (Spec.get fusion 1) = Tensor.zeros [hiddenDim, hiddenDim])
    (hHigh : (Spec.get fusion 2) = identityTensorSpec hiddenDim) :
    FeatureFusion.forward fusion features = (Spec.get features 2) := by
  rw [forward, hLow, hMiddle, hHigh]
  rw [vecMatMul_fill_zero, vecMatMul_fill_zero, vecMatMul_identity]
  simp

end FeatureFusion

/-- One-layer MTP/EAGLE-3 draft head. Its two input matrices represent the bias-free linear map
applied to the concatenation of the current token embedding and fused target feature. The decoder
layer has the same KDA-or-MLA and channel-mixer structure as a backbone layer. -/
structure Model (cfg : TextConfig) (decayRank : Nat) where
  /-- Three-source feature-fusion projection. -/
  featureFusion : Tensor ℝ [3, cfg.hiddenDim, cfg.hiddenDim]
  /-- Token embedding table shared with the draft input path. -/
  tokenEmbedding : Tensor ℝ [cfg.vocabSize, cfg.hiddenDim]
  /-- Projection applied to the current token embedding. -/
  tokenInputWeight : Tensor ℝ [cfg.hiddenDim, cfg.hiddenDim]
  /-- Projection applied to the fused target-model feature. -/
  featureInputWeight : Tensor ℝ [cfg.hiddenDim, cfg.hiddenDim]
  /-- Single causal decoder layer used by the draft model. -/
  decoderLayer : BackboneLayer ℝ cfg decayRank
  /-- Query used to read the draft layer's final AttnRes depth state. -/
  finalQuery : Tensor ℝ [cfg.hiddenDim]
  /-- RMS-normalization scale before the vocabulary projection. -/
  finalNormScale : Tensor ℝ [cfg.hiddenDim]
  /-- Projection from hidden states to draft-token logits. -/
  vocabularyHead : Tensor ℝ [cfg.hiddenDim, cfg.vocabSize]
  vocabSize_pos : 0 < cfg.vocabSize
  activeExperts_le : cfg.activeExperts ≤ cfg.numRoutedExperts

namespace Model

variable {cfg : TextConfig} {decayRank : Nat}

/-- Form a draft decoder input from a token embedding and one already fused feature. The feature is
the target model's three-level fusion at the first training-time-test step and the preceding draft
hidden state at every later step. -/
noncomputable def inputFromFeature (model : Model cfg decayRank) (token : Fin cfg.vocabSize)
    (feature : Tensor ℝ [cfg.hiddenDim]) : Tensor ℝ [cfg.hiddenDim] :=
  vecMatMulSpec (Spec.get model.tokenEmbedding token) model.tokenInputWeight +
    vecMatMulSpec feature model.featureInputWeight

/-- Form the first draft input by fusing the low-, middle-, and high-level target features. -/
noncomputable def input (model : Model cfg decayRank) (token : Fin cfg.vocabSize)
    (features : Tensor ℝ [3, cfg.hiddenDim]) : Tensor ℝ [cfg.hiddenDim] :=
  model.inputFromFeature token (FeatureFusion.forward model.featureFusion features)

/-- Causal KDA or MLA state retained by the single draft decoder layer. -/
abbrev CausalState (model : Model cfg decayRank) :=
  model.decoderLayer.sequence.CausalState

/-- Empty causal state at the beginning of a draft window. -/
noncomputable def initialState (model : Model cfg decayRank) : model.CausalState :=
  model.decoderLayer.sequence.initialState

/-- Advance the draft decoder from one fused feature, returning the sequence-mixer state for the
next draft position and the normalized hidden state. -/
noncomputable def stepHiddenFromFeature (model : Model cfg decayRank) (state : model.CausalState)
    (token : Fin cfg.vocabSize) (feature : Tensor ℝ [cfg.hiddenDim]) :
    model.CausalState × Tensor ℝ [cfg.hiddenDim] :=
  let depthState := BackboneLayer.initialDepthState (model.inputFromFeature token feature)
  let result := model.decoderLayer.forwardToken model.activeExperts_le state depthState
  (result.1, RMSNorm.scale (result.2.retrieve model.finalQuery) model.finalNormScale)

/-- Advance the first draft position from target-model features. Later training-time-test positions
use `stepHiddenFromFeature` with the preceding draft hidden state. -/
noncomputable def stepHidden (model : Model cfg decayRank) (state : model.CausalState)
    (token : Fin cfg.vocabSize) (features : Tensor ℝ [3, cfg.hiddenDim]) :
    model.CausalState × Tensor ℝ [cfg.hiddenDim] :=
  model.stepHiddenFromFeature state token (FeatureFusion.forward model.featureFusion features)

/-- Run the first draft position from an empty causal state. -/
noncomputable def forwardHidden (model : Model cfg decayRank)
    (token : Fin cfg.vocabSize) (features : Tensor ℝ [3, cfg.hiddenDim]) :
    Tensor ℝ [cfg.hiddenDim] :=
  (model.stepHidden model.initialState token features).2

/-- Advance one causal draft position and return its next-token logits. -/
noncomputable def stepLogits (model : Model cfg decayRank) (state : model.CausalState)
    (token : Fin cfg.vocabSize) (features : Tensor ℝ [3, cfg.hiddenDim]) :
    model.CausalState × Tensor ℝ [cfg.vocabSize] :=
  let result := model.stepHidden state token features
  (result.1, vecMatMulSpec result.2 model.vocabularyHead)

/-- Advance one continuation position using the preceding draft hidden state as the feature
input. -/
noncomputable def stepLogitsFromFeature (model : Model cfg decayRank) (state : model.CausalState)
    (token : Fin cfg.vocabSize) (feature : Tensor ℝ [cfg.hiddenDim]) :
    model.CausalState ×
      (Tensor ℝ [cfg.hiddenDim] × Tensor ℝ [cfg.vocabSize]) :=
  let result := model.stepHiddenFromFeature state token feature
  (result.1, result.2, vecMatMulSpec result.2 model.vocabularyHead)

/-- Draft next-token logits. -/
noncomputable def logits (model : Model cfg decayRank)
    (token : Fin cfg.vocabSize) (features : Tensor ℝ [3, cfg.hiddenDim]) :
    Tensor ℝ [cfg.vocabSize] :=
  (model.stepLogits model.initialState token features).2

/-- With the report's `[0 0 I]` initialization, the EAGLE-3 fused input equals the pretrained MTP
input built from the high-level feature alone. This is why the draft head can start from the MTP
layer. -/
theorem input_at_eagle_initialization (model : Model cfg decayRank)
    (token : Fin cfg.vocabSize) (features : Tensor ℝ [3, cfg.hiddenDim])
    (hLow : (Spec.get model.featureFusion 0) =
      Tensor.zeros [cfg.hiddenDim, cfg.hiddenDim])
    (hMiddle : (Spec.get model.featureFusion 1) =
      Tensor.zeros [cfg.hiddenDim, cfg.hiddenDim])
    (hHigh : (Spec.get model.featureFusion 2) = identityTensorSpec cfg.hiddenDim) :
    model.input token features =
      vecMatMulSpec (Spec.get model.tokenEmbedding token) model.tokenInputWeight +
        vecMatMulSpec (Spec.get features 2) model.featureInputWeight := by
  rw [input, inputFromFeature,
    FeatureFusion.initial_fusion_eq_high features model.featureFusion hLow hMiddle hHigh]

end Model

/-- A probability vector over the finite vocabulary, as a point of Mathlib's standard simplex. -/
abbrev Distribution (vocabSize : Nat) := Convexity.StdSimplex ℝ (Fin vocabSize)

namespace Distribution

/-- Interpret a nonempty vocabulary-logit vector as a probability distribution through TorchLean's
stable softmax specification. Positivity and normalization come from TorchLean's softmax
theorems, not from new assumptions. -/
noncomputable def ofLogits {vocabSize : Nat} (hVocab : 0 < vocabSize)
    (logits : Tensor ℝ [vocabSize]) : Distribution vocabSize := by
  cases vocabSize with
  | zero => omega
  | succ n =>
      let probabilities := Activation.softmaxVecSpec logits
      exact {
        weights := Finsupp.equivFunOnFinite.symm (Tensor.getScalar probabilities)
        nonneg := fun token => by
          simpa using le_of_lt (Proofs.softmax_vec_spec_pos logits token)
        total := by
          simpa [Finsupp.sum_fintype, probabilities, Spec.sum_spec_vec] using
            Proofs.sum_spec_softmax_vec_spec logits }

end Distribution

/-- Target-model distribution at one sequence position, computed from the logits of the complete
K3 language backbone. -/
noncomputable def targetDistributionAt {cfg : TextConfig} {decayRank sequenceLength : Nat}
    (model : LanguageModel ℝ cfg decayRank)
    (embeddings : Tensor ℝ [sequenceLength, cfg.hiddenDim])
    (position : Fin sequenceLength) : Distribution cfg.vocabSize :=
  Distribution.ofLogits model.vocabSize_pos (Spec.get (model.logits embeddings) position)

namespace Model

/-- Next-token distribution emitted by the EAGLE-3 draft head. -/
noncomputable def distribution (model : Model cfg decayRank)
    (token : Fin cfg.vocabSize) (features : Tensor ℝ [3, cfg.hiddenDim]) :
    Distribution cfg.vocabSize :=
  Distribution.ofLogits model.vocabSize_pos (model.logits token features)

/-- Advance one causal draft position and return the normalized distribution used by lossless
speculative sampling. -/
noncomputable def stepDistribution (model : Model cfg decayRank) (state : model.CausalState)
    (token : Fin cfg.vocabSize) (features : Tensor ℝ [3, cfg.hiddenDim]) :
    model.CausalState × Distribution cfg.vocabSize :=
  let result := model.stepLogits state token features
  (result.1, Distribution.ofLogits model.vocabSize_pos result.2)

/-- Advance a continuation position and expose both the hidden feature fed into the next step and
the normalized token distribution. -/
noncomputable def stepDistributionFromFeature (model : Model cfg decayRank)
    (state : model.CausalState) (token : Fin cfg.vocabSize)
    (feature : Tensor ℝ [cfg.hiddenDim]) :
    model.CausalState ×
      (Tensor ℝ [cfg.hiddenDim] × Distribution cfg.vocabSize) :=
  let result := model.stepLogitsFromFeature state token feature
  (result.1, result.2.1, Distribution.ofLogits model.vocabSize_pos result.2.2)

end Model

/-- Lossless speculative-sampling acceptance rate `Σ_x min(p(x), q(x))`. -/
noncomputable def acceptanceRate {vocabSize : Nat}
    (target draft : Distribution vocabSize) : ℝ :=
  ∑ token, min (target.weights token) (draft.weights token)

/-- Total-variation distance between target and draft next-token distributions. -/
noncomputable def totalVariation {vocabSize : Nat}
    (target draft : Distribution vocabSize) : ℝ :=
  (1 / 2 : ℝ) * ∑ token, |target.weights token - draft.weights token|

/-- Swapping target and draft does not change speculative acceptance. -/
theorem acceptanceRate_comm {vocabSize : Nat} (target draft : Distribution vocabSize) :
    acceptanceRate target draft = acceptanceRate draft target := by
  simp only [acceptanceRate, min_comm]

/-- Total variation is nonnegative. -/
theorem totalVariation_nonneg {vocabSize : Nat} (target draft : Distribution vocabSize) :
    0 ≤ totalVariation target draft :=
  mul_nonneg (by norm_num) (Finset.sum_nonneg fun _ _ => abs_nonneg _)

/-- Acceptance rates lie in `[0, 1]`. -/
theorem acceptanceRate_mem_unitInterval {vocabSize : Nat}
    (target draft : Distribution vocabSize) :
    0 ≤ acceptanceRate target draft ∧ acceptanceRate target draft ≤ 1 := by
  constructor
  · exact Finset.sum_nonneg (fun token _ =>
      le_min (target.nonneg token) (draft.nonneg token))
  · calc
      acceptanceRate target draft
          ≤ ∑ token, target.weights token :=
            Finset.sum_le_sum (fun token _ => min_le_left _ _)
      _ = 1 := target.total_of_fintype

/-- Speculative acceptance equals one minus the total-variation distance, so the Eq. 16 loss
`-log(acceptance)` depends on target and draft only through that distance. -/
theorem acceptanceRate_eq_one_sub_totalVariation {vocabSize : Nat}
    (target draft : Distribution vocabSize) :
    acceptanceRate target draft = 1 - totalVariation target draft := by
  have htarget : ∑ token, target.weights token = 1 := by
    exact target.total_of_fintype
  have hdraft : ∑ token, draft.weights token = 1 := by
    exact draft.total_of_fintype
  have hpointwise : ∀ token : Fin vocabSize,
      min (target.weights token) (draft.weights token) =
        (target.weights token + draft.weights token - |target.weights token - draft.weights token|) / 2 := by
    intro token
    by_cases h : target.weights token ≤ draft.weights token
    · rw [min_eq_left h, abs_of_nonpos (sub_nonpos.mpr h)]
      ring_nf
    · have h' : draft.weights token ≤ target.weights token := le_of_not_ge h
      rw [min_eq_right h', abs_of_nonneg (sub_nonneg.mpr h')]
      ring_nf
  rw [acceptanceRate, totalVariation]
  simp_rw [hpointwise]
  rw [← Finset.sum_div]
  rw [Finset.sum_sub_distrib, Finset.sum_add_distrib,
    htarget, hdraft]
  ring_nf

/-- Speculative acceptance is perfect exactly when the draft and target distributions agree. -/
theorem acceptanceRate_eq_one_iff {vocabSize : Nat}
    (target draft : Distribution vocabSize) :
    acceptanceRate target draft = 1 ↔ target = draft := by
  constructor
  · intro hAcceptance
    have hIdentity := acceptanceRate_eq_one_sub_totalVariation target draft
    rw [hAcceptance] at hIdentity
    have hVariation : totalVariation target draft = 0 := by linarith
    have hSum : ∑ token, |target.weights token - draft.weights token| = 0 := by
      rw [totalVariation] at hVariation
      linarith
    apply Convexity.StdSimplex.ext
    apply Finsupp.ext
    intro token
    have hCoordinate :
        |target.weights token - draft.weights token| = 0 :=
      (Finset.sum_eq_zero_iff_of_nonneg fun _ _ => abs_nonneg _).mp hSum token
        (Finset.mem_univ token)
    exact sub_eq_zero.mp (abs_eq_zero.mp hCoordinate)
  · rintro rfl
    rw [acceptanceRate]
    simp only [min_self]
    exact target.total_of_fintype

/-- Rejection probability: the target mass not shared with the draft. -/
noncomputable def residualMass {vocabSize : Nat}
    (target draft : Distribution vocabSize) : ℝ :=
  1 - acceptanceRate target draft

/-- Unnormalized correction mass for one token after speculative rejection. -/
noncomputable def residualNumerator {vocabSize : Nat}
    (target draft : Distribution vocabSize) (token : Fin vocabSize) : ℝ :=
  max 0 (target.weights token - draft.weights token)

/-- The positive part of `target - draft` is the target mass not covered by acceptance. -/
theorem residualNumerator_eq_sub_min {vocabSize : Nat}
    (target draft : Distribution vocabSize) (token : Fin vocabSize) :
    residualNumerator target draft token =
      target.weights token - min (target.weights token) (draft.weights token) := by
  simpa only [residualNumerator, sub_self] using
    (max_sub_sub_left (target.weights token) (target.weights token) (draft.weights token))

/-- Correction numerators sum to the rejection probability. -/
theorem sum_residualNumerator {vocabSize : Nat}
    (target draft : Distribution vocabSize) :
    ∑ token, residualNumerator target draft token = residualMass target draft := by
  have htarget : ∑ token, target.weights token = 1 := by
    exact target.total_of_fintype
  simp_rw [residualNumerator_eq_sub_min]
  rw [Finset.sum_sub_distrib, htarget]
  rfl

/-- Distribution sampled after a rejection, defined when target and draft differ (acceptance below
one). Its density is the normalized positive part of `target - draft`. -/
noncomputable def correctionDistribution {vocabSize : Nat}
    (target draft : Distribution vocabSize)
    (hReject : acceptanceRate target draft < 1) : Distribution vocabSize := by
  have hMass : 0 < residualMass target draft := by
    simpa [residualMass] using sub_pos.mpr hReject
  exact {
    weights := Finsupp.equivFunOnFinite.symm
      (fun token => residualNumerator target draft token / residualMass target draft)
    nonneg := fun token => by
      simpa [residualNumerator] using
        div_nonneg (le_max_left 0 (target.weights token - draft.weights token)) hMass.le
    total := by
      simpa [Finsupp.sum_fintype, ← Finset.sum_div, sum_residualNumerator] using
        div_self (ne_of_gt hMass) }

/-- One-token speculative sampling is lossless. For every token, the mass accepted from the draft
plus the mass drawn from the rejection correction equals the target-model mass. -/
theorem accepted_add_corrected_eq_target {vocabSize : Nat}
    (target draft : Distribution vocabSize)
    (hReject : acceptanceRate target draft < 1) (token : Fin vocabSize) :
    min (target.weights token) (draft.weights token) +
        residualMass target draft *
          (correctionDistribution target draft hReject).weights token =
      target.weights token := by
  have hMass : residualMass target draft ≠ 0 := by
    apply ne_of_gt
    simpa [residualMass] using sub_pos.mpr hReject
  rw [correctionDistribution]
  change min (target.weights token) (draft.weights token) +
      residualMass target draft *
        (residualNumerator target draft token / residualMass target draft) =
    target.weights token
  rw [mul_div_cancel₀ _ hMass]
  rw [residualNumerator_eq_sub_min]
  ring

namespace Model

/-- State of EAGLE-3 training-time testing after target features have been fused once. The second
component is the draft hidden feature consumed by the following step. -/
abbrev TrainingTimeTestState (model : Model cfg decayRank) :=
  model.CausalState × Tensor ℝ [cfg.hiddenDim]

/-- Initial training-time-test state: an empty draft cache and the fused target-model feature. -/
noncomputable def initialTrainingTimeTestState (model : Model cfg decayRank)
    (features : Tensor ℝ [3, cfg.hiddenDim]) : model.TrainingTimeTestState :=
  (model.initialState, FeatureFusion.forward model.featureFusion features)

/-- One EAGLE-3 training-time-test transition. It consumes the current token and
feature, then stores its own hidden output as the feature for the next transition. No fresh
target-model feature is requested after initialization. -/
noncomputable def trainingTimeTestStep (model : Model cfg decayRank) :
    model.TrainingTimeTestState → Fin cfg.vocabSize →
      model.TrainingTimeTestState × Distribution cfg.vocabSize :=
  fun state token =>
    let result := model.stepDistributionFromFeature state.1 token state.2
    ((result.1, result.2.1), result.2.2)

/-- Run EAGLE-3 training-time testing over shifted training tokens, starting from the fused
target-model features. Kimi K3 supplies seven tokens. -/
noncomputable def trainingTimeTest (model : Model cfg decayRank)
    (tokens : List (Fin cfg.vocabSize)) (features : Tensor ℝ [3, cfg.hiddenDim]) :
    model.TrainingTimeTestState × List (Distribution cfg.vocabSize) :=
  CausalScan.run model.trainingTimeTestStep
    (model.initialTrainingTimeTestState features) tokens

/-- Training-time testing emits one draft distribution for every shifted training token. -/
theorem trainingTimeTest_length (model : Model cfg decayRank)
    (tokens : List (Fin cfg.vocabSize)) (features : Tensor ℝ [3, cfg.hiddenDim]) :
    (model.trainingTimeTest tokens features).2.length = tokens.length :=
  CausalScan.outputs_length model.trainingTimeTestStep
    (model.initialTrainingTimeTestState features) tokens

/-- Splitting a training-time-test token sequence into chunks does not change its final state or
draft distributions, so the seven reported steps can be replayed incrementally. -/
theorem trainingTimeTest_append (model : Model cfg decayRank)
    (initialTokens remainingTokens : List (Fin cfg.vocabSize))
    (features : Tensor ℝ [3, cfg.hiddenDim]) :
    model.trainingTimeTest (initialTokens ++ remainingTokens) features =
      let first := model.trainingTimeTest initialTokens features
      let second := CausalScan.run model.trainingTimeTestStep first.1 remainingTokens
      (second.1, first.2 ++ second.2) := by
  exact CausalScan.run_append model.trainingTimeTestStep
    (model.initialTrainingTimeTestState features) initialTokens remainingTokens

/-- The seven draft distributions used by K3 fine-tuning. Indexing by `Fin 7` puts the reported
training-time-test depth into the type. -/
noncomputable def sevenStepDistributions (model : Model cfg decayRank)
    (tokens : Fin 7 → Fin cfg.vocabSize) (features : Tensor ℝ [3, cfg.hiddenDim]) :
    Fin 7 → Distribution cfg.vocabSize := by
  let outputs := (model.trainingTimeTest (List.ofFn tokens) features).2
  have hLength : outputs.length = 7 := by
    rw [model.trainingTimeTest_length]
    simp
  exact fun step => outputs.get ⟨step.val, by rw [hLength]; exact step.isLt⟩

end Model

/-- Likelihood-based draft loss in Eq. 16.

The codomain is `EReal` because disjoint target and draft distributions have zero acceptance and
so infinite negative-log loss.  Lean's convention `Real.log 0 = 0` would hide that case.
-/
noncomputable def likelihoodLoss {vocabSize : Nat}
    (target draft : Distribution vocabSize) : EReal :=
  if acceptanceRate target draft = 0 then ⊤
  else ((-Real.log (acceptanceRate target draft) : ℝ) : EReal)

/-- Likelihood loss is infinite exactly when the acceptance rate is zero. -/
theorem likelihoodLoss_eq_top_iff {vocabSize : Nat}
    (target draft : Distribution vocabSize) :
    likelihoodLoss target draft = ⊤ ↔ acceptanceRate target draft = 0 := by
  simp [likelihoodLoss]

/-- At positive acceptance, the extended loss agrees with the ordinary real negative log. -/
theorem likelihoodLoss_eq_neg_log_of_pos {vocabSize : Nat}
    (target draft : Distribution vocabSize)
    (hAcceptance : 0 < acceptanceRate target draft) :
    likelihoodLoss target draft =
      ((-Real.log (acceptanceRate target draft) : ℝ) : EReal) := by
  simp [likelihoodLoss, ne_of_gt hAcceptance]

namespace Model

/-- Kimi K3's seven-step EAGLE-3 objective against a frozen target language model. Each target
distribution comes from the target model at the corresponding training position, and each draft
distribution comes from the recurrent training-time-test pass. As in Eq. 16 of the report, there
is no auxiliary ground-truth cross-entropy term. -/
noncomputable def sevenStepLikelihoodLoss {targetDecayRank draftDecayRank sequenceLength : Nat}
    (target : LanguageModel ℝ cfg targetDecayRank) (draft : Model cfg draftDecayRank)
    (embeddings : Tensor ℝ [sequenceLength, cfg.hiddenDim])
    (positions : Fin 7 → Fin sequenceLength) (tokens : Fin 7 → Fin cfg.vocabSize)
    (features : Tensor ℝ [3, cfg.hiddenDim]) : EReal :=
  ∑ step : Fin 7,
    likelihoodLoss (targetDistributionAt target embeddings (positions step))
      (draft.sevenStepDistributions tokens features step)

end Model

end Draft

end KimiK3
