/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Model
public import KimiK3.Microscaling
public import NN.MLTheory.Optimization.Muon
public import NN.Proofs.Analysis.Softmax
public import NN.Spec.Layers.Loss
public import NN.Proofs.Tensor.Basic.Algebra
public import NN.Tensor
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Data.EReal.Basic

/-!
# Kimi K3 training and deployment specifications

This module records the mathematical parts of the K3 training recipe that can be stated from the
published equations without reproducing Moonshot's private data or distributed system:

* Per-Head Muon as independent orthogonalization contracts for attention-head blocks;
* masked next-token pretraining and its derivative with respect to the model logits;
* the clipped MOPD reward from Eq. 15;
* OCP microscaling formats used by routed-expert weights and activations;
* EAGLE-3 feature fusion, training-time testing, and lossless speculative acceptance.

This does not turn empirical claims in the report into theorems.  Dataset quality, scaling
efficiency, benchmark scores, and distributed-system throughput remain measured properties.

Reference: Kimi Team, "Kimi K3: Open Frontier Intelligence", 2026, Sections 2.5, 3.3--3.4,
and 4.1, https://arxiv.org/abs/2607.24653.
-/

@[expose] public section

namespace KimiK3

open Spec
open Tensor
open scoped BigOperators

/-! ## Per-Head Muon -/

namespace PerHeadMuon

open Optim.Muon

/-- One momentum matrix for every attention head. -/
abbrev HeadMatrices (α : Type) (heads rows columns : Nat) :=
  Fin heads → MatrixTensor α rows columns

/-- Partition a projection matrix along its output-head axis.

For a matrix of shape `rows × (heads * columns)`, output column `(head, column)` belongs to the
`head`-th `rows × columns` block. This is the layout used when Per-Head Muon treats Q, K, and V
momentum buffers independently by attention head.
-/
def split {α : Type} {heads rows columns : Nat}
    (momentum : MatrixTensor α rows (heads * columns)) :
    HeadMatrices α heads rows columns :=
  fun head => Tensor.dim fun row => Tensor.dim fun column =>
    Tensor.scalar (get2 momentum row (finProdFinEquiv (head, column)))

/-- Reassemble per-head matrices into the original projection layout. -/
def merge {α : Type} {heads rows columns : Nat}
    (blocks : HeadMatrices α heads rows columns) :
    MatrixTensor α rows (heads * columns) :=
  Tensor.dim fun row => Tensor.dim fun outputColumn =>
    let headColumn := finProdFinEquiv.symm outputColumn
    Tensor.scalar (get2 (blocks headColumn.1) row headColumn.2)

/-- Splitting and reassembling a projection momentum matrix preserves every entry. -/
theorem merge_split {α : Type} {heads rows columns : Nat}
    (momentum : MatrixTensor α rows (heads * columns)) :
    merge (split momentum) = momentum := by
  cases momentum with
  | dim matrixRows =>
      apply congrArg Tensor.dim
      funext row
      cases hRow : matrixRows row with
      | dim matrixColumns =>
          apply congrArg Tensor.dim
          funext outputColumn
          cases hValue : matrixColumns outputColumn with
          | scalar value =>
              have hIndex :
                  finProdFinEquiv (finProdFinEquiv.symm outputColumn) = outputColumn :=
                Equiv.apply_symm_apply finProdFinEquiv outputColumn
              simp only [split]
              change Tensor.scalar
                (get2 (Tensor.dim matrixRows) row
                  (finProdFinEquiv (finProdFinEquiv.symm outputColumn))) = Tensor.scalar value
              rw [hIndex]
              simp [get2, Spec.get, Spec.get, hRow, hValue]

/-- Reassembling and splitting an indexed head family returns the same family. -/
theorem split_merge {α : Type} {heads rows columns : Nat}
    (blocks : HeadMatrices α heads rows columns) :
    split (merge blocks) = blocks := by
  funext head
  cases hBlock : blocks head with
  | dim blockRows =>
      apply congrArg Tensor.dim
      funext row
      cases hRow : blockRows row with
      | dim blockColumns =>
          apply congrArg Tensor.dim
          funext column
          cases hValue : blockColumns column with
          | scalar value =>
              have hIndex :
                  finProdFinEquiv.symm (finProdFinEquiv (head, column)) = (head, column) :=
                Equiv.symm_apply_apply finProdFinEquiv (head, column)
              simp only [merge]
              change Tensor.scalar
                (get2 (blocks (finProdFinEquiv.symm (finProdFinEquiv (head, column))).1)
                  row (finProdFinEquiv.symm (finProdFinEquiv (head, column))).2) =
                  Tensor.scalar value
              rw [hIndex]
              simp [get2, Spec.get, Spec.get, hBlock, hRow, hValue]

/-- Orthogonalize every output-head block independently and restore the projection layout. -/
def orthogonalize {α : Type} [Context α] {heads rows columns : Nat}
    (orthogonalizer : Orthogonalizer α (.dim rows (.dim columns .scalar)))
    (momentum : MatrixTensor α rows (heads * columns)) :
    MatrixTensor α rows (heads * columns) :=
  merge fun head => orthogonalizer.apply (split momentum head)

/-- Splitting a Per-Head Muon direction recovers the independently orthogonalized head blocks. -/
theorem split_orthogonalize {α : Type} [Context α] {heads rows columns : Nat}
    (orthogonalizer : Orthogonalizer α (.dim rows (.dim columns .scalar)))
    (momentum : MatrixTensor α rows (heads * columns)) :
    split (orthogonalize orthogonalizer momentum) =
      fun head => orthogonalizer.apply (split momentum head) := by
  exact split_merge _

/-- Exact Per-Head Muon applies the matrix orthogonality contract separately to every head. -/
def HasExactDirections {α : Type} [Context α] {heads rows columns : Nat}
    (orthogonalizer : Orthogonalizer α (.dim rows (.dim columns .scalar)))
    (momentum : HeadMatrices α heads rows columns) : Prop :=
  ∀ head, HasExactColumnGram (orthogonalizer.apply (momentum head))

/-- Approximate Per-Head Muon carries one Gram-residual bound per head. -/
def HasApproxDirections {α : Type} [Context α] {heads rows columns : Nat} (epsilon : α)
    (orthogonalizer : Orthogonalizer α (.dim rows (.dim columns .scalar)))
    (momentum : HeadMatrices α heads rows columns) : Prop :=
  ∀ head, HasApproxColumnGram epsilon (orthogonalizer.apply (momentum head))

/-- An exact Muon backend certifies every head block of the reassembled projection update. -/
theorem orthogonalize_hasExactColumnGram {α : Type} [Context α]
    {heads rows columns : Nat}
    (orthogonalizer : Orthogonalizer α (.dim rows (.dim columns .scalar)))
    (hExact : ExactMatrixOrthogonalizer orthogonalizer)
    (momentum : MatrixTensor α rows (heads * columns)) :
    ∀ head, HasExactColumnGram (split (orthogonalize orthogonalizer momentum) head) := by
  intro head
  rw [split_orthogonalize]
  exact hExact (split momentum head)

/-- An approximate Muon backend transfers its Gram-residual bound to every reassembled head. -/
theorem orthogonalize_hasApproxColumnGram {α : Type} [Context α]
    {heads rows columns : Nat} (epsilon : α)
    (orthogonalizer : Orthogonalizer α (.dim rows (.dim columns .scalar)))
    (hApprox : ApproxMatrixOrthogonalizer epsilon orthogonalizer)
    (momentum : MatrixTensor α rows (heads * columns)) :
    ∀ head, HasApproxColumnGram epsilon
      (split (orthogonalize orthogonalizer momentum) head) := by
  intro head
  rw [split_orthogonalize]
  exact hApprox (split momentum head)

end PerHeadMuon

/-! ## Joint text-and-vision next-token objective -/

namespace Pretraining

/-- Negative log-likelihood of one target token, evaluated from unnormalized vocabulary logits. -/
noncomputable def tokenLoss {vocabSize : Nat}
    (logits : Tensor ℝ (.dim vocabSize .scalar)) (target : Fin vocabSize) : ℝ :=
  -Tensor.getScalar (Activation.logSoftmaxVecSpec logits) target

/-- Mean next-token loss over the positions selected by `supervised`.

The model may receive both text and visual embeddings, but its targets are vocabulary tokens. The
finite set makes masking explicit: visual input positions, padding, and input-only protocol tokens
can be excluded without assigning them fabricated output labels.
-/
noncomputable def nextTokenLoss {sequenceLength vocabSize : Nat}
    (logits : Tensor ℝ (.dim sequenceLength (.dim vocabSize .scalar)))
    (targets : Fin sequenceLength → Fin vocabSize)
    (supervised : Finset (Fin sequenceLength)) (_hSupervised : supervised.Nonempty) : ℝ :=
  (∑ position ∈ supervised, tokenLoss (Spec.get logits position) (targets position)) /
    supervised.card

/-- Gradient of the mean next-token objective with respect to its logits.

Rows omitted from `supervised` have zero gradient. A supervised row uses the standard
`softmax(logits) - oneHot(target)` derivative, scaled by the number of supervised positions.
-/
noncomputable def logitGradient {sequenceLength vocabSize : Nat}
    (logits : Tensor ℝ (.dim sequenceLength (.dim vocabSize .scalar)))
    (targets : Fin sequenceLength → Fin vocabSize)
    (supervised : Finset (Fin sequenceLength)) (_hSupervised : supervised.Nonempty) :
    Tensor ℝ (.dim sequenceLength (.dim vocabSize .scalar)) :=
  Tensor.dim fun position =>
    if position ∈ supervised then
      scaleSpec
        (crossEntropyLogitsDerivSpec 0 (Spec.get logits position)
          (TorchLean.Tensor.oneHot vocabSize (targets position)))
        (1 / supervised.card)
    else
      Spec.fill 0 (.dim vocabSize .scalar)

/-- On every supervised position, the declared gradient is exactly TorchLean's stable
cross-entropy-on-logits derivative, with the outer mean reduction applied. -/
theorem logitGradient_at_supervised {sequenceLength vocabSize : Nat}
    (logits : Tensor ℝ (.dim sequenceLength (.dim vocabSize .scalar)))
    (targets : Fin sequenceLength → Fin vocabSize)
    (supervised : Finset (Fin sequenceLength)) (hSupervised : supervised.Nonempty)
    (position : Fin sequenceLength) (hPosition : position ∈ supervised) :
    Spec.get (logitGradient logits targets supervised hSupervised) position =
      scaleSpec
        (crossEntropyLogitsDerivSpec 0 (Spec.get logits position)
          (TorchLean.Tensor.oneHot vocabSize (targets position)))
        (1 / supervised.card) := by
  simp only [logitGradient, Spec.get_dim]
  rw [if_pos hPosition]

/-- A masked position contributes no gradient to the shared backbone or vocabulary head. -/
theorem logitGradient_at_masked {sequenceLength vocabSize : Nat}
    (logits : Tensor ℝ (.dim sequenceLength (.dim vocabSize .scalar)))
    (targets : Fin sequenceLength → Fin vocabSize)
    (supervised : Finset (Fin sequenceLength)) (hSupervised : supervised.Nonempty)
    (position : Fin sequenceLength) (hPosition : position ∉ supervised) :
    Spec.get (logitGradient logits targets supervised hSupervised) position =
      Spec.fill 0 (.dim vocabSize .scalar) := by
  simp only [logitGradient, Spec.get_dim]
  rw [if_neg hPosition]

end Pretraining

namespace LanguageModel

/-- Evaluate the masked next-token objective on the logits produced by the complete language
backbone. This is the pretraining objective of a concrete model, rather than a loss function left
detached from the architecture that produced its logits. -/
noncomputable def pretrainingLoss {cfg : TextConfig} {decayRank sequenceLength : Nat}
    (model : LanguageModel ℝ cfg decayRank)
    (embeddings : Tensor ℝ (.dim sequenceLength (.dim cfg.hiddenDim .scalar)))
    (targets : Fin sequenceLength → Fin cfg.vocabSize)
    (supervised : Finset (Fin sequenceLength)) (hSupervised : supervised.Nonempty) : ℝ :=
  Pretraining.nextTokenLoss (model.logits embeddings) targets supervised hSupervised

end LanguageModel

namespace MOPD

/-- A probability that may safely appear inside the logarithmic MOPD reward.

The teacher and student probabilities in Eq. 15 come from softmax distributions and are therefore
strictly positive.  Keeping that fact in the type prevents Lean's totalized real logarithm from
silently assigning a value to an impossible zero- or negative-probability input.
-/
structure StrictProbability where
  value : ℝ
  positive : 0 < value
  atMostOne : value ≤ 1

/-- Symmetric clipping to `[-bound, bound]`. -/
noncomputable def clip (bound value : ℝ) : ℝ := max (-bound) (min bound value)

/-- Per-token teacher/student reward from Eq. 15, before the stop-gradient annotation. -/
noncomputable def reward (maxReward : ℝ)
    (teacherProbability studentProbability : StrictProbability) : ℝ :=
  clip maxReward (Real.log (teacherProbability.value / studentProbability.value))

/-- Clipping enforces the report's reward interval for every log-ratio. -/
theorem reward_mem_interval {maxReward : ℝ} (h : 0 ≤ maxReward)
    (teacherProbability studentProbability : StrictProbability) :
    -maxReward ≤ reward maxReward teacherProbability studentProbability ∧
      reward maxReward teacherProbability studentProbability ≤ maxReward := by
  constructor
  · exact le_max_left _ _
  · exact max_le (by linarith) (min_le_left _ _)

end MOPD


/-! ## EAGLE-3 draft fine-tuning and lossless acceptance -/

namespace Draft

/-- Low-, middle-, and high-depth target-model features used by the draft layer. -/
structure FeatureTriplet (α : Type) (hiddenDim : Nat) where
  low : Tensor α (.dim hiddenDim .scalar)
  middle : Tensor α (.dim hiddenDim .scalar)
  high : Tensor α (.dim hiddenDim .scalar)

namespace FeatureTriplet

/-- Read the first, fourth, and final AttnRes block outputs named by the K3 report. The hypothesis
rules out applying the EAGLE-3 head before four block outputs exist. -/
def ofBlockState {hiddenDim : Nat} (state : AttnRes.BlockState ℝ hiddenDim)
    (hFour : 4 ≤ state.blockOutputs.length) : FeatureTriplet ℝ hiddenDim :=
  let outputs := state.blockOutputs
  have hFourOutputs : 4 ≤ outputs.length := by simpa [outputs] using hFour
  have hNonempty : outputs ≠ [] := List.ne_nil_of_length_pos (by omega)
  { low := outputs.get ⟨0, by omega⟩
    middle := outputs.get ⟨3, by omega⟩
    high := outputs.getLast hNonempty }

end FeatureTriplet

namespace LanguageModel

/-- Extract the EAGLE-3 feature triplet for one token after a target-model forward pass. The result
is tied directly to that token's retained AttnRes state, rather than supplied as unrelated data. -/
noncomputable def eagleFeaturesAt {cfg : TextConfig} {decayRank sequenceLength : Nat}
    (model : LanguageModel ℝ cfg decayRank)
    (embeddings : Tensor ℝ (.dim sequenceLength (.dim cfg.hiddenDim .scalar)))
    (position : Fin sequenceLength)
    (hFour : 4 ≤ (model.forwardDepthStates embeddings position).blockOutputs.length) :
    FeatureTriplet ℝ cfg.hiddenDim :=
  FeatureTriplet.ofBlockState (model.forwardDepthStates embeddings position) hFour

end LanguageModel

/-- Bias-free feature-fusion projection `W_E3`, represented as one matrix per source depth. -/
structure FeatureFusion (α : Type) (hiddenDim : Nat) where
  weight : Fin 3 → Tensor α (.dim hiddenDim (.dim hiddenDim .scalar))

namespace FeatureFusion

variable {hiddenDim : Nat}

/-- Every entry of a constant square matrix is its fill value. -/
@[simp]
private theorem get2_fill (value : ℝ) (row column : Fin hiddenDim) :
    get2 (Spec.fill value (.dim hiddenDim (.dim hiddenDim .scalar))) row column = value := by
  rfl

/-- Multiplying a real vector by an all-zero square matrix gives the zero vector. -/
@[simp]
private theorem vecMatMul_fill_zero (values : Tensor ℝ (.dim hiddenDim .scalar)) :
    vecMatMulSpec values (Spec.fill 0 (.dim hiddenDim (.dim hiddenDim .scalar))) =
      Spec.fill 0 (.dim hiddenDim .scalar) := by
  let result := vecMatMulSpec values
    (Spec.fill 0 (.dim hiddenDim (.dim hiddenDim .scalar)))
  let zero : Tensor ℝ (.dim hiddenDim .scalar) := Spec.fill 0 (.dim hiddenDim .scalar)
  have coordinates : Tensor.getScalar result = Tensor.getScalar zero := by
    funext column
    dsimp [result]
    rw [getScalar_vec_mat_mul_spec]
    calc
      (∑ row : Fin hiddenDim, Tensor.getScalar values row *
          get2 (Spec.fill 0 (.dim hiddenDim (.dim hiddenDim .scalar))) row column) = 0 := by
            simp only [get2_fill, mul_zero, Finset.sum_const_zero]
      _ = Tensor.getScalar zero column := by rfl
  exact Tensor.ext_vector (fun column => congrFun coordinates column)

/-- Entries of the spec identity matrix are the Kronecker delta. -/
@[simp]
private theorem get2_identity (row column : Fin hiddenDim) :
    get2 (identityTensorSpec hiddenDim :
      Tensor ℝ (.dim hiddenDim (.dim hiddenDim .scalar))) row column =
      if row = column then 1 else 0 := by
  cases hiddenDim with
  | zero => exact Fin.elim0 row
  | succ n =>
      simp only [identityTensorSpec, get2, _root_.Spec.get, _root_.Spec.get]
      simp [Fin.ext_iff]

/-- Multiplying a real row vector by the identity matrix returns that vector. -/
@[simp]
private theorem vecMatMul_identity (values : Tensor ℝ (.dim hiddenDim .scalar)) :
    vecMatMulSpec values (identityTensorSpec hiddenDim) = values := by
  let result := vecMatMulSpec values (identityTensorSpec hiddenDim)
  have coordinates : Tensor.getScalar result = Tensor.getScalar values := by
    funext column
    simp [result, getScalar_vec_mat_mul_spec]
  exact Tensor.ext_vector (fun column => congrFun coordinates column)

variable {α : Type} [Context α]

/-- Concatenate-and-project written as a sum of three matrix products. -/
def forward (fusion : FeatureFusion α hiddenDim) (features : FeatureTriplet α hiddenDim) :
    Tensor α (.dim hiddenDim .scalar) :=
  addSpec
    (addSpec
      (vecMatMulSpec features.low (fusion.weight 0))
      (vecMatMulSpec features.middle (fusion.weight 1)))
    (vecMatMulSpec features.high (fusion.weight 2))

/-- The report's `[0 0 I]` initialization returns the high-level feature exactly. -/
theorem initial_fusion_eq_high (features : FeatureTriplet ℝ hiddenDim)
    (fusion : FeatureFusion ℝ hiddenDim)
    (hLow : fusion.weight 0 = Spec.fill 0 (.dim hiddenDim (.dim hiddenDim .scalar)))
    (hMiddle : fusion.weight 1 = Spec.fill 0 (.dim hiddenDim (.dim hiddenDim .scalar)))
    (hHigh : fusion.weight 2 = identityTensorSpec hiddenDim) :
    fusion.forward features = features.high := by
  rw [forward, hLow, hMiddle, hHigh]
  rw [vecMatMul_fill_zero, vecMatMul_fill_zero, vecMatMul_identity]
  simp

end FeatureFusion

/-- The one-layer MTP/EAGLE-3 draft head. The two input matrices represent the bias-free linear map
applied to the concatenation of the current token embedding and fused target feature. The decoder
layer has the same KDA-or-MLA and channel-mixer structure as a backbone layer. -/
structure Model (cfg : TextConfig) (decayRank : Nat) where
  featureFusion : FeatureFusion ℝ cfg.hiddenDim
  tokenEmbedding : Tensor ℝ (.dim cfg.vocabSize (.dim cfg.hiddenDim .scalar))
  tokenInputWeight : Tensor ℝ (.dim cfg.hiddenDim (.dim cfg.hiddenDim .scalar))
  featureInputWeight : Tensor ℝ (.dim cfg.hiddenDim (.dim cfg.hiddenDim .scalar))
  decoderLayer : BackboneLayer ℝ cfg decayRank
  finalQuery : Tensor ℝ (.dim cfg.hiddenDim .scalar)
  finalNormScale : Tensor ℝ (.dim cfg.hiddenDim .scalar)
  vocabularyHead : Tensor ℝ (.dim cfg.hiddenDim (.dim cfg.vocabSize .scalar))
  vocabSize_pos : 0 < cfg.vocabSize
  activeExperts_le : cfg.activeExperts ≤ cfg.numRoutedExperts

namespace Model

variable {cfg : TextConfig} {decayRank : Nat}

/-- Form a draft decoder input from a token embedding and one already fused feature. The feature is
the target model's three-level fusion at the first training-time-test step and the preceding draft
hidden state at every later step. -/
noncomputable def inputFromFeature (model : Model cfg decayRank) (token : Fin cfg.vocabSize)
    (feature : Tensor ℝ (.dim cfg.hiddenDim .scalar)) : Tensor ℝ (.dim cfg.hiddenDim .scalar) :=
  vecMatMulSpec (Spec.get model.tokenEmbedding token) model.tokenInputWeight +
    vecMatMulSpec feature model.featureInputWeight

/-- Form the first draft input by fusing the low-, middle-, and high-level target features. -/
noncomputable def input (model : Model cfg decayRank) (token : Fin cfg.vocabSize)
    (features : FeatureTriplet ℝ cfg.hiddenDim) : Tensor ℝ (.dim cfg.hiddenDim .scalar) :=
  model.inputFromFeature token (model.featureFusion.forward features)

/-- Causal KDA or MLA state retained by the single draft decoder layer. -/
abbrev CausalState (model : Model cfg decayRank) :=
  model.decoderLayer.sequence.CausalState

/-- Empty causal state at the beginning of a draft window. -/
noncomputable def initialState (model : Model cfg decayRank) : model.CausalState :=
  model.decoderLayer.sequence.initialState

/-- Advance the draft decoder from one fused feature and retain the sequence-mixer state needed by
the next draft position. -/
noncomputable def stepHiddenFromFeature (model : Model cfg decayRank) (state : model.CausalState)
    (token : Fin cfg.vocabSize) (feature : Tensor ℝ (.dim cfg.hiddenDim .scalar)) :
    model.CausalState × Tensor ℝ (.dim cfg.hiddenDim .scalar) :=
  let depthState := BackboneLayer.initialDepthState (model.inputFromFeature token feature)
  let result := model.decoderLayer.forwardToken model.activeExperts_le state depthState
  (result.1, RMSNorm.scale (result.2.retrieve model.finalQuery) model.finalNormScale)

/-- Advance the first draft position from target-model features. Later training-time-test positions
use `stepHiddenFromFeature` with the preceding draft hidden state. -/
noncomputable def stepHidden (model : Model cfg decayRank) (state : model.CausalState)
    (token : Fin cfg.vocabSize) (features : FeatureTriplet ℝ cfg.hiddenDim) :
    model.CausalState × Tensor ℝ (.dim cfg.hiddenDim .scalar) :=
  model.stepHiddenFromFeature state token (model.featureFusion.forward features)

/-- Run the first draft position from an empty causal state. -/
noncomputable def forwardHidden (model : Model cfg decayRank)
    (token : Fin cfg.vocabSize) (features : FeatureTriplet ℝ cfg.hiddenDim) :
    Tensor ℝ (.dim cfg.hiddenDim .scalar) :=
  (model.stepHidden model.initialState token features).2

/-- Advance one causal draft position and return its next-token logits. -/
noncomputable def stepLogits (model : Model cfg decayRank) (state : model.CausalState)
    (token : Fin cfg.vocabSize) (features : FeatureTriplet ℝ cfg.hiddenDim) :
    model.CausalState × Tensor ℝ (.dim cfg.vocabSize .scalar) :=
  let result := model.stepHidden state token features
  (result.1, vecMatMulSpec result.2 model.vocabularyHead)

/-- Advance one continuation position using the preceding draft hidden state as the feature input. -/
noncomputable def stepLogitsFromFeature (model : Model cfg decayRank) (state : model.CausalState)
    (token : Fin cfg.vocabSize) (feature : Tensor ℝ (.dim cfg.hiddenDim .scalar)) :
    model.CausalState ×
      (Tensor ℝ (.dim cfg.hiddenDim .scalar) × Tensor ℝ (.dim cfg.vocabSize .scalar)) :=
  let result := model.stepHiddenFromFeature state token feature
  (result.1, result.2, vecMatMulSpec result.2 model.vocabularyHead)

/-- Draft next-token logits. -/
noncomputable def logits (model : Model cfg decayRank)
    (token : Fin cfg.vocabSize) (features : FeatureTriplet ℝ cfg.hiddenDim) :
    Tensor ℝ (.dim cfg.vocabSize .scalar) :=
  (model.stepLogits model.initialState token features).2

/-- With the report's `[0 0 I]` initialization, replacing the pretrained MTP input by EAGLE-3
feature fusion leaves that feature input exactly unchanged. This is the formal compatibility fact
that justifies initializing the draft head from the MTP layer. -/
theorem input_at_eagle_initialization (model : Model cfg decayRank)
    (token : Fin cfg.vocabSize) (features : FeatureTriplet ℝ cfg.hiddenDim)
    (hLow : model.featureFusion.weight 0 =
      Spec.fill 0 (.dim cfg.hiddenDim (.dim cfg.hiddenDim .scalar)))
    (hMiddle : model.featureFusion.weight 1 =
      Spec.fill 0 (.dim cfg.hiddenDim (.dim cfg.hiddenDim .scalar)))
    (hHigh : model.featureFusion.weight 2 = identityTensorSpec cfg.hiddenDim) :
    model.input token features =
      vecMatMulSpec (Spec.get model.tokenEmbedding token) model.tokenInputWeight +
        vecMatMulSpec features.high model.featureInputWeight := by
  rw [input, inputFromFeature,
    FeatureFusion.initial_fusion_eq_high features model.featureFusion hLow hMiddle hHigh]

end Model

/-- A probability mass function over a finite vocabulary. -/
structure Distribution (vocabSize : Nat) where
  probability : Fin vocabSize → ℝ
  nonnegative : ∀ token, 0 ≤ probability token
  sumsToOne : ∑ token, probability token = 1

namespace Distribution

/-- Interpret a nonempty vocabulary-logit vector as a probability distribution using TorchLean's
stable softmax specification. Positivity and normalization are inherited from the corresponding
analysis theorems rather than postulated by the K3 development. -/
noncomputable def ofLogits {vocabSize : Nat} (hVocab : 0 < vocabSize)
    (logits : Tensor ℝ (.dim vocabSize .scalar)) : Distribution vocabSize := by
  cases vocabSize with
  | zero => omega
  | succ n =>
      let probabilities := Activation.softmaxVecSpec logits
      exact
        { probability := Spec.Tensor.getScalar probabilities
          nonnegative := fun token =>
            le_of_lt (Proofs.softmax_vec_spec_pos logits token)
          sumsToOne := by
            simpa [probabilities, Spec.sum_spec_vec] using
              Proofs.sum_spec_softmax_vec_spec logits }

end Distribution

/-- Target-model distribution at one sequence position. The positivity and normalization facts come
from TorchLean's softmax theorems, while the logits come from the complete K3 language backbone. -/
noncomputable def targetDistributionAt {cfg : TextConfig} {decayRank sequenceLength : Nat}
    (model : LanguageModel ℝ cfg decayRank)
    (embeddings : Tensor ℝ (.dim sequenceLength (.dim cfg.hiddenDim .scalar)))
    (position : Fin sequenceLength) : Distribution cfg.vocabSize :=
  Distribution.ofLogits model.vocabSize_pos (Spec.get (model.logits embeddings) position)

namespace Model

/-- The next-token distribution emitted by the EAGLE-3 draft head. -/
noncomputable def distribution (model : Model cfg decayRank)
    (token : Fin cfg.vocabSize) (features : FeatureTriplet ℝ cfg.hiddenDim) :
    Distribution cfg.vocabSize :=
  Distribution.ofLogits model.vocabSize_pos (model.logits token features)

/-- Advance one causal draft position and return the normalized distribution used by lossless
speculative sampling. -/
noncomputable def stepDistribution (model : Model cfg decayRank) (state : model.CausalState)
    (token : Fin cfg.vocabSize) (features : FeatureTriplet ℝ cfg.hiddenDim) :
    model.CausalState × Distribution cfg.vocabSize :=
  let result := model.stepLogits state token features
  (result.1, Distribution.ofLogits model.vocabSize_pos result.2)

/-- Advance a continuation position and expose both the hidden feature fed into the next step and
the normalized token distribution. -/
noncomputable def stepDistributionFromFeature (model : Model cfg decayRank)
    (state : model.CausalState) (token : Fin cfg.vocabSize)
    (feature : Tensor ℝ (.dim cfg.hiddenDim .scalar)) :
    model.CausalState ×
      (Tensor ℝ (.dim cfg.hiddenDim .scalar) × Distribution cfg.vocabSize) :=
  let result := model.stepLogitsFromFeature state token feature
  (result.1, result.2.1, Distribution.ofLogits model.vocabSize_pos result.2.2)

end Model

/-- Lossless speculative-sampling acceptance rate `Σ_x min(p(x), q(x))`. -/
noncomputable def acceptanceRate {vocabSize : Nat}
    (target draft : Distribution vocabSize) : ℝ :=
  ∑ token, min (target.probability token) (draft.probability token)

/-- Total-variation distance between target and draft next-token distributions. -/
noncomputable def totalVariation {vocabSize : Nat}
    (target draft : Distribution vocabSize) : ℝ :=
  (1 / 2 : ℝ) * ∑ token, |target.probability token - draft.probability token|

/-- The acceptance rate is always a probability. -/
theorem acceptanceRate_mem_unitInterval {vocabSize : Nat}
    (target draft : Distribution vocabSize) :
    0 ≤ acceptanceRate target draft ∧ acceptanceRate target draft ≤ 1 := by
  constructor
  · exact Finset.sum_nonneg (fun token _ =>
      le_min (target.nonnegative token) (draft.nonnegative token))
  · calc
      acceptanceRate target draft
          ≤ ∑ token, target.probability token :=
            Finset.sum_le_sum (fun token _ => min_le_left _ _)
      _ = 1 := target.sumsToOne

/-- Lossless speculative acceptance is exactly one minus total-variation distance. This identifies
the paper's likelihood objective with a standard statistical distance, rather than merely bounding
it between zero and one. -/
theorem acceptanceRate_eq_one_sub_totalVariation {vocabSize : Nat}
    (target draft : Distribution vocabSize) :
    acceptanceRate target draft = 1 - totalVariation target draft := by
  have hpointwise : ∀ token : Fin vocabSize,
      min (target.probability token) (draft.probability token) =
        (target.probability token + draft.probability token -
          |target.probability token - draft.probability token|) / 2 := by
    intro token
    by_cases h : target.probability token ≤ draft.probability token
    · rw [min_eq_left h, abs_of_nonpos (sub_nonpos.mpr h)]
      ring_nf
    · have h' : draft.probability token ≤ target.probability token := le_of_not_ge h
      rw [min_eq_right h', abs_of_nonneg (sub_nonneg.mpr h')]
      ring_nf
  rw [acceptanceRate, totalVariation]
  simp_rw [hpointwise]
  rw [← Finset.sum_div]
  rw [Finset.sum_sub_distrib, Finset.sum_add_distrib,
    target.sumsToOne, draft.sumsToOne]
  ring_nf

/-- Probability mass left after rejecting the part shared by target and draft distributions. -/
noncomputable def residualMass {vocabSize : Nat}
    (target draft : Distribution vocabSize) : ℝ :=
  1 - acceptanceRate target draft

/-- Unnormalized correction mass for one token after speculative rejection. -/
noncomputable def residualNumerator {vocabSize : Nat}
    (target draft : Distribution vocabSize) (token : Fin vocabSize) : ℝ :=
  max 0 (target.probability token - draft.probability token)

/-- The positive part of `target - draft` is exactly the target mass not covered by acceptance. -/
theorem residualNumerator_eq_sub_min {vocabSize : Nat}
    (target draft : Distribution vocabSize) (token : Fin vocabSize) :
    residualNumerator target draft token =
      target.probability token - min (target.probability token) (draft.probability token) := by
  unfold residualNumerator
  by_cases h : target.probability token ≤ draft.probability token
  · rw [min_eq_left h, max_eq_left (sub_nonpos.mpr h), sub_self]
  · have h' : draft.probability token ≤ target.probability token := le_of_not_ge h
    rw [min_eq_right h', max_eq_right]
    exact sub_nonneg.mpr h'

/-- The correction numerators sum to the rejection probability. -/
theorem sum_residualNumerator {vocabSize : Nat}
    (target draft : Distribution vocabSize) :
    ∑ token, residualNumerator target draft token = residualMass target draft := by
  simp_rw [residualNumerator_eq_sub_min]
  rw [Finset.sum_sub_distrib, target.sumsToOne]
  rfl

/-- Distribution sampled after rejection when target and draft are not identical in total
variation. Its density is the normalized positive part of `target - draft`. -/
noncomputable def correctionDistribution {vocabSize : Nat}
    (target draft : Distribution vocabSize)
    (hReject : acceptanceRate target draft < 1) : Distribution vocabSize := by
  have hMass : 0 < residualMass target draft := by
    simpa [residualMass] using sub_pos.mpr hReject
  exact
    { probability := fun token =>
        residualNumerator target draft token / residualMass target draft
      nonnegative := fun token =>
        div_nonneg (le_max_left 0 _) hMass.le
      sumsToOne := by
        rw [← Finset.sum_div, sum_residualNumerator]
        exact div_self (ne_of_gt hMass) }

/-- One-token speculative sampling is lossless. For every token, the mass accepted from the draft
plus the mass drawn from the rejection correction is exactly the target-model mass. -/
theorem accepted_add_corrected_eq_target {vocabSize : Nat}
    (target draft : Distribution vocabSize)
    (hReject : acceptanceRate target draft < 1) (token : Fin vocabSize) :
    min (target.probability token) (draft.probability token) +
        residualMass target draft *
          (correctionDistribution target draft hReject).probability token =
      target.probability token := by
  have hMass : residualMass target draft ≠ 0 := by
    apply ne_of_gt
    simpa [residualMass] using sub_pos.mpr hReject
  rw [correctionDistribution]
  change min (target.probability token) (draft.probability token) +
      residualMass target draft *
        (residualNumerator target draft token / residualMass target draft) =
    target.probability token
  rw [mul_div_cancel₀ _ hMass]
  rw [residualNumerator_eq_sub_min]
  ring

namespace Model

/-- State of EAGLE-3 training-time testing after target features have been fused once. The second
component is the draft hidden feature consumed by the following step. -/
abbrev TrainingTimeTestState (model : Model cfg decayRank) :=
  model.CausalState × Tensor ℝ (.dim cfg.hiddenDim .scalar)

/-- Initial training-time-test state: an empty draft cache and the fused target-model feature. -/
noncomputable def initialTrainingTimeTestState (model : Model cfg decayRank)
    (features : FeatureTriplet ℝ cfg.hiddenDim) : model.TrainingTimeTestState :=
  (model.initialState, model.featureFusion.forward features)

/-- One EAGLE-3 training-time-test transition. The transition consumes the current token and feature,
then stores its own hidden output as the feature for the next transition. No fresh target-model
feature is requested after initialization. -/
noncomputable def trainingTimeTestStep (model : Model cfg decayRank) :
    model.TrainingTimeTestState → Fin cfg.vocabSize →
      model.TrainingTimeTestState × Distribution cfg.vocabSize :=
  fun state token =>
    let result := model.stepDistributionFromFeature state.1 token state.2
    ((result.1, result.2.1), result.2.2)

/-- Run EAGLE-3 training-time testing over shifted training tokens. Kimi K3 supplies seven tokens.
Only the initial state contains target-model features; each transition stores its own hidden output
as the feature consumed by the next transition. -/
noncomputable def trainingTimeTest (model : Model cfg decayRank)
    (tokens : List (Fin cfg.vocabSize)) (features : FeatureTriplet ℝ cfg.hiddenDim) :
    model.TrainingTimeTestState × List (Distribution cfg.vocabSize) :=
  CausalScan.run model.trainingTimeTestStep
    (model.initialTrainingTimeTestState features) tokens

/-- Training-time testing emits one draft distribution for every shifted training token. -/
theorem trainingTimeTest_length (model : Model cfg decayRank)
    (tokens : List (Fin cfg.vocabSize)) (features : FeatureTriplet ℝ cfg.hiddenDim) :
    (model.trainingTimeTest tokens features).2.length = tokens.length :=
  CausalScan.outputs_length model.trainingTimeTestStep
    (model.initialTrainingTimeTestState features) tokens

/-- Splitting a training-time-test token sequence into chunks does not change its final state or
draft distributions. This permits the seven reported steps to be replayed incrementally. -/
theorem trainingTimeTest_append (model : Model cfg decayRank)
    (initialTokens remainingTokens : List (Fin cfg.vocabSize))
    (features : FeatureTriplet ℝ cfg.hiddenDim) :
    model.trainingTimeTest (initialTokens ++ remainingTokens) features =
      let first := model.trainingTimeTest initialTokens features
      let second := CausalScan.run model.trainingTimeTestStep first.1 remainingTokens
      (second.1, first.2 ++ second.2) := by
  exact CausalScan.run_append model.trainingTimeTestStep
    (model.initialTrainingTimeTestState features) initialTokens remainingTokens

/-- The seven draft distributions used by K3 fine-tuning. Indexing by `Fin 7` makes the reported
training-time-test depth part of the interface instead of a comment or runtime assertion. -/
noncomputable def sevenStepDistributions (model : Model cfg decayRank)
    (tokens : Fin 7 → Fin cfg.vocabSize) (features : FeatureTriplet ℝ cfg.hiddenDim) :
    Fin 7 → Distribution cfg.vocabSize := by
  let outputs := (model.trainingTimeTest (List.ofFn tokens) features).2
  have hLength : outputs.length = 7 := by
    rw [model.trainingTimeTest_length]
    simp
  exact fun step => outputs.get ⟨step.val, by rw [hLength]; exact step.isLt⟩

end Model

/-- Likelihood-based draft loss in Eq. 16.

The codomain is `EReal` because disjoint target and draft distributions have zero acceptance and
therefore infinite negative-log loss.  Using an extended real here preserves that boundary case
instead of inheriting Lean's convention `Real.log 0 = 0`.
-/
noncomputable def likelihoodLoss {vocabSize : Nat}
    (target draft : Distribution vocabSize) : EReal :=
  if acceptanceRate target draft = 0 then ⊤
  else ((-Real.log (acceptanceRate target draft) : ℝ) : EReal)

/-- Zero speculative-sampling acceptance is exactly the infinite-loss case. -/
theorem likelihoodLoss_eq_top_iff {vocabSize : Nat}
    (target draft : Distribution vocabSize) :
    likelihoodLoss target draft = ⊤ ↔ acceptanceRate target draft = 0 := by
  simp [likelihoodLoss]

/-- At positive acceptance, the extended loss agrees with the ordinary real negative log. -/
theorem likelihoodLoss_of_pos {vocabSize : Nat}
    (target draft : Distribution vocabSize)
    (hAcceptance : 0 < acceptanceRate target draft) :
    likelihoodLoss target draft =
      ((-Real.log (acceptanceRate target draft) : ℝ) : EReal) := by
  simp [likelihoodLoss, ne_of_gt hAcceptance]

namespace Model

/-- Kimi K3's seven-step EAGLE-3 objective against a frozen target language model. Each target
distribution comes from the target model at the corresponding training position, and each draft
distribution comes from the recurrent training-time-test pass. There is no auxiliary ground-truth
cross-entropy term in this objective, matching Eq. 16 of the report. -/
noncomputable def sevenStepLikelihoodLoss {targetDecayRank draftDecayRank sequenceLength : Nat}
    (target : LanguageModel ℝ cfg targetDecayRank) (draft : Model cfg draftDecayRank)
    (embeddings : Tensor ℝ (.dim sequenceLength (.dim cfg.hiddenDim .scalar)))
    (positions : Fin 7 → Fin sequenceLength) (tokens : Fin 7 → Fin cfg.vocabSize)
    (features : FeatureTriplet ℝ cfg.hiddenDim) : EReal :=
  ∑ step : Fin 7,
    likelihoodLoss (targetDistributionAt target embeddings (positions step))
      (draft.sevenStepDistributions tokens features step)

end Model

end Draft

end KimiK3
