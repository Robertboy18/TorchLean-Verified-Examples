/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.FeedForward
public import KimiK3.Sequence
public import KimiK3.Vision

/-!
# Composed Kimi K3 language model

The preceding Kimi K3 modules specify individual operations. This file puts them in the order used
by the backbone: AttnRes retrieves the input to a sequence mixer, the sequence mixer is either KDA
or Gated MLA, a second AttnRes retrieval feeds the dense/Stable-LatentMoE channel mixer, and the
result is recorded into the current depth block.  A final AttnRes query aggregates the retained
block representations before the vocabulary projection. Sparse layers compute their top-k routes
from their own adjusted router scores. Exact score ties are resolved by expert index, as specified in
`Route.chooseTopK`.
-/

@[expose] public section

namespace KimiK3

open Spec
open Tensor

/-- Either recurrent KDA or global Gated MLA at one backbone layer. -/
inductive SequenceMixer (α : Type) (cfg : TextConfig) (decayRank : Nat) where
  | kda :
      KDALayer α cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
        cfg.shortConvWidth decayRank → α → SequenceMixer α cfg decayRank
  | mla :
      GatedMLA α cfg.hiddenDim cfg.numHeads cfg.queryLatentDim cfg.kvLatentDim
        cfg.qkNopeHeadDim cfg.qkReservedHeadDim cfg.valueHeadDim → SequenceMixer α cfg decayRank

namespace SequenceMixer

variable {α : Type} [Context α]
variable {cfg : TextConfig} {decayRank : Nat}

/-- The schedule tag of a concrete sequence mixer. -/
def kind : SequenceMixer α cfg decayRank → AttentionKind
  | .kda _ _ => .kda
  | .mla _ => .mla

/-- Cache/state carried between causal token steps. KDA retains its recurrent matrices and short
convolution history; MLA retains the compressed latent cache. -/
def CausalState : (mixer : SequenceMixer α cfg decayRank) → Type
  | .kda _ _ =>
      KDALayer.ScanState α cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
  | .mla _ => GatedMLA.Cache α cfg.kvLatentDim cfg.qkReservedHeadDim

/-- Empty causal state for either sequence mixer. -/
def initialState (mixer : SequenceMixer α cfg decayRank) : mixer.CausalState :=
  match mixer with
  | .kda _ _ =>
      let headState : KDALayer.State α cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim :=
        Spec.fill 0
          (.dim cfg.numHeads (.dim cfg.kdaHeadDim (.dim cfg.kdaValueDim .scalar)))
      (headState, [])
  | .mla _ => []

/-- Advance either KDA or Gated MLA by one token while retaining its causal state. -/
def step (mixer : SequenceMixer α cfg decayRank) (state : mixer.CausalState)
    (token : Tensor α (.dim cfg.hiddenDim .scalar)) :
    mixer.CausalState × Tensor α (.dim cfg.hiddenDim .scalar) :=
  match mixer with
  | .kda layer logFloor => layer.scanStep logFloor state token
  | .mla layer => layer.step state token

/-- Run one sequence-mixing layer from an empty sequence cache/state. -/
def forward (mixer : SequenceMixer α cfg decayRank)
    (tokens : List (Tensor α (.dim cfg.hiddenDim .scalar))) :
    List (Tensor α (.dim cfg.hiddenDim .scalar)) :=
  (CausalScan.run mixer.step mixer.initialState tokens).2

/-- Both sequence mixers return one representation per input token. -/
theorem forward_length (mixer : SequenceMixer α cfg decayRank)
    (tokens : List (Tensor α (.dim cfg.hiddenDim .scalar))) :
    (mixer.forward tokens).length = tokens.length := by
  exact CausalScan.outputs_length mixer.step mixer.initialState tokens

end SequenceMixer

/-- Dense first-layer FFN or sparse Stable LatentMoE used by later layers. -/
inductive ChannelMixer (α : Type) (cfg : TextConfig) where
  | dense : Expert α cfg.hiddenDim cfg.denseHiddenDim cfg.hiddenDim → ChannelMixer α cfg
  | sparse :
      StableLatentMoE α cfg.hiddenDim cfg.routedLatentDim cfg.routedExpertHiddenDim
        cfg.routedExpertHiddenDim cfg.numSharedExperts cfg.numRoutedExperts cfg.activeExperts →
      ChannelMixer α cfg

namespace ChannelMixer

variable {α : Type} [Context α]
variable {cfg : TextConfig}

/-- Whether this value is the initial dense channel mixer. -/
def isDense : ChannelMixer α cfg → Bool
  | .dense _ => true
  | .sparse _ => false

/-- Evaluate one token using an explicitly supplied route. This lower-level operation is useful when
checking a backend selector against the mathematical top-k route. -/
def forwardGivenRoute (mixer : ChannelMixer α cfg)
    (route : Route cfg.numRoutedExperts cfg.activeExperts)
    (x : Tensor α (.dim cfg.hiddenDim .scalar)) : Tensor α (.dim cfg.hiddenDim .scalar) :=
  match mixer with
  | .dense expert => expert.forward (cfg.situGateCap : α) (cfg.situUpCap : α) x
  | .sparse moe => moe.forward route (cfg.situGateCap : α) (cfg.situUpCap : α) x

/-- Evaluate one real-valued channel mixer. Sparse layers choose their experts from the adjusted
router scores; dense layers require no routing decision. -/
noncomputable def forward (mixer : ChannelMixer ℝ cfg)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (x : Tensor ℝ (.dim cfg.hiddenDim .scalar)) : Tensor ℝ (.dim cfg.hiddenDim .scalar) :=
  match mixer with
  | .dense expert => expert.forward cfg.situGateCap cfg.situUpCap x
  | .sparse moe => moe.forwardReal hActive cfg.situGateCap cfg.situUpCap x

end ChannelMixer

/-- One K3 backbone layer, including separate AttnRes queries for its two submodules. -/
structure BackboneLayer (α : Type) (cfg : TextConfig) (decayRank : Nat) where
  sequenceQuery : Tensor α (.dim cfg.hiddenDim .scalar)
  sequenceNormScale : Tensor α (.dim cfg.hiddenDim .scalar)
  sequence : SequenceMixer α cfg decayRank
  channelQuery : Tensor α (.dim cfg.hiddenDim .scalar)
  channelNormScale : Tensor α (.dim cfg.hiddenDim .scalar)
  channel : ChannelMixer α cfg

namespace BackboneLayer

variable {α : Type} [Context α]
variable {cfg : TextConfig} {decayRank : Nat}

/-- Initialize the depth-memory state of one token from its embedding. -/
def initialDepthState (embedding : Tensor α (.dim cfg.hiddenDim .scalar)) :
    AttnRes.BlockState α cfg.hiddenDim :=
  { embedding
    completedBlocks := []
    partialBlock := Spec.fill 0 (.dim cfg.hiddenDim .scalar)
    partialSize := 0 }

/-- Retrieve one input per token from the current depth state. -/
def retrieveAll (states : List (AttnRes.BlockState α cfg.hiddenDim))
    (query : Tensor α (.dim cfg.hiddenDim .scalar)) :
    List (Tensor α (.dim cfg.hiddenDim .scalar)) :=
  states.map (fun state => state.retrieve query)

/-- Retrieve channel inputs after including the sequence output from the same decoder layer. -/
def retrieveAfterSequenceAll (states : List (AttnRes.BlockState α cfg.hiddenDim))
    (sequenceOutputs : List (Tensor α (.dim cfg.hiddenDim .scalar)))
    (query : Tensor α (.dim cfg.hiddenDim .scalar)) :
    List (Tensor α (.dim cfg.hiddenDim .scalar)) :=
  List.zipWith (fun state output => state.retrieveAfterSequence output query) states sequenceOutputs

/-- Commit both submodule outputs while advancing each token's block state once. -/
def finishAll (states : List (AttnRes.BlockState α cfg.hiddenDim))
    (sequenceOutputs channelOutputs : List (Tensor α (.dim cfg.hiddenDim .scalar))) :
    List (AttnRes.BlockState α cfg.hiddenDim) :=
  List.zipWith
    (fun state outputs => state.finishLayer cfg.attnResBlockSize outputs.1 outputs.2)
    states (List.zip sequenceOutputs channelOutputs)

/-- Apply one backbone layer to all tokenwise depth states. -/
noncomputable def forward (layer : BackboneLayer ℝ cfg decayRank)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (states : List (AttnRes.BlockState ℝ cfg.hiddenDim)) :
    List (AttnRes.BlockState ℝ cfg.hiddenDim) :=
  let sequenceInput := retrieveAll states layer.sequenceQuery
  let normalizedSequenceInput := sequenceInput.map
    (fun token => RMSNorm.scale token layer.sequenceNormScale)
  let sequenceOutput := layer.sequence.forward normalizedSequenceInput
  let channelInput := retrieveAfterSequenceAll states sequenceOutput layer.channelQuery
  let channelOutput := channelInput.mapIdx (fun _ token =>
    layer.channel.forward hActive (RMSNorm.scale token layer.channelNormScale))
  finishAll states sequenceOutput channelOutput

/-- A layer preserves the number of tokenwise depth states. -/
theorem forward_length (layer : BackboneLayer ℝ cfg decayRank)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (states : List (AttnRes.BlockState ℝ cfg.hiddenDim)) :
    (layer.forward hActive states).length = states.length := by
  let sequenceInput := retrieveAll states layer.sequenceQuery
  let normalizedSequenceInput := sequenceInput.map
    (fun token => RMSNorm.scale token layer.sequenceNormScale)
  let sequenceOutput := layer.sequence.forward normalizedSequenceInput
  let channelInput := retrieveAfterSequenceAll states sequenceOutput layer.channelQuery
  let channelOutput := channelInput.mapIdx (fun _ token =>
    layer.channel.forward hActive (RMSNorm.scale token layer.channelNormScale))
  have hSequence : sequenceOutput.length = states.length := by
    rw [layer.sequence.forward_length]
    simp [normalizedSequenceInput, sequenceInput, retrieveAll]
  have hChannelInput : channelInput.length = states.length := by
    simp [channelInput, retrieveAfterSequenceAll, hSequence]
  have hChannel : channelOutput.length = states.length := by
    simp [channelOutput, hChannelInput]
  change (finishAll states sequenceOutput channelOutput).length = states.length
  simp [finishAll, hSequence, hChannel]

/-- Apply one backbone layer to a single tokenwise depth state. -/
noncomputable def forwardOne (layer : BackboneLayer ℝ cfg decayRank)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (state : AttnRes.BlockState ℝ cfg.hiddenDim) : AttnRes.BlockState ℝ cfg.hiddenDim := by
  let outputs := layer.forward hActive [state]
  have hLength : outputs.length = 1 := layer.forward_length hActive [state]
  exact outputs.get ⟨0, by omega⟩

/-- Advance one backbone layer at a single autoregressive token while preserving the sequence
mixer's causal state. This is the primitive used by the EAGLE draft and streaming decoders. -/
noncomputable def forwardToken (layer : BackboneLayer ℝ cfg decayRank)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (sequenceState : layer.sequence.CausalState)
    (depthState : AttnRes.BlockState ℝ cfg.hiddenDim) :
    layer.sequence.CausalState × AttnRes.BlockState ℝ cfg.hiddenDim :=
  let sequenceInput := RMSNorm.scale (depthState.retrieve layer.sequenceQuery)
    layer.sequenceNormScale
  let sequenceResult := layer.sequence.step sequenceState sequenceInput
  let channelInput := depthState.retrieveAfterSequence sequenceResult.2 layer.channelQuery
  let channelOutput := layer.channel.forward hActive
    (RMSNorm.scale channelInput layer.channelNormScale)
  (sequenceResult.1,
    depthState.finishLayer cfg.attnResBlockSize sequenceResult.2 channelOutput)

end BackboneLayer

/-- The complete language-backbone parameter bundle.

The final three fields make the architectural invariants part of the model value. In particular,
there is no separately supplied proof that expert routing is possible, and a caller cannot attach an
arbitrary sequence of KDA, MLA, dense, and sparse layers to a K3 configuration.
-/
structure LanguageModel (α : Type) (cfg : TextConfig) (decayRank : Nat) where
  tokenEmbedding : Tensor α (.dim cfg.vocabSize (.dim cfg.hiddenDim .scalar))
  layer : Fin cfg.numLayers → BackboneLayer α cfg decayRank
  finalQuery : Tensor α (.dim cfg.hiddenDim .scalar)
  finalNormScale : Tensor α (.dim cfg.hiddenDim .scalar)
  vocabularyHead : Tensor α (.dim cfg.hiddenDim (.dim cfg.vocabSize .scalar))
  vocabSize_pos : 0 < cfg.vocabSize
  activeExperts_le : cfg.activeExperts ≤ cfg.numRoutedExperts
  sequenceSchedule : ∀ index, (layer index).sequence.kind = cfg.attentionKindAt index
  channelSchedule :
    ∀ index, (layer index).channel.isDense = decide (index.val < cfg.firstDenseLayers)

/-- One position in the sequence presented to the shared language backbone.

Text positions index the vocabulary embedding table. Visual positions index the sequence emitted
by MoonViT after temporal pooling, spatial merging, and projection to the language width. Keeping
the alternatives typed avoids assigning an ordinary vocabulary meaning to the repeated media
placeholder used by serialized prompts.
-/
inductive InputSlot (vocabSize visualTokens : Nat) where
  | text (token : Fin vocabSize)
  | visual (token : Fin visualTokens)
  deriving DecidableEq

namespace LanguageModel

variable {α : Type} [Context α]
variable {cfg : TextConfig} {decayRank : Nat}

/-- Look up a fixed-length sequence of text-token embeddings. -/
def embedText {sequenceLength : Nat} (model : LanguageModel α cfg decayRank)
    (tokens : Fin sequenceLength → Fin cfg.vocabSize) :
    Tensor α (.dim sequenceLength (.dim cfg.hiddenDim .scalar)) :=
  Tensor.dim fun position => Spec.get model.tokenEmbedding (tokens position)

/-- Resolve a mixed text/visual sequence into language-width embeddings. The slot list is the
formal counterpart of expanding a serialized media placeholder into the visual tokens produced by
MoonViT. -/
def embedMultimodal {sequenceLength visualTokens : Nat}
    (model : LanguageModel α cfg decayRank)
    (visual : Tensor α (.dim visualTokens (.dim cfg.hiddenDim .scalar)))
    (slots : Fin sequenceLength → InputSlot cfg.vocabSize visualTokens) :
    Tensor α (.dim sequenceLength (.dim cfg.hiddenDim .scalar)) :=
  Tensor.dim fun position =>
    match slots position with
    | .text token => Spec.get model.tokenEmbedding token
    | .visual token => Spec.get visual token

/-- Run MoonViT and splice its projected tokens into a mixed language-model sequence. -/
noncomputable def embedVisionText {visionCfg : VisionConfig}
    {frames rows columns patchFeatures sequenceLength visualTokens : Nat}
    (model : LanguageModel ℝ cfg decayRank)
    (vision : MoonViT.Model ℝ visionCfg frames
      (rows * visionCfg.mergeHeight) (columns * visionCfg.mergeWidth) patchFeatures)
    (patches : MoonViT.Grid ℝ frames
      (rows * visionCfg.mergeHeight) (columns * visionCfg.mergeWidth) patchFeatures)
    (hFrames : 0 < frames)
    (hSpatial : 0 < (rows * visionCfg.mergeHeight) *
      (columns * visionCfg.mergeWidth))
    (hVisionHidden : 0 < visionCfg.hiddenDim)
    (hVisionText : 0 < visionCfg.textHiddenDim)
    (hVisualTokens : rows * columns = visualTokens)
    (hHidden : visionCfg.textHiddenDim = cfg.hiddenDim)
    (slots : Fin sequenceLength → InputSlot cfg.vocabSize visualTokens) :
    Tensor ℝ (.dim sequenceLength (.dim cfg.hiddenDim .scalar)) := by
  let encoded := vision.forward patches hFrames hSpatial hVisionHidden hVisionText
  have aligned : Tensor ℝ (.dim visualTokens (.dim cfg.hiddenDim .scalar)) := by
    simpa [hVisualTokens, hHidden] using encoded
  exact model.embedMultimodal aligned slots

/-- Run every backbone layer and retain the final AttnRes state at each sequence position. Exposing
this state is needed by EAGLE-3, whose draft input reads intermediate AttnRes block outputs. -/
noncomputable def forwardDepthStates {sequenceLength : Nat}
    (model : LanguageModel ℝ cfg decayRank)
    (embeddings : Tensor ℝ (.dim sequenceLength (.dim cfg.hiddenDim .scalar))) :
    Fin sequenceLength → AttnRes.BlockState ℝ cfg.hiddenDim := by
  let initialEmbeddings := List.ofFn fun position => Spec.get embeddings position
  let initial := initialEmbeddings.map BackboneLayer.initialDepthState
  let finalStates := (List.finRange cfg.numLayers).foldl
    (fun states index => (model.layer index).forward model.activeExperts_le states) initial
  have hFoldGeneral : ∀ (indices : List (Fin cfg.numLayers))
      (states : List (AttnRes.BlockState ℝ cfg.hiddenDim)),
      (indices.foldl (fun states index =>
        (model.layer index).forward model.activeExperts_le states) states).length = states.length := by
    intro indices
    induction indices with
    | nil => simp
    | cons index rest ih =>
        intro states
        simp only [List.foldl_cons]
        rw [ih]
        exact BackboneLayer.forward_length (model.layer index) model.activeExperts_le states
  have hLength : finalStates.length = sequenceLength := by
    calc
      finalStates.length = initial.length :=
        hFoldGeneral (List.finRange cfg.numLayers) initial
      _ = sequenceLength := by simp [initial, initialEmbeddings]
  exact fun position => finalStates.get ⟨position.val, by rw [hLength]; exact position.isLt⟩

/-- Run the language backbone from already embedded text and/or visual tokens. -/
noncomputable def forwardHidden {sequenceLength : Nat}
    (model : LanguageModel ℝ cfg decayRank)
    (embeddings : Tensor ℝ (.dim sequenceLength (.dim cfg.hiddenDim .scalar))) :
    Tensor ℝ (.dim sequenceLength (.dim cfg.hiddenDim .scalar)) :=
  Tensor.dim fun position =>
    let state := model.forwardDepthStates embeddings position
    RMSNorm.scale (state.retrieve model.finalQuery) model.finalNormScale

/-- Project final hidden representations to vocabulary logits. -/
noncomputable def logits {sequenceLength : Nat} (model : LanguageModel ℝ cfg decayRank)
    (embeddings : Tensor ℝ (.dim sequenceLength (.dim cfg.hiddenDim .scalar))) :
    Tensor ℝ (.dim sequenceLength (.dim cfg.vocabSize .scalar)) :=
  Tensor.dim fun position =>
    vecMatMulSpec (Spec.get (model.forwardHidden embeddings) position) model.vocabularyHead

end LanguageModel

end KimiK3
