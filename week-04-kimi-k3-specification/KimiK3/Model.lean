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

The other Kimi K3 modules specify individual operations, and this file puts them in backbone
order. AttnRes retrieves the input to a sequence mixer (KDA or Gated MLA), a second AttnRes
retrieval feeds the dense or Stable LatentMoE channel mixer, and the result is recorded into the
current depth block.  A final AttnRes query aggregates the retained block representations before
the vocabulary projection. Sparse layers compute their top-k routes from their own adjusted router
scores, and `Route.chooseTopK` breaks score ties by expert index.

Reference: Kimi Team, "Kimi K3: Open Frontier Intelligence", 2026, Figure 2, p. 3, and
Sections 2.1--2.4, pp. 4--10, https://arxiv.org/abs/2607.24653.
-/

@[expose] public section

namespace KimiK3

open Spec TorchLean
open Tensor

/-- Either recurrent KDA or global Gated MLA at one backbone layer. -/
inductive SequenceMixer (α : Type) [Storage α] (cfg : TextConfig) (decayRank : Nat) where
  | kda :
      KDA.Layer α cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
        cfg.shortConvWidth decayRank → α → SequenceMixer α cfg decayRank
  | mla :
      GatedMLA α cfg.hiddenDim cfg.numHeads cfg.queryLatentDim cfg.kvLatentDim
        cfg.qkNopeHeadDim cfg.qkReservedHeadDim cfg.valueHeadDim → SequenceMixer α cfg decayRank

namespace SequenceMixer

variable {α : Type} [Storage α] [Context α]
variable {cfg : TextConfig} {decayRank : Nat}

/-- Schedule tag of a concrete sequence mixer. -/
def kind : SequenceMixer α cfg decayRank → AttentionKind
  | .kda _ _ => .kda
  | .mla _ => .mla

/-- Cache or state carried between causal token steps: KDA's recurrent matrices and short
convolution history, or MLA's compressed latent cache. -/
def CausalState : (mixer : SequenceMixer α cfg decayRank) → Type
  | .kda _ _ =>
      KDA.Layer.ScanState α cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
  | .mla _ => GatedMLA.Cache α cfg.kvLatentDim cfg.qkReservedHeadDim

/-- Empty causal state for either sequence mixer. -/
def initialState (mixer : SequenceMixer α cfg decayRank) : mixer.CausalState :=
  match mixer with
  | .kda _ _ =>
      let headState : KDA.Layer.State α cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim :=
        Tensor.zeros [cfg.numHeads, cfg.kdaHeadDim, cfg.kdaValueDim]
      (headState, [])
  | .mla _ => []

/-- Advance KDA or Gated MLA by one token, returning the updated causal state and the output. -/
def step (mixer : SequenceMixer α cfg decayRank) (state : mixer.CausalState)
    (token : Tensor α [cfg.hiddenDim]) :
    mixer.CausalState × Tensor α [cfg.hiddenDim] :=
  match mixer with
  | .kda layer logFloor => layer.scanStep logFloor state token
  | .mla layer => layer.step state token

/-- Run one sequence mixer over a token list, starting from its empty state. -/
def forward (mixer : SequenceMixer α cfg decayRank)
    (tokens : List (Tensor α [cfg.hiddenDim])) :
    List (Tensor α [cfg.hiddenDim]) :=
  (CausalScan.run mixer.step mixer.initialState tokens).2

/-- An empty sequence produces no mixed representations. -/
@[simp] theorem forward_nil (mixer : SequenceMixer α cfg decayRank) :
    mixer.forward [] = [] := by
  simp [forward]

/-- Both sequence mixers return one representation per input token. -/
theorem forward_length (mixer : SequenceMixer α cfg decayRank)
    (tokens : List (Tensor α [cfg.hiddenDim])) :
    (mixer.forward tokens).length = tokens.length := by
  exact CausalScan.outputs_length mixer.step mixer.initialState tokens

end SequenceMixer

/-- Dense FFN of the leading layers, or the sparse Stable LatentMoE used by later layers. -/
inductive ChannelMixer (α : Type) [Storage α] (cfg : TextConfig) where
  | dense : Expert α cfg.hiddenDim cfg.denseHiddenDim cfg.hiddenDim → ChannelMixer α cfg
  | sparse :
      StableLatentMoE α cfg.hiddenDim cfg.routedLatentDim cfg.routedExpertHiddenDim
        cfg.routedExpertHiddenDim cfg.numSharedExperts cfg.numRoutedExperts cfg.activeExperts →
      ChannelMixer α cfg

namespace ChannelMixer

variable {α : Type} [Storage α] [Context α]
variable {cfg : TextConfig}

/-- Whether this channel mixer is the dense FFN. -/
def isDense : ChannelMixer α cfg → Bool
  | .dense _ => true
  | .sparse _ => false

/-- Evaluate one token with a given route (dense layers ignore it). Useful for checking a backend
selector against the mathematical top-k route. -/
def forwardGivenRoute (mixer : ChannelMixer α cfg)
    (route : Route cfg.numRoutedExperts cfg.activeExperts)
    (x : Tensor α [cfg.hiddenDim]) : Tensor α [cfg.hiddenDim] :=
  match mixer with
  | .dense expert => expert.forward (cfg.situGateCap : α) (cfg.situUpCap : α) x
  | .sparse moe => moe.forward route (cfg.situGateCap : α) (cfg.situUpCap : α) x

/-- Evaluate a real-valued channel mixer on one token. Sparse layers choose their experts from the
adjusted router scores, and dense layers need no routing decision. -/
noncomputable def forward (mixer : ChannelMixer ℝ cfg)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (x : Tensor ℝ [cfg.hiddenDim]) : Tensor ℝ [cfg.hiddenDim] :=
  match mixer with
  | .dense expert => expert.forward cfg.situGateCap cfg.situUpCap x
  | .sparse moe => moe.forwardReal hActive cfg.situGateCap cfg.situUpCap x

end ChannelMixer

/-- One K3 backbone layer, including separate AttnRes queries for its two submodules. -/
structure BackboneLayer (α : Type) [Storage α] (cfg : TextConfig) (decayRank : Nat) where
  /-- Learned depth query for the sequence mixer input. -/
  sequenceQuery : Tensor α [cfg.hiddenDim]
  /-- RMSNorm scale applied before sequence mixing. -/
  sequenceNormScale : Tensor α [cfg.hiddenDim]
  /-- KDA or MLA parameters selected by this layer's schedule position. -/
  sequence : SequenceMixer α cfg decayRank
  /-- Learned depth query for the channel mixer input. -/
  channelQuery : Tensor α [cfg.hiddenDim]
  /-- RMSNorm scale applied before channel mixing. -/
  channelNormScale : Tensor α [cfg.hiddenDim]
  /-- Dense FFN or routed Stable LatentMoE parameters. -/
  channel : ChannelMixer α cfg

namespace BackboneLayer

variable {α : Type} [Storage α] [Context α]
variable {cfg : TextConfig} {decayRank : Nat}

/-- Initialize the depth-memory state of one token from its embedding. -/
def initialDepthState (embedding : Tensor α [cfg.hiddenDim]) :
    AttnRes.BlockState α cfg.hiddenDim :=
  { embedding
    completedBlocks := []
    partialBlock := Tensor.zeros [cfg.hiddenDim]
    partialSize := 0 }

/-- An initial depth state is valid for every positive AttnRes block size. -/
theorem initialDepthState_wf (embedding : Tensor α [cfg.hiddenDim])
    (hBlockSize : 0 < cfg.attnResBlockSize) :
    (initialDepthState embedding).WF cfg.attnResBlockSize := by
  exact hBlockSize

/-- Apply one backbone layer to all tokenwise depth states. -/
noncomputable def forward (layer : BackboneLayer ℝ cfg decayRank)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (states : List (AttnRes.BlockState ℝ cfg.hiddenDim)) :
    List (AttnRes.BlockState ℝ cfg.hiddenDim) :=
  let sequenceInput := states.map (fun state => state.retrieve layer.sequenceQuery)
  let normalizedSequenceInput := sequenceInput.map
    (fun token => RMSNorm.scale token layer.sequenceNormScale)
  let sequenceOutput := layer.sequence.forward normalizedSequenceInput
  let channelInput := List.zipWith
    (fun state output => state.retrieveAfterSequence output layer.channelQuery) states
    sequenceOutput
  let channelOutput := channelInput.map (fun token =>
    layer.channel.forward hActive (RMSNorm.scale token layer.channelNormScale))
  List.zipWith
    (fun state outputs =>
      state.finishLayer cfg.attnResBlockSize outputs.1 outputs.2)
    states (List.zip sequenceOutput channelOutput)

/-- A layer preserves the number of tokenwise depth states. -/
theorem forward_length (layer : BackboneLayer ℝ cfg decayRank)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (states : List (AttnRes.BlockState ℝ cfg.hiddenDim)) :
    (layer.forward hActive states).length = states.length := by
  let sequenceInput := states.map (fun state => state.retrieve layer.sequenceQuery)
  let normalizedSequenceInput := sequenceInput.map
    (fun token => RMSNorm.scale token layer.sequenceNormScale)
  let sequenceOutput := layer.sequence.forward normalizedSequenceInput
  let channelInput := List.zipWith
    (fun state output => state.retrieveAfterSequence output layer.channelQuery) states
    sequenceOutput
  let channelOutput := channelInput.map (fun token =>
    layer.channel.forward hActive (RMSNorm.scale token layer.channelNormScale))
  have hSequence : sequenceOutput.length = states.length := by
    rw [layer.sequence.forward_length]
    simp [normalizedSequenceInput, sequenceInput]
  have hChannelInput : channelInput.length = states.length := by
    simp [channelInput, hSequence]
  have hChannel : channelOutput.length = states.length := by
    simp [channelOutput, hChannelInput]
  change (List.zipWith _ states (List.zip sequenceOutput channelOutput)).length = states.length
  simp [hSequence, hChannel]

/-- Advance one backbone layer by a single autoregressive token, threading the sequence mixer's
causal state. Both the streaming decoder and the EAGLE draft are built from this step. -/
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

/-- Evaluating a layer over the whole sequence agrees with advancing its causal state one token
at a time. This covers the full layer: both AttnRes retrievals, both normalizations, the
sequence mixer, and the routed channel mixer. -/
theorem forward_eq_scan (layer : BackboneLayer ℝ cfg decayRank)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (states : List (AttnRes.BlockState ℝ cfg.hiddenDim)) :
    layer.forward hActive states =
      (CausalScan.run (layer.forwardToken hActive) layer.sequence.initialState states).2 := by
  let prepare := fun state : AttnRes.BlockState ℝ cfg.hiddenDim =>
    RMSNorm.scale (state.retrieve layer.sequenceQuery) layer.sequenceNormScale
  let finish := fun (state : AttnRes.BlockState ℝ cfg.hiddenDim)
      (output : Tensor ℝ [cfg.hiddenDim]) =>
    state.finishLayer cfg.attnResBlockSize output
      (layer.channel.forward hActive
        (RMSNorm.scale (state.retrieveAfterSequence output layer.channelQuery)
          layer.channelNormScale))
  have hRun := CausalScan.run_map layer.sequence.step prepare finish
    layer.sequence.initialState states
  change layer.forward hActive states = (CausalScan.run
    (fun current input =>
      let result := layer.sequence.step current (prepare input)
      (result.1, finish input result.2)) layer.sequence.initialState states).2
  rw [hRun]
  simp only
  unfold forward SequenceMixer.forward
  simp only [List.map_map]
  have hZip (xs : List (AttnRes.BlockState ℝ cfg.hiddenDim))
      (outputs : List (Tensor ℝ [cfg.hiddenDim])) :
      List.zipWith (fun state outputs =>
        state.finishLayer cfg.attnResBlockSize outputs.1 outputs.2) xs
        (outputs.zip ((xs.zipWith
          (fun state output => state.retrieveAfterSequence output layer.channelQuery)
          outputs).map (fun token => layer.channel.forward hActive
            (RMSNorm.scale token layer.channelNormScale)))) =
        List.zipWith finish xs outputs := by
    induction xs generalizing outputs with
    | nil => simp
    | cons state rest ih =>
      cases outputs with
      | nil => simp
      | cons output outputs =>
        simp only [List.zipWith_cons_cons, List.map_cons, List.zip_cons_cons]
        exact congrArg (List.cons (finish state output)) (ih outputs)
  exact hZip states _

/-- A one-token backbone step preserves the AttnRes block-position invariant. -/
theorem forwardToken_depth_wf (layer : BackboneLayer ℝ cfg decayRank)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (sequenceState : layer.sequence.CausalState)
    (depthState : AttnRes.BlockState ℝ cfg.hiddenDim)
    (hDepth : depthState.WF cfg.attnResBlockSize) :
    (layer.forwardToken hActive sequenceState depthState).2.WF cfg.attnResBlockSize := by
  simp only [forwardToken]
  exact depthState.finishLayer_wf cfg.attnResBlockSize _ _ hDepth

end BackboneLayer

/-- Complete language-backbone parameter bundle.

Its four proof fields make the architectural invariants part of the model value. Routing
feasibility needs no separate proof, and a caller cannot attach an arbitrary sequence of KDA, MLA,
dense, and sparse layers to a K3 configuration.
-/
structure LanguageModel (α : Type) [Storage α] (cfg : TextConfig) (decayRank : Nat) where
  /-- Shared vocabulary embedding table. -/
  tokenEmbedding : Tensor α [cfg.vocabSize, cfg.hiddenDim]
  /-- Shape-indexed parameters for every configured backbone layer. -/
  layer : Fin cfg.numLayers → BackboneLayer α cfg decayRank
  /-- Final learned query over AttnRes block summaries. -/
  finalQuery : Tensor α [cfg.hiddenDim]
  /-- RMSNorm scale applied after final depth retrieval. -/
  finalNormScale : Tensor α [cfg.hiddenDim]
  /-- Projection from hidden width to vocabulary logits. -/
  vocabularyHead : Tensor α [cfg.hiddenDim, cfg.vocabSize]
  /-- Nonemptiness required by typed vocabulary operations. -/
  vocabSize_pos : 0 < cfg.vocabSize
  /-- Routing capacity required by every sparse channel mixer. -/
  activeExperts_le : cfg.activeExperts ≤ cfg.numRoutedExperts
  /-- Stored sequence mixers agree with the configured KDA/MLA schedule. -/
  sequenceSchedule : ∀ index, (layer index).sequence.kind = cfg.attentionKindAt index
  /-- Stored channel mixers agree with the configured dense-prefix schedule. -/
  channelSchedule :
    ∀ index, (layer index).channel.isDense = decide (index.val < cfg.firstDenseLayers)

/-- One position in the sequence presented to the shared language backbone.

Text positions index the vocabulary embedding table. Visual positions index the sequence emitted
by MoonViT after temporal pooling, spatial merging, and projection to the language width. With
typed alternatives, the repeated media placeholder of serialized prompts never gets an ordinary
vocabulary meaning.
-/
inductive InputSlot (vocabSize visualTokens : Nat) where
  | text (token : Fin vocabSize)
  | visual (token : Fin visualTokens)
  deriving DecidableEq

namespace LanguageModel

variable {α : Type} [Storage α] [Context α]
variable {cfg : TextConfig} {decayRank : Nat}

/-- Look up a fixed-length sequence of text-token embeddings. -/
def embedText {sequenceLength : Nat} (model : LanguageModel α cfg decayRank)
    (tokens : Fin sequenceLength → Fin cfg.vocabSize) :
    Tensor α [sequenceLength, cfg.hiddenDim] :=
  Tensor.stackLeading fun position => Spec.get model.tokenEmbedding (tokens position)

omit [Context α] in
/-- Text embedding is a row lookup in the shared vocabulary table. -/
@[simp] theorem get_embedText {sequenceLength : Nat} (model : LanguageModel α cfg decayRank)
    (tokens : Fin sequenceLength → Fin cfg.vocabSize) (position : Fin sequenceLength) :
    Spec.get (model.embedText tokens) position =
      Spec.get model.tokenEmbedding (tokens position) := by
  simp [embedText, Spec.get]

/-- Resolve a mixed text/visual sequence into language-width embeddings. Slots play the role of
expanding a serialized media placeholder into MoonViT's visual tokens. -/
def embedMultimodal {sequenceLength visualTokens : Nat}
    (model : LanguageModel α cfg decayRank)
    (visual : Tensor α [visualTokens, cfg.hiddenDim])
    (slots : Fin sequenceLength → InputSlot cfg.vocabSize visualTokens) :
    Tensor α [sequenceLength, cfg.hiddenDim] :=
  Tensor.stackLeading fun position =>
    match slots position with
    | .text token => Spec.get model.tokenEmbedding token
    | .visual token => Spec.get visual token

omit [Context α] in
/-- A text slot is resolved from the vocabulary embedding table. -/
theorem get_embedMultimodal_text {sequenceLength visualTokens : Nat}
    (model : LanguageModel α cfg decayRank)
    (visual : Tensor α [visualTokens, cfg.hiddenDim])
    (slots : Fin sequenceLength → InputSlot cfg.vocabSize visualTokens)
    (position : Fin sequenceLength) (token : Fin cfg.vocabSize)
    (hSlot : slots position = .text token) :
    Spec.get (model.embedMultimodal visual slots) position =
      Spec.get model.tokenEmbedding token := by
  simp [embedMultimodal, hSlot]

omit [Context α] in
/-- A visual slot is resolved from MoonViT's projected token sequence. -/
theorem get_embedMultimodal_visual {sequenceLength visualTokens : Nat}
    (model : LanguageModel α cfg decayRank)
    (visual : Tensor α [visualTokens, cfg.hiddenDim])
    (slots : Fin sequenceLength → InputSlot cfg.vocabSize visualTokens)
    (position : Fin sequenceLength) (token : Fin visualTokens)
    (hSlot : slots position = .visual token) :
    Spec.get (model.embedMultimodal visual slots) position = Spec.get visual token := by
  simp [embedMultimodal, hSlot]

/-- Run MoonViT and splice its projected tokens into a mixed language-model sequence. -/
noncomputable def embedVisionText {visionCfg : VisionConfig}
    {frames rows columns patchFeatures sequenceLength visualTokens : Nat}
    (model : LanguageModel ℝ cfg decayRank)
    (vision : MoonViT.Model ℝ visionCfg patchFeatures)
    (patches : MoonViT.Grid ℝ frames
      (rows * visionCfg.mergeHeight) (columns * visionCfg.mergeWidth) patchFeatures)
    (hFrames : 0 < frames) (hMaxFrames : frames ≤ MoonViT.maxFrames)
    (hSpatial : 0 < (rows * visionCfg.mergeHeight) *
      (columns * visionCfg.mergeWidth))
    (hVisionHidden : 0 < visionCfg.hiddenDim)
    (hVisionText : 0 < visionCfg.textHiddenDim)
    (hVisualTokens : rows * columns = visualTokens)
    (hHidden : visionCfg.textHiddenDim = cfg.hiddenDim)
    (slots : Fin sequenceLength → InputSlot cfg.vocabSize visualTokens) :
    Tensor ℝ [sequenceLength, cfg.hiddenDim] := by
  let encoded := vision.forward patches hFrames hMaxFrames hSpatial hVisionHidden hVisionText
  have aligned : Tensor ℝ [visualTokens, cfg.hiddenDim] := by
    simpa [hVisualTokens, hHidden] using encoded
  exact model.embedMultimodal aligned slots

/-- Run every backbone layer and keep the final AttnRes state at each sequence position. EAGLE-3
needs this state because its draft input reads intermediate AttnRes block outputs. -/
noncomputable def forwardDepthStates {sequenceLength : Nat}
    (model : LanguageModel ℝ cfg decayRank)
    (embeddings : Tensor ℝ [sequenceLength, cfg.hiddenDim]) :
    Fin sequenceLength → AttnRes.BlockState ℝ cfg.hiddenDim := by
  let initialEmbeddings := List.ofFn fun position => Spec.get embeddings position
  let initial := initialEmbeddings.map BackboneLayer.initialDepthState
  let finalStates := (List.finRange cfg.numLayers).foldl
    (fun states index => (model.layer index).forward model.activeExperts_le states) initial
  have hFoldGeneral : ∀ (indices : List (Fin cfg.numLayers))
      (states : List (AttnRes.BlockState ℝ cfg.hiddenDim)),
      (indices.foldl (fun states index =>
        (model.layer index).forward model.activeExperts_le states) states).length =
        states.length := by
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
    (embeddings : Tensor ℝ [sequenceLength, cfg.hiddenDim]) :
    Tensor ℝ [sequenceLength, cfg.hiddenDim] :=
  Tensor.stackLeading fun position =>
    let state := model.forwardDepthStates embeddings position
    RMSNorm.scale (state.retrieve model.finalQuery) model.finalNormScale

/-- Project final hidden representations to vocabulary logits. -/
noncomputable def logits {sequenceLength : Nat} (model : LanguageModel ℝ cfg decayRank)
    (embeddings : Tensor ℝ [sequenceLength, cfg.hiddenDim]) :
    Tensor ℝ [sequenceLength, cfg.vocabSize] :=
  Tensor.stackLeading fun position =>
    vecMatMulSpec (Spec.get (model.forwardHidden embeddings) position) model.vocabularyHead

end LanguageModel

end KimiK3
