/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Backbone

/-!
# Decoder layer graphs

Each layer selects the existing KDA or MLA graph and its dense or sparse channel mixer.
The denotation theorem covers those four cases with the same parameter and state layout.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec

open Spec TorchLean NN.GraphSpec.DAG

/-- Split graph arguments at the parameter/state boundary of a decoder component. -/
def splitArgs {Γ : List Shape} : {left right : List Shape} →
    Args Γ (left ++ right) → Args Γ left × Args Γ right
  | [], _, args => (.nil, args)
  | _ :: _, _, .cons value rest =>
      let parts := splitArgs rest
      (.cons value parts.1, parts.2)

/-- Evaluating the left argument group selects the corresponding input tensors. -/
@[simp] theorem eval_splitArgs_left {α : Type} [Storage α] [Context α]
    {Γ left right : List Shape} (env : TorchLean.TensorPack α Γ)
    (args : Args Γ (left ++ right)) :
    Term.evalArgs env (splitArgs args).1 = (TorchLean.TensorPack.split (Term.evalArgs env args)).1 := by
  induction left with
  | nil => rfl
  | cons _ _ ih =>
      cases args with
      | cons value rest => simp [splitArgs, Term.evalArgs, TorchLean.TensorPack.split, ih]

/-- Evaluating the remaining argument group selects the remaining input tensors. -/
@[simp] theorem eval_splitArgs_right {α : Type} [Storage α] [Context α]
    {Γ left right : List Shape} (env : TorchLean.TensorPack α Γ)
    (args : Args Γ (left ++ right)) :
    Term.evalArgs env (splitArgs args).2 = (TorchLean.TensorPack.split (Term.evalArgs env args)).2 := by
  induction left with
  | nil => rfl
  | cons _ _ ih =>
      cases args with
      | cons value rest => simp [splitArgs, Term.evalArgs, TorchLean.TensorPack.split, ih]

namespace DecoderLayer

open Spec TorchLean
open TorchLean.Tensor
open NN.GraphSpec.DAG
open Runtime.Autograd.Torch

variable {cfg : TextConfig} {decayRank : Nat}

/-- Parameter shapes selected by a sequence mixer and channel mixer. -/
abbrev Params (sequence : SequenceMixer ℝ cfg decayRank) (channel : ChannelMixer ℝ cfg) :
    List Shape :=
  match sequence, channel with
  | .kda _ _, .dense _ =>
      Backbone.KDADense.Params cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
        cfg.shortConvWidth decayRank cfg.denseHiddenDim
  | .kda _ _, .sparse _ =>
      Backbone.KDASparse.Params cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
        cfg.shortConvWidth decayRank cfg.routedLatentDim cfg.routedExpertHiddenDim
        cfg.numSharedExperts cfg.numRoutedExperts
  | .mla _, .dense _ =>
      Backbone.MLADense.Params cfg.hiddenDim cfg.numHeads cfg.queryLatentDim cfg.kvLatentDim
        cfg.qkNopeHeadDim cfg.qkReservedHeadDim cfg.valueHeadDim cfg.denseHiddenDim
  | .mla _, .sparse _ =>
      Backbone.MLASparse.Params cfg.hiddenDim cfg.numHeads cfg.queryLatentDim cfg.kvLatentDim
        cfg.qkNopeHeadDim cfg.qkReservedHeadDim cfg.valueHeadDim cfg.routedLatentDim
        cfg.routedExpertHiddenDim cfg.numSharedExperts cfg.numRoutedExperts

/-- Causal state presented to one decoder layer before processing the current token. -/
abbrev StateShapes (pastTokens : Nat) : SequenceMixer ℝ cfg decayRank → List Shape
  | .kda _ _ =>
      [ [cfg.shortConvWidth, cfg.hiddenDim],
        [cfg.numHeads, cfg.kdaHeadDim, cfg.kdaValueDim] ]
  | .mla _ =>
      [ [pastTokens, cfg.kvLatentDim],
        [pastTokens, cfg.qkReservedHeadDim] ]

/-- Causal state returned after the current token has passed through one decoder layer. -/
abbrev NextStateShapes (pastTokens : Nat) : SequenceMixer ℝ cfg decayRank → List Shape
  | .kda _ _ =>
      [ [cfg.shortConvWidth, cfg.hiddenDim],
        [cfg.numHeads, cfg.kdaHeadDim, cfg.kdaValueDim] ]
  | .mla _ =>
      [ [pastTokens + 1, cfg.kvLatentDim],
        [pastTokens + 1, cfg.qkReservedHeadDim] ]

/-- Input shapes of the concrete layer graph.

KDA receives its fixed convolution window and recurrent matrix, and MLA its two growing
compressed caches. The remaining scalar inputs are the numerical constants that the component
graphs already take.
-/
abbrev Inputs (pastTokens completedCount : Nat)
    (sequence : SequenceMixer ℝ cfg decayRank) : List Shape :=
  match sequence with
  | .kda _ _ =>
      Backbone.DepthInputs completedCount cfg.hiddenDim ++
        [ [cfg.shortConvWidth, cfg.hiddenDim],
          [cfg.numHeads, cfg.kdaHeadDim, cfg.kdaValueDim],
          [], [], [], [] ]
  | .mla _ =>
      Backbone.DepthInputs completedCount cfg.hiddenDim ++
        [ [pastTokens, cfg.kvLatentDim],
          [pastTokens, cfg.qkReservedHeadDim],
          [], [], [] ]

/-- Output shapes of the concrete layer graph: updated causal state, both residual contributions,
and the updated partial AttnRes block. -/
abbrev Outputs (pastTokens : Nat) (sequence : SequenceMixer ℝ cfg decayRank) : List Shape :=
  match sequence with
  | .kda _ _ =>
      [ [cfg.shortConvWidth, cfg.hiddenDim],
        [cfg.numHeads, cfg.kdaHeadDim, cfg.kdaValueDim],
        [cfg.hiddenDim], [cfg.hiddenDim], [cfg.hiddenDim] ]
  | .mla _ =>
      [ [pastTokens + 1, cfg.kvLatentDim],
        [pastTokens + 1, cfg.qkReservedHeadDim],
        [cfg.hiddenDim], [cfg.hiddenDim], [cfg.hiddenDim] ]

/-- Numerical scalar consumed by a sequence-mixing graph. -/
def controlValue (sequence : SequenceMixer ℝ cfg decayRank) (mlaScoreScale : ℝ) : ℝ :=
  match sequence with
  | .kda _ logFloor => logFloor
  | .mla _ => mlaScoreScale

/-- Select the GraphSpec implementation (one of the four backbone graphs) for a decoder layer. -/
def model (pastTokens completedCount : Nat) (layer : BackboneLayer ℝ cfg decayRank)
    (route : Route cfg.numRoutedExperts cfg.activeExperts) (partialPresent : Bool)
    (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hHeads : 0 < cfg.numHeads) (hKey : 0 < cfg.kdaHeadDim)
    (hValue : 0 < cfg.kdaValueDim) (hWidth : 0 < cfg.shortConvWidth)
    (hQueryLatent : 0 < cfg.queryLatentDim) (hKVLatent : 0 < cfg.kvLatentDim)
    (hRoutedLatent : 0 < cfg.routedLatentDim) :
    NN.GraphSpec.DAG.MultiModel (Params layer.sequence layer.channel)
      (Inputs pastTokens completedCount layer.sequence) (Outputs pastTokens layer.sequence) :=
  match layer.sequence, layer.channel with
  | .kda _ _, .dense _ =>
      Backbone.KDADense.model completedCount cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim
        cfg.kdaValueDim cfg.shortConvWidth decayRank cfg.denseHiddenDim partialPresent hCompleted
        hModel hHeads hKey hValue hWidth
  | .kda _ _, .sparse _ =>
      Backbone.KDASparse.model completedCount cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim
        cfg.kdaValueDim cfg.shortConvWidth decayRank cfg.routedLatentDim
        cfg.routedExpertHiddenDim cfg.numSharedExperts cfg.numRoutedExperts cfg.activeExperts route
        partialPresent hCompleted hModel hHeads hKey hValue hWidth hRoutedLatent
  | .mla _, .dense _ =>
      Backbone.MLADense.model pastTokens completedCount cfg.hiddenDim cfg.numHeads
        cfg.queryLatentDim cfg.kvLatentDim cfg.qkNopeHeadDim cfg.qkReservedHeadDim
        cfg.valueHeadDim cfg.denseHiddenDim partialPresent hCompleted hModel hQueryLatent
        hKVLatent
  | .mla _, .sparse _ =>
      Backbone.MLASparse.model pastTokens completedCount cfg.hiddenDim cfg.numHeads
        cfg.queryLatentDim cfg.kvLatentDim cfg.qkNopeHeadDim cfg.qkReservedHeadDim
        cfg.valueHeadDim cfg.routedLatentDim cfg.routedExpertHiddenDim cfg.numSharedExperts
        cfg.numRoutedExperts cfg.activeExperts route partialPresent hCompleted hModel hQueryLatent
        hKVLatent hRoutedLatent

/-- Extract a concrete layer's parameters in the ABI selected by `model`. -/
def parameters (layer : BackboneLayer ℝ cfg decayRank) :
    TorchLean.TensorPack ℝ (Params layer.sequence layer.channel) :=
  match layer.sequence, layer.channel with
  | .kda kda _, .dense expert => Backbone.KDADense.parameters layer kda expert
  | .kda kda _, .sparse moe => Backbone.KDASparse.parameters layer kda moe
  | .mla mla, .dense expert => Backbone.MLADense.parameters layer mla expert
  | .mla mla, .sparse moe => Backbone.MLASparse.parameters layer mla moe

/-- Package a layer's depth state, causal state, and scalar constants for its selected graph. -/
noncomputable def inputs (pastTokens : Nat) (sequence : SequenceMixer ℝ cfg decayRank)
    {completedCount : Nat}
    (completed : Tensor ℝ [completedCount, cfg.hiddenDim])
    (partialState : Tensor ℝ [cfg.hiddenDim])
    (state : TorchLean.TensorPack ℝ (StateShapes pastTokens sequence))
    (scoreScale : ℝ) : TorchLean.TensorPack ℝ (Inputs pastTokens completedCount sequence) :=
  match sequence with
  | .kda _ logFloor =>
      match state with
      | .cons previousWindow (.cons previousState .nil) =>
          Backbone.KDADense.inputs completed partialState previousWindow previousState logFloor
            Normalize.l2Epsilon cfg.situGateCap cfg.situUpCap
  | .mla _ =>
      match state with
      | .cons pastLatentCache (.cons pastSharedKeyCache .nil) =>
          Backbone.MLADense.inputs completed partialState pastLatentCache pastSharedKeyCache
            scoreScale cfg.situGateCap cfg.situUpCap

/-- Construct the inputs of a concrete layer inside a larger graph.

`sequenceControl` is the recurrent log floor for KDA and the score multiplier for MLA. Epsilon
(used only by KDA) and the two SiTU caps are graph values, so compilation cannot quietly replace a
numerical convention chosen by the model or checkpoint.
-/
def inputTerms {Γ : List Shape} (pastTokens : Nat)
    (sequence : SequenceMixer ℝ cfg decayRank) {completedCount : Nat}
    (completed : Term Γ [completedCount, cfg.hiddenDim])
    (partialState : Term Γ [cfg.hiddenDim])
    (state : Args Γ (StateShapes pastTokens sequence))
    (sequenceControl epsilon gateCap upCap : Term Γ []) :
    Args Γ (Inputs pastTokens completedCount sequence) :=
  match sequence with
  | .kda _ _ =>
      match state with
      | .cons previousWindow (.cons previousState .nil) =>
          .cons completed <| .cons partialState <| .cons previousWindow <|
            .cons previousState <| .cons sequenceControl <| .cons epsilon <|
              .cons gateCap <| .cons upCap .nil
  | .mla _ =>
      match state with
      | .cons pastLatentCache (.cons pastSharedKeyCache .nil) =>
          .cons completed <| .cons partialState <| .cons pastLatentCache <|
            .cons pastSharedKeyCache <| .cons sequenceControl <| .cons gateCap <|
              .cons upCap .nil

/-- Select the updated causal state from a layer graph's result list. -/
def nextStateTerms {Γ : List Shape} (pastTokens : Nat) :
    (sequence : SequenceMixer ℝ cfg decayRank) →
      Args Γ (Outputs pastTokens sequence) → Args Γ (NextStateShapes pastTokens sequence)
  | .kda _ _, .cons nextWindow (.cons nextState
      (.cons _sequenceOutput (.cons _channelOutput (.cons _nextPartial .nil)))) =>
      .cons nextWindow (.cons nextState .nil)
  | .mla _, .cons nextLatentCache (.cons nextSharedKeyCache
      (.cons _sequenceOutput (.cons _channelOutput (.cons _nextPartial .nil)))) =>
      .cons nextLatentCache (.cons nextSharedKeyCache .nil)

/-- Select the updated partial AttnRes sum from a layer graph's result list. -/
def nextPartialTerm {Γ : List Shape} (pastTokens : Nat) :
    (sequence : SequenceMixer ℝ cfg decayRank) →
      Args Γ (Outputs pastTokens sequence) → Term Γ [cfg.hiddenDim]
  | .kda _ _, .cons _nextWindow (.cons _nextState
      (.cons _sequenceOutput (.cons _channelOutput (.cons nextPartial .nil)))) =>
      nextPartial
  | .mla _, .cons _nextLatentCache (.cons _nextSharedKeyCache
      (.cons _sequenceOutput (.cons _channelOutput (.cons nextPartial .nil)))) =>
      nextPartial

/-- Select the updated causal state from the pure result of a layer specification. -/
def nextStateValues (pastTokens : Nat) :
    (sequence : SequenceMixer ℝ cfg decayRank) →
      TorchLean.TensorPack ℝ (Outputs pastTokens sequence) →
        TorchLean.TensorPack ℝ (NextStateShapes pastTokens sequence)
  | .kda _ _, .cons nextWindow (.cons nextState
      (.cons _sequenceOutput (.cons _channelOutput (.cons _nextPartial .nil)))) =>
      .cons nextWindow (.cons nextState .nil)
  | .mla _, .cons nextLatentCache (.cons nextSharedKeyCache
      (.cons _sequenceOutput (.cons _channelOutput (.cons _nextPartial .nil)))) =>
      .cons nextLatentCache (.cons nextSharedKeyCache .nil)

/-- Select the updated partial AttnRes sum from the pure result of a layer specification. -/
def nextPartialValue (pastTokens : Nat) :
    (sequence : SequenceMixer ℝ cfg decayRank) →
      TorchLean.TensorPack ℝ (Outputs pastTokens sequence) → Tensor ℝ [cfg.hiddenDim]
  | .kda _ _, .cons _nextWindow (.cons _nextState
      (.cons _sequenceOutput (.cons _channelOutput (.cons nextPartial .nil)))) =>
      nextPartial
  | .mla _, .cons _nextLatentCache (.cons _nextSharedKeyCache
      (.cons _sequenceOutput (.cons _channelOutput (.cons nextPartial .nil)))) =>
      nextPartial

/-- Evaluating the graph-level state projection agrees with projecting the evaluated layer
result. A composed decoder can then keep each layer's own kind of cache without reopening the
implementation of KDA or MLA. -/
theorem evalArgs_nextStateTerms {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ) (pastTokens : Nat)
    (sequence : SequenceMixer ℝ cfg decayRank)
    (results : Args Γ (Outputs pastTokens sequence)) :
    Term.evalArgs env (nextStateTerms pastTokens sequence results) =
      nextStateValues pastTokens sequence (Term.evalArgs env results) := by
  cases sequence with
  | kda kda logFloor =>
      cases results with
      | cons nextWindow rest =>
          cases rest with
          | cons nextState rest =>
              cases rest with
              | cons sequenceOutput rest =>
                  cases rest with
                  | cons channelOutput rest =>
                      cases rest with
                      | cons nextPartial rest => cases rest; rfl
  | mla mla =>
      cases results with
      | cons nextLatentCache rest =>
          cases rest with
          | cons nextSharedKeyCache rest =>
              cases rest with
              | cons sequenceOutput rest =>
                  cases rest with
                  | cons channelOutput rest =>
                      cases rest with
                      | cons nextPartial rest => cases rest; rfl

/-- Evaluating the graph-level AttnRes projection returns the partial sum selected from the
mathematical layer result. -/
theorem eval_nextPartialTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ) (pastTokens : Nat)
    (sequence : SequenceMixer ℝ cfg decayRank)
    (results : Args Γ (Outputs pastTokens sequence)) :
    Term.eval env (nextPartialTerm pastTokens sequence results) =
      nextPartialValue pastTokens sequence (Term.evalArgs env results) := by
  cases sequence with
  | kda kda logFloor =>
      cases results with
      | cons nextWindow rest =>
          cases rest with
          | cons nextState rest =>
              cases rest with
              | cons sequenceOutput rest =>
                  cases rest with
                  | cons channelOutput rest =>
                      cases rest with
                      | cons nextPartial rest => cases rest; rfl
  | mla mla =>
      cases results with
      | cons nextLatentCache rest =>
          cases rest with
          | cons nextSharedKeyCache rest =>
              cases rest with
              | cons sequenceOutput rest =>
                  cases rest with
                  | cons channelOutput rest =>
                      cases rest with
                      | cons nextPartial rest => cases rest; rfl

/-- When the control terms evaluate to the model's constants, `inputTerms` evaluates to `inputs`.
KDA takes its recurrent log floor from the layer and MLA takes the supplied score scale. -/
theorem evalArgs_inputTerms {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ) (pastTokens : Nat)
    (sequence : SequenceMixer ℝ cfg decayRank) {completedCount : Nat}
    (completed : Term Γ [completedCount, cfg.hiddenDim])
    (partialState : Term Γ [cfg.hiddenDim])
    (state : Args Γ (StateShapes pastTokens sequence))
    (control epsilon gateCap upCap : Term Γ []) (scoreScale : ℝ)
    (hControl : Term.eval env control = Tensor.scalar (controlValue sequence scoreScale))
    (hEpsilon : Term.eval env epsilon = Tensor.scalar Normalize.l2Epsilon)
    (hGateCap : Term.eval env gateCap = Tensor.scalar (cfg.situGateCap : ℝ))
    (hUpCap : Term.eval env upCap = Tensor.scalar (cfg.situUpCap : ℝ)) :
    Term.evalArgs env
        (inputTerms pastTokens sequence completed partialState state control epsilon gateCap
          upCap) =
      inputs pastTokens sequence (Term.eval env completed) (Term.eval env partialState)
        (Term.evalArgs env state) scoreScale := by
  cases sequence with
  | kda kda logFloor =>
      cases state with
      | cons previousWindow rest =>
          cases rest with
          | cons previousState rest =>
              cases rest
              simp only [controlValue] at hControl
              simp only [inputTerms, inputs, Term.evalArgs,
                Backbone.KDADense.inputs, Backbone.depthInputs,
                TorchLean.TensorPack.append, hControl, hEpsilon, hGateCap, hUpCap]
  | mla mla =>
      cases state with
      | cons pastLatentCache rest =>
          cases rest with
          | cons pastSharedKeyCache rest =>
              cases rest
              simp only [controlValue] at hControl
              simp only [inputTerms, inputs, Term.evalArgs,
                Backbone.MLADense.inputs, Backbone.depthInputs,
                TorchLean.TensorPack.append, hControl, hGateCap, hUpCap]

/-- Mathematical transition denoted by the graph selected for `layer`.

It only dispatches to the four branch equations in `Backbone`, which carry the substance. The
common result type lets the decoder recurse while keeping track of which causal state each layer
owns. The latent-width hypotheses mirror the graph-construction interface, although the branch
equations do not use them.
-/
noncomputable def specStep (pastTokens : Nat) (layer : BackboneLayer ℝ cfg decayRank)
    {completedCount : Nat} (route : Route cfg.numRoutedExperts cfg.activeExperts)
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hWidth : 0 < cfg.shortConvWidth) (_hQueryLatent : 0 < cfg.queryLatentDim)
    (_hKVLatent : 0 < cfg.kvLatentDim) (_hRoutedLatent : 0 < cfg.routedLatentDim)
    (completed : Tensor ℝ [completedCount, cfg.hiddenDim])
    (partialState : Tensor ℝ [cfg.hiddenDim])
    (state : TorchLean.TensorPack ℝ (StateShapes pastTokens layer.sequence)) (scoreScale : ℝ) :
    TorchLean.TensorPack ℝ (Outputs pastTokens layer.sequence) :=
  match layer with
  | ⟨sequenceQuery, sequenceNormScale, sequence, channelQuery, channelNormScale, channel⟩ =>
    match sequence with
    | .kda kda logFloor =>
      match channel with
      | .dense expert =>
          match state with
          | .cons previousWindow (.cons previousState .nil) =>
              Backbone.KDADense.specStep partialPresent hCompleted hModel hWidth kda expert
                sequenceQuery sequenceNormScale channelQuery channelNormScale completed partialState
                previousWindow previousState logFloor
                cfg.situGateCap cfg.situUpCap
      | .sparse moe =>
          match state with
          | .cons previousWindow (.cons previousState .nil) =>
              Backbone.KDASparse.specStep partialPresent hCompleted hModel hWidth kda moe route
                sequenceQuery sequenceNormScale channelQuery channelNormScale completed partialState
                previousWindow previousState logFloor
                cfg.situGateCap cfg.situUpCap
    | .mla mla =>
      match channel with
      | .dense expert =>
          match state with
          | .cons pastLatentCache (.cons pastSharedKeyCache .nil) =>
              Backbone.MLADense.specStep partialPresent hCompleted hModel mla expert sequenceQuery
                sequenceNormScale channelQuery channelNormScale completed
                partialState pastLatentCache pastSharedKeyCache
                scoreScale cfg.situGateCap cfg.situUpCap
      | .sparse moe =>
          match state with
          | .cons pastLatentCache (.cons pastSharedKeyCache .nil) =>
              Backbone.MLASparse.specStep partialPresent hCompleted hModel mla moe route
                sequenceQuery sequenceNormScale channelQuery channelNormScale completed
                partialState pastLatentCache pastSharedKeyCache
                scoreScale cfg.situGateCap cfg.situUpCap

/-- Expert route determined by the channel input at this decoder layer.

Sparse layers apply K3's deterministic top-k rule to the bias-adjusted router scores. Dense layers
never read a route, so they get the zero-score route only to fill the shared layer interface. The
latent-width hypotheses are there to match `specStep` and `RouteAgrees`, and the route calculation
does not use them.
-/
noncomputable def expectedRoute (pastTokens : Nat)
    (layer : BackboneLayer ℝ cfg decayRank) {completedCount : Nat}
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hWidth : 0 < cfg.shortConvWidth) (_hQueryLatent : 0 < cfg.queryLatentDim)
    (_hKVLatent : 0 < cfg.kvLatentDim)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (completed : Tensor ℝ [completedCount, cfg.hiddenDim])
    (partialState : Tensor ℝ [cfg.hiddenDim])
    (state : TorchLean.TensorPack ℝ (StateShapes pastTokens layer.sequence)) (scoreScale : ℝ) :
    Route cfg.numRoutedExperts cfg.activeExperts :=
  match _hSequence : layer.sequence, _hChannel : layer.channel, state with
  | .kda _ _, .dense _, _ =>
      Route.chooseTopK (fun _ => 0) hActive
  | .kda kda logFloor, .sparse moe,
      .cons previousWindow (.cons previousState .nil) =>
      let sequenceInput := PackedDepth.sequenceInput partialPresent hCompleted hModel
        layer.sequenceQuery layer.sequenceNormScale completed partialState
      let nextWindow := KDA.Layer.rollWindow hWidth sequenceInput previousWindow
      let sequenceResult := kda.stepWindow hWidth logFloor nextWindow previousState
      let channelInput := PackedDepth.channelInput hModel layer.channelQuery
        layer.channelNormScale completed partialState sequenceResult.2
      moe.route hActive channelInput
  | .mla _, .dense _, _ =>
      Route.chooseTopK (fun _ => 0) hActive
  | .mla mla, .sparse moe,
      .cons pastLatentCache (.cons pastSharedKeyCache .nil) =>
      let sequenceInput := PackedDepth.sequenceInput partialPresent hCompleted hModel
        layer.sequenceQuery layer.sequenceNormScale completed partialState
      let sequenceResult := mla.stepFixed pastTokens pastLatentCache
        pastSharedKeyCache sequenceInput scoreScale
      let channelInput := PackedDepth.channelInput hModel layer.channelQuery
        layer.channelNormScale completed partialState sequenceResult.2.2
      moe.route hActive channelInput

/-- Whether a route staged into the numerical graph agrees with K3's router at this layer.

It holds trivially for a dense channel mixer. For a sparse mixer it is equality with
`expectedRoute`, the deterministic route computed from the actual intermediate channel input
(router bias and the lower-index tie break of `Route.chooseTopK` included).
-/
def RouteAgrees (pastTokens : Nat) (layer : BackboneLayer ℝ cfg decayRank)
    {completedCount : Nat} (route : Route cfg.numRoutedExperts cfg.activeExperts)
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hWidth : 0 < cfg.shortConvWidth) (hQueryLatent : 0 < cfg.queryLatentDim)
    (hKVLatent : 0 < cfg.kvLatentDim)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (completed : Tensor ℝ [completedCount, cfg.hiddenDim])
    (partialState : Tensor ℝ [cfg.hiddenDim])
    (state : TorchLean.TensorPack ℝ (StateShapes pastTokens layer.sequence))
    (scoreScale : ℝ) : Prop :=
  match layer.channel with
  | .dense _ => True
  | .sparse _ =>
      route = expectedRoute pastTokens layer partialPresent hCompleted hModel hWidth hQueryLatent
        hKVLatent hActive completed partialState state scoreScale

/-- Decoder-layer semantics with sparse routes computed from the current hidden representation. -/
noncomputable def specStepAuto (pastTokens : Nat) (layer : BackboneLayer ℝ cfg decayRank)
    {completedCount : Nat} (partialPresent : Bool) (hCompleted : 0 < completedCount)
    (hModel : 0 < cfg.hiddenDim) (hWidth : 0 < cfg.shortConvWidth)
    (hQueryLatent : 0 < cfg.queryLatentDim) (hKVLatent : 0 < cfg.kvLatentDim)
    (hRoutedLatent : 0 < cfg.routedLatentDim)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (completed : Tensor ℝ [completedCount, cfg.hiddenDim])
    (partialState : Tensor ℝ [cfg.hiddenDim])
    (state : TorchLean.TensorPack ℝ (StateShapes pastTokens layer.sequence)) (scoreScale : ℝ) :
    TorchLean.TensorPack ℝ (Outputs pastTokens layer.sequence) :=
  specStep pastTokens layer
    (expectedRoute pastTokens layer partialPresent hCompleted hModel hWidth hQueryLatent
      hKVLatent hActive completed partialState state scoreScale)
    partialPresent hCompleted hModel hWidth hQueryLatent hKVLatent hRoutedLatent completed
    partialState state scoreScale

/-- If the staged route passes `RouteAgrees`, `specStep` equals the automatically routed
`specStepAuto`. -/
theorem specStep_eq_specStepAuto_of_routeAgrees
    (pastTokens : Nat) (layer : BackboneLayer ℝ cfg decayRank)
    {completedCount : Nat} (route : Route cfg.numRoutedExperts cfg.activeExperts)
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hWidth : 0 < cfg.shortConvWidth) (hQueryLatent : 0 < cfg.queryLatentDim)
    (hKVLatent : 0 < cfg.kvLatentDim) (hRoutedLatent : 0 < cfg.routedLatentDim)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (completed : Tensor ℝ [completedCount, cfg.hiddenDim])
    (partialState : Tensor ℝ [cfg.hiddenDim])
    (state : TorchLean.TensorPack ℝ (StateShapes pastTokens layer.sequence)) (scoreScale : ℝ)
    (hRoute : RouteAgrees pastTokens layer route partialPresent hCompleted hModel hWidth
      hQueryLatent hKVLatent hActive completed partialState state scoreScale) :
    specStep pastTokens layer route partialPresent hCompleted hModel hWidth hQueryLatent
        hKVLatent hRoutedLatent completed partialState state scoreScale =
      specStepAuto pastTokens layer partialPresent hCompleted hModel hWidth hQueryLatent
        hKVLatent hRoutedLatent hActive completed partialState state scoreScale := by
  rcases layer with ⟨sequenceQuery, sequenceNormScale, sequence, channelQuery,
    channelNormScale, channel⟩
  cases sequence with
  | kda kda logFloor =>
      cases channel with
      | dense expert =>
          cases state with
          | cons previousWindow rest =>
              cases rest with
              | cons previousState rest => cases rest; rfl
      | sparse moe =>
          cases state with
          | cons previousWindow rest =>
              cases rest with
              | cons previousState rest =>
                  cases rest
                  simp only [RouteAgrees] at hRoute
                  subst route
                  rfl
  | mla mla =>
      cases channel with
      | dense expert =>
          cases state with
          | cons pastLatentCache rest =>
              cases rest with
              | cons pastSharedKeyCache rest => cases rest; rfl
      | sparse moe =>
          cases state with
          | cons pastLatentCache rest =>
              cases rest with
              | cons pastSharedKeyCache rest =>
                  cases rest
                  simp only [RouteAgrees] at hRoute
                  subst route
                  rfl

/-- Every concrete decoder-layer graph computes `specStep` for its layer. -/
theorem model_specFwd_eq_specStep
    (pastTokens : Nat) (layer : BackboneLayer ℝ cfg decayRank)
    {completedCount : Nat} (route : Route cfg.numRoutedExperts cfg.activeExperts)
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hHeads : 0 < cfg.numHeads) (hKey : 0 < cfg.kdaHeadDim)
    (hValue : 0 < cfg.kdaValueDim) (hWidth : 0 < cfg.shortConvWidth)
    (hQueryLatent : 0 < cfg.queryLatentDim) (hKVLatent : 0 < cfg.kvLatentDim)
    (hRoutedLatent : 0 < cfg.routedLatentDim) (hActive : 0 < cfg.activeExperts)
    (completed : Tensor ℝ [completedCount, cfg.hiddenDim])
    (partialState : Tensor ℝ [cfg.hiddenDim])
    (state : TorchLean.TensorPack ℝ (StateShapes pastTokens layer.sequence)) (scoreScale : ℝ) :
    (model pastTokens completedCount layer route partialPresent hCompleted hModel hHeads hKey
      hValue hWidth hQueryLatent hKVLatent hRoutedLatent).specFwd (parameters layer)
        (inputs pastTokens layer.sequence completed partialState state scoreScale) =
      specStep pastTokens layer route partialPresent hCompleted hModel hWidth hQueryLatent
        hKVLatent hRoutedLatent completed partialState state scoreScale := by
  rcases layer with ⟨sequenceQuery, sequenceNormScale, sequence, channelQuery,
    channelNormScale, channel⟩
  change TorchLean.TensorPack ℝ (StateShapes pastTokens sequence) at state
  cases sequence with
  | kda kda logFloor =>
      cases channel with
      | dense expert =>
          cases state with
          | cons previousWindow rest =>
              cases rest with
              | cons previousState rest =>
                  cases rest
                  simpa only [model, parameters, inputs, specStep,
                    Backbone.KDADense.inputs, Backbone.KDASparse.inputs] using
                    Backbone.KDADense.model_specFwd_eq_specStep partialPresent hCompleted hModel
                      hHeads hKey hValue hWidth
                      (⟨sequenceQuery, sequenceNormScale, .kda kda logFloor, channelQuery,
                        channelNormScale, .dense expert⟩ : BackboneLayer ℝ cfg decayRank)
                      kda expert completed partialState
                      previousWindow previousState logFloor Normalize.l2Epsilon cfg.situGateCap
                      cfg.situUpCap rfl
      | sparse moe =>
          cases state with
          | cons previousWindow rest =>
              cases rest with
              | cons previousState rest =>
                  cases rest
                  simpa only [model, parameters, inputs, specStep,
                    Backbone.KDADense.inputs, Backbone.KDASparse.inputs] using
                    Backbone.KDASparse.model_specFwd_eq_specStep partialPresent hCompleted hModel
                      hHeads hKey hValue hWidth hRoutedLatent hActive
                      (⟨sequenceQuery, sequenceNormScale, .kda kda logFloor, channelQuery,
                        channelNormScale, .sparse moe⟩ : BackboneLayer ℝ cfg decayRank)
                      kda moe route
                      completed partialState previousWindow previousState logFloor
                      Normalize.l2Epsilon
                      cfg.situGateCap cfg.situUpCap rfl
  | mla mla =>
      cases channel with
      | dense expert =>
          cases state with
          | cons pastLatentCache rest =>
              cases rest with
              | cons pastSharedKeyCache rest =>
                  cases rest
                  simpa only [model, parameters, inputs, specStep,
                    Backbone.MLADense.inputs, Backbone.MLASparse.inputs] using
                    Backbone.MLADense.model_specFwd_eq_specStep partialPresent hCompleted hModel
                      hQueryLatent hKVLatent
                      (⟨sequenceQuery, sequenceNormScale, .mla mla, channelQuery,
                        channelNormScale, .dense expert⟩ : BackboneLayer ℝ cfg decayRank)
                      mla expert completed partialState pastLatentCache
                      pastSharedKeyCache scoreScale cfg.situGateCap cfg.situUpCap
      | sparse moe =>
          cases state with
          | cons pastLatentCache rest =>
              cases rest with
              | cons pastSharedKeyCache rest =>
                  cases rest
                  simpa only [model, parameters, inputs, specStep,
                    Backbone.MLADense.inputs, Backbone.MLASparse.inputs] using
                    Backbone.MLASparse.model_specFwd_eq_specStep partialPresent hCompleted hModel
                      hQueryLatent hKVLatent hRoutedLatent hActive
                      (⟨sequenceQuery, sequenceNormScale, .mla mla, channelQuery,
                        channelNormScale, .sparse moe⟩ : BackboneLayer ℝ cfg decayRank)
                      mla moe route completed
                      partialState pastLatentCache pastSharedKeyCache scoreScale cfg.situGateCap
                      cfg.situUpCap

end DecoderLayer

end GraphSpec
end KimiK3
