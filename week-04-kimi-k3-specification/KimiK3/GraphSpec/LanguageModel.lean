/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Backbone

/-!
# Kimi K3 language-model graph

This file assembles the individually verified decoder graphs into the typed graph of a complete
language model.  It deliberately reuses `KimiK3.LanguageModel` and `KimiK3.BackboneLayer`: the
GraphSpec layer is a lowering of those mathematical objects, not a second architecture hierarchy.

The first step is to give every concrete backbone layer one dependent graph ABI.  Its parameter
and state shapes are computed from the existing `SequenceMixer` and `ChannelMixer` constructors.
Pattern matching then selects one of the four refinement theorems proved in `Backbone.lean`.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec
namespace DecoderLayer

open Spec
open Spec.Tensor
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
      [ .dim cfg.shortConvWidth (.dim cfg.hiddenDim .scalar),
        .dim cfg.numHeads (.dim cfg.kdaHeadDim (.dim cfg.kdaValueDim .scalar)) ]
  | .mla _ =>
      [ .dim pastTokens (.dim cfg.kvLatentDim .scalar),
        .dim pastTokens (.dim cfg.qkReservedHeadDim .scalar) ]

/-- Causal state returned after the current token has passed through one decoder layer. -/
abbrev NextStateShapes (pastTokens : Nat) : SequenceMixer ℝ cfg decayRank → List Shape
  | .kda _ _ =>
      [ .dim cfg.shortConvWidth (.dim cfg.hiddenDim .scalar),
        .dim cfg.numHeads (.dim cfg.kdaHeadDim (.dim cfg.kdaValueDim .scalar)) ]
  | .mla _ =>
      [ .dim (pastTokens + 1) (.dim cfg.kvLatentDim .scalar),
        .dim (pastTokens + 1) (.dim cfg.qkReservedHeadDim .scalar) ]

/-- Input shapes of the concrete layer graph.

KDA receives its fixed convolution window and recurrent matrix.  MLA receives its two growing
compressed caches.  The remaining scalar inputs are the numerical constants already exposed by
the component graphs.
-/
abbrev Inputs (pastTokens completedCount : Nat)
    (sequence : SequenceMixer ℝ cfg decayRank) : List Shape :=
  match sequence with
  | .kda _ _ =>
      Backbone.DepthInputs completedCount cfg.hiddenDim ++
        [ .dim cfg.shortConvWidth (.dim cfg.hiddenDim .scalar),
          .dim cfg.numHeads (.dim cfg.kdaHeadDim (.dim cfg.kdaValueDim .scalar)),
          .scalar, .scalar, .scalar, .scalar ]
  | .mla _ =>
      Backbone.DepthInputs completedCount cfg.hiddenDim ++
        [ .dim pastTokens (.dim cfg.kvLatentDim .scalar),
          .dim pastTokens (.dim cfg.qkReservedHeadDim .scalar),
          .scalar, .scalar, .scalar ]

/-- Output shapes of the concrete layer graph: updated causal state, both residual contributions,
and the updated partial AttnRes block. -/
abbrev Outputs (pastTokens : Nat) (sequence : SequenceMixer ℝ cfg decayRank) : List Shape :=
  match sequence with
  | .kda _ _ =>
      [ .dim cfg.shortConvWidth (.dim cfg.hiddenDim .scalar),
        .dim cfg.numHeads (.dim cfg.kdaHeadDim (.dim cfg.kdaValueDim .scalar)),
        .dim cfg.hiddenDim .scalar, .dim cfg.hiddenDim .scalar, .dim cfg.hiddenDim .scalar ]
  | .mla _ =>
      [ .dim (pastTokens + 1) (.dim cfg.kvLatentDim .scalar),
        .dim (pastTokens + 1) (.dim cfg.qkReservedHeadDim .scalar),
        .dim cfg.hiddenDim .scalar, .dim cfg.hiddenDim .scalar, .dim cfg.hiddenDim .scalar ]

/-- Numerical scalar consumed by a sequence-mixing graph. -/
def controlValue (sequence : SequenceMixer ℝ cfg decayRank) (mlaScoreScale : ℝ) : ℝ :=
  match sequence with
  | .kda _ logFloor => logFloor
  | .mla _ => mlaScoreScale

/-- Select the already verified GraphSpec implementation of a concrete decoder layer. -/
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

/-- Extract a concrete layer's parameters in precisely the ABI selected by `model`. -/
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
    (completed : Tensor ℝ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Tensor ℝ (.dim cfg.hiddenDim .scalar))
    (state : TorchLean.TensorPack ℝ (StateShapes pastTokens sequence))
    (scoreScale : ℝ) : TorchLean.TensorPack ℝ (Inputs pastTokens completedCount sequence) :=
  match sequence with
  | .kda _ logFloor =>
      match state with
      | .cons previousWindow (.cons previousState .nil) =>
          Backbone.KDADense.inputs completed partialState previousWindow previousState logFloor
            Numbers.epsilon cfg.situGateCap cfg.situUpCap
  | .mla _ =>
      match state with
      | .cons pastLatentCache (.cons pastSharedKeyCache .nil) =>
          Backbone.MLADense.inputs completed partialState pastLatentCache pastSharedKeyCache
            scoreScale cfg.situGateCap cfg.situUpCap

/-- Construct the inputs of a concrete layer inside a larger graph.

`sequenceControl` is the recurrent log floor for KDA and the score multiplier for MLA. Epsilon and
the two SiTU caps remain explicit graph values, so compilation cannot silently replace a numerical
convention chosen by the model or checkpoint.
-/
def inputTerms {Γ : List Shape} (pastTokens : Nat)
    (sequence : SequenceMixer ℝ cfg decayRank) {completedCount : Nat}
    (completed : Term Γ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Term Γ (.dim cfg.hiddenDim .scalar))
    (state : Args Γ (StateShapes pastTokens sequence))
    (sequenceControl epsilon gateCap upCap : Term Γ .scalar) :
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
      Args Γ (Outputs pastTokens sequence) → Term Γ (.dim cfg.hiddenDim .scalar)
  | .kda _ _, .cons _nextWindow (.cons _nextState
      (.cons _sequenceOutput (.cons _channelOutput (.cons nextPartial .nil)))) =>
      nextPartial
  | .mla _, .cons _nextLatentCache (.cons _nextSharedKeyCache
      (.cons _sequenceOutput (.cons _channelOutput (.cons nextPartial .nil)))) =>
      nextPartial

/-- Select the updated causal state from the pure result of a layer specification. -/
def nextStateValues (pastTokens : Nat) :
    (sequence : SequenceMixer ℝ cfg decayRank) →
      TorchLean.TensorPack ℝ (Outputs pastTokens sequence) → TorchLean.TensorPack ℝ (NextStateShapes pastTokens sequence)
  | .kda _ _, .cons nextWindow (.cons nextState
      (.cons _sequenceOutput (.cons _channelOutput (.cons _nextPartial .nil)))) =>
      .cons nextWindow (.cons nextState .nil)
  | .mla _, .cons nextLatentCache (.cons nextSharedKeyCache
      (.cons _sequenceOutput (.cons _channelOutput (.cons _nextPartial .nil)))) =>
      .cons nextLatentCache (.cons nextSharedKeyCache .nil)

/-- Select the updated partial AttnRes sum from the pure result of a layer specification. -/
def nextPartialValue (pastTokens : Nat) :
    (sequence : SequenceMixer ℝ cfg decayRank) →
      TorchLean.TensorPack ℝ (Outputs pastTokens sequence) → Tensor ℝ (.dim cfg.hiddenDim .scalar)
  | .kda _ _, .cons _nextWindow (.cons _nextState
      (.cons _sequenceOutput (.cons _channelOutput (.cons nextPartial .nil)))) =>
      nextPartial
  | .mla _, .cons _nextLatentCache (.cons _nextSharedKeyCache
      (.cons _sequenceOutput (.cons _channelOutput (.cons nextPartial .nil)))) =>
      nextPartial

/-- Evaluating the graph-level state projection agrees with projecting the evaluated layer
result.  This lets a composed decoder retain each layer's heterogeneous cache without reopening
the implementation of KDA or MLA. -/
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

/-- The explicit graph controls denote the numerical constants used by the mathematical layer
specification.  KDA reads its recurrent log floor from the layer; MLA reads the supplied score
scale. -/
theorem evalArgs_inputTerms {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ) (pastTokens : Nat)
    (sequence : SequenceMixer ℝ cfg decayRank) {completedCount : Nat}
    (completed : Term Γ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Term Γ (.dim cfg.hiddenDim .scalar))
    (state : Args Γ (StateShapes pastTokens sequence))
    (control epsilon gateCap upCap : Term Γ .scalar) (scoreScale : ℝ)
    (hControl : Term.eval env control = .scalar (controlValue sequence scoreScale))
    (hEpsilon : Term.eval env epsilon = .scalar Numbers.epsilon)
    (hGateCap : Term.eval env gateCap = .scalar (cfg.situGateCap : ℝ))
    (hUpCap : Term.eval env upCap = .scalar (cfg.situUpCap : ℝ)) :
    Term.evalArgs env
        (inputTerms pastTokens sequence completed partialState state control epsilon gateCap upCap) =
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

This definition is intentionally only a dispatcher.  The four branch equations remain the
substantive specifications in `Backbone`; this common result type lets the complete decoder recurse
without erasing which causal state each layer owns.
-/
noncomputable def specStep (pastTokens : Nat) (layer : BackboneLayer ℝ cfg decayRank)
    {completedCount : Nat} (route : Route cfg.numRoutedExperts cfg.activeExperts)
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hWidth : 0 < cfg.shortConvWidth) (hQueryLatent : 0 < cfg.queryLatentDim)
    (hKVLatent : 0 < cfg.kvLatentDim) (_hRoutedLatent : 0 < cfg.routedLatentDim)
    (completed : Tensor ℝ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Tensor ℝ (.dim cfg.hiddenDim .scalar))
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
              Backbone.MLASparse.specStep partialPresent hCompleted hModel mla moe route sequenceQuery
                sequenceNormScale channelQuery channelNormScale completed
                partialState pastLatentCache pastSharedKeyCache
                scoreScale cfg.situGateCap cfg.situUpCap

/-- Expert route determined by the channel input encountered at this decoder layer.

Sparse layers apply K3's deterministic top-k rule to the bias-adjusted router scores. Dense layers
do not inspect a route; the canonical zero-score route merely supplies the otherwise unused argument
of the common layer interface. Thus this definition does not introduce routing into a dense layer.
-/
noncomputable def expectedRoute (pastTokens : Nat)
    (layer : BackboneLayer ℝ cfg decayRank) {completedCount : Nat}
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hWidth : 0 < cfg.shortConvWidth) (hQueryLatent : 0 < cfg.queryLatentDim)
    (hKVLatent : 0 < cfg.kvLatentDim)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (completed : Tensor ℝ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Tensor ℝ (.dim cfg.hiddenDim .scalar))
    (state : TorchLean.TensorPack ℝ (StateShapes pastTokens layer.sequence)) (scoreScale : ℝ) :
    Route cfg.numRoutedExperts cfg.activeExperts :=
  match _hSequence : layer.sequence, _hChannel : layer.channel, state with
  | .kda _ _, .dense _, _ =>
      Route.chooseTopK (fun _ => 0) hActive
  | .kda kda logFloor, .sparse moe,
      .cons previousWindow (.cons previousState .nil) =>
      let sequenceInput := PackedDepth.sequenceInput partialPresent hCompleted hModel
        layer.sequenceQuery layer.sequenceNormScale completed partialState
      let nextWindow := KDALayer.rollWindow hWidth sequenceInput previousWindow
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

The condition is vacuous for a dense channel mixer. For a sparse mixer it is equality with the
deterministic route computed from the actual intermediate channel input, including the router bias
and the lower-index tie break formalized by `Route.chooseTopK`.
-/
def RouteAgrees (pastTokens : Nat) (layer : BackboneLayer ℝ cfg decayRank)
    {completedCount : Nat} (route : Route cfg.numRoutedExperts cfg.activeExperts)
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hWidth : 0 < cfg.shortConvWidth) (hQueryLatent : 0 < cfg.queryLatentDim)
    (hKVLatent : 0 < cfg.kvLatentDim)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (completed : Tensor ℝ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Tensor ℝ (.dim cfg.hiddenDim .scalar))
    (state : TorchLean.TensorPack ℝ (StateShapes pastTokens layer.sequence)) (scoreScale : ℝ) : Prop :=
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
    (completed : Tensor ℝ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Tensor ℝ (.dim cfg.hiddenDim .scalar))
    (state : TorchLean.TensorPack ℝ (StateShapes pastTokens layer.sequence)) (scoreScale : ℝ) :
    TorchLean.TensorPack ℝ (Outputs pastTokens layer.sequence) :=
  specStep pastTokens layer
    (expectedRoute pastTokens layer partialPresent hCompleted hModel hWidth hQueryLatent
      hKVLatent hActive completed partialState state scoreScale)
    partialPresent hCompleted hModel hWidth hQueryLatent hKVLatent hRoutedLatent completed
    partialState state scoreScale

/-- A checked staged route gives exactly the automatically routed decoder-layer transition. -/
theorem specStep_eq_specStepAuto_of_routeAgrees
    (pastTokens : Nat) (layer : BackboneLayer ℝ cfg decayRank)
    {completedCount : Nat} (route : Route cfg.numRoutedExperts cfg.activeExperts)
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hWidth : 0 < cfg.shortConvWidth) (hQueryLatent : 0 < cfg.queryLatentDim)
    (hKVLatent : 0 < cfg.kvLatentDim) (hRoutedLatent : 0 < cfg.routedLatentDim)
    (hActive : cfg.activeExperts ≤ cfg.numRoutedExperts)
    (completed : Tensor ℝ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Tensor ℝ (.dim cfg.hiddenDim .scalar))
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

/-- Every concrete decoder-layer graph denotes the corresponding mathematical K3 transition. -/
theorem model_specFwd_eq_specStep
    (pastTokens : Nat) (layer : BackboneLayer ℝ cfg decayRank)
    {completedCount : Nat} (route : Route cfg.numRoutedExperts cfg.activeExperts)
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hHeads : 0 < cfg.numHeads) (hKey : 0 < cfg.kdaHeadDim)
    (hValue : 0 < cfg.kdaValueDim) (hWidth : 0 < cfg.shortConvWidth)
    (hQueryLatent : 0 < cfg.queryLatentDim) (hKVLatent : 0 < cfg.kvLatentDim)
    (hRoutedLatent : 0 < cfg.routedLatentDim) (hActive : 0 < cfg.activeExperts)
    (completed : Tensor ℝ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Tensor ℝ (.dim cfg.hiddenDim .scalar))
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
                      previousWindow previousState logFloor Numbers.epsilon cfg.situGateCap
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
                      completed partialState previousWindow previousState logFloor Numbers.epsilon
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

namespace DepthSchedule

open Spec
open Spec.Tensor
open NN.GraphSpec.DAG
open Runtime.Autograd.Torch

/-- Number of completed AttnRes sources after `processedLayers` decoder layers.

The token embedding is always the first source.  Every full block contributes one additional
source, while the current incomplete block is carried separately.
-/
def completedCount (blockSize processedLayers : Nat) : Nat :=
  1 + processedLayers / blockSize

/-- Whether the current AttnRes block already contains an earlier decoder layer. -/
def partialPresent (blockSize processedLayers : Nat) : Bool :=
  decide (processedLayers % blockSize ≠ 0)

/-- The packed depth state always contains at least the embedding row. -/
theorem completedCount_pos (blockSize processedLayers : Nat) :
    0 < completedCount blockSize processedLayers := by
  simp [completedCount]

/-- Completing a depth block increases the number of packed sources by one. -/
theorem completedCount_succ_of_boundary {blockSize processedLayers : Nat}
    (_hBlockSize : 0 < blockSize) (hBoundary : (processedLayers + 1) % blockSize = 0) :
    completedCount blockSize (processedLayers + 1) =
      completedCount blockSize processedLayers + 1 := by
  have hDvd : blockSize ∣ processedLayers + 1 :=
    Nat.dvd_iff_mod_eq_zero.mpr hBoundary
  simp only [completedCount, Nat.succ_div_of_dvd hDvd]
  omega

/-- Away from a depth-block boundary, the completed-source tensor keeps the same row count. -/
theorem completedCount_succ_of_not_boundary {blockSize processedLayers : Nat}
    (_hBlockSize : 0 < blockSize) (hBoundary : (processedLayers + 1) % blockSize ≠ 0) :
    completedCount blockSize (processedLayers + 1) =
      completedCount blockSize processedLayers := by
  have hNotDvd : ¬blockSize ∣ processedLayers + 1 := by
    intro hDvd
    exact hBoundary (Nat.mod_eq_zero_of_dvd hDvd)
  simp [completedCount, Nat.succ_div_of_not_dvd hNotDvd]

/-- Shapes of the packed AttnRes state after a fixed number of decoder layers. -/
abbrev Shapes (blockSize processedLayers modelDim : Nat) : List Shape :=
  [ .dim (completedCount blockSize processedLayers) (.dim modelDim .scalar),
    .dim modelDim .scalar ]

/-- Update the mathematical packed depth state after one decoder layer.

At a block boundary the accumulated residual becomes a new completed source and the partial sum is
reset.  At every other layer only the partial sum changes.  The result type records the new source
count, so an invalid boundary update cannot be passed to the next graph.
-/
def advance (blockSize processedLayers modelDim : Nat) (hBlockSize : 0 < blockSize)
    (completed : Tensor ℝ
      (.dim (completedCount blockSize processedLayers) (.dim modelDim .scalar)))
    (nextPartial : Tensor ℝ (.dim modelDim .scalar)) :
    TorchLean.TensorPack ℝ (Shapes blockSize (processedLayers + 1) modelDim) :=
  if hBoundary : (processedLayers + 1) % blockSize = 0 then
    let row : Tensor ℝ (.dim 1 (.dim modelDim .scalar)) :=
      Tensor.reshapeSpec nextPartial (by simp [Shape.size])
    let appended := Tensor.concatAxisSpec .scalar completed row
    have hCount := completedCount_succ_of_boundary hBlockSize hBoundary
    have hShape :
        Shape.dim (completedCount blockSize processedLayers + 1)
            (Shape.dim modelDim Shape.scalar) =
          Shape.dim (completedCount blockSize (processedLayers + 1))
            (Shape.dim modelDim Shape.scalar) :=
      congrArg (fun rows => Shape.dim rows (Shape.dim modelDim Shape.scalar)) hCount.symm
    let nextCompleted : Tensor ℝ
        (.dim (completedCount blockSize (processedLayers + 1)) (.dim modelDim .scalar)) :=
      Tensor.castShape appended hShape
    .cons nextCompleted <| .cons (Spec.fill 0 (.dim modelDim .scalar)) .nil
  else
    have hCount := completedCount_succ_of_not_boundary hBlockSize hBoundary
    have hShape :
        Shape.dim (completedCount blockSize processedLayers) (Shape.dim modelDim Shape.scalar) =
          Shape.dim (completedCount blockSize (processedLayers + 1))
            (Shape.dim modelDim Shape.scalar) :=
      congrArg (fun rows => Shape.dim rows (Shape.dim modelDim Shape.scalar)) hCount.symm
    let nextCompleted : Tensor ℝ
        (.dim (completedCount blockSize (processedLayers + 1)) (.dim modelDim .scalar)) :=
      Tensor.castShape completed hShape
    .cons nextCompleted (.cons nextPartial .nil)

/-- Graph block implementing `advance` from the current completed tensor and partial sum. -/
def advanceBlock {Γ : List Shape} (blockSize processedLayers modelDim : Nat)
    (hBlockSize : 0 < blockSize)
    (completed : Term Γ
      (.dim (completedCount blockSize processedLayers) (.dim modelDim .scalar)))
    (nextPartial : Term Γ (.dim modelDim .scalar)) :
    Block Γ (Shapes blockSize (processedLayers + 1) modelDim) :=
  if hBoundary : (processedLayers + 1) % blockSize = 0 then
    let row : Term Γ (.dim 1 (.dim modelDim .scalar)) :=
      Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
        (.cons nextPartial .nil)
    let appended : Term Γ
        (.dim (completedCount blockSize processedLayers + 1) (.dim modelDim .scalar)) :=
      GraphSpec.concatAxisZeroTerm (completedCount blockSize processedLayers) 1
        (.dim modelDim .scalar) completed row
    have hCount := completedCount_succ_of_boundary hBlockSize hBoundary
    let nextCompleted : Term Γ
        (.dim (completedCount blockSize (processedLayers + 1)) (.dim modelDim .scalar)) :=
      Term.cast appended
        (congrArg (fun rows => Shape.dim rows (Shape.dim modelDim Shape.scalar)) hCount.symm)
    let zeroPartial : Term Γ (.dim modelDim .scalar) :=
      Term.op (NN.GraphSpec.DAG.PrimOp.zero (.dim modelDim .scalar)) .nil
    .ret (.cons nextCompleted (.cons zeroPartial .nil))
  else
    have hCount := completedCount_succ_of_not_boundary hBlockSize hBoundary
    let nextCompleted : Term Γ
        (.dim (completedCount blockSize (processedLayers + 1)) (.dim modelDim .scalar)) :=
      Term.cast completed
        (congrArg (fun rows => Shape.dim rows (Shape.dim modelDim Shape.scalar)) hCount.symm)
    .ret (.cons nextCompleted (.cons nextPartial .nil))

/-- The depth-transition graph has exactly the mathematical AttnRes block semantics. -/
theorem eval_advanceBlock {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (blockSize processedLayers modelDim : Nat) (hBlockSize : 0 < blockSize)
    (completed : Term Γ
      (.dim (completedCount blockSize processedLayers) (.dim modelDim .scalar)))
    (nextPartial : Term Γ (.dim modelDim .scalar)) :
    Block.eval env
        (advanceBlock blockSize processedLayers modelDim hBlockSize completed nextPartial) =
      advance blockSize processedLayers modelDim hBlockSize (Term.eval env completed)
        (Term.eval env nextPartial) := by
  by_cases hBoundary : (processedLayers + 1) % blockSize = 0
  · rw [advanceBlock.eq_def, dif_pos hBoundary]
    simp only [Block.eval, Term.evalArgs, Term.eval_cast]
    rw [GraphSpec.eval_concatAxisZeroTerm]
    simp only [Term.eval_op, Term.evalArgs, NN.GraphSpec.DAG.PrimOp.reshape_specFwd,
      NN.GraphSpec.DAG.PrimOp.zero_specFwd]
    rw [advance.eq_def, dif_pos hBoundary]
    rw [Tensor.eqRec_eq_cast_shape]
  · rw [advanceBlock.eq_def, dif_neg hBoundary]
    simp only [Block.eval, Term.evalArgs, Term.eval_cast]
    rw [advance.eq_def, dif_neg hBoundary]
    rw [Tensor.eqRec_eq_cast_shape]

end DepthSchedule

namespace Decoder

open Spec
open NN.GraphSpec.DAG
open Runtime.Autograd.Torch

variable {cfg : TextConfig} {decayRank : Nat}

/-!
### Typed ABI of a decoder schedule

The complete decoder does not have a uniform cache at every layer.  A KDA layer owns a fixed
convolution window and recurrent matrix, whereas an MLA layer owns two caches whose leading
dimension grows with the context.  Flattening these values into an untyped array would discard the
distinction that the formalization is meant to preserve.  The functions below concatenate the
already established layer ABIs while retaining every shape in the type.
-/

/-- Parameter shapes of the selected layers, in execution order. -/
def ParamsFor (model : LanguageModel ℝ cfg decayRank) :
    List (Fin cfg.numLayers) → List Shape
  | [] => []
  | index :: rest =>
      DecoderLayer.Params (model.layer index).sequence (model.layer index).channel ++
        ParamsFor model rest

/-- Causal-state shapes before the selected layers process the current token. -/
def StateShapesFor (pastTokens : Nat) (model : LanguageModel ℝ cfg decayRank) :
    List (Fin cfg.numLayers) → List Shape
  | [] => []
  | index :: rest =>
      DecoderLayer.StateShapes pastTokens (model.layer index).sequence ++
        StateShapesFor pastTokens model rest

/-- Causal-state shapes after the selected layers process the current token. -/
def NextStateShapesFor (pastTokens : Nat) (model : LanguageModel ℝ cfg decayRank) :
    List (Fin cfg.numLayers) → List Shape
  | [] => []
  | index :: rest =>
      DecoderLayer.NextStateShapes pastTokens (model.layer index).sequence ++
        NextStateShapesFor pastTokens model rest

/-- One explicit numerical control for each sequence mixer. -/
def ControlShapesFor (model : LanguageModel ℝ cfg decayRank) :
    List (Fin cfg.numLayers) → List Shape
  | [] => []
  | _index :: rest => .scalar :: ControlShapesFor model rest

/-- Number of processed layers after executing a concrete schedule segment. -/
def processedAfter {Layer : Type} : Nat → List Layer → Nat
  | processedLayers, [] => processedLayers
  | processedLayers, _layer :: rest => processedAfter (processedLayers + 1) rest

/-- Executing a schedule advances the depth counter by its length. -/
theorem processedAfter_eq_add_length {Layer : Type} (processedLayers : Nat)
    (layers : List Layer) :
    processedAfter processedLayers layers = processedLayers + layers.length := by
  induction layers generalizing processedLayers with
  | nil => simp [processedAfter]
  | cons layer rest ih =>
      simp only [processedAfter, ih, List.length_cons]
      omega

/-- Inputs to a decoder segment: depth state, causal states, numerical controls, and shared scalars. -/
abbrev InputsFor (pastTokens processedLayers : Nat) (model : LanguageModel ℝ cfg decayRank)
    (indices : List (Fin cfg.numLayers)) : List Shape :=
  DepthSchedule.Shapes cfg.attnResBlockSize processedLayers cfg.hiddenDim ++
    (StateShapesFor pastTokens model indices ++
      (ControlShapesFor model indices ++ [.scalar, .scalar, .scalar]))

/-- Outputs of a decoder segment: updated causal states followed by the final packed depth state. -/
abbrev OutputsFor (pastTokens processedLayers : Nat) (model : LanguageModel ℝ cfg decayRank)
    (indices : List (Fin cfg.numLayers)) : List Shape :=
  NextStateShapesFor pastTokens model indices ++
    DepthSchedule.Shapes cfg.attnResBlockSize (processedAfter processedLayers indices) cfg.hiddenDim

/-- Extract the selected layers' parameters from the existing language-model value. -/
def parametersFor (model : LanguageModel ℝ cfg decayRank) :
    (indices : List (Fin cfg.numLayers)) → TorchLean.TensorPack ℝ (ParamsFor model indices)
  | [] => .nil
  | index :: rest =>
      TorchLean.TensorPack.append
        (DecoderLayer.parameters (model.layer index)) (parametersFor model rest)

/-- Pack one causal state per layer into the decoder's heterogeneous state ABI. -/
def packStates (pastTokens : Nat) (model : LanguageModel ℝ cfg decayRank)
    (state : ∀ index, TorchLean.TensorPack ℝ
      (DecoderLayer.StateShapes pastTokens (model.layer index).sequence)) :
    (indices : List (Fin cfg.numLayers)) → TorchLean.TensorPack ℝ (StateShapesFor pastTokens model indices)
  | [] => .nil
  | index :: rest =>
      TorchLean.TensorPack.append (state index)
        (packStates pastTokens model state rest)

/-- Empty fixed-shape cache corresponding to a concrete sequence mixer. -/
def emptyState (sequence : SequenceMixer ℝ cfg decayRank) :
    TorchLean.TensorPack ℝ (DecoderLayer.StateShapes 0 sequence) :=
  match sequence with
  | .kda _ _ =>
      .cons (Spec.fill 0 (.dim cfg.shortConvWidth (.dim cfg.hiddenDim .scalar))) <|
        .cons (Spec.fill 0
          (.dim cfg.numHeads (.dim cfg.kdaHeadDim (.dim cfg.kdaValueDim .scalar)))) .nil
  | .mla _ =>
      .cons (Spec.fill 0 (.dim 0 (.dim cfg.kvLatentDim .scalar))) <|
        .cons (Spec.fill 0 (.dim 0 (.dim cfg.qkReservedHeadDim .scalar))) .nil

/-- Empty causal states for a fresh decoder run. -/
def initialStatesFor (model : LanguageModel ℝ cfg decayRank) :
    (indices : List (Fin cfg.numLayers)) → TorchLean.TensorPack ℝ (StateShapesFor 0 model indices) :=
  packStates 0 model (fun index => emptyState (model.layer index).sequence)

/-- Numerical control consumed by one sequence mixer.

KDA stores its recurrent log floor in the model. MLA uses the externally recorded score scale,
normally `1 / sqrt(qkNopeHeadDim + qkReservedHeadDim)` for K3 checkpoints.
-/
def sequenceControl (layer : BackboneLayer ℝ cfg decayRank) (mlaScoreScale : ℝ) : ℝ :=
  match layer.sequence with
  | .kda _ logFloor => logFloor
  | .mla _ => mlaScoreScale

/-- Pack the numerical control of every selected sequence mixer. -/
def controlsFor (model : LanguageModel ℝ cfg decayRank) (mlaScoreScale : ℝ) :
    (indices : List (Fin cfg.numLayers)) → TorchLean.TensorPack ℝ (ControlShapesFor model indices)
  | [] => .nil
  | index :: rest =>
      .cons (.scalar (sequenceControl (model.layer index) mlaScoreScale))
        (controlsFor model mlaScoreScale rest)

/-- Package the values consumed by a decoder segment in its public input order. -/
noncomputable def inputsFor (pastTokens processedLayers : Nat)
    (model : LanguageModel ℝ cfg decayRank)
    (indices : List (Fin cfg.numLayers))
    (depth : TorchLean.TensorPack ℝ
      (DepthSchedule.Shapes cfg.attnResBlockSize processedLayers cfg.hiddenDim))
    (states : TorchLean.TensorPack ℝ (StateShapesFor pastTokens model indices)) (mlaScoreScale : ℝ) :
    TorchLean.TensorPack ℝ (InputsFor pastTokens processedLayers model indices) :=
  TorchLean.TensorPack.append depth <|
    TorchLean.TensorPack.append states <|
      TorchLean.TensorPack.append (controlsFor model mlaScoreScale indices) <|
        .cons (.scalar Numbers.epsilon) <| .cons (.scalar cfg.situGateCap) <|
          .cons (.scalar cfg.situUpCap) .nil

/-- The empty decoder schedule returns its depth state unchanged. -/
theorem outputsFor_nil (pastTokens processedLayers : Nat)
    (model : LanguageModel ℝ cfg decayRank) :
    DepthSchedule.Shapes cfg.attnResBlockSize processedLayers cfg.hiddenDim =
      OutputsFor pastTokens processedLayers model [] := by
  simp [OutputsFor, NextStateShapesFor, processedAfter]

/-- One decoder step prepends its updated causal state to the remaining schedule outputs. -/
theorem outputsFor_cons (pastTokens processedLayers : Nat)
    (model : LanguageModel ℝ cfg decayRank) (index : Fin cfg.numLayers)
    (rest : List (Fin cfg.numLayers)) :
    DecoderLayer.NextStateShapes pastTokens (model.layer index).sequence ++
        OutputsFor pastTokens (processedLayers + 1) model rest =
      OutputsFor pastTokens processedLayers model (index :: rest) := by
  simp [OutputsFor, NextStateShapesFor, processedAfter, List.append_assoc]

/-- Pure mathematical execution of a decoder schedule segment.

This is the schedule-level counterpart of `runBlock`. It calls the already proved layer
specification, advances the mathematical AttnRes state, and recurses. It contains no graph syntax
and no backend operations.
-/
noncomputable def runSpec {architecture : Config} {decayRank : Nat}
    (hcfg : architecture.WF) (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens processedLayers : Nat) :
    (indices : List (Fin architecture.text.numLayers)) →
      TorchLean.TensorPack ℝ (DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim) →
      TorchLean.TensorPack ℝ (StateShapesFor pastTokens model indices) → ℝ →
      TorchLean.TensorPack ℝ (OutputsFor pastTokens processedLayers model indices)
  | [], depth, _states, _mlaScoreScale =>
      outputsFor_nil pastTokens processedLayers model ▸ depth
  | index :: rest, depth, states, mlaScoreScale => by
      let layer := model.layer index
      let stateParts := TorchLean.TensorPack.split
        (ss₁ := DecoderLayer.StateShapes pastTokens layer.sequence)
        (ss₂ := StateShapesFor pastTokens model rest) states
      let completed := Env.tget depth Var.head
      let partialState := Env.tget depth (.tail .head)
      let result := DecoderLayer.specStep pastTokens layer (route index)
        (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
        (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
        hcfg.hiddenDim_pos hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos
        hcfg.kvLatentDim_pos hcfg.routedLatentDim_pos completed partialState stateParts.1
        mlaScoreScale
      let nextDepth := DepthSchedule.advance architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim hcfg.attnResBlockSize_pos completed
        (DecoderLayer.nextPartialValue pastTokens layer.sequence result)
      let restResult := runSpec hcfg model route pastTokens (processedLayers + 1) rest nextDepth
        stateParts.2 mlaScoreScale
      exact outputsFor_cons pastTokens processedLayers model index rest ▸
        TorchLean.TensorPack.append
          (DecoderLayer.nextStateValues pastTokens layer.sequence result) restResult
termination_by indices => indices.length

/-- Pure decoder semantics with every sparse route recomputed from the intermediate channel input.

This is the architecture-level meaning of K3 routing. The route-staged graph below is an executable
specialization of this function; `RoutesAgree` records the condition under which that specialization
is valid for the current token and causal state.
-/
noncomputable def runSpecAuto {architecture : Config} {decayRank : Nat}
    (hcfg : architecture.WF) (model : LanguageModel ℝ architecture.text decayRank)
    (pastTokens processedLayers : Nat) :
    (indices : List (Fin architecture.text.numLayers)) →
      TorchLean.TensorPack ℝ (DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim) →
      TorchLean.TensorPack ℝ (StateShapesFor pastTokens model indices) → ℝ →
      TorchLean.TensorPack ℝ (OutputsFor pastTokens processedLayers model indices)
  | [], depth, _states, _mlaScoreScale =>
      outputsFor_nil pastTokens processedLayers model ▸ depth
  | index :: rest, depth, states, mlaScoreScale => by
      let layer := model.layer index
      let stateParts := TorchLean.TensorPack.split
        (ss₁ := DecoderLayer.StateShapes pastTokens layer.sequence)
        (ss₂ := StateShapesFor pastTokens model rest) states
      let completed := Env.tget depth Var.head
      let partialState := Env.tget depth (.tail .head)
      let result := DecoderLayer.specStepAuto pastTokens layer
        (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
        (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
        hcfg.hiddenDim_pos hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos
        hcfg.kvLatentDim_pos hcfg.routedLatentDim_pos hcfg.activeExperts_le completed partialState
        stateParts.1 mlaScoreScale
      let nextDepth := DepthSchedule.advance architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim hcfg.attnResBlockSize_pos completed
        (DecoderLayer.nextPartialValue pastTokens layer.sequence result)
      let restResult := runSpecAuto hcfg model pastTokens (processedLayers + 1) rest nextDepth
        stateParts.2 mlaScoreScale
      exact outputsFor_cons pastTokens processedLayers model index rest ▸
        TorchLean.TensorPack.append
          (DecoderLayer.nextStateValues pastTokens layer.sequence result) restResult
termination_by indices => indices.length

/-- Every staged route agrees with the deterministic router along the actual decoder execution.

The recursive call follows `runSpecAuto`: consequently, the route for layer `i + 1` is checked using
the representation produced by the automatically routed layers through `i`. Dense layers contribute
the proposition `True`; sparse layers require equality with `StableLatentMoE.route`.
-/
noncomputable def RoutesAgree {architecture : Config} {decayRank : Nat}
    (hcfg : architecture.WF) (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens processedLayers : Nat) :
    (indices : List (Fin architecture.text.numLayers)) →
      TorchLean.TensorPack ℝ (DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim) →
      TorchLean.TensorPack ℝ (StateShapesFor pastTokens model indices) → ℝ → Prop
  | [], _depth, _states, _mlaScoreScale => True
  | index :: rest, depth, states, mlaScoreScale =>
      let layer := model.layer index
      let stateParts := TorchLean.TensorPack.split
        (ss₁ := DecoderLayer.StateShapes pastTokens layer.sequence)
        (ss₂ := StateShapesFor pastTokens model rest) states
      let completed := Env.tget depth Var.head
      let partialState := Env.tget depth (.tail .head)
      let routeAgrees := DecoderLayer.RouteAgrees pastTokens layer (route index)
        (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
        (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
        hcfg.hiddenDim_pos hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos
        hcfg.kvLatentDim_pos hcfg.activeExperts_le completed partialState stateParts.1 mlaScoreScale
      let result := DecoderLayer.specStepAuto pastTokens layer
        (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
        (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
        hcfg.hiddenDim_pos hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos
        hcfg.kvLatentDim_pos hcfg.routedLatentDim_pos hcfg.activeExperts_le completed partialState
        stateParts.1 mlaScoreScale
      let nextDepth := DepthSchedule.advance architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim hcfg.attnResBlockSize_pos completed
        (DecoderLayer.nextPartialValue pastTokens layer.sequence result)
      routeAgrees ∧ RoutesAgree hcfg model route pastTokens (processedLayers + 1) rest nextDepth
        stateParts.2 mlaScoreScale
termination_by indices => indices.length

/-- A route-staged decoder run equals K3's automatically routed semantics when its trace agrees. -/
theorem runSpec_eq_runSpecAuto_of_routesAgree
    {architecture : Config} {decayRank : Nat} (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens processedLayers : Nat) (indices : List (Fin architecture.text.numLayers))
    (depth : TorchLean.TensorPack ℝ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim))
    (states : TorchLean.TensorPack ℝ (StateShapesFor pastTokens model indices)) (mlaScoreScale : ℝ)
    (hRoutes : RoutesAgree hcfg model route pastTokens processedLayers indices depth states
      mlaScoreScale) :
    runSpec hcfg model route pastTokens processedLayers indices depth states mlaScoreScale =
      runSpecAuto hcfg model pastTokens processedLayers indices depth states mlaScoreScale := by
  induction indices generalizing processedLayers with
  | nil =>
      rw [runSpec.eq_def, runSpecAuto.eq_def]
  | cons index rest ih =>
      simp only [RoutesAgree] at hRoutes
      rw [runSpec, runSpecAuto]
      rw [DecoderLayer.specStep_eq_specStepAuto_of_routeAgrees
        (hRoute := hRoutes.1)]
      rw [ih _ _ _ hRoutes.2]

/-!
### Recursive decoder graph
-/

/-- Inline a schedule segment into one typed DAG block.

Each recursive step performs three operations: execute the selected decoder-layer graph, advance
the packed AttnRes state, and continue with the remaining layers.  The route is static graph data;
a later theorem can discharge its `Route.IsTopK` obligation without placing a sorting primitive
inside the numerical graph.
-/
def runBlock {architecture : Config} {decayRank : Nat} (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    {Γ : List Shape} (pastTokens processedLayers : Nat) :
    (indices : List (Fin architecture.text.numLayers)) →
      Args Γ (ParamsFor model indices) →
      Args Γ (DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim) →
      Args Γ (StateShapesFor pastTokens model indices) →
      Args Γ (ControlShapesFor model indices) →
      Term Γ .scalar → Term Γ .scalar → Term Γ .scalar →
      Block Γ (OutputsFor pastTokens processedLayers model indices)
  | [], params, depth, states, controls, _epsilon, _gateCap, _upCap => by
      cases params
      cases states
      cases controls
      exact Block.castOutputs (outputsFor_nil pastTokens processedLayers model)
        (Block.ret depth)
  | index :: rest, params, depth, states, controls, epsilon, gateCap, upCap => by
      let layer := model.layer index
      let paramParts := Args.splitAppend
        (left := DecoderLayer.Params layer.sequence layer.channel)
        (right := ParamsFor model rest) params
      let stateParts := Args.splitAppend
        (left := DecoderLayer.StateShapes pastTokens layer.sequence)
        (right := StateShapesFor pastTokens model rest) states
      let completed := Args.get depth Var.head
      let partialState := Args.get depth (.tail .head)
      let controlParts := Args.splitAppend
        (left := [.scalar]) (right := ControlShapesFor model rest) controls
      let sequenceControl := Args.get controlParts.1 Var.head
      let layerGraph := DecoderLayer.model pastTokens
        (DepthSchedule.completedCount architecture.text.attnResBlockSize processedLayers)
        layer (route index)
        (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
        (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
        hcfg.hiddenDim_pos hcfg.numHeads_pos hcfg.kdaHeadDim_pos hcfg.kdaValueDim_pos
        hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos hcfg.kvLatentDim_pos
        hcfg.routedLatentDim_pos
      let layerInputs := DecoderLayer.inputTerms pastTokens layer.sequence completed partialState
        stateParts.1 sequenceControl epsilon gateCap upCap
      let layerBlock := layerGraph.inline paramParts.1 layerInputs
      let layerOutputs := DecoderLayer.Outputs pastTokens layer.sequence
      exact layerBlock.andThen <|
        let results : Args (Γ ++ layerOutputs) layerOutputs :=
          Args.rename (Var.inRight Γ) (Args.vars layerOutputs)
        let nextStates := DecoderLayer.nextStateTerms pastTokens layer.sequence results
        let nextPartial := DecoderLayer.nextPartialTerm pastTokens layer.sequence results
        let completed' := Term.weakenAppend layerOutputs completed
        let depthBlock := DepthSchedule.advanceBlock architecture.text.attnResBlockSize
          processedLayers architecture.text.hiddenDim hcfg.attnResBlockSize_pos completed'
          nextPartial
        depthBlock.andThen <|
          let nextDepthShapes := DepthSchedule.Shapes architecture.text.attnResBlockSize
            (processedLayers + 1) architecture.text.hiddenDim
          let extended := Γ ++ layerOutputs
          let depthResults : Args (extended ++ nextDepthShapes) nextDepthShapes :=
            Args.rename (Var.inRight extended) (Args.vars nextDepthShapes)
          let liftOriginal := fun {sh : Shape} (term : Term Γ sh) =>
            Term.weakenAppend nextDepthShapes (Term.weakenAppend layerOutputs term)
          let liftOriginalArgs := fun {shapes : List Shape} (args : Args Γ shapes) =>
            Args.rename (Var.inLeft nextDepthShapes)
              (Args.rename (Var.inLeft layerOutputs) args)
          let recursive := runBlock hcfg model route pastTokens (processedLayers + 1) rest
            (liftOriginalArgs paramParts.2) depthResults (liftOriginalArgs stateParts.2)
            (liftOriginalArgs controlParts.2) (liftOriginal epsilon) (liftOriginal gateCap)
            (liftOriginal upCap)
          let recursiveOutputs := OutputsFor pastTokens (processedLayers + 1) model rest
          recursive.andThen <|
            let recursiveResults : Args ((extended ++ nextDepthShapes) ++ recursiveOutputs)
                recursiveOutputs :=
              Args.rename (Var.inRight (extended ++ nextDepthShapes))
                (Args.vars recursiveOutputs)
            let nextStates' := Args.rename (Var.inLeft nextDepthShapes) nextStates
            let nextStates'' := Args.rename (Var.inLeft recursiveOutputs) nextStates'
            Block.castOutputs (outputsFor_cons pastTokens processedLayers model index rest) <|
              Block.ret (Args.append nextStates'' recursiveResults)
termination_by indices => indices.length

/-- The recursively composed decoder block denotes the pure decoder schedule.

The hypotheses identify the learned parameters and the four shared numerical controls in the
surrounding graph environment.  Causal and AttnRes states remain arbitrary: the theorem therefore
applies both to a fresh prompt and to continuation from any well-typed cache. -/
theorem eval_runBlock {architecture : Config} {decayRank : Nat} (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ) (pastTokens processedLayers : Nat)
    (indices : List (Fin architecture.text.numLayers))
    (params : Args Γ (ParamsFor languageModel indices))
    (depth : Args Γ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim))
    (states : Args Γ (StateShapesFor pastTokens languageModel indices))
    (controls : Args Γ (ControlShapesFor languageModel indices))
    (epsilon gateCap upCap : Term Γ .scalar) (mlaScoreScale : ℝ)
    (hParams : Term.evalArgs env params = parametersFor languageModel indices)
    (hControls : Term.evalArgs env controls =
      controlsFor languageModel mlaScoreScale indices)
    (hEpsilon : Term.eval env epsilon = .scalar Numbers.epsilon)
    (hGateCap : Term.eval env gateCap =
      .scalar (architecture.text.situGateCap : ℝ))
    (hUpCap : Term.eval env upCap = .scalar (architecture.text.situUpCap : ℝ)) :
    Block.eval env
        (runBlock hcfg languageModel route pastTokens processedLayers indices params depth states
          controls epsilon gateCap upCap) =
      runSpec hcfg languageModel route pastTokens processedLayers indices
        (Term.evalArgs env depth) (Term.evalArgs env states) mlaScoreScale := by
  induction indices generalizing Γ processedLayers with
  | nil =>
      rw [runBlock.eq_def, runSpec.eq_def]
      cases params
      cases states
      cases controls
      simp only [Block.eval_castOutputs, Block.eval]
      rfl
  | cons index rest ih =>
      rw [runBlock.eq_def]
      let layer := languageModel.layer index
      change Args Γ
        (DecoderLayer.Params layer.sequence layer.channel ++ ParamsFor languageModel rest) at params
      change Args Γ
        (DecoderLayer.StateShapes pastTokens layer.sequence ++
          StateShapesFor pastTokens languageModel rest) at states
      change Args Γ ([.scalar] ++ ControlShapesFor languageModel rest) at controls
      let paramParts := Args.splitAppend
        (left := DecoderLayer.Params layer.sequence layer.channel)
        (right := ParamsFor languageModel rest) params
      let stateParts := Args.splitAppend
        (left := DecoderLayer.StateShapes pastTokens layer.sequence)
        (right := StateShapesFor pastTokens languageModel rest) states
      let controlParts := Args.splitAppend
        (left := [.scalar]) (right := ControlShapesFor languageModel rest) controls
      let completed := Args.get depth Var.head
      let partialState := Args.get depth (.tail .head)
      let control := Args.get controlParts.1 Var.head
      have hParamParts := Term.evalArgs_splitAppend env params
      have hStateParts := Term.evalArgs_splitAppend env states
      have hControlParts := Term.evalArgs_splitAppend env controls
      dsimp [paramParts, stateParts, controlParts] at hParamParts hStateParts hControlParts
      have hParamParts' := hParamParts.trans <| congrArg
        (TorchLean.TensorPack.split
          (ss₁ := DecoderLayer.Params layer.sequence layer.channel)
          (ss₂ := ParamsFor languageModel rest)) hParams
      simp only [parametersFor,
        TorchLean.TensorPack.split_append] at hParamParts'
      have hControlParts' := hControlParts.trans <| congrArg
        (TorchLean.TensorPack.split
          (ss₁ := [.scalar]) (ss₂ := ControlShapesFor languageModel rest)) hControls
      simp only [controlsFor] at hControlParts'
      have hLayerParams : Term.evalArgs env paramParts.1 =
          DecoderLayer.parameters layer := congrArg Prod.fst hParamParts'
      have hRestParams : Term.evalArgs env paramParts.2 =
          parametersFor languageModel rest := congrArg Prod.snd hParamParts'
      have hControlValues : Term.evalArgs env controlParts.1 =
          .cons (.scalar (sequenceControl layer mlaScoreScale)) .nil :=
        congrArg Prod.fst hControlParts'
      have hRestControls : Term.evalArgs env controlParts.2 =
          controlsFor languageModel mlaScoreScale rest := congrArg Prod.snd hControlParts'
      have hControl : Term.eval env control =
          .scalar (sequenceControl layer mlaScoreScale) := by
        calc
          Term.eval env control = Env.tget (Term.evalArgs env controlParts.1) Var.head :=
            Term.eval_get env controlParts.1 Var.head
          _ = .scalar (sequenceControl layer mlaScoreScale) := by rw [hControlValues]; rfl
      have hLayerResult :
          Block.eval env
              ((DecoderLayer.model pastTokens
                (DepthSchedule.completedCount architecture.text.attnResBlockSize processedLayers)
                layer (route index)
                (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
                (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
                hcfg.hiddenDim_pos hcfg.numHeads_pos hcfg.kdaHeadDim_pos hcfg.kdaValueDim_pos
                hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos hcfg.kvLatentDim_pos
                hcfg.routedLatentDim_pos).inline paramParts.1
                  (DecoderLayer.inputTerms pastTokens layer.sequence completed partialState stateParts.1
                    control epsilon gateCap upCap)) =
            DecoderLayer.specStep pastTokens layer (route index)
              (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
              (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
              hcfg.hiddenDim_pos hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos
              hcfg.kvLatentDim_pos hcfg.routedLatentDim_pos (Term.eval env completed)
              (Term.eval env partialState) (Term.evalArgs env stateParts.1)
              mlaScoreScale := by
        rw [NN.GraphSpec.DAG.MultiModel.eval_inline, hLayerParams]
        rw [DecoderLayer.evalArgs_inputTerms env pastTokens layer.sequence completed partialState
          stateParts.1 control epsilon gateCap upCap mlaScoreScale]
        · exact DecoderLayer.model_specFwd_eq_specStep pastTokens layer (route index)
            (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
            (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
            hcfg.hiddenDim_pos hcfg.numHeads_pos hcfg.kdaHeadDim_pos hcfg.kdaValueDim_pos
            hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos hcfg.kvLatentDim_pos
            hcfg.routedLatentDim_pos hcfg.activeExperts_pos (Term.eval env completed)
            (Term.eval env partialState) (Term.evalArgs env stateParts.1) mlaScoreScale
        · simpa only [sequenceControl, DecoderLayer.controlValue] using hControl
        · exact hEpsilon
        · exact hGateCap
        · exact hUpCap
      rw [Block.eval_andThen, hLayerResult]
      rw [Block.eval_andThen]
      rw [DepthSchedule.eval_advanceBlock]
      simp only [Term.eval_weakenAppend, DecoderLayer.eval_nextPartialTerm,
        Term.evalArgs_rename_inRight, Term.evalArgs_vars]
      set layerResult := DecoderLayer.specStep pastTokens layer (route index)
        (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
        (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
        hcfg.hiddenDim_pos hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos
        hcfg.kvLatentDim_pos hcfg.routedLatentDim_pos (Term.eval env completed)
        (Term.eval env partialState) (Term.evalArgs env stateParts.1) mlaScoreScale
      set nextDepth := DepthSchedule.advance architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim hcfg.attnResBlockSize_pos (Term.eval env completed)
        (DecoderLayer.nextPartialValue pastTokens layer.sequence layerResult)
      set envDepth := TorchLean.TensorPack.append
        (TorchLean.TensorPack.append env layerResult) nextDepth
      let layerOutputs := DecoderLayer.Outputs pastTokens layer.sequence
      let nextDepthShapes := DepthSchedule.Shapes architecture.text.attnResBlockSize
        (processedLayers + 1) architecture.text.hiddenDim
      let extended := Γ ++ layerOutputs
      let depthResults : Args (extended ++ nextDepthShapes) nextDepthShapes :=
        Args.rename (Var.inRight extended) (Args.vars nextDepthShapes)
      let liftOriginal := fun {shape : Shape} (term : Term Γ shape) =>
        Term.weakenAppend nextDepthShapes (Term.weakenAppend layerOutputs term)
      let liftOriginalArgs := fun {shapes : List Shape} (args : Args Γ shapes) =>
        Args.rename (Var.inLeft nextDepthShapes) (Args.rename (Var.inLeft layerOutputs) args)
      have hDepthResults : Term.evalArgs envDepth depthResults = nextDepth := by
        change Term.evalArgs
            (TorchLean.TensorPack.append
              (TorchLean.TensorPack.append env layerResult) nextDepth)
            (Args.rename (Var.inRight (Γ ++ layerOutputs))
              (Args.vars nextDepthShapes)) = nextDepth
        rw [Term.evalArgs_rename_inRight, Term.evalArgs_vars]
      have hLiftedParams :
          Term.evalArgs envDepth (liftOriginalArgs paramParts.2) =
            parametersFor languageModel rest := by
        rw [show envDepth =
            TorchLean.TensorPack.append
              (TorchLean.TensorPack.append env layerResult) nextDepth by rfl]
        change Term.evalArgs
            (TorchLean.TensorPack.append
              (TorchLean.TensorPack.append env layerResult) nextDepth)
            (Args.rename (Var.inLeft nextDepthShapes)
              (Args.rename (Var.inLeft layerOutputs) paramParts.2)) = _
        rw [Term.evalArgs_rename_inLeft, Term.evalArgs_rename_inLeft]
        exact hRestParams
      have hLiftedStates :
          Term.evalArgs envDepth (liftOriginalArgs stateParts.2) =
            Term.evalArgs env stateParts.2 := by
        rw [show envDepth =
            TorchLean.TensorPack.append
              (TorchLean.TensorPack.append env layerResult) nextDepth by rfl]
        change Term.evalArgs
            (TorchLean.TensorPack.append
              (TorchLean.TensorPack.append env layerResult) nextDepth)
            (Args.rename (Var.inLeft nextDepthShapes)
              (Args.rename (Var.inLeft layerOutputs) stateParts.2)) = _
        rw [Term.evalArgs_rename_inLeft, Term.evalArgs_rename_inLeft]
      have hLiftedControls :
          Term.evalArgs envDepth (liftOriginalArgs controlParts.2) =
            controlsFor languageModel mlaScoreScale rest := by
        rw [show envDepth =
            TorchLean.TensorPack.append
              (TorchLean.TensorPack.append env layerResult) nextDepth by rfl]
        change Term.evalArgs
            (TorchLean.TensorPack.append
              (TorchLean.TensorPack.append env layerResult) nextDepth)
            (Args.rename (Var.inLeft nextDepthShapes)
              (Args.rename (Var.inLeft layerOutputs) controlParts.2)) = _
        rw [Term.evalArgs_rename_inLeft, Term.evalArgs_rename_inLeft]
        exact hRestControls
      have hLiftedEpsilon :
          Term.eval envDepth (liftOriginal epsilon) = Tensor.scalar Numbers.epsilon := by
        rw [show envDepth =
            TorchLean.TensorPack.append
              (TorchLean.TensorPack.append env layerResult) nextDepth by rfl]
        change Term.eval
            (TorchLean.TensorPack.append
              (TorchLean.TensorPack.append env layerResult) nextDepth)
            (Term.weakenAppend nextDepthShapes
              (Term.weakenAppend layerOutputs epsilon)) = _
        rw [Term.eval_weakenAppend, Term.eval_weakenAppend]
        exact hEpsilon
      have hLiftedGateCap :
          Term.eval envDepth (liftOriginal gateCap) =
            Tensor.scalar (architecture.text.situGateCap : ℝ) := by
        rw [show envDepth =
            TorchLean.TensorPack.append
              (TorchLean.TensorPack.append env layerResult) nextDepth by rfl]
        change Term.eval
            (TorchLean.TensorPack.append
              (TorchLean.TensorPack.append env layerResult) nextDepth)
            (Term.weakenAppend nextDepthShapes
              (Term.weakenAppend layerOutputs gateCap)) = _
        rw [Term.eval_weakenAppend, Term.eval_weakenAppend]
        exact hGateCap
      have hLiftedUpCap :
          Term.eval envDepth (liftOriginal upCap) =
            Tensor.scalar (architecture.text.situUpCap : ℝ) := by
        rw [show envDepth =
            TorchLean.TensorPack.append
              (TorchLean.TensorPack.append env layerResult) nextDepth by rfl]
        change Term.eval
            (TorchLean.TensorPack.append
              (TorchLean.TensorPack.append env layerResult) nextDepth)
            (Term.weakenAppend nextDepthShapes
              (Term.weakenAppend layerOutputs upCap)) = _
        rw [Term.eval_weakenAppend, Term.eval_weakenAppend]
        exact hUpCap
      have hRecursive :
          Block.eval envDepth
              (runBlock hcfg languageModel route pastTokens (processedLayers + 1) rest
                (liftOriginalArgs paramParts.2) depthResults
                (liftOriginalArgs stateParts.2) (liftOriginalArgs controlParts.2)
                (liftOriginal epsilon) (liftOriginal gateCap) (liftOriginal upCap)) =
            runSpec hcfg languageModel route pastTokens (processedLayers + 1) rest nextDepth
              (Term.evalArgs env stateParts.2) mlaScoreScale := by
        rw [ih envDepth (processedLayers + 1) (liftOriginalArgs paramParts.2)
          depthResults (liftOriginalArgs stateParts.2) (liftOriginalArgs controlParts.2)
          (liftOriginal epsilon) (liftOriginal gateCap) (liftOriginal upCap)
          hLiftedParams hLiftedControls hLiftedEpsilon hLiftedGateCap hLiftedUpCap,
          hDepthResults, hLiftedStates]
      rw [Block.eval_andThen]
      rw [hRecursive]
      simp only [Block.eval_castOutputs, Block.eval, Term.evalArgs_append, envDepth,
        Term.evalArgs_rename_inLeft,
        DecoderLayer.evalArgs_nextStateTerms, Term.evalArgs_rename_inRight,
        Term.evalArgs_vars, layer]
      have hCompletedValue :
          Env.tget (Term.evalArgs env depth) Var.head = Term.eval env completed :=
        (Term.eval_get env depth Var.head).symm
      have hPartialValue :
          Env.tget (Term.evalArgs env depth) (.tail .head) =
            Term.eval env partialState :=
        (Term.eval_get env depth (.tail .head)).symm
      rw [runSpec]
      rw [hCompletedValue, hPartialValue]
      simp only [StateShapesFor, layer] at states hStateParts ⊢
      rw [← hStateParts]

/-- Default GraphSpec parameters for a decoder segment.

These are the component models' existing initial values and are intended for graph construction and
shape checks.  A released checkpoint replaces them through the same `ParamsFor` ABI.
-/
def initialParamsFor {architecture : Config} {decayRank : Nat} (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens processedLayers : Nat) :
    (indices : List (Fin architecture.text.numLayers)) → TorchLean.TensorPack Float (ParamsFor model indices)
  | [] => .nil
  | index :: rest =>
      let layer := model.layer index
      let layerGraph := DecoderLayer.model pastTokens
        (DepthSchedule.completedCount architecture.text.attnResBlockSize processedLayers)
        layer (route index)
        (DepthSchedule.partialPresent architecture.text.attnResBlockSize processedLayers)
        (DepthSchedule.completedCount_pos architecture.text.attnResBlockSize processedLayers)
        hcfg.hiddenDim_pos hcfg.numHeads_pos hcfg.kdaHeadDim_pos hcfg.kdaValueDim_pos
        hcfg.shortConvWidth_pos hcfg.queryLatentDim_pos hcfg.kvLatentDim_pos
        hcfg.routedLatentDim_pos
      TorchLean.TensorPack.append layerGraph.initParams
        (initialParamsFor hcfg model route pastTokens (processedLayers + 1) rest)

/-- A complete typed GraphSpec model for a decoder schedule segment. -/
def model {architecture : Config} {decayRank : Nat} (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens processedLayers : Nat)
    (indices : List (Fin architecture.text.numLayers)) :
    NN.GraphSpec.DAG.MultiModel (ParamsFor languageModel indices)
      (InputsFor pastTokens processedLayers languageModel indices)
      (OutputsFor pastTokens processedLayers languageModel indices) :=
  { initParams := initialParamsFor hcfg languageModel route pastTokens processedLayers indices
    body := by
      let all := Args.vars
        (ParamsFor languageModel indices ++
          InputsFor pastTokens processedLayers languageModel indices)
      let parameterAndInput := Args.splitAppend
        (left := ParamsFor languageModel indices)
        (right := InputsFor pastTokens processedLayers languageModel indices) all
      let depthAndRest := Args.splitAppend
        (left := DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
          architecture.text.hiddenDim)
        (right := StateShapesFor pastTokens languageModel indices ++
          (ControlShapesFor languageModel indices ++ [.scalar, .scalar, .scalar]))
        parameterAndInput.2
      let stateAndRest := Args.splitAppend
        (left := StateShapesFor pastTokens languageModel indices)
        (right := ControlShapesFor languageModel indices ++ [.scalar, .scalar, .scalar])
        depthAndRest.2
      let controlAndShared := Args.splitAppend
        (left := ControlShapesFor languageModel indices)
        (right := [.scalar, .scalar, .scalar]) stateAndRest.2
      let epsilon := Args.get controlAndShared.2 Var.head
      let gateCap := Args.get controlAndShared.2 (.tail .head)
      let upCap := Args.get controlAndShared.2 (.tail (.tail .head))
      exact runBlock hcfg languageModel route pastTokens processedLayers indices
        parameterAndInput.1 depthAndRest.1 stateAndRest.1 controlAndShared.1
        epsilon gateCap upCap }

/-- The public decoder model denotes the recursive mathematical schedule on arbitrary typed state.

Unlike the component theorems, this statement covers a heterogeneous list of KDA and MLA layers.
The output includes every updated causal state followed by the packed AttnRes depth state.
-/
theorem model_specFwd_eq_runSpec {architecture : Config} {decayRank : Nat}
    (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens processedLayers : Nat)
    (indices : List (Fin architecture.text.numLayers))
    (depth : TorchLean.TensorPack ℝ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize processedLayers
        architecture.text.hiddenDim))
    (states : TorchLean.TensorPack ℝ (StateShapesFor pastTokens languageModel indices))
    (mlaScoreScale : ℝ) :
    (model hcfg languageModel route pastTokens processedLayers indices).specFwd
        (parametersFor languageModel indices)
        (inputsFor pastTokens processedLayers languageModel indices depth states mlaScoreScale) =
      runSpec hcfg languageModel route pastTokens processedLayers indices depth states
        mlaScoreScale := by
  simp only [NN.GraphSpec.DAG.MultiModel.specFwd, model, InputsFor, inputsFor]
  rw [eval_runBlock (mlaScoreScale := mlaScoreScale)]
  all_goals
    simp only [Term.evalArgs_splitAppend_fst, Term.evalArgs_splitAppend_snd,
      Term.evalArgs_vars, TorchLean.TensorPack.split_append,
      Term.eval_get, Env.tget]

/-- The full configured decoder schedule in its canonical zero-based layer order. -/
def fullModel {architecture : Config} {decayRank : Nat} (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) :
    NN.GraphSpec.DAG.MultiModel
      (ParamsFor languageModel (List.finRange architecture.text.numLayers))
      (InputsFor pastTokens 0 languageModel (List.finRange architecture.text.numLayers))
      (OutputsFor pastTokens 0 languageModel (List.finRange architecture.text.numLayers)) :=
  model hcfg languageModel route pastTokens 0 (List.finRange architecture.text.numLayers)

end Decoder

namespace TokenStep

open Spec
open NN.GraphSpec.DAG
open Runtime.Autograd.Torch

variable {architecture : Config} {decayRank : Nat}

/-- Parameters used by one complete autoregressive token step.

The embedding table comes first, followed by every decoder-layer parameter and the three learned
objects used after the decoder: the final AttnRes query, RMSNorm scale, and vocabulary projection.
-/
abbrev Params (model : LanguageModel ℝ architecture.text decayRank) : List Shape :=
  .dim architecture.text.vocabSize (.dim architecture.text.hiddenDim .scalar) ::
    (Decoder.ParamsFor model (List.finRange architecture.text.numLayers) ++
      [.dim architecture.text.hiddenDim .scalar,
        .dim architecture.text.hiddenDim .scalar,
        .dim architecture.text.hiddenDim (.dim architecture.text.vocabSize .scalar)])

/-- Runtime inputs used by one complete autoregressive token step. -/
abbrev Inputs (pastTokens : Nat)
    (model : LanguageModel ℝ architecture.text decayRank) : List Shape :=
  Decoder.StateShapesFor pastTokens model (List.finRange architecture.text.numLayers) ++
    (Decoder.ControlShapesFor model (List.finRange architecture.text.numLayers) ++
      [.scalar, .scalar, .scalar])

/-- Updated causal states and the current token's vocabulary logits. -/
abbrev Outputs (pastTokens : Nat)
    (model : LanguageModel ℝ architecture.text decayRank) : List Shape :=
  Decoder.NextStateShapesFor pastTokens model (List.finRange architecture.text.numLayers) ++
    [.dim architecture.text.vocabSize .scalar]

/-- Learned real-valued tensors in the `Params` order. -/
def parameters (model : LanguageModel ℝ architecture.text decayRank) : TorchLean.TensorPack ℝ (Params model) := by
  let decoder := Decoder.parametersFor model (List.finRange architecture.text.numLayers)
  let final : TorchLean.TensorPack ℝ
      [.dim architecture.text.hiddenDim .scalar,
        .dim architecture.text.hiddenDim .scalar,
        .dim architecture.text.hiddenDim (.dim architecture.text.vocabSize .scalar)] :=
    .cons model.finalQuery <| .cons model.finalNormScale <| .cons model.vocabularyHead .nil
  exact .cons model.tokenEmbedding <|
    TorchLean.TensorPack.append
      (ss₁ := Decoder.ParamsFor model (List.finRange architecture.text.numLayers))
      (ss₂ :=
        [.dim architecture.text.hiddenDim .scalar,
          .dim architecture.text.hiddenDim .scalar,
          .dim architecture.text.hiddenDim (.dim architecture.text.vocabSize .scalar)])
      decoder final

/-- Runtime values in the `Inputs` order. -/
noncomputable def inputs (pastTokens : Nat) (model : LanguageModel ℝ architecture.text decayRank)
    (states : TorchLean.TensorPack ℝ
      (Decoder.StateShapesFor pastTokens model (List.finRange architecture.text.numLayers)))
    (mlaScoreScale : ℝ) : TorchLean.TensorPack ℝ (Inputs pastTokens model) :=
  TorchLean.TensorPack.append states <|
    TorchLean.TensorPack.append
      (Decoder.controlsFor model mlaScoreScale (List.finRange architecture.text.numLayers)) <|
        .cons (.scalar Numbers.epsilon) <| .cons (.scalar architecture.text.situGateCap) <|
          .cons (.scalar architecture.text.situUpCap) .nil

/-- Pack a token embedding into the initial AttnRes state. -/
def initialDepth (embedding : Tensor ℝ (.dim architecture.text.hiddenDim .scalar)) :
    TorchLean.TensorPack ℝ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize 0 architecture.text.hiddenDim) :=
  .cons (Tensor.reshapeSpec embedding (by
      simp [Shape.size, DepthSchedule.completedCount])) <|
    .cons (Spec.fill 0 (.dim architecture.text.hiddenDim .scalar)) .nil

/-- Graph terms for the initial AttnRes state of one token. -/
def initialDepthTerms {Γ : List Shape}
    (embedding : Term Γ (.dim architecture.text.hiddenDim .scalar)) :
    Args Γ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize 0 architecture.text.hiddenDim) :=
  .cons
    (Term.op
      (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by
        simp [Shape.size, DepthSchedule.completedCount]))
      (.cons embedding .nil)) <|
    .cons (Term.op
      (NN.GraphSpec.DAG.PrimOp.zero (.dim architecture.text.hiddenDim .scalar)) .nil) .nil

/-- Evaluating the initial-depth terms gives the mathematical initial AttnRes state. -/
theorem eval_initialDepthTerms {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (embedding : Term Γ (.dim architecture.text.hiddenDim .scalar)) :
    Term.evalArgs env (initialDepthTerms (architecture := architecture) embedding) =
      initialDepth (Term.eval env embedding) := by
  simp only [initialDepthTerms, initialDepth, Term.eval, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.reshape_specFwd, NN.GraphSpec.DAG.PrimOp.zero_specFwd]

/-- Number of depth representations visible to the final AttnRes query. -/
def finalSourceCount : Nat :=
  DepthSchedule.completedCount architecture.text.attnResBlockSize architecture.text.numLayers +
    if architecture.text.numLayers % architecture.text.attnResBlockSize = 0 then 0 else 1

/-- The final AttnRes query always sees at least the token embedding. -/
theorem finalSourceCount_pos : 0 < finalSourceCount (architecture := architecture) := by
  simp [finalSourceCount, DepthSchedule.completedCount]

/-- Convert the final packed depth state into the matrix consumed by AttnRes.

Completed depth blocks are already rows of the first tensor. If the final block is incomplete, its
partial sum is appended as one additional row.
-/
def finalSources
    (depth : TorchLean.TensorPack ℝ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize architecture.text.numLayers
        architecture.text.hiddenDim)) :
    Tensor ℝ (.dim (finalSourceCount (architecture := architecture))
      (.dim architecture.text.hiddenDim .scalar)) :=
  let completed := Env.tget depth Var.head
  let partialState := Env.tget depth (.tail .head)
  if hBoundary :
      architecture.text.numLayers % architecture.text.attnResBlockSize = 0 then
    Tensor.castShape completed (by simp [finalSourceCount, hBoundary])
  else
    let row : Tensor ℝ (.dim 1 (.dim architecture.text.hiddenDim .scalar)) :=
      Tensor.reshapeSpec partialState (by simp [Shape.size])
    Tensor.castShape (Tensor.concatAxisSpec .scalar completed row)
      (by simp [finalSourceCount, hBoundary])

/-- Graph term that exposes exactly the depth rows visible to the final AttnRes query. -/
def finalSourcesTerm {Γ : List Shape}
    (depth : Args Γ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize architecture.text.numLayers
        architecture.text.hiddenDim)) :
    Term Γ (.dim (finalSourceCount (architecture := architecture))
      (.dim architecture.text.hiddenDim .scalar)) :=
  let completed := Args.get depth Var.head
  let partialState := Args.get depth (.tail .head)
  if hBoundary :
      architecture.text.numLayers % architecture.text.attnResBlockSize = 0 then
    Term.cast completed (by simp [finalSourceCount, hBoundary])
  else
    let row : Term Γ (.dim 1 (.dim architecture.text.hiddenDim .scalar)) :=
      Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
        (.cons partialState .nil)
    let appended : Term Γ
        (.dim (DepthSchedule.completedCount architecture.text.attnResBlockSize
          architecture.text.numLayers + 1) (.dim architecture.text.hiddenDim .scalar)) :=
      GraphSpec.concatAxisZeroTerm
        (DepthSchedule.completedCount architecture.text.attnResBlockSize
          architecture.text.numLayers) 1 (.dim architecture.text.hiddenDim .scalar) completed row
    Term.cast appended (by simp [finalSourceCount, hBoundary])

/-- The final-source graph term denotes the mathematical packed source matrix. -/
theorem eval_finalSourcesTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (depth : Args Γ
      (DepthSchedule.Shapes architecture.text.attnResBlockSize architecture.text.numLayers
        architecture.text.hiddenDim)) :
    Term.eval env (finalSourcesTerm (architecture := architecture) depth) =
      finalSources (Term.evalArgs env depth) := by
  by_cases hBoundary :
      architecture.text.numLayers % architecture.text.attnResBlockSize = 0
  · have hCompleted := Term.eval_get env depth (Var.head)
    simp only [finalSourcesTerm, finalSources, hBoundary, dite_true, Term.eval_cast]
    rw [hCompleted, Tensor.eqRec_eq_cast_shape]
  · have hCompleted := Term.eval_get env depth (Var.head)
    have hPartial := Term.eval_get env depth (.tail .head)
    simp only [finalSourcesTerm, finalSources, hBoundary, dite_false, Term.eval_cast,
      Term.evalArgs]
    rw [GraphSpec.eval_concatAxisZeroTerm]
    simp only [Term.eval_op, Term.evalArgs, NN.GraphSpec.DAG.PrimOp.reshape_specFwd]
    rw [hCompleted, hPartial]
    rw [Tensor.eqRec_eq_cast_shape]

private theorem processedAfter_full :
    Decoder.processedAfter 0 (List.finRange architecture.text.numLayers) =
      architecture.text.numLayers := by
  rw [Decoder.processedAfter_eq_add_length]
  simp

/-- The complete decoder returns every updated layer cache followed by its final depth state. -/
theorem outputsFor_full (pastTokens : Nat)
    (model : LanguageModel ℝ architecture.text decayRank) :
    Decoder.OutputsFor pastTokens 0 model (List.finRange architecture.text.numLayers) =
      Decoder.NextStateShapesFor pastTokens model (List.finRange architecture.text.numLayers) ++
        DepthSchedule.Shapes architecture.text.attnResBlockSize architecture.text.numLayers
          architecture.text.hiddenDim := by
  simp only [Decoder.OutputsFor]
  rw [processedAfter_full]

/-- Apply K3's final AttnRes retrieval, normalization, and vocabulary projection to a decoder run. -/
noncomputable def finishDecoder (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (pastTokens : Nat)
    (decoderResult : TorchLean.TensorPack ℝ
      (Decoder.OutputsFor pastTokens 0 model (List.finRange architecture.text.numLayers))) :
    TorchLean.TensorPack ℝ (Outputs pastTokens model) :=
  let normalizedResult : TorchLean.TensorPack ℝ
      (Decoder.NextStateShapesFor pastTokens model
          (List.finRange architecture.text.numLayers) ++
        DepthSchedule.Shapes architecture.text.attnResBlockSize architecture.text.numLayers
          architecture.text.hiddenDim) :=
    outputsFor_full pastTokens model ▸ decoderResult
  let parts := TorchLean.TensorPack.split normalizedResult
  let sources := finalSources parts.2
  let retrieved := AttnRes.attendPacked hcfg.hiddenDim_pos
    model.finalQuery sources
  let hidden := RMSNorm.scale retrieved model.finalNormScale
  let logits := vecMatMulSpec hidden model.vocabularyHead
  TorchLean.TensorPack.append parts.1 (.cons logits .nil)

/-- Pure semantics of one complete autoregressive token step with a staged route trace. -/
noncomputable def runSpec (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) (token : Fin architecture.text.vocabSize)
    (states : TorchLean.TensorPack ℝ
      (Decoder.StateShapesFor pastTokens model (List.finRange architecture.text.numLayers)))
    (mlaScoreScale : ℝ) : TorchLean.TensorPack ℝ (Outputs pastTokens model) :=
  let embedding := Spec.get model.tokenEmbedding token
  let decoderResult := Decoder.runSpec hcfg model route pastTokens 0
    (List.finRange architecture.text.numLayers) (initialDepth embedding) states mlaScoreScale
  finishDecoder hcfg model pastTokens decoderResult

/-- Complete token-step semantics with sparse routes computed at the layers where they are used. -/
noncomputable def runSpecAuto (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (pastTokens : Nat) (token : Fin architecture.text.vocabSize)
    (states : TorchLean.TensorPack ℝ
      (Decoder.StateShapesFor pastTokens model (List.finRange architecture.text.numLayers)))
    (mlaScoreScale : ℝ) : TorchLean.TensorPack ℝ (Outputs pastTokens model) :=
  let embedding := Spec.get model.tokenEmbedding token
  let decoderResult := Decoder.runSpecAuto hcfg model pastTokens 0
    (List.finRange architecture.text.numLayers) (initialDepth embedding) states mlaScoreScale
  finishDecoder hcfg model pastTokens decoderResult

/-- Route-trace condition for the complete token graph at a concrete token and causal state. -/
noncomputable def RoutesAgree (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) (token : Fin architecture.text.vocabSize)
    (states : TorchLean.TensorPack ℝ
      (Decoder.StateShapesFor pastTokens model (List.finRange architecture.text.numLayers)))
    (mlaScoreScale : ℝ) : Prop :=
  Decoder.RoutesAgree hcfg model route pastTokens 0
    (List.finRange architecture.text.numLayers)
    (initialDepth (Spec.get model.tokenEmbedding token)) states mlaScoreScale

/-- Checked route traces preserve the complete automatically routed token-step semantics. -/
theorem runSpec_eq_runSpecAuto_of_routesAgree (hcfg : architecture.WF)
    (model : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) (token : Fin architecture.text.vocabSize)
    (states : TorchLean.TensorPack ℝ
      (Decoder.StateShapesFor pastTokens model (List.finRange architecture.text.numLayers)))
    (mlaScoreScale : ℝ)
    (hRoutes : RoutesAgree hcfg model route pastTokens token states mlaScoreScale) :
    runSpec hcfg model route pastTokens token states mlaScoreScale =
      runSpecAuto hcfg model pastTokens token states mlaScoreScale := by
  apply congrArg (finishDecoder hcfg model pastTokens)
  exact Decoder.runSpec_eq_runSpecAuto_of_routesAgree hcfg model route pastTokens 0
    (List.finRange architecture.text.numLayers)
    (initialDepth (Spec.get model.tokenEmbedding token)) states mlaScoreScale hRoutes

/-- Shape-correct default tensors for the complete token-step graph.

These values support graph construction and compilation. A checkpoint supplies the learned values
through the same `Params` ABI.
-/
def initialParams (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) : TorchLean.TensorPack Float (Params languageModel) :=
  .cons (Spec.fill 0
      (.dim architecture.text.vocabSize (.dim architecture.text.hiddenDim .scalar))) <|
    TorchLean.TensorPack.append
      (ss₁ := Decoder.ParamsFor languageModel (List.finRange architecture.text.numLayers))
      (ss₂ :=
        [.dim architecture.text.hiddenDim .scalar,
          .dim architecture.text.hiddenDim .scalar,
          .dim architecture.text.hiddenDim (.dim architecture.text.vocabSize .scalar)])
      (Decoder.fullModel hcfg languageModel route pastTokens).initParams <|
        .cons (Spec.fill 0 (.dim architecture.text.hiddenDim .scalar)) <|
          .cons (Spec.fill 0 (.dim architecture.text.hiddenDim .scalar)) <|
            .cons (Spec.fill 0
              (.dim architecture.text.hiddenDim
                (.dim architecture.text.vocabSize .scalar))) .nil

def embeddingTableTerm {Γ : List Shape}
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (params : Args Γ (Params languageModel)) :
    Term Γ (.dim architecture.text.vocabSize (.dim architecture.text.hiddenDim .scalar)) :=
  Args.get params Var.head

def decoderParameterTerms {Γ : List Shape}
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (params : Args Γ (Params languageModel)) :
    Args Γ (Decoder.ParamsFor languageModel (List.finRange architecture.text.numLayers)) :=
  match params with
  | .cons _ rest => (Args.splitAppend rest).1

def finalParameterTerms {Γ : List Shape}
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (params : Args Γ (Params languageModel)) : Args Γ
      [.dim architecture.text.hiddenDim .scalar,
        .dim architecture.text.hiddenDim .scalar,
        .dim architecture.text.hiddenDim (.dim architecture.text.vocabSize .scalar)] :=
  match params with
  | .cons _ rest => (Args.splitAppend rest).2

def decoderInputTerms {Γ : List Shape} (pastTokens : Nat)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (embedding : Term Γ (.dim architecture.text.hiddenDim .scalar))
    (allInputs : Args Γ (Inputs pastTokens languageModel)) :
    Args Γ (Decoder.InputsFor pastTokens 0 languageModel
      (List.finRange architecture.text.numLayers)) :=
  let stateAndRest := Args.splitAppend
    (left := Decoder.StateShapesFor pastTokens languageModel
      (List.finRange architecture.text.numLayers))
    (right :=
      Decoder.ControlShapesFor languageModel (List.finRange architecture.text.numLayers) ++
        [.scalar, .scalar, .scalar])
    allInputs
  let controlAndShared := Args.splitAppend
    (left := Decoder.ControlShapesFor languageModel
      (List.finRange architecture.text.numLayers))
    (right := [.scalar, .scalar, .scalar]) stateAndRest.2
  Args.append (initialDepthTerms embedding) <|
    Args.append stateAndRest.1 <| Args.append controlAndShared.1 controlAndShared.2

theorem eval_decoderParameterTerms {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (params : Args Γ (Params languageModel)) :
    Term.evalArgs env (decoderParameterTerms languageModel params) =
      (TorchLean.TensorPack.split
        (match Term.evalArgs env params with
        | .cons _ rest => rest)).1 := by
  cases params with
  | cons table rest =>
      exact Term.evalArgs_splitAppend_fst env rest

theorem eval_finalParameterTerms {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (params : Args Γ (Params languageModel)) :
    Term.evalArgs env (finalParameterTerms languageModel params) =
      (TorchLean.TensorPack.split
        (match Term.evalArgs env params with
        | .cons _ rest => rest)).2 := by
  cases params with
  | cons table rest =>
      exact Term.evalArgs_splitAppend_snd env rest

theorem eval_decoderInputTerms {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (pastTokens : Nat) (languageModel : LanguageModel ℝ architecture.text decayRank)
    (embedding : Term Γ (.dim architecture.text.hiddenDim .scalar))
    (allInputs : Args Γ (Inputs pastTokens languageModel)) :
    Term.evalArgs env (decoderInputTerms pastTokens languageModel embedding allInputs) =
      let stateAndRest := TorchLean.TensorPack.split
        (Term.evalArgs env allInputs)
      let controlAndShared := TorchLean.TensorPack.split stateAndRest.2
      TorchLean.TensorPack.append (initialDepth (Term.eval env embedding)) <|
        TorchLean.TensorPack.append stateAndRest.1 <|
          TorchLean.TensorPack.append controlAndShared.1 controlAndShared.2 := by
  simp only [decoderInputTerms, Term.evalArgs_append, eval_initialDepthTerms,
    Term.evalArgs_splitAppend_fst, Term.evalArgs_splitAppend_snd]

/-- Complete typed DAG for one autoregressive token step, from embedding lookup to logits. -/
def graph (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) (token : Fin architecture.text.vocabSize) :
    NN.GraphSpec.DAG.MultiModel (Params languageModel) (Inputs pastTokens languageModel)
    (Outputs pastTokens languageModel) :=
  { initParams := initialParams hcfg languageModel route pastTokens
    body := by
      let all := Args.vars (Params languageModel ++ Inputs pastTokens languageModel)
      let parameterAndInput := Args.splitAppend
        (left := Params languageModel) (right := Inputs pastTokens languageModel) all
      let decoderParameters := decoderParameterTerms languageModel parameterAndInput.1
      let finalParameters := finalParameterTerms languageModel parameterAndInput.1
      let embedding := StableLatentMoE.selectLeadingTerm architecture.text.vocabSize
        (.dim architecture.text.hiddenDim .scalar) token
        (embeddingTableTerm languageModel parameterAndInput.1)
      let decoderInputs := decoderInputTerms pastTokens languageModel embedding parameterAndInput.2
      let decoder := Decoder.fullModel hcfg languageModel route pastTokens
      let decoderBlock := Block.castOutputs (outputsFor_full pastTokens languageModel)
        (decoder.inline decoderParameters decoderInputs)
      exact decoderBlock.andThen <|
        let decoderOutputs :=
          Decoder.NextStateShapesFor pastTokens languageModel
              (List.finRange architecture.text.numLayers) ++
            DepthSchedule.Shapes architecture.text.attnResBlockSize
              architecture.text.numLayers architecture.text.hiddenDim
        let extended := Params languageModel ++ Inputs pastTokens languageModel
        let decoderResults : Args (extended ++ decoderOutputs) decoderOutputs :=
          Args.rename (Var.inRight extended) (Args.vars decoderOutputs)
        let resultParts := Args.splitAppend decoderResults
        let liftOriginal := fun {shape : Shape} (term : Term extended shape) =>
          Term.weakenAppend decoderOutputs term
        let finalQuery := liftOriginal (Args.get finalParameters Var.head)
        let finalNormScale := liftOriginal (Args.get finalParameters (.tail .head))
        let vocabularyHead := liftOriginal
          (Args.get finalParameters (.tail (.tail .head)))
        let sources := finalSourcesTerm resultParts.2
        let retrieved := AttnRes.term finalSourceCount architecture.text.hiddenDim
          hcfg.hiddenDim_pos finalQuery sources
        let hidden := Term.op
          (NN.GraphSpec.DAG.PrimOp.rmsNorm .scalar architecture.text.hiddenDim
            hcfg.hiddenDim_pos)
          (.cons retrieved (.cons finalNormScale .nil))
        let logits := Term.op
          (NN.GraphSpec.DAG.PrimOp.vecMat architecture.text.hiddenDim
            architecture.text.vocabSize)
          (.cons hidden (.cons vocabularyHead .nil))
        Block.ret (Args.append resultParts.1 (.cons logits .nil)) }

/-- The complete token DAG has exactly the mathematical token-step semantics. -/
theorem graph_specFwd_eq_runSpec (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) (token : Fin architecture.text.vocabSize)
    (states : TorchLean.TensorPack ℝ
      (Decoder.StateShapesFor pastTokens languageModel
        (List.finRange architecture.text.numLayers)))
    (mlaScoreScale : ℝ) :
    (graph hcfg languageModel route pastTokens token).specFwd
        (parameters languageModel) (inputs pastTokens languageModel states mlaScoreScale) =
      runSpec hcfg languageModel route pastTokens token states mlaScoreScale := by
  simp only [NN.GraphSpec.DAG.MultiModel.specFwd, graph, Params, Inputs, Outputs,
    parameters, inputs, Decoder.InputsFor]
  rw [Block.eval_andThen]
  rw [Block.eval_castOutputs]
  rw [NN.GraphSpec.DAG.MultiModel.eval_inline]
  rw [eval_decoderParameterTerms, eval_decoderInputTerms]
  simp only [Term.evalArgs_splitAppend_fst, Term.evalArgs_splitAppend_snd,
    Term.evalArgs_vars, TorchLean.TensorPack.split_append,
    Term.eval, Term.evalArgs, Term.eval_get, Env.tget, embeddingTableTerm,
    StableLatentMoE.eval_selectLeadingTerm]
  have hDecoder :
      (Decoder.fullModel hcfg languageModel route pastTokens).specFwd
          (Decoder.parametersFor languageModel (List.finRange architecture.text.numLayers))
          (TorchLean.TensorPack.append
            (initialDepth (Spec.get languageModel.tokenEmbedding token)) <|
              TorchLean.TensorPack.append states <|
                TorchLean.TensorPack.append
                  (Decoder.controlsFor languageModel mlaScoreScale
                    (List.finRange architecture.text.numLayers)) <|
                    .cons (.scalar Numbers.epsilon) <|
                      .cons (.scalar architecture.text.situGateCap) <|
                        .cons (.scalar architecture.text.situUpCap) .nil) =
        Decoder.runSpec hcfg languageModel route pastTokens 0
          (List.finRange architecture.text.numLayers)
          (initialDepth (Spec.get languageModel.tokenEmbedding token)) states mlaScoreScale := by
    exact Decoder.model_specFwd_eq_runSpec hcfg languageModel route pastTokens 0
      (List.finRange architecture.text.numLayers)
      (initialDepth (Spec.get languageModel.tokenEmbedding token)) states mlaScoreScale
  rw [hDecoder]
  simp only [Block.eval, Term.evalArgs_append, Term.evalArgs,
    Term.evalArgs_rename_inRight, Term.evalArgs_vars, Term.eval_weakenAppend,
    eval_finalParameterTerms, Term.evalArgs_splitAppend_fst,
    Term.evalArgs_splitAppend_snd, TorchLean.TensorPack.split_append,
    Term.eval, Term.eval_get, Env.tget, eval_finalSourcesTerm, AttnRes.eval_term,
    NN.GraphSpec.DAG.PrimOp.rmsNorm_specFwd,
    NN.GraphSpec.DAG.PrimOp.vecMat]
  rw [runSpec]
  rw [GraphSpec.rmsNormSemantics_scalar_eq_scale]
  rfl

/-- The complete token DAG refines K3's automatically routed semantics for a valid route trace.

This is the end-to-end routing statement: the graph begins with token embedding lookup, checks each
sparse layer against the scores computed from that layer's actual hidden representation, and ends at
the vocabulary logits. No claim about routing is delegated to an arbitrary caller-provided choice.
-/
theorem graph_specFwd_eq_runSpecAuto (hcfg : architecture.WF)
    (languageModel : LanguageModel ℝ architecture.text decayRank)
    (route : ∀ _index : Fin architecture.text.numLayers,
      Route architecture.text.numRoutedExperts architecture.text.activeExperts)
    (pastTokens : Nat) (token : Fin architecture.text.vocabSize)
    (states : TorchLean.TensorPack ℝ
      (Decoder.StateShapesFor pastTokens languageModel
        (List.finRange architecture.text.numLayers)))
    (mlaScoreScale : ℝ)
    (hRoutes : RoutesAgree hcfg languageModel route pastTokens token states mlaScoreScale) :
    (graph hcfg languageModel route pastTokens token).specFwd
        (parameters languageModel) (inputs pastTokens languageModel states mlaScoreScale) =
      runSpecAuto hcfg languageModel pastTokens token states mlaScoreScale := by
  rw [graph_specFwd_eq_runSpec]
  exact runSpec_eq_runSpecAuto_of_routesAgree hcfg languageModel route pastTokens token states
    mlaScoreScale hRoutes

end TokenStep

end GraphSpec
end KimiK3
