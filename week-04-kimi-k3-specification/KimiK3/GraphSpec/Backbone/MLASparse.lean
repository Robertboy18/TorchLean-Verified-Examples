/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Backbone.Depth
public import KimiK3.GraphSpec.MoE
public import KimiK3.GraphSpec.MLA

/-!
# Sparse MLA backbone graph

Compose the fixed-cache MLA transition with the shared AttnRes depth interface and a
route-specialized Stable LatentMoE. The denotation theorem identifies every returned state and
output tensor.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec

open Spec TorchLean
open TorchLean.Tensor
open NN.GraphSpec.DAG
open Runtime.Autograd.Torch

namespace Backbone

namespace MLASparse

/-- Parameters of one fixed-cache MLA layer followed by depth controls and Stable LatentMoE. -/
abbrev Params (modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim
    latentDim expertHidden numShared numRouted : Nat) : List Shape :=
  MLA.LayerParams modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim ++
    (DepthParams modelDim ++
      StableLatentMoE.Params modelDim latentDim expertHidden expertHidden numShared numRouted)

/-- Packed depth memory, MLA caches, score scale, and SiTU caps for one token. -/
abbrev Inputs (pastTokens completedCount modelDim kvLatentDim sharedKeyDim : Nat) : List Shape :=
  DepthInputs completedCount modelDim ++
    [ [pastTokens, kvLatentDim],
      [pastTokens, sharedKeyDim], [], [], [] ]

/-- Updated MLA caches, both submodule outputs, and the new AttnRes partial sum. -/
abbrev Outputs (pastTokens modelDim kvLatentDim sharedKeyDim : Nat) : List Shape :=
  [ [pastTokens + 1, kvLatentDim],
    [pastTokens + 1, sharedKeyDim],
    [modelDim], [modelDim], [modelDim] ]

/-- Sparse MLA decoder-layer transition for a fixed expert route. -/
def model (pastTokens completedCount modelDim heads queryLatentDim kvLatentDim contentKeyDim
    sharedKeyDim valueDim latentDim expertHidden numShared numRouted activeExperts : Nat)
    (route : Route numRouted activeExperts) (partialPresent : Bool)
    (hCompleted : 0 < completedCount) (hModel : 0 < modelDim)
    (hQueryLatent : 0 < queryLatentDim) (hKVLatent : 0 < kvLatentDim)
    (hLatent : 0 < latentDim) :
    NN.GraphSpec.DAG.MultiModel
      (Params modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim
        latentDim expertHidden numShared numRouted)
      (Inputs pastTokens completedCount modelDim kvLatentDim sharedKeyDim)
      (Outputs pastTokens modelDim kvLatentDim sharedKeyDim) :=
  let mixerParams :=
    MLA.LayerParams modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim
  let depthParams := DepthParams modelDim
  let channelParams :=
    StableLatentMoE.Params modelDim latentDim expertHidden expertHidden numShared numRouted
  let depthIns := DepthInputs completedCount modelDim
  let stateIns : List Shape :=
    [ [pastTokens, kvLatentDim],
      [pastTokens, sharedKeyDim], [], [], [] ]
  let params := mixerParams ++ (depthParams ++ channelParams)
  let ins := depthIns ++ stateIns
  let Γ := params ++ ins
  let mixerArgs : Args Γ mixerParams :=
    Args.rename (Var.inLeft ins) <|
      Args.rename (Var.inLeft (depthParams ++ channelParams)) (Args.vars mixerParams)
  let depthArgs : Args Γ depthParams :=
    Args.rename (Var.inLeft ins) <| Args.rename (Var.inRight mixerParams) <|
      Args.rename (Var.inLeft channelParams) (Args.vars depthParams)
  let channelArgs : Args Γ channelParams :=
    Args.rename (Var.inLeft ins) <| Args.rename (Var.inRight mixerParams) <|
      Args.rename (Var.inRight depthParams) (Args.vars channelParams)
  let depthInputArgs : Args Γ depthIns :=
    Args.rename (Var.inRight params) <|
      Args.rename (Var.inLeft stateIns) (Args.vars depthIns)
  let stateArgs : Args Γ stateIns :=
    Args.rename (Var.inRight params) <|
      Args.rename (Var.inRight depthIns) (Args.vars stateIns)
  let sequenceQuery := Args.get depthArgs .head
  let sequenceScale := Args.get depthArgs (.tail .head)
  let channelQuery := Args.get depthArgs (.tail (.tail .head))
  let channelScale := Args.get depthArgs (.tail (.tail (.tail .head)))
  let completed := Args.get depthInputArgs .head
  let partialState := Args.get depthInputArgs (.tail .head)
  let pastLatentCache := Args.get stateArgs .head
  let pastSharedKeyCache := Args.get stateArgs (.tail .head)
  let scoreScale := Args.get stateArgs (.tail (.tail .head))
  let gateCap := Args.get stateArgs (.tail (.tail (.tail .head)))
  let upCap := Args.get stateArgs (.tail (.tail (.tail (.tail .head))))
  let sequenceInput := PackedDepth.sequenceInputTerm completedCount modelDim partialPresent
    hCompleted hModel sequenceQuery sequenceScale completed partialState
  let mixer := MLA.stepModel pastTokens modelDim heads queryLatentDim kvLatentDim contentKeyDim
    sharedKeyDim valueDim hQueryLatent hKVLatent
  let mixerBlock := mixer.inline mixerArgs
    (.cons pastLatentCache <| .cons pastSharedKeyCache <| .cons sequenceInput <|
      .cons scoreScale .nil)
  let mixerOuts := MLA.StepOutputs pastTokens modelDim kvLatentDim sharedKeyDim
  let Γ' := Γ ++ mixerOuts
  let original {s : Shape} (term : Term Γ s) : Term Γ' s := Term.weakenAppend mixerOuts term
  let nextLatentCache : Term Γ' [pastTokens + 1, kvLatentDim] :=
    Term.var (Var.inRight Γ .head)
  let nextSharedKeyCache : Term Γ' [pastTokens + 1, sharedKeyDim] :=
    Term.var (Var.inRight Γ (.tail .head))
  let sequenceOutput : Term Γ' [modelDim] :=
    Term.var (Var.inRight Γ (.tail (.tail .head)))
  let channelInput := PackedDepth.channelInputTerm completedCount modelDim hModel
    (original channelQuery) (original channelScale) (original completed)
    (original partialState) sequenceOutput
  let channelOutput :=
    (StableLatentMoE.modelGivenRoute modelDim latentDim expertHidden expertHidden numShared
      numRouted activeExperts route hLatent).inline
        (Args.rename (Var.inLeft mixerOuts) channelArgs)
        (.cons channelInput (.cons (original gateCap) (.cons (original upCap) .nil)))
  let nextPartial := PackedDepth.finishPartialTerm modelDim (original partialState)
    sequenceOutput channelOutput
  { initParams :=
      TorchLean.TensorPack.append
        (MLA.initialLayerParams modelDim heads queryLatentDim kvLatentDim contentKeyDim
          sharedKeyDim valueDim)
        (TorchLean.TensorPack.append (initialDepthParams modelDim)
          (StableLatentMoE.initialParams modelDim latentDim expertHidden expertHidden numShared
            numRouted))
    body := mixerBlock.andThen <| Block.ret <|
      .cons nextLatentCache <| .cons nextSharedKeyCache <| .cons sequenceOutput <|
        .cons channelOutput <| .cons nextPartial .nil }

/-- Mathematical transition represented by the sparse MLA graph. -/
noncomputable def specStep
    {pastTokens completedCount modelDim heads queryLatentDim kvLatentDim contentKeyDim
      sharedKeyDim valueDim latentDim expertHidden numShared numRouted activeExperts : Nat}
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < modelDim)
    (layer : GatedMLA ℝ modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim
      valueDim)
    (moe : KimiK3.StableLatentMoE ℝ modelDim latentDim expertHidden expertHidden numShared
      numRouted activeExperts)
    (route : Route numRouted activeExperts)
    (sequenceQuery sequenceScale channelQuery channelScale : Tensor ℝ [modelDim])
    (completed : Tensor ℝ [completedCount, modelDim])
    (partialState : Tensor ℝ [modelDim])
    (pastLatentCache : Tensor ℝ [pastTokens, kvLatentDim])
    (pastSharedKeyCache : Tensor ℝ [pastTokens, sharedKeyDim])
    (scoreScale gateCap upCap : ℝ) :
    TorchLean.TensorPack ℝ (Outputs pastTokens modelDim kvLatentDim sharedKeyDim) :=
  let sequenceInput := PackedDepth.sequenceInput partialPresent hCompleted hModel sequenceQuery
    sequenceScale completed partialState
  let sequenceResult := layer.stepFixed pastTokens pastLatentCache
    pastSharedKeyCache sequenceInput scoreScale
  let channelInput := PackedDepth.channelInput hModel channelQuery channelScale completed
    partialState sequenceResult.2.2
  let channelOutput := moe.forward route gateCap upCap channelInput
  let nextPartial := PackedDepth.finishPartial partialState sequenceResult.2.2 channelOutput
  .cons sequenceResult.1 <| .cons sequenceResult.2.1 <| .cons sequenceResult.2.2 <|
    .cons channelOutput <| .cons nextPartial .nil

/-- Pack MLA, depth-control, and sparse-channel parameters in graph order. -/
def parameters {cfg : TextConfig} {decayRank : Nat}
    (backbone : BackboneLayer ℝ cfg decayRank)
    (mla : GatedMLA ℝ cfg.hiddenDim cfg.numHeads cfg.queryLatentDim cfg.kvLatentDim
      cfg.qkNopeHeadDim cfg.qkReservedHeadDim cfg.valueHeadDim)
    (moe : KimiK3.StableLatentMoE ℝ cfg.hiddenDim cfg.routedLatentDim
      cfg.routedExpertHiddenDim cfg.routedExpertHiddenDim cfg.numSharedExperts
      cfg.numRoutedExperts cfg.activeExperts) :
    TorchLean.TensorPack ℝ
      (Params cfg.hiddenDim cfg.numHeads cfg.queryLatentDim cfg.kvLatentDim
        cfg.qkNopeHeadDim cfg.qkReservedHeadDim cfg.valueHeadDim cfg.routedLatentDim
        cfg.routedExpertHiddenDim cfg.numSharedExperts cfg.numRoutedExperts) :=
  TorchLean.TensorPack.append (MLA.layerParameters mla)
    (TorchLean.TensorPack.append (depthParameters backbone)
      (StableLatentMoE.parameters moe))

/-- Package one sparse MLA decoder state in graph order. -/
def inputs {cfg : TextConfig} {pastTokens completedCount : Nat}
    (completed : Tensor ℝ [completedCount, cfg.hiddenDim])
    (partialState : Tensor ℝ [cfg.hiddenDim])
    (pastLatentCache : Tensor ℝ [pastTokens, cfg.kvLatentDim])
    (pastSharedKeyCache : Tensor ℝ
      [pastTokens, cfg.qkReservedHeadDim])
    (scoreScale gateCap upCap : ℝ) :
    TorchLean.TensorPack ℝ
      (Inputs pastTokens completedCount cfg.hiddenDim cfg.kvLatentDim
        cfg.qkReservedHeadDim) :=
  TorchLean.TensorPack.append (depthInputs completed partialState) <|
    .cons pastLatentCache <| .cons pastSharedKeyCache <| .cons (Tensor.scalar scoreScale) <|
      .cons (Tensor.scalar gateCap) <| .cons (Tensor.scalar upCap) .nil

/-- The sparse MLA graph computes `specStep` for the supplied route. -/
theorem model_specFwd_eq_specStep
    {cfg : TextConfig} {pastTokens completedCount decayRank : Nat}
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hQueryLatent : 0 < cfg.queryLatentDim) (hKVLatent : 0 < cfg.kvLatentDim)
    (hLatent : 0 < cfg.routedLatentDim) (hActive : 0 < cfg.activeExperts)
    (backbone : BackboneLayer ℝ cfg decayRank)
    (mla : GatedMLA ℝ cfg.hiddenDim cfg.numHeads cfg.queryLatentDim cfg.kvLatentDim
      cfg.qkNopeHeadDim cfg.qkReservedHeadDim cfg.valueHeadDim)
    (moe : KimiK3.StableLatentMoE ℝ cfg.hiddenDim cfg.routedLatentDim
      cfg.routedExpertHiddenDim cfg.routedExpertHiddenDim cfg.numSharedExperts
      cfg.numRoutedExperts cfg.activeExperts)
    (route : Route cfg.numRoutedExperts cfg.activeExperts)
    (completed : Tensor ℝ [completedCount, cfg.hiddenDim])
    (partialState : Tensor ℝ [cfg.hiddenDim])
    (pastLatentCache : Tensor ℝ [pastTokens, cfg.kvLatentDim])
    (pastSharedKeyCache : Tensor ℝ
      [pastTokens, cfg.qkReservedHeadDim])
    (scoreScale gateCap upCap : ℝ) :
    (model pastTokens completedCount cfg.hiddenDim cfg.numHeads cfg.queryLatentDim
      cfg.kvLatentDim cfg.qkNopeHeadDim cfg.qkReservedHeadDim cfg.valueHeadDim
      cfg.routedLatentDim cfg.routedExpertHiddenDim cfg.numSharedExperts cfg.numRoutedExperts
      cfg.activeExperts route partialPresent hCompleted hModel hQueryLatent hKVLatent
      hLatent).specFwd (parameters backbone mla moe)
        (inputs completed partialState pastLatentCache pastSharedKeyCache scoreScale gateCap
          upCap) =
      specStep partialPresent hCompleted hModel mla moe route
        backbone.sequenceQuery backbone.sequenceNormScale backbone.channelQuery
        backbone.channelNormScale completed partialState pastLatentCache pastSharedKeyCache
        scoreScale gateCap upCap := by
  simp only [NN.GraphSpec.DAG.MultiModel.specFwd, model]
  rw [Block.eval_andThen]
  rw [NN.GraphSpec.DAG.MultiModel.eval_inline]
  simp only [parameters, inputs]
  simp only [Args.get_rename, Args.get_vars, Term.evalArgs_rename_inLeft,
    Term.evalArgs_vars, Term.eval_rename_inRight, Term.evalArgs]
  simp only [depthParameters, depthInputs, Term.eval, Env.tget,
    PackedDepth.eval_sequenceInputTerm]
  simp only [Term.eval_rename_inLeft, Term.eval_rename_inRight]
  simp only [Term.eval, Env.tget]
  let sequenceInput := PackedDepth.sequenceInput partialPresent hCompleted hModel
    backbone.sequenceQuery backbone.sequenceNormScale completed partialState
  have hMixer :
      (MLA.stepModel pastTokens cfg.hiddenDim cfg.numHeads cfg.queryLatentDim cfg.kvLatentDim
        cfg.qkNopeHeadDim cfg.qkReservedHeadDim cfg.valueHeadDim hQueryLatent
        hKVLatent).specFwd (MLA.layerParameters mla)
          (.cons pastLatentCache <| .cons pastSharedKeyCache <| .cons sequenceInput <|
            .cons (Tensor.scalar scoreScale) .nil) =
        let result := mla.stepFixed pastTokens pastLatentCache
          pastSharedKeyCache sequenceInput scoreScale
        .cons result.1 (.cons result.2.1 (.cons result.2.2 .nil)) := by
    simpa only [MLA.stepGraphOutputs, MLA.stepFixedOutputs, MLA.stepInputs] using
      MLA.stepModel_specFwd_eq_stepFixed hQueryLatent hKVLatent mla pastLatentCache
        pastSharedKeyCache sequenceInput scoreScale
  let sequenceResult := mla.stepFixed pastTokens pastLatentCache
    pastSharedKeyCache sequenceInput scoreScale
  let channelInput := PackedDepth.channelInput hModel backbone.channelQuery
    backbone.channelNormScale completed partialState sequenceResult.2.2
  have hChannel :
      (StableLatentMoE.modelGivenRoute cfg.hiddenDim cfg.routedLatentDim
        cfg.routedExpertHiddenDim cfg.routedExpertHiddenDim cfg.numSharedExperts
        cfg.numRoutedExperts cfg.activeExperts route hLatent).specFwd
          (StableLatentMoE.parameters moe)
          (.cons channelInput <| .cons (Tensor.scalar gateCap) <| .cons (Tensor.scalar upCap) .nil) =
        moe.forward route gateCap upCap channelInput := by
    simpa only [StableLatentMoE.inputs] using
      StableLatentMoE.modelGivenRoute_specFwd_eq_forward hLatent hActive moe route channelInput
        gateCap upCap
  change Block.eval _ _ = _
  rw [hMixer]
  simp only [Block.eval, Term.evalArgs, Term.eval_weakenAppend, Term.eval,
    Env.tget_append_inRight, PackedDepth.eval_channelInputTerm,
    PackedDepth.eval_finishPartialTerm, NN.GraphSpec.DAG.Model.eval_inline,
    Term.evalArgs_rename_inLeft, Term.evalArgs_rename_inRight, Term.evalArgs_vars,
    Term.eval_rename_inLeft, Term.eval_rename_inRight, Env.tget, specStep]
  rw [hChannel]

end MLASparse

end Backbone
end GraphSpec
end KimiK3
