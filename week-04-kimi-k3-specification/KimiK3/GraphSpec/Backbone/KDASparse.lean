/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Backbone.Depth
public import KimiK3.GraphSpec.MoE
public import KimiK3.GraphSpec.KDA

/-!
# Sparse KDA backbone graph

Compose the KDA recurrence and convolution window with the shared AttnRes depth interface and a
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

namespace KDASparse

/-- Parameters of one KDA layer followed by its depth controls and Stable LatentMoE. -/
abbrev Params (modelDim heads keyDim valueDim convWidth decayRank latentDim expertHidden
    numShared numRouted : Nat) : List Shape :=
  KDA.LayerParams modelDim heads keyDim valueDim convWidth decayRank ++
    (DepthParams modelDim ++
      StableLatentMoE.Params modelDim latentDim expertHidden expertHidden numShared numRouted)

/-- Runtime inputs of a sparse KDA layer. The route is not among them: it is a static argument of
the graph builder. -/
abbrev Inputs (completedCount modelDim heads keyDim valueDim convWidth : Nat) : List Shape :=
  DepthInputs completedCount modelDim ++
    [ [convWidth, modelDim],
      [heads, keyDim, valueDim],
      [], [], [], [] ]

/-- Updated KDA state, both submodule outputs, and the new AttnRes partial sum. -/
abbrev Outputs (modelDim heads keyDim valueDim convWidth : Nat) : List Shape :=
  [ [convWidth, modelDim],
    [heads, keyDim, valueDim],
    [modelDim], [modelDim], [modelDim] ]

/-- Sparse KDA decoder-layer transition for a fixed expert route. -/
def model (completedCount modelDim heads keyDim valueDim convWidth decayRank latentDim
    expertHidden numShared numRouted activeExperts : Nat)
    (route : Route numRouted activeExperts) (partialPresent : Bool)
    (hCompleted : 0 < completedCount) (hModel : 0 < modelDim) (hHeads : 0 < heads)
    (hKey : 0 < keyDim) (hValue : 0 < valueDim) (hWidth : 0 < convWidth)
    (hLatent : 0 < latentDim) :
    NN.GraphSpec.DAG.MultiModel
      (Params modelDim heads keyDim valueDim convWidth decayRank latentDim expertHidden
        numShared numRouted)
      (Inputs completedCount modelDim heads keyDim valueDim convWidth)
      (Outputs modelDim heads keyDim valueDim convWidth) :=
  let mixerParams := KDA.LayerParams modelDim heads keyDim valueDim convWidth decayRank
  let depthParams := DepthParams modelDim
  let channelParams :=
    StableLatentMoE.Params modelDim latentDim expertHidden expertHidden numShared numRouted
  let depthIns := DepthInputs completedCount modelDim
  let stateIns : List Shape :=
    [ [convWidth, modelDim],
      [heads, keyDim, valueDim],
      [], [], [], [] ]
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
  let previousWindow := Args.get stateArgs .head
  let previousState := Args.get stateArgs (.tail .head)
  let logFloor := Args.get stateArgs (.tail (.tail .head))
  let epsilon := Args.get stateArgs (.tail (.tail (.tail .head)))
  let gateCap := Args.get stateArgs (.tail (.tail (.tail (.tail .head))))
  let upCap := Args.get stateArgs (.tail (.tail (.tail (.tail (.tail .head)))))
  let sequenceInput := PackedDepth.sequenceInputTerm completedCount modelDim partialPresent
    hCompleted hModel sequenceQuery sequenceScale completed partialState
  let nextWindow := KDA.rollWindowTerm modelDim convWidth hWidth sequenceInput previousWindow
  let mixer := KDA.layerModel modelDim heads keyDim valueDim convWidth decayRank
    hModel hHeads hKey hValue hWidth
  let mixerBlock := mixer.inline mixerArgs
    (.cons nextWindow (.cons previousState (.cons logFloor (.cons epsilon .nil))))
  let mixerOuts := KDA.LayerOutputs modelDim heads keyDim valueDim
  let Γ' := Γ ++ mixerOuts
  let original {s : Shape} (term : Term Γ s) : Term Γ' s := Term.weakenAppend mixerOuts term
  let nextState : Term Γ' [heads, keyDim, valueDim] :=
    Term.var (Var.inRight Γ .head)
  let sequenceOutput : Term Γ' [modelDim] :=
    Term.var (Var.inRight Γ (.tail .head))
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
        (KDA.initialLayerParams modelDim heads keyDim valueDim convWidth decayRank)
        (TorchLean.TensorPack.append (initialDepthParams modelDim)
          (StableLatentMoE.initialParams modelDim latentDim expertHidden expertHidden numShared
            numRouted))
    body := mixerBlock.andThen <| Block.ret <|
      .cons (original nextWindow) <| .cons nextState <| .cons sequenceOutput <|
        .cons channelOutput <| .cons nextPartial .nil }

/-- Mathematical transition represented by the sparse KDA graph. -/
noncomputable def specStep
    {completedCount modelDim heads keyDim valueDim convWidth decayRank latentDim expertHidden
      numShared numRouted activeExperts : Nat}
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < modelDim)
    (hWidth : 0 < convWidth)
    (layer : KDA.Layer ℝ modelDim heads keyDim valueDim convWidth decayRank)
    (moe : KimiK3.StableLatentMoE ℝ modelDim latentDim expertHidden expertHidden numShared
      numRouted activeExperts)
    (route : Route numRouted activeExperts)
    (sequenceQuery sequenceScale channelQuery channelScale : Tensor ℝ [modelDim])
    (completed : Tensor ℝ [completedCount, modelDim])
    (partialState : Tensor ℝ [modelDim])
    (previousWindow : Tensor ℝ [convWidth, modelDim])
    (previousState : KDA.Layer.State ℝ heads keyDim valueDim)
    (logFloor gateCap upCap : ℝ) :
    TorchLean.TensorPack ℝ (Outputs modelDim heads keyDim valueDim convWidth) :=
  let sequenceInput := PackedDepth.sequenceInput partialPresent hCompleted hModel sequenceQuery
    sequenceScale completed partialState
  let nextWindow := KDA.Layer.rollWindow (α := ℝ) hWidth sequenceInput previousWindow
  let sequenceResult := layer.stepWindow hWidth logFloor nextWindow previousState
  let channelInput := PackedDepth.channelInput hModel channelQuery channelScale completed
    partialState sequenceResult.2
  let channelOutput := moe.forward route gateCap upCap channelInput
  let nextPartial := PackedDepth.finishPartial partialState sequenceResult.2 channelOutput
  .cons nextWindow <| .cons sequenceResult.1 <| .cons sequenceResult.2 <|
    .cons channelOutput <| .cons nextPartial .nil

/-- Pack the KDA, depth-control, and sparse-channel parameters in graph order. -/
def parameters
    {cfg : TextConfig} {decayRank : Nat}
    (layer : BackboneLayer ℝ cfg decayRank)
    (kda : KDA.Layer ℝ cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
      cfg.shortConvWidth decayRank)
    (moe : KimiK3.StableLatentMoE ℝ cfg.hiddenDim cfg.routedLatentDim
      cfg.routedExpertHiddenDim cfg.routedExpertHiddenDim cfg.numSharedExperts
      cfg.numRoutedExperts cfg.activeExperts) :
    TorchLean.TensorPack ℝ
      (Params cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim cfg.shortConvWidth
        decayRank cfg.routedLatentDim cfg.routedExpertHiddenDim cfg.numSharedExperts
        cfg.numRoutedExperts) :=
  TorchLean.TensorPack.append (KDA.layerParameters kda)
    (TorchLean.TensorPack.append (depthParameters layer)
      (StableLatentMoE.parameters moe))

/-- Package one sparse KDA decoder state in graph order. -/
def inputs
    {cfg : TextConfig} {completedCount : Nat}
    (completed : Tensor ℝ [completedCount, cfg.hiddenDim])
    (partialState : Tensor ℝ [cfg.hiddenDim])
    (previousWindow : Tensor ℝ [cfg.shortConvWidth, cfg.hiddenDim])
    (previousState : KDA.Layer.State ℝ cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim)
    (logFloor epsilon gateCap upCap : ℝ) :
    TorchLean.TensorPack ℝ
      (Inputs completedCount cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
        cfg.shortConvWidth) :=
  TorchLean.TensorPack.append (depthInputs completed partialState) <|
    .cons previousWindow <| .cons previousState <| .cons (Tensor.scalar logFloor) <|
      .cons (Tensor.scalar epsilon) <| .cons (Tensor.scalar gateCap) <| .cons (Tensor.scalar upCap) .nil

/-- With `epsilon = Normalize.l2Epsilon`, the sparse KDA graph computes `specStep` for the
supplied route. -/
theorem model_specFwd_eq_specStep
    {cfg : TextConfig} {completedCount decayRank : Nat}
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hHeads : 0 < cfg.numHeads) (hKey : 0 < cfg.kdaHeadDim)
    (hValue : 0 < cfg.kdaValueDim) (hWidth : 0 < cfg.shortConvWidth)
    (hLatent : 0 < cfg.routedLatentDim) (hActive : 0 < cfg.activeExperts)
    (layer : BackboneLayer ℝ cfg decayRank)
    (kda : KDA.Layer ℝ cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
      cfg.shortConvWidth decayRank)
    (moe : KimiK3.StableLatentMoE ℝ cfg.hiddenDim cfg.routedLatentDim
      cfg.routedExpertHiddenDim cfg.routedExpertHiddenDim cfg.numSharedExperts
      cfg.numRoutedExperts cfg.activeExperts)
    (route : Route cfg.numRoutedExperts cfg.activeExperts)
    (completed : Tensor ℝ [completedCount, cfg.hiddenDim])
    (partialState : Tensor ℝ [cfg.hiddenDim])
    (previousWindow : Tensor ℝ [cfg.shortConvWidth, cfg.hiddenDim])
    (previousState : KDA.Layer.State ℝ cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim)
    (logFloor epsilon gateCap upCap : ℝ) (hEpsilon : epsilon = Normalize.l2Epsilon) :
    (model completedCount cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
      cfg.shortConvWidth decayRank cfg.routedLatentDim cfg.routedExpertHiddenDim
      cfg.numSharedExperts cfg.numRoutedExperts cfg.activeExperts route partialPresent hCompleted
      hModel hHeads hKey hValue hWidth hLatent).specFwd
        (parameters layer kda moe)
        (inputs completed partialState previousWindow previousState logFloor epsilon gateCap
          upCap) =
      specStep partialPresent hCompleted hModel hWidth kda moe route layer.sequenceQuery
        layer.sequenceNormScale layer.channelQuery layer.channelNormScale completed partialState
        previousWindow previousState logFloor gateCap upCap := by
  subst epsilon
  simp only [NN.GraphSpec.DAG.MultiModel.specFwd, model]
  rw [Block.eval_andThen]
  rw [NN.GraphSpec.DAG.MultiModel.eval_inline]
  simp only [parameters, inputs]
  simp only [Args.get_rename, Args.get_vars, Term.evalArgs_rename_inLeft,
    Term.evalArgs_vars, Term.eval_rename_inRight, Term.evalArgs]
  simp only [depthParameters, depthInputs, Term.eval, Env.tget,
    PackedDepth.eval_sequenceInputTerm, KDA.eval_rollWindowTerm]
  simp only [Term.eval_rename_inLeft, Term.eval_rename_inRight]
  simp only [Term.eval, Env.tget]
  let sequenceInput := PackedDepth.sequenceInput partialPresent hCompleted hModel
    layer.sequenceQuery layer.sequenceNormScale completed partialState
  let nextWindow := KDA.Layer.rollWindow hWidth sequenceInput previousWindow
  have hMixer :
      (KDA.layerModel cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
        cfg.shortConvWidth decayRank hModel hHeads hKey hValue hWidth).specFwd
          (KDA.layerParameters kda)
          (.cons nextWindow <| .cons previousState <| .cons (Tensor.scalar logFloor) <|
            .cons (Tensor.scalar Normalize.l2Epsilon) .nil) =
        let result := kda.stepWindow hWidth logFloor nextWindow previousState
        .cons result.1 (.cons result.2 .nil) := by
    simpa only [KDA.layerInputs] using
      KDA.layerModel_specFwd_eq_stepWindow hModel hHeads hKey hValue hWidth kda
        nextWindow previousState logFloor
  let sequenceResult := kda.stepWindow hWidth logFloor nextWindow previousState
  let channelInput := PackedDepth.channelInput hModel layer.channelQuery layer.channelNormScale
    completed partialState sequenceResult.2
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
    Env.tget_append_inRight, KDA.eval_rollWindowTerm,
    PackedDepth.eval_sequenceInputTerm, PackedDepth.eval_channelInputTerm,
    PackedDepth.eval_finishPartialTerm, NN.GraphSpec.DAG.Model.eval_inline,
    Term.evalArgs_rename_inLeft, Term.evalArgs_rename_inRight, Term.evalArgs_vars,
    Term.eval_rename_inLeft, Term.eval_rename_inRight, Env.tget, specStep]
  rw [hChannel]

end KDASparse

end Backbone
end GraphSpec
end KimiK3
