/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.AttnRes
public import KimiK3.GraphSpec.Expert
public import KimiK3.GraphSpec.KDA
public import KimiK3.GraphSpec.MLA
public import KimiK3.GraphSpec.MoE
public import KimiK3.Model

/-!
# Composed Kimi K3 backbone graphs

The component graphs become useful only when they are assembled in the order used by a decoder
layer.  This file starts that composition at the causal-token boundary.  A packed depth state is
queried by AttnRes, normalized, passed through a sequence mixer, queried again for the channel
mixer, and finally accumulated into the current AttnRes block.

Top-k routing remains a discrete boundary.  Sparse graphs are parameterized by a `Route` together
with its mathematical `IsTopK` contract; all arithmetic after that decision remains an ordinary
TorchLean graph and therefore retains the usual forward and backward interpretation.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec

open Spec
open Spec.Tensor
open NN.GraphSpec.DAG
open Runtime.Autograd.Torch

namespace PackedDepth

/-- Number of depth representations visible to the sequence mixer. `completedCount` counts the token
embedding and every completed AttnRes block; a nonempty partial block contributes one more row. -/
def sourceCount (completedCount : Nat) (partialPresent : Bool) : Nat :=
  completedCount + if partialPresent then 1 else 0

/-- Pack the depth representations visible before the sequence mixer. -/
def sequenceSources {α : Type} [Context α] {completedCount modelDim : Nat}
    (partialPresent : Bool)
    (completed : Tensor α (.dim completedCount (.dim modelDim .scalar)))
    (partialState : Tensor α (.dim modelDim .scalar)) :
    Tensor α (.dim (sourceCount completedCount partialPresent) (.dim modelDim .scalar)) :=
  match partialPresent with
  | false => completed
  | true =>
      let partialRow : Tensor α (.dim 1 (.dim modelDim .scalar)) :=
        Tensor.reshapeSpec partialState (by simp [Shape.size])
      Tensor.concatAxisSpec .scalar completed partialRow

/-- After the sequence mixer, its output is added to the current partial block and exposed as the
last AttnRes source for the channel mixer. -/
def channelSources {α : Type} [Context α] {completedCount modelDim : Nat}
    (completed : Tensor α (.dim completedCount (.dim modelDim .scalar)))
    (partialState sequenceOutput : Tensor α (.dim modelDim .scalar)) :
    Tensor α (.dim (completedCount + 1) (.dim modelDim .scalar)) :=
  let current := partialState + sequenceOutput
  let currentRow : Tensor α (.dim 1 (.dim modelDim .scalar)) :=
    Tensor.reshapeSpec current (by simp [Shape.size])
  Tensor.concatAxisSpec .scalar completed currentRow

/-- Packed counterpart of `BackboneLayer`'s first AttnRes retrieval and RMS normalization. -/
noncomputable def sequenceInput {completedCount modelDim : Nat} (partialPresent : Bool)
    (hCompleted : 0 < completedCount) (hModel : 0 < modelDim)
    (query scale : Tensor ℝ (.dim modelDim .scalar))
    (completed : Tensor ℝ (.dim completedCount (.dim modelDim .scalar)))
    (partialState : Tensor ℝ (.dim modelDim .scalar)) :
    Tensor ℝ (.dim modelDim .scalar) :=
  let sources := sequenceSources partialPresent completed partialState
  have hSources : 0 < sourceCount completedCount partialPresent := by
    simp [sourceCount]
    omega
  RMSNorm.scale (AttnRes.attendPacked hModel query sources) scale

/-- Packed counterpart of the second AttnRes retrieval and RMS normalization. -/
noncomputable def channelInput {completedCount modelDim : Nat} (hModel : 0 < modelDim)
    (query scale : Tensor ℝ (.dim modelDim .scalar))
    (completed : Tensor ℝ (.dim completedCount (.dim modelDim .scalar)))
    (partialState sequenceOutput : Tensor ℝ (.dim modelDim .scalar)) :
    Tensor ℝ (.dim modelDim .scalar) :=
  let sources := channelSources completed partialState sequenceOutput
  RMSNorm.scale (AttnRes.attendPacked hModel query sources) scale

/-- Accumulate both submodule outputs into the current AttnRes block.  Whether this vector is kept
as a partial block or committed as a completed block is determined statically by the layer index. -/
def finishPartial {α : Type} [Context α] {modelDim : Nat}
    (partialState sequenceOutput channelOutput : Tensor α (.dim modelDim .scalar)) :
    Tensor α (.dim modelDim .scalar) :=
  partialState + sequenceOutput + channelOutput

/-- Graph term that packs the depth sources visible to a sequence mixer. -/
def sequenceSourcesTerm {Γ : List Shape} (completedCount modelDim : Nat)
    (partialPresent : Bool)
    (completed : Term Γ (.dim completedCount (.dim modelDim .scalar)))
    (partialState : Term Γ (.dim modelDim .scalar)) :
    Term Γ (.dim (sourceCount completedCount partialPresent) (.dim modelDim .scalar)) :=
  match partialPresent with
  | false => completed
  | true =>
      let partialRow : Term Γ (.dim 1 (.dim modelDim .scalar)) :=
        Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
          (.cons partialState .nil)
      Term.op (NN.GraphSpec.DAG.PrimOp.concatAxis
          (.dim completedCount (.dim modelDim .scalar)) 0 completedCount 1)
        (.cons completed (.cons partialRow .nil))

/-- Graph term that appends the current sequence residual to completed depth blocks. -/
def channelSourcesTerm {Γ : List Shape} (completedCount modelDim : Nat)
    (completed : Term Γ (.dim completedCount (.dim modelDim .scalar)))
    (partialState sequenceOutput : Term Γ (.dim modelDim .scalar)) :
    Term Γ (.dim (completedCount + 1) (.dim modelDim .scalar)) :=
  let current := Term.op (NN.GraphSpec.DAG.PrimOp.add (.dim modelDim .scalar))
    (.cons partialState (.cons sequenceOutput .nil))
  let currentRow : Term Γ (.dim 1 (.dim modelDim .scalar)) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons current .nil)
  Term.op (NN.GraphSpec.DAG.PrimOp.concatAxis
      (.dim completedCount (.dim modelDim .scalar)) 0 completedCount 1)
    (.cons completed (.cons currentRow .nil))

/-- Graph term for the first AttnRes retrieval and RMS normalization in a decoder layer. -/
def sequenceInputTerm {Γ : List Shape} (completedCount modelDim : Nat)
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < modelDim)
    (query scale : Term Γ (.dim modelDim .scalar))
    (completed : Term Γ (.dim completedCount (.dim modelDim .scalar)))
    (partialState : Term Γ (.dim modelDim .scalar)) : Term Γ (.dim modelDim .scalar) :=
  let sources := sequenceSourcesTerm completedCount modelDim partialPresent completed partialState
  have hSources : 0 < sourceCount completedCount partialPresent := by
    simp [sourceCount]
    omega
  let retrieved := AttnRes.term (sourceCount completedCount partialPresent) modelDim hModel
    query sources
  Term.op (NN.GraphSpec.DAG.PrimOp.rmsNorm .scalar modelDim hModel)
    (.cons retrieved (.cons scale .nil))

/-- Graph term for the second AttnRes retrieval and RMS normalization in a decoder layer. -/
def channelInputTerm {Γ : List Shape} (completedCount modelDim : Nat)
    (hModel : 0 < modelDim) (query scale : Term Γ (.dim modelDim .scalar))
    (completed : Term Γ (.dim completedCount (.dim modelDim .scalar)))
    (partialState sequenceOutput : Term Γ (.dim modelDim .scalar)) :
    Term Γ (.dim modelDim .scalar) :=
  let sources := channelSourcesTerm completedCount modelDim completed partialState sequenceOutput
  let retrieved := AttnRes.term (completedCount + 1) modelDim hModel query sources
  Term.op (NN.GraphSpec.DAG.PrimOp.rmsNorm .scalar modelDim hModel)
    (.cons retrieved (.cons scale .nil))

/-- Graph term that accumulates sequence and channel outputs into the current depth block. -/
def finishPartialTerm {Γ : List Shape} (modelDim : Nat)
    (partialState sequenceOutput channelOutput : Term Γ (.dim modelDim .scalar)) :
    Term Γ (.dim modelDim .scalar) :=
  let withSequence := Term.op (NN.GraphSpec.DAG.PrimOp.add (.dim modelDim .scalar))
    (.cons partialState (.cons sequenceOutput .nil))
  Term.op (NN.GraphSpec.DAG.PrimOp.add (.dim modelDim .scalar))
    (.cons withSequence (.cons channelOutput .nil))

@[simp] theorem eval_sequenceSourcesTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (completedCount modelDim : Nat) (partialPresent : Bool)
    (completed : Term Γ (.dim completedCount (.dim modelDim .scalar)))
    (partialState : Term Γ (.dim modelDim .scalar)) :
    Term.eval env
        (sequenceSourcesTerm completedCount modelDim partialPresent completed partialState) =
      sequenceSources partialPresent (Term.eval env completed) (Term.eval env partialState) := by
  cases partialPresent
  · rfl
  · change Tensor.concatAxisSpec .scalar (Term.eval env completed)
      ((Term.eval env partialState).reshapeSpec _) = _
    rfl

@[simp] theorem eval_channelSourcesTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (completedCount modelDim : Nat)
    (completed : Term Γ (.dim completedCount (.dim modelDim .scalar)))
    (partialState sequenceOutput : Term Γ (.dim modelDim .scalar)) :
    Term.eval env
        (channelSourcesTerm completedCount modelDim completed partialState sequenceOutput) =
      channelSources (Term.eval env completed) (Term.eval env partialState)
        (Term.eval env sequenceOutput) := by
  change
    Tensor.concatAxisSpec .scalar (Term.eval env completed)
        (((Term.eval env partialState + Term.eval env sequenceOutput).reshapeSpec _)) = _
  rfl

theorem eval_sequenceInputTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (completedCount modelDim : Nat) (partialPresent : Bool)
    (hCompleted : 0 < completedCount) (hModel : 0 < modelDim)
    (query scale : Term Γ (.dim modelDim .scalar))
    (completed : Term Γ (.dim completedCount (.dim modelDim .scalar)))
    (partialState : Term Γ (.dim modelDim .scalar)) :
    Term.eval env
        (sequenceInputTerm completedCount modelDim partialPresent hCompleted hModel query scale
          completed partialState) =
      sequenceInput (completedCount := completedCount) (modelDim := modelDim)
        partialPresent hCompleted hModel (Term.eval env query) (Term.eval env scale)
        (Term.eval env completed) (Term.eval env partialState) := by
  simp only [sequenceInputTerm, sequenceInput, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.rmsNorm_specFwd]
  rw [AttnRes.eval_term, eval_sequenceSourcesTerm]
  change NN.GraphSpec.DAG.PrimOp.Internal.rmsNormVectorSemantics hModel _ _ = _
  exact GraphSpec.rmsNormVectorSemantics_eq_scale hModel _ _

theorem eval_channelInputTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (completedCount modelDim : Nat) (hModel : 0 < modelDim)
    (query scale : Term Γ (.dim modelDim .scalar))
    (completed : Term Γ (.dim completedCount (.dim modelDim .scalar)))
    (partialState sequenceOutput : Term Γ (.dim modelDim .scalar)) :
    Term.eval env
        (channelInputTerm completedCount modelDim hModel query scale completed partialState
          sequenceOutput) =
      channelInput (completedCount := completedCount) (modelDim := modelDim)
        hModel (Term.eval env query) (Term.eval env scale)
        (Term.eval env completed) (Term.eval env partialState) (Term.eval env sequenceOutput) := by
  simp only [channelInputTerm, channelInput, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.rmsNorm_specFwd]
  rw [AttnRes.eval_term, eval_channelSourcesTerm]
  change NN.GraphSpec.DAG.PrimOp.Internal.rmsNormVectorSemantics hModel _ _ = _
  exact GraphSpec.rmsNormVectorSemantics_eq_scale hModel _ _

@[simp] theorem eval_finishPartialTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (modelDim : Nat)
    (partialState sequenceOutput channelOutput : Term Γ (.dim modelDim .scalar)) :
    Term.eval env (finishPartialTerm modelDim partialState sequenceOutput channelOutput) =
      finishPartial (Term.eval env partialState) (Term.eval env sequenceOutput)
        (Term.eval env channelOutput) := by
  rfl

end PackedDepth

namespace Backbone

/-- Learned AttnRes queries and RMSNorm scales shared by the dense and sparse graph builders. -/
abbrev DepthParams (modelDim : Nat) : List Shape :=
  [ .dim modelDim .scalar, .dim modelDim .scalar,
    .dim modelDim .scalar, .dim modelDim .scalar ]

/-- Packed depth memory supplied to one statically positioned decoder layer. -/
abbrev DepthInputs (completedCount modelDim : Nat) : List Shape :=
  [.dim completedCount (.dim modelDim .scalar), .dim modelDim .scalar]

/-- Zero queries and unit normalization scales for a freshly initialized depth controller. -/
def initialDepthParams (modelDim : Nat) : TorchLean.TensorPack Float (DepthParams modelDim) :=
  .cons (Spec.fill 0 (.dim modelDim .scalar)) <|
    .cons (Spec.fill 1 (.dim modelDim .scalar)) <|
      .cons (Spec.fill 0 (.dim modelDim .scalar)) <|
        .cons (Spec.fill 1 (.dim modelDim .scalar)) .nil

/-- Extract a layer's learned depth queries and normalization scales in graph order. -/
def depthParameters {α : Type} {cfg : TextConfig} {decayRank : Nat}
    (layer : BackboneLayer α cfg decayRank) : TorchLean.TensorPack α (DepthParams cfg.hiddenDim) :=
  .cons layer.sequenceQuery <| .cons layer.sequenceNormScale <|
    .cons layer.channelQuery <| .cons layer.channelNormScale .nil

/-- Package completed depth blocks and the current partial block in graph order. -/
def depthInputs {α : Type} {completedCount modelDim : Nat}
    (completed : Tensor α (.dim completedCount (.dim modelDim .scalar)))
    (partialState : Tensor α (.dim modelDim .scalar)) :
    TorchLean.TensorPack α (DepthInputs completedCount modelDim) :=
  .cons completed (.cons partialState .nil)

namespace KDADense

/-- Parameters of one KDA layer followed by its depth controls and dense SiTU expert. -/
abbrev Params (modelDim heads keyDim valueDim convWidth decayRank denseHidden : Nat) : List Shape :=
  KDA.LayerParams modelDim heads keyDim valueDim convWidth decayRank ++
    (DepthParams modelDim ++ Expert.Params modelDim denseHidden modelDim)

/-- Runtime state of a dense KDA decoder layer at one token. -/
abbrev Inputs (completedCount modelDim heads keyDim valueDim convWidth : Nat) : List Shape :=
  DepthInputs completedCount modelDim ++
    [ .dim convWidth (.dim modelDim .scalar),
      .dim heads (.dim keyDim (.dim valueDim .scalar)),
      .scalar, .scalar, .scalar, .scalar ]

/-- Updated KDA window and recurrence, both submodule outputs, and the new AttnRes partial sum. -/
abbrev Outputs (modelDim heads keyDim valueDim convWidth : Nat) : List Shape :=
  [ .dim convWidth (.dim modelDim .scalar),
    .dim heads (.dim keyDim (.dim valueDim .scalar)),
    .dim modelDim .scalar, .dim modelDim .scalar, .dim modelDim .scalar ]

/-- A complete dense KDA decoder-layer token transition. -/
def model (completedCount modelDim heads keyDim valueDim convWidth decayRank denseHidden : Nat)
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < modelDim)
    (hHeads : 0 < heads) (hKey : 0 < keyDim) (hValue : 0 < valueDim)
    (hWidth : 0 < convWidth) :
    NN.GraphSpec.DAG.MultiModel
      (Params modelDim heads keyDim valueDim convWidth decayRank denseHidden)
      (Inputs completedCount modelDim heads keyDim valueDim convWidth)
      (Outputs modelDim heads keyDim valueDim convWidth) :=
  let mixerParams := KDA.LayerParams modelDim heads keyDim valueDim convWidth decayRank
  let depthParams := DepthParams modelDim
  let expertParams := Expert.Params modelDim denseHidden modelDim
  let depthIns := DepthInputs completedCount modelDim
  let stateIns : List Shape :=
    [ .dim convWidth (.dim modelDim .scalar),
      .dim heads (.dim keyDim (.dim valueDim .scalar)),
      .scalar, .scalar, .scalar, .scalar ]
  let params := mixerParams ++ (depthParams ++ expertParams)
  let ins := depthIns ++ stateIns
  let Γ := params ++ ins
  let mixerArgs : Args Γ mixerParams :=
    Args.rename (Var.inLeft ins) <|
      Args.rename (Var.inLeft (depthParams ++ expertParams)) (Args.vars mixerParams)
  let depthArgs : Args Γ depthParams :=
    Args.rename (Var.inLeft ins) <| Args.rename (Var.inRight mixerParams) <|
      Args.rename (Var.inLeft expertParams) (Args.vars depthParams)
  let expertArgs : Args Γ expertParams :=
    Args.rename (Var.inLeft ins) <| Args.rename (Var.inRight mixerParams) <|
      Args.rename (Var.inRight depthParams) (Args.vars expertParams)
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
  let nextState : Term Γ' (.dim heads (.dim keyDim (.dim valueDim .scalar))) :=
    Term.var (Var.inRight Γ .head)
  let sequenceOutput : Term Γ' (.dim modelDim .scalar) :=
    Term.var (Var.inRight Γ (.tail .head))
  let channelInput := PackedDepth.channelInputTerm completedCount modelDim hModel
    (original channelQuery) (original channelScale) (original completed)
    (original partialState) sequenceOutput
  let channelOutput := (Expert.model modelDim denseHidden modelDim).inline
    (Args.rename (Var.inLeft mixerOuts) expertArgs)
    (.cons channelInput (.cons (original gateCap) (.cons (original upCap) .nil)))
  let nextPartial := PackedDepth.finishPartialTerm modelDim (original partialState)
    sequenceOutput channelOutput
  { initParams :=
      TorchLean.TensorPack.append
        (KDA.initialLayerParams modelDim heads keyDim valueDim convWidth decayRank)
        (TorchLean.TensorPack.append (initialDepthParams modelDim)
          (Expert.initialParams modelDim denseHidden modelDim))
    body := mixerBlock.andThen <| Block.ret <|
      .cons (original nextWindow) <| .cons nextState <| .cons sequenceOutput <|
        .cons channelOutput <| .cons nextPartial .nil }

/-- Mathematical result returned by the dense KDA graph. -/
noncomputable def specStep
    {completedCount modelDim heads keyDim valueDim convWidth decayRank denseHidden : Nat}
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < modelDim)
    (hWidth : 0 < convWidth)
    (layer : KDALayer ℝ modelDim heads keyDim valueDim convWidth decayRank)
    (expert : Expert ℝ modelDim denseHidden modelDim)
    (sequenceQuery sequenceScale channelQuery channelScale :
      Tensor ℝ (.dim modelDim .scalar))
    (completed : Tensor ℝ (.dim completedCount (.dim modelDim .scalar)))
    (partialState : Tensor ℝ (.dim modelDim .scalar))
    (previousWindow : Tensor ℝ (.dim convWidth (.dim modelDim .scalar)))
    (previousState : KDALayer.State ℝ heads keyDim valueDim)
    (logFloor gateCap upCap : ℝ) :
    TorchLean.TensorPack ℝ (Outputs modelDim heads keyDim valueDim convWidth) :=
  let sequenceInput := PackedDepth.sequenceInput partialPresent hCompleted hModel sequenceQuery
    sequenceScale completed partialState
  let nextWindow := KDALayer.rollWindow (α := ℝ) hWidth sequenceInput previousWindow
  let sequenceResult := layer.stepWindow hWidth logFloor nextWindow previousState
  let channelInput := PackedDepth.channelInput hModel channelQuery channelScale completed
    partialState sequenceResult.2
  let channelOutput := expert.forward gateCap upCap channelInput
  let nextPartial := PackedDepth.finishPartial partialState sequenceResult.2 channelOutput
  .cons nextWindow <| .cons sequenceResult.1 <| .cons sequenceResult.2 <|
    .cons channelOutput <| .cons nextPartial .nil

/-- Package the KDA, depth-control, and expert parameters in the composed model's order. -/
def parameters
    {cfg : TextConfig} {decayRank : Nat}
    (layer : BackboneLayer ℝ cfg decayRank)
    (kda : KDALayer ℝ cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
      cfg.shortConvWidth decayRank)
    (expert : Expert ℝ cfg.hiddenDim cfg.denseHiddenDim cfg.hiddenDim) :
    TorchLean.TensorPack ℝ
      (Params cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim cfg.shortConvWidth
        decayRank cfg.denseHiddenDim) :=
  TorchLean.TensorPack.append (KDA.layerParameters kda)
    (TorchLean.TensorPack.append (depthParameters layer) (Expert.parameters expert))

/-- Package one dense KDA decoder-layer state in the composed model's input order. -/
def inputs
    {cfg : TextConfig} {completedCount : Nat}
    (completed : Tensor ℝ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Tensor ℝ (.dim cfg.hiddenDim .scalar))
    (previousWindow : Tensor ℝ (.dim cfg.shortConvWidth (.dim cfg.hiddenDim .scalar)))
    (previousState : KDALayer.State ℝ cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim)
    (logFloor epsilon gateCap upCap : ℝ) :
    TorchLean.TensorPack ℝ
      (Inputs completedCount cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
        cfg.shortConvWidth) :=
  TorchLean.TensorPack.append (depthInputs completed partialState) <|
    .cons previousWindow <| .cons previousState <| .cons (.scalar logFloor) <|
      .cons (.scalar epsilon) <| .cons (.scalar gateCap) <| .cons (.scalar upCap) .nil

/-- The complete dense KDA graph denotes the packed mathematical decoder-layer transition. -/
theorem model_specFwd_eq_specStep
    {cfg : TextConfig} {completedCount decayRank : Nat}
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hHeads : 0 < cfg.numHeads) (hKey : 0 < cfg.kdaHeadDim)
    (hValue : 0 < cfg.kdaValueDim) (hWidth : 0 < cfg.shortConvWidth)
    (layer : BackboneLayer ℝ cfg decayRank)
    (kda : KDALayer ℝ cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
      cfg.shortConvWidth decayRank)
    (expert : Expert ℝ cfg.hiddenDim cfg.denseHiddenDim cfg.hiddenDim)
    (completed : Tensor ℝ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Tensor ℝ (.dim cfg.hiddenDim .scalar))
    (previousWindow : Tensor ℝ (.dim cfg.shortConvWidth (.dim cfg.hiddenDim .scalar)))
    (previousState : KDALayer.State ℝ cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim)
    (logFloor epsilon gateCap upCap : ℝ) (hEpsilon : epsilon = Numbers.epsilon) :
    (model completedCount cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
      cfg.shortConvWidth decayRank cfg.denseHiddenDim partialPresent hCompleted hModel hHeads hKey
      hValue hWidth).specFwd
        (parameters layer kda expert)
        (inputs completed partialState previousWindow previousState logFloor epsilon gateCap upCap) =
      specStep partialPresent hCompleted hModel hWidth kda expert layer.sequenceQuery
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
  let nextWindow := KDALayer.rollWindow hWidth sequenceInput previousWindow
  have hMixer :
      (KDA.layerModel cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
        cfg.shortConvWidth decayRank hModel hHeads hKey hValue hWidth).specFwd
          (KDA.layerParameters kda)
          (.cons nextWindow <| .cons previousState <| .cons (.scalar logFloor) <|
            .cons (.scalar Numbers.epsilon) .nil) =
        let result := kda.stepWindow hWidth logFloor nextWindow previousState
        .cons result.1 (.cons result.2 .nil) := by
    simpa only [KDA.layerInputs] using
      KDA.layerModel_specFwd_eq_stepWindow hModel hHeads hKey hValue hWidth kda
        nextWindow previousState logFloor
  let sequenceResult := kda.stepWindow hWidth logFloor nextWindow previousState
  let channelInput := PackedDepth.channelInput hModel layer.channelQuery layer.channelNormScale
    completed partialState sequenceResult.2
  have hExpert :
      (Expert.model cfg.hiddenDim cfg.denseHiddenDim cfg.hiddenDim).specFwd
          (Expert.parameters expert)
          (.cons channelInput <| .cons (.scalar gateCap) <| .cons (.scalar upCap) .nil) =
        expert.forward gateCap upCap channelInput := by
    simpa only [Expert.inputs] using
      Expert.specFwd_eq_forward expert channelInput gateCap upCap
  change Block.eval _ _ = _
  rw [hMixer]
  simp only [Block.eval, Term.evalArgs, Term.eval_weakenAppend, Term.eval,
    Env.tget_append_inRight, KDA.eval_rollWindowTerm,
    PackedDepth.eval_sequenceInputTerm, PackedDepth.eval_channelInputTerm,
    PackedDepth.eval_finishPartialTerm, NN.GraphSpec.DAG.Model.eval_inline,
    Term.evalArgs_rename_inLeft, Term.evalArgs_rename_inRight, Term.evalArgs_vars,
    Term.eval_rename_inLeft, Term.eval_rename_inRight, Env.tget,
    specStep]
  rw [hExpert]

end KDADense

namespace KDASparse

/-- Parameters of one KDA layer followed by its depth controls and Stable LatentMoE. -/
abbrev Params (modelDim heads keyDim valueDim convWidth decayRank latentDim expertHidden
    numShared numRouted : Nat) : List Shape :=
  KDA.LayerParams modelDim heads keyDim valueDim convWidth decayRank ++
    (DepthParams modelDim ++
      StableLatentMoE.Params modelDim latentDim expertHidden expertHidden numShared numRouted)

/-- Runtime inputs of a sparse KDA layer.  Routing is a proved static graph argument, not an
untyped tensor payload. -/
abbrev Inputs (completedCount modelDim heads keyDim valueDim convWidth : Nat) : List Shape :=
  DepthInputs completedCount modelDim ++
    [ .dim convWidth (.dim modelDim .scalar),
      .dim heads (.dim keyDim (.dim valueDim .scalar)),
      .scalar, .scalar, .scalar, .scalar ]

/-- Updated KDA state, both submodule outputs, and the new AttnRes partial sum. -/
abbrev Outputs (modelDim heads keyDim valueDim convWidth : Nat) : List Shape :=
  [ .dim convWidth (.dim modelDim .scalar),
    .dim heads (.dim keyDim (.dim valueDim .scalar)),
    .dim modelDim .scalar, .dim modelDim .scalar, .dim modelDim .scalar ]

/-- Complete sparse KDA decoder-layer transition for one already certified expert route. -/
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
    [ .dim convWidth (.dim modelDim .scalar),
      .dim heads (.dim keyDim (.dim valueDim .scalar)),
      .scalar, .scalar, .scalar, .scalar ]
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
  let nextState : Term Γ' (.dim heads (.dim keyDim (.dim valueDim .scalar))) :=
    Term.var (Var.inRight Γ .head)
  let sequenceOutput : Term Γ' (.dim modelDim .scalar) :=
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
    (layer : KDALayer ℝ modelDim heads keyDim valueDim convWidth decayRank)
    (moe : KimiK3.StableLatentMoE ℝ modelDim latentDim expertHidden expertHidden numShared
      numRouted activeExperts)
    (route : Route numRouted activeExperts)
    (sequenceQuery sequenceScale channelQuery channelScale : Tensor ℝ (.dim modelDim .scalar))
    (completed : Tensor ℝ (.dim completedCount (.dim modelDim .scalar)))
    (partialState : Tensor ℝ (.dim modelDim .scalar))
    (previousWindow : Tensor ℝ (.dim convWidth (.dim modelDim .scalar)))
    (previousState : KDALayer.State ℝ heads keyDim valueDim)
    (logFloor gateCap upCap : ℝ) :
    TorchLean.TensorPack ℝ (Outputs modelDim heads keyDim valueDim convWidth) :=
  let sequenceInput := PackedDepth.sequenceInput partialPresent hCompleted hModel sequenceQuery
    sequenceScale completed partialState
  let nextWindow := KDALayer.rollWindow (α := ℝ) hWidth sequenceInput previousWindow
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
    (kda : KDALayer ℝ cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
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
    (completed : Tensor ℝ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Tensor ℝ (.dim cfg.hiddenDim .scalar))
    (previousWindow : Tensor ℝ (.dim cfg.shortConvWidth (.dim cfg.hiddenDim .scalar)))
    (previousState : KDALayer.State ℝ cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim)
    (logFloor epsilon gateCap upCap : ℝ) :
    TorchLean.TensorPack ℝ
      (Inputs completedCount cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
        cfg.shortConvWidth) :=
  TorchLean.TensorPack.append (depthInputs completed partialState) <|
    .cons previousWindow <| .cons previousState <| .cons (.scalar logFloor) <|
      .cons (.scalar epsilon) <| .cons (.scalar gateCap) <| .cons (.scalar upCap) .nil

/-- The sparse KDA graph denotes the mathematical transition for the supplied route. -/
theorem model_specFwd_eq_specStep
    {cfg : TextConfig} {completedCount decayRank : Nat}
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hHeads : 0 < cfg.numHeads) (hKey : 0 < cfg.kdaHeadDim)
    (hValue : 0 < cfg.kdaValueDim) (hWidth : 0 < cfg.shortConvWidth)
    (hLatent : 0 < cfg.routedLatentDim) (hActive : 0 < cfg.activeExperts)
    (layer : BackboneLayer ℝ cfg decayRank)
    (kda : KDALayer ℝ cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
      cfg.shortConvWidth decayRank)
    (moe : KimiK3.StableLatentMoE ℝ cfg.hiddenDim cfg.routedLatentDim
      cfg.routedExpertHiddenDim cfg.routedExpertHiddenDim cfg.numSharedExperts
      cfg.numRoutedExperts cfg.activeExperts)
    (route : Route cfg.numRoutedExperts cfg.activeExperts)
    (completed : Tensor ℝ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Tensor ℝ (.dim cfg.hiddenDim .scalar))
    (previousWindow : Tensor ℝ (.dim cfg.shortConvWidth (.dim cfg.hiddenDim .scalar)))
    (previousState : KDALayer.State ℝ cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim)
    (logFloor epsilon gateCap upCap : ℝ) (hEpsilon : epsilon = Numbers.epsilon) :
    (model completedCount cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
      cfg.shortConvWidth decayRank cfg.routedLatentDim cfg.routedExpertHiddenDim
      cfg.numSharedExperts cfg.numRoutedExperts cfg.activeExperts route partialPresent hCompleted
      hModel hHeads hKey hValue hWidth hLatent).specFwd
        (parameters layer kda moe)
        (inputs completed partialState previousWindow previousState logFloor epsilon gateCap upCap) =
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
  let nextWindow := KDALayer.rollWindow hWidth sequenceInput previousWindow
  have hMixer :
      (KDA.layerModel cfg.hiddenDim cfg.numHeads cfg.kdaHeadDim cfg.kdaValueDim
        cfg.shortConvWidth decayRank hModel hHeads hKey hValue hWidth).specFwd
          (KDA.layerParameters kda)
          (.cons nextWindow <| .cons previousState <| .cons (.scalar logFloor) <|
            .cons (.scalar Numbers.epsilon) .nil) =
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
          (.cons channelInput <| .cons (.scalar gateCap) <| .cons (.scalar upCap) .nil) =
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

namespace MLADense

/-- Parameters of one fixed-cache MLA layer followed by depth controls and a dense expert. -/
abbrev Params (modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim
    denseHidden : Nat) : List Shape :=
  MLA.LayerParams modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim ++
    (DepthParams modelDim ++ Expert.Params modelDim denseHidden modelDim)

/-- Packed depth memory, MLA caches, score scale, and SiTU caps for one token. -/
abbrev Inputs (pastTokens completedCount modelDim kvLatentDim sharedKeyDim : Nat) : List Shape :=
  DepthInputs completedCount modelDim ++
    [ .dim pastTokens (.dim kvLatentDim .scalar),
      .dim pastTokens (.dim sharedKeyDim .scalar), .scalar, .scalar, .scalar ]

/-- Updated MLA caches, both submodule outputs, and the new AttnRes partial sum. -/
abbrev Outputs (pastTokens modelDim kvLatentDim sharedKeyDim : Nat) : List Shape :=
  [ .dim (pastTokens + 1) (.dim kvLatentDim .scalar),
    .dim (pastTokens + 1) (.dim sharedKeyDim .scalar),
    .dim modelDim .scalar, .dim modelDim .scalar, .dim modelDim .scalar ]

/-- Complete dense MLA decoder-layer transition at a fixed cache length. -/
def model (pastTokens completedCount modelDim heads queryLatentDim kvLatentDim contentKeyDim
    sharedKeyDim valueDim denseHidden : Nat) (partialPresent : Bool)
    (hCompleted : 0 < completedCount) (hModel : 0 < modelDim)
    (hQueryLatent : 0 < queryLatentDim) (hKVLatent : 0 < kvLatentDim) :
    NN.GraphSpec.DAG.MultiModel
      (Params modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim
        denseHidden)
      (Inputs pastTokens completedCount modelDim kvLatentDim sharedKeyDim)
      (Outputs pastTokens modelDim kvLatentDim sharedKeyDim) :=
  let mixerParams :=
    MLA.LayerParams modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim
  let depthParams := DepthParams modelDim
  let channelParams := Expert.Params modelDim denseHidden modelDim
  let depthIns := DepthInputs completedCount modelDim
  let stateIns : List Shape :=
    [ .dim pastTokens (.dim kvLatentDim .scalar),
      .dim pastTokens (.dim sharedKeyDim .scalar), .scalar, .scalar, .scalar ]
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
  let nextLatentCache : Term Γ' (.dim (pastTokens + 1) (.dim kvLatentDim .scalar)) :=
    Term.var (Var.inRight Γ .head)
  let nextSharedKeyCache : Term Γ' (.dim (pastTokens + 1) (.dim sharedKeyDim .scalar)) :=
    Term.var (Var.inRight Γ (.tail .head))
  let sequenceOutput : Term Γ' (.dim modelDim .scalar) :=
    Term.var (Var.inRight Γ (.tail (.tail .head)))
  let channelInput := PackedDepth.channelInputTerm completedCount modelDim hModel
    (original channelQuery) (original channelScale) (original completed)
    (original partialState) sequenceOutput
  let channelOutput := (Expert.model modelDim denseHidden modelDim).inline
    (Args.rename (Var.inLeft mixerOuts) channelArgs)
    (.cons channelInput (.cons (original gateCap) (.cons (original upCap) .nil)))
  let nextPartial := PackedDepth.finishPartialTerm modelDim (original partialState)
    sequenceOutput channelOutput
  { initParams :=
      TorchLean.TensorPack.append
        (MLA.initialLayerParams modelDim heads queryLatentDim kvLatentDim contentKeyDim
          sharedKeyDim valueDim)
        (TorchLean.TensorPack.append (initialDepthParams modelDim)
          (Expert.initialParams modelDim denseHidden modelDim))
    body := mixerBlock.andThen <| Block.ret <|
      .cons nextLatentCache <| .cons nextSharedKeyCache <| .cons sequenceOutput <|
        .cons channelOutput <| .cons nextPartial .nil }

/-- Mathematical transition represented by the dense MLA graph. -/
noncomputable def specStep
    {pastTokens completedCount modelDim heads queryLatentDim kvLatentDim contentKeyDim
      sharedKeyDim valueDim denseHidden : Nat}
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < modelDim)
    (layer : GatedMLA ℝ modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim
      valueDim)
    (expert : Expert ℝ modelDim denseHidden modelDim)
    (sequenceQuery sequenceScale channelQuery channelScale : Tensor ℝ (.dim modelDim .scalar))
    (completed : Tensor ℝ (.dim completedCount (.dim modelDim .scalar)))
    (partialState : Tensor ℝ (.dim modelDim .scalar))
    (pastLatentCache : Tensor ℝ (.dim pastTokens (.dim kvLatentDim .scalar)))
    (pastSharedKeyCache : Tensor ℝ (.dim pastTokens (.dim sharedKeyDim .scalar)))
    (scoreScale gateCap upCap : ℝ) :
    TorchLean.TensorPack ℝ (Outputs pastTokens modelDim kvLatentDim sharedKeyDim) :=
  let sequenceInput := PackedDepth.sequenceInput partialPresent hCompleted hModel sequenceQuery
    sequenceScale completed partialState
  let sequenceResult := layer.stepFixed pastTokens pastLatentCache
    pastSharedKeyCache sequenceInput scoreScale
  let channelInput := PackedDepth.channelInput hModel channelQuery channelScale completed
    partialState sequenceResult.2.2
  let channelOutput := expert.forward gateCap upCap channelInput
  let nextPartial := PackedDepth.finishPartial partialState sequenceResult.2.2 channelOutput
  .cons sequenceResult.1 <| .cons sequenceResult.2.1 <| .cons sequenceResult.2.2 <|
    .cons channelOutput <| .cons nextPartial .nil

/-- Pack MLA, depth-control, and dense-channel parameters in graph order. -/
def parameters {cfg : TextConfig} {decayRank : Nat}
    (backbone : BackboneLayer ℝ cfg decayRank)
    (mla : GatedMLA ℝ cfg.hiddenDim cfg.numHeads cfg.queryLatentDim cfg.kvLatentDim
      cfg.qkNopeHeadDim cfg.qkReservedHeadDim cfg.valueHeadDim)
    (expert : Expert ℝ cfg.hiddenDim cfg.denseHiddenDim cfg.hiddenDim) :
    TorchLean.TensorPack ℝ
      (Params cfg.hiddenDim cfg.numHeads cfg.queryLatentDim cfg.kvLatentDim
        cfg.qkNopeHeadDim cfg.qkReservedHeadDim cfg.valueHeadDim cfg.denseHiddenDim) :=
  TorchLean.TensorPack.append (MLA.layerParameters mla)
    (TorchLean.TensorPack.append (depthParameters backbone)
      (Expert.parameters expert))

/-- Package one dense MLA decoder state in graph order. -/
def inputs {cfg : TextConfig} {pastTokens completedCount : Nat}
    (completed : Tensor ℝ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Tensor ℝ (.dim cfg.hiddenDim .scalar))
    (pastLatentCache : Tensor ℝ (.dim pastTokens (.dim cfg.kvLatentDim .scalar)))
    (pastSharedKeyCache : Tensor ℝ
      (.dim pastTokens (.dim cfg.qkReservedHeadDim .scalar)))
    (scoreScale gateCap upCap : ℝ) :
    TorchLean.TensorPack ℝ
      (Inputs pastTokens completedCount cfg.hiddenDim cfg.kvLatentDim
        cfg.qkReservedHeadDim) :=
  TorchLean.TensorPack.append (depthInputs completed partialState) <|
    .cons pastLatentCache <| .cons pastSharedKeyCache <| .cons (.scalar scoreScale) <|
      .cons (.scalar gateCap) <| .cons (.scalar upCap) .nil

/-- The dense MLA graph denotes its mathematical fixed-cache decoder transition. -/
theorem model_specFwd_eq_specStep
    {cfg : TextConfig} {pastTokens completedCount decayRank : Nat}
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < cfg.hiddenDim)
    (hQueryLatent : 0 < cfg.queryLatentDim) (hKVLatent : 0 < cfg.kvLatentDim)
    (backbone : BackboneLayer ℝ cfg decayRank)
    (mla : GatedMLA ℝ cfg.hiddenDim cfg.numHeads cfg.queryLatentDim cfg.kvLatentDim
      cfg.qkNopeHeadDim cfg.qkReservedHeadDim cfg.valueHeadDim)
    (expert : Expert ℝ cfg.hiddenDim cfg.denseHiddenDim cfg.hiddenDim)
    (completed : Tensor ℝ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Tensor ℝ (.dim cfg.hiddenDim .scalar))
    (pastLatentCache : Tensor ℝ (.dim pastTokens (.dim cfg.kvLatentDim .scalar)))
    (pastSharedKeyCache : Tensor ℝ
      (.dim pastTokens (.dim cfg.qkReservedHeadDim .scalar)))
    (scoreScale gateCap upCap : ℝ) :
    (model pastTokens completedCount cfg.hiddenDim cfg.numHeads cfg.queryLatentDim
      cfg.kvLatentDim cfg.qkNopeHeadDim cfg.qkReservedHeadDim cfg.valueHeadDim
      cfg.denseHiddenDim partialPresent hCompleted hModel hQueryLatent hKVLatent).specFwd
        (parameters backbone mla expert)
        (inputs completed partialState pastLatentCache pastSharedKeyCache scoreScale gateCap upCap) =
      specStep partialPresent hCompleted hModel mla expert
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
            .cons (.scalar scoreScale) .nil) =
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
      (Expert.model cfg.hiddenDim cfg.denseHiddenDim cfg.hiddenDim).specFwd
          (Expert.parameters expert)
          (.cons channelInput <| .cons (.scalar gateCap) <| .cons (.scalar upCap) .nil) =
        expert.forward gateCap upCap channelInput := by
    simpa only [Expert.inputs] using
      Expert.specFwd_eq_forward expert channelInput gateCap upCap
  change Block.eval _ _ = _
  rw [hMixer]
  simp only [Block.eval, Term.evalArgs, Term.eval_weakenAppend, Term.eval,
    Env.tget_append_inRight, PackedDepth.eval_channelInputTerm,
    PackedDepth.eval_finishPartialTerm,
    NN.GraphSpec.DAG.Model.eval_inline, Term.evalArgs_rename_inLeft,
    Term.evalArgs_rename_inRight, Term.evalArgs_vars, Term.eval_rename_inLeft,
    Term.eval_rename_inRight, Env.tget, specStep]
  rw [hChannel]

end MLADense

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
    [ .dim pastTokens (.dim kvLatentDim .scalar),
      .dim pastTokens (.dim sharedKeyDim .scalar), .scalar, .scalar, .scalar ]

/-- Updated MLA caches, both submodule outputs, and the new AttnRes partial sum. -/
abbrev Outputs (pastTokens modelDim kvLatentDim sharedKeyDim : Nat) : List Shape :=
  [ .dim (pastTokens + 1) (.dim kvLatentDim .scalar),
    .dim (pastTokens + 1) (.dim sharedKeyDim .scalar),
    .dim modelDim .scalar, .dim modelDim .scalar, .dim modelDim .scalar ]

/-- Complete sparse MLA decoder-layer transition for one already certified expert route. -/
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
    [ .dim pastTokens (.dim kvLatentDim .scalar),
      .dim pastTokens (.dim sharedKeyDim .scalar), .scalar, .scalar, .scalar ]
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
  let nextLatentCache : Term Γ' (.dim (pastTokens + 1) (.dim kvLatentDim .scalar)) :=
    Term.var (Var.inRight Γ .head)
  let nextSharedKeyCache : Term Γ' (.dim (pastTokens + 1) (.dim sharedKeyDim .scalar)) :=
    Term.var (Var.inRight Γ (.tail .head))
  let sequenceOutput : Term Γ' (.dim modelDim .scalar) :=
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
    (sequenceQuery sequenceScale channelQuery channelScale : Tensor ℝ (.dim modelDim .scalar))
    (completed : Tensor ℝ (.dim completedCount (.dim modelDim .scalar)))
    (partialState : Tensor ℝ (.dim modelDim .scalar))
    (pastLatentCache : Tensor ℝ (.dim pastTokens (.dim kvLatentDim .scalar)))
    (pastSharedKeyCache : Tensor ℝ (.dim pastTokens (.dim sharedKeyDim .scalar)))
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
    (completed : Tensor ℝ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Tensor ℝ (.dim cfg.hiddenDim .scalar))
    (pastLatentCache : Tensor ℝ (.dim pastTokens (.dim cfg.kvLatentDim .scalar)))
    (pastSharedKeyCache : Tensor ℝ
      (.dim pastTokens (.dim cfg.qkReservedHeadDim .scalar)))
    (scoreScale gateCap upCap : ℝ) :
    TorchLean.TensorPack ℝ
      (Inputs pastTokens completedCount cfg.hiddenDim cfg.kvLatentDim
        cfg.qkReservedHeadDim) :=
  TorchLean.TensorPack.append (depthInputs completed partialState) <|
    .cons pastLatentCache <| .cons pastSharedKeyCache <| .cons (.scalar scoreScale) <|
      .cons (.scalar gateCap) <| .cons (.scalar upCap) .nil

/-- The sparse MLA graph denotes its mathematical transition for the supplied route. -/
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
    (completed : Tensor ℝ (.dim completedCount (.dim cfg.hiddenDim .scalar)))
    (partialState : Tensor ℝ (.dim cfg.hiddenDim .scalar))
    (pastLatentCache : Tensor ℝ (.dim pastTokens (.dim cfg.kvLatentDim .scalar)))
    (pastSharedKeyCache : Tensor ℝ
      (.dim pastTokens (.dim cfg.qkReservedHeadDim .scalar)))
    (scoreScale gateCap upCap : ℝ) :
    (model pastTokens completedCount cfg.hiddenDim cfg.numHeads cfg.queryLatentDim
      cfg.kvLatentDim cfg.qkNopeHeadDim cfg.qkReservedHeadDim cfg.valueHeadDim
      cfg.routedLatentDim cfg.routedExpertHiddenDim cfg.numSharedExperts cfg.numRoutedExperts
      cfg.activeExperts route partialPresent hCompleted hModel hQueryLatent hKVLatent
      hLatent).specFwd (parameters backbone mla moe)
        (inputs completed partialState pastLatentCache pastSharedKeyCache scoreScale gateCap upCap) =
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
            .cons (.scalar scoreScale) .nil) =
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
          (.cons channelInput <| .cons (.scalar gateCap) <| .cons (.scalar upCap) .nil) =
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
