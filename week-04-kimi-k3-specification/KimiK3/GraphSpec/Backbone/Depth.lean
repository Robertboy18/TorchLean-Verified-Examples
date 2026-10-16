/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.AttnRes
public import KimiK3.Model

/-!
# Packed depth state for backbone graphs

The shared depth interface packs completed AttnRes blocks and the current partial block. Its term
builders retrieve and normalize the sequence and channel inputs, then accumulate both outputs. The
four decoder-layer graphs all use the parameter and input shapes below.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec

open Spec TorchLean
open TorchLean.Tensor
open NN.GraphSpec.DAG
open Runtime.Autograd.Torch

namespace PackedDepth

/-- Number of depth representations visible to the sequence mixer. `completedCount` counts the token
embedding and every completed AttnRes block, and a nonempty partial block adds one more row. -/
def sourceCount (completedCount : Nat) (partialPresent : Bool) : Nat :=
  completedCount + if partialPresent then 1 else 0

/-- Pack the depth representations visible before the sequence mixer. -/
def sequenceSources {α : Type} [Storage α] [Context α] {completedCount modelDim : Nat}
    (partialPresent : Bool)
    (completed : Tensor α [completedCount, modelDim])
    (partialState : Tensor α [modelDim]) :
    Tensor α [sourceCount completedCount partialPresent, modelDim] :=
  match partialPresent with
  | false => completed
  | true =>
      let partialRow : Tensor α [1, modelDim] :=
        Tensor.reshapeSpec partialState (by simp [Shape.size])
      Tensor.concat completed partialRow

/-- Depth sources for the channel mixer: the completed blocks, then the partial block with the
sequence output added as the last row. -/
def channelSources {α : Type} [Storage α] [Context α] {completedCount modelDim : Nat}
    (completed : Tensor α [completedCount, modelDim])
    (partialState sequenceOutput : Tensor α [modelDim]) :
    Tensor α [completedCount + 1, modelDim] :=
  let current := partialState + sequenceOutput
  let currentRow : Tensor α [1, modelDim] :=
    Tensor.reshapeSpec current (by simp [Shape.size])
  Tensor.concat completed currentRow

/-- Packed counterpart of `BackboneLayer`'s first AttnRes retrieval and RMS normalization. -/
noncomputable def sequenceInput {completedCount modelDim : Nat} (partialPresent : Bool)
    (hCompleted : 0 < completedCount) (hModel : 0 < modelDim)
    (query scale : Tensor ℝ [modelDim])
    (completed : Tensor ℝ [completedCount, modelDim])
    (partialState : Tensor ℝ [modelDim]) :
    Tensor ℝ [modelDim] :=
  let sources := sequenceSources partialPresent completed partialState
  have hSources : 0 < sourceCount completedCount partialPresent := by
    simp [sourceCount]
    omega
  RMSNorm.scale (AttnRes.attendPacked hModel query sources) scale

/-- Packed counterpart of the second AttnRes retrieval and RMS normalization. -/
noncomputable def channelInput {completedCount modelDim : Nat} (hModel : 0 < modelDim)
    (query scale : Tensor ℝ [modelDim])
    (completed : Tensor ℝ [completedCount, modelDim])
    (partialState sequenceOutput : Tensor ℝ [modelDim]) :
    Tensor ℝ [modelDim] :=
  let sources := channelSources completed partialState sequenceOutput
  RMSNorm.scale (AttnRes.attendPacked hModel query sources) scale

/-- Accumulate both submodule outputs into the current AttnRes block. The layer index decides
statically whether this vector stays a partial block or is committed as a completed one. -/
def finishPartial {α : Type} [Storage α] [Context α] {modelDim : Nat}
    (partialState sequenceOutput channelOutput : Tensor α [modelDim]) :
    Tensor α [modelDim] :=
  partialState + sequenceOutput + channelOutput

/-- Graph term that packs the depth sources visible to a sequence mixer. -/
def sequenceSourcesTerm {Γ : List Shape} (completedCount modelDim : Nat)
    (partialPresent : Bool)
    (completed : Term Γ [completedCount, modelDim])
    (partialState : Term Γ [modelDim]) :
    Term Γ [sourceCount completedCount partialPresent, modelDim] :=
  match partialPresent with
  | false => completed
  | true =>
      let partialRow : Term Γ [1, modelDim] :=
        Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
          (.cons partialState .nil)
      Term.op (NN.GraphSpec.DAG.PrimOp.concatAxis
          [completedCount, modelDim] 0 completedCount 1)
        (.cons completed (.cons partialRow .nil))

/-- Graph term that appends the current sequence residual to completed depth blocks. -/
def channelSourcesTerm {Γ : List Shape} (completedCount modelDim : Nat)
    (completed : Term Γ [completedCount, modelDim])
    (partialState sequenceOutput : Term Γ [modelDim]) :
    Term Γ [completedCount + 1, modelDim] :=
  let current := Term.op (NN.GraphSpec.DAG.PrimOp.add [modelDim])
    (.cons partialState (.cons sequenceOutput .nil))
  let currentRow : Term Γ [1, modelDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons current .nil)
  Term.op (NN.GraphSpec.DAG.PrimOp.concatAxis
      [completedCount, modelDim] 0 completedCount 1)
    (.cons completed (.cons currentRow .nil))

/-- Graph term for the first AttnRes retrieval and RMS normalization in a decoder layer. -/
def sequenceInputTerm {Γ : List Shape} (completedCount modelDim : Nat)
    (partialPresent : Bool) (hCompleted : 0 < completedCount) (hModel : 0 < modelDim)
    (query scale : Term Γ [modelDim])
    (completed : Term Γ [completedCount, modelDim])
    (partialState : Term Γ [modelDim]) : Term Γ [modelDim] :=
  let sources := sequenceSourcesTerm completedCount modelDim partialPresent completed partialState
  have hSources : 0 < sourceCount completedCount partialPresent := by
    simp [sourceCount]
    omega
  let retrieved := AttnRes.term (sourceCount completedCount partialPresent) modelDim hModel
    query sources
  Term.op (NN.GraphSpec.DAG.PrimOp.rmsNorm [] modelDim hModel)
    (.cons retrieved (.cons scale .nil))

/-- Graph term for the second AttnRes retrieval and RMS normalization in a decoder layer. -/
def channelInputTerm {Γ : List Shape} (completedCount modelDim : Nat)
    (hModel : 0 < modelDim) (query scale : Term Γ [modelDim])
    (completed : Term Γ [completedCount, modelDim])
    (partialState sequenceOutput : Term Γ [modelDim]) :
    Term Γ [modelDim] :=
  let sources := channelSourcesTerm completedCount modelDim completed partialState sequenceOutput
  let retrieved := AttnRes.term (completedCount + 1) modelDim hModel query sources
  Term.op (NN.GraphSpec.DAG.PrimOp.rmsNorm [] modelDim hModel)
    (.cons retrieved (.cons scale .nil))

/-- Graph term that accumulates sequence and channel outputs into the current depth block. -/
def finishPartialTerm {Γ : List Shape} (modelDim : Nat)
    (partialState sequenceOutput channelOutput : Term Γ [modelDim]) :
    Term Γ [modelDim] :=
  let withSequence := Term.op (NN.GraphSpec.DAG.PrimOp.add [modelDim])
    (.cons partialState (.cons sequenceOutput .nil))
  Term.op (NN.GraphSpec.DAG.PrimOp.add [modelDim])
    (.cons withSequence (.cons channelOutput .nil))

@[simp] theorem eval_sequenceSourcesTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (completedCount modelDim : Nat) (partialPresent : Bool)
    (completed : Term Γ [completedCount, modelDim])
    (partialState : Term Γ [modelDim]) :
    Term.eval env
        (sequenceSourcesTerm completedCount modelDim partialPresent completed partialState) =
      sequenceSources partialPresent (Term.eval env completed) (Term.eval env partialState) := by
  cases partialPresent
  · rfl
  · rfl

@[simp] theorem eval_channelSourcesTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (completedCount modelDim : Nat)
    (completed : Term Γ [completedCount, modelDim])
    (partialState sequenceOutput : Term Γ [modelDim]) :
    Term.eval env
        (channelSourcesTerm completedCount modelDim completed partialState sequenceOutput) =
      channelSources (Term.eval env completed) (Term.eval env partialState)
        (Term.eval env sequenceOutput) := by
  rfl

theorem eval_sequenceInputTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (completedCount modelDim : Nat) (partialPresent : Bool)
    (hCompleted : 0 < completedCount) (hModel : 0 < modelDim)
    (query scale : Term Γ [modelDim])
    (completed : Term Γ [completedCount, modelDim])
    (partialState : Term Γ [modelDim]) :
    Term.eval env
        (sequenceInputTerm completedCount modelDim partialPresent hCompleted hModel query scale
          completed partialState) =
      sequenceInput (completedCount := completedCount) (modelDim := modelDim)
        partialPresent hCompleted hModel (Term.eval env query) (Term.eval env scale)
        (Term.eval env completed) (Term.eval env partialState) := by
  simp only [sequenceInputTerm, sequenceInput, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.rmsNorm_specFwd]
  rw [AttnRes.eval_term, eval_sequenceSourcesTerm]
  change NN.GraphSpec.DAG.PrimOp.Internal.rmsNormVectorSpec hModel _ _ = _
  exact GraphSpec.rmsNormVectorSpec_eq_scale hModel _ _

theorem eval_channelInputTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (completedCount modelDim : Nat) (hModel : 0 < modelDim)
    (query scale : Term Γ [modelDim])
    (completed : Term Γ [completedCount, modelDim])
    (partialState sequenceOutput : Term Γ [modelDim]) :
    Term.eval env
        (channelInputTerm completedCount modelDim hModel query scale completed partialState
          sequenceOutput) =
      channelInput (completedCount := completedCount) (modelDim := modelDim)
        hModel (Term.eval env query) (Term.eval env scale)
        (Term.eval env completed) (Term.eval env partialState) (Term.eval env sequenceOutput) := by
  simp only [channelInputTerm, channelInput, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.rmsNorm_specFwd]
  rw [AttnRes.eval_term, eval_channelSourcesTerm]
  change NN.GraphSpec.DAG.PrimOp.Internal.rmsNormVectorSpec hModel _ _ = _
  exact GraphSpec.rmsNormVectorSpec_eq_scale hModel _ _

@[simp] theorem eval_finishPartialTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (modelDim : Nat)
    (partialState sequenceOutput channelOutput : Term Γ [modelDim]) :
    Term.eval env (finishPartialTerm modelDim partialState sequenceOutput channelOutput) =
      finishPartial (Term.eval env partialState) (Term.eval env sequenceOutput)
        (Term.eval env channelOutput) := by
  rfl

end PackedDepth

namespace Backbone

/-- Shapes of the learned AttnRes queries and RMSNorm scales, used by every graph builder. -/
abbrev DepthParams (modelDim : Nat) : List Shape :=
  [ [modelDim], [modelDim],
    [modelDim], [modelDim] ]

/-- Packed depth memory supplied to one statically positioned decoder layer. -/
abbrev DepthInputs (completedCount modelDim : Nat) : List Shape :=
  [[completedCount, modelDim], [modelDim]]

/-- Zero queries and unit normalization scales for a freshly initialized depth controller. -/
def initialDepthParams (modelDim : Nat) : TorchLean.TensorPack Float (DepthParams modelDim) :=
  .cons (Tensor.zeros [modelDim]) <|
    .cons (Tensor.ones [modelDim]) <|
      .cons (Tensor.zeros [modelDim]) <|
        .cons (Tensor.ones [modelDim]) .nil

/-- Extract a layer's learned depth queries and normalization scales in graph order. -/
def depthParameters {α : Type} [Storage α] {cfg : TextConfig} {decayRank : Nat}
    (layer : BackboneLayer α cfg decayRank) : TorchLean.TensorPack α (DepthParams cfg.hiddenDim) :=
  .cons layer.sequenceQuery <| .cons layer.sequenceNormScale <|
    .cons layer.channelQuery <| .cons layer.channelNormScale .nil

/-- Package completed depth blocks and the current partial block in graph order. -/
def depthInputs {α : Type} [Storage α] {completedCount modelDim : Nat}
    (completed : Tensor α [completedCount, modelDim])
    (partialState : Tensor α [modelDim]) :
    TorchLean.TensorPack α (DepthInputs completedCount modelDim) :=
  .cons completed (.cons partialState .nil)

end Backbone
end GraphSpec
end KimiK3
