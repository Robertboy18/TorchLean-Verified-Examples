/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Primitives
public import KimiK3.Sequence
public import NN.Proofs.Tensor.Basic.LinearAlgebra

/-!
# Executable Kimi Delta Attention recurrence

This module lowers one KDA recurrence to TorchLean's typed DAG. The graph has no parameters because
it starts after the query, key, value, retention, and write-strength projections. It consumes the
previous recurrent state and those projected values, then returns both the next state and the
current-token readout.

The state update is built from ordinary tensor operations. In particular, the graph does not hide
`KDA.update` inside a custom runtime call. This makes the recurrence available to every TorchLean
backend and exposes all of its arithmetic to autograd and numerical analysis.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec
namespace KDA

open Spec
open Spec.Tensor
open NN.GraphSpec.DAG
open Runtime.Autograd.Torch

attribute [local simp] Spec.Tensor.mapSpec_dim

/-- Inputs to one projected KDA step, beginning with the recurrent state. -/
abbrev Inputs (keyDim valueDim : Nat) : List Shape :=
  [ .dim keyDim (.dim valueDim .scalar),
    .dim keyDim .scalar,
    .dim keyDim .scalar,
    .dim valueDim .scalar,
    .dim keyDim .scalar,
    .scalar ]

/-- The updated recurrent state and current-token value readout. -/
abbrev Outputs (keyDim valueDim : Nat) : List Shape :=
  [.dim keyDim (.dim valueDim .scalar), .dim valueDim .scalar]

/-- One projected KDA step represented as a shared, multi-output TorchLean graph. -/
def model (keyDim valueDim : Nat) :
    NN.GraphSpec.DAG.MultiModel [] (Inputs keyDim valueDim) (Outputs keyDim valueDim) :=
  let stateShape := .dim keyDim (.dim valueDim .scalar)
  let keyShape := .dim keyDim .scalar
  let valueShape := .dim valueDim .scalar
  let Γ := [] ++ Inputs keyDim valueDim
  let state : Term Γ stateShape := Term.var .head
  let queryVar : Var Γ keyShape := .tail .head
  let key : Term Γ keyShape := Term.var (.tail (.tail .head))
  let value : Term Γ valueShape := Term.var (.tail (.tail (.tail .head)))
  let retention : Term Γ keyShape := Term.var (.tail (.tail (.tail (.tail .head))))
  let writeStrength : Term Γ .scalar :=
    Term.var (.tail (.tail (.tail (.tail (.tail .head)))))
  let decayed : Term Γ stateShape :=
    Term.op (PrimOp.rowScale keyDim valueDim) (.cons retention (.cons state .nil))
  let correction : Term Γ valueShape :=
    Term.op (PrimOp.vecMat keyDim valueDim) (.cons key (.cons decayed .nil))
  let correctionOuter : Term Γ stateShape :=
    Term.op (PrimOp.outer keyDim valueDim) (.cons key (.cons correction .nil))
  let writeOuter : Term Γ stateShape :=
    Term.op (PrimOp.outer keyDim valueDim) (.cons key (.cons value .nil))
  let scaledCorrection : Term Γ stateShape :=
    Term.op (PrimOp.scalarMul stateShape)
      (.cons writeStrength (.cons correctionOuter .nil))
  let scaledWrite : Term Γ stateShape :=
    Term.op (PrimOp.scalarMul stateShape) (.cons writeStrength (.cons writeOuter .nil))
  let corrected : Term Γ stateShape :=
    Term.op (PrimOp.sub stateShape) (.cons decayed (.cons scaledCorrection .nil))
  let nextState : Term Γ stateShape :=
    Term.op (NN.GraphSpec.DAG.PrimOp.add stateShape)
      (.cons corrected (.cons scaledWrite .nil))
  let nextState' : Term (Γ ++ [stateShape]) stateShape := Term.var (Var.last Γ)
  let query' : Term (Γ ++ [stateShape]) keyShape := Term.var (Var.weakenRight queryVar)
  let output : Term (Γ ++ [stateShape]) valueShape :=
    Term.op (PrimOp.vecMat keyDim valueDim) (.cons query' (.cons nextState' .nil))
  { initParams := .nil
    body := Block.let1 nextState <| Block.ret <|
      .cons nextState' (.cons output .nil) }

/-- Package the mathematical KDA state and projected input in the graph ABI. -/
def inputs {α : Type} {keyDim valueDim : Nat}
    (state : KimiK3.KDA.State α keyDim valueDim)
    (input : KDAStepInput α keyDim valueDim) : TorchLean.TensorPack α (Inputs keyDim valueDim) :=
  .cons state <| .cons input.query <| .cons input.key <| .cons input.value <|
    .cons input.retention <| .cons (.scalar input.writeStrength) .nil

/-- The executable KDA graph denotes the mathematical recurrence and its state-dependent readout. -/
theorem specFwd_eq_step {keyDim valueDim : Nat}
    (state : KimiK3.KDA.State ℝ keyDim valueDim)
    (input : KDAStepInput ℝ keyDim valueDim) :
    (model keyDim valueDim).specFwd .nil (inputs state input) =
      .cons (KimiK3.KDA.update state input)
        (.cons (KimiK3.KDA.read (KimiK3.KDA.update state input) input.query) .nil) := by
  simp only [model, inputs, NN.GraphSpec.DAG.MultiModel.specFwd, Block.eval,
    Term.eval, Term.evalArgs]
  simp only [Env.tget_append_last, Env.tget_append_weakenRight]
  simp [TorchLean.TensorPack.append, Env.tget, PrimOp.rowScale, PrimOp.vecMat,
    PrimOp.outer, PrimOp.scalarMul, PrimOp.sub, NN.GraphSpec.DAG.PrimOp.add,
    KimiK3.KDA.update, KimiK3.KDA.read]

/-! ## Complete multi-head layer -/

/-- Projection, short-convolution, write-gate, and decay parameters for a packed KDA layer.

The leading axis of every head-local tensor is the head axis. This is a view of `KDALayer.head`,
not a second parameter record: `layerParameters` below packs the existing dependent function into
the batched tensor ABI consumed by `bmm` and elementwise operations.
-/
abbrev PreparationParams (modelDim heads keyDim valueDim convWidth decayRank : Nat) : List Shape :=
  [ .dim heads (.dim modelDim (.dim keyDim .scalar)),
    .dim heads (.dim modelDim (.dim keyDim .scalar)),
    .dim heads (.dim modelDim (.dim valueDim .scalar)),
    .dim heads (.dim convWidth (.dim keyDim .scalar)),
    .dim heads (.dim keyDim .scalar),
    .dim heads (.dim convWidth (.dim keyDim .scalar)),
    .dim heads (.dim keyDim .scalar),
    .dim heads (.dim convWidth (.dim valueDim .scalar)),
    .dim heads (.dim valueDim .scalar),
    .dim heads (.dim modelDim .scalar),
    .dim heads (.dim modelDim (.dim decayRank .scalar)),
    .dim heads (.dim decayRank (.dim keyDim .scalar)),
    .dim heads (.dim keyDim .scalar),
    .dim heads .scalar ]

/-- Gate, projection, and normalization parameters used after the recurrent update. -/
abbrev ReadoutParams (modelDim heads valueDim : Nat) : List Shape :=
  [ .dim modelDim (.dim heads (.dim valueDim .scalar)),
    .dim heads (.dim valueDim (.dim modelDim .scalar)),
    .dim heads (.dim valueDim .scalar) ]

/-- Parameter shapes for a complete packed KDA layer. -/
abbrev LayerParams (modelDim heads keyDim valueDim convWidth decayRank : Nat) : List Shape :=
  PreparationParams modelDim heads keyDim valueDim convWidth decayRank ++
    ReadoutParams modelDim heads valueDim

/-- Runtime inputs for one fixed-window KDA token step. -/
abbrev LayerInputs (modelDim heads keyDim valueDim convWidth : Nat) : List Shape :=
  [ .dim convWidth (.dim modelDim .scalar),
    .dim heads (.dim keyDim (.dim valueDim .scalar)),
    .scalar,
    .scalar ]

/-- Updated recurrent matrices followed by the projected model-width output. -/
abbrev LayerOutputs (modelDim heads keyDim valueDim : Nat) : List Shape :=
  [.dim heads (.dim keyDim (.dim valueDim .scalar)), .dim modelDim .scalar]

/-- Zero-filled defaults for standalone graph construction.

Training and checkpoint loading replace every entry. The semantic theorem quantifies over an
arbitrary `KDALayer`, so no property depends on this initialization.
-/
def initialLayerParams (modelDim heads keyDim valueDim convWidth decayRank : Nat) :
    TorchLean.TensorPack Float (LayerParams modelDim heads keyDim valueDim convWidth decayRank) :=
  .cons (Spec.fill 0 (.dim heads (.dim modelDim (.dim keyDim .scalar)))) <|
  .cons (Spec.fill 0 (.dim heads (.dim modelDim (.dim keyDim .scalar)))) <|
  .cons (Spec.fill 0 (.dim heads (.dim modelDim (.dim valueDim .scalar)))) <|
  .cons (Spec.fill 0 (.dim heads (.dim convWidth (.dim keyDim .scalar)))) <|
  .cons (Spec.fill 0 (.dim heads (.dim keyDim .scalar))) <|
  .cons (Spec.fill 0 (.dim heads (.dim convWidth (.dim keyDim .scalar)))) <|
  .cons (Spec.fill 0 (.dim heads (.dim keyDim .scalar))) <|
  .cons (Spec.fill 0 (.dim heads (.dim convWidth (.dim valueDim .scalar)))) <|
  .cons (Spec.fill 0 (.dim heads (.dim valueDim .scalar))) <|
  .cons (Spec.fill 0 (.dim heads (.dim modelDim .scalar))) <|
  .cons (Spec.fill 0 (.dim heads (.dim modelDim (.dim decayRank .scalar)))) <|
  .cons (Spec.fill 0 (.dim heads (.dim decayRank (.dim keyDim .scalar)))) <|
  .cons (Spec.fill 0 (.dim heads (.dim keyDim .scalar))) <|
  .cons (Spec.fill 0 (.dim heads .scalar)) <|
  .cons (Spec.fill 0 (.dim modelDim (.dim heads (.dim valueDim .scalar)))) <|
  .cons (Spec.fill 0 (.dim heads (.dim valueDim (.dim modelDim .scalar)))) <|
  .cons (Spec.fill 0 (.dim heads (.dim valueDim .scalar))) .nil

/-- Pack the existing head-indexed KDA records into GraphSpec's parameter ABI. -/
def layerParameters {α : Type} {modelDim heads keyDim valueDim convWidth decayRank : Nat}
    (layer : KDALayer α modelDim heads keyDim valueDim convWidth decayRank) :
    TorchLean.TensorPack α (LayerParams modelDim heads keyDim valueDim convWidth decayRank) :=
  .cons (Tensor.dim fun head => (layer.head head).queryWeight) <|
  .cons (Tensor.dim fun head => (layer.head head).keyWeight) <|
  .cons (Tensor.dim fun head => (layer.head head).valueWeight) <|
  .cons (Tensor.dim fun head => (layer.head head).queryConv.weight) <|
  .cons (Tensor.dim fun head => (layer.head head).queryConv.bias) <|
  .cons (Tensor.dim fun head => (layer.head head).keyConv.weight) <|
  .cons (Tensor.dim fun head => (layer.head head).keyConv.bias) <|
  .cons (Tensor.dim fun head => (layer.head head).valueConv.weight) <|
  .cons (Tensor.dim fun head => (layer.head head).valueConv.bias) <|
  .cons (Tensor.dim fun head => (layer.head head).betaWeight) <|
  .cons (Tensor.dim fun head => (layer.head head).decayDown) <|
  .cons (Tensor.dim fun head => (layer.head head).decayUp) <|
  .cons (Tensor.dim fun head => (layer.head head).decayBias) <|
  .cons (Tensor.dim fun head => Tensor.scalar (layer.head head).decayLogScale) <|
  .cons layer.gateWeight <| .cons layer.outputWeight <| .cons layer.outputNormScale .nil

/-- Package a fixed causal window and recurrent state for the complete KDA graph. -/
def layerInputs {α : Type} {modelDim heads keyDim valueDim convWidth : Nat}
    (window : Tensor α (.dim convWidth (.dim modelDim .scalar)))
    (state : KDALayer.State α heads keyDim valueDim) (logFloor epsilon : α) :
    TorchLean.TensorPack α (LayerInputs modelDim heads keyDim valueDim convWidth) :=
  .cons window <| .cons state <| .cons (.scalar logFloor) <|
    .cons (.scalar epsilon) .nil

/-- Uncast graph term underlying `rollWindowTerm`. -/
def rollWindowRawTerm {Γ : List Shape} (modelDim convWidth : Nat)
    (current : Term Γ (.dim modelDim .scalar))
    (previous : Term Γ (.dim convWidth (.dim modelDim .scalar))) :
    Term Γ (.dim (1 + (convWidth - 1)) (.dim modelDim .scalar)) :=
  let currentRow : Term Γ (.dim 1 (.dim modelDim .scalar)) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size])) (.cons current .nil)
  let retained : Term Γ (.dim (convWidth - 1) (.dim modelDim .scalar)) :=
    Term.op
      (NN.GraphSpec.DAG.PrimOp.sliceAxisRange
        (.dim convWidth (.dim modelDim .scalar)) 0 0 (convWidth - 1)
          (by simpa only [zero_add, Shape.axisSize_zero] using Nat.sub_le convWidth 1))
      (.cons previous .nil)
  Term.op
    (NN.GraphSpec.DAG.PrimOp.concatAxis
      (.dim 1 (.dim modelDim .scalar)) 0 1 (convWidth - 1))
    (.cons currentRow (.cons retained .nil))

/-- Graph term for the fixed KDA history update.

This uses only reshape, leading-axis slice, and concatenation, so eager and compiled execution use
their ordinary differentiable implementations.  No KDA-specific runtime primitive is introduced.
-/
def rollWindowTerm {Γ : List Shape} (modelDim convWidth : Nat) (hWidth : 0 < convWidth)
    (current : Term Γ (.dim modelDim .scalar))
    (previous : Term Γ (.dim convWidth (.dim modelDim .scalar))) :
    Term Γ (.dim convWidth (.dim modelDim .scalar)) :=
  Term.cast (rollWindowRawTerm modelDim convWidth current previous)
    (KDALayer.rollWindowShape hWidth)

/-- The graph-level rolling window has exactly the KDA streaming semantics. -/
@[simp] theorem eval_rollWindowTerm {Γ : List Shape}
    (env : TorchLean.TensorPack ℝ Γ) (modelDim convWidth : Nat) (hWidth : 0 < convWidth)
    (current : Term Γ (.dim modelDim .scalar))
    (previous : Term Γ (.dim convWidth (.dim modelDim .scalar))) :
    Term.eval env (rollWindowTerm modelDim convWidth hWidth current previous) =
      KDALayer.rollWindow hWidth (Term.eval env current) (Term.eval env previous) := by
  simp [rollWindowTerm, rollWindowRawTerm, Term.eval, Term.eval_cast, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.reshape_specFwd,
    NN.GraphSpec.DAG.PrimOp.sliceAxisRange,
    NN.GraphSpec.DAG.PrimOp.concatAxis, NN.GraphSpec.DAG.PrimOp.concatAxisSpec,
    Shape.replaceAxis, Tensor.permuteByAdjacentSwaps,
    KDALayer.rollWindow, KDALayer.rollWindowRaw]
  cases hCurrent : Term.eval env current
  cases hPrevious : Term.eval env previous
  rfl

/-- Project a fixed causal window and apply one packed depthwise short convolution.

The same term is used for the query, key, and value paths. Naming it once keeps the complete KDA
graph readable while leaving every arithmetic operation visible to GraphSpec. -/
def shortConvTerm {Γ : List Shape} (heads convWidth modelDim channels : Nat)
    (hHeads : 0 < heads) (hWidth : 0 < convWidth) (hChannels : 0 < channels)
    (window : Term Γ (.dim convWidth (.dim modelDim .scalar)))
    (projection : Term Γ (.dim heads (.dim modelDim (.dim channels .scalar))))
    (kernel : Term Γ (.dim heads (.dim convWidth (.dim channels .scalar))))
    (bias : Term Γ (.dim heads (.dim channels .scalar))) :
    Term Γ (.dim heads (.dim channels .scalar)) :=
  letI : Fact (0 < heads) := ⟨hHeads⟩
  letI : Fact (0 < channels) := ⟨hChannels⟩
  let projected := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul .scalar [heads] [heads]
      convWidth modelDim channels (.scalarTo [heads]) (.refl [heads]))
    (.cons window (.cons projection .nil))
  let summed := Term.op
    (NN.GraphSpec.DAG.PrimOp.batchedDepthwiseWeightedSum
      heads convWidth channels hHeads hWidth hChannels)
    (.cons projected (.cons kernel .nil))
  Term.op (NN.GraphSpec.DAG.PrimOp.add (.dim heads (.dim channels .scalar)))
    (.cons summed (.cons bias .nil))

/-- Pure semantics of the packed projection and depthwise short-convolution term. -/
@[simp] theorem eval_shortConvTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (heads convWidth modelDim channels : Nat)
    (hHeads : 0 < heads) (hWidth : 0 < convWidth) (hChannels : 0 < channels)
    (window : Term Γ (.dim convWidth (.dim modelDim .scalar)))
    (projection : Term Γ (.dim heads (.dim modelDim (.dim channels .scalar))))
    (kernel : Term Γ (.dim heads (.dim convWidth (.dim channels .scalar))))
    (bias : Term Γ (.dim heads (.dim channels .scalar))) :
    Term.eval env (shortConvTerm heads convWidth modelDim channels hHeads hWidth hChannels
      window projection kernel bias) =
      Spec.Tensor.addSpec
        (Tensor.dim fun head =>
          Spec.Tensor.reduceSum 0
            (Spec.Tensor.mulSpec
              (Spec.matMulSpec (Term.eval env window)
                (Spec.get (Term.eval env projection) head))
              (Spec.get (Term.eval env kernel) head))
            (Shape.hasNonemptyAxisZeroOfPos hWidth).proof)
        (Term.eval env bias) := by
  letI : Shape.HasNonemptyAxis 0 (.dim convWidth (.dim channels .scalar)) :=
    Shape.hasNonemptyAxisZeroOfPos hWidth
  simp only [shortConvTerm, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.matmul_specFwd,
    NN.GraphSpec.DAG.PrimOp.batchedDepthwiseWeightedSum_specFwd,
    NN.GraphSpec.DAG.PrimOp.add_specFwd]
  simp only [Spec.Tensor.matmulSpec,
    Spec.Tensor.Internal.matmulCommonBatchSpec,
    Spec.Tensor.Internal.extendBroadcastSuffix,
    Spec.Tensor.broadcastTo]
  cases hWindow : Term.eval env window
  cases hProjection : Term.eval env projection
  rw [Spec.Tensor.Internal.broadcastTo_extendBroadcastSuffix_scalarTo_dim,
    Spec.Tensor.Internal.broadcastTo_extendBroadcastSuffix_refl]
  rfl

/-- Short-convolution projection followed by SiLU. This is the complete value preparation path. -/
def activatedProjectionTerm {Γ : List Shape}
    (heads convWidth modelDim channels : Nat) (hHeads : 0 < heads)
    (hWidth : 0 < convWidth) (hChannels : 0 < channels)
    (window : Term Γ (.dim convWidth (.dim modelDim .scalar)))
    (projection : Term Γ (.dim heads (.dim modelDim (.dim channels .scalar))))
    (kernel : Term Γ (.dim heads (.dim convWidth (.dim channels .scalar))))
    (bias : Term Γ (.dim heads (.dim channels .scalar))) :
    Term Γ (.dim heads (.dim channels .scalar)) :=
  let projected := shortConvTerm heads convWidth modelDim channels hHeads hWidth hChannels
    window projection kernel bias
  Term.op (NN.GraphSpec.DAG.PrimOp.silu (.dim heads (.dim channels .scalar)))
    (.cons projected .nil)

/-- Evaluation of the activated projection is SiLU applied to the shared short convolution. -/
@[simp] theorem eval_activatedProjectionTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (heads convWidth modelDim channels : Nat)
    (hHeads : 0 < heads) (hWidth : 0 < convWidth) (hChannels : 0 < channels)
    (window : Term Γ (.dim convWidth (.dim modelDim .scalar)))
    (projection : Term Γ (.dim heads (.dim modelDim (.dim channels .scalar))))
    (kernel : Term Γ (.dim heads (.dim convWidth (.dim channels .scalar))))
    (bias : Term Γ (.dim heads (.dim channels .scalar))) :
    Term.eval env (activatedProjectionTerm heads convWidth modelDim channels
      hHeads hWidth hChannels window projection kernel bias) =
      Spec.Tensor.mapSpec Activation.Math.swishSpec
        (Term.eval env (shortConvTerm heads convWidth modelDim channels
          hHeads hWidth hChannels window projection kernel bias)) := by
  simp only [activatedProjectionTerm, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.silu_specFwd, Activation.swishSpec]

/-- Query/key preparation: short convolution, SiLU, and row-wise L2 normalization. -/
def normalizedProjectionTerm {Γ : List Shape}
    (heads convWidth modelDim channels : Nat) (hHeads : 0 < heads)
    (hWidth : 0 < convWidth) (hChannels : 0 < channels)
    (window : Term Γ (.dim convWidth (.dim modelDim .scalar)))
    (projection : Term Γ (.dim heads (.dim modelDim (.dim channels .scalar))))
    (kernel : Term Γ (.dim heads (.dim convWidth (.dim channels .scalar))))
    (bias : Term Γ (.dim heads (.dim channels .scalar))) (epsilon : Term Γ .scalar) :
    Term Γ (.dim heads (.dim channels .scalar)) :=
  let activated := activatedProjectionTerm heads convWidth modelDim channels hHeads hWidth
    hChannels window projection kernel bias
  Term.op (NN.GraphSpec.DAG.PrimOp.l2Normalize [heads] channels hChannels)
    (.cons activated (.cons epsilon .nil))

/-- Evaluation of the normalized projection normalizes each head along its channel axis. -/
@[simp] theorem eval_normalizedProjectionTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (heads convWidth modelDim channels : Nat)
    (hHeads : 0 < heads) (hWidth : 0 < convWidth) (hChannels : 0 < channels)
    (window : Term Γ (.dim convWidth (.dim modelDim .scalar)))
    (projection : Term Γ (.dim heads (.dim modelDim (.dim channels .scalar))))
    (kernel : Term Γ (.dim heads (.dim convWidth (.dim channels .scalar))))
    (bias : Term Γ (.dim heads (.dim channels .scalar))) (epsilon : Term Γ .scalar) :
    Term.eval env (normalizedProjectionTerm heads convWidth modelDim channels
      hHeads hWidth hChannels window projection kernel bias epsilon) =
      Tensor.dim fun head => KimiK3.Normalize.regularizedL2
        (Spec.get
          (Term.eval env (activatedProjectionTerm heads convWidth modelDim channels
            hHeads hWidth hChannels window projection kernel bias)) head)
        (Tensor.item (Term.eval env epsilon)) := by
  cases hEpsilon : Term.eval env epsilon with
  | scalar epsilonValue =>
      simp only [normalizedProjectionTerm, Term.eval_op, Term.evalArgs,
        NN.GraphSpec.DAG.PrimOp.l2Normalize_specFwd,
        Tensor.item_scalar, hEpsilon]
      cases hActivated : Term.eval env
        (activatedProjectionTerm heads convWidth modelDim channels
          hHeads hWidth hChannels window projection kernel bias)
      rw [NN.GraphSpec.DAG.PrimOp.l2NormalizeSemantics_dim]
      congr

/-- Multiply one shared vector by a matrix for each head. -/
def batchedSharedVecMatTerm {Γ : List Shape} (heads inputDim outputDim : Nat)
    (vector : Term Γ (.dim inputDim .scalar))
    (matrices : Term Γ (.dim heads (.dim inputDim (.dim outputDim .scalar)))) :
    Term Γ (.dim heads (.dim outputDim .scalar)) :=
  Term.op (NN.GraphSpec.DAG.PrimOp.broadcastVecMat .scalar [heads] [heads]
    inputDim outputDim (.scalarTo [heads]) (.refl [heads]))
    (.cons vector (.cons matrices .nil))

/-- Multiply one vector by one matrix for each head. -/
def batchedVecMatTerm {Γ : List Shape} (heads inputDim outputDim : Nat)
    (vectors : Term Γ (.dim heads (.dim inputDim .scalar)))
    (matrices : Term Γ (.dim heads (.dim inputDim (.dim outputDim .scalar)))) :
    Term Γ (.dim heads (.dim outputDim .scalar)) :=
  Term.op (NN.GraphSpec.DAG.PrimOp.broadcastVecMat [heads] [heads] [heads]
    inputDim outputDim (.refl [heads]) (.refl [heads]))
    (.cons vectors (.cons matrices .nil))

/-- Pure semantics of a shared vector multiplied by one matrix per head. -/
@[simp] theorem eval_batchedSharedVecMatTerm {Γ : List Shape}
    (env : TorchLean.TensorPack ℝ Γ) (heads inputDim outputDim : Nat)
    (vector : Term Γ (.dim inputDim .scalar))
    (matrices : Term Γ (.dim heads (.dim inputDim (.dim outputDim .scalar)))) :
    Term.eval env (batchedSharedVecMatTerm heads inputDim outputDim vector matrices) =
      .dim (fun head => vecMatMulSpec (Term.eval env vector)
        (Spec.get (Term.eval env matrices) head)) := by
  simp only [batchedSharedVecMatTerm, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.broadcastVecMat_specFwd,
    NN.GraphSpec.DAG.PrimOp.Internal.vecMatCommonBatchSpec,
    Spec.Tensor.Internal.extendBroadcastSuffix, Spec.Tensor.broadcastTo]
  cases hVector : Term.eval env vector
  cases hMatrices : Term.eval env matrices
  rw [Spec.Tensor.Internal.broadcastTo_extendBroadcastSuffix_scalarTo_dim,
    Spec.Tensor.Internal.broadcastTo_extendBroadcastSuffix_refl,
    NN.GraphSpec.DAG.PrimOp.Internal.vecMatCommonBatchSpec_dim]
  rfl

/-- Pure semantics of independently multiplying one vector and matrix per head. -/
@[simp] theorem eval_batchedVecMatTerm {Γ : List Shape}
    (env : TorchLean.TensorPack ℝ Γ) (heads inputDim outputDim : Nat)
    (vectors : Term Γ (.dim heads (.dim inputDim .scalar)))
    (matrices : Term Γ (.dim heads (.dim inputDim (.dim outputDim .scalar)))) :
    Term.eval env (batchedVecMatTerm heads inputDim outputDim vectors matrices) =
      .dim (fun head => vecMatMulSpec
        (Spec.get (Term.eval env vectors) head)
        (Spec.get (Term.eval env matrices) head)) := by
  simp only [batchedVecMatTerm, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.broadcastVecMat_specFwd,
    NN.GraphSpec.DAG.PrimOp.Internal.vecMatCommonBatchSpec,
    Spec.Tensor.Internal.extendBroadcastSuffix, Spec.Tensor.broadcastTo]
  cases hVectors : Term.eval env vectors
  cases hMatrices : Term.eval env matrices
  rw [Spec.Tensor.Internal.broadcastTo_extendBroadcastSuffix_refl,
    Spec.Tensor.Internal.broadcastTo_extendBroadcastSuffix_refl,
    NN.GraphSpec.DAG.PrimOp.Internal.vecMatCommonBatchSpec_dim]
  rfl

/-- Shapes of the current token and the five quantities prepared before the recurrent update.

This is an `Args` signature rather than a second KDA state record.  It records the equation-level
boundary between the projection/gating calculation and the recurrent matrix update: current token,
normalized query, normalized key, activated value, retention, and write strength, in that order. -/
abbrev Prepared (modelDim heads keyDim valueDim : Nat) : List Shape :=
  [ .dim modelDim .scalar,
    .dim heads (.dim keyDim .scalar),
    .dim heads (.dim keyDim .scalar),
    .dim heads (.dim valueDim .scalar),
    .dim heads (.dim keyDim .scalar),
    .dim heads .scalar ]

/-- Prepare the query, key, value, retention, and write-strength terms for one KDA token.

The function follows the KDA equations in their dependency order.  Query and key use the same
short-convolution and SiLU path as value, followed by row-wise L2 normalization.  Retention is
`exp(logFloor * sigmoid(exp(logScale) * decayLogit))`; write strength is the sigmoid of the
per-head linear projection named `beta` in the paper. -/
def prepareTerms {Γ : List Shape}
    (modelDim heads keyDim valueDim convWidth decayRank : Nat)
    (hHeads : 0 < heads) (hKey : 0 < keyDim) (hValue : 0 < valueDim)
    (hWidth : 0 < convWidth)
    (queryWeight keyWeight : Term Γ (.dim heads (.dim modelDim (.dim keyDim .scalar))))
    (valueWeight : Term Γ (.dim heads (.dim modelDim (.dim valueDim .scalar))))
    (queryConvWeight keyConvWeight :
      Term Γ (.dim heads (.dim convWidth (.dim keyDim .scalar))))
    (queryConvBias keyConvBias : Term Γ (.dim heads (.dim keyDim .scalar)))
    (valueConvWeight : Term Γ (.dim heads (.dim convWidth (.dim valueDim .scalar))))
    (valueConvBias : Term Γ (.dim heads (.dim valueDim .scalar)))
    (betaWeight : Term Γ (.dim heads (.dim modelDim .scalar)))
    (decayDown : Term Γ (.dim heads (.dim modelDim (.dim decayRank .scalar))))
    (decayUp : Term Γ (.dim heads (.dim decayRank (.dim keyDim .scalar))))
    (decayBias : Term Γ (.dim heads (.dim keyDim .scalar)))
    (decayLogScale : Term Γ (.dim heads .scalar))
    (window : Term Γ (.dim convWidth (.dim modelDim .scalar)))
    (logFloor epsilon : Term Γ .scalar) :
    Args Γ (Prepared modelDim heads keyDim valueDim) :=
  let current : Term Γ (.dim modelDim .scalar) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.select
      (.dim convWidth (.dim modelDim .scalar)) 0 ⟨0, hWidth⟩)
      (.cons window .nil)
  let query := normalizedProjectionTerm heads convWidth modelDim keyDim hHeads hWidth hKey
    window queryWeight queryConvWeight queryConvBias epsilon
  let key := normalizedProjectionTerm heads convWidth modelDim keyDim hHeads hWidth hKey
    window keyWeight keyConvWeight keyConvBias epsilon
  let value := activatedProjectionTerm heads convWidth modelDim valueDim hHeads hWidth hValue
    window valueWeight valueConvWeight valueConvBias
  let decayHidden : Term Γ (.dim heads (.dim decayRank .scalar)) :=
    batchedSharedVecMatTerm heads modelDim decayRank current decayDown
  let decayLogit : Term Γ (.dim heads (.dim keyDim .scalar)) :=
    batchedVecMatTerm heads decayRank keyDim decayHidden decayUp
  let shiftedDecayLogit :=
    Term.op (NN.GraphSpec.DAG.PrimOp.add (.dim heads (.dim keyDim .scalar)))
      (.cons decayLogit (.cons decayBias .nil))
  let decayScale :=
    Term.op (NN.GraphSpec.DAG.PrimOp.exp (.dim heads .scalar)) (.cons decayLogScale .nil)
  let scaledDecayLogit :=
    Term.op (NN.GraphSpec.DAG.PrimOp.batchedScale heads (.dim keyDim .scalar))
      (.cons decayScale (.cons shiftedDecayLogit .nil))
  let decayGate :=
    Term.op (NN.GraphSpec.DAG.PrimOp.sigmoid (.dim heads (.dim keyDim .scalar)))
      (.cons scaledDecayLogit .nil)
  let retentionExponent :=
    Term.op (NN.GraphSpec.DAG.PrimOp.scalarMul (.dim heads (.dim keyDim .scalar)))
      (.cons logFloor (.cons decayGate .nil))
  let retention :=
    Term.op (NN.GraphSpec.DAG.PrimOp.exp (.dim heads (.dim keyDim .scalar)))
      (.cons retentionExponent .nil)
  let betaLogit : Term Γ (.dim heads .scalar) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.batchedSharedDot heads modelDim)
      (.cons current (.cons betaWeight .nil))
  let beta :=
    Term.op (NN.GraphSpec.DAG.PrimOp.sigmoid (.dim heads .scalar)) (.cons betaLogit .nil)
  .cons current <|
    .cons query (.cons key (.cons value (.cons retention (.cons beta .nil))))

/-- Read the updated recurrent matrix with each query and normalize each head independently. -/
def normalizedHeadReadTerm {Γ : List Shape} (heads keyDim valueDim : Nat)
    (hHeads : 0 < heads) (hValue : 0 < valueDim)
    (query : Term Γ (.dim heads (.dim keyDim .scalar)))
    (state : Term Γ (.dim heads (.dim keyDim (.dim valueDim .scalar))))
    (outputNormScale : Term Γ (.dim heads (.dim valueDim .scalar))) :
    Term Γ (.dim heads (.dim valueDim .scalar)) :=
  let headOutput := batchedVecMatTerm heads keyDim valueDim query state
  Term.op (NN.GraphSpec.DAG.PrimOp.rmsNormElementwise [heads] valueDim hValue)
    (.cons headOutput (.cons outputNormScale .nil))

/-- The normalized head-read term denotes the corresponding KDA head equations. -/
@[simp] theorem eval_normalizedHeadReadTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (heads keyDim valueDim : Nat) (hHeads : 0 < heads) (hValue : 0 < valueDim)
    (query : Term Γ (.dim heads (.dim keyDim .scalar)))
    (state : Term Γ (.dim heads (.dim keyDim (.dim valueDim .scalar))))
    (outputNormScale : Term Γ (.dim heads (.dim valueDim .scalar))) :
    Term.eval env (normalizedHeadReadTerm heads keyDim valueDim hHeads hValue
      query state outputNormScale) =
      .dim (fun head => RMSNorm.scale
        (KDA.read (Spec.get (Term.eval env state) head)
          (Spec.get (Term.eval env query) head))
        (Spec.get (Term.eval env outputNormScale) head)) := by
  simp only [normalizedHeadReadTerm, Term.eval_op, Term.evalArgs,
    eval_batchedVecMatTerm,
    NN.GraphSpec.DAG.PrimOp.rmsNormElementwise_specFwd, KDA.read]
  cases hScale : Term.eval env outputNormScale
  rw [NN.GraphSpec.DAG.PrimOp.rmsNormElementwiseSemantics_dim]
  congr
  funext head
  exact GraphSpec.rmsNormVectorSemantics_eq_scale hValue _ _

/-- Compute the full-rank output gate for every KDA head. -/
def outputGateTerm {Γ : List Shape} (modelDim heads valueDim : Nat)
    (current : Term Γ (.dim modelDim .scalar))
    (gateWeight : Term Γ (.dim modelDim (.dim heads (.dim valueDim .scalar)))) :
    Term Γ (.dim heads (.dim valueDim .scalar)) :=
  let gateMatrices :=
    Term.op (NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth
      (.dim modelDim (.dim heads (.dim valueDim .scalar))) 0) (.cons gateWeight .nil)
  let gateLogit := batchedSharedVecMatTerm heads modelDim valueDim current gateMatrices
  let gate :=
    Term.op (NN.GraphSpec.DAG.PrimOp.sigmoid (.dim heads (.dim valueDim .scalar)))
      (.cons gateLogit .nil)
  gate

/-- The output-gate term has the packed coordinate semantics used by `KDALayer.gates`. -/
@[simp] theorem eval_outputGateTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (modelDim heads valueDim : Nat)
    (current : Term Γ (.dim modelDim .scalar))
    (gateWeight : Term Γ (.dim modelDim (.dim heads (.dim valueDim .scalar)))) :
    Term.eval env (outputGateTerm modelDim heads valueDim current gateWeight) =
      .dim (fun head => mapSpec Activation.Math.sigmoidSpec <|
        vecMatMulSpec (Term.eval env current) <|
          .dim (fun input => Spec.get (Spec.get (Term.eval env gateWeight) input) head)) := by
  simp only [outputGateTerm, Term.eval_op, Term.evalArgs,
    eval_batchedSharedVecMatTerm,
    NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth_specFwd,
    NN.GraphSpec.DAG.PrimOp.sigmoid_specFwd, Activation.sigmoidSpec]
  cases hGateWeight : Term.eval env gateWeight with
  | dim gateValues =>
      rw [Spec.Tensor.swapAdjacentAxes_zero]
      rfl

/-- Gate, project, and sum the packed KDA head outputs. -/
def projectHeadReadoutTerm {Γ : List Shape} (modelDim heads valueDim : Nat)
    (hModel : 0 < modelDim) (hHeads : 0 < heads)
    (gate normalizedHeadOutput : Term Γ (.dim heads (.dim valueDim .scalar)))
    (outputWeight : Term Γ (.dim heads (.dim valueDim (.dim modelDim .scalar)))) :
    Term Γ (.dim modelDim .scalar) :=
  letI : Fact (0 < modelDim) := ⟨hModel⟩
  letI : Fact (0 < heads) := ⟨hHeads⟩
  letI : NeZero modelDim := ⟨Nat.ne_of_gt hModel⟩
  letI : NeZero heads := ⟨Nat.ne_of_gt hHeads⟩
  letI : Shape.HasNonemptyAxis 0 (.dim heads (.dim modelDim .scalar)) :=
    Shape.hasNonemptyAxisZeroOfPos hHeads
  let gated := Term.op (NN.GraphSpec.DAG.PrimOp.mul (.dim heads (.dim valueDim .scalar)))
    (.cons gate (.cons normalizedHeadOutput .nil))
  let projectedHeads := batchedVecMatTerm heads valueDim modelDim gated outputWeight
  Term.op (NN.GraphSpec.DAG.PrimOp.reduceSum
    (.dim heads (.dim modelDim .scalar)) 0) (.cons projectedHeads .nil)

/-- The final packed projection is exactly a sum of gated per-head projections. -/
@[simp] theorem eval_projectHeadReadoutTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (modelDim heads valueDim : Nat) (hHeads : 0 < heads)
    (hModel : 0 < modelDim)
    (gate normalizedHeadOutput : Term Γ (.dim heads (.dim valueDim .scalar)))
    (outputWeight : Term Γ (.dim heads (.dim valueDim (.dim modelDim .scalar)))) :
    Term.eval env (projectHeadReadoutTerm modelDim heads valueDim hModel hHeads
      gate normalizedHeadOutput outputWeight) =
      Tensor.reduceSum 0
        (.dim (fun head => vecMatMulSpec
          (Spec.get (mulSpec (Term.eval env gate) (Term.eval env normalizedHeadOutput)) head)
          (Spec.get (Term.eval env outputWeight) head)))
        (Shape.hasNonemptyAxisZeroOfPos hHeads).proof := by
  letI : Shape.HasNonemptyAxis 0 (.dim heads (.dim modelDim .scalar)) :=
    Shape.hasNonemptyAxisZeroOfPos hHeads
  simp only [projectHeadReadoutTerm, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.mul_specFwd,
    eval_batchedVecMatTerm,
    NN.GraphSpec.DAG.PrimOp.reduceSum_specFwd]

/-- Post-update KDA readout as a compositional TorchLean term.

The recurrent state is an ordinary term argument, so callers may supply either an input state or a
shared `let`-bound update. No K3 operation is hidden behind a primitive node.
-/
def readoutTerm {Γ : List Shape} (modelDim heads keyDim valueDim : Nat)
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hValue : 0 < valueDim)
    (current : Term Γ (.dim modelDim .scalar))
    (query : Term Γ (.dim heads (.dim keyDim .scalar)))
    (state : Term Γ (.dim heads (.dim keyDim (.dim valueDim .scalar))))
    (gateWeight : Term Γ (.dim modelDim (.dim heads (.dim valueDim .scalar))))
    (outputWeight : Term Γ (.dim heads (.dim valueDim (.dim modelDim .scalar))))
    (outputNormScale : Term Γ (.dim heads (.dim valueDim .scalar))) :
    Term Γ (.dim modelDim .scalar) :=
  let normalized := normalizedHeadReadTerm heads keyDim valueDim hHeads hValue
    query state outputNormScale
  let gate := outputGateTerm modelDim heads valueDim current gateWeight
  projectHeadReadoutTerm modelDim heads valueDim hModel hHeads gate normalized outputWeight

private theorem reduceSum_outer_matrix_eq_foldl {rows columns : Nat}
    (tensor : Tensor ℝ (.dim rows (.dim columns .scalar)))
    (h : Shape.NonemptyAxis 0 (.dim rows (.dim columns .scalar))) :
    Tensor.reduceSum 0 tensor h = Tensor.dim fun column => Tensor.scalar <|
      (List.finRange rows).foldl
        (fun total row => total + Tensor.getScalar (Spec.get tensor row) column) 0 := by
  cases tensor with
  | dim slices =>
      simp only [Tensor.reduceSum, Tensor.reduceDim, Tensor.Internal.reduceDimCore,
        Tensor.Internal.reduceOuterAxis]
      apply congrArg Tensor.dim
      funext column
      apply congrArg Tensor.scalar
      rw [sum_spec_vec, List.finRange_foldl_add_eq_finset_sum]
      rfl

/-- The compositional readout term denotes KDA's packed multi-head readout equation. -/
theorem eval_readoutTerm {Γ : List Shape}
    (env : TorchLean.TensorPack ℝ Γ) (modelDim heads keyDim valueDim : Nat)
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hValue : 0 < valueDim)
    (current : Term Γ (.dim modelDim .scalar))
    (query : Term Γ (.dim heads (.dim keyDim .scalar)))
    (state : Term Γ (.dim heads (.dim keyDim (.dim valueDim .scalar))))
    (gateWeight : Term Γ (.dim modelDim (.dim heads (.dim valueDim .scalar))))
    (outputWeight : Term Γ (.dim heads (.dim valueDim (.dim modelDim .scalar))))
    (outputNormScale : Term Γ (.dim heads (.dim valueDim .scalar))) :
    Term.eval env (readoutTerm modelDim heads keyDim valueDim hModel hHeads hValue
      current query state gateWeight outputWeight outputNormScale) =
      KDALayer.readout (Term.eval env state) (Term.eval env query)
        (Tensor.dim fun head =>
          mapSpec Activation.Math.sigmoidSpec <|
            vecMatMulSpec (Term.eval env current) <| Tensor.dim fun input =>
              Spec.get (Spec.get (Term.eval env gateWeight) input) head)
        (Term.eval env outputWeight) (Term.eval env outputNormScale) := by
  simp only [readoutTerm, eval_projectHeadReadoutTerm, eval_outputGateTerm,
    eval_normalizedHeadReadTerm, KDALayer.readout]
  rw [reduceSum_outer_matrix_eq_foldl]

/-- A shared recurrent update can feed the readout without changing any earlier graph value. -/
@[simp] theorem eval_readoutTerm_after_append {Γ : List Shape}
    (env : TorchLean.TensorPack ℝ Γ) (modelDim heads keyDim valueDim : Nat)
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hValue : 0 < valueDim)
    (current : Term Γ (.dim modelDim .scalar))
    (query : Term Γ (.dim heads (.dim keyDim .scalar)))
    (nextState : Tensor ℝ (.dim heads (.dim keyDim (.dim valueDim .scalar))))
    (gateWeight : Term Γ (.dim modelDim (.dim heads (.dim valueDim .scalar))))
    (outputWeight : Term Γ (.dim heads (.dim valueDim (.dim modelDim .scalar))))
    (outputNormScale : Term Γ (.dim heads (.dim valueDim .scalar))) :
    Term.eval (TorchLean.TensorPack.append env (.cons nextState .nil))
        (readoutTerm modelDim heads keyDim valueDim hModel hHeads hValue
          (Term.weakenRight current) (Term.weakenRight query) (Term.var (Var.last Γ))
          (Term.weakenRight gateWeight) (Term.weakenRight outputWeight)
          (Term.weakenRight outputNormScale)) =
      KDALayer.readout nextState (Term.eval env query)
        (Tensor.dim fun head =>
          mapSpec Activation.Math.sigmoidSpec <|
            vecMatMulSpec (Term.eval env current) <| Tensor.dim fun input =>
              Spec.get (Spec.get (Term.eval env gateWeight) input) head)
        (Term.eval env outputWeight) (Term.eval env outputNormScale) := by
  rw [eval_readoutTerm]
  rw [show Term.eval
      (TorchLean.TensorPack.append env (.cons nextState .nil))
      (Term.var (Var.last Γ)) = nextState by
    simpa only [Term.eval] using Env.tget_append_last env nextState]
  simp only [Term.eval_weakenRight]

/-- First half of the fixed-window KDA graph: project the current window and form the recurrent
inputs for every head.

The six outputs are exactly the quantities listed by `Prepared`.  Making this an ordinary
multi-output block gives the recurrent update a typed interface while keeping all projection,
convolution, normalization, and gate operations visible in the DAG. -/
def prepareBlock (modelDim heads keyDim valueDim convWidth decayRank : Nat)
    (hHeads : 0 < heads) (hKey : 0 < keyDim)
    (hValue : 0 < valueDim) (hWidth : 0 < convWidth) :
    Block
      (LayerParams modelDim heads keyDim valueDim convWidth decayRank ++
        LayerInputs modelDim heads keyDim valueDim convWidth)
      (Prepared modelDim heads keyDim valueDim) :=
  let params := LayerParams modelDim heads keyDim valueDim convWidth decayRank
  let inputs := LayerInputs modelDim heads keyDim valueDim convWidth
  let Γ := params ++ inputs
  let envTerms : Args Γ
      (LayerParams modelDim heads keyDim valueDim convWidth decayRank ++
        LayerInputs modelDim heads keyDim valueDim convWidth) := by
    simpa [Γ, params, inputs] using Args.vars Γ
  let .cons queryWeight
      (.cons keyWeight
      (.cons valueWeight
      (.cons queryConvWeight
      (.cons queryConvBias
      (.cons keyConvWeight
      (.cons keyConvBias
      (.cons valueConvWeight
      (.cons valueConvBias
      (.cons betaWeight
      (.cons decayDown
      (.cons decayUp
      (.cons decayBias
      (.cons decayLogScale
      (.cons gateWeight
      (.cons outputWeight
      (.cons outputNormScale
      (.cons window
      (.cons state
      (.cons logFloor
      (.cons epsilon .nil)))))))))))))))))))) := envTerms
  let .cons current
      (.cons query (.cons key (.cons value (.cons retention (.cons beta .nil))))) :=
    prepareTerms modelDim heads keyDim valueDim convWidth decayRank hHeads hKey hValue hWidth
      queryWeight keyWeight valueWeight queryConvWeight keyConvWeight queryConvBias keyConvBias
      valueConvWeight valueConvBias betaWeight decayDown decayUp decayBias decayLogScale window
      logFloor epsilon
  Block.ret <| .cons current <|
    .cons query (.cons key (.cons value (.cons retention (.cons beta .nil))))

/-- Pack the mathematical KDA preparation equations in the order expected by `prepareBlock`.

This is the semantic view of the graph boundary, not another KDA state type.  Its entries are the
current token and the five fields obtained by applying `KDAHead.prepareWindow` independently to
each head. -/
def preparedTensors {α : Type} [Context α]
    {modelDim heads keyDim valueDim convWidth decayRank : Nat}
    (layer : KDALayer α modelDim heads keyDim valueDim convWidth decayRank)
    (hWidth : 0 < convWidth) (logFloor : α)
    (window : Tensor α (.dim convWidth (.dim modelDim .scalar))) :
    TorchLean.TensorPack α (Prepared modelDim heads keyDim valueDim) :=
  let current := Spec.get window ⟨0, hWidth⟩
  let prepared := fun head => (layer.head head).prepareWindow hWidth logFloor window
  .cons current <|
    .cons (Tensor.dim fun head => (prepared head).query) <|
    .cons (Tensor.dim fun head => (prepared head).key) <|
    .cons (Tensor.dim fun head => (prepared head).value) <|
    .cons (Tensor.dim fun head => (prepared head).retention) <|
    .cons (Tensor.dim fun head => Tensor.scalar (prepared head).writeStrength) .nil

/-- The first graph block computes exactly the fixed-window KDA preparation equations. -/
theorem eval_prepareBlock_eq_preparedTensors
    {modelDim heads keyDim valueDim convWidth decayRank : Nat}
    (hHeads : 0 < heads) (hKey : 0 < keyDim) (hValue : 0 < valueDim)
    (hWidth : 0 < convWidth)
    (layer : KDALayer ℝ modelDim heads keyDim valueDim convWidth decayRank)
    (window : Tensor ℝ (.dim convWidth (.dim modelDim .scalar)))
    (state : KDALayer.State ℝ heads keyDim valueDim) (logFloor : ℝ) :
    Block.eval
        (TorchLean.TensorPack.append (layerParameters layer)
          (layerInputs window state logFloor Numbers.epsilon))
        (prepareBlock modelDim heads keyDim valueDim convWidth decayRank
          hHeads hKey hValue hWidth) =
      preparedTensors layer hWidth logFloor window := by
  simp [prepareBlock, preparedTensors, prepareTerms, layerParameters, layerInputs, Block.eval,
    TorchLean.TensorPack.append,
    Args.vars, Args.weakenLeft, Term.weakenLeft, Term.rename, Term.eval, Term.evalArgs,
    Env.tget,
    NN.GraphSpec.DAG.PrimOp.select, NN.GraphSpec.DAG.PrimOp.add,
    NN.GraphSpec.DAG.PrimOp.exp, NN.GraphSpec.DAG.PrimOp.sigmoid,
    KDAHead.prepareWindow, ShortConv.forwardWindow, Activation.sigmoidSpec]
  congr
  apply congrArg Tensor.dim
  funext head
  simp only [Spec.Tensor.addSpec, Spec.Tensor.get_map2Spec,
    Spec.get_dim, Spec.Tensor.mapSpec_dim, Spec.Tensor.toScalar_mapSpec]
  unfold Spec.Tensor.expSpec
  rw [Spec.map_spec_comp, Spec.map_spec_comp]
  change Spec.Tensor.mapSpec MathFunctions.exp (Spec.Tensor.mapSpec _ _) = _
  rw [Spec.map_spec_comp]
  rfl

/-- Packed recurrent update used by the second half of a KDA layer.

This term contains only the state transition from Eq. 7 of the K3 report.  Projection,
convolution, and output readout remain separate graph terms, so each equation has an independently
checkable refinement theorem. -/
def updateTerm {Γ : List Shape} (heads keyDim valueDim : Nat)
    (state : Term Γ (.dim heads (.dim keyDim (.dim valueDim .scalar))))
    (key : Term Γ (.dim heads (.dim keyDim .scalar)))
    (value : Term Γ (.dim heads (.dim valueDim .scalar)))
    (retention : Term Γ (.dim heads (.dim keyDim .scalar)))
    (beta : Term Γ (.dim heads .scalar)) :
    Term Γ (.dim heads (.dim keyDim (.dim valueDim .scalar))) :=
  let decayed := Term.op
    (NN.GraphSpec.DAG.PrimOp.batchedRowScale heads keyDim valueDim)
    (.cons retention (.cons state .nil))
  let correction := batchedVecMatTerm heads keyDim valueDim key decayed
  let correctionOuter := Term.op
    (NN.GraphSpec.DAG.PrimOp.batchedOuter heads keyDim valueDim)
    (.cons key (.cons correction .nil))
  let writeOuter := Term.op
    (NN.GraphSpec.DAG.PrimOp.batchedOuter heads keyDim valueDim)
    (.cons key (.cons value .nil))
  let scaledCorrection := Term.op
    (NN.GraphSpec.DAG.PrimOp.batchedScale heads (.dim keyDim (.dim valueDim .scalar)))
    (.cons beta (.cons correctionOuter .nil))
  let scaledWrite := Term.op
    (NN.GraphSpec.DAG.PrimOp.batchedScale heads (.dim keyDim (.dim valueDim .scalar)))
    (.cons beta (.cons writeOuter .nil))
  let corrected := Term.op
    (NN.GraphSpec.DAG.PrimOp.sub (.dim heads (.dim keyDim (.dim valueDim .scalar))))
    (.cons decayed (.cons scaledCorrection .nil))
  Term.op
    (NN.GraphSpec.DAG.PrimOp.add (.dim heads (.dim keyDim (.dim valueDim .scalar))))
    (.cons corrected (.cons scaledWrite .nil))

/-- `updateTerm` denotes one independent KDA recurrence update for every packed head. -/
theorem eval_updateTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (heads keyDim valueDim : Nat)
    (state : Term Γ (.dim heads (.dim keyDim (.dim valueDim .scalar))))
    (query : Term Γ (.dim heads (.dim keyDim .scalar)))
    (key : Term Γ (.dim heads (.dim keyDim .scalar)))
    (value : Term Γ (.dim heads (.dim valueDim .scalar)))
    (retention : Term Γ (.dim heads (.dim keyDim .scalar)))
    (beta : Term Γ (.dim heads .scalar)) :
    Term.eval env (updateTerm heads keyDim valueDim state key value retention beta) =
      Tensor.dim fun head =>
        KDA.update (Spec.get (Term.eval env state) head)
          { query := Spec.get (Term.eval env query) head
            key := Spec.get (Term.eval env key) head
            value := Spec.get (Term.eval env value) head
            retention := Spec.get (Term.eval env retention) head
            writeStrength := (Spec.get (Term.eval env beta) head).item } := by
  simp only [updateTerm, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.batchedRowScale_specFwd,
    eval_batchedVecMatTerm,
    NN.GraphSpec.DAG.PrimOp.batchedOuter_specFwd,
    NN.GraphSpec.DAG.PrimOp.batchedScale_specFwd,
    NN.GraphSpec.DAG.PrimOp.sub_specFwd, NN.GraphSpec.DAG.PrimOp.add_specFwd]
  congr

/-- Assemble the recurrent update and readout from terms in an arbitrary graph environment. -/
def updateReadoutTermsBlock {Γ : List Shape}
    (modelDim heads keyDim valueDim : Nat)
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hValue : 0 < valueDim)
    (state : Term Γ (.dim heads (.dim keyDim (.dim valueDim .scalar))))
    (current : Term Γ (.dim modelDim .scalar))
    (query key : Term Γ (.dim heads (.dim keyDim .scalar)))
    (value : Term Γ (.dim heads (.dim valueDim .scalar)))
    (retention : Term Γ (.dim heads (.dim keyDim .scalar)))
    (beta : Term Γ (.dim heads .scalar))
    (gateWeight : Term Γ (.dim modelDim (.dim heads (.dim valueDim .scalar))))
    (outputWeight : Term Γ (.dim heads (.dim valueDim (.dim modelDim .scalar))))
    (outputNormScale : Term Γ (.dim heads (.dim valueDim .scalar))) :
    Block Γ (LayerOutputs modelDim heads keyDim valueDim) :=
  let nextState := updateTerm heads keyDim valueDim state key value retention beta
  let Γnext := Γ ++ [.dim heads (.dim keyDim (.dim valueDim .scalar))]
  let nextState' : Term Γnext (.dim heads (.dim keyDim (.dim valueDim .scalar))) :=
    Term.var (Var.last Γ)
  let output := readoutTerm modelDim heads keyDim valueDim hModel hHeads hValue
    (Term.weakenRight current) (Term.weakenRight query) nextState'
    (Term.weakenRight gateWeight) (Term.weakenRight outputWeight)
    (Term.weakenRight outputNormScale)
  Block.let1 nextState <| Block.ret <| .cons nextState' (.cons output .nil)

/-- The generic update/readout block denotes `KDALayer.stepPrepared`. -/
theorem eval_updateReadoutTermsBlock {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    {modelDim heads keyDim valueDim : Nat}
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hValue : 0 < valueDim)
    (state : Term Γ (.dim heads (.dim keyDim (.dim valueDim .scalar))))
    (current : Term Γ (.dim modelDim .scalar))
    (query key : Term Γ (.dim heads (.dim keyDim .scalar)))
    (value : Term Γ (.dim heads (.dim valueDim .scalar)))
    (retention : Term Γ (.dim heads (.dim keyDim .scalar)))
    (beta : Term Γ (.dim heads .scalar))
    (gateWeight : Term Γ (.dim modelDim (.dim heads (.dim valueDim .scalar))))
    (outputWeight : Term Γ (.dim heads (.dim valueDim (.dim modelDim .scalar))))
    (outputNormScale : Term Γ (.dim heads (.dim valueDim .scalar))) :
    Block.eval env
        (updateReadoutTermsBlock modelDim heads keyDim valueDim hModel hHeads hValue
          state current query key value retention beta gateWeight outputWeight outputNormScale) =
      let prepared : Fin heads → KDAStepInput ℝ keyDim valueDim := fun head =>
        { query := Spec.get (Term.eval env query) head
          key := Spec.get (Term.eval env key) head
          value := Spec.get (Term.eval env value) head
          retention := Spec.get (Term.eval env retention) head
          writeStrength := (Spec.get (Term.eval env beta) head).item }
      let next := Tensor.dim fun head => KDA.update (Spec.get (Term.eval env state) head)
        (prepared head)
      .cons next <| .cons
        (KDALayer.readout next (Term.eval env query)
          (Tensor.dim fun head =>
            mapSpec Activation.Math.sigmoidSpec <|
              vecMatMulSpec (Term.eval env current) <| Tensor.dim fun input =>
                Spec.get (Spec.get (Term.eval env gateWeight) input) head)
          (Term.eval env outputWeight) (Term.eval env outputNormScale)) .nil := by
  simp only [updateReadoutTermsBlock, Block.eval]
  rw [eval_updateTerm env heads keyDim valueDim state query key value retention beta]
  simp only [Term.evalArgs, Term.eval_var_last_append]
  rw [eval_readoutTerm_after_append]

/-- Second half of the fixed-window KDA graph.

The original parameter/input environment is retained on the left and the six prepared tensors are
appended on the right.  The new recurrent state is bound once because it is both returned and read
by the output projection. -/
def updateReadoutBlock (modelDim heads keyDim valueDim convWidth decayRank : Nat)
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hValue : 0 < valueDim) :
    Block
      ((LayerParams modelDim heads keyDim valueDim convWidth decayRank ++
          LayerInputs modelDim heads keyDim valueDim convWidth) ++
        Prepared modelDim heads keyDim valueDim)
      (LayerOutputs modelDim heads keyDim valueDim) :=
  letI : Fact (0 < modelDim) := ⟨hModel⟩
  let prepParams := PreparationParams modelDim heads keyDim valueDim convWidth decayRank
  let readoutParams := ReadoutParams modelDim heads valueDim
  let params := prepParams ++ readoutParams
  let inputs := LayerInputs modelDim heads keyDim valueDim convWidth
  let prepared := Prepared modelDim heads keyDim valueDim
  let base := params ++ inputs
  let Γ := base ++ prepared
  let gateWeight : Term Γ (.dim modelDim (.dim heads (.dim valueDim .scalar))) :=
    .var <| Var.inLeft prepared <| Var.inLeft inputs <| Var.inRight prepParams .head
  let outputWeight : Term Γ (.dim heads (.dim valueDim (.dim modelDim .scalar))) :=
    .var <| Var.inLeft prepared <| Var.inLeft inputs <|
      Var.inRight prepParams (.tail .head)
  let outputNormScale : Term Γ (.dim heads (.dim valueDim .scalar)) :=
    .var <| Var.inLeft prepared <| Var.inLeft inputs <|
      Var.inRight prepParams (.tail (.tail .head))
  let state : Term Γ (.dim heads (.dim keyDim (.dim valueDim .scalar))) :=
    .var <| Var.inLeft prepared <| Var.inRight params (.tail .head)
  let current : Term Γ (.dim modelDim .scalar) :=
    .var <| Var.inRight base .head
  let query : Term Γ (.dim heads (.dim keyDim .scalar)) :=
    .var <| Var.inRight base (.tail .head)
  let key : Term Γ (.dim heads (.dim keyDim .scalar)) :=
    .var <| Var.inRight base (.tail (.tail .head))
  let value : Term Γ (.dim heads (.dim valueDim .scalar)) :=
    .var <| Var.inRight base (.tail (.tail (.tail .head)))
  let retention : Term Γ (.dim heads (.dim keyDim .scalar)) :=
    .var <| Var.inRight base (.tail (.tail (.tail (.tail .head))))
  let beta : Term Γ (.dim heads .scalar) :=
    .var <| Var.inRight base (.tail (.tail (.tail (.tail (.tail .head)))))
  updateReadoutTermsBlock modelDim heads keyDim valueDim hModel hHeads hValue
    state current query key value retention beta gateWeight outputWeight outputNormScale

/-- The second graph block is the recurrent KDA update followed by the shared layer readout.

The six prepared tensors form a typed boundary between projection/convolution and recurrence.  The
statement reconstructs `KDAStepInput` only to identify those packed tensors with the mathematical
arguments of `KDALayer.stepPrepared`; it does not introduce another executable KDA path. -/
theorem eval_updateReadoutBlock_eq_stepPrepared
    {modelDim heads keyDim valueDim convWidth decayRank : Nat}
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hValue : 0 < valueDim)
    (layer : KDALayer ℝ modelDim heads keyDim valueDim convWidth decayRank)
    (window : Tensor ℝ (.dim convWidth (.dim modelDim .scalar)))
    (state : KDALayer.State ℝ heads keyDim valueDim) (logFloor epsilon : ℝ)
    (current : Tensor ℝ (.dim modelDim .scalar))
    (query key : Tensor ℝ (.dim heads (.dim keyDim .scalar)))
    (value : Tensor ℝ (.dim heads (.dim valueDim .scalar)))
    (retention : Tensor ℝ (.dim heads (.dim keyDim .scalar)))
    (beta : Tensor ℝ (.dim heads .scalar)) :
    Block.eval
        (TorchLean.TensorPack.append
          (TorchLean.TensorPack.append (layerParameters layer)
            (layerInputs window state logFloor epsilon))
          (.cons current <| .cons query <| .cons key <| .cons value <|
            .cons retention <| .cons beta .nil))
        (updateReadoutBlock modelDim heads keyDim valueDim convWidth decayRank
          hModel hHeads hValue) =
      let prepared : Fin heads → KDAStepInput ℝ keyDim valueDim := fun head =>
        { query := Spec.get query head
          key := Spec.get key head
          value := Spec.get value head
          retention := Spec.get retention head
          writeStrength := (Spec.get beta head).item }
      let result := layer.stepPrepared state current prepared
      .cons result.1 (.cons result.2 .nil) := by
  simp only [updateReadoutBlock]
  rw [eval_updateReadoutTermsBlock]
  simp [Term.eval, Var.inLeft, Var.inRight, Env.tget,
    TorchLean.TensorPack.append,
    layerParameters, layerInputs,
    KDALayer.stepPrepared, KDALayer.gates]
  cases query
  rfl

/-- Complete fixed-window multi-head KDA token step as a typed TorchLean graph. -/
def layerModel (modelDim heads keyDim valueDim convWidth decayRank : Nat)
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hKey : 0 < keyDim)
    (hValue : 0 < valueDim) (hWidth : 0 < convWidth) :
    NN.GraphSpec.DAG.MultiModel
      (LayerParams modelDim heads keyDim valueDim convWidth decayRank)
      (LayerInputs modelDim heads keyDim valueDim convWidth)
      (LayerOutputs modelDim heads keyDim valueDim) :=
  { initParams := initialLayerParams modelDim heads keyDim valueDim convWidth decayRank
    body := (prepareBlock modelDim heads keyDim valueDim convWidth decayRank
      hHeads hKey hValue hWidth).andThen <|
        updateReadoutBlock modelDim heads keyDim valueDim convWidth decayRank
          hModel hHeads hValue }

/-- The complete packed KDA graph denotes the fixed-window mathematical layer step. -/
theorem layerModel_specFwd_eq_stepWindow
    {modelDim heads keyDim valueDim convWidth decayRank : Nat}
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hKey : 0 < keyDim)
    (hValue : 0 < valueDim) (hWidth : 0 < convWidth)
    (layer : KDALayer ℝ modelDim heads keyDim valueDim convWidth decayRank)
    (window : Tensor ℝ (.dim convWidth (.dim modelDim .scalar)))
    (state : KDALayer.State ℝ heads keyDim valueDim) (logFloor : ℝ) :
    (layerModel modelDim heads keyDim valueDim convWidth decayRank
      hModel hHeads hKey hValue hWidth).specFwd (layerParameters layer)
        (layerInputs window state logFloor Numbers.epsilon) =
      let result := layer.stepWindow hWidth logFloor window state
      .cons result.1 (.cons result.2 .nil) := by
  simp only [NN.GraphSpec.DAG.MultiModel.specFwd, layerModel]
  rw [Block.eval_andThen]
  rw [eval_prepareBlock_eq_preparedTensors hHeads hKey hValue hWidth layer window state logFloor]
  simp only [preparedTensors]
  rw [eval_updateReadoutBlock_eq_stepPrepared]
  rfl

end KDA
end GraphSpec
end KimiK3
