/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Primitives
public import KimiK3.Sequence.KDA
public import TorchLean.Tensor.Reduce.Real

/-!
# Kimi Delta Attention graphs

`model` starts from prepared query, key, value, retention, and write-strength tensors. It has no
learned parameters and returns the updated recurrent state and its query readout. The later
`prepareBlock` and `layerModel` add the learned projections, the short-convolution window,
per-head normalization, gating, and the output projection for a complete token step.

The graphs use ordinary TorchLean primitives instead of an opaque call to `KDA.update`, and their
evaluation theorems identify the returned tensors with the reference equations over the reals.
Running them on a backend still needs implementations of those primitives.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec
namespace KDA

open TorchLean
open FloatLib.Numerics (MathFunctions)
open TorchLean.Tensor

attribute [local simp] TorchLean.Tensor.map_dim

private theorem scale_rows {α : Type} [Storage α] [Mul α] {rows columns : Nat}
    (weights : Tensor α [rows]) (matrix : Tensor α [rows, columns]) :
    Tensor.zipWith (s := [rows, columns]) (· * ·)
      (Tensor.mapLeading [rows] (fun x => Tensor.full [columns] x.item) weights) matrix =
      Tensor.dim (fun i => Tensor.dim (fun j =>
        Tensor.scalar (weights.getScalar i * Tensor.get2 matrix i j))) := by
  apply TorchLean.Tensor.Rep.ext
  rintro ⟨i, j, ⟨⟩⟩
  simp [Tensor.mapLeading, Tensor.dim_apply, Tensor.unstack_apply, Tensor.getScalar_eq_apply,
    Tensor.get2_eq_apply, Tensor.item]

private theorem map_mul_left {α : Type} [Storage α] [Mul α] {s : Shape}
    (c : α) (x : Tensor α s) :
    Tensor.map (c * ·) x = Tensor.zipWith (· * ·) (Tensor.full s c) x := by
  apply Tensor.Rep.ext
  intro i
  simp

/-- Inputs to one projected KDA step, beginning with the recurrent state. -/
abbrev Inputs (keyDim valueDim : Nat) : List Shape :=
  [ [keyDim, valueDim],
    [keyDim],
    [keyDim],
    [valueDim],
    [keyDim],
    [] ]

/-- Outputs of one step: the updated recurrent state and the current token's value readout. -/
abbrev Outputs (keyDim valueDim : Nat) : List Shape :=
  [[keyDim, valueDim], [valueDim]]

/-- One projected KDA step as a multi-output TorchLean graph that binds the new state once. -/
def model (keyDim valueDim : Nat) :
    MultiModel [] (Inputs keyDim valueDim) (Outputs keyDim valueDim) :=
  let stateShape : Shape := [keyDim, valueDim]
  let keyShape : Shape := [keyDim]
  let valueShape : Shape := [valueDim]
  let Γ := [] ++ Inputs keyDim valueDim
  let envTerms : Args Γ (Inputs keyDim valueDim) := by
    simpa [Γ] using Args.vars Γ
  let .cons state <| .cons query <| .cons key <| .cons value <|
      .cons retention <| .cons writeStrength .nil := envTerms
  let decayed : Term Γ stateShape :=
    Term.op (PrimOp.mul stateShape)
      (.cons (expandTerm [keyDim] [valueDim] retention) (.cons state .nil))
  let correction : Term Γ valueShape :=
    Term.op (PrimOp.broadcastVecMat [] [] [] keyDim valueDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
      (.cons key (.cons decayed .nil))
  let correctionOuter : Term Γ stateShape :=
    outerTerm [] keyDim valueDim key correction
  let writeOuter : Term Γ stateShape :=
    outerTerm [] keyDim valueDim key value
  let scaledCorrection : Term Γ stateShape :=
    scaleTerm stateShape writeStrength correctionOuter
  let scaledWrite : Term Γ stateShape :=
    scaleTerm stateShape writeStrength writeOuter
  let corrected : Term Γ stateShape :=
    Term.op (PrimOp.sub stateShape) (.cons decayed (.cons scaledCorrection .nil))
  let nextState : Term Γ stateShape :=
    Term.op (PrimOp.add stateShape)
      (.cons corrected (.cons scaledWrite .nil))
  let nextState' : Term (Γ ++ [stateShape]) stateShape := Term.var (Var.last Γ)
  let query' : Term (Γ ++ [stateShape]) keyShape := Term.weakenRight query
  let output : Term (Γ ++ [stateShape]) valueShape :=
    Term.op (PrimOp.broadcastVecMat [] [] [] keyDim valueDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
      (.cons query' (.cons nextState' .nil))
  { initParams := .nil
    body := Block.let1 nextState <| Block.ret <|
      .cons nextState' (.cons output .nil) }

/-- Package the mathematical KDA state and projected input in the graph ABI. -/
def inputs {α : Type} [Storage α] {keyDim valueDim : Nat}
    (state : KimiK3.KDA.State α keyDim valueDim)
    (input : KDA.Input α keyDim valueDim) : TorchLean.TensorPack α (Inputs keyDim valueDim) :=
  .cons state <| .cons input.query <| .cons input.key <| .cons input.value <|
    .cons input.retention <| .cons (Tensor.scalar input.writeStrength) .nil

/-- `model` computes `KDA.step`: the recurrence and the readout of the new state. -/
theorem specFwd_eq_step {keyDim valueDim : Nat}
    (state : KimiK3.KDA.State ℝ keyDim valueDim)
    (input : KDA.Input ℝ keyDim valueDim) :
    (model keyDim valueDim).specFwd .nil (inputs state input) =
      let result := KimiK3.KDA.step state input
      .cons result.1 (.cons result.2 .nil) := by
  change Block.eval (inputs state input) (model keyDim valueDim).body = _
  simp [model, inputs, Inputs, Args.vars, Args.weakenLeft, Args.rename,
    Term.evalArgs, Env.tget, KimiK3.KDA.step, KimiK3.KDA.update, KimiK3.KDA.read,
    Tensor.zipEach, PrimOp.Internal.vecMatCommonBatchSpec, Tensor.add_def,
    Tensor.sub_def, Tensor.mul_def, scale_rows, map_mul_left]

/-! ## Complete multi-head layer -/

/-- Projection, short-convolution, write-gate, and decay parameters for a packed KDA layer.

Every head-local tensor has the head axis first. This is a packed view of `KDA.Layer.head`, not a
second parameter record, and `layerParameters` below builds it for `bmm` and elementwise
operations.
-/
abbrev PreparationParams (modelDim heads keyDim valueDim convWidth decayRank : Nat) : List Shape :=
  [ [heads, modelDim, keyDim],
    [heads, modelDim, keyDim],
    [heads, modelDim, valueDim],
    [heads, convWidth, keyDim],
    [heads, keyDim],
    [heads, convWidth, keyDim],
    [heads, keyDim],
    [heads, convWidth, valueDim],
    [heads, valueDim],
    [heads, modelDim],
    [heads, modelDim, decayRank],
    [heads, decayRank, keyDim],
    [heads, keyDim],
    [heads] ]

/-- Gate, projection, and normalization parameters used after the recurrent update. -/
abbrev ReadoutParams (modelDim heads valueDim : Nat) : List Shape :=
  [ [modelDim, heads, valueDim],
    [heads, valueDim, modelDim],
    [heads, valueDim] ]

/-- Parameter shapes for a complete packed KDA layer. -/
abbrev LayerParams (modelDim heads keyDim valueDim convWidth decayRank : Nat) : List Shape :=
  PreparationParams modelDim heads keyDim valueDim convWidth decayRank ++
    ReadoutParams modelDim heads valueDim

/-- Runtime inputs for one fixed-window KDA token step. -/
abbrev LayerInputs (modelDim heads keyDim valueDim convWidth : Nat) : List Shape :=
  [ [convWidth, modelDim],
    [heads, keyDim, valueDim],
    [],
    [] ]

/-- Updated recurrent matrices followed by the projected model-width output. -/
abbrev LayerOutputs (modelDim heads keyDim valueDim : Nat) : List Shape :=
  [[heads, keyDim, valueDim], [modelDim]]

/-- Zero-filled defaults for standalone graph construction.

Training and checkpoint loading replace every entry, and the semantic theorem holds for an
arbitrary `KDA.Layer`.
-/
def initialLayerParams (modelDim heads keyDim valueDim convWidth decayRank : Nat) :
    TorchLean.TensorPack Float (LayerParams modelDim heads keyDim valueDim convWidth decayRank) :=
  TorchLean.TensorPack.zero

/-- Pack the existing head-indexed KDA records into GraphSpec's parameter ABI. -/
def layerParameters {α : Type} [Storage α]
    {modelDim heads keyDim valueDim convWidth decayRank : Nat}
    (layer : KDA.Layer α modelDim heads keyDim valueDim convWidth decayRank) :
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
def layerInputs {α : Type} [Storage α] {modelDim heads keyDim valueDim convWidth : Nat}
    (window : Tensor α [convWidth, modelDim])
    (state : KDA.Layer.State α heads keyDim valueDim) (logFloor epsilon : α) :
    TorchLean.TensorPack α (LayerInputs modelDim heads keyDim valueDim convWidth) :=
  .cons window <| .cons state <| .cons (Tensor.scalar logFloor) <|
    .cons (Tensor.scalar epsilon) .nil

/-- Uncast graph term underlying `rollWindowTerm`. -/
def rollWindowRawTerm {Γ : List Shape} (modelDim convWidth : Nat)
    (current : Term Γ [modelDim])
    (previous : Term Γ [convWidth, modelDim]) :
    Term Γ [1 + (convWidth - 1), modelDim] :=
  let currentRow : Term Γ [1, modelDim] :=
    Term.op (PrimOp.reshape _ _ (by simp [Shape.size])) (.cons current .nil)
  let retained : Term Γ [convWidth - 1, modelDim] :=
    Term.op (PrimOp.unary fun x => Tensor.take x 0 (convWidth - 1)
      (by simp [Shape.axisSize])) (.cons previous .nil)
  concatLeadingTerm 1 (convWidth - 1) [modelDim] currentRow retained

/-- Graph term for the fixed KDA history update.

The current token becomes row `0` and the oldest row is dropped. Only reshape, leading-axis slice,
and concatenation are ordinary TorchLean tensor operations.
-/
def rollWindowTerm {Γ : List Shape} (modelDim convWidth : Nat) (hWidth : 0 < convWidth)
    (current : Term Γ [modelDim])
    (previous : Term Γ [convWidth, modelDim]) :
    Term Γ [convWidth, modelDim] :=
  Term.cast (rollWindowRawTerm modelDim convWidth current previous)
    (KDA.Layer.rollWindow_shape_eq hWidth)

/-- Evaluating `rollWindowTerm` gives the streaming window update `KDA.Layer.rollWindow`. -/
@[simp] theorem eval_rollWindowTerm {Γ : List Shape}
    (env : TorchLean.TensorPack ℝ Γ) (modelDim convWidth : Nat) (hWidth : 0 < convWidth)
    (current : Term Γ [modelDim])
    (previous : Term Γ [convWidth, modelDim]) :
    Term.eval env (rollWindowTerm modelDim convWidth hWidth current previous) =
      KDA.Layer.rollWindow hWidth (Term.eval env current) (Term.eval env previous) := by
  simp only [rollWindowTerm, Term.eval_cast, KDA.Layer.rollWindow, Tensor.castShape]
  rfl

/-- Project a fixed causal window and apply one packed depthwise short convolution.

The query, key, and value paths all use this term. -/
def shortConvTerm {Γ : List Shape} (heads convWidth modelDim channels : Nat)
    (_hHeads : 0 < heads) (_hWidth : 0 < convWidth) (_hChannels : 0 < channels)
    (window : Term Γ [convWidth, modelDim])
    (projection : Term Γ [heads, modelDim, channels])
    (kernel : Term Γ [heads, convWidth, channels])
    (bias : Term Γ [heads, channels]) :
    Term Γ [heads, channels] :=
  let projected := Term.op (PrimOp.binary fun x w =>
    Tensor.mapLeading [heads] (Tensor.matmul x) w) (.cons window (.cons projection .nil))
  let weighted := Term.op (PrimOp.mul [heads, convWidth, channels])
    (.cons projected (.cons kernel .nil))
  let summed : Term Γ [heads, channels] := Term.op (PrimOp.unary fun x => Tensor.mapLeading [heads]
    (fun (row : Tensor _ [convWidth, channels]) =>
      Tensor.reduce .sum row [⟨0, Nat.zero_lt_succ 1⟩] false) x)
    (.cons weighted .nil)
  Term.op (PrimOp.add [heads, channels]) (.cons summed (.cons bias .nil))

/-- Pure semantics of the packed projection and depthwise short-convolution term. -/
@[simp] theorem eval_shortConvTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (heads convWidth modelDim channels : Nat)
    (hHeads : 0 < heads) (hWidth : 0 < convWidth) (hChannels : 0 < channels)
    (window : Term Γ [convWidth, modelDim])
    (projection : Term Γ [heads, modelDim, channels])
    (kernel : Term Γ [heads, convWidth, channels])
    (bias : Term Γ [heads, channels]) :
    Term.eval env (shortConvTerm heads convWidth modelDim channels hHeads hWidth hChannels
      window projection kernel bias) =
      Tensor.zipWith (· + ·) (Tensor.dim fun head =>
        Tensor.reduce .sum
          (Tensor.matmul (Term.eval env window) ((Term.eval env projection).unstack head) *
            (Term.eval env kernel).unstack head)
          [⟨0, Nat.zero_lt_succ 1⟩] false) (Term.eval env bias) := by
  change Tensor.zipWith (· + ·)
    (Tensor.dim fun head => Tensor.reduce .sum
      (Tensor.unstack (Tensor.zipWith (· * ·)
        (Tensor.dim fun i => Tensor.matmul (Term.eval env window)
          ((Term.eval env projection).unstack i)) (Term.eval env kernel)) head)
      [⟨0, Nat.zero_lt_succ 1⟩] false) (Term.eval env bias) = _
  simp only [Tensor.unstack_zipWith, Tensor.unstack_dim, Tensor.mul_def]

/-- Short-convolution projection followed by SiLU, the whole value preparation path. -/
def activatedProjectionTerm {Γ : List Shape}
    (heads convWidth modelDim channels : Nat) (hHeads : 0 < heads)
    (hWidth : 0 < convWidth) (hChannels : 0 < channels)
    (window : Term Γ [convWidth, modelDim])
    (projection : Term Γ [heads, modelDim, channels])
    (kernel : Term Γ [heads, convWidth, channels])
    (bias : Term Γ [heads, channels]) :
    Term Γ [heads, channels] :=
  let projected := shortConvTerm heads convWidth modelDim channels hHeads hWidth hChannels
    window projection kernel bias
  Term.op (PrimOp.silu [heads, channels])
    (.cons projected .nil)

/-- Evaluating the activated projection applies SiLU to the short convolution. -/
@[simp] theorem eval_activatedProjectionTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (heads convWidth modelDim channels : Nat)
    (hHeads : 0 < heads) (hWidth : 0 < convWidth) (hChannels : 0 < channels)
    (window : Term Γ [convWidth, modelDim])
    (projection : Term Γ [heads, modelDim, channels])
    (kernel : Term Γ [heads, convWidth, channels])
    (bias : Term Γ [heads, channels]) :
    Term.eval env (activatedProjectionTerm heads convWidth modelDim channels
      hHeads hWidth hChannels window projection kernel bias) =
      TorchLean.Tensor.map Activation.silu
        (Term.eval env (shortConvTerm heads convWidth modelDim channels
          hHeads hWidth hChannels window projection kernel bias)) := by
  simp only [activatedProjectionTerm, Term.eval_op, Term.evalArgs,
    PrimOp.silu_specFwd]

/-- Query/key preparation: short convolution, SiLU, and row-wise L2 normalization. -/
def normalizedProjectionTerm {Γ : List Shape}
    (heads convWidth modelDim channels : Nat) (hHeads : 0 < heads)
    (hWidth : 0 < convWidth) (hChannels : 0 < channels)
    (window : Term Γ [convWidth, modelDim])
    (projection : Term Γ [heads, modelDim, channels])
    (kernel : Term Γ [heads, convWidth, channels])
    (bias : Term Γ [heads, channels]) (epsilon : Term Γ []) :
    Term Γ [heads, channels] :=
  let activated := activatedProjectionTerm heads convWidth modelDim channels hHeads hWidth
    hChannels window projection kernel bias
  Term.op (PrimOp.normalizeL2Regularized [heads] channels hChannels)
    (.cons activated (.cons epsilon .nil))

/-- Evaluation of the normalized projection normalizes each head along its channel axis. -/
@[simp] theorem eval_normalizedProjectionTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (heads convWidth modelDim channels : Nat)
    (hHeads : 0 < heads) (hWidth : 0 < convWidth) (hChannels : 0 < channels)
    (window : Term Γ [convWidth, modelDim])
    (projection : Term Γ [heads, modelDim, channels])
    (kernel : Term Γ [heads, convWidth, channels])
    (bias : Term Γ [heads, channels]) (epsilon : Term Γ []) :
    Term.eval env (normalizedProjectionTerm heads convWidth modelDim channels
      hHeads hWidth hChannels window projection kernel bias epsilon) =
      Tensor.dim fun head => Tensor.normalizeL2
        (Tensor.unstack
          (Term.eval env (activatedProjectionTerm heads convWidth modelDim channels
            hHeads hWidth hChannels window projection kernel bias)) head)
        (Tensor.item (Term.eval env epsilon)) := by
  rfl

/-- Multiply one shared vector by a matrix for each head. -/
def batchedSharedVecMatTerm {Γ : List Shape} (heads inputDim outputDim : Nat)
    (vector : Term Γ [inputDim])
    (matrices : Term Γ [heads, inputDim, outputDim]) :
    Term Γ [heads, outputDim] :=
  Term.op (PrimOp.binary fun x w => Tensor.mapLeading [heads] (Tensor.vecmat x) w)
    (.cons vector (.cons matrices .nil))

/-- Multiply one vector by one matrix for each head. -/
def batchedVecMatTerm {Γ : List Shape} (heads inputDim outputDim : Nat)
    (vectors : Term Γ [heads, inputDim])
    (matrices : Term Γ [heads, inputDim, outputDim]) :
    Term Γ [heads, outputDim] :=
  Term.op (PrimOp.binary fun x w => Tensor.zipEach [heads] [outputDim] Tensor.vecmat x w)
    (.cons vectors (.cons matrices .nil))

/-- Pure semantics of a shared vector multiplied by one matrix per head. -/
@[simp] theorem eval_batchedSharedVecMatTerm {Γ : List Shape}
    (env : TorchLean.TensorPack ℝ Γ) (heads inputDim outputDim : Nat)
    (vector : Term Γ [inputDim])
    (matrices : Term Γ [heads, inputDim, outputDim]) :
    Term.eval env (batchedSharedVecMatTerm heads inputDim outputDim vector matrices) =
      .dim (fun head => Tensor.vecmat (Term.eval env vector)
        (Tensor.unstack (Term.eval env matrices) head)) := by
  rfl

/-- Pure semantics of independently multiplying one vector and matrix per head. -/
@[simp] theorem eval_batchedVecMatTerm {Γ : List Shape}
    (env : TorchLean.TensorPack ℝ Γ) (heads inputDim outputDim : Nat)
    (vectors : Term Γ [heads, inputDim])
    (matrices : Term Γ [heads, inputDim, outputDim]) :
    Term.eval env (batchedVecMatTerm heads inputDim outputDim vectors matrices) =
      .dim (fun head => Tensor.vecmat
        (Tensor.unstack (Term.eval env vectors) head)
        (Tensor.unstack (Term.eval env matrices) head)) := by
  rfl

/-- Shapes of the current token and the five quantities prepared before the recurrent update.

In order: current token, normalized query, normalized key, activated value, retention, and write
strength. This `Args` signature marks the boundary between the projection and gating calculation
and the recurrent matrix update. -/
abbrev Prepared (modelDim heads keyDim valueDim : Nat) : List Shape :=
  [ [modelDim],
    [heads, keyDim],
    [heads, keyDim],
    [heads, valueDim],
    [heads, keyDim],
    [heads] ]

/-- Prepare the current token and the query, key, value, retention, and write-strength terms for
one KDA token, in the dependency order of the KDA equations.

Query and key take the value path (short convolution and SiLU) followed by row-wise L2
normalization. Retention is `exp(logFloor * sigmoid(exp(logScale) * (decayLogit + decayBias)))`,
and write strength is the sigmoid of the per-head linear projection the paper calls `beta`. -/
def prepareTerms {Γ : List Shape}
    (modelDim heads keyDim valueDim convWidth decayRank : Nat)
    (hHeads : 0 < heads) (hKey : 0 < keyDim) (hValue : 0 < valueDim)
    (hWidth : 0 < convWidth)
    (queryWeight keyWeight : Term Γ [heads, modelDim, keyDim])
    (valueWeight : Term Γ [heads, modelDim, valueDim])
    (queryConvWeight keyConvWeight :
      Term Γ [heads, convWidth, keyDim])
    (queryConvBias keyConvBias : Term Γ [heads, keyDim])
    (valueConvWeight : Term Γ [heads, convWidth, valueDim])
    (valueConvBias : Term Γ [heads, valueDim])
    (betaWeight : Term Γ [heads, modelDim])
    (decayDown : Term Γ [heads, modelDim, decayRank])
    (decayUp : Term Γ [heads, decayRank, keyDim])
    (decayBias : Term Γ [heads, keyDim])
    (decayLogScale : Term Γ [heads])
    (window : Term Γ [convWidth, modelDim])
    (logFloor epsilon : Term Γ []) :
    Args Γ (Prepared modelDim heads keyDim valueDim) :=
  let current : Term Γ [modelDim] :=
    Term.op (PrimOp.unary fun x => x.unstack ⟨0, hWidth⟩) (.cons window .nil)
  let query := normalizedProjectionTerm heads convWidth modelDim keyDim hHeads hWidth hKey
    window queryWeight queryConvWeight queryConvBias epsilon
  let key := normalizedProjectionTerm heads convWidth modelDim keyDim hHeads hWidth hKey
    window keyWeight keyConvWeight keyConvBias epsilon
  let value := activatedProjectionTerm heads convWidth modelDim valueDim hHeads hWidth hValue
    window valueWeight valueConvWeight valueConvBias
  let decayHidden : Term Γ [heads, decayRank] :=
    batchedSharedVecMatTerm heads modelDim decayRank current decayDown
  let decayLogit : Term Γ [heads, keyDim] :=
    batchedVecMatTerm heads decayRank keyDim decayHidden decayUp
  let shiftedDecayLogit :=
    Term.op (PrimOp.add [heads, keyDim])
      (.cons decayLogit (.cons decayBias .nil))
  let decayScale :=
    Term.op (PrimOp.exp [heads]) (.cons decayLogScale .nil)
  let scaledDecayLogit :=
    Term.op (PrimOp.mul [heads, keyDim])
      (.cons (expandTerm [heads] [keyDim] decayScale) (.cons shiftedDecayLogit .nil))
  let decayGate :=
    Term.op (PrimOp.sigmoid [heads, keyDim])
      (.cons scaledDecayLogit .nil)
  let retentionExponent :=
    scaleTerm [heads, keyDim] logFloor decayGate
  let retention :=
    Term.op (PrimOp.exp [heads, keyDim])
      (.cons retentionExponent .nil)
  let betaLogit : Term Γ [heads] := Term.op (PrimOp.binary fun x w =>
    Tensor.mapLeading [heads] (fun row => Tensor.scalar (Tensor.dot x row)) w)
    (.cons current (.cons betaWeight .nil))
  let beta :=
    Term.op (PrimOp.sigmoid [heads]) (.cons betaLogit .nil)
  .cons current <|
    .cons query (.cons key (.cons value (.cons retention (.cons beta .nil))))

/-- Read the updated recurrent matrix with each query and RMS-normalize each head separately.

The RMSNorm primitive runs with unit scale, and a pointwise multiplication then applies each head's
learned scale. `_hHeads` is kept for the graph-building interface, because normalization
within a head needs just `0 < valueDim`. -/
def normalizedHeadReadTerm {Γ : List Shape} (heads keyDim valueDim : Nat)
    (_hHeads : 0 < heads) (hValue : 0 < valueDim)
    (query : Term Γ [heads, keyDim])
    (state : Term Γ [heads, keyDim, valueDim])
    (outputNormScale : Term Γ [heads, valueDim]) :
    Term Γ [heads, valueDim] :=
  let headOutput := batchedVecMatTerm heads keyDim valueDim query state
  Term.op (PrimOp.binary fun output gamma => Tensor.zipEach [heads] [valueDim]
    (RMSNorm.scalePositive hValue) output gamma)
    (.cons headOutput (.cons outputNormScale .nil))

/-- Evaluating the head-read term gives the normalized KDA head readouts. -/
@[simp] theorem eval_normalizedHeadReadTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (heads keyDim valueDim : Nat) (hHeads : 0 < heads) (hValue : 0 < valueDim)
    (query : Term Γ [heads, keyDim])
    (state : Term Γ [heads, keyDim, valueDim])
    (outputNormScale : Term Γ [heads, valueDim]) :
    Term.eval env (normalizedHeadReadTerm heads keyDim valueDim hHeads hValue
      query state outputNormScale) =
      .dim (fun head => RMSNorm.scale
        (KDA.read (Tensor.unstack (Term.eval env state) head)
          (Tensor.unstack (Term.eval env query) head))
        (Tensor.unstack (Term.eval env outputNormScale) head)) := by
  simp [normalizedHeadReadTerm, Term.evalArgs, PrimOp.binary, KDA.read, Tensor.zipEach,
    RMSNorm.scale_eq_scalePositive hValue, Pack.head, Pack.tail]

/-- Compute the full-rank output gate for every KDA head. -/
def outputGateTerm {Γ : List Shape} (modelDim heads valueDim : Nat)
    (current : Term Γ [modelDim])
    (gateWeight : Term Γ [modelDim, heads, valueDim]) :
    Term Γ [heads, valueDim] :=
  let gateMatrices :=
    Term.op (PrimOp.swapAdjacentAtDepth
      [modelDim, heads, valueDim] 0) (.cons gateWeight .nil)
  let gateLogit := batchedSharedVecMatTerm heads modelDim valueDim current gateMatrices
  let gate :=
    Term.op (PrimOp.sigmoid [heads, valueDim])
      (.cons gateLogit .nil)
  gate

/-- Evaluating the output-gate term gives the packed gate coordinates used by `KDA.Layer.gates`. -/
@[simp] theorem eval_outputGateTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (modelDim heads valueDim : Nat)
    (current : Term Γ [modelDim])
    (gateWeight : Term Γ [modelDim, heads, valueDim]) :
    Term.eval env (outputGateTerm modelDim heads valueDim current gateWeight) =
      .dim (fun head => Tensor.map Context.sigmoid <|
        Tensor.vecmat (Term.eval env current) <|
          .dim (fun input => Tensor.unstack (Tensor.unstack (Term.eval env gateWeight) input) head)) := by
  simp only [outputGateTerm, Term.eval_op, Term.evalArgs,
    eval_batchedSharedVecMatTerm, PrimOp.swapAdjacentAtDepth_specFwd,
    PrimOp.sigmoid_specFwd, Tensor.sigmoid, Tensor.map_dim]
  congr 1
  funext head
  congr 2
  apply Tensor.Rep.ext
  rintro ⟨input, column⟩
  rw [Tensor.unstack_apply, Tensor.permute_adjacentSwap_zero_apply,
    Tensor.dim_apply, Tensor.unstack_apply, Tensor.unstack_apply]

/-- Gate, project, and sum the packed KDA head outputs. -/
def projectHeadReadoutTerm {Γ : List Shape} (modelDim heads valueDim : Nat)
    (_hModel : 0 < modelDim) (_hHeads : 0 < heads)
    (gate normalizedHeadOutput : Term Γ [heads, valueDim])
    (outputWeight : Term Γ [heads, valueDim, modelDim]) :
    Term Γ [modelDim] :=
  let gated := Term.op (PrimOp.mul [heads, valueDim])
    (.cons gate (.cons normalizedHeadOutput .nil))
  let projectedHeads := batchedVecMatTerm heads valueDim modelDim gated outputWeight
  Term.op (PrimOp.reduceSum
    [heads, modelDim] 0) (.cons projectedHeads .nil)

/-- Evaluating the packed projection sums the gated per-head projections over heads. -/
@[simp] theorem eval_projectHeadReadoutTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (modelDim heads valueDim : Nat) (hHeads : 0 < heads)
    (hModel : 0 < modelDim)
    (gate normalizedHeadOutput : Term Γ [heads, valueDim])
    (outputWeight : Term Γ [heads, valueDim, modelDim]) :
    Term.eval env (projectHeadReadoutTerm modelDim heads valueDim hModel hHeads
      gate normalizedHeadOutput outputWeight) =
      Tensor.reduceDimWith Tensor.sum [heads, modelDim] 0
        (.dim (fun head => Tensor.vecmat
          (Tensor.unstack (Term.eval env gate * Term.eval env normalizedHeadOutput) head)
          (Tensor.unstack (Term.eval env outputWeight) head))) := by
  rfl

/-- Post-update KDA readout as a compositional TorchLean term.

The recurrent state is an ordinary term argument, so callers may pass either an input state or a
shared `let`-bound update.
-/
def readoutTerm {Γ : List Shape} (modelDim heads keyDim valueDim : Nat)
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hValue : 0 < valueDim)
    (current : Term Γ [modelDim])
    (query : Term Γ [heads, keyDim])
    (state : Term Γ [heads, keyDim, valueDim])
    (gateWeight : Term Γ [modelDim, heads, valueDim])
    (outputWeight : Term Γ [heads, valueDim, modelDim])
    (outputNormScale : Term Γ [heads, valueDim]) :
    Term Γ [modelDim] :=
  let normalized := normalizedHeadReadTerm heads keyDim valueDim hHeads hValue
    query state outputNormScale
  let gate := outputGateTerm modelDim heads valueDim current gateWeight
  projectHeadReadoutTerm modelDim heads valueDim hModel hHeads gate normalized outputWeight

private theorem reduceSum_outer_matrix_eq_foldl {rows columns : Nat}
    (tensor : Tensor ℝ [rows, columns]) :
    Tensor.reduceDimWith Tensor.sum [rows, columns] 0 tensor =
      Tensor.dim fun column => Tensor.scalar <|
        (List.finRange rows).foldl
          (fun total row => total + Tensor.getScalar (Tensor.unstack tensor row) column) 0 := by
  have vectorSum (values : Tensor ℝ [rows]) :
      Tensor.sum values = ∑ i : Fin rows, Tensor.getScalar values i := by
    classical
    rw [Tensor.sum_eq_coord_sum, Fintype.sum_prod_type]
    simp [Tensor.getScalar_eq_apply]
  change Tensor.reduceOuterAxis Tensor.sum tensor = _
  apply Tensor.ext_vector
  intro column
  simp only [Tensor.getScalar_eq_apply, Tensor.reduceOuterAxis_apply, Tensor.dim_apply,
    Tensor.scalar_apply]
  rw [vectorSum, Fin.sum_univ_def, List.sum_eq_foldl, List.foldl_map]
  simp only [Tensor.getScalar_ofFn, Tensor.unstack_apply]

/-- Evaluating the readout term gives `KDA.Layer.readout`. -/
theorem eval_readoutTerm {Γ : List Shape}
    (env : TorchLean.TensorPack ℝ Γ) (modelDim heads keyDim valueDim : Nat)
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hValue : 0 < valueDim)
    (current : Term Γ [modelDim])
    (query : Term Γ [heads, keyDim])
    (state : Term Γ [heads, keyDim, valueDim])
    (gateWeight : Term Γ [modelDim, heads, valueDim])
    (outputWeight : Term Γ [heads, valueDim, modelDim])
    (outputNormScale : Term Γ [heads, valueDim]) :
    Term.eval env (readoutTerm modelDim heads keyDim valueDim hModel hHeads hValue
      current query state gateWeight outputWeight outputNormScale) =
      KDA.Layer.readout (Term.eval env state) (Term.eval env query)
        (Tensor.dim fun head =>
          Tensor.map Context.sigmoid <|
            Tensor.vecmat (Term.eval env current) <| Tensor.dim fun input =>
              Tensor.unstack (Tensor.unstack (Term.eval env gateWeight) input) head)
        (Term.eval env outputWeight) (Term.eval env outputNormScale) := by
  simp only [readoutTerm, eval_projectHeadReadoutTerm, eval_outputGateTerm,
    eval_normalizedHeadReadTerm, KDA.Layer.readout]
  rw [reduceSum_outer_matrix_eq_foldl]

/-- Readout of a `let`-bound recurrent update, with every other argument weakened past it. -/
@[simp] theorem eval_readoutTerm_after_append {Γ : List Shape}
    (env : TorchLean.TensorPack ℝ Γ) (modelDim heads keyDim valueDim : Nat)
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hValue : 0 < valueDim)
    (current : Term Γ [modelDim])
    (query : Term Γ [heads, keyDim])
    (nextState : Tensor ℝ [heads, keyDim, valueDim])
    (gateWeight : Term Γ [modelDim, heads, valueDim])
    (outputWeight : Term Γ [heads, valueDim, modelDim])
    (outputNormScale : Term Γ [heads, valueDim]) :
    Term.eval (TorchLean.Pack.append env (.cons nextState .nil))
        (readoutTerm modelDim heads keyDim valueDim hModel hHeads hValue
          (Term.weakenRight current) (Term.weakenRight query) (Term.var (Var.last Γ))
          (Term.weakenRight gateWeight) (Term.weakenRight outputWeight)
          (Term.weakenRight outputNormScale)) =
      KDA.Layer.readout nextState (Term.eval env query)
        (Tensor.dim fun head =>
          Tensor.map Context.sigmoid <|
            Tensor.vecmat (Term.eval env current) <| Tensor.dim fun input =>
              Tensor.unstack (Tensor.unstack (Term.eval env gateWeight) input) head)
        (Term.eval env outputWeight) (Term.eval env outputNormScale) := by
  rw [eval_readoutTerm]
  simp only [Term.eval_var_last_append, Term.eval_weakenRight]

/-- First half of the fixed-window KDA graph: project the current window and form the recurrent
inputs for every head.

The six outputs are the quantities listed in `Prepared`, which gives the recurrent update a typed
interface. -/
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

/-- Reference values of the `prepareBlock` outputs, in the same order.

The entries are the current token and the five fields of `KDA.Head.prepareWindow` applied to each
head. -/
def preparedTensors {α : Type} [Storage α] [Context α]
    {modelDim heads keyDim valueDim convWidth decayRank : Nat}
    (layer : KDA.Layer α modelDim heads keyDim valueDim convWidth decayRank)
    (hWidth : 0 < convWidth) (logFloor : α)
    (window : Tensor α [convWidth, modelDim]) :
    TorchLean.TensorPack α (Prepared modelDim heads keyDim valueDim) :=
  let current := Tensor.unstack window ⟨0, hWidth⟩
  let prepared := fun head => (layer.head head).prepareWindow hWidth logFloor window
  .cons current <|
    .cons (Tensor.dim fun head => (prepared head).query) <|
    .cons (Tensor.dim fun head => (prepared head).key) <|
    .cons (Tensor.dim fun head => (prepared head).value) <|
    .cons (Tensor.dim fun head => (prepared head).retention) <|
    .cons (Tensor.dim fun head => Tensor.scalar (prepared head).writeStrength) .nil

set_option backward.isDefEq.respectTransparency false in
/-- With epsilon `Normalize.l2Epsilon`, the first graph block computes `preparedTensors`. -/
theorem eval_prepareBlock_eq_preparedTensors
    {modelDim heads keyDim valueDim convWidth decayRank : Nat}
    (hHeads : 0 < heads) (hKey : 0 < keyDim) (hValue : 0 < valueDim)
    (hWidth : 0 < convWidth)
    (layer : KDA.Layer ℝ modelDim heads keyDim valueDim convWidth decayRank)
    (window : Tensor ℝ [convWidth, modelDim])
    (state : KDA.Layer.State ℝ heads keyDim valueDim) (logFloor : ℝ) :
    Block.eval
        (TorchLean.Pack.append (layerParameters layer)
          (layerInputs window state logFloor Normalize.l2Epsilon))
        (prepareBlock modelDim heads keyDim valueDim convWidth decayRank
          hHeads hKey hValue hWidth) =
      preparedTensors layer hWidth logFloor window := by
  simp [prepareBlock, preparedTensors, prepareTerms, layerParameters, layerInputs,
    TorchLean.Pack.append, Args.vars, Args.weakenLeft, Args.rename, Term.evalArgs, Env.tget,
    PrimOp.unary, PrimOp.binary, PrimOp.exp, Pack.head, Pack.tail,
    KDA.Head.prepareWindow, KDA.Head.activate, ShortConv.forwardWindow,
    Tensor.add_def, Tensor.mul_def, Tensor.mapLeading, Tensor.map_dim, Tensor.sigmoid,
    Tensor.exp, Tensor.zipWith_dim]
  apply Tensor.Rep.ext
  rintro ⟨head, coordinate⟩
  simp [Tensor.dim_apply]

/-- Packed recurrent update used by the second half of a KDA layer.

It contains only the state transition of Eq. 1 in the K3 report. Projection, convolution, and
readout are separate terms, each with its own refinement theorem. -/
def updateTerm {Γ : List Shape} (heads keyDim valueDim : Nat)
    (state : Term Γ [heads, keyDim, valueDim])
    (key : Term Γ [heads, keyDim])
    (value : Term Γ [heads, valueDim])
    (retention : Term Γ [heads, keyDim])
    (beta : Term Γ [heads]) :
    Term Γ [heads, keyDim, valueDim] :=
  let decayed := Term.op (PrimOp.mul [heads, keyDim, valueDim])
    (.cons (expandTerm [heads, keyDim] [valueDim] retention) (.cons state .nil))
  let correction := batchedVecMatTerm heads keyDim valueDim key decayed
  let correctionOuter := outerTerm [heads] keyDim valueDim key correction
  let writeOuter := outerTerm [heads] keyDim valueDim key value
  let scaledCorrection := Term.op (PrimOp.mul [heads, keyDim, valueDim])
    (.cons (expandTerm [heads] [keyDim, valueDim] beta) (.cons correctionOuter .nil))
  let scaledWrite := Term.op (PrimOp.mul [heads, keyDim, valueDim])
    (.cons (expandTerm [heads] [keyDim, valueDim] beta) (.cons writeOuter .nil))
  let corrected := Term.op
    (PrimOp.sub [heads, keyDim, valueDim])
    (.cons decayed (.cons scaledCorrection .nil))
  Term.op
    (PrimOp.add [heads, keyDim, valueDim])
    (.cons corrected (.cons scaledWrite .nil))

/-- `updateTerm` denotes one independent KDA recurrence update for every packed head. -/
theorem eval_updateTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (heads keyDim valueDim : Nat)
    (state : Term Γ [heads, keyDim, valueDim])
    (query : Term Γ [heads, keyDim])
    (key : Term Γ [heads, keyDim])
    (value : Term Γ [heads, valueDim])
    (retention : Term Γ [heads, keyDim])
    (beta : Term Γ [heads]) :
    Term.eval env (updateTerm heads keyDim valueDim state key value retention beta) =
      Tensor.dim fun head =>
        KDA.update (Tensor.unstack (Term.eval env state) head)
          { query := Tensor.unstack (Term.eval env query) head
            key := Tensor.unstack (Term.eval env key) head
            value := Tensor.unstack (Term.eval env value) head
            retention := Tensor.unstack (Term.eval env retention) head
            writeStrength := (Tensor.unstack (Term.eval env beta) head).item } := by
  have decayed :
      Tensor.zipWith (s := [heads, keyDim, valueDim]) (· * ·)
        (Tensor.mapLeading [heads, keyDim] (fun x => Tensor.full [valueDim] x.item)
          (Term.eval env retention)) (Term.eval env state) =
        Tensor.dim (fun head => Tensor.dim (fun row => Tensor.dim (fun column =>
          Tensor.scalar (((Term.eval env retention).unstack head).getScalar row *
            Tensor.get2 ((Term.eval env state).unstack head) row column)))) := by
    apply TorchLean.Tensor.Rep.ext
    rintro ⟨head, row, column, ⟨⟩⟩
    simp [Tensor.mapLeading, Tensor.dim_apply, Tensor.unstack_apply,
      Tensor.getScalar_eq_apply, Tensor.get2_eq_apply, Tensor.item]
  simp only [updateTerm, Term.eval_op, Term.evalArgs,
    eval_expandTerm, PrimOp.mul_specFwd,
    eval_batchedVecMatTerm,
    eval_outerTerm,
    PrimOp.sub_specFwd, PrimOp.add_specFwd]
  simp only [Tensor.mul_def]
  rw [decayed]
  apply Tensor.Rep.ext
  rintro ⟨head, row, column, ⟨⟩⟩
  simp [KDA.update, Tensor.zipEach, Tensor.mapLeading,
    Tensor.dim_apply, Tensor.unstack_apply, Tensor.getScalar_eq_apply,
    Tensor.item]

/-- Assemble the recurrent update and readout from terms in an arbitrary graph environment. -/
def updateReadoutTermsBlock {Γ : List Shape}
    (modelDim heads keyDim valueDim : Nat)
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hValue : 0 < valueDim)
    (state : Term Γ [heads, keyDim, valueDim])
    (current : Term Γ [modelDim])
    (query key : Term Γ [heads, keyDim])
    (value : Term Γ [heads, valueDim])
    (retention : Term Γ [heads, keyDim])
    (beta : Term Γ [heads])
    (gateWeight : Term Γ [modelDim, heads, valueDim])
    (outputWeight : Term Γ [heads, valueDim, modelDim])
    (outputNormScale : Term Γ [heads, valueDim]) :
    Block Γ (LayerOutputs modelDim heads keyDim valueDim) :=
  let nextState := updateTerm heads keyDim valueDim state key value retention beta
  let Γnext := Γ ++ [[heads, keyDim, valueDim]]
  let nextState' : Term Γnext [heads, keyDim, valueDim] :=
    Term.var (Var.last Γ)
  let output := readoutTerm modelDim heads keyDim valueDim hModel hHeads hValue
    (Term.weakenRight current) (Term.weakenRight query) nextState'
    (Term.weakenRight gateWeight) (Term.weakenRight outputWeight)
    (Term.weakenRight outputNormScale)
  Block.let1 nextState <| Block.ret <| .cons nextState' (.cons output .nil)

/-- Evaluating the update/readout block gives the per-head update followed by
`KDA.Layer.readout`, as in `KDA.Layer.stepPrepared`. -/
theorem eval_updateReadoutTermsBlock {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    {modelDim heads keyDim valueDim : Nat}
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hValue : 0 < valueDim)
    (state : Term Γ [heads, keyDim, valueDim])
    (current : Term Γ [modelDim])
    (query key : Term Γ [heads, keyDim])
    (value : Term Γ [heads, valueDim])
    (retention : Term Γ [heads, keyDim])
    (beta : Term Γ [heads])
    (gateWeight : Term Γ [modelDim, heads, valueDim])
    (outputWeight : Term Γ [heads, valueDim, modelDim])
    (outputNormScale : Term Γ [heads, valueDim]) :
    Block.eval env
        (updateReadoutTermsBlock modelDim heads keyDim valueDim hModel hHeads hValue
          state current query key value retention beta gateWeight outputWeight outputNormScale) =
      let prepared : Fin heads → KDA.Input ℝ keyDim valueDim := fun head =>
        { query := Tensor.unstack (Term.eval env query) head
          key := Tensor.unstack (Term.eval env key) head
          value := Tensor.unstack (Term.eval env value) head
          retention := Tensor.unstack (Term.eval env retention) head
          writeStrength := (Tensor.unstack (Term.eval env beta) head).item }
      let next := Tensor.dim fun head => KDA.update (Tensor.unstack (Term.eval env state) head)
        (prepared head)
      .cons next <| .cons
        (KDA.Layer.readout next (Term.eval env query)
          (Tensor.dim fun head =>
            Tensor.map Context.sigmoid <|
              Tensor.vecmat (Term.eval env current) <| Tensor.dim fun input =>
                Tensor.unstack (Tensor.unstack (Term.eval env gateWeight) input) head)
          (Term.eval env outputWeight) (Term.eval env outputNormScale)) .nil := by
  simp only [updateReadoutTermsBlock, Block.eval_let1, Block.eval_ret]
  rw [eval_updateTerm env heads keyDim valueDim state query key value retention beta]
  simp only [Term.evalArgs, Term.eval_var_last_append]
  rw [eval_readoutTerm_after_append]

/-- Second half of the fixed-window KDA graph.

The parameter and input environment stays on the left, with the six prepared tensors appended on
the right. The new recurrent state is bound once because it is both returned and read by the
readout. -/
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
  let gateWeight : Term Γ [modelDim, heads, valueDim] :=
    .var <| Var.inLeft prepared <| Var.inLeft inputs <| Var.inRight prepParams .head
  let outputWeight : Term Γ [heads, valueDim, modelDim] :=
    .var <| Var.inLeft prepared <| Var.inLeft inputs <|
      Var.inRight prepParams (.tail .head)
  let outputNormScale : Term Γ [heads, valueDim] :=
    .var <| Var.inLeft prepared <| Var.inLeft inputs <|
      Var.inRight prepParams (.tail (.tail .head))
  let state : Term Γ [heads, keyDim, valueDim] :=
    .var <| Var.inLeft prepared <| Var.inRight params (.tail .head)
  let current : Term Γ [modelDim] :=
    .var <| Var.inRight base .head
  let query : Term Γ [heads, keyDim] :=
    .var <| Var.inRight base (.tail .head)
  let key : Term Γ [heads, keyDim] :=
    .var <| Var.inRight base (.tail (.tail .head))
  let value : Term Γ [heads, valueDim] :=
    .var <| Var.inRight base (.tail (.tail (.tail .head)))
  let retention : Term Γ [heads, keyDim] :=
    .var <| Var.inRight base (.tail (.tail (.tail (.tail .head))))
  let beta : Term Γ [heads] :=
    .var <| Var.inRight base (.tail (.tail (.tail (.tail (.tail .head)))))
  updateReadoutTermsBlock modelDim heads keyDim valueDim hModel hHeads hValue
    state current query key value retention beta gateWeight outputWeight outputNormScale

-- Keep tensor computations folded while elaborating the packed result's type.
attribute [local irreducible] KDA.Layer.stepPrepared KDA.Layer.readout KDA.update

/-- `updateReadoutBlock` computes `KDA.Layer.stepPrepared` on the six prepared tensors.

The statement rebuilds `KDA.Input` only to match those tensors with the arguments of
`stepPrepared`. -/
theorem eval_updateReadoutBlock_eq_stepPrepared
    {modelDim heads keyDim valueDim convWidth decayRank : Nat}
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hValue : 0 < valueDim)
    (layer : KDA.Layer ℝ modelDim heads keyDim valueDim convWidth decayRank)
    (window : Tensor ℝ [convWidth, modelDim])
    (state : KDA.Layer.State ℝ heads keyDim valueDim) (logFloor epsilon : ℝ)
    (current : Tensor ℝ [modelDim])
    (query key : Tensor ℝ [heads, keyDim])
    (value : Tensor ℝ [heads, valueDim])
    (retention : Tensor ℝ [heads, keyDim])
    (beta : Tensor ℝ [heads]) :
    Block.eval
        (TorchLean.Pack.append
          (TorchLean.Pack.append (layerParameters layer)
            (layerInputs window state logFloor epsilon))
          (.cons current <| .cons query <| .cons key <| .cons value <|
            .cons retention <| .cons beta .nil))
        (updateReadoutBlock modelDim heads keyDim valueDim convWidth decayRank
          hModel hHeads hValue) =
      let prepared : Fin heads → KDA.Input ℝ keyDim valueDim := fun head =>
        { query := Tensor.unstack query head
          key := Tensor.unstack key head
          value := Tensor.unstack value head
          retention := Tensor.unstack retention head
          writeStrength := (Tensor.unstack beta head).item }
      let result := layer.stepPrepared state current prepared
      .cons result.1 (.cons result.2 .nil) := by
  simp only [updateReadoutBlock]
  rw [eval_updateReadoutTermsBlock]
  simp [Term.eval_var, Var.inLeft, Var.inRight, Env.tget,
    TorchLean.Pack.append,
    layerParameters, layerInputs,
    KDA.Layer.stepPrepared, KDA.Layer.gates]

/-- Complete fixed-window multi-head KDA token step as a typed TorchLean graph. -/
def layerModel (modelDim heads keyDim valueDim convWidth decayRank : Nat)
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hKey : 0 < keyDim)
    (hValue : 0 < valueDim) (hWidth : 0 < convWidth) :
    MultiModel
      (LayerParams modelDim heads keyDim valueDim convWidth decayRank)
      (LayerInputs modelDim heads keyDim valueDim convWidth)
      (LayerOutputs modelDim heads keyDim valueDim) :=
  { initParams := initialLayerParams modelDim heads keyDim valueDim convWidth decayRank
    body := (prepareBlock modelDim heads keyDim valueDim convWidth decayRank
      hHeads hKey hValue hWidth).andThen <|
        updateReadoutBlock modelDim heads keyDim valueDim convWidth decayRank
          hModel hHeads hValue }

/-- With epsilon `Normalize.l2Epsilon`, the packed KDA graph computes `KDA.Layer.stepWindow`. -/
theorem layerModel_specFwd_eq_stepWindow
    {modelDim heads keyDim valueDim convWidth decayRank : Nat}
    (hModel : 0 < modelDim) (hHeads : 0 < heads) (hKey : 0 < keyDim)
    (hValue : 0 < valueDim) (hWidth : 0 < convWidth)
    (layer : KDA.Layer ℝ modelDim heads keyDim valueDim convWidth decayRank)
    (window : Tensor ℝ [convWidth, modelDim])
    (state : KDA.Layer.State ℝ heads keyDim valueDim) (logFloor : ℝ) :
    (layerModel modelDim heads keyDim valueDim convWidth decayRank
      hModel hHeads hKey hValue hWidth).specFwd (layerParameters layer)
        (layerInputs window state logFloor Normalize.l2Epsilon) =
      let result := layer.stepWindow hWidth logFloor window state
      .cons result.1 (.cons result.2 .nil) := by
  simp only [MultiModel.specFwd, layerModel]
  rw [Block.eval_andThen]
  rw [eval_prepareBlock_eq_preparedTensors hHeads hKey hValue hWidth layer window state logFloor]
  simp only [preparedTensors]
  rw [eval_updateReadoutBlock_eq_stepPrepared]
  simp [KDA.Layer.stepWindow]

end KDA
end GraphSpec
end KimiK3
