/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Common
public import KimiK3.Sequence.Scan
public import KimiK3.Config
public import NN.Spec.Layers.Activation
public import Mathlib.Analysis.SpecialFunctions.Sigmoid

/-!
# Kimi Delta Attention

`update` decays the recurrent matrix and writes a key-aligned value correction. `Layer` adds
the learned projections, newest-first short-convolution windows, normalization, and output
gating. Sequence and streaming functions share this token transition, and their scan laws
compare runs with the same operations and inputs.

Reference: Kimi K3 technical report, Sections 2.1--2.2, pp. 4--6, Eqs. 1--10.
https://arxiv.org/abs/2607.24653.
-/

@[expose] public section

namespace KimiK3

open Spec TorchLean
open Tensor

/-- Parameters of one causal depthwise short convolution. -/
structure ShortConv (α : Type) [Storage α] (width channels : Nat) where
  /-- One scalar coefficient for each tap and channel. -/
  weight : Tensor α [width, channels]
  /-- Per-channel bias. -/
  bias : Tensor α [channels]

namespace ShortConv

variable {α : Type} [Storage α] [Context α]
variable {width channels : Nat}

/--
Apply a short convolution to a newest-first projected-token history.  Missing old tokens are zero
padded, so the definition is causal for every history length.
-/
def forward (conv : ShortConv α width channels)
    (history : List (Tensor α [channels])) : Tensor α [channels] :=
  let zero : Tensor α [channels] := Tensor.zeros [channels]
  Tensor.stackLeading (fun channel => Tensor.scalar <|
    (List.finRange width).foldl (fun total tap =>
      total + Tensor.getScalar (history.getD tap.val zero) channel * get2 conv.weight tap channel)
      (Tensor.getScalar conv.bias channel))

/-- A short convolution never reads beyond its window. Truncating the history keeps the addition
order and every scalar operation, so this holds in rounded scalar contexts too. -/
theorem forward_take (conv : ShortConv α width channels)
    (history : List (Tensor α [channels])) :
    conv.forward (history.take width) = conv.forward history := by
  unfold forward
  apply congrArg Tensor.stackLeading
  funext channel
  apply congrArg Tensor.scalar
  apply List.foldl_ext
  intro total tap _
  simp only [List.getD_eq_getElem?_getD, List.getElem?_take_of_lt tap.isLt]

/-- Apply the causal convolution to a full newest-first window.

The leading axis is the tap axis: row zero is the current projected token and later rows are
successively older tokens. Streaming code keeps this fixed-size window, while the list-based
`forward` serves the zero-padded causal specification. Here the bias is added after the tap sum
(`forward` starts its fold from the bias), so rounded results need not agree with `forward`.
-/
def forwardWindow (conv : ShortConv α width channels) (hWidth : 0 < width)
    (window : Tensor α [width, channels]) :
    Tensor α [channels] :=
  Tensor.addSpec
    (Tensor.reduceSum 0 (Tensor.mulSpec window conv.weight)
      (Shape.hasNonemptyAxisZeroOfPos hWidth).proof) conv.bias

end ShortConv

/-- Input to the mathematical single-head KDA recurrence in Eq. 1. -/
structure KDA.Input (α : Type) [Storage α] (keyDim valueDim : Nat) where
  /-- Query used to read the updated recurrent matrix. -/
  query : Tensor α [keyDim]
  /-- Unit-regularized key used by the delta update. -/
  key : Tensor α [keyDim]
  /-- Value written into the recurrent matrix. -/
  value : Tensor α [valueDim]
  /-- Coordinate-wise multiplicative retention. -/
  retention : Tensor α [keyDim]
  /-- Scalar delta-rule write gate. -/
  writeStrength : α

namespace KDA

variable {α : Type} [Storage α] [Context α]
variable {keyDim valueDim : Nat}

/-- Recurrent KDA state `S ∈ R^(d_k × d_v)`. -/
abbrev State (α : Type) [Storage α] (keyDim valueDim : Nat) :=
  Tensor α [keyDim, valueDim]

/--
One single-head KDA recurrence (Eq. 1):

`S_t = (I - β_t k_t k_tᵀ) Diag(α_t) S_(t-1) + β_t k_t v_tᵀ`.

It is implemented in the factorized form of an efficient kernel: decay the rows of the state,
read the decayed state with the key, subtract the key-aligned correction, then add the rank-one
write.
-/
def update (state : State α keyDim valueDim) (input : KDA.Input α keyDim valueDim) :
    State α keyDim valueDim :=
  let decayed := Tensor.stackLeading fun row => Tensor.stackLeading fun column => Tensor.scalar <|
    Tensor.getScalar input.retention row * get2 state row column
  let correction := vecMatMulSpec input.key decayed
  Tensor.addSpec
    (Tensor.subSpec decayed
      (Tensor.mulSpec
        (Tensor.full [keyDim, valueDim] input.writeStrength)
        (outerProductSpec input.key correction)))
    (Tensor.mulSpec
      (Tensor.full [keyDim, valueDim] input.writeStrength)
      (outerProductSpec input.key input.value))

/-- Coordinate form of the factorized KDA update.

Its correction coordinate reads the key from the already decayed state, so decay comes before the
delta correction.
-/
theorem update_get2 (state : State ℝ keyDim valueDim)
    (input : KDA.Input ℝ keyDim valueDim) (row : Fin keyDim) (column : Fin valueDim) :
  let decayed := Tensor.stackLeading fun source => Tensor.stackLeading fun target => Tensor.scalar
    (Tensor.getScalar input.retention source * get2 state source target)
  get2 (update state input) row column =
      Tensor.getScalar input.retention row * get2 state row column -
        input.writeStrength * Tensor.getScalar input.key row *
          Tensor.getScalar (vecMatMulSpec input.key decayed) column +
        input.writeStrength * Tensor.getScalar input.key row *
          Tensor.getScalar input.value column := by
  simp [update, Tensor.stackLeading, Tensor.addSpec, Tensor.subSpec, Tensor.mulSpec]
  ring

/-- Read the current output after the state update: `o_t = S_tᵀ q_t`. -/
def read (state : State α keyDim valueDim) (query : Tensor α [keyDim]) :
    Tensor α [valueDim] :=
  vecMatMulSpec query state

/-- One recurrence step returns the updated state and its current-token output. -/
def step (state : State α keyDim valueDim) (input : KDA.Input α keyDim valueDim) :
    State α keyDim valueDim × Tensor α [valueDim] :=
  let next := update state input
  (next, read next input.query)

/-- Sequential semantics for a projected KDA sequence. -/
def run (state : State α keyDim valueDim) (inputs : List (KDA.Input α keyDim valueDim)) :=
  CausalScan.run step state inputs

/-- A single-head KDA scan emits one value vector per projected token. -/
theorem run_outputs_length (state : State α keyDim valueDim)
    (inputs : List (KDA.Input α keyDim valueDim)) :
    (run state inputs).2.length = inputs.length := by
  exact CausalScan.outputs_length step state inputs

/-- Chunk boundaries do not change the KDA recurrence or emitted outputs. -/
theorem run_append (state : State α keyDim valueDim)
    (left right : List (KDA.Input α keyDim valueDim)) :
    run state (left ++ right) =
      let first := run state left
      let second := run first.1 right
      (second.1, first.2 ++ second.2) :=
  CausalScan.run_append step state left right

/-- Lower-bounded real retention from Eq. 5. -/
noncomputable def retentionReal (logFloor logScale logit : ℝ) : ℝ :=
  Real.exp (logFloor * Real.sigmoid (Real.exp logScale * logit))

/--
K3's retention parameterization lies strictly between `exp(logFloor)` and one, so no channel's
retention can approach zero.
-/
theorem retentionReal_bounds {logFloor : ℝ} (hFloor : logFloor < 0) (logScale logit : ℝ) :
    Real.exp logFloor < retentionReal logFloor logScale logit ∧
      retentionReal logFloor logScale logit < 1 := by
  have hSigPos : 0 < Real.sigmoid (Real.exp logScale * logit) := Real.sigmoid_pos _
  have hSigLt : Real.sigmoid (Real.exp logScale * logit) < 1 := Real.sigmoid_lt_one _
  have hLower : logFloor < logFloor * Real.sigmoid (Real.exp logScale * logit) := by
    nlinarith
  have hUpper : logFloor * Real.sigmoid (Real.exp logScale * logit) < 0 := by
    exact mul_neg_of_neg_of_pos hFloor hSigPos
  constructor
  · exact Real.exp_lt_exp.mpr hLower
  · simpa [retentionReal] using Real.exp_lt_one_iff.mpr hUpper

/-- With the paper's floor `logFloor = -5`, every one-step retention lies strictly between
`exp(-5)` and `1`. -/
theorem paper_retention_bounds (logScale logit : ℝ) :
    Real.exp (-5) < retentionReal (-5) logScale logit ∧
      retentionReal (-5) logScale logit < 1 := by
  exact retentionReal_bounds (by norm_num) logScale logit

end KDA

/-- Projection and gating parameters for one KDA head. -/
structure KDA.Head (α : Type) [Storage α] (modelDim keyDim valueDim convWidth decayRank : Nat) where
  /-- Model-to-query projection. -/
  queryWeight : Tensor α [modelDim, keyDim]
  /-- Model-to-key projection. -/
  keyWeight : Tensor α [modelDim, keyDim]
  /-- Model-to-value projection. -/
  valueWeight : Tensor α [modelDim, valueDim]
  /-- Depthwise causal convolution for projected queries. -/
  queryConv : ShortConv α convWidth keyDim
  /-- Depthwise causal convolution for projected keys. -/
  keyConv : ShortConv α convWidth keyDim
  /-- Depthwise causal convolution for projected values. -/
  valueConv : ShortConv α convWidth valueDim
  /-- Projection producing the scalar delta-rule write gate. -/
  betaWeight : Tensor α [modelDim]
  /-- First factor of the low-rank retention projection. -/
  decayDown : Tensor α [modelDim, decayRank]
  /-- Second factor of the low-rank retention projection. -/
  decayUp : Tensor α [decayRank, keyDim]
  /-- Coordinate-wise retention bias. -/
  decayBias : Tensor α [keyDim]
  /-- Learned logarithm of the retention-logit scale. -/
  decayLogScale : α

namespace KDA.Head

variable {α : Type} [Storage α] [Context α]
variable {modelDim keyDim valueDim convWidth decayRank : Nat}

/-- Build the recurrence input from the three short-convolution outputs: swish activations, L2
normalization of query and key, and the retention and write gates of the current token. -/
def activate (head : KDA.Head α modelDim keyDim valueDim convWidth decayRank)
    (logFloor : α) (current : Tensor α [modelDim])
    (qPre kPre : Tensor α [keyDim]) (vPre : Tensor α [valueDim]) :
    KDA.Input α keyDim valueDim :=
  let query := Spec.normalizeL2RegularizedSpec (Tensor.map Activation.Math.swishSpec qPre)
    Normalize.l2Epsilon
  let key := Spec.normalizeL2RegularizedSpec (Tensor.map Activation.Math.swishSpec kPre)
    Normalize.l2Epsilon
  let value := Tensor.map Activation.Math.swishSpec vPre
  let decayLogit :=
    Tensor.addSpec (vecMatMulSpec (vecMatMulSpec current head.decayDown) head.decayUp)
      head.decayBias
  let scale := MathFunctions.exp head.decayLogScale
  let retention := Tensor.map
    (fun z => MathFunctions.exp (logFloor * Activation.Math.sigmoidSpec (scale * z))) decayLogit
  let beta := Activation.Math.sigmoidSpec (Tensor.dotSpec current head.betaWeight)
  { query, key, value, retention, writeStrength := beta }

/-- Prepare one recurrence step, projecting only the tokens read by the convolution window. -/
def prepare (head : KDA.Head α modelDim keyDim valueDim convWidth decayRank)
    (logFloor : α) (current : Tensor α [modelDim])
    (history : List (Tensor α [modelDim])) : KDA.Input α keyDim valueDim :=
  let window := (current :: history).take convWidth
  head.activate logFloor current
    (head.queryConv.forward (window.map (vecMatMulSpec · head.queryWeight)))
    (head.keyConv.forward (window.map (vecMatMulSpec · head.keyWeight)))
    (head.valueConv.forward (window.map (vecMatMulSpec · head.valueWeight)))

/-- Projecting only the convolution window agrees with projecting the complete history, with no
associativity or real-arithmetic assumption. -/
theorem prepare_eq_full_history (head : KDA.Head α modelDim keyDim valueDim convWidth decayRank)
    (logFloor : α) (current : Tensor α [modelDim])
    (history : List (Tensor α [modelDim])) :
    head.prepare logFloor current history = head.activate logFloor current
      (head.queryConv.forward ((current :: history).map (vecMatMulSpec · head.queryWeight)))
      (head.keyConv.forward ((current :: history).map (vecMatMulSpec · head.keyWeight)))
      (head.valueConv.forward ((current :: history).map (vecMatMulSpec · head.valueWeight))) := by
  simp [prepare, List.map_take, ShortConv.forward_take]

/-- Construct one head's recurrence input from a fixed newest-first model-width window.

This is the tensor ABI of the executable graph. One matrix multiplication projects the whole
window before the three depthwise convolutions. L2 normalization uses the `10⁻⁶` stabilizer of
the released FlashKDA reference and CUDA kernel.
-/
def prepareWindow (head : KDA.Head α modelDim keyDim valueDim convWidth decayRank)
    (hWidth : 0 < convWidth) (logFloor : α)
    (window : Tensor α [convWidth, modelDim]) :
    KDA.Input α keyDim valueDim :=
  let current := Spec.get window ⟨0, hWidth⟩
  let qPre := head.queryConv.forwardWindow hWidth (matMulSpec window head.queryWeight)
  let kPre := head.keyConv.forwardWindow hWidth (matMulSpec window head.keyWeight)
  let vPre := head.valueConv.forwardWindow hWidth (matMulSpec window head.valueWeight)
  head.activate logFloor current qPre kPre vPre

end KDA.Head

/-- Complete multi-head KDA layer with a full-rank output gate (Eq. 6). -/
structure KDA.Layer (α : Type) [Storage α]
    (modelDim heads keyDim valueDim convWidth decayRank : Nat) where
  /-- Parameters indexed by attention head. -/
  head : Fin heads → KDA.Head α modelDim keyDim valueDim convWidth decayRank
  /-- Full-rank output-gate projection. -/
  gateWeight : Tensor α [modelDim, heads, valueDim]
  /-- Per-head projection back to model width. -/
  outputWeight : Tensor α [heads, valueDim, modelDim]
  /-- Per-head RMSNorm scale applied before output gating. -/
  outputNormScale : Tensor α [heads, valueDim]

namespace KDA.Layer

variable {α : Type} [Storage α] [Context α]
variable {modelDim heads keyDim valueDim convWidth decayRank : Nat}

/-- State carried by every KDA head.

Heads run along the leading tensor axis.  A packed tensor, rather than a function
`Fin heads → KDA.State`, makes the specification state the same value that GraphSpec and the
runtime pass around.  `Spec.get state head` still gives one head's matrix.
-/
abbrev State (α : Type) [Storage α] (heads keyDim valueDim : Nat) :=
  Tensor α [heads, keyDim, valueDim]

/-- Full-rank output gates, arranged as one row per attention head.

In the published layout the gate weight has shape `(modelDim, heads, valueDim)`, so each head
multiplies the hidden vector by its own `(modelDim, valueDim)` slice and applies a sigmoid.  Its
packed result has the same layout as the GraphSpec graph.
-/
def gates (layer : KDA.Layer α modelDim heads keyDim valueDim convWidth decayRank)
    (x : Tensor α [modelDim]) :
    Tensor α [heads, valueDim] :=
  Tensor.stackLeading fun head =>
    Tensor.map Activation.Math.sigmoidSpec <|
      vecMatMulSpec x <| Tensor.stackLeading fun input =>
        Spec.get (Spec.get layer.gateWeight input) head

/-- Read every recurrent head, normalize it, apply its output gate, and project back to model
width.

It takes the already updated state and no projection or convolution parameters, so
GraphSpec reuses it as the target of the post-update subgraph.
-/
def readout (state : Tensor α [heads, keyDim, valueDim])
    (query : Tensor α [heads, keyDim])
    (gate : Tensor α [heads, valueDim])
    (outputWeight : Tensor α [heads, valueDim, modelDim])
    (outputNormScale : Tensor α [heads, valueDim]) :
    Tensor α [modelDim] :=
  let headOutput := Tensor.stackLeading fun h =>
    RMSNorm.scale (KDA.read (Spec.get state h) (Spec.get query h))
      (Spec.get outputNormScale h)
  let gated := mulSpec gate headOutput
  let projectedHeads := Tensor.stackLeading fun h =>
    vecMatMulSpec (Spec.get gated h) (Spec.get outputWeight h)
  Tensor.stackLeading fun output => Tensor.scalar <|
    (List.finRange heads).foldl (fun total head =>
      total + Tensor.getScalar (Spec.get projectedHeads head) output) 0

/-- Complete one layer step after every head's recurrence inputs have been prepared.

`step` and `stepWindow` both call it, so the list-based causal specification and the
fixed-window graph target share one copy of the recurrence, output normalization, full-rank gate,
and output projection.
-/
def stepPrepared (layer : KDA.Layer α modelDim heads keyDim valueDim convWidth decayRank)
    (state : State α heads keyDim valueDim) (x : Tensor α [modelDim])
    (prepared : Fin heads → KDA.Input α keyDim valueDim) :
    State α heads keyDim valueDim × Tensor α [modelDim] :=
  let next := Tensor.stackLeading fun h => KDA.update (Spec.get state h) (prepared h)
  let query := Tensor.stackLeading fun h => (prepared h).query
  (next, readout next query (layer.gates x) layer.outputWeight layer.outputNormScale)

/-- One token step through every KDA head and the shared output projection. -/
def step (layer : KDA.Layer α modelDim heads keyDim valueDim convWidth decayRank)
    (logFloor : α) (history : List (Tensor α [modelDim]))
    (state : State α heads keyDim valueDim) (x : Tensor α [modelDim]) :
    State α heads keyDim valueDim × Tensor α [modelDim] :=
  layer.stepPrepared state x fun head => (layer.head head).prepare logFloor x history

/-- One complete KDA layer step from a fixed newest-first model-width window. -/
def stepWindow (layer : KDA.Layer α modelDim heads keyDim valueDim convWidth decayRank)
    (hWidth : 0 < convWidth) (logFloor : α)
    (window : Tensor α [convWidth, modelDim])
    (state : State α heads keyDim valueDim) :
    State α heads keyDim valueDim × Tensor α [modelDim] :=
  let current := Spec.get window ⟨0, hWidth⟩
  layer.stepPrepared state current fun head =>
    (layer.head head).prepareWindow hWidth logFloor window

/-- Appending one row to a positive-width window prefix restores the original width. -/
theorem rollWindow_shape_eq (hWidth : 0 < convWidth) :
    ([1 + (convWidth - 1), modelDim] : Shape) =
      [convWidth, modelDim] := by
  congr 1
  omega

/-- Uncast tensor operation underlying `rollWindow`. -/
def rollWindowRaw (current : Tensor α [modelDim])
    (previous : Tensor α [convWidth, modelDim]) :
    Tensor α [1 + (convWidth - 1), modelDim] :=
  let currentRow : Tensor α [1, modelDim] :=
    Tensor.reshapeSpec current (by simp [Shape.size])
  let retained := Tensor.sliceAxisRangeSpec 0 previous 0 (convWidth - 1) (by simp)
  Tensor.concat currentRow retained

/-- Advance the fixed newest-first short-convolution window by one token.

The current token becomes row zero, the previous first `convWidth - 1` rows move back by one
position, and the oldest row is discarded.  This is the transition the executable graph refines.
-/
def rollWindow (hWidth : 0 < convWidth)
    (current : Tensor α [modelDim])
    (previous : Tensor α [convWidth, modelDim]) :
    Tensor α [convWidth, modelDim] :=
  KDA.Layer.rollWindow_shape_eq hWidth ▸ KDA.Layer.rollWindowRaw current previous

/-- Streaming KDA state: recurrent head matrices together with newest-first token history for the
causal short convolutions. -/
abbrev ScanState (α : Type) [Storage α] (modelDim heads keyDim valueDim : Nat) :=
  State α heads keyDim valueDim × List (Tensor α [modelDim])

/-- One streaming layer step, including the short-convolution history update. -/
def scanStep (layer : KDA.Layer α modelDim heads keyDim valueDim convWidth decayRank)
    (logFloor : α) (state : ScanState α modelDim heads keyDim valueDim)
    (x : Tensor α [modelDim]) :
    ScanState α modelDim heads keyDim valueDim × Tensor α [modelDim] :=
  let result := layer.step logFloor state.2 state.1 x
  ((result.1, x :: state.2), result.2)

/-- A streaming KDA step puts the current token at the front of the convolution history. -/
theorem scanStep_history
    (layer : KDA.Layer α modelDim heads keyDim valueDim convWidth decayRank)
    (logFloor : α) (state : ScanState α modelDim heads keyDim valueDim)
    (x : Tensor α [modelDim]) :
    (layer.scanStep logFloor state x).1.2 = x :: state.2 := by
  rfl

/-- Run a KDA layer from a given streaming state. -/
def run (layer : KDA.Layer α modelDim heads keyDim valueDim convWidth decayRank)
    (logFloor : α) (state : ScanState α modelDim heads keyDim valueDim)
    (tokens : List (Tensor α [modelDim])) :=
  CausalScan.run (layer.scanStep logFloor) state tokens

/-- A multi-head KDA scan emits one model-width vector per token. -/
theorem run_outputs_length
    (layer : KDA.Layer α modelDim heads keyDim valueDim convWidth decayRank)
    (logFloor : α) (state : ScanState α modelDim heads keyDim valueDim)
    (tokens : List (Tensor α [modelDim])) :
    (layer.run logFloor state tokens).2.length = tokens.length := by
  exact CausalScan.outputs_length (layer.scanStep logFloor) state tokens

/-- Chunked KDA execution preserves the recurrent matrices, convolution history, and outputs. -/
theorem run_append
    (layer : KDA.Layer α modelDim heads keyDim valueDim convWidth decayRank)
    (logFloor : α) (state : ScanState α modelDim heads keyDim valueDim)
    (left right : List (Tensor α [modelDim])) :
    layer.run logFloor state (left ++ right) =
      let first := layer.run logFloor state left
      let second := layer.run logFloor first.1 right
      (second.1, first.2 ++ second.2) :=
  CausalScan.run_append (layer.scanStep logFloor) state left right

end KDA.Layer

end KimiK3
