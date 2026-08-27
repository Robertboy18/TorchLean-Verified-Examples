/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Common
public import KimiK3.Config
public import NN.Spec.Core.TensorReductionShape
public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Normalization
public import Mathlib.Analysis.SpecialFunctions.Sigmoid
import NN.Proofs.Tensor.Basic.Algebra

/-!
# Kimi K3 sequence and depth mixing

This module contains theorem-oriented specifications of the three mechanisms that move information
through the Kimi K3 backbone:

* Kimi Delta Attention (KDA), including its short convolutions, lower-bounded retention gate,
  recurrent delta update, and full-rank output gate;
* Gated Multi-head Latent Attention (MLA), whose cache stores a shared compressed KV latent and
  whose query/key computation intentionally contains no positional encoding;
* Block Attention Residuals (AttnRes), which attend over the embedding, completed block sums, and
  the current partial block sum.

The definitions follow Eqs. 1--10 of the Kimi K3 report.  They are reference semantics, not a claim
that the released fused kernels execute these definitions directly.

Reference: Kimi Team, "Kimi K3: Open Frontier Intelligence", 2026, Sections 2.1--2.2,
https://arxiv.org/abs/2607.24653.
-/

@[expose] public section

namespace KimiK3

open Spec
open Tensor

/-! ## A reusable causal scan -/

namespace CausalScan

/-- Run a state transition over a token list, returning the final state and one output per token. -/
def run {State Input Output : Type} (step : State → Input → State × Output) :
    State → List Input → State × List Output
  | state, [] => (state, [])
  | state, token :: rest =>
      let (next, output) := step state token
      let (final, outputs) := run step next rest
      (final, output :: outputs)

/-- A causal scan emits exactly one output for every input. -/
theorem outputs_length {State Input Output : Type} (step : State → Input → State × Output)
    (state : State) (tokens : List Input) :
    (run step state tokens).2.length = tokens.length := by
  induction tokens generalizing state with
  | nil => simp [run]
  | cons token rest ih =>
      simp [run, ih]

/--
Splitting a causal scan into chunks does not change its state or outputs.  This is the semantic
contract used by chunkwise KDA implementations: chunking may change evaluation order inside a
kernel, but not the recurrence being implemented.
-/
theorem run_append {State Input Output : Type} (step : State → Input → State × Output)
    (state : State) (left right : List Input) :
    run step state (left ++ right) =
      let first := run step state left
      let second := run step first.1 right
      (second.1, first.2 ++ second.2) := by
  induction left generalizing state with
  | nil => simp [run]
  | cons token rest ih =>
      simp [run, ih]

end CausalScan

/-! ## Kimi Delta Attention -/

/-- Parameters of one causal depthwise short convolution. -/
structure ShortConv (α : Type) (width channels : Nat) where
  weight : Tensor α (.dim width (.dim channels .scalar))
  bias : Tensor α (.dim channels .scalar)

namespace ShortConv

variable {α : Type} [Context α]
variable {width channels : Nat}

/--
Apply a short convolution to a newest-first projected-token history.  Missing old tokens are zero
padded, so the definition is causal for every history length.
-/
def forward (conv : ShortConv α width channels)
    (history : List (Tensor α (.dim channels .scalar))) : Tensor α (.dim channels .scalar) :=
  let zero : Tensor α (.dim channels .scalar) := Spec.fill 0 (.dim channels .scalar)
  Tensor.dim (fun channel => Tensor.scalar <|
    (List.finRange width).foldl (fun total tap =>
      total + Tensor.getScalar (history.getD tap.val zero) channel * get2 conv.weight tap channel)
      (Tensor.getScalar conv.bias channel))

/-- Apply the same causal convolution to a full newest-first window.

The leading axis is the tap axis: row zero is the current projected token and later rows are
successively older tokens. Streaming code can maintain this fixed-size window directly, while the
list-based `forward` above remains convenient for the ideal causal specification with zero padding.
-/
def forwardWindow (conv : ShortConv α width channels) (hWidth : 0 < width)
    (window : Tensor α (.dim width (.dim channels .scalar))) :
    Tensor α (.dim channels .scalar) :=
  Tensor.addSpec
    (Tensor.reduceSum 0 (Tensor.mulSpec window conv.weight)
      (Shape.hasNonemptyAxisZeroOfPos hWidth).proof) conv.bias

end ShortConv

/-- Input to the mathematical single-head KDA recurrence in Eq. 1. -/
structure KDAStepInput (α : Type) (keyDim valueDim : Nat) where
  query : Tensor α (.dim keyDim .scalar)
  key : Tensor α (.dim keyDim .scalar)
  value : Tensor α (.dim valueDim .scalar)
  retention : Tensor α (.dim keyDim .scalar)
  writeStrength : α

namespace KDA

variable {α : Type} [Context α]
variable {keyDim valueDim : Nat}

/-- Recurrent KDA state `S ∈ R^(d_k × d_v)`. -/
abbrev State (α : Type) (keyDim valueDim : Nat) :=
  Tensor α (.dim keyDim (.dim valueDim .scalar))

/--
One single-head KDA recurrence (Eq. 1):

`S_t = (I - β_t k_t k_tᵀ) Diag(α_t) S_(t-1) + β_t k_t v_tᵀ`.

The implementation follows the factorized form used by an efficient kernel: decay the rows of the
state, read the decayed state with the key, subtract the key-aligned correction, and finally add
the rank-one write. Every operation is an ordinary TorchLean tensor operation.
-/
def update (state : State α keyDim valueDim) (input : KDAStepInput α keyDim valueDim) :
    State α keyDim valueDim :=
  let decayed := Tensor.dim fun row => Tensor.dim fun column => Tensor.scalar <|
    Tensor.getScalar input.retention row * get2 state row column
  let correction := vecMatMulSpec input.key decayed
  Tensor.addSpec
    (Tensor.subSpec decayed
      (Tensor.mulSpec
        (Spec.fill input.writeStrength (.dim keyDim (.dim valueDim .scalar)))
        (outerProductSpec input.key correction)))
    (Tensor.mulSpec
      (Spec.fill input.writeStrength (.dim keyDim (.dim valueDim .scalar)))
      (outerProductSpec input.key input.value))

/-- Coordinate form of the factorized KDA update.

The correction coordinate is the key's read from the already decayed state. This statement makes
the order of decay and delta correction explicit without duplicating the implementation.
-/
theorem update_get2 (state : State ℝ keyDim valueDim)
    (input : KDAStepInput ℝ keyDim valueDim) (row : Fin keyDim) (column : Fin valueDim) :
  let decayed := Tensor.dim fun source => Tensor.dim fun target => Tensor.scalar
    (Tensor.getScalar input.retention source * get2 state source target)
  get2 (update state input) row column =
      Tensor.getScalar input.retention row * get2 state row column -
        input.writeStrength * Tensor.getScalar input.key row *
          Tensor.getScalar (vecMatMulSpec input.key decayed) column +
        input.writeStrength * Tensor.getScalar input.key row * Tensor.getScalar input.value column := by
  simp [update, Tensor.addSpec, Tensor.subSpec, Tensor.mulSpec]
  ring

/-- Read the current output after the state update: `o_t = S_tᵀ q_t`. -/
def read (state : State α keyDim valueDim) (query : Tensor α (.dim keyDim .scalar)) :
    Tensor α (.dim valueDim .scalar) :=
  vecMatMulSpec query state

/-- One recurrence step returns the updated state and its current-token output. -/
def step (state : State α keyDim valueDim) (input : KDAStepInput α keyDim valueDim) :
    State α keyDim valueDim × Tensor α (.dim valueDim .scalar) :=
  let next := update state input
  (next, read next input.query)

/-- Sequential semantics for a projected KDA sequence. -/
def run (state : State α keyDim valueDim) (inputs : List (KDAStepInput α keyDim valueDim)) :=
  CausalScan.run step state inputs

/-- Chunk boundaries do not change the KDA recurrence or emitted outputs. -/
theorem run_append (state : State α keyDim valueDim)
    (left right : List (KDAStepInput α keyDim valueDim)) :
    run state (left ++ right) =
      let first := run state left
      let second := run first.1 right
      (second.1, first.2 ++ second.2) :=
  CausalScan.run_append step state left right

/-- Lower-bounded real retention from Eq. 5. -/
noncomputable def retentionReal (logFloor logScale logit : ℝ) : ℝ :=
  Real.exp (logFloor * Real.sigmoid (Real.exp logScale * logit))

/--
The K3 retention parameterization stays strictly above `exp(logFloor)` and at most one.  For the
paper value `logFloor = -5`, this prevents an individual channel retention from approaching zero.
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

/-- For Kimi K3, every one-step retention lies strictly between `exp(-5)` and `1`. -/
theorem paper_retention_bounds (logScale logit : ℝ) :
    Real.exp (-5) < retentionReal (-5) logScale logit ∧
      retentionReal (-5) logScale logit < 1 := by
  exact retentionReal_bounds (by norm_num) logScale logit

end KDA

/-- Projection and gating parameters for one KDA head. -/
structure KDAHead (α : Type) (modelDim keyDim valueDim convWidth decayRank : Nat) where
  queryWeight : Tensor α (.dim modelDim (.dim keyDim .scalar))
  keyWeight : Tensor α (.dim modelDim (.dim keyDim .scalar))
  valueWeight : Tensor α (.dim modelDim (.dim valueDim .scalar))
  queryConv : ShortConv α convWidth keyDim
  keyConv : ShortConv α convWidth keyDim
  valueConv : ShortConv α convWidth valueDim
  betaWeight : Tensor α (.dim modelDim .scalar)
  decayDown : Tensor α (.dim modelDim (.dim decayRank .scalar))
  decayUp : Tensor α (.dim decayRank (.dim keyDim .scalar))
  decayBias : Tensor α (.dim keyDim .scalar)
  decayLogScale : α

namespace KDAHead

variable {α : Type} [Context α]
variable {modelDim keyDim valueDim convWidth decayRank : Nat}

/-- Project a newest-first input history through one matrix. -/
def projectedHistory (weight : Tensor α (.dim modelDim (.dim keyDim .scalar)))
    (history : List (Tensor α (.dim modelDim .scalar))) :
    List (Tensor α (.dim keyDim .scalar)) :=
  history.map (fun token => vecMatMulSpec token weight)

/-- Construct all single-head recurrence inputs from a causal token history. -/
def prepare (head : KDAHead α modelDim keyDim valueDim convWidth decayRank)
    (logFloor : α) (current : Tensor α (.dim modelDim .scalar))
    (history : List (Tensor α (.dim modelDim .scalar))) : KDAStepInput α keyDim valueDim :=
  let withCurrent := current :: history
  let qPre := head.queryConv.forward (projectedHistory head.queryWeight withCurrent)
  let kPre := head.keyConv.forward (projectedHistory head.keyWeight withCurrent)
  let vHistory := history.map (fun token => vecMatMulSpec token head.valueWeight)
  let vPre := head.valueConv.forward (vecMatMulSpec current head.valueWeight :: vHistory)
  let query := Normalize.regularizedL2 (Tensor.mapSpec Activation.Math.swishSpec qPre)
    Numbers.epsilon
  let key := Normalize.regularizedL2 (Tensor.mapSpec Activation.Math.swishSpec kPre)
    Numbers.epsilon
  let value := Tensor.mapSpec Activation.Math.swishSpec vPre
  let decayLogit :=
    Tensor.addSpec (vecMatMulSpec (vecMatMulSpec current head.decayDown) head.decayUp)
      head.decayBias
  let scale := MathFunctions.exp head.decayLogScale
  let retention := Tensor.mapSpec
    (fun z => MathFunctions.exp (logFloor * Activation.Math.sigmoidSpec (scale * z))) decayLogit
  let beta := Activation.Math.sigmoidSpec (Tensor.dotSpec current head.betaWeight)
  { query, key, value, retention, writeStrength := beta }

/-- Construct one head's recurrence input from a fixed newest-first model-width window.

This is the tensor ABI used by the executable graph. Projection is performed for the entire
window in one matrix multiplication before the three depthwise convolutions. The normalization
uses the `10⁻⁶` stabilizer implemented by the released FlashKDA reference and CUDA kernel.
-/
def prepareWindow (head : KDAHead α modelDim keyDim valueDim convWidth decayRank)
    (hWidth : 0 < convWidth) (logFloor : α)
    (window : Tensor α (.dim convWidth (.dim modelDim .scalar))) :
    KDAStepInput α keyDim valueDim :=
  let current := Spec.get window ⟨0, hWidth⟩
  let qPre := head.queryConv.forwardWindow hWidth (matMulSpec window head.queryWeight)
  let kPre := head.keyConv.forwardWindow hWidth (matMulSpec window head.keyWeight)
  let vPre := head.valueConv.forwardWindow hWidth (matMulSpec window head.valueWeight)
  let query := Normalize.regularizedL2 (Tensor.mapSpec Activation.Math.swishSpec qPre)
    Numbers.epsilon
  let key := Normalize.regularizedL2 (Tensor.mapSpec Activation.Math.swishSpec kPre)
    Numbers.epsilon
  let value := Tensor.mapSpec Activation.Math.swishSpec vPre
  let decayLogit :=
    Tensor.addSpec (vecMatMulSpec (vecMatMulSpec current head.decayDown) head.decayUp)
      head.decayBias
  let scale := MathFunctions.exp head.decayLogScale
  let retention := Tensor.mapSpec
    (fun z => MathFunctions.exp (logFloor * Activation.Math.sigmoidSpec (scale * z))) decayLogit
  let beta := Activation.Math.sigmoidSpec (Tensor.dotSpec current head.betaWeight)
  { query, key, value, retention, writeStrength := beta }

end KDAHead

/-- Complete multi-head KDA layer with a full-rank output gate (Eq. 6). -/
structure KDALayer (α : Type)
    (modelDim heads keyDim valueDim convWidth decayRank : Nat) where
  head : Fin heads → KDAHead α modelDim keyDim valueDim convWidth decayRank
  gateWeight : Tensor α (.dim modelDim (.dim heads (.dim valueDim .scalar)))
  outputWeight : Tensor α (.dim heads (.dim valueDim (.dim modelDim .scalar)))
  outputNormScale : Tensor α (.dim heads (.dim valueDim .scalar))

namespace KDALayer

variable {α : Type} [Context α]
variable {modelDim heads keyDim valueDim convWidth decayRank : Nat}

/-- State carried by every KDA head.

The leading tensor axis is the head axis.  Using the packed representation here, rather than a
function `Fin heads → KDA.State`, makes the mathematical state identical to the value passed
through GraphSpec and the runtime.  Individual head equations remain available through
`Spec.get state head`.
-/
abbrev State (α : Type) (heads keyDim valueDim : Nat) :=
  Tensor α (.dim heads (.dim keyDim (.dim valueDim .scalar)))

/-- Full-rank output gates, arranged as one row per attention head.

The published gate weight has shape `(modelDim, heads, valueDim)`.  For each head we select its
`(modelDim, valueDim)` matrix, multiply the current hidden vector by that matrix, and apply the
sigmoid coordinatewise.  Keeping the result packed gives the specification the same layout as the
GraphSpec graph and avoids a second scalar-loop implementation of the gate.
-/
def gates (layer : KDALayer α modelDim heads keyDim valueDim convWidth decayRank)
    (x : Tensor α (.dim modelDim .scalar)) :
    Tensor α (.dim heads (.dim valueDim .scalar)) :=
  Tensor.dim fun head =>
    mapSpec Activation.Math.sigmoidSpec <|
      vecMatMulSpec x <| Tensor.dim fun input =>
        Spec.get (Spec.get layer.gateWeight input) head

/-- Coordinate view of the packed full-rank output gate. -/
def gateAt (layer : KDALayer α modelDim heads keyDim valueDim convWidth decayRank)
    (x : Tensor α (.dim modelDim .scalar)) (head : Fin heads) (channel : Fin valueDim) : α :=
  Tensor.getScalar (Spec.get (layer.gates x) head) channel

/-- Read every recurrent head, apply its output gate, and project back to model width.

The recurrence graph computes the new state once and then passes it to this function.  Keeping the
readout independent of projection and convolution parameters also gives GraphSpec a compact,
reusable semantic target for the shared post-update subgraph.
-/
def readout (state : Tensor α (.dim heads (.dim keyDim (.dim valueDim .scalar))))
    (query : Tensor α (.dim heads (.dim keyDim .scalar)))
    (gate : Tensor α (.dim heads (.dim valueDim .scalar)))
    (outputWeight : Tensor α (.dim heads (.dim valueDim (.dim modelDim .scalar))))
    (outputNormScale : Tensor α (.dim heads (.dim valueDim .scalar))) :
    Tensor α (.dim modelDim .scalar) :=
  let headOutput := Tensor.dim fun h =>
    RMSNorm.scale (KDA.read (Spec.get state h) (Spec.get query h))
      (Spec.get outputNormScale h)
  let gated := mulSpec gate headOutput
  let projectedHeads := Tensor.dim fun h =>
    vecMatMulSpec (Spec.get gated h) (Spec.get outputWeight h)
  Tensor.dim fun output => Tensor.scalar <|
    (List.finRange heads).foldl (fun total head =>
      total + Tensor.getScalar (Spec.get projectedHeads head) output) 0

/-- Complete one layer step after every head's recurrence inputs have been prepared.

Both the list-based causal specification and the fixed-window executable graph use this function.
Keeping the recurrence, output normalization, full-rank gate, and output projection here prevents
the two front ends from acquiring subtly different layer equations.
-/
def stepPrepared (layer : KDALayer α modelDim heads keyDim valueDim convWidth decayRank)
    (state : State α heads keyDim valueDim) (x : Tensor α (.dim modelDim .scalar))
    (prepared : Fin heads → KDAStepInput α keyDim valueDim) :
    State α heads keyDim valueDim × Tensor α (.dim modelDim .scalar) :=
  let next := Tensor.dim fun h => KDA.update (Spec.get state h) (prepared h)
  let query := Tensor.dim fun h => (prepared h).query
  (next, readout next query (layer.gates x) layer.outputWeight layer.outputNormScale)

/-- One token step through every KDA head and the shared output projection. -/
def step (layer : KDALayer α modelDim heads keyDim valueDim convWidth decayRank)
    (logFloor : α) (history : List (Tensor α (.dim modelDim .scalar)))
    (state : State α heads keyDim valueDim) (x : Tensor α (.dim modelDim .scalar)) :
    State α heads keyDim valueDim × Tensor α (.dim modelDim .scalar) :=
  layer.stepPrepared state x fun head => (layer.head head).prepare logFloor x history

/-- One complete KDA layer step from a fixed newest-first model-width window. -/
def stepWindow (layer : KDALayer α modelDim heads keyDim valueDim convWidth decayRank)
    (hWidth : 0 < convWidth) (logFloor : α)
    (window : Tensor α (.dim convWidth (.dim modelDim .scalar)))
    (state : State α heads keyDim valueDim) :
    State α heads keyDim valueDim × Tensor α (.dim modelDim .scalar) :=
  let current := Spec.get window ⟨0, hWidth⟩
  layer.stepPrepared state current fun head =>
    (layer.head head).prepareWindow hWidth logFloor window

/-- Appending one row to a positive-width window prefix restores the original width. -/
theorem rollWindowShape (hWidth : 0 < convWidth) :
    Shape.dim (1 + (convWidth - 1)) (.dim modelDim .scalar) =
      .dim convWidth (.dim modelDim .scalar) := by
  congr 1
  omega

/-- The uncast tensor operation underlying `rollWindow`. -/
def rollWindowRaw (current : Tensor α (.dim modelDim .scalar))
    (previous : Tensor α (.dim convWidth (.dim modelDim .scalar))) :
    Tensor α (.dim (1 + (convWidth - 1)) (.dim modelDim .scalar)) :=
  let currentRow : Tensor α (.dim 1 (.dim modelDim .scalar)) :=
    Tensor.reshapeSpec current (by simp [Shape.size])
  let retained := Tensor.sliceAxisRangeSpec 0 previous 0 (convWidth - 1) (by simp)
  Tensor.concatAxisSpec .scalar currentRow retained

/-- Advance the fixed newest-first short-convolution window by one token.

The current token becomes row zero, the previous first `convWidth - 1` rows move back by one
position, and the oldest row is discarded.  Keeping this operation at the specification level
gives the executable graph a precise state transition to refine.
-/
def rollWindow (hWidth : 0 < convWidth)
    (current : Tensor α (.dim modelDim .scalar))
    (previous : Tensor α (.dim convWidth (.dim modelDim .scalar))) :
    Tensor α (.dim convWidth (.dim modelDim .scalar)) :=
  KDALayer.rollWindowShape hWidth ▸ KDALayer.rollWindowRaw current previous

/-- Streaming KDA state: recurrent head matrices together with newest-first token history for the
causal short convolutions. -/
abbrev ScanState (α : Type) (modelDim heads keyDim valueDim : Nat) :=
  State α heads keyDim valueDim × List (Tensor α (.dim modelDim .scalar))

/-- One streaming layer step, including the short-convolution history update. -/
def scanStep (layer : KDALayer α modelDim heads keyDim valueDim convWidth decayRank)
    (logFloor : α) (state : ScanState α modelDim heads keyDim valueDim)
    (x : Tensor α (.dim modelDim .scalar)) :
    ScanState α modelDim heads keyDim valueDim × Tensor α (.dim modelDim .scalar) :=
  let result := layer.step logFloor state.2 state.1 x
  ((result.1, x :: state.2), result.2)

/-- Run a KDA layer from an explicit streaming state. -/
def run (layer : KDALayer α modelDim heads keyDim valueDim convWidth decayRank)
    (logFloor : α) (state : ScanState α modelDim heads keyDim valueDim)
    (tokens : List (Tensor α (.dim modelDim .scalar))) :=
  CausalScan.run (layer.scanStep logFloor) state tokens

/-- Chunked KDA execution preserves the recurrent matrices, convolution history, and outputs. -/
theorem run_append
    (layer : KDALayer α modelDim heads keyDim valueDim convWidth decayRank)
    (logFloor : α) (state : ScanState α modelDim heads keyDim valueDim)
    (left right : List (Tensor α (.dim modelDim .scalar))) :
    layer.run logFloor state (left ++ right) =
      let first := layer.run logFloor state left
      let second := layer.run logFloor first.1 right
      (second.1, first.2 ++ second.2) :=
  CausalScan.run_append (layer.scanStep logFloor) state left right

end KDALayer

/-! ## Gated Multi-head Latent Attention -/

/-- Per-head up-projections from MLA's compressed query and KV latents. -/
structure MLAHead (α : Type)
    (queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim : Nat) where
  queryContentUp : Tensor α (.dim queryLatentDim (.dim contentKeyDim .scalar))
  querySharedUp : Tensor α (.dim queryLatentDim (.dim sharedKeyDim .scalar))
  keyUp : Tensor α (.dim kvLatentDim (.dim contentKeyDim .scalar))
  valueUp : Tensor α (.dim kvLatentDim (.dim valueDim .scalar))

/-- One cached MLA token: a normalized KV latent and a shared, unrotated key component. -/
structure MLACacheEntry (α : Type) (kvLatentDim sharedKeyDim : Nat) where
  latent : Tensor α (.dim kvLatentDim .scalar)
  sharedKey : Tensor α (.dim sharedKeyDim .scalar)

/-- Gated MLA parameters.  No positional-encoding operation appears in this structure. -/
structure GatedMLA (α : Type)
    (modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim : Nat) where
  queryDown : Tensor α (.dim modelDim (.dim queryLatentDim .scalar))
  queryNormScale : Tensor α (.dim queryLatentDim .scalar)
  kvDown : Tensor α (.dim modelDim (.dim kvLatentDim .scalar))
  kvNormScale : Tensor α (.dim kvLatentDim .scalar)
  sharedKeyDown : Tensor α (.dim modelDim (.dim sharedKeyDim .scalar))
  head : Fin heads →
    MLAHead α queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim
  gateWeight : Tensor α (.dim modelDim (.dim heads (.dim valueDim .scalar)))
  outputWeight : Tensor α (.dim heads (.dim valueDim (.dim modelDim .scalar)))

namespace GatedMLA

variable {α : Type} [Context α]
variable {modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim : Nat}

/-- The cache stores the two shared representations needed to reconstruct every head. -/
abbrev Cache (α : Type) (kvLatentDim sharedKeyDim : Nat) :=
  List (MLACacheEntry α kvLatentDim sharedKeyDim)

/-- Unnormalized global content attention for one head over a nonempty latent cache. -/
def attendHead
    (head : MLAHead α queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (queryLatent : Tensor α (.dim queryLatentDim .scalar))
    (cache : Cache α kvLatentDim sharedKeyDim) : Tensor α (.dim valueDim .scalar) :=
  let queryContent := vecMatMulSpec queryLatent head.queryContentUp
  let queryShared := vecMatMulSpec queryLatent head.querySharedUp
  let score := fun entry =>
    MathFunctions.exp ((Tensor.dotSpec queryContent (vecMatMulSpec entry.latent head.keyUp) +
      Tensor.dotSpec queryShared entry.sharedKey) /
      MathFunctions.sqrt (contentKeyDim + sharedKeyDim : α))
  let denominator := cache.foldl (fun total entry => total + score entry) 0
  cache.foldl (fun output entry =>
    let weight := score entry / denominator
    output + Tensor.mapSpec (fun value => weight * value)
      (vecMatMulSpec entry.latent head.valueUp))
    (Spec.fill 0 (.dim valueDim .scalar))

/-- Evaluate one MLA head from a fixed-length tensor cache.

The streaming interface above uses a list because its cache grows by one entry at each token.  A
compiled graph, however, has a statically known context length.  This definition presents the same
latent-attention calculation in that fixed representation: cache rows are tokens, the first cache
stores normalized KV latents, and the second stores the shared NoPE key coordinates.

The caller supplies the score scale explicitly.  For K3 this is
`1 / sqrt (contentKeyDim + sharedKeyDim)`; keeping it explicit lets the graph record the numerical
value used by a checkpoint rather than silently recomputing it in a backend.
-/
def attendHeadFixed (tokens : Nat)
    (head : MLAHead α queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (queryLatent : Tensor α (.dim queryLatentDim .scalar))
    (latentCache : Tensor α (.dim tokens (.dim kvLatentDim .scalar)))
    (sharedKeyCache : Tensor α (.dim tokens (.dim sharedKeyDim .scalar)))
    (scoreScale : α) : Tensor α (.dim valueDim .scalar) :=
  let queryContent := vecMatMulSpec queryLatent head.queryContentUp
  let queryShared := vecMatMulSpec queryLatent head.querySharedUp
  let keys := matMulSpec latentCache head.keyUp
  let values := matMulSpec latentCache head.valueUp
  let contentScores := vecMatMulSpec queryContent (Tensor.swapAdjacentAxes keys 0)
  let sharedScores := vecMatMulSpec queryShared (Tensor.swapAdjacentAxes sharedKeyCache 0)
  let scores := Tensor.mulSpec (Spec.fill scoreScale (.dim tokens .scalar))
    (Tensor.addSpec contentScores sharedScores)
  vecMatMulSpec (Activation.softmaxSpec 0 scores) values

/-- Pack the content-query up-projections of all heads along a leading head axis. -/
def queryContentUpPacked
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim) :
    Tensor α (.dim heads (.dim queryLatentDim (.dim contentKeyDim .scalar))) :=
  Tensor.dim fun head => (layer.head head).queryContentUp

/-- Pack the shared-query up-projections of all heads along a leading head axis. -/
def querySharedUpPacked
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim) :
    Tensor α (.dim heads (.dim queryLatentDim (.dim sharedKeyDim .scalar))) :=
  Tensor.dim fun head => (layer.head head).querySharedUp

/-- Pack the content-key up-projections of all heads along a leading head axis. -/
def keyUpPacked
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim) :
    Tensor α (.dim heads (.dim kvLatentDim (.dim contentKeyDim .scalar))) :=
  Tensor.dim fun head => (layer.head head).keyUp

/-- Pack the value up-projections of all heads along a leading head axis. -/
def valueUpPacked
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim) :
    Tensor α (.dim heads (.dim kvLatentDim (.dim valueDim .scalar))) :=
  Tensor.dim fun head => (layer.head head).valueUp

/-- One complete Gated MLA step at a fixed context length.

Unlike `step`, whose list cache is convenient for inductive causal proofs, this operation uses
rank-two cache tensors and performs every head in one batched calculation.  It is the mathematical
interface used by the typed executable graph.  The result contains the two caches after appending
the current token and the projected model output.
-/
def stepFixed (pastTokens : Nat)
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (pastLatentCache : Tensor α (.dim pastTokens (.dim kvLatentDim .scalar)))
    (pastSharedKeyCache : Tensor α (.dim pastTokens (.dim sharedKeyDim .scalar)))
    (x : Tensor α (.dim modelDim .scalar)) (scoreScale : α) :
    Tensor α (.dim (pastTokens + 1) (.dim kvLatentDim .scalar)) ×
      Tensor α (.dim (pastTokens + 1) (.dim sharedKeyDim .scalar)) ×
      Tensor α (.dim modelDim .scalar) :=
  let queryProjected := vecMatMulSpec x layer.queryDown
  let queryLatent := RMSNorm.scale queryProjected layer.queryNormScale
  let kvProjected := vecMatMulSpec x layer.kvDown
  let currentKV := RMSNorm.scale kvProjected layer.kvNormScale
  let currentShared := vecMatMulSpec x layer.sharedKeyDown
  let currentKVRow : Tensor α (.dim 1 (.dim kvLatentDim .scalar)) :=
    Tensor.reshapeSpec currentKV (by simp [Shape.size])
  let currentSharedRow : Tensor α (.dim 1 (.dim sharedKeyDim .scalar)) :=
    Tensor.reshapeSpec currentShared (by simp [Shape.size])
  let latentCache := Tensor.concatAxisSpec .scalar pastLatentCache currentKVRow
  let sharedKeyCache := Tensor.concatAxisSpec .scalar pastSharedKeyCache currentSharedRow
  let queryBatchSource : Tensor α (.dim 1 (.dim 1 (.dim queryLatentDim .scalar))) :=
    Tensor.reshapeSpec queryLatent (by simp [Shape.size])
  let queryBatch : Tensor α (.dim heads (.dim 1 (.dim queryLatentDim .scalar))) :=
    Tensor.broadcastTo
      (Shape.CanBroadcastTo.dim_1_to_n
        (Shape.CanBroadcastTo.dim_eq
          (Shape.CanBroadcastTo.dim_eq
            (Shape.CanBroadcastTo.scalarTo .scalar)))) queryBatchSource
  let latentBatchSource :
      Tensor α (.dim 1 (.dim (pastTokens + 1) (.dim kvLatentDim .scalar))) :=
    Tensor.reshapeSpec latentCache (by simp [Shape.size])
  let latentBatch :
      Tensor α (.dim heads (.dim (pastTokens + 1) (.dim kvLatentDim .scalar))) :=
    Tensor.broadcastTo
      (Shape.CanBroadcastTo.dim_1_to_n
        (Shape.CanBroadcastTo.dim_eq
          (Shape.CanBroadcastTo.dim_eq
            (Shape.CanBroadcastTo.scalarTo .scalar)))) latentBatchSource
  let sharedBatchSource :
      Tensor α (.dim 1 (.dim (pastTokens + 1) (.dim sharedKeyDim .scalar))) :=
    Tensor.reshapeSpec sharedKeyCache (by simp [Shape.size])
  let sharedBatch :
      Tensor α (.dim heads (.dim (pastTokens + 1) (.dim sharedKeyDim .scalar))) :=
    Tensor.broadcastTo
      (Shape.CanBroadcastTo.dim_1_to_n
        (Shape.CanBroadcastTo.dim_eq
          (Shape.CanBroadcastTo.dim_eq
            (Shape.CanBroadcastTo.scalarTo .scalar)))) sharedBatchSource
  let sameBatch := Shape.CanBroadcastTo.refl [heads]
  let queryContent := Tensor.matmulSpec sameBatch sameBatch queryBatch layer.queryContentUpPacked
  let queryShared := Tensor.matmulSpec sameBatch sameBatch queryBatch layer.querySharedUpPacked
  let keys := Tensor.matmulSpec sameBatch sameBatch latentBatch layer.keyUpPacked
  let values := Tensor.matmulSpec sameBatch sameBatch latentBatch layer.valueUpPacked
  let contentScores := Tensor.matmulSpec sameBatch sameBatch queryContent
    (Tensor.swapAdjacentAxes keys 1)
  let sharedScores :=
    Tensor.matmulSpec sameBatch sameBatch queryShared (Tensor.swapAdjacentAxes sharedBatch 1)
  let scoreShape := .dim heads (.dim 1 (.dim (pastTokens + 1) .scalar))
  let scores := Tensor.mulSpec (Spec.fill scoreScale scoreShape)
    (Tensor.addSpec contentScores sharedScores)
  let headOutput3 := Tensor.matmulSpec sameBatch sameBatch
    (Activation.softmaxSpec 2 scores) values
  let headOutput : Tensor α (.dim heads (.dim valueDim .scalar)) :=
    Tensor.reshapeSpec headOutput3 (by simp [Shape.size])
  let gateMatrix : Tensor α (.dim modelDim (.dim (heads * valueDim) .scalar)) :=
    Tensor.reshapeSpec layer.gateWeight (by simp [Shape.size])
  let gateFlat := vecMatMulSpec x gateMatrix
  let gate : Tensor α (.dim heads (.dim valueDim .scalar)) :=
    Activation.sigmoidSpec (Tensor.reshapeSpec gateFlat (by simp [Shape.size]))
  let gatedHeads := Tensor.mulSpec gate headOutput
  let gatedFlat : Tensor α (.dim (heads * valueDim) .scalar) :=
    Tensor.reshapeSpec gatedHeads (by simp [Shape.size])
  let outputMatrix : Tensor α (.dim (heads * valueDim) (.dim modelDim .scalar)) :=
    Tensor.reshapeSpec layer.outputWeight (by simp [Shape.size, Nat.mul_assoc])
  (latentCache, sharedKeyCache, vecMatMulSpec gatedFlat outputMatrix)

/-- One causal MLA token step.  The current latent is inserted before attention is evaluated. -/
def step
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (cache : Cache α kvLatentDim sharedKeyDim) (x : Tensor α (.dim modelDim .scalar)) :
    Cache α kvLatentDim sharedKeyDim × Tensor α (.dim modelDim .scalar) :=
  let queryLatent := RMSNorm.scale (vecMatMulSpec x layer.queryDown) layer.queryNormScale
  let currentKV := RMSNorm.scale (vecMatMulSpec x layer.kvDown) layer.kvNormScale
  let current : MLACacheEntry α kvLatentDim sharedKeyDim :=
    { latent := currentKV, sharedKey := vecMatMulSpec x layer.sharedKeyDown }
  let cache' := cache ++ [current]
  let headOutput := fun h => attendHead (layer.head h) queryLatent cache'
  let gate := fun h channel => Activation.Math.sigmoidSpec <|
    (List.finRange modelDim).foldl (fun total input =>
      total + Tensor.getScalar x input * get2 (Spec.get layer.gateWeight input) h channel) 0
  let output := Tensor.dim (fun feature => Tensor.scalar <|
    (List.finRange heads).foldl (fun acrossHeads h =>
      (List.finRange valueDim).foldl (fun total channel =>
        total + gate h channel * Tensor.getScalar (headOutput h) channel *
          get2 (Spec.get layer.outputWeight h) channel feature) acrossHeads) 0)
  (cache', output)

/-- MLA's cache grows by exactly one compressed latent at every token. -/
theorem step_cache_length
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (cache : Cache α kvLatentDim sharedKeyDim) (x : Tensor α (.dim modelDim .scalar)) :
    (layer.step cache x).1.length = cache.length + 1 := by
  simp [step]

/-- Chunking a Gated MLA causal pass preserves its cache and outputs. -/
theorem run_append
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (cache : Cache α kvLatentDim sharedKeyDim)
    (left right : List (Tensor α (.dim modelDim .scalar))) :
    CausalScan.run layer.step cache (left ++ right) =
      let first := CausalScan.run layer.step cache left
      let second := CausalScan.run layer.step first.1 right
      (second.1, first.2 ++ second.2) :=
  CausalScan.run_append layer.step cache left right

end GatedMLA

/-! ## Block Attention Residuals -/

namespace AttnRes

variable {α : Type} [Context α]
variable {modelDim : Nat}

/-- Positive softmax kernel `exp(qᵀ RMSNorm(k))` from Eq. 9. -/
def kernel (query key : Tensor α (.dim modelDim .scalar)) : α :=
  MathFunctions.exp (Tensor.dotSpec query (RMSNorm.unit key))

/--
Attention over depth representations. Callers include the embedding, so `sources` is nonempty.
-/
def normalizer (query : Tensor α (.dim modelDim .scalar))
    (sources : List (Tensor α (.dim modelDim .scalar))) : α :=
  sources.foldl (fun total source => total + kernel query source) 0

/-- The depth-attention normalizer is positive whenever at least one source is present. Hence the
real-valued specification cannot divide by zero during AttnRes retrieval. -/
theorem normalizer_pos_of_ne_nil (query : Tensor ℝ (.dim modelDim .scalar))
    (sources : List (Tensor ℝ (.dim modelDim .scalar))) (hSources : sources ≠ []) :
    0 < normalizer query sources := by
  cases sources with
  | nil => exact (hSources rfl).elim
  | cons first rest =>
      simp only [normalizer, List.foldl_cons]
      clear hSources
      have hFirst : 0 < (0 : ℝ) + kernel query first := by
        change 0 < 0 + Real.exp _
        positivity
      generalize (0 : ℝ) + kernel query first = accumulator at hFirst ⊢
      induction rest generalizing accumulator with
      | nil => simpa using hFirst
      | cons source rest ih =>
          simp only [List.foldl_cons]
          apply ih
          exact add_pos_of_pos_of_nonneg hFirst (Real.exp_pos _).le

/--
Attention over depth representations. Callers include the embedding, so `sources` is nonempty.
-/
def attend (query : Tensor α (.dim modelDim .scalar))
    (sources : List (Tensor α (.dim modelDim .scalar))) : Tensor α (.dim modelDim .scalar) :=
  let denominator := normalizer query sources
  sources.foldl (fun output source =>
    let weight := kernel query source / denominator
    output + Tensor.mapSpec (fun value => weight * value) source)
    (Spec.fill 0 (.dim modelDim .scalar))

/-- Fixed-shape form of depth attention used at graph and backend boundaries.

The first axis enumerates the visible depth representations. Rows are RMS-normalized, scored
against the query, exponentiated, and normalized by their total mass. The resulting weights form a
weighted vector--matrix product with the unnormalized sources, exactly as in Eq. 9--10.
-/
def attendPacked {sources : Nat} (hModel : 0 < modelDim)
    (query : Tensor α (.dim modelDim .scalar))
    (values : Tensor α (.dim sources (.dim modelDim .scalar))) :
    Tensor α (.dim modelDim .scalar) :=
  let unitScale := Spec.fill 1 (.dim modelDim .scalar)
  let normalized := Tensor.dim fun row =>
    RMSNorm.scalePositive hModel (Spec.get values row) unitScale
  let scores := vecMatMulSpec query (Tensor.swapAdjacentAxes normalized 0)
  let kernels := Tensor.expSpec scores
  let denominator := Tensor.sumSpec kernels
  let weights := Tensor.mapSpec (fun weight => weight / denominator) kernels
  vecMatMulSpec weights values

/-- State retained by Block AttnRes while traversing layers. -/
structure BlockState (α : Type) (modelDim : Nat) where
  embedding : Tensor α (.dim modelDim .scalar)
  completedBlocks : List (Tensor α (.dim modelDim .scalar))
  partialBlock : Tensor α (.dim modelDim .scalar)
  partialSize : Nat

namespace BlockState

/-- A valid partial AttnRes block has not yet reached its block boundary. -/
def WF (state : BlockState α modelDim) (blockSize : Nat) : Prop :=
  state.partialSize < blockSize

/-- Sources in Eq. 10: embedding, completed block sums, and an optional current partial sum. -/
def sources (state : BlockState α modelDim) : List (Tensor α (.dim modelDim .scalar)) :=
  state.embedding :: state.completedBlocks ++
    if state.partialSize = 0 then [] else [state.partialBlock]

/-- Outputs of the AttnRes blocks completed so far. A nonempty partial block is included as the
current final block; the token embedding is not a block output. -/
def blockOutputs (state : BlockState α modelDim) : List (Tensor α (.dim modelDim .scalar)) :=
  state.completedBlocks ++ if state.partialSize = 0 then [] else [state.partialBlock]

omit [Context α] in
/-- The number of visible block outputs is the number of completed blocks, plus the current partial
block when at least one layer has contributed to it. -/
theorem blockOutputs_length (state : BlockState α modelDim) :
    state.blockOutputs.length = state.completedBlocks.length + if state.partialSize = 0 then 0 else 1 := by
  by_cases h : state.partialSize = 0 <;> simp [blockOutputs, h]

/-- Every valid depth state has a positive real-valued attention normalizer because the embedding
is always retained. -/
theorem normalizer_pos (state : BlockState ℝ modelDim)
    (query : Tensor ℝ (.dim modelDim .scalar)) :
    0 < AttnRes.normalizer query state.sources := by
  apply normalizer_pos_of_ne_nil
  simp [sources]

/-- Retrieve the input to a layer using its learned depth pseudo-query. -/
def retrieve (state : BlockState α modelDim) (query : Tensor α (.dim modelDim .scalar)) :
    Tensor α (.dim modelDim .scalar) :=
  attend query state.sources

/--
Depth representations visible after the sequence mixer but before the channel mixer. The current
sequence output contributes to the partial block without advancing the decoder-layer counter.
-/
def sourcesAfterSequence (state : BlockState α modelDim)
    (sequenceOutput : Tensor α (.dim modelDim .scalar)) :
    List (Tensor α (.dim modelDim .scalar)) :=
  state.embedding :: state.completedBlocks ++ [state.partialBlock + sequenceOutput]

/-- Retrieve the channel-mixer input after adding the current layer's sequence output. -/
def retrieveAfterSequence (state : BlockState α modelDim)
    (sequenceOutput query : Tensor α (.dim modelDim .scalar)) :
    Tensor α (.dim modelDim .scalar) :=
  attend query (state.sourcesAfterSequence sequenceOutput)

/--
Commit one complete decoder layer. Attention and channel outputs belong to the same AttnRes block,
so `partialSize` advances once after both submodules have run.
-/
def finishLayer (state : BlockState α modelDim) (blockSize : Nat)
    (sequenceOutput channelOutput : Tensor α (.dim modelDim .scalar)) : BlockState α modelDim :=
  let partialSum := state.partialBlock + sequenceOutput + channelOutput
  if state.partialSize + 1 = blockSize then
    { state with
      completedBlocks := state.completedBlocks ++ [partialSum]
      partialBlock := Spec.fill 0 (.dim modelDim .scalar)
      partialSize := 0 }
  else
    { state with partialBlock := partialSum, partialSize := state.partialSize + 1 }

/-- Committing a full block adds exactly one retained block representation. -/
private theorem finishLayer_completed_length (state : BlockState α modelDim) (blockSize : Nat)
    (sequenceOutput channelOutput : Tensor α (.dim modelDim .scalar))
    (hFull : state.partialSize + 1 = blockSize) :
    (state.finishLayer blockSize sequenceOutput channelOutput).completedBlocks.length =
      state.completedBlocks.length + 1 := by
  simp [finishLayer, hFull]

/-- A non-boundary decoder layer leaves the completed-block list unchanged. -/
private theorem finishLayer_completed_length_of_ne (state : BlockState α modelDim) (blockSize : Nat)
    (sequenceOutput channelOutput : Tensor α (.dim modelDim .scalar))
    (hNotFull : state.partialSize + 1 ≠ blockSize) :
    (state.finishLayer blockSize sequenceOutput channelOutput).completedBlocks.length =
      state.completedBlocks.length := by
  simp [finishLayer, hNotFull]

/-- Every completed decoder layer advances the current block position exactly once. -/
private theorem finishLayer_partialSize (state : BlockState α modelDim) (blockSize : Nat)
    (sequenceOutput channelOutput : Tensor α (.dim modelDim .scalar)) :
    (state.finishLayer blockSize sequenceOutput channelOutput).partialSize =
      if state.partialSize + 1 = blockSize then 0 else state.partialSize + 1 := by
  by_cases hFull : state.partialSize + 1 = blockSize
  · simp [finishLayer, hFull]
  · simp [finishLayer, hFull]

/-- Committing a decoder layer preserves the block-position invariant. This rules out a malformed
state silently growing past its declared AttnRes block size. -/
theorem finishLayer_wf (state : BlockState α modelDim) (blockSize : Nat)
    (sequenceOutput channelOutput : Tensor α (.dim modelDim .scalar))
    (hState : state.WF blockSize) :
    (state.finishLayer blockSize sequenceOutput channelOutput).WF blockSize := by
  rw [WF] at hState
  rw [WF, finishLayer_partialSize]
  split
  · omega
  · omega

end BlockState

end AttnRes

end KimiK3
