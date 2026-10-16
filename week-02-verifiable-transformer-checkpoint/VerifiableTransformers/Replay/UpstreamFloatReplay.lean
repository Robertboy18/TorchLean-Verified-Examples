/-
Lean Float replay of the upstream-code checkpoint.

This executable recomputes the small GPT's forward pass from the generated Float arrays and
checks the projected candidate decision on all 256 quote/bracket prompts. It also compares its
candidate logits with the PyTorch score trace, checks BandNorm's empty-support fallback, and
reruns the two extracted circuits with upstream's zero ablation. None of this is a theorem
about floating-point arithmetic or the exporter's correctness.

The replay follows Neel Somani's `scripts/small/train.py`, `scripts/small/extract.py`, and
`scripts/small/extract_weights.py` at the pinned upstream commit: row-major linear maps,
Signed-L1-BandNorm, one-head causal sparsemax attention, LeakyReLU, the untied `lm_head`,
and `controlled_forward` for circuits.

Two differences from PyTorch: Lean `Float` is binary64 while the checkpoint ran in float32,
and the replay's sparsemax scans only the `pos + 1` visible scores, where PyTorch scans all
six with the hidden ones set to `-1e4` (those never enter the support).

The rational SMT contract is in `Spec/UpstreamSmallGPT.lean`.
-/

import VerifiableTransformers.Certificate.FiniteEval
import VerifiableTransformers.Generated.UpstreamCheckpointPayload
import VerifiableTransformers.Generated.UpstreamEvalTrace
import VerifiableTransformers.Generated.UpstreamCircuitSummaries

set_option maxRecDepth 4096

namespace VerifiableTransformers.Replay.UpstreamFloatReplay

open VerifiableTransformers.Certificate
open VerifiableTransformers.Certificate.FiniteEval
open VerifiableTransformers.Certificate.Circuit
open VerifiableTransformers.Generated.UpstreamCheckpointPayload

/-- One exported vector, preserving the checkpoint's coordinate order. -/
abbrev FloatVec := Array Float

/-- Exported linear weights, one row per output coordinate. -/
abbrev FloatMatrix := Array (Array Float)

/-- Hidden vectors, one per token position. -/
abbrev HiddenSeq := Array FloatVec

/-! ## Array helpers for generated Float weights -/

/-- Total array lookup with `0.0` outside the exported coordinate range. -/
def vecGetD (xs : FloatVec) (i : Nat) : Float :=
  xs.getD i 0.0

/-- Row lookup, with the empty row outside the matrix. -/
def matrixRowD (M : FloatMatrix) (i : Nat) : FloatVec :=
  M.getD i #[]

/-- Float vector `f 0, ..., f (n - 1)`. -/
def tabulateVec (n : Nat) (f : Nat → Float) : FloatVec :=
  (List.range n).map f |>.toArray

/-- Hidden sequence `f 0, ..., f (n - 1)`. -/
def tabulateSeq (n : Nat) (f : Nat → FloatVec) : HiddenSeq :=
  (List.range n).map f |>.toArray

/-- Left-to-right Float sum of `f 0, ..., f (n - 1)`. -/
def sumRangeFloat (n : Nat) (f : Nat → Float) : Float :=
  (List.range n).foldl (fun acc i => acc + f i) 0.0

/-- Float maximum (`x` on ties). -/
def maxFloat (x y : Float) : Float :=
  if x >= y then x else y

/-- Float ReLU. -/
def reluFloat (x : Float) : Float :=
  maxFloat x 0.0

/-- Float LeakyReLU with the payload's negative slope `leakySlope`. -/
def leakyReluFloat (x : Float) : Float :=
  if x >= 0.0 then x else leakySlope * x

/-- Dot product over the first `n` coordinates, summed in coordinate order. -/
def dotFloat (x y : FloatVec) (n : Nat) : Float :=
  sumRangeFloat n (fun i => vecGetD x i * vecGetD y i)

/-- Row-major affine map `W x + b` for the upstream exported arrays. -/
def affineMapFloat (W : FloatMatrix) (b : FloatVec) (x : FloatVec)
    (outDim inDim : Nat) : FloatVec :=
  tabulateVec outDim (fun i => dotFloat (matrixRowD W i) x inDim + vecGetD b i)

/-- Coordinatewise vector addition for residual connections. -/
def addFloatVec (x y : FloatVec) (n : Nat) : FloatVec :=
  tabulateVec n (fun i => vecGetD x i + vecGetD y i)

/-- Subtract the Float mean from each coordinate. -/
def centerFloatVec (x : FloatVec) (n : Nat) : FloatVec :=
  let mu := sumRangeFloat n (fun i => vecGetD x i) / Float.ofNat n
  tabulateVec n (fun i => vecGetD x i - mu)

/-- Per-coordinate affine scale and shift used by BandNorm. -/
def affineFloatVec (gamma beta x : FloatVec) (n : Nat) : FloatVec :=
  tabulateVec n (fun i => vecGetD gamma i * vecGetD x i + vecGetD beta i)

/-! ## Float replay of the upstream custom operators

These are plain Float code, so the forward pass depends only on the embedded checkpoint
arrays.
-/

/-- Insert a Float into a descending list. -/
def insertDescFloat (x : Float) : List Float → List Float
  | [] => [x]
  | y :: ys => if x >= y then x :: y :: ys else y :: insertDescFloat x ys

/-- Descending sort used by the projection threshold scan. -/
def sortDescFloat : List Float → List Float
  | [] => []
  | x :: xs => insertDescFloat x (sortDescFloat xs)

/-- Top-k threshold scan shared by sparsemax and nonnegative L1 projection. -/
def projectionThresholdAux (radius : Float) :
    List Float → Nat → Float → Float → Float
  | [], _, _, best => best
  | y :: ys, k, pref, best =>
      let k' := k + 1
      let pref' := pref + y
      let tau := (pref' - radius) / Float.ofNat k'
      let best' := if y > tau then tau else best
      projectionThresholdAux radius ys k' pref' best'

/-- Projection threshold computed from unsorted Float values. -/
def projectionThreshold (radius : Float) (values : List Float) : Float :=
  projectionThresholdAux radius (sortDescFloat values) 0 0.0 0.0

/-- Nonnegative L1 projection with the same mass/threshold rule as upstream `train.py`. -/
def nonnegativeL1ProjectFloat (radius : Float) (y : FloatVec) (n : Nat) : FloatVec :=
  let mass := sumRangeFloat n (fun i => vecGetD y i)
  if mass <= radius then
    y
  else
    let tau := projectionThreshold radius ((List.range n).map (fun i => vecGetD y i))
    tabulateVec n (fun i => reluFloat (vecGetD y i - tau))

/-- Fallback mask for the positive half when its support is empty: `1` on even coordinates. -/
def positiveFallbackMask (n : Nat) : FloatVec :=
  tabulateVec n (fun i => if i % 2 == 0 then 1.0 else 0.0)

/-- Fallback mask for the negative half: `1` on odd coordinates. -/
def negativeFallbackMask (n : Nat) : FloatVec :=
  tabulateVec n (fun i => if i % 2 == 0 then 0.0 else 1.0)

/-- Whether a projected half has an active positive coordinate. -/
def hasPositiveCoordinate (y : FloatVec) (n : Nat) : Bool :=
  (List.range n).any (fun i => vecGetD y i > 0.0)

/-- Active set used by low-mass lifting, with the branch fallback for empty support. -/
def activeMaskOrFallbackFloat (fallback y : FloatVec) (n : Nat) : FloatVec :=
  if hasPositiveCoordinate y n then
    tabulateVec n (fun i => if vecGetD y i > 0.0 then 1.0 else 0.0)
  else
    fallback

/-- Raise a projected half to the lower L1 target along the active support. -/
def liftLowMassFloat (target : Float) (fallback y : FloatVec) (n : Nat) : FloatVec :=
  let mass := sumRangeFloat n (fun i => vecGetD y i)
  let deficit := reluFloat (target - mass)
  let active := activeMaskOrFallbackFloat fallback y n
  let activeCount := maxFloat 1.0 (sumRangeFloat n (fun i => vecGetD active i))
  let delta := deficit / activeCount
  tabulateVec n (fun i => vecGetD y i + vecGetD active i * delta)

/-- Float replay of the upstream Signed-L1-BandNorm layer. -/
def signedL1BandNormFloat (gamma beta x : FloatVec) : FloatVec :=
  let n := dModel
  let centered := centerFloatVec x n
  let pos := tabulateVec n (fun i => reluFloat (vecGetD centered i))
  let neg := tabulateVec n (fun i => reluFloat (-(vecGetD centered i)))
  let posProj := nonnegativeL1ProjectFloat halfHigh pos n
  let negProj := nonnegativeL1ProjectFloat halfHigh neg n
  let posLift := liftLowMassFloat halfLow (positiveFallbackMask n) posProj n
  let negLift := liftLowMassFloat halfLow (negativeFallbackMask n) negProj n
  let recombined := tabulateVec n (fun i => vecGetD posLift i - vecGetD negLift i)
  affineFloatVec gamma beta (centerFloatVec recombined n) n

/-- Sparsemax in Float: subtract the largest score, then project onto the simplex. -/
def sparsemaxFloat (scores : FloatVec) : FloatVec :=
  let n := scores.size
  let maxScore :=
    match (List.range n).map (fun i => vecGetD scores i) with
    | [] => 0.0
    | x :: xs => xs.foldl maxFloat x
  let shifted := tabulateVec n (fun i => vecGetD scores i - maxScore)
  let tau := projectionThreshold 1.0 ((List.range n).map (fun i => vecGetD shifted i))
  tabulateVec n (fun i => reluFloat (vecGetD shifted i - tau))

/-! ## Two-layer GPT replay -/

/--
One-head causal sparsemax attention at one sequence position.

The upstream default has `n_heads = 1` and `head_dim = d_model = 16`, so the replay
splits no heads.
-/
def causalSparsemaxAttentionAt (queries keys values : HiddenSeq) (pos : Nat) : FloatVec :=
  let q := queries.getD pos #[]
  let count := pos + 1
  let scores :=
    tabulateVec count (fun t =>
      dotFloat q (keys.getD t #[]) dModel / Float.sqrt (Float.ofNat dModel))
  let weights := sparsemaxFloat scores
  tabulateVec dModel (fun j =>
    sumRangeFloat count (fun t => vecGetD weights t * vecGetD (values.getD t #[]) j))

/-- Exported parameter bundle for one upstream transformer block. -/
structure LayerWeights where
  attnNormGamma : FloatVec
  attnNormBeta : FloatVec
  Wq : FloatMatrix
  bq : FloatVec
  Wk : FloatMatrix
  bk : FloatVec
  Wv : FloatMatrix
  bv : FloatVec
  Wo : FloatMatrix
  bo : FloatVec
  mlpNormGamma : FloatVec
  mlpNormBeta : FloatVec
  Wup : FloatMatrix
  bup : FloatVec
  Wdown : FloatMatrix
  bdown : FloatVec

/-- First transformer block weights from the generated checkpoint values. -/
def layer0 : LayerWeights :=
  { attnNormGamma := attn0normgamma, attnNormBeta := attn0normbeta,
    Wq := attn0Wq, bq := attn0bq, Wk := attn0Wk, bk := attn0bk,
    Wv := attn0Wv, bv := attn0bv, Wo := attn0Wo, bo := attn0bo,
    mlpNormGamma := mlp0normgamma, mlpNormBeta := mlp0normbeta,
    Wup := mlp0Wup, bup := mlp0bup,
    Wdown := mlp0Wdown, bdown := mlp0bdown }

/-- Second transformer block weights from the generated checkpoint values. -/
def layer1 : LayerWeights :=
  { attnNormGamma := attn1normgamma, attnNormBeta := attn1normbeta,
    Wq := attn1Wq, bq := attn1bq, Wk := attn1Wk, bk := attn1bk,
    Wv := attn1Wv, bv := attn1bv, Wo := attn1Wo, bo := attn1bo,
    mlpNormGamma := mlp1normgamma, mlpNormBeta := mlp1normbeta,
    Wup := mlp1Wup, bup := mlp1bup,
    Wdown := mlp1Wdown, bdown := mlp1bdown }

/-- Attention sublayer output `c_proj(attn(BandNorm(x)))`, before the residual add. -/
def attentionBlock (weights : LayerWeights) (hidden : HiddenSeq) : HiddenSeq :=
  let normedAttn := tabulateSeq seqLen (fun pos =>
    signedL1BandNormFloat weights.attnNormGamma weights.attnNormBeta
      (hidden.getD pos #[]))
  let queries := tabulateSeq seqLen (fun pos =>
    affineMapFloat weights.Wq weights.bq (normedAttn.getD pos #[]) dModel dModel)
  let keys := tabulateSeq seqLen (fun pos =>
    affineMapFloat weights.Wk weights.bk (normedAttn.getD pos #[]) dModel dModel)
  let values := tabulateSeq seqLen (fun pos =>
    affineMapFloat weights.Wv weights.bv (normedAttn.getD pos #[]) dModel dModel)
  tabulateSeq seqLen (fun pos =>
    affineMapFloat weights.Wo weights.bo (causalSparsemaxAttentionAt queries keys values pos)
      dModel dModel)

/-- MLP sublayer output `W_down(LeakyReLU(W_up BandNorm(x)))`, before the residual add. -/
def mlpBlock (weights : LayerWeights) (hidden : HiddenSeq) : HiddenSeq :=
  tabulateSeq seqLen (fun pos =>
    let normed := signedL1BandNormFloat weights.mlpNormGamma weights.mlpNormBeta
      (hidden.getD pos #[])
    let up := affineMapFloat weights.Wup weights.bup normed dFF dModel
    let act := tabulateVec dFF (fun i => leakyReluFloat (vecGetD up i))
    affineMapFloat weights.Wdown weights.bdown act dModel dFF)

/-- Positionwise sum of two hidden sequences. -/
def addSeq (x y : HiddenSeq) : HiddenSeq :=
  tabulateSeq seqLen (fun pos => addFloatVec (x.getD pos #[]) (y.getD pos #[]) dModel)

/-- Replay one residual attention-plus-MLP block with the exported weights. -/
def runTransformerLayer (weights : LayerWeights) (hidden : HiddenSeq) : HiddenSeq :=
  let afterAttn := addSeq hidden (attentionBlock weights hidden)
  addSeq afterAttn (mlpBlock weights afterAttn)

/-- Embed a finite-domain prompt using token and position tables from the checkpoint. -/
def embedPrompt (input : List Nat) : HiddenSeq :=
  tabulateSeq seqLen (fun pos =>
    let tok := input.getD pos 0
    addFloatVec (matrixRowD wte tok) (matrixRowD wpe pos) dModel)

/-- Hidden vector at the last position after both blocks and the final BandNorm. -/
def finalHiddenState (input : List Nat) : FloatVec :=
  let h0 := embedPrompt input
  let h1 := runTransformerLayer layer0 h0
  let h2 := runTransformerLayer layer1 h1
  signedL1BandNormFloat finalnormgamma finalnormbeta (h2.getD (seqLen - 1) #[])

/-- Untied output logit for one candidate token. -/
def outputLogit (input : List Nat) (tok : Nat) : Float :=
  dotFloat (matrixRowD lmhead tok) (finalHiddenState input) dModel

/-! ## Exhaustive finite-domain check -/

/-- Target and alternate logits of the canonical prompt at `idx`. -/
def candidateLogitsAt (idx : Nat) : Float × Float :=
  let hidden := finalHiddenState (expectedInput idx)
  (dotFloat (matrixRowD lmhead (expectedTarget idx)) hidden dModel,
   dotFloat (matrixRowD lmhead (expectedAlternate idx)) hidden dModel)

/-- Candidate logits of all 256 prompts, computed once. -/
def candidateLogits : Array (Float × Float) :=
  (List.range 256).map candidateLogitsAt |>.toArray

def targetMarginAt (idx : Nat) : Float :=
  let (t, a) := candidateLogits.getD idx (0.0, 0.0)
  t - a

/-- Whether one finite-domain prompt gives a positive projected target margin. -/
def rowPasses (idx : Nat) : Bool :=
  targetMarginAt idx > 0.0

/-- Exhaustive check over all 256 quote/bracket prompts. -/
def allRowsPass : Bool :=
  (List.range 256).all rowPasses

/-- Prompt indices whose projected target margin is non-positive. -/
def failures : List Nat :=
  (List.range 256).filter (fun idx => ! rowPasses idx)

/-- Minimum of a list of floats (`0.0` for the empty list). -/
def minList : List Float → Float
  | [] => 0.0
  | x :: xs => xs.foldl (fun acc m => if m < acc then m else acc) x

/-- Maximum of a list of floats (`0.0` for the empty list). -/
def maxList : List Float → Float
  | [] => 0.0
  | x :: xs => xs.foldl maxFloat x

/-- Minimum target-vs-alternate margin across the finite domain. -/
def minMargin : Float :=
  minList ((List.range 256).map targetMarginAt)

/-- Passing rows in the quote-close half of the finite task. -/
def quotePasses : Nat :=
  (List.range 128).filter rowPasses |>.length

/-- Passing rows in the bracket-type half of the finite task. -/
def bracketPasses : Nat :=
  ((List.range 128).map (fun i => i + 128)).filter rowPasses |>.length

/-! ## Agreement with the PyTorch score trace -/

/-- Largest allowed gap between a replayed logit and the trace, in logits. The trace is float32
rounded to micro-logits, so a correct replay stays within a few micro-logits. -/
def traceTolerance : Float := 1e-5

/-- Gap between the replayed candidate logits and the trace row at `idx`. -/
def traceGapAt (idx : Nat) : Float :=
  match Generated.UpstreamEvalTrace.evalRows[idx]? with
  | none => 1.0e9
  | some row =>
      let (t, a) := candidateLogits.getD idx (0.0, 0.0)
      maxFloat (Float.abs (t - Float.ofInt row.targetScoreMicros / 1.0e6))
        (Float.abs (a - Float.ofInt row.alternateScoreMicros / 1.0e6))

/-- Largest gap over the 256 rows. -/
def maxTraceGap : Float :=
  maxList ((List.range 256).map traceGapAt)

/-! ## BandNorm's empty-support fallback -/

/-- Constant vector of the hidden width. -/
def constantVec (c : Float) : FloatVec :=
  tabulateVec dModel (fun _ => c)

/-- BandNorm with unit scale and zero shift applied to a constant vector. -/
def bandNormOfConstant (c : Float) : FloatVec :=
  signedL1BandNormFloat (constantVec 1.0) (constantVec 0.0) (constantVec c)

/-- Largest distance from the alternating `+halfLow/8, -halfLow/8` fallback pattern. -/
def fallbackDistance (y : FloatVec) : Float :=
  let step := halfLow / 8.0
  maxList ((List.range dModel).map (fun i =>
    Float.abs (vecGetD y i - (if i % 2 == 0 then step else -step))))

/-! ## Circuits with zero ablation, as in upstream `controlled_forward` -/

/-- An all-zero hidden sequence: the contribution of an ablated edge. -/
def zeroSeq : HiddenSeq :=
  tabulateSeq seqLen (fun _ => constantVec 0.0)

/-- Output of an already computed node, by upstream node name. -/
def nodeOutput (outs : List (String × HiddenSeq)) (name : String) : HiddenSeq :=
  match outs.find? (fun o => o.1 == name) with
  | some o => o.2
  | none => zeroSeq

/-- Residual input of `node`: the sum of its parents whose edge is kept. -/
def nodeInput (keep : List Edge) (outs : List (String × HiddenSeq)) (node : String) :
    HiddenSeq :=
  smallCircuitGraphEdges.foldl (fun acc e =>
    if e.target == node && keep.contains e then addSeq acc (nodeOutput outs e.source) else acc)
    zeroSeq

/-- Final hidden vector when only the edges in `keep` carry activations. -/
def controlledFinalHidden (keep : List Edge) (input : List Nat) : FloatVec :=
  let outs := [("emb", embedPrompt input)]
  let outs := outs ++ [("attn_0", attentionBlock layer0 (nodeInput keep outs "attn_0"))]
  let outs := outs ++ [("mlp_0", mlpBlock layer0 (nodeInput keep outs "mlp_0"))]
  let outs := outs ++ [("attn_1", attentionBlock layer1 (nodeInput keep outs "attn_1"))]
  let outs := outs ++ [("mlp_1", mlpBlock layer1 (nodeInput keep outs "mlp_1"))]
  let final := nodeInput keep outs "logits"
  signedL1BandNormFloat finalnormgamma finalnormbeta (final.getD (seqLen - 1) #[])

/-- Target-minus-alternate margin of the circuit `keep` on the prompt at `idx`. -/
def controlledMarginAt (keep : List Edge) (idx : Nat) : Float :=
  let hidden := controlledFinalHidden keep (expectedInput idx)
  dotFloat (matrixRowD lmhead (expectedTarget idx)) hidden dModel -
    dotFloat (matrixRowD lmhead (expectedAlternate idx)) hidden dModel

/-- Largest margin gap between the controlled forward with all 15 edges kept and the plain
forward pass. -/
def fullGraphGap : Float :=
  maxList ((List.range 256).map (fun idx =>
    Float.abs (controlledMarginAt smallCircuitGraphEdges idx - targetMarginAt idx)))

/-- What the replay finds for one circuit. -/
structure CircuitReplay where
  label : String
  summary : VerificationSummary
  offset : Nat
  /-- Margins of the circuit on the task's 128 prompts. -/
  margins : Array Float
  /-- For each retained edge, the first prompt where removing it flips the decision. -/
  witnesses : List (Edge × Option Nat)

/-- Run one circuit over its task's prompts, and each one-edge ablation of it. -/
def replayCircuit (label : String) (s : VerificationSummary) (offset : Nat) : CircuitReplay :=
  let keep := s.circuit.edges
  let margins := (List.range 128).map (fun i => controlledMarginAt keep (offset + i)) |>.toArray
  let witness (e : Edge) : Option Nat :=
    let ablated := keep.erase e
    (List.range 128).find? (fun i =>
      (margins.getD i 0.0 > 0.0) != (controlledMarginAt ablated (offset + i) > 0.0))
  { label, summary := s, offset, margins, witnesses := keep.map (fun e => (e, witness e)) }

def CircuitReplay.correct (r : CircuitReplay) : Nat :=
  (r.margins.toList.filter (fun m => m > 0.0)).length

def CircuitReplay.necessary (r : CircuitReplay) : Nat :=
  (r.witnesses.filter (fun w => w.2.isSome)).length

/-- Whether the replay matches the recorded PyTorch failure count and the solver's
edge-necessity count, for each check that ran. -/
def CircuitReplay.agrees (r : CircuitReplay) : Bool :=
  let s := r.summary
  (s.pytorchStatus == "ERROR" || 128 - r.correct == s.pytorchFailures) &&
  (!(s.edgeStatus == "VERIFIED" || s.edgeStatus == "FAILED") || r.necessary == s.edgeNecessary)

def quoteReplay : CircuitReplay :=
  replayCircuit "quote" Generated.UpstreamCircuitSummaries.quoteVerification 0

def bracketReplay : CircuitReplay :=
  replayCircuit "bracket" Generated.UpstreamCircuitSummaries.bracketVerification 128

def edgeText (e : Edge) : String :=
  s!"{e.source} -> {e.target}"

def reportCircuit (r : CircuitReplay) : IO Unit := do
  let s := r.summary
  IO.println s!"{r.label} circuit ({s.circuit.edges.length} edges): \
    {r.correct}/128 prompts decided as the target, \
    smallest margin {minList r.margins.toList}"
  IO.println s!"  recorded: PyTorch {s.pytorchStatus} ({s.pytorchFailures} failures), \
    functional {s.functionalStatus}, edge necessity {s.edgeStatus} \
    ({s.edgeNecessary}/{s.edgeTotal}), robustness {s.robustnessStatus}"
  for (e, w) in r.witnesses do
    match w with
    | some i =>
        IO.println s!"  {edgeText e}: removing it flips the decision on \
          {expectedInput (r.offset + i)}"
    | none => IO.println s!"  {edgeText e}: removing it flips no decision"

/--
Print the replay results for all 256 prompts and both circuits.

Any failed comparison makes the command fail, so a local run or CI needs only the exit code.
-/
def main : IO Unit := do
  let traceSha := Generated.UpstreamEvalTrace.evalCertificate.sourceWeightsSha256
  IO.println s!"Lean Float replay of smt_weights.json {sourceSha256}"
  IO.println s!"quote rows passing:   {quotePasses}/128"
  IO.println s!"bracket rows passing: {bracketPasses}/128"
  IO.println s!"minimum target-vs-alternate margin: {minMargin}"
  IO.println s!"largest gap to the PyTorch trace: {maxTraceGap * 1.0e6} micro-logits \
    (tolerance {traceTolerance * 1.0e6})"
  let (t0, a0) := candidateLogits.getD 0 (0.0, 0.0)
  IO.println s!"prompt {expectedInput 0}: logit {expectedTarget 0} = {t0}, \
    logit {expectedAlternate 0} = {a0}"
  for c in [1.0, 0.3] do
    let y := bandNormOfConstant c
    IO.println s!"BandNorm of a constant {c} vector starts [{vecGetD y 0}, {vecGetD y 1}, ...], \
      distance to the +-{halfLow / 8.0} fallback pattern {fallbackDistance y}"
  IO.println s!"controlled forward with all 15 edges matches the plain forward pass: \
    {decide (fullGraphGap <= 1.0e-9)}"
  reportCircuit quoteReplay
  reportCircuit bracketReplay
  let mut problems : List String := []
  if sourceSha256 != Fingerprint.smtWeightsSha256 || sourceSha256 != traceSha then
    problems := problems ++ ["payload fingerprint differs from the pinned or traced one"]
  if !allRowsPass then
    problems := problems ++ [s!"rows with non-positive margin: {failures}"]
  if !(maxTraceGap <= traceTolerance) then
    problems := problems ++ ["replayed logits differ from the PyTorch trace"]
  if !(fallbackDistance (bandNormOfConstant 1.0) <= 1.0e-12) then
    problems := problems ++ ["BandNorm fallback on a constant vector is not the alternating pattern"]
  if !(fullGraphGap <= 1.0e-9) then
    problems := problems ++ ["controlled forward with every edge differs from the forward pass"]
  for r in [quoteReplay, bracketReplay] do
    if !r.agrees then
      problems := problems ++ [s!"{r.label} circuit replay disagrees with the recorded summary"]
  if problems.isEmpty then
    IO.println "PASS: all 256 finite-domain prompts satisfy the projected decision property, \
      and the replay agrees with the trace and the circuit summaries."
  else
    throw <| IO.userError s!"FAIL: {problems}"

end VerifiableTransformers.Replay.UpstreamFloatReplay

/-- Executable entrypoint for the upstream Float replay. -/
def main : IO Unit :=
  VerifiableTransformers.Replay.UpstreamFloatReplay.main
