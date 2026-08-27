/-
Lean Float replay of Neel Somani's exported checkpoint.

The strongest checkpoint check happens here: the generated Float arrays are
embedded in Lean, the small GPT forward pass is recomputed, and all 256
quote/bracket prompts are checked again.

The replay follows `scripts/small/train.py` and `scripts/small/extract_weights.py`:
row-major linear maps, Signed-L1-BandNorm, one-head causal sparsemax attention,
LeakyReLU, and the untied `lm_head`.

The replay stays compact on purpose. The rational SMT contract is in
`Spec/UpstreamSmallGPT.lean`, while the archived TorchLean comparison has its own generated trace.
Keeping those claims separate makes the checkpoint replay easy to audit.
-/

import VerifiableTransformers.Certificate.FiniteEval
import VerifiableTransformers.Generated.UpstreamCheckpointPayload

set_option maxRecDepth 4096

namespace VerifiableTransformers.Replay.UpstreamFloatReplay

open VerifiableTransformers.Certificate.FiniteEval
open VerifiableTransformers.Generated.UpstreamCheckpointPayload

abbrev FloatVec := Array Float
abbrev FloatMatrix := Array (Array Float)
abbrev HiddenSeq := Array FloatVec

/-! ## Array helpers for generated Float weights -/

/-- Total array lookup with `0.0` outside the exported coordinate range. -/
def vecGetD (xs : FloatVec) (i : Nat) : Float :=
  xs.getD i 0.0

/-- Total row lookup for row-major generated matrices. -/
def matrixRowD (M : FloatMatrix) (i : Nat) : FloatVec :=
  M.getD i #[]

/-- Build a generated-style Float vector by enumerating coordinates. -/
def tabulateVec (n : Nat) (f : Nat → Float) : FloatVec :=
  (List.range n).map f |>.toArray

/-- Build a sequence of hidden vectors by enumerating positions. -/
def tabulateSeq (n : Nat) (f : Nat → FloatVec) : HiddenSeq :=
  (List.range n).map f |>.toArray

/-- Left-folded Float summation in the checkpoint replay's coordinate order. -/
def sumRangeFloat (n : Nat) (f : Nat → Float) : Float :=
  (List.range n).foldl (fun acc i => acc + f i) 0.0

/-- Maximum operation used by ReLU and sparsemax thresholding. -/
def maxFloat (x y : Float) : Float :=
  if x >= y then x else y

/-- Scalar ReLU in the upstream Float replay. -/
def reluFloat (x : Float) : Float :=
  maxFloat x 0.0

/-- Scalar LeakyReLU with the exported upstream negative slope. -/
def leakyReluFloat (x : Float) : Float :=
  if x >= 0.0 then x else leakySlope * x

/-- Coordinate-order dot product for exported row-major weights. -/
def dotFloat (x y : FloatVec) (n : Nat) : Float :=
  sumRangeFloat n (fun i => vecGetD x i * vecGetD y i)

/-- Row-major affine map `W x + b` for the upstream exported arrays. -/
def affineMapFloat (W : FloatMatrix) (b : FloatVec) (x : FloatVec)
    (outDim inDim : Nat) : FloatVec :=
  tabulateVec outDim (fun i => dotFloat (matrixRowD W i) x inDim + vecGetD b i)

/-- Coordinatewise vector addition for residual connections. -/
def addFloatVec (x y : FloatVec) (n : Nat) : FloatVec :=
  tabulateVec n (fun i => vecGetD x i + vecGetD y i)

/-- Subtract the replay's Float mean from each coordinate. -/
def centerFloatVec (x : FloatVec) (n : Nat) : FloatVec :=
  let mu := sumRangeFloat n (fun i => vecGetD x i) / Float.ofNat n
  tabulateVec n (fun i => vecGetD x i - mu)

/-- Per-coordinate affine scale and shift used by BandNorm. -/
def affineFloatVec (gamma beta x : FloatVec) (n : Nat) : FloatVec :=
  tabulateVec n (fun i => vecGetD gamma i * vecGetD x i + vecGetD beta i)

/-! ## Float replay of the upstream custom operators

TorchLean provides runtime implementations for these operator families. The
versions here are plain Float code so the replay executable depends
only on the embedded checkpoint arrays and the finite-domain task definition.
-/

/-- Insert one Float into a descending list, using the replay's comparison order. -/
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

/-- Nonnegative L1 projection with the same mass/threshold rule as Neel's run. -/
def nonnegativeL1ProjectFloat (radius : Float) (y : FloatVec) (n : Nat) : FloatVec :=
  let mass := sumRangeFloat n (fun i => vecGetD y i)
  if mass <= radius then
    y
  else
    let tau := projectionThreshold radius ((List.range n).map (fun i => vecGetD y i))
    tabulateVec n (fun i => reluFloat (vecGetD y i - tau))

/-- Alternating positive-side fallback mask from the upstream BandNorm branch policy. -/
def positiveFallbackMask (n : Nat) : FloatVec :=
  tabulateVec n (fun i => if i % 2 == 0 then 1.0 else 0.0)

/-- Alternating negative-side fallback mask from the upstream BandNorm branch policy. -/
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

/-- Float replay of sparsemax as projection of shifted scores onto the simplex. -/
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

The upstream default has `n_heads = 1` and `head_dim = d_model = 16`, so there
is no additional head splitting in this replay.
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

/-- Replay one residual attention-plus-MLP block with the exported weights. -/
def runTransformerLayer (weights : LayerWeights) (hidden : HiddenSeq) : HiddenSeq :=
  let normedAttn := tabulateSeq seqLen (fun pos =>
    signedL1BandNormFloat weights.attnNormGamma weights.attnNormBeta
      (hidden.getD pos #[]))
  let queries := tabulateSeq seqLen (fun pos =>
    affineMapFloat weights.Wq weights.bq (normedAttn.getD pos #[]) dModel dModel)
  let keys := tabulateSeq seqLen (fun pos =>
    affineMapFloat weights.Wk weights.bk (normedAttn.getD pos #[]) dModel dModel)
  let values := tabulateSeq seqLen (fun pos =>
    affineMapFloat weights.Wv weights.bv (normedAttn.getD pos #[]) dModel dModel)
  let attnRaw := tabulateSeq seqLen (fun pos =>
    affineMapFloat weights.Wo weights.bo (causalSparsemaxAttentionAt queries keys values pos)
      dModel dModel)
  let afterAttn := tabulateSeq seqLen (fun pos =>
    addFloatVec (hidden.getD pos #[]) (attnRaw.getD pos #[]) dModel)
  let normedMlp := tabulateSeq seqLen (fun pos =>
    signedL1BandNormFloat weights.mlpNormGamma weights.mlpNormBeta
      (afterAttn.getD pos #[]))
  let mlpOut := tabulateSeq seqLen (fun pos =>
    let up := affineMapFloat weights.Wup weights.bup (normedMlp.getD pos #[]) dFF dModel
    let act := tabulateVec dFF (fun i => leakyReluFloat (vecGetD up i))
    affineMapFloat weights.Wdown weights.bdown act dModel dFF)
  tabulateSeq seqLen (fun pos =>
    addFloatVec (afterAttn.getD pos #[]) (mlpOut.getD pos #[]) dModel)

/-- Embed a finite-domain prompt using token and position tables from the checkpoint. -/
def embedPrompt (input : List Nat) : HiddenSeq :=
  tabulateSeq seqLen (fun pos =>
    let tok := input.getD pos 0
    addFloatVec (matrixRowD wte tok) (matrixRowD wpe pos) dModel)

/-- Final hidden vector at the prediction position after both replayed blocks. -/
def finalHiddenState (input : List Nat) : FloatVec :=
  let h0 := embedPrompt input
  let h1 := runTransformerLayer layer0 h0
  let h2 := runTransformerLayer layer1 h1
  signedL1BandNormFloat finalnormgamma finalnormbeta (h2.getD (seqLen - 1) #[])

/-- Untied output logit for one candidate token. -/
def outputLogit (input : List Nat) (tok : Nat) : Float :=
  dotFloat (matrixRowD lmhead tok) (finalHiddenState input) dModel

/-! ## Exhaustive finite-domain check -/

def targetMarginAt (idx : Nat) : Float :=
  let input := expectedInput idx
  let target := expectedTarget idx
  let alternate := expectedAlternate idx
  outputLogit input target - outputLogit input alternate

/-- Whether one finite-domain prompt gives a positive projected target margin. -/
def rowPasses (idx : Nat) : Bool :=
  targetMarginAt idx > 0.0

/-- Exhaustive check over all 256 quote/bracket prompts. -/
def allRowsPass : Bool :=
  (List.range 256).all rowPasses

/-- Prompt indices whose projected target margin is non-positive. -/
def failures : List Nat :=
  (List.range 256).filter (fun idx => ! rowPasses idx)

/-- Minimum target-vs-alternate margin across the finite domain. -/
def minMargin : Float :=
  match (List.range 256).map targetMarginAt with
  | [] => 0.0
  | x :: xs => xs.foldl (fun acc m => if m < acc then m else acc) x

/-- Passing rows in the quote-close half of the finite task. -/
def quotePasses : Nat :=
  (List.range 128).filter rowPasses |>.length

/-- Passing rows in the bracket-type half of the finite task. -/
def bracketPasses : Nat :=
  ((List.range 128).map (fun i => i + 128)).filter rowPasses |>.length

/-- Print the replay summary reported by `lake exe verify_upstream_forward`. -/
def main : IO Unit := do
  IO.println "Lean Float replay for neel-small-gpt"
  IO.println s!"quote rows passing:   {quotePasses}/128"
  IO.println s!"bracket rows passing: {bracketPasses}/128"
  IO.println s!"minimum target-vs-alternate margin: {minMargin}"
  if allRowsPass then
    IO.println "PASS: all 256 finite-domain prompts satisfy the projected decision property."
  else
    IO.println s!"FAIL: rows with non-positive margin: {failures}"

end VerifiableTransformers.Replay.UpstreamFloatReplay

/-- Executable entrypoint for the upstream Float replay. -/
def main : IO Unit :=
  VerifiableTransformers.Replay.UpstreamFloatReplay.main
