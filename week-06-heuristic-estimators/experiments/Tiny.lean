/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
Authors: Robert Joseph George
-/

import Lean.Data.Json.Printer
import NN.Tensor
import VerifiableTransformers.Replay.UpstreamFloatReplay

/-!
# Removing score coordinates in a trained sparsemax transformer

We reuse Week 2's checkpoint, finite prompts, normalization, sparsemax, and feed-forward replay.
Only the first attention block's query–key contraction changes. Its retained coordinates are
packed into TorchLean tensors; value aggregation keeps the full hidden width. The second block,
residual paths, Signed-L1-BandNorm, LeakyReLU, and output head are unchanged.

Half the finite prompts fit a coordinate ordering by squared score error. The split balances
both tasks and both candidate labels; its other half is reserved for evaluation. Those prompts
are held out from this coordinate search, not necessarily from the checkpoint's original training.
We also report the complete 256-prompt domain, separately from the held-out results.

The rank-16 control must reproduce the original replay on every prompt before reduced ranks are
reported. These are executable binary64 comparisons, not a new theorem about floating-point
evaluation. Candidate-token decisions are the finite task's projected decisions, not top-1 over
the complete vocabulary. Exact binary64 words accompany the displayed decimal observations.
-/

open TorchLean Lean
open VerifiableTransformers.Certificate.FiniteEval
open VerifiableTransformers.Generated.UpstreamCheckpointPayload
open VerifiableTransformers.Replay.UpstreamFloatReplay

namespace TinyExperiment

/-- Content-token parity splits both labels of both tasks evenly. -/
private def calibration (index : Nat) : Bool :=
  ((index / 2) % 4 + (index / 8) % 4 + (index / 32) % 4) % 2 == 0

private def asTensor (hidden : HiddenSeq) : Tensor Float [seqLen, dModel] :=
  Tensor.matrix fun i j => vecGetD (hidden.getD i.val #[]) j.val

/-- The original affine projections and normalization, without changing their arithmetic. -/
private def projections (weights : LayerWeights) (hidden : HiddenSeq) :
    Tensor Float [seqLen, dModel] × Tensor Float [seqLen, dModel] ×
      Tensor Float [seqLen, dModel] :=
  let normalized := tabulateSeq seqLen fun position =>
    signedL1BandNormFloat weights.attnNormGamma weights.attnNormBeta
      (hidden.getD position #[])
  let project (matrix : FloatMatrix) (bias : FloatVec) : Tensor Float [seqLen, dModel] :=
    asTensor <| tabulateSeq seqLen fun position =>
      affineMapFloat matrix bias (normalized.getD position #[]) dModel dModel
  (project weights.Wq weights.bq, project weights.Wk weights.bk,
    project weights.Wv weights.bv)

/-- Causal sparsemax keeps the original `sqrt(16)` scale at every retained rank. -/
private def attend (kept : Array (Fin dModel))
    (queries keys values : Tensor Float [seqLen, dModel]) (position : Nat) : FloatVec :=
  let count := min (position + 1) seqLen
  let query : Tensor Float [kept.size] :=
    Tensor.ofFn fun i =>
      queries[((⟨position % seqLen, Nat.mod_lt _ (by decide)⟩ : Fin seqLen), kept[i])]
  let bank : Tensor Float [count, kept.size] :=
    Tensor.matrix fun i j => keys[((⟨i.val, Nat.lt_of_lt_of_le i.isLt (Nat.min_le_right _ _)⟩ :
      Fin seqLen), kept[j])]
  let scores := Tensor.map (fun value => value / Float.sqrt (Float.ofNat dModel))
    (Tensor.matvec bank query)
  let probabilities := sparsemaxFloat (Tensor.to scores (Array Float))
  let weights : Tensor Float [count] := Tensor.ofFn fun i => vecGetD probabilities i.val
  let valueBank : Tensor Float [count, dModel] := Tensor.matrix fun i j =>
    values[((⟨i.val, Nat.lt_of_lt_of_le i.isLt (Nat.min_le_right _ _)⟩ : Fin seqLen), j)]
  Tensor.to (Tensor.vecmat weights valueBank) (Array Float)

/-- Reuse the first block's residual and MLP calculations after replacing its score contraction. -/
private def firstBlock (kept : Array (Fin dModel)) (hidden : HiddenSeq) : HiddenSeq :=
  let (queries, keys, values) := projections layer0 hidden
  let attended := tabulateSeq seqLen fun position =>
    affineMapFloat layer0.Wo layer0.bo (attend kept queries keys values position) dModel dModel
  let residual := tabulateSeq seqLen fun position =>
    addFloatVec (hidden.getD position #[]) (attended.getD position #[]) dModel
  let normalized := tabulateSeq seqLen fun position =>
    signedL1BandNormFloat layer0.mlpNormGamma layer0.mlpNormBeta (residual.getD position #[])
  let feedForward := tabulateSeq seqLen fun position =>
    let up := affineMapFloat layer0.Wup layer0.bup (normalized.getD position #[]) dFF dModel
    let activated := tabulateVec dFF fun i => leakyReluFloat (vecGetD up i)
    affineMapFloat layer0.Wdown layer0.bdown activated dModel dFF
  tabulateSeq seqLen fun position =>
    addFloatVec (residual.getD position #[]) (feedForward.getD position #[]) dModel

private def logits (hidden : FloatVec) : Tensor Float [vocabSize] :=
  Tensor.ofFn fun token => dotFloat (matrixRowD lmhead token.val) hidden dModel

private def forward (kept : Array (Fin dModel)) (index : Nat) : Tensor Float [vocabSize] :=
  let first := firstBlock kept (embedPrompt (expectedInput index))
  let second := runTransformerLayer layer1 first
  logits <| signedL1BandNormFloat finalnormgamma finalnormbeta (second.getD (seqLen - 1) #[])

/-- Squared-score error includes coordinate cross terms, not just coordinate magnitudes. -/
private def fit (indices : Array Nat) : Tensor Float [dModel, dModel] :=
  let projected := indices.map fun index => projections layer0 (embedPrompt (expectedInput index))
  let count := indices.size * seqLen * seqLen
  let features : Tensor Float [count, dModel] := Tensor.matrix fun row coordinate =>
    let sample := row.val / (seqLen * seqLen)
    let query := row.val / seqLen % seqLen
    let key := row.val % seqLen
    let (queries, keys, _) := projected[sample]!
    if key ≤ query then
      queries[((⟨query, Nat.mod_lt _ (by decide)⟩ : Fin seqLen), coordinate)] *
        keys[((⟨key, Nat.mod_lt _ (by decide)⟩ : Fin seqLen), coordinate)] /
        Float.sqrt (Float.ofNat dModel)
    else 0
  let transposed : Tensor Float [dModel, count] := Tensor.matrix fun i j => features[(j, i)]
  Tensor.matmul transposed features

/-- Greedily reduce the calibration quadratic error; ties keep the smaller coordinate. -/
private def ordering (matrix : Tensor Float [dModel, dModel]) : Array Nat := Id.run do
  let coordinates := Array.ofFn fun i : Fin dModel => i
  let mut order : Array Nat := #[]
  for _ in [0:dModel] do
    let remaining := coordinates.filter fun i => !order.contains i.val
    let gain (i : Fin dModel) : Float :=
      2 * Tensor.sum (Tensor.ofFn fun j : Fin dModel =>
        if order.contains j.val then 0 else matrix[(i, j)]) - matrix[(i, i)]
    let first := remaining.getD 0 ⟨0, by decide⟩
    let best := remaining.foldl (fun best i => if gain i > gain best then i else best) first
    order := order.push best.val
  return order

/-- Retained coordinates keep their original summation order. -/
private def retained (order : Array Nat) (rank : Nat) : Array (Fin dModel) :=
  (Array.ofFn fun i : Fin dModel => i).filter fun i => (order.extract 0 rank).contains i.val

private def floatJson (value : Float) : Json :=
  match JsonNumber.fromFloat? value with
  | .inr number => .num number
  | .inl spelling => .str spelling

private def margin (index : Nat) (scores : Tensor Float [vocabSize]) : Float :=
  scores.getScalar ⟨expectedTarget index % vocabSize, Nat.mod_lt _ (by decide)⟩ -
    scores.getScalar ⟨expectedAlternate index % vocabSize, Nat.mod_lt _ (by decide)⟩

private def observation (index rank : Nat) (reference actual : Tensor Float [vocabSize]) : Json :=
  Json.mkObj [
    ("index", toJson index), ("rank", toJson rank),
    ("calibration", toJson (calibration index)),
    ("task", toJson (if index < 128 then "quote" else "bracket")),
    ("target", toJson (expectedTarget index)),
    ("baseline_margin", floatJson (margin index reference)),
    ("margin", floatJson (margin index actual)),
    ("baseline_margin_bits", toJson (margin index reference).toBits),
    ("margin_bits", toJson (margin index actual).toBits),
    ("max_logit_error", floatJson (Tensor.maxAbsDiff reference actual)),
    ("output_bits", toJson ((Tensor.to actual (Array Float)).map Float.toBits)),
    ("baseline_bits", toJson ((Tensor.to reference (Array Float)).map Float.toBits))]

/-- Write observations after checking the unchanged-width control on the complete finite domain. -/
def run : IO Unit := do
  unless checkpointPayloadOk do throw <| IO.userError "checkpoint tensor shape mismatch"
  let calibrationIndices := (Array.range 256).filter calibration
  let evaluationIndices := (Array.range 256).filter fun i => !calibration i
  unless calibrationIndices.size == 128 && evaluationIndices.size == 128 do
    throw <| IO.userError "invalid calibration/evaluation partition"
  for task in [0:2] do
    for label in [0:2] do
      let members := calibrationIndices.filter fun i => i / 128 == task && i % 2 == label
      unless members.size == 32 do throw <| IO.userError "calibration split is not balanced"
  let started ← IO.monoNanosNow
  let matrix ← IO.lazyPure fun _ => fit calibrationIndices
  let order ← IO.lazyPure fun _ => ordering matrix
  let fitted ← IO.monoNanosNow
  let directory : System.FilePath := "/tmp/torchlean-multimodel/tiny"
  IO.FS.createDirAll directory
  let metadata := Json.mkObj [
    ("checkpoint_sha256", toJson sourceSha256),
    ("calibration_indices", toJson calibrationIndices),
    ("evaluation_indices", toJson evaluationIndices),
    ("ordering", toJson order),
    ("fit_ns", toJson (fitted - started)),
    ("scope", toJson "first-block QK coordinates; two candidate tokens; finite 256-prompt domain")]
  IO.FS.writeFile (directory / "metadata.json") metadata.pretty
  IO.FS.withFile (directory / "observations.jsonl") .write fun destination => do
    let complete := Array.ofFn fun i : Fin dModel => i
    for index in [0:256] do
      let reference ← IO.lazyPure fun _ => logits (finalHiddenState (expectedInput index))
      let control ← IO.lazyPure fun _ => forward complete index
      unless Tensor.maxAbsDiff reference control == 0 do
        throw <| IO.userError s!"rank-16 control differs from original replay at prompt {index}"
      for rank in #[16, 12, 8, 4] do
        let kept := retained order rank
        let actual ← IO.lazyPure fun _ => forward kept index
        unless Tensor.foldl (fun valid value => valid && !value.isNaN && !value.isInf)
            true actual do
          throw <| IO.userError s!"nonfinite result at prompt {index}, rank {rank}"
        destination.putStrLn (observation index rank reference actual).compress
    destination.flush
  IO.println s!"sparsemax transformer: 256 prompts, balanced 128/128 split; results in {directory}"

end TinyExperiment

-- The imported replay owns `main`; checking this experiment runs its separate calculation.
#eval TinyExperiment.run
