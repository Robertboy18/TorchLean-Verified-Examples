/-
Generated TorchLean CUDA checkpoint summary and eval trace.

Lean checks two facts here: the exported parameter summary has the expected
shape/count metadata, and the finite eval rows satisfy the projected
quote/bracket property.

Sources: artifacts/torchlean-cuda-small-gpt.parambits.json and artifacts/torchlean-cuda-small-gpt.eval.json.
-/

import VerifiableTransformers.Certificate.FiniteEval
import VerifiableTransformers.Generated.UpstreamExportSummary

set_option maxRecDepth 4096

namespace VerifiableTransformers.Generated.TorchLeanEvalTrace

open VerifiableTransformers.Certificate.FiniteEval

def checkpointPath : String :=
  "artifacts/torchlean-cuda-small-gpt.parambits.json"

def checkpointSha256 : String :=
  "78cbaf4162354e8e2333bca30d37debb0565875dde4577eb72a7778bc81493d8"

def formatTag : String :=
  "torchlean_paramlist_bits_v1"

/-- TorchLean runtime parameter order for `TorchLean.TrainSmallGPT.mkTrainableModel`. -/
def paramTensorShapes : List (List Nat) :=
  [[32, 16], [6, 16], [16], [16], [16, 16], [16], [16, 16], [16], [16, 16], [16], [16, 16], [16], [16], [16], [64, 16], [64], [16, 64], [16], [16], [16], [16, 16], [16], [16, 16], [16], [16, 16], [16], [16, 16], [16], [16], [16], [64, 16], [64], [16, 64], [16], [16], [16], [32, 16]]

def paramValueCounts : List Nat :=
  [512, 96, 16, 16, 256, 16, 256, 16, 256, 16, 256, 16, 16, 16, 1024, 64, 1024, 16, 16, 16, 256, 16, 256, 16, 256, 16, 256, 16, 16, 16, 1024, 64, 1024, 16, 16, 16, 512]

def first16WeightBits : List Nat :=
  [13804512869032656896, 4581204437959180288, 4565868175970795520, 13789136080806608896, 4573201333875638272, 13795885263819374592, 13795846579585810432, 4572294467100344320, 13800249230838726656, 4575980389243289600, 4579601405840457728, 13801914890559946752, 13799361570620309504, 13777349279649628160, 4578844464562307072, 4573554774787489792]

def totalValueCount : Nat :=
  paramValueCounts.foldl (fun acc n => acc + n) 0

def checkpointSummaryOk : Bool :=
  paramTensorShapes.length == 37 &&
  totalValueCount == 7712 &&
  totalValueCount ==
    VerifiableTransformers.Certificate.Weights.tensorEntryCount
      VerifiableTransformers.Generated.UpstreamExportSummary.exportSummary.tensorFields &&
  first16WeightBits.length == 16 &&
  VerifiableTransformers.Generated.UpstreamExportSummary.exportSummary.vocabSize == 32 &&
  VerifiableTransformers.Generated.UpstreamExportSummary.exportSummary.maxSeqLen == 6 &&
  VerifiableTransformers.Generated.UpstreamExportSummary.exportSummary.dModel == 16 &&
  VerifiableTransformers.Generated.UpstreamExportSummary.exportSummary.nLayers == 2

theorem checkpointSummary_ok :
    checkpointSummaryOk = true := by
  rfl

def evalRows : List CandidateEval :=
[
  { task := .quoteClose, input := [1, 2, 5, 9, 5, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 5, 10, 5, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 5, 9, 5, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 5, 10, 5, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 5, 9, 5, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 5, 10, 5, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 5, 9, 5, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 5, 10, 5, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .quoteClose, input := [1, 2, 5, 9, 6, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 5, 10, 6, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 5, 9, 6, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 5, 10, 6, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 5, 9, 6, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 5, 10, 6, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 5, 9, 6, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 5, 10, 6, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .quoteClose, input := [1, 2, 5, 9, 7, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 5, 10, 7, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 5, 9, 7, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 5, 10, 7, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 5, 9, 7, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 5, 10, 7, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 5, 9, 7, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 5, 10, 7, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .quoteClose, input := [1, 2, 5, 9, 8, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 5, 10, 8, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 5, 9, 8, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 5, 10, 8, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 5, 9, 8, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 5, 10, 8, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 5, 9, 8, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 5, 10, 8, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .quoteClose, input := [1, 2, 6, 9, 5, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 6, 10, 5, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 6, 9, 5, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 6, 10, 5, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 6, 9, 5, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 6, 10, 5, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 6, 9, 5, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 6, 10, 5, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .quoteClose, input := [1, 2, 6, 9, 6, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 6, 10, 6, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 6, 9, 6, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 6, 10, 6, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 6, 9, 6, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 6, 10, 6, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 6, 9, 6, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 6, 10, 6, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .quoteClose, input := [1, 2, 6, 9, 7, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 6, 10, 7, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 6, 9, 7, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 6, 10, 7, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 6, 9, 7, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 6, 10, 7, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 6, 9, 7, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 6, 10, 7, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .quoteClose, input := [1, 2, 6, 9, 8, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 6, 10, 8, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 6, 9, 8, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 6, 10, 8, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 6, 9, 8, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 6, 10, 8, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 6, 9, 8, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 6, 10, 8, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .quoteClose, input := [1, 2, 7, 9, 5, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 7, 10, 5, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 7, 9, 5, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 7, 10, 5, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 7, 9, 5, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 7, 10, 5, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 7, 9, 5, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 7, 10, 5, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .quoteClose, input := [1, 2, 7, 9, 6, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 7, 10, 6, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 7, 9, 6, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 7, 10, 6, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 7, 9, 6, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 7, 10, 6, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 7, 9, 6, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 7, 10, 6, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .quoteClose, input := [1, 2, 7, 9, 7, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 7, 10, 7, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 7, 9, 7, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 7, 10, 7, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 7, 9, 7, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 7, 10, 7, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 7, 9, 7, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 7, 10, 7, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .quoteClose, input := [1, 2, 7, 9, 8, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 7, 10, 8, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 7, 9, 8, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 7, 10, 8, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 7, 9, 8, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 7, 10, 8, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 7, 9, 8, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 7, 10, 8, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .quoteClose, input := [1, 2, 8, 9, 5, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 8, 10, 5, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 8, 9, 5, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 8, 10, 5, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 8, 9, 5, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 8, 10, 5, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 8, 9, 5, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 8, 10, 5, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .quoteClose, input := [1, 2, 8, 9, 6, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 8, 10, 6, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 8, 9, 6, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 8, 10, 6, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 8, 9, 6, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 8, 10, 6, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 8, 9, 6, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 8, 10, 6, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .quoteClose, input := [1, 2, 8, 9, 7, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 8, 10, 7, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 8, 9, 7, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 8, 10, 7, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 8, 9, 7, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 8, 10, 7, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 8, 9, 7, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 8, 10, 7, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .quoteClose, input := [1, 2, 8, 9, 8, 5], target := 9, alternate := 10, targetScoreMicros := 3176348, alternateScoreMicros := 2164653 },
  { task := .quoteClose, input := [1, 2, 8, 10, 8, 5], target := 10, alternate := 9, targetScoreMicros := 3998022, alternateScoreMicros := 3719960 },
  { task := .quoteClose, input := [1, 2, 8, 9, 8, 6], target := 9, alternate := 10, targetScoreMicros := 3169240, alternateScoreMicros := 2187928 },
  { task := .quoteClose, input := [1, 2, 8, 10, 8, 6], target := 10, alternate := 9, targetScoreMicros := 4014421, alternateScoreMicros := 3725699 },
  { task := .quoteClose, input := [1, 2, 8, 9, 8, 7], target := 9, alternate := 10, targetScoreMicros := 3167773, alternateScoreMicros := 2188036 },
  { task := .quoteClose, input := [1, 2, 8, 10, 8, 7], target := 10, alternate := 9, targetScoreMicros := 4013872, alternateScoreMicros := 3725513 },
  { task := .quoteClose, input := [1, 2, 8, 9, 8, 8], target := 9, alternate := 10, targetScoreMicros := 3165252, alternateScoreMicros := 2184325 },
  { task := .quoteClose, input := [1, 2, 8, 10, 8, 8], target := 10, alternate := 9, targetScoreMicros := 4009623, alternateScoreMicros := 3724019 },
  { task := .bracketType, input := [1, 3, 5, 11, 5, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 5, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 5, 11, 5, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 5, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 5, 11, 5, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 5, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 5, 11, 5, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 5, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 },
  { task := .bracketType, input := [1, 3, 5, 11, 6, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 6, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 5, 11, 6, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 6, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 5, 11, 6, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 6, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 5, 11, 6, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 6, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 },
  { task := .bracketType, input := [1, 3, 5, 11, 7, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 7, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 5, 11, 7, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 7, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 5, 11, 7, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 7, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 5, 11, 7, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 7, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 },
  { task := .bracketType, input := [1, 3, 5, 11, 8, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 8, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 5, 11, 8, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 8, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 5, 11, 8, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 8, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 5, 11, 8, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 5, 12, 8, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 },
  { task := .bracketType, input := [1, 3, 6, 11, 5, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 5, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 6, 11, 5, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 5, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 6, 11, 5, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 5, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 6, 11, 5, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 5, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 },
  { task := .bracketType, input := [1, 3, 6, 11, 6, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 6, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 6, 11, 6, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 6, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 6, 11, 6, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 6, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 6, 11, 6, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 6, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 },
  { task := .bracketType, input := [1, 3, 6, 11, 7, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 7, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 6, 11, 7, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 7, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 6, 11, 7, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 7, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 6, 11, 7, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 7, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 },
  { task := .bracketType, input := [1, 3, 6, 11, 8, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 8, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 6, 11, 8, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 8, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 6, 11, 8, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 8, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 6, 11, 8, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 6, 12, 8, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 },
  { task := .bracketType, input := [1, 3, 7, 11, 5, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 5, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 7, 11, 5, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 5, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 7, 11, 5, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 5, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 7, 11, 5, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 5, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 },
  { task := .bracketType, input := [1, 3, 7, 11, 6, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 6, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 7, 11, 6, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 6, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 7, 11, 6, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 6, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 7, 11, 6, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 6, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 },
  { task := .bracketType, input := [1, 3, 7, 11, 7, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 7, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 7, 11, 7, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 7, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 7, 11, 7, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 7, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 7, 11, 7, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 7, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 },
  { task := .bracketType, input := [1, 3, 7, 11, 8, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 8, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 7, 11, 8, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 8, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 7, 11, 8, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 8, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 7, 11, 8, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 7, 12, 8, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 },
  { task := .bracketType, input := [1, 3, 8, 11, 5, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 5, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 8, 11, 5, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 5, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 8, 11, 5, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 5, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 8, 11, 5, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 5, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 },
  { task := .bracketType, input := [1, 3, 8, 11, 6, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 6, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 8, 11, 6, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 6, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 8, 11, 6, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 6, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 8, 11, 6, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 6, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 },
  { task := .bracketType, input := [1, 3, 8, 11, 7, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 7, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 8, 11, 7, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 7, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 8, 11, 7, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 7, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 8, 11, 7, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 7, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 },
  { task := .bracketType, input := [1, 3, 8, 11, 8, 5], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 8, 5], target := 14, alternate := 13, targetScoreMicros := 3900815, alternateScoreMicros := 3609976 },
  { task := .bracketType, input := [1, 3, 8, 11, 8, 6], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 8, 6], target := 14, alternate := 13, targetScoreMicros := 3869058, alternateScoreMicros := 3621370 },
  { task := .bracketType, input := [1, 3, 8, 11, 8, 7], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 8, 7], target := 14, alternate := 13, targetScoreMicros := 3867515, alternateScoreMicros := 3623626 },
  { task := .bracketType, input := [1, 3, 8, 11, 8, 8], target := 13, alternate := 14, targetScoreMicros := 2728338, alternateScoreMicros := 2224913 },
  { task := .bracketType, input := [1, 3, 8, 12, 8, 8], target := 14, alternate := 13, targetScoreMicros := 3868959, alternateScoreMicros := 3627482 }
]

def evalCertificate : EvalCertificate where
  sourceWeightsSha256 := "78cbaf4162354e8e2333bca30d37debb0565875dde4577eb72a7778bc81493d8"
  checkpointPath := checkpointPath
  scoreScale := 1000000
  rows := evalRows

theorem evalTrace_ok :
    checkEvalCertificateWithSha
      checkpointSha256
      evalCertificate = true := by
  rfl

end VerifiableTransformers.Generated.TorchLeanEvalTrace
