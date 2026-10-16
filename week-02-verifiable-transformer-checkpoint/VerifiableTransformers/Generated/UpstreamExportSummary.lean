/-
Generated summary certificate for:

  artifacts/upstream-small-gpt/smt_weights.json
  artifacts/upstream-small-gpt/config.json
  artifacts/upstream-small-gpt/provenance.json

Lean records the metadata, operator tags, and tensor-shape summaries here, while the full
floating-point arrays live in the checkpoint payload file. The generator records the
export's SHA-256, and Lean compares it with `Certificate.Fingerprint` without recomputing it
from the checkpoint bytes. Do not edit by hand; regenerate with
tools/generate_export_certificate.py.
-/

import VerifiableTransformers.Certificate.Weights

namespace VerifiableTransformers.Generated.UpstreamExportSummary

open VerifiableTransformers.Certificate.Weights
open VerifiableTransformers.Spec.UpstreamSmallGPT

def tensorFields : List TensorSummary :=
  [ tensor .wte [32, 16]
  , tensor .wpe [6, 16]
  , tensor (.attnWq 0) [16, 16]
  , tensor (.attnWk 0) [16, 16]
  , tensor (.attnWv 0) [16, 16]
  , tensor (.attnBq 0) [16]
  , tensor (.attnBk 0) [16]
  , tensor (.attnBv 0) [16]
  , tensor (.attnWo 0) [16, 16]
  , tensor (.attnBo 0) [16]
  , tensor (.mlpWUp 0) [64, 16]
  , tensor (.mlpBUp 0) [64]
  , tensor (.mlpWDown 0) [16, 64]
  , tensor (.mlpBDown 0) [16]
  , tensor (.attnNormGamma 0) [16]
  , tensor (.attnNormBeta 0) [16]
  , tensor (.mlpNormGamma 0) [16]
  , tensor (.mlpNormBeta 0) [16]
  , tensor (.attnWq 1) [16, 16]
  , tensor (.attnWk 1) [16, 16]
  , tensor (.attnWv 1) [16, 16]
  , tensor (.attnBq 1) [16]
  , tensor (.attnBk 1) [16]
  , tensor (.attnBv 1) [16]
  , tensor (.attnWo 1) [16, 16]
  , tensor (.attnBo 1) [16]
  , tensor (.mlpWUp 1) [64, 16]
  , tensor (.mlpBUp 1) [64]
  , tensor (.mlpWDown 1) [16, 64]
  , tensor (.mlpBDown 1) [16]
  , tensor (.attnNormGamma 1) [16]
  , tensor (.attnNormBeta 1) [16]
  , tensor (.mlpNormGamma 1) [16]
  , tensor (.mlpNormBeta 1) [16]
  , tensor .finalNormGamma [16]
  , tensor .finalNormBeta [16]
  , tensor .lmHead [32, 16]
  ]

def exportSummary : ExportSummary where
  sourcePath := "artifacts/upstream-small-gpt/smt_weights.json"
  sha256 := "ab61ac5dce70292ab9fa4dcd5265d98e04c34c5d89b809384ab297ef9e7abc34"
  upstreamCommit := "8e3c54bad2494c17c91109fdb58cf9e2f1c98efc"
  fieldCount := 48
  vocabSize := 32
  maxSeqLen := 6
  dModel := 16
  nLayers := 2
  nHeads := 1
  dMlp := 64
  headDim := 16
  normVariant := "signed_l1_band_norm"
  attnVariant := "sparsemax"
  activationVariant := "leaky_relu"
  halfLow := 44 / 10
  halfHigh := 84 / 10
  tensorFields := tensorFields

/- The imported export summary passes the checker. -/
theorem exportSummary_ok :
    checkExportSummary exportSummary = true := by
  decide +kernel

/-- The small-model configuration saved by the training run (`config.json`). -/
def trainedConfig : SmallGPTConfig where
  vocabSize := 32
  maxSeqLen := 6
  padTokenId := 0
  bosTokenId := 1
  dModel := 16
  nLayers := 2
  nHeads := 1
  dMlp := 64
  normVariant := "signed_l1_band_norm"
  attnVariant := "sparsemax"
  activationVariant := "leaky_relu"
  normL1LowPerDim := 55 / 100
  normL1HighPerDim := 105 / 100
  leakyReluNegativeSlope := 1 / 100
  tieEmbeddings := false
  useBias := true
  dropout := 0
  attnPdrop := 0
  residPdrop := 0
  embdPdrop := 0

/- The run's saved configuration is the one transcribed in `Spec.UpstreamSmallGPT`. -/
theorem trainedConfig_eq_repoDefault :
    trainedConfig = repoDefaultSmallConfig := by
  decide +kernel

end VerifiableTransformers.Generated.UpstreamExportSummary
