/-
Generated summary certificate for:

  artifacts/upstream-small-gpt/smt_weights.json

Lean records the metadata and tensor-shape summaries here, while the full
floating-point arrays live in the checkpoint values file. The SHA-256 field ties
this summary to the exact exported checkpoint data.
-/

import VerifiableTransformers.Certificate.Weights

namespace VerifiableTransformers.Generated.UpstreamExportSummary

open VerifiableTransformers.Certificate.Weights

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
  sha256 := "a647e7e07512596b70b2b41740ab53e1c1fa94e1fce5d13eceb8d86caec61e88"
  fieldCount := 48
  vocabSize := 32
  maxSeqLen := 6
  dModel := 16
  nLayers := 2
  nHeads := 1
  dMlp := 64
  headDim := 16
  normVariant := .signedL1BandNorm
  attnVariant := .sparsemax
  activationVariant := .leakyRelu
  halfLow := 22 / 5
  halfHigh := 42 / 5
  tensorFields := tensorFields

/- The imported export summary passes the checker. -/
theorem exportSummary_ok :
    checkExportSummary exportSummary = true := by
  simp [checkExportSummary, exportSummary, tensorFields,
    tensor, shapeEntries, TensorSummary.wellShaped,
    tensorEntryCount, repoDefaultTensorFields]

end VerifiableTransformers.Generated.UpstreamExportSummary
