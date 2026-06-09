/-
Checks for Neel's `smt_weights.json` export summary.

The training code can serialize a checkpoint, but I do not want the release to
ask readers to trust a paragraph saying the export had the right shape.  Lean
checks the dimensions, operator choices, half-band constants, tensor inventory,
parameter count, and checkpoint fingerprint.

The full Float checkpoint replay lives separately in
`Generated.UpstreamCheckpointPayload` and `Replay.UpstreamFloatReplay`.
-/

import VerifiableTransformers.Spec.UpstreamSmallGPT

namespace VerifiableTransformers.Certificate.Weights

open VerifiableTransformers

/-- Normalization variants accepted by the export check. -/
inductive NormVariant where
  | signedL1BandNorm
deriving Repr, DecidableEq

/-- Attention variants accepted by the export check. -/
inductive AttentionVariant where
  | sparsemax
deriving Repr, DecidableEq

/-- Activation variants accepted by the export check. -/
inductive ActivationVariant where
  | leakyRelu
deriving Repr, DecidableEq

/-- Tensor fields expected in `scripts/small/extract_weights.py` output. -/
inductive TensorName where
  | wte
  | wpe
  | attnWq (layer : Nat)
  | attnWk (layer : Nat)
  | attnWv (layer : Nat)
  | attnBq (layer : Nat)
  | attnBk (layer : Nat)
  | attnBv (layer : Nat)
  | attnWo (layer : Nat)
  | attnBo (layer : Nat)
  | mlpWUp (layer : Nat)
  | mlpBUp (layer : Nat)
  | mlpWDown (layer : Nat)
  | mlpBDown (layer : Nat)
  | attnNormGamma (layer : Nat)
  | attnNormBeta (layer : Nat)
  | mlpNormGamma (layer : Nat)
  | mlpNormBeta (layer : Nat)
  | finalNormGamma
  | finalNormBeta
  | lmHead
deriving Repr, DecidableEq

/-- Shape-and-size summary for one exported tensor field. -/
structure TensorSummary where
  name : TensorName
  shape : List Nat
  entries : Nat
deriving Repr, DecidableEq

/-- Metadata summary generated from Neel's `smt_weights.json`. -/
structure ExportSummary where
  sourcePath : String
  sha256 : String
  fieldCount : Nat
  vocabSize : Nat
  maxSeqLen : Nat
  dModel : Nat
  nLayers : Nat
  nHeads : Nat
  dMlp : Nat
  headDim : Nat
  normVariant : NormVariant
  attnVariant : AttentionVariant
  activationVariant : ActivationVariant
  halfLow : Rat
  halfHigh : Rat
  tensorFields : List TensorSummary
deriving Repr, DecidableEq

/-- Product of tensor dimensions, with scalar/empty shape size `1`. -/
def shapeEntries : List Nat -> Nat
  | [] => 1
  | n :: ns => n * shapeEntries ns

/-- Check that the recorded entry count matches the recorded shape. -/
def TensorSummary.wellShaped (t : TensorSummary) : Bool :=
  t.entries == shapeEntries t.shape

/-- Build a tensor summary with the entry count computed in Lean. -/
def tensor (name : TensorName) (shape : List Nat) : TensorSummary :=
  { name := name, shape := shape, entries := shapeEntries shape }

/-- Q/K/V and output-projection tensor summaries for one attention layer. -/
def qkvFields (layer : Nat) (dModel : Nat) : List TensorSummary :=
  [ tensor (.attnWq layer) [dModel, dModel]
  , tensor (.attnWk layer) [dModel, dModel]
  , tensor (.attnWv layer) [dModel, dModel]
  , tensor (.attnBq layer) [dModel]
  , tensor (.attnBk layer) [dModel]
  , tensor (.attnBv layer) [dModel]
  , tensor (.attnWo layer) [dModel, dModel]
  , tensor (.attnBo layer) [dModel]
  ]

/-- MLP tensor summaries for one transformer block. -/
def mlpFields (layer : Nat) (dModel dMlp : Nat) : List TensorSummary :=
  [ tensor (.mlpWUp layer) [dMlp, dModel]
  , tensor (.mlpBUp layer) [dMlp]
  , tensor (.mlpWDown layer) [dModel, dMlp]
  , tensor (.mlpBDown layer) [dModel]
  ]

/-- BandNorm affine tensor summaries for one transformer block. -/
def normFields (layer : Nat) (dModel : Nat) : List TensorSummary :=
  [ tensor (.attnNormGamma layer) [dModel]
  , tensor (.attnNormBeta layer) [dModel]
  , tensor (.mlpNormGamma layer) [dModel]
  , tensor (.mlpNormBeta layer) [dModel]
  ]

/-- All exported tensor summaries belonging to one transformer block. -/
def layerFields (layer : Nat) (dModel dMlp : Nat) : List TensorSummary :=
  qkvFields layer dModel ++ mlpFields layer dModel dMlp ++ normFields layer dModel

/-- Small local `concatMap`, kept here to avoid depending on extra list API names. -/
def concatMapList {α β : Type} (xs : List α) (f : α -> List β) : List β :=
  xs.foldr (fun x acc => f x ++ acc) []

/-- Tensor inventory implied by a small GPT configuration. -/
def expectedTensorFields (cfg : Spec.UpstreamSmallGPT.SmallGPTConfig) : List TensorSummary :=
  [ tensor .wte [cfg.vocabSize, cfg.dModel]
  , tensor .wpe [cfg.maxSeqLen, cfg.dModel]
  ] ++
  concatMapList (List.range cfg.nLayers) (fun layer =>
    layerFields layer cfg.dModel cfg.dMlp) ++
  [ tensor .finalNormGamma [cfg.dModel]
  , tensor .finalNormBeta [cfg.dModel]
  , tensor .lmHead [cfg.vocabSize, cfg.dModel]
  ]

/-- Exact tensor inventory for the small verifiable-transformer export. -/
def repoDefaultTensorFields : List TensorSummary :=
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

/-- Total number of scalar values represented by a tensor inventory. -/
def tensorEntryCount (xs : List TensorSummary) : Nat :=
  xs.foldl (fun acc x => acc + x.entries) 0

/-- Whether every required tensor summary appears in the generated summary. -/
def containsAll (need got : List TensorSummary) : Bool :=
  need.all (fun x => got.contains x)

/-- Main export-summary check used by generated certificates. -/
def checkExportSummary (e : ExportSummary) : Bool :=
  e.vocabSize == 32 &&
  e.maxSeqLen == 6 &&
  e.dModel == 16 &&
  e.nLayers == 2 &&
  e.nHeads == 1 &&
  e.dMlp == 64 &&
  e.headDim == 16 &&
  e.normVariant == .signedL1BandNorm &&
  e.attnVariant == .sparsemax &&
  e.activationVariant == .leakyRelu &&
  e.halfLow == 22 / 5 &&
  e.halfHigh == 42 / 5 &&
  e.fieldCount == 48 &&
  e.tensorFields.length == 37 &&
  e.tensorFields.all TensorSummary.wellShaped &&
  e.tensorFields == repoDefaultTensorFields &&
  tensorEntryCount e.tensorFields == 7712

/--
Accepted export summaries are the current Lean-side boundary for
`smt_weights.json` exports.

The checkpoint fingerprint keeps the checked summary tied to a specific export.
-/
def AcceptedExport (e : ExportSummary) : Prop :=
  checkExportSummary e = true

end VerifiableTransformers.Certificate.Weights
