/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.CausalTraining
public import TorchLeanGPT.Batching
public import TorchLeanGPT.Validation
public import TorchLeanGPT.DialogueRecords
public import LeanProfiler
public import NN.Spec.Core.Tensor.Constructors

/-!
# Training and completion runner

`train_torchlean_gpt` reads the little-endian `uint16` token shards written by
`tools/prepare_dataset.py`, plus its target masks and dialogue records for instruction tuning. The
format is plain: every GPT-2 token id fits in 16 bits, and a shard holds no executable code or
Python object serialization.

A run writes:

* a shape-checked TorchLean parameter checkpoint, when `--save-params` is supplied
* resumable checkpoints (parameters, AdamW moments, metrics, manifest), when `--checkpoint-dir`
  and `--checkpoint-every` are supplied
* a metric file with the observed training and validation losses
* a passport with the model dimensions, parameter count, backend profile, data paths, and the
  Lean theorem names that apply

Set `LEAN_PROFILE=1` to add a LeanProfiler trace and summary. Without it, no profiling events are
kept.
-/

@[expose] public section

open TorchLean

namespace TorchLeanGPT
namespace Run

/-- Name used in diagnostics. -/
def exeName : String := "train_torchlean_gpt"

/-- Default data directory (relative to the repository root) of the dataset preparation script. -/
def defaultDataDir : System.FilePath :=
  "week-03-gpt-training/data/tinyshakespeare"

/--
Default repeat penalty of every Week 3 generator (`--repeat-penalty` overrides it).

Before greedy or top-k selection, each token's score drops by this value once per occurrence among
the last `defaultRepeatWindow` ids, prompt included. The recorded generations in this example use
these defaults.
-/
def defaultRepeatPenalty : Float := 1.05

/-- Number of recent token ids, prompt included, seen by the repeat penalty. -/
def defaultRepeatWindow : Nat := 64

/-- One row of the metric file: a loss with its phase, step, and optional rate and timing. -/
structure MetricPoint where
  phase : String
  step : Nat
  loss : Float
  learningRate : Option Float := none
  durationMs : Option Nat := none
  tokensPerSecond : Option Float := none
  deriving Repr

/-- Command-line settings, after applying a preset and any flag overrides. -/
structure RunConfig where
  presetName : String
  model : ModelConfig
  batch : Nat
  steps : Nat
  tokenBudget? : Option Nat
  learningRate : Float
  minLearningRate : Float
  warmupSteps : Nat
  weightDecay : Float
  evalEvery : Nat
  evalBatches : Nat
  seed : Nat
  trainBin : System.FilePath
  valBin : System.FilePath
  trainMask? : Option System.FilePath
  valMask? : Option System.FilePath
  trainRecords? : Option System.FilePath
  valRecords? : Option System.FilePath
  loadParams? : Option System.FilePath
  saveParams? : Option System.FilePath
  resume? : Option System.FilePath
  checkpointDir? : Option System.FilePath
  checkpointEvery : Nat
  metricsPath : System.FilePath
  passportPath : System.FilePath
  tokenizerVocab? : Option System.FilePath
  tokenizerMerges? : Option System.FilePath
  prompt : String
  generate : Nat
  temperature : Float
  topK : Nat
  repeatPenalty : Float
  repeatWindow : Nat
  deriving Repr

/-- Command-line help for training, evaluation, checkpointing, and completion. -/
def usage : String :=
  String.intercalate "\n"
    [ "Train a GPT-style causal Transformer in TorchLean."
    , ""
    , "Usage:"
    , "  lake -R -K cuda=true exe train_torchlean_gpt --device cuda [options]"
    , ""
    , "Main options:"
    , "  --preset quick|gpt2-small    model preset (default: quick)"
    , "  --train-bin PATH             little-endian uint16 training tokens"
    , "  --val-bin PATH               little-endian uint16 validation tokens"
    , "  --train-mask PATH            optional uint8 target mask for weighted loss"
    , "  --val-mask PATH              paired validation target mask"
    , "  --train-records PATH         bounded dialogue windows and target intervals"
    , "  --val-records PATH           paired validation dialogue records"
    , "  --steps N                    optimizer updates"
    , "  --token-budget N             derive updates from a total token budget"
    , "  --batch N                    windows per update"
    , "  --load-params PATH           load an exact-bit TorchLean checkpoint"
    , "  --save-params PATH           save final parameters"
    , "  --resume PATH                resume parameters, AdamW moments, and global step"
    , "  --checkpoint-dir PATH        directory for resumable training checkpoints"
    , "  --checkpoint-every N         save every N completed updates (0 disables)"
    , ""
    , "Model dimensions:"
    , "  --context N --vocab N --width N --heads N --layers N --dropout P"
    , ""
    , "Evaluation and output:"
    , "  --eval-every N --eval-batches N --lr X --min-lr X --warmup-steps N"
    , "  --weight-decay X --seed N"
    , "  --metrics PATH --passport PATH"
    , "  --tokenizer-vocab PATH --tokenizer-merges PATH"
    , "  --prompt TEXT --generate N --temperature X --top-k N"
    , "  --repeat-penalty X --repeat-window N"
    , "      subtract X per occurrence among the last N ids, prompt included"
    , "      (default 1.05 over 64 ids; --repeat-penalty 0 disables it)"
    , ""
    , "Set LEAN_PROFILE=1 to write LeanProfiler trace and summary JSON files."
    ]

/-- Reject non-finite command-line values before they reach training or sampling. -/
def requireFinite (flag : String) (value : Float) : Except String Unit :=
  if value.isNaN || value.isInf then
    .error s!"--{flag} must be finite"
  else
    .ok ()

/-- Read the model-dimension flags, defaulting each to `base`, and validate the result. -/
def parseModelConfigFor (command : String) (args : List String) (base : ModelConfig) :
    IO (ModelConfig × List String) := do
  let (context, args) ← CLI.orThrow command <|
    CLI.takePositiveNatFlag args command "context" base.context
  let (vocab, args) ← CLI.orThrow command <|
    CLI.takePositiveNatFlag args command "vocab" base.vocab
  let (width, args) ← CLI.orThrow command <|
    CLI.takePositiveNatFlag args command "width" base.width
  let (heads, args) ← CLI.orThrow command <|
    CLI.takePositiveNatFlag args command "heads" base.heads
  let (layers, args) ← CLI.orThrow command <|
    CLI.takePositiveNatFlag args command "layers" base.layers
  let (dropout, args) ← CLI.orThrow command <|
    CLI.takeNonnegativeFloatFlag args command "dropout" base.dropout
  let model : ModelConfig :=
    { context, vocab, width, heads, layers, dropout }
  CLI.orThrow command model.validate
  pure (model, args)

/-- Parse training, artifact, tokenizer, and generation options. -/
def RunConfig.parse (seed : Nat) (args : List String) : IO RunConfig := do
  let (presetName, args) ← CLI.orThrow exeName <|
    CLI.takeFlagValue args "preset" "quick"
  let preset ← CLI.orThrow exeName <| ModelConfig.ofName presetName
  let (model, args) ← parseModelConfigFor exeName args preset
  let defaultSteps := if presetName.toLower == "gpt2-small" then 1000 else 2
  let defaultBatch := if presetName.toLower == "gpt2-small" then 4 else 2
  let (steps?, args) ← CLI.orThrow exeName <| CLI.takeNatFlag? args "steps"
  let (tokenBudget?, args) ← CLI.orThrow exeName <| CLI.takeNatFlag? args "token-budget"
  let (batch, args) ← CLI.orThrow exeName <|
    CLI.takePositiveNatFlag args exeName "batch" defaultBatch
  if steps?.isSome && tokenBudget?.isSome then
    throw <| IO.userError s!"{exeName}: pass either --steps or --token-budget, not both"
  if tokenBudget? == some 0 then
    throw <| IO.userError s!"{exeName}: --token-budget must be positive"
  let tokensPerStep := batch * model.context
  let steps :=
    match tokenBudget? with
    | some budget => (budget + tokensPerStep - 1) / tokensPerStep
    | none => steps?.getD defaultSteps
  let defaultLearningRate := if presetName.toLower == "gpt2-small" then 6e-4 else 3e-4
  let (learningRate, args) ← CLI.orThrow exeName <|
    CLI.takePositiveFloatFlag args exeName "lr" defaultLearningRate
  let (minLearningRate, args) ← CLI.orThrow exeName <|
    CLI.takeNonnegativeFloatFlag args exeName "min-lr" (learningRate * 0.1)
  let defaultWarmupSteps :=
    if presetName.toLower == "gpt2-small" then
      min 2000 (steps / 10)
    else
      0
  let (warmupSteps, args) ← CLI.orThrow exeName <|
    CLI.takeNatFlag args "warmup-steps" defaultWarmupSteps
  let (weightDecay, args) ← CLI.orThrow exeName <|
    CLI.takeNonnegativeFloatFlag args exeName "weight-decay" 0.1
  let (evalEvery, args) ← CLI.orThrow exeName <|
    CLI.takeNatFlag args "eval-every" (if steps <= 10 then 1 else 100)
  let (evalBatches, args) ← CLI.orThrow exeName <|
    CLI.takePositiveNatFlag args exeName "eval-batches" 4
  let (trainBin, args) ← CLI.orThrow exeName <|
    CLI.takePathFlag args "train-bin" (defaultDataDir / "train.bin")
  let (valBin, args) ← CLI.orThrow exeName <|
    CLI.takePathFlag args "val-bin" (defaultDataDir / "val.bin")
  let ((trainMask?, valMask?), args) ← CLI.orThrow exeName <|
    CLI.takePairedPathFlags args "train-mask" "val-mask"
  let ((trainRecords?, valRecords?), args) ← CLI.orThrow exeName <|
    CLI.takePairedPathFlags args "train-records" "val-records"
  if trainMask?.isSome != trainRecords?.isSome then
    throw <| IO.userError <|
      s!"{exeName}: target masks and dialogue records must be supplied together"
  let (loadParams?, args) ← CLI.orThrow exeName <| CLI.takePathFlag? args "load-params"
  let (saveParams?, args) ← CLI.orThrow exeName <| CLI.takePathFlag? args "save-params"
  let (resume?, args) ← CLI.orThrow exeName <| CLI.takePathFlag? args "resume"
  let (checkpointDir?, args) ← CLI.orThrow exeName <| CLI.takePathFlag? args "checkpoint-dir"
  let (checkpointEvery, args) ← CLI.orThrow exeName <| CLI.takeNatFlag args "checkpoint-every" 0
  if loadParams?.isSome && resume?.isSome then
    throw <| IO.userError s!"{exeName}: --load-params and --resume are mutually exclusive"
  if checkpointEvery != 0 && checkpointDir?.isNone then
    throw <| IO.userError s!"{exeName}: --checkpoint-every requires --checkpoint-dir"
  let (metricsPath, args) ← CLI.orThrow exeName <|
    CLI.takePathFlag args "metrics"
      "week-03-gpt-training/artifacts/training-metrics.json"
  let (passportPath, args) ← CLI.orThrow exeName <|
    CLI.takePathFlag args "passport"
      "week-03-gpt-training/artifacts/run-passport.json"
  let ((tokenizerVocab?, tokenizerMerges?), args) ← CLI.orThrow exeName <|
    CLI.takePairedPathFlags args "tokenizer-vocab" "tokenizer-merges"
  let (prompt, args) ← CLI.orThrow exeName <|
    CLI.takeFlagValue args "prompt" "The meaning of verification is"
  let (generate, args) ← CLI.orThrow exeName <| CLI.takeNatFlag args "generate" 0
  let (temperature, args) ← CLI.orThrow exeName <|
    CLI.takePositiveFloatFlag args exeName "temperature" 0.8
  let (topK, args) ← CLI.orThrow exeName <| CLI.takeNatFlag args "top-k" 40
  let (repeatPenalty, args) ← CLI.orThrow exeName <|
    CLI.takeNonnegativeFloatFlag args exeName "repeat-penalty" defaultRepeatPenalty
  let (repeatWindow, args) ← CLI.orThrow exeName <|
    CLI.takeNatFlag args "repeat-window" defaultRepeatWindow
  CLI.orThrow exeName <| CLI.checkNoArgs args
  CLI.orThrow exeName <| requireFinite "lr" learningRate
  CLI.orThrow exeName <| requireFinite "min-lr" minLearningRate
  if minLearningRate > learningRate then
    throw <| IO.userError s!"{exeName}: --min-lr cannot exceed --lr"
  CLI.orThrow exeName <| requireFinite "weight-decay" weightDecay
  CLI.orThrow exeName <| requireFinite "temperature" temperature
  pure
    { presetName
      model
      batch
      steps
      tokenBudget?
      learningRate
      minLearningRate
      warmupSteps
      weightDecay
      evalEvery
      evalBatches
      seed
      trainBin
      valBin
      trainMask?
      valMask?
      trainRecords?
      valRecords?
      loadParams?
      saveParams?
      resume?
      checkpointDir?
      checkpointEvery
      metricsPath
      passportPath
      tokenizerVocab?
      tokenizerMerges?
      prompt
      generate
      temperature
      topK
      repeatPenalty
      repeatWindow }

/-- Read one little-endian `uint16`, returning zero for an incomplete trailing word. -/
def readUInt16LE (bytes : ByteArray) (offset : Nat) : Nat :=
  if hHigh : offset + 1 < bytes.size then
    have hLow : offset < bytes.size := Nat.lt_trans (Nat.lt_succ_self offset) hHigh
    (bytes.get offset hLow).toNat + 256 * (bytes.get (offset + 1) hHigh).toNat
  else
    0

/-- Token shard kept as its on-disk bytes, with a content hash. -/
structure TokenShard where
  bytes : ByteArray
  contentHash : UInt64

/-- Number of complete `uint16` token ids in a compact shard. -/
def TokenShard.size (shard : TokenShard) : Nat :=
  shard.bytes.size / 2

/-- Read one token id, or `fallback` beyond the end of the shard. -/
def TokenShard.getD (shard : TokenShard) (index fallback : Nat) : Nat :=
  if index < shard.size then
    readUInt16LE shard.bytes (2 * index)
  else
    fallback

/-- Byte-per-token target mask, kept as bytes rather than boxed booleans. -/
structure TargetMask where
  bytes : ByteArray
  contentHash : UInt64

/-- Number of mask entries. -/
def TargetMask.size (mask : TargetMask) : Nat :=
  mask.bytes.size

/-- Read one target-mask bit, defaulting to `false` beyond the mask. -/
def TargetMask.getD (mask : TargetMask) (index : Nat) : Bool :=
  if h : index < mask.size then mask.bytes.get index h != 0 else false

/-- One dialogue record: a contiguous window of a token shard and its assistant-target interval. -/
structure DialogueRecord where
  offset : Nat
  length : Nat
  targetOffset : Nat
  targetLength : Nat
  deriving Repr, DecidableEq, Inhabited

/-- Dialogue records and their content identity. -/
structure DialogueRecords where
  entries : Array DialogueRecord
  contentHash : UInt64

/-- Read one little-endian `uint64`, returning zero for an incomplete trailing word. -/
def readUInt64LE (bytes : ByteArray) (offset : Nat) : Nat :=
  Id.run do
    let mut value : UInt64 := 0
    for i in [0:8] do
      if h : offset + i < bytes.size then
        value := value ||| (UInt64.ofNat (bytes.get (offset + i) h).toNat <<< UInt64.ofNat (8 * i))
    return value.toNat

/--
Load dialogue records, rejecting any that leave the token shard, have invalid targets, or exceed
`context + 1` tokens.

The writer limits records to `context + 1` tokens (one input window plus its final next-token
target). Rejecting longer records here stops a run with a smaller context from silently truncating
an example.
-/
def readDialogueRecords
    (path : System.FilePath) (shardTokens context : Nat) : IO DialogueRecords := do
  let bytes ← IO.FS.readBinFile path
  if bytes.size % 32 != 0 then
    throw <| IO.userError s!"{exeName}: {path} is not a sequence of four-uint64 records"
  let mut entries := Array.mkEmpty (bytes.size / 32)
  for i in [0:bytes.size / 32] do
    let offset := readUInt64LE bytes (32 * i)
    let length := readUInt64LE bytes (32 * i + 8)
    let targetOffset := readUInt64LE bytes (32 * i + 16)
    let targetLength := readUInt64LE bytes (32 * i + 24)
    if length < 2 then
      throw <| IO.userError s!"{exeName}: dialogue record {i} in {path} has fewer than two tokens"
    if length > context + 1 then
      throw <| IO.userError <|
        s!"{exeName}: dialogue record {i} in {path} has {length} tokens, " ++
          s!"exceeding this model's {context + 1}-token record limit"
    if offset > shardTokens || length > shardTokens - offset then
      throw <| IO.userError s!"{exeName}: dialogue record {i} in {path} leaves the token shard"
    if targetLength = 0 || targetOffset <= offset || targetOffset > offset + length ||
        targetLength > offset + length - targetOffset then
      throw <| IO.userError s!"{exeName}: dialogue record {i} in {path} has invalid targets"
    entries := entries.push { offset, length, targetOffset, targetLength }
  if entries.isEmpty then
    throw <| IO.userError s!"{exeName}: {path} contains no dialogue records"
  pure { entries, contentHash := hash bytes }

/--
Check that the record target intervals, taken in order, partition the active target mask.

Every recorded target must be active, intervals must be ordered and disjoint, and their total length
must equal the number of active mask bytes. So no assistant target is dropped, counted twice, or
invented on its way into the weighted objective.
-/
def validateDialogueRecordTargets
    (path : System.FilePath) (records : DialogueRecords) (targetMask : TargetMask) : IO Unit := do
  let mut previousTargetEnd := 0
  let mut recordedTargets := 0
  for i in [0:records.entries.size] do
    let record := records.entries[i]!
    if record.targetOffset < previousTargetEnd then
      throw <| IO.userError s!"{exeName}: target intervals overlap or are unordered in {path}"
    for j in [0:record.targetLength] do
      if !targetMask.getD (record.targetOffset + j) then
        throw <| IO.userError <|
          s!"{exeName}: record {i} in {path} names an inactive target at " ++
            s!"{record.targetOffset + j}"
    previousTargetEnd := record.targetOffset + record.targetLength
    recordedTargets := recordedTargets + record.targetLength
  let activeTargets := targetMask.bytes.foldl
    (fun count byte => if byte = 0 then count else count + 1) 0
  if recordedTargets != activeTargets then
    throw <| IO.userError <|
      s!"{exeName}: {path} covers {recordedTargets} targets, but the mask has {activeTargets}"

/-- Token counts and content hashes of the shards, masks, and records used by one run. -/
structure DatasetIdentity where
  trainTokens : Nat
  validationTokens : Nat
  trainHash : UInt64
  validationHash : UInt64
  trainMaskHash? : Option UInt64
  validationMaskHash? : Option UInt64
  trainRecordsHash? : Option UInt64
  validationRecordsHash? : Option UInt64

/--
Load a raw little-endian `uint16` token shard.

Files with an odd byte count and ids outside the model vocabulary are rejected before any tensor
indexing.
-/
def readTokenShard (path : System.FilePath) (vocab : Nat) : IO TokenShard := do
  let bytes ← IO.FS.readBinFile path
  if bytes.size % 2 != 0 then
    throw <| IO.userError s!"{exeName}: {path} has an odd byte count"
  if let some i := Validation.firstInvalid (bytes.size / 2)
      (fun i => readUInt16LE bytes (2 * i) < vocab) then
    let token := readUInt16LE bytes (2 * i)
    throw <| IO.userError
      s!"{exeName}: token id {token} in {path} is outside vocabulary [0, {vocab})"
  pure { bytes, contentHash := hash bytes }

/--
Load a byte-per-token target mask, rejecting any byte other than `0` or `1`.

Each sampled batch gets its loss normalization from the Lean runner (`activeRowWeight`), not from
a Python preprocessing convention.
-/
def readTargetMask (path : System.FilePath) : IO TargetMask := do
  let bytes ← IO.FS.readBinFile path
  if let some i := Validation.firstInvalid bytes.size (fun i => bytes[i]!.toNat ≤ 1) then
    let value := bytes[i]!.toNat
    throw <| IO.userError
      s!"{exeName}: target mask {path} contains byte {value} at offset {i}; expected 0 or 1"
  pure { bytes, contentHash := hash bytes }

/--
Build one deterministic next-token batch, reading ids from the shard bytes instead of expanding
the corpus into an `Array Nat`.
-/
def causalLmTokenBatchFromShard
    (vocab batch seqLen : Nat) [NeZero vocab]
    (shard : TokenShard) (seed step : Nat) (padId : Nat := 0) :
    Tensor (Fin vocab) [batch, seqLen] × Tensor (Fin vocab) [batch, seqLen] :=
  let offsetAt := text.Corpus.randomBatchOffsets shard.size seqLen batch seed step
  (Tensor.dim fun bi => Tensor.ofFn fun i =>
      Fin.ofNat vocab (shard.getD (offsetAt.getScalar bi + i.val) padId),
    Tensor.dim fun bi => Tensor.ofFn fun i =>
      Fin.ofNat vocab (shard.getD (offsetAt.getScalar bi + i.val + 1) padId))

/-- The dialogue record sampled for one batch row at one optimizer step. -/
def sampledRecord (records : DialogueRecords) (seed step batchIndex : Nat) : DialogueRecord :=
  records.entries[Spec.Random.sampleNat
    (Spec.Random.keyOf seed step) batchIndex records.entries.size]!

/--
Input token, next-token target, and loss-mask bit for row `row` of one dialogue record.

Both the target and the mask bit are read at `DialogueRecords.targetIndex record.offset row`, the
corpus position after the input. Rows at or after the end of the record are padding.
-/
def recordRow (shard : TokenShard) (targetMask : TargetMask) (padId : Nat)
    (record : DialogueRecord) (row : Nat) : Nat × Nat × Bool :=
  if row + 1 < record.length then
    let targetIndex := TorchLeanGPT.DialogueRecords.targetIndex record.offset row
    let inRecordTargets := TorchLeanGPT.DialogueRecords.targetRowEnabled
      record.offset record.targetOffset record.targetLength row
    (shard.getD (record.offset + row) padId,
      shard.getD targetIndex padId, inRecordTargets && targetMask.getD targetIndex)
  else
    (padId, padId, false)

/-- Row `k` of a dialogue batch in row-major order: row `k % seqLen` of batch entry `k / seqLen`. -/
def recordBatchRow (shard : TokenShard) (targetMask : TargetMask) (records : DialogueRecords)
    (seed step padId seqLen k : Nat) : Nat × Nat × Bool :=
  recordRow shard targetMask padId (sampledRecord records seed step (k / seqLen)) (k % seqLen)

/-- All rows of a dialogue batch in row-major order, packed by `Batching.appendRows`. -/
def recordBatchRows (shard : TokenShard) (targetMask : TargetMask) (records : DialogueRecords)
    (seed step padId batch seqLen : Nat) : Array Nat × Array Nat × Array Bool :=
  Batching.appendRows (batch * seqLen)
    (recordBatchRow shard targetMask records seed step padId seqLen) (#[], #[], #[])

/-- Weight shared by every enabled row: one over the number of enabled rows, or zero if none. -/
def activeRowWeight (enabled : Array Bool) : Float :=
  let activeCount := enabled.foldl (fun count active =>
    if active then count + 1 else count) 0
  if activeCount = 0 then 0 else 1.0 / Float.ofNat activeCount

/-- Build a deterministic dialogue-bounded batch from compact token, mask, and record shards. -/
def causalLmMaskedTokenBatchFromRecords
    {α : Type} [Storage α] [_root_.Context α] [Runtime.FromFloat α]
    (vocab batch seqLen : Nat) [NeZero vocab]
    (shard : TokenShard) (targetMask : TargetMask)
    (records : DialogueRecords)
    (seed step : Nat) (padId : Nat := 0) :
    Tensor (Fin vocab) [batch, seqLen] ×
      Tensor (Fin vocab) [batch, seqLen] × Tensor α [batch, seqLen] :=
  let rows := recordBatchRows shard targetMask records seed step padId batch seqLen
  let activeWeight := activeRowWeight rows.2.2
  let rowWeights := rows.2.2.map (fun active => if active then activeWeight else 0)
  (Tensor.dim fun bi => Tensor.ofFn fun i =>
      Fin.ofNat vocab (rows.1.getD (bi.val * seqLen + i.val) padId),
    Tensor.dim fun bi => Tensor.ofFn fun i =>
      Fin.ofNat vocab (rows.2.1.getD (bi.val * seqLen + i.val) padId),
    Tensor.dim fun bi => Tensor.ofFn fun i =>
      Runtime.ofFloat (rowWeights.getD (bi.val * seqLen + i.val) 0.0))

/-! ## Targets and loss masks of the batches the trainer builds -/

/-- Every entry of a pretraining batch is read from the shard at its window offset. -/
theorem causalLmTokenBatchFromShard_get
    (vocab batch seqLen : Nat) [NeZero vocab]
    (shard : TokenShard) (seed step padId : Nat) (bi : Fin batch) (i : Fin seqLen) :
    let batchTensors := causalLmTokenBatchFromShard vocab batch seqLen shard seed step padId
    let start := (text.Corpus.randomBatchOffsets shard.size seqLen batch seed step).getScalar bi
    (batchTensors.1.get bi).getScalar i = Fin.ofNat vocab (shard.getD (start + i.val) padId) ∧
      (batchTensors.2.get bi).getScalar i =
        Fin.ofNat vocab (shard.getD (start + i.val + 1) padId) := by
  simp only [causalLmTokenBatchFromShard, Spec.get_dim, Tensor.getScalar_ofFn, and_self]

/-- Batch entry `bi`, row `i` is packed at array index `bi * seqLen + i`. -/
theorem recordBatchRows_getElem?
    (shard : TokenShard) (targetMask : TargetMask) (records : DialogueRecords)
    (seed step padId batch seqLen : Nat) (bi : Fin batch) (i : Fin seqLen) :
    let rows := recordBatchRows shard targetMask records seed step padId batch seqLen
    let row := recordRow shard targetMask padId (sampledRecord records seed step bi.val) i.val
    rows.1[bi.val * seqLen + i.val]? = some row.1 ∧
      rows.2.1[bi.val * seqLen + i.val]? = some row.2.1 ∧
      rows.2.2[bi.val * seqLen + i.val]? = some row.2.2 := by
  have hIndex : bi.val * seqLen + i.val < batch * seqLen := by
    have := Nat.mul_le_mul_right seqLen (Nat.succ_le_of_lt bi.isLt)
    rw [Nat.succ_mul] at this
    omega
  have hDiv : (bi.val * seqLen + i.val) / seqLen = bi.val := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt),
      Nat.div_eq_of_lt i.isLt, Nat.zero_add]
  have hMod : (bi.val * seqLen + i.val) % seqLen = i.val := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt i.isLt]
  have h := Batching.appendRows_getElem? (batch * seqLen)
    (recordBatchRow shard targetMask records seed step padId seqLen) _ hIndex
  rw [recordBatchRow, hDiv, hMod] at h
  exact h

/--
Row `i` of batch entry `bi` in a dialogue batch is row `i` of the record sampled for `bi`, and
its weight is the shared weight if the row's loss-mask bit is set.

Later proofs rewrite with this equation rather than make Lean's kernel evaluate `recordRow` on
symbolic records.
-/
theorem causalLmMaskedTokenBatchFromRecords_get
    {α : Type} [Storage α] [_root_.Context α] [Runtime.FromFloat α]
    (vocab batch seqLen : Nat) [NeZero vocab]
    (shard : TokenShard) (targetMask : TargetMask) (records : DialogueRecords)
    (seed step padId : Nat) (bi : Fin batch) (i : Fin seqLen) :
    let batchTensors := causalLmMaskedTokenBatchFromRecords (α := α)
      vocab batch seqLen shard targetMask records seed step padId
    let row := recordRow shard targetMask padId (sampledRecord records seed step bi.val) i.val
    let weight :=
      activeRowWeight (recordBatchRows shard targetMask records seed step padId batch seqLen).2.2
    (batchTensors.1.get bi).getScalar i = Fin.ofNat vocab row.1 ∧
      (batchTensors.2.1.get bi).getScalar i = Fin.ofNat vocab row.2.1 ∧
      (batchTensors.2.2.get bi).getScalar i =
        Runtime.ofFloat (if row.2.2 then weight else 0) := by
  obtain ⟨hInputs, hTargets, hEnabled⟩ :=
    recordBatchRows_getElem? shard targetMask records seed step padId batch seqLen bi i
  simp only [causalLmMaskedTokenBatchFromRecords, Spec.get_dim,
    Tensor.getScalar_ofFn, Array.getD_eq_getD_getElem?, Array.getElem?_map]
  rw [hInputs, hTargets, hEnabled, Option.getD_some, Option.getD_some, Option.map_some,
    Option.getD_some]
  exact ⟨rfl, rfl, rfl⟩

/-- A row inside its record reads its input and target from the shard. -/
theorem recordRow_of_lt (shard : TokenShard) (targetMask : TargetMask) (padId : Nat)
    (record : DialogueRecord) (row : Nat) (hInside : row + 1 < record.length) :
    recordRow shard targetMask padId record row =
      (shard.getD (record.offset + row) padId,
        shard.getD (TorchLeanGPT.DialogueRecords.targetIndex record.offset row) padId,
        TorchLeanGPT.DialogueRecords.targetRowEnabled
            record.offset record.targetOffset record.targetLength row &&
          targetMask.getD (TorchLeanGPT.DialogueRecords.targetIndex record.offset row)) := by
  rw [recordRow, ite_eq_left hInside]

/-- A row at or after the end of its record is padding with no loss weight. -/
theorem recordRow_of_le (shard : TokenShard) (targetMask : TargetMask) (padId : Nat)
    (record : DialogueRecord) (row : Nat) (hEnd : record.length ≤ row + 1) :
    recordRow shard targetMask padId record row = (padId, padId, false) := by
  rw [recordRow, ite_eq_right (Nat.not_lt.mpr hEnd)]

end Run

/--
Every target in a pretraining batch is the corpus token after its input.

`Run.causalLmTokenBatchFromShard` is the batch builder `train_torchlean_gpt` uses for ordinary
next-token training. Row `i` of batch entry `bi` predicts the shard token at `start + i + 1`, and
that token is also the input of row `i + 1`.
-/
theorem causal_window_target_is_next_token
    (vocab batch seqLen : Nat) [NeZero vocab]
    (shard : Run.TokenShard) (seed step padId : Nat) (bi : Fin batch) (i : Fin seqLen) :
    let batchTensors := Run.causalLmTokenBatchFromShard vocab batch seqLen shard seed step padId
    let start := (text.Corpus.randomBatchOffsets shard.size seqLen batch seed step).getScalar bi
    (batchTensors.2.get bi).getScalar i =
        Fin.ofNat vocab (shard.getD (start + i.val + 1) padId) ∧
      ∀ j : Fin seqLen, j.val = i.val + 1 →
        (batchTensors.2.get bi).getScalar i = (batchTensors.1.get bi).getScalar j := by
  intro batchTensors start
  have hRow := Run.causalLmTokenBatchFromShard_get vocab batch seqLen shard seed step padId bi i
  refine ⟨hRow.2, fun j hj => ?_⟩
  rw [hRow.2, (Run.causalLmTokenBatchFromShard_get vocab batch seqLen shard seed step padId bi j).1,
    hj, Nat.add_assoc]

/--
Every target inside a dialogue record is the corpus token after its input.

`Run.causalLmMaskedTokenBatchFromRecords` is the batch builder used for instruction tuning. A row
inside its sampled record predicts the token at `DialogueRecords.targetIndex record.offset i`,
which is also the input of the next row while that row is still inside the record. Rows at or after
the end of the record are padding in both tensors.
-/
theorem causal_record_target_is_next_token
    {α : Type} [Storage α] [_root_.Context α] [Runtime.FromFloat α]
    (vocab batch seqLen : Nat) [NeZero vocab]
    (shard : Run.TokenShard) (targetMask : Run.TargetMask) (records : Run.DialogueRecords)
    (seed step padId : Nat) (bi : Fin batch) (i : Fin seqLen) :
    let batchTensors := Run.causalLmMaskedTokenBatchFromRecords (α := α)
      vocab batch seqLen shard targetMask records seed step padId
    let record := Run.sampledRecord records seed step bi.val
    (i.val + 1 < record.length →
      (batchTensors.2.1.get bi).getScalar i =
          Fin.ofNat vocab (shard.getD (DialogueRecords.targetIndex record.offset i.val) padId) ∧
        ∀ j : Fin seqLen, j.val = i.val + 1 → j.val + 1 < record.length →
          (batchTensors.2.1.get bi).getScalar i = (batchTensors.1.get bi).getScalar j) ∧
      (record.length ≤ i.val + 1 →
        (batchTensors.1.get bi).getScalar i = Fin.ofNat vocab padId ∧
          (batchTensors.2.1.get bi).getScalar i = Fin.ofNat vocab padId) := by
  intro batchTensors record
  have hRow := Run.causalLmMaskedTokenBatchFromRecords_get (α := α)
    vocab batch seqLen shard targetMask records seed step padId bi i
  refine ⟨fun hInside => ⟨?_, fun j hj hNext => ?_⟩, fun hEnd => ?_⟩
  · rw [hRow.2.1, Run.recordRow_of_lt shard targetMask padId record i.val hInside]
  · have hNextRow := Run.causalLmMaskedTokenBatchFromRecords_get (α := α)
      vocab batch seqLen shard targetMask records seed step padId bi j
    rw [hRow.2.1, hNextRow.1, Run.recordRow_of_lt shard targetMask padId record i.val hInside,
      Run.recordRow_of_lt shard targetMask padId record j.val hNext, hj,
      DialogueRecords.targetIndex, Nat.add_assoc]
  · rw [hRow.1, hRow.2.1, Run.recordRow_of_le shard targetMask padId record i.val hEnd]
    exact ⟨rfl, rfl⟩

/--
Instruction-tuning loss weights follow the prediction target, not the input token.

For row `i` of a dialogue record, the weight reads the target-mask bit at
`DialogueRecords.targetIndex record.offset i`, the same corpus position that supplies the row's
target in `causal_record_target_is_next_token`. A row gets the shared batch weight
(`Run.activeRowWeight`) when it lies inside the record, inside the assistant-target interval, and
its target is marked. Every other row gets zero.
-/
theorem causal_window_mask_is_next_target
    {α : Type} [Storage α] [_root_.Context α] [Runtime.FromFloat α]
    (vocab batch seqLen : Nat) [NeZero vocab]
    (shard : Run.TokenShard) (targetMask : Run.TargetMask) (records : Run.DialogueRecords)
    (seed step padId : Nat) (bi : Fin batch) (i : Fin seqLen) :
    let batchTensors := Run.causalLmMaskedTokenBatchFromRecords (α := α)
      vocab batch seqLen shard targetMask records seed step padId
    let record := Run.sampledRecord records seed step bi.val
    let weight := Run.activeRowWeight
      (Run.recordBatchRows shard targetMask records seed step padId batch seqLen).2.2
    (batchTensors.2.2.get bi).getScalar i =
      Runtime.ofFloat
        (if decide (i.val + 1 < record.length) &&
            DialogueRecords.targetRowEnabled
              record.offset record.targetOffset record.targetLength i.val &&
            targetMask.getD (DialogueRecords.targetIndex record.offset i.val) then
          weight
        else
          0) := by
  intro batchTensors record weight
  rw [(Run.causalLmMaskedTokenBatchFromRecords_get (α := α)
    vocab batch seqLen shard targetMask records seed step padId bi i).2.2]
  by_cases hInside : i.val + 1 < record.length
  · rw [Run.recordRow_of_lt shard targetMask padId record i.val hInside,
      decide_eq_true hInside, Bool.true_and]
  · rw [Run.recordRow_of_le shard targetMask padId record i.val (Nat.not_lt.mp hInside),
      decide_eq_false hInside, Bool.false_and, Bool.false_and]

namespace Run

/-- Create the parent directory of an output file if it is missing. -/
def ensureParent (path : System.FilePath) : IO Unit :=
  match path.parent with
  | some parent => IO.FS.createDirAll parent
  | none => pure ()

/-- JSON encoding that preserves non-finite Float diagnostics as strings. -/
def floatJson (value : Float) : Lean.Json :=
  match Lean.JsonNumber.fromFloat? value with
  | .inr number => .num number
  | .inl spelling => .str spelling

/-- Name of the backend profile selected by the runtime options. -/
def backendProfileName (opts : Runtime.Config) : String :=
  match opts.resolveBackendProfile with
  | .ok profile => profile.name
  | .error _ => "unresolved"

/--
Decoding settings of a run's optional completion.

Metric files, passports, and checkpoint manifests all record them, so a "greedy" (`top_k = 1`)
completion is not mistaken for an unpenalized argmax.
-/
def generationJson (cfg : RunConfig) : Lean.Json :=
  Lean.Json.mkObj
    [ ("new_tokens", .num (Lean.JsonNumber.fromNat cfg.generate))
    , ("temperature", floatJson cfg.temperature)
    , ("top_k", .num (Lean.JsonNumber.fromNat cfg.topK))
    , ("repeat_penalty", floatJson cfg.repeatPenalty)
    , ("repeat_window", .num (Lean.JsonNumber.fromNat cfg.repeatWindow))
    , ("sampling_seed", .num (Lean.JsonNumber.fromNat cfg.seed))
    ]

/-- Write the observed losses and timings in a small stable JSON schema. -/
def writeMetrics (path : System.FilePath) (cfg : RunConfig) (points : Array MetricPoint) :
    IO Unit := do
  ensureParent path
  let rows := points.map fun point =>
    Lean.Json.mkObj
      [ ("phase", .str point.phase)
      , ("step", .num (Lean.JsonNumber.fromNat point.step))
      , ("loss", floatJson point.loss)
      , ("learning_rate",
          point.learningRate.map floatJson |>.getD .null)
      , ("duration_ms",
          point.durationMs.map (fun value => .num (Lean.JsonNumber.fromNat value)) |>.getD .null)
      , ("tokens_per_second",
          point.tokensPerSecond.map floatJson |>.getD .null)
      ]
  let artifact := Lean.Json.mkObj
    [ ("schema", .str "torchlean.gpt-training.metrics.v1")
    , ("dtype", .str "float64")
    , ("generation", generationJson cfg)
    , ("points", .arr rows)
    ]
  IO.FS.writeFile path artifact.pretty

/-! ## Resumable training checkpoints -/

/-- Run settings that determine the optimizer trajectory. Version 1 manifests store nothing else. -/
def checkpointConfigJsonV1 (cfg : RunConfig) : Lean.Json :=
  Lean.Json.mkObj
    [ ("preset", .str cfg.presetName)
    , ("context", .num (Lean.JsonNumber.fromNat cfg.model.context))
    , ("vocab", .num (Lean.JsonNumber.fromNat cfg.model.vocab))
    , ("width", .num (Lean.JsonNumber.fromNat cfg.model.width))
    , ("heads", .num (Lean.JsonNumber.fromNat cfg.model.heads))
    , ("layers", .num (Lean.JsonNumber.fromNat cfg.model.layers))
    , ("dropout_bits", .num (Lean.JsonNumber.fromNat cfg.model.dropout.toBits.toNat))
    , ("batch", .num (Lean.JsonNumber.fromNat cfg.batch))
    , ("steps", .num (Lean.JsonNumber.fromNat cfg.steps))
    , ("learning_rate_bits", .num (Lean.JsonNumber.fromNat cfg.learningRate.toBits.toNat))
    , ("minimum_learning_rate_bits",
        .num (Lean.JsonNumber.fromNat cfg.minLearningRate.toBits.toNat))
    , ("warmup_steps", .num (Lean.JsonNumber.fromNat cfg.warmupSteps))
    , ("weight_decay_bits", .num (Lean.JsonNumber.fromNat cfg.weightDecay.toBits.toNat))
    , ("seed", .num (Lean.JsonNumber.fromNat cfg.seed))
    , ("train_shard", .str cfg.trainBin.toString)
    , ("validation_shard", .str cfg.valBin.toString)
    , ("train_mask", cfg.trainMask?.map (fun p => .str p.toString) |>.getD .null)
    , ("validation_mask", cfg.valMask?.map (fun p => .str p.toString) |>.getD .null)
    , ("train_records", cfg.trainRecords?.map (fun p => .str p.toString) |>.getD .null)
    , ("validation_records", cfg.valRecords?.map (fun p => .str p.toString) |>.getD .null)
    ]

/-- Resume identity of new (v2) checkpoints: run settings, content hashes, and runtime. -/
def checkpointConfigJson
    (cfg : RunConfig) (opts : Runtime.Config) (data : DatasetIdentity) : Lean.Json :=
  Lean.Json.mkObj
    [ ("run", checkpointConfigJsonV1 cfg)
    , ("train_content_hash64",
        .num (Lean.JsonNumber.fromNat data.trainHash.toNat))
    , ("validation_content_hash64",
        .num (Lean.JsonNumber.fromNat data.validationHash.toNat))
    , ("train_mask_content_hash64",
        data.trainMaskHash?.map (fun value =>
          .num (Lean.JsonNumber.fromNat value.toNat)) |>.getD .null)
    , ("validation_mask_content_hash64",
        data.validationMaskHash?.map (fun value =>
          .num (Lean.JsonNumber.fromNat value.toNat)) |>.getD .null)
    , ("train_records_content_hash64",
        data.trainRecordsHash?.map (fun value =>
          .num (Lean.JsonNumber.fromNat value.toNat)) |>.getD .null)
    , ("validation_records_content_hash64",
        data.validationRecordsHash?.map (fun value =>
          .num (Lean.JsonNumber.fromNat value.toNat)) |>.getD .null)
    , ("device", .str opts.deviceName)
    , ("backend_profile", .str (backendProfileName opts))
    , ("lean_version", .str Lean.versionString)
    , ("optimizer", .str "TorchLean CUDA AdamW")
    ]

/-- Manifest committed last inside each complete checkpoint directory. -/
def checkpointManifest
    (cfg : RunConfig) (opts : Runtime.Config) (data : DatasetIdentity)
    (completedStep : Nat) : Lean.Json :=
  Lean.Json.mkObj
    [ ("schema", .str "torchlean.gpt-training.resume.v2")
    , ("completed_step", .num (Lean.JsonNumber.fromNat completedStep))
    , ("config", checkpointConfigJson cfg opts data)
    -- Decoding does not change the optimizer trajectory, so it stays outside the resume identity.
    , ("generation", generationJson cfg)
    ]

/-- Resolve either a concrete checkpoint directory or a root containing a `LATEST` pointer. -/
def resolveCheckpointDirectory (path : System.FilePath) : IO System.FilePath := do
  if ← (path / "manifest.json").pathExists then
    pure path
  else
    let latestPath := path / "LATEST"
    if !(← latestPath.pathExists) then
      throw <| IO.userError s!"{exeName}: no checkpoint manifest or LATEST pointer at {path}"
    let name := (String.trimAscii (← IO.FS.readFile latestPath)).toString
    if name.isEmpty then
      throw <| IO.userError s!"{exeName}: empty checkpoint pointer at {latestPath}"
    pure (path / name)

/-- Parse and validate a checkpoint manifest before mutating model or optimizer state. -/
def readCheckpointStep
    (cfg : RunConfig) (opts : Runtime.Config) (data : DatasetIdentity)
    (directory : System.FilePath) : IO Nat := do
  let path := directory / "manifest.json"
  let json ← match Lean.Json.parse (← IO.FS.readFile path) with
    | .ok value => pure value
    | .error message =>
        throw <| IO.userError s!"{exeName}: malformed checkpoint manifest {path}: {message}"
  let schema ← match json.getObjValAs? String "schema" with
    | .ok value => pure value
    | .error error => throw <| IO.userError s!"{exeName}: invalid checkpoint schema: {error}"
  let savedConfig ← match json.getObjVal? "config" with
    | .ok value => pure value
    | .error error => throw <| IO.userError s!"{exeName}: missing checkpoint config: {error}"
  let expectedConfig ←
    if schema == "torchlean.gpt-training.resume.v2" then
      pure (checkpointConfigJson cfg opts data)
    else if schema == "torchlean.gpt-training.resume.v1" then
      IO.eprintln <|
        s!"{exeName}: warning: resuming a v1 checkpoint without content or runtime hashes"
      pure (checkpointConfigJsonV1 cfg)
    else
      throw <| IO.userError s!"{exeName}: unsupported checkpoint schema {schema}"
  if savedConfig != expectedConfig then
    throw <| IO.userError
      s!"{exeName}: checkpoint configuration does not match this training command"
  let completed ← match json.getObjValAs? Nat "completed_step" with
    | .ok value => pure value
    | .error error => throw <| IO.userError s!"{exeName}: invalid checkpoint step: {error}"
  if completed > cfg.steps then
    throw <| IO.userError
      s!"{exeName}: checkpoint step {completed} exceeds configured step count {cfg.steps}"
  pure completed

/--
Write one resumable CUDA AdamW training checkpoint and point `LATEST` at it.

Parameters and optimizer moments are streamed separately into a temporary directory, which is
renamed into place only after both payloads, the metric snapshot, and the manifest are written.
-/
def saveTrainingCheckpoint
    {β : Type} [Storage β] {stateShapes inputShapes dataInputShapes : List Shape}
    (cfg : RunConfig) (opts : Runtime.Config)
    (module : Module.Objective Float β stateShapes inputShapes dataInputShapes)
    (data : DatasetIdentity)
    (root : System.FilePath) (completedStep : Nat)
    (points : Array MetricPoint) : IO System.FilePath := do
  IO.FS.createDirAll root
  let name := s!"step-{completedStep}"
  let destination := root / name
  if ← destination.pathExists then
    throw <| IO.userError s!"{exeName}: checkpoint already exists: {destination}"
  let temporary := root / s!".{name}.tmp"
  if ← temporary.pathExists then
    throw <| IO.userError <|
      s!"{exeName}: incomplete checkpoint already exists: {temporary}; inspect or move it first"
  IO.FS.createDirAll temporary
  Checkpoint.save module (temporary / "parameters.tlf32")
  Checkpoint.Optimizer.save module (temporary / "optimizer.tladam")
  writeMetrics (temporary / "metrics.json") cfg points
  IO.FS.writeFile (temporary / "manifest.json")
    (checkpointManifest cfg opts data completedStep).pretty
  IO.FS.rename temporary destination
  let latestTemporary := root / ".LATEST.tmp"
  IO.FS.writeFile latestTemporary (name ++ "\n")
  IO.FS.rename latestTemporary (root / "LATEST")
  pure destination

/-- Restore model parameters, CUDA AdamW moments, and the completed global step. -/
def loadTrainingCheckpoint
    {β : Type} [Storage β] {stateShapes inputShapes dataInputShapes : List Shape}
    (cfg : RunConfig) (opts : Runtime.Config)
    (module : Module.Objective Float β stateShapes inputShapes dataInputShapes)
    (data : DatasetIdentity)
    (path : System.FilePath) : IO (Nat × System.FilePath) := do
  let directory ← resolveCheckpointDirectory path
  let completed ← readCheckpointStep cfg opts data directory
  Checkpoint.load module (directory / "parameters.tlf32")
  Checkpoint.Optimizer.load module (directory / "optimizer.tladam")
  pure (completed, directory)

/-- Write the run passport: model, schedule, data identity, runtime, and applicable theorems. -/
def writePassport
    (path : System.FilePath) (cfg : RunConfig) (opts : Runtime.Config)
    (data : DatasetIdentity) (actualParameters : Nat) : IO Unit := do
  ensureParent path
  let model := cfg.model
  let residualProjectionStd := 0.02 / Float.sqrt (Float.ofNat (2 * model.layers))
  let artifact := Lean.Json.mkObj
    [ ("schema", .str "torchlean.gpt-training.passport.v2")
    , ("preset", .str cfg.presetName)
    , ("context", .num (Lean.JsonNumber.fromNat model.context))
    , ("vocab", .num (Lean.JsonNumber.fromNat model.vocab))
    , ("width", .num (Lean.JsonNumber.fromNat model.width))
    , ("heads", .num (Lean.JsonNumber.fromNat model.heads))
    , ("layers", .num (Lean.JsonNumber.fromNat model.layers))
    , ("batch", .num (Lean.JsonNumber.fromNat cfg.batch))
    , ("optimizer_steps", .num (Lean.JsonNumber.fromNat cfg.steps))
    , ("requested_token_budget",
        cfg.tokenBudget?.map (fun n => .num (Lean.JsonNumber.fromNat n)) |>.getD .null)
    , ("scheduled_training_tokens",
        .num (Lean.JsonNumber.fromNat (cfg.steps * cfg.batch * model.context)))
    , ("learning_rate_schedule", .str "linear-warmup-cosine")
    , ("peak_learning_rate", floatJson cfg.learningRate)
    , ("minimum_learning_rate", floatJson cfg.minLearningRate)
    , ("warmup_steps", .num (Lean.JsonNumber.fromNat cfg.warmupSteps))
    , ("weight_decay", floatJson cfg.weightDecay)
    , ("generation", generationJson cfg)
    , ("fresh_parameter_standard_deviation", floatJson 0.02)
    , ("fresh_residual_projection_standard_deviation", floatJson residualProjectionStd)
    , ("loaded_checkpoint",
        cfg.loadParams?.map (fun checkpoint => .str checkpoint.toString) |>.getD .null)
    , ("resumed_training_checkpoint",
        cfg.resume?.map (fun checkpoint => .str checkpoint.toString) |>.getD .null)
    , ("saved_checkpoint",
        cfg.saveParams?.map (fun checkpoint => .str checkpoint.toString) |>.getD .null)
    , ("stored_parameters", .num (Lean.JsonNumber.fromNat actualParameters))
    , ("train_tokens", .num (Lean.JsonNumber.fromNat data.trainTokens))
    , ("validation_tokens", .num (Lean.JsonNumber.fromNat data.validationTokens))
    , ("train_content_hash64", .num (Lean.JsonNumber.fromNat data.trainHash.toNat))
    , ("validation_content_hash64",
        .num (Lean.JsonNumber.fromNat data.validationHash.toNat))
    , ("train_mask_content_hash64",
        data.trainMaskHash?.map (fun value =>
          .num (Lean.JsonNumber.fromNat value.toNat)) |>.getD .null)
    , ("validation_mask_content_hash64",
        data.validationMaskHash?.map (fun value =>
          .num (Lean.JsonNumber.fromNat value.toNat)) |>.getD .null)
    , ("train_records_content_hash64",
        data.trainRecordsHash?.map (fun value =>
          .num (Lean.JsonNumber.fromNat value.toNat)) |>.getD .null)
    , ("validation_records_content_hash64",
        data.validationRecordsHash?.map (fun value =>
          .num (Lean.JsonNumber.fromNat value.toNat)) |>.getD .null)
    , ("lean_version", .str Lean.versionString)
    , ("device", .str opts.deviceName)
    , ("backend_profile", .str (backendProfileName opts))
    , ("scalar_type", .str "Float")
    , ("dtype", .str "float64")
    , ("train_shard", .str cfg.trainBin.toString)
    , ("validation_shard", .str cfg.valBin.toString)
    , ("objective",
        .str (if cfg.trainRecords?.isSome then
          "dialogue-bounded-assistant-next-token"
        else
          "next-token-mean"))
    , ("train_target_mask",
        cfg.trainMask?.map (fun path => .str path.toString) |>.getD .null)
    , ("validation_target_mask",
        cfg.valMask?.map (fun path => .str path.toString) |>.getD .null)
    , ("train_dialogue_records",
        cfg.trainRecords?.map (fun path => .str path.toString) |>.getD .null)
    , ("validation_dialogue_records",
        cfg.valRecords?.map (fun path => .str path.toString) |>.getD .null)
    , ("training_target_theorem",
        .str "TorchLeanGPT.causal_window_target_is_next_token")
    , ("training_record_target_theorem",
        .str "TorchLeanGPT.causal_record_target_is_next_token")
    , ("training_target_mask_theorem",
        .str "TorchLeanGPT.causal_window_mask_is_next_target")
    , ("dialogue_record_bound_theorem",
        .str "TorchLeanGPT.DialogueRecords.target_index_lt_record_end")
    , ("dialogue_record_selector_theorem",
        .str "TorchLeanGPT.DialogueRecords.targetRowEnabled_eq_true_iff")
    , ("dialogue_padding_theorem",
        .str "TorchLeanGPT.DialogueRecords.not_target_row_of_record_end_le")
    , ("weighted_objective_support_theorem",
        .str "TorchLeanGPT.weighted_rows_eq_of_eq_on_support")
    , ("causal_attention_theorem",
        .str "TorchLeanGPT.causal_attention_blocks_future_forward_and_backward")
    , ("training_resume_theorem",
        .str "TorchLeanGPT.Training.resume_eq_uninterrupted")
    , ("runtime_boundary", .str <|
        "The theorem names refer to specification-level results; " ++
        "device and backend_profile identify the execution path.")
    ]
  IO.FS.writeFile path artifact.pretty

/-- Predictor with discrete token ids and floating-point vocabulary logits. -/
abbrev Predictor (cfg : nn.models.CausalTransformer.Config) (batch : Nat) :=
  Tensor (Fin cfg.vocabularySize) (cfg.tokenShape [batch]) →
    IO (Tensor Float (cfg.vocabularyShape [batch]))

/-- Truncate or zero-pad one token row to the context length and repeat it across the batch. -/
def tokenBatchTensor
    (cfg : nn.models.CausalTransformer.Config) [NeZero cfg.vocabularySize]
    (batch : Nat) (tokens : List Nat) :
    Tensor (Fin cfg.vocabularySize) (cfg.tokenShape [batch]) :=
  let row := (tokens.take cfg.sequenceLength ++
    List.replicate (cfg.sequenceLength - Nat.min tokens.length cfg.sequenceLength) 0).toArray
  Tensor.dim fun _ => Tensor.ofFn fun position =>
    Fin.ofNat cfg.vocabularySize (row.getD position.val 0)

/--
Sample up to `steps` tokens after `promptIds`, returning the prompt followed by the new tokens.

Learned absolute positions are never restarted, so a prompt plus continuation longer than the
context is rejected. Sampling stops early at `stopToken?`, which is not appended.
-/
partial def generateIds
    (cfg : nn.models.CausalTransformer.Config) [NeZero cfg.vocabularySize]
    (batch : Nat) (predict : Predictor cfg batch)
    (promptIds : List Nat)
    (steps : Nat) (temperature : Float) (topK seed : Nat)
    (repeatPenalty : Float := defaultRepeatPenalty) (repeatWindow : Nat := defaultRepeatWindow)
    (stopToken? : Option Nat := none) : IO (List Nat) := do
  if promptIds.length + steps > cfg.sequenceLength then
    throw <| IO.userError <|
      s!"generation: prompt ({promptIds.length}) plus requested continuation ({steps}) " ++
        s!"exceeds context length {cfg.sequenceLength}"
  else if hBatch : batch = 0 then
    pure promptIds
  else if hSeq : cfg.sequenceLength = 0 then
    pure promptIds
  else
    letI : NeZero cfg.sequenceLength := ⟨hSeq⟩
    let generation : text.GenerationOptions :=
      { prompt := ""
        newTokenCount := steps
        temperature
        topK
        repeatPenalty
        repeatWindow
        seed
        asciiOnly := false }
    let firstBatch : Fin batch := ⟨0, Nat.pos_of_ne_zero hBatch⟩
    let rec loop (ids : List Nat) : Nat → IO (List Nat)
      | 0 => pure ids
      | remaining + 1 => do
          let generatedSoFar := generation.newTokenCount - (remaining + 1)
          let predictionPosition := if ids.isEmpty then 0 else ids.length - 1
          let x := tokenBatchTensor cfg batch ids
          let logits ← predict x
          let scores := Tensor.get (Tensor.get logits firstBatch)
            (Fin.ofNat cfg.sequenceLength predictionPosition)
          let recent :=
            if generation.repeatWindow = 0 then
              #[]
            else
              (ids.drop (ids.length - Nat.min ids.length generation.repeatWindow)).toArray
          let recentTensor : Tensor Nat [recent.size] := Tensor.ofFn fun i => recent[i]
          let nextToken ← CLI.orThrow exeName <|
            text.next scores generation generatedSoFar recentTensor
          if stopToken? = some nextToken.val then
            pure ids
          else
            loop (ids ++ [nextToken.val]) remaining
    loop promptIds generation.newTokenCount

/-- Load the optional GPT-2 BPE tokenizer pair. -/
def loadTokenizer? (cfg : RunConfig) : IO (Option text.GPT2BPE.Tokenizer) :=
  match cfg.tokenizerVocab?, cfg.tokenizerMerges? with
  | some vocab, some merges =>
      some <$> LeanProfiler.span "tokenizer.load"
        (text.GPT2BPE.load vocab merges (progress := true) (label := exeName))
        (metadata := { phase := some "data", activity := some "tokenizer" })
  | _, _ => pure none

/-- Generate and print one completion when tokenizer assets were supplied. -/
def printCompletion
    (runCfg : RunConfig) (modelCfg : nn.models.CausalTransformer.Config)
    [NeZero modelCfg.vocabularySize]
    (predict : Predictor modelCfg runCfg.batch) (tokenizer : text.GPT2BPE.Tokenizer) : IO Unit := do
  let promptIds ← CLI.orThrow exeName <| text.GPT2BPE.encode tokenizer runCfg.prompt
  let outputIds ← LeanProfiler.span "generation.complete"
    (generateIds modelCfg runCfg.batch predict promptIds.toList runCfg.generate
      runCfg.temperature runCfg.topK runCfg.seed runCfg.repeatPenalty runCfg.repeatWindow)
    (metadata :=
      { phase := some "generation"
        activity := some "autoregressive"
        stepIndex := some runCfg.generate })
  let output ← CLI.orThrow exeName <| text.GPT2BPE.decode tokenizer outputIds.toArray
  IO.println "\n--- completion ---"
  IO.println output
  IO.println "------------------"

/-- Update the schedule's learning rate without resetting AdamW moments. -/
def setAdamWLearningRate {shapes : List Shape} (learningRate : Float) :
    _root_.Runtime.Autograd.Model.Optim.StateList
      (fun _ _ => _root_.Optim.AdamW.State Float) Float shapes →
    _root_.Runtime.Autograd.Model.Optim.StateList
      (fun _ _ => _root_.Optim.AdamW.State Float) Float shapes
  | .nil => .nil
  | .cons state rest =>
      .cons { state with lr := learningRate } (setAdamWLearningRate learningRate rest)

/--
Train either the ordinary next-token objective or its masked variant and return the parameters.

Both paths share initialization, AdamW state, checkpointing, evaluation, metrics, and profiling.
The weighted path passes its row weights as a continuous input, apart from the discrete tokens and
targets.
-/
def optimizeObjective
    {inputShapes dataInputShapes : List Shape}
    (runCfg : RunConfig) (opts : Runtime.Config)
    (cfg : nn.models.CausalTransformer.Config) [NeZero cfg.vocabularySize]
    (model : nn.IndexedModel (cfg.tokenShape [runCfg.batch]) (cfg.vocabularyShape [runCfg.batch])
      (Fin cfg.vocabularySize))
    (data : DatasetIdentity) (actualParameterCount : Nat)
    (trainDef : Module.ObjectiveDefinition (Fin cfg.vocabularySize)
      model.stateShapes inputShapes dataInputShapes)
    (evalDef : Module.ObjectiveDefinition (Fin cfg.vocabularySize)
      model.stateShapes inputShapes dataInputShapes)
    (trainSample validationSample : Nat →
      TensorPack Float inputShapes × TensorPack (Fin cfg.vocabularySize) dataInputShapes) :
    IO (_root_.Runtime.Autograd.Torch.ParamList Float
      model.stateShapes) := do
  let module ← LeanProfiler.span "model.initialize"
    (Module.instantiate trainDef opts (α := Float))
    (metadata :=
      { phase := some "initialization"
        activity := some "parameters"
        backend := some (backendProfileName opts)
        dtype := some "Float"
        device := some opts.deviceName })
  match runCfg.loadParams? with
  | none => pure ()
  | some checkpoint =>
      LeanProfiler.span "checkpoint.load"
        (Checkpoint.load module checkpoint)
        (metadata := { phase := some "checkpoint", activity := some "load" })

  let optimizer := _root_.Runtime.Autograd.Model.Optim.adamw
    (α := Float) (paramShapes := model.stateShapes)
    runCfg.learningRate runCfg.weightDecay 0.9 0.999 1e-8
  let optimizerState ← LeanProfiler.span "optimizer.initialize"
    (module.initOptimizer optimizer)
    (metadata :=
      { phase := some "initialization"
        activity := some "adamw"
        device := some opts.deviceName })
  let optimizerStateRef ← IO.mkRef optimizerState
  let metricPoints ← IO.mkRef (#[] : Array MetricPoint)
  let schedule := Trainer.Scheduler.warmupCosine
    runCfg.learningRate runCfg.minLearningRate runCfg.warmupSteps runCfg.steps
  let parameters := (Module.Objective.Internal.runtime module).trainer.state
  let evaluator ← Runtime.Autograd.Model.Module.ObjectiveDef.evaluator
    evalDef opts parameters

  let startStep ← match runCfg.resume? with
    | none => pure 0
    | some checkpoint => do
        let (completed, directory) ← LeanProfiler.span "checkpoint.resume"
          (loadTrainingCheckpoint runCfg opts module data checkpoint)
          (metadata := { phase := some "checkpoint", activity := some "resume" })
        IO.println s!"resumed checkpoint: {directory} (completed_step={completed})"
        pure completed

  let evaluate (_step : Nat) : IO Float := do
    let losses ← (List.range runCfg.evalBatches).mapM fun batchIndex => do
      let (inputs, dataInputs) := validationSample batchIndex
      let loss ← Runtime.Autograd.Model.Module.Evaluator.run evaluator inputs dataInputs
      pure (Tensor.item loss)
    pure (losses.foldl (· + ·) 0.0 / Float.ofNat losses.length)

  let initialValidation ← LeanProfiler.span "evaluation.validation"
    (evaluate startStep)
    (metadata :=
      { phase := some "evaluation"
        activity := some "validation"
        stepIndex := some startStep
        device := some opts.deviceName })
  metricPoints.modify
    (·.push { phase := "validation", step := startStep, loss := initialValidation })
  IO.println s!"step={startStep} validation_loss={initialValidation}"

  let trainOneStep (step : Nat) : IO Unit := do
    let state ← optimizerStateRef.get
    let learningRate := schedule.rate step
    let state := setAdamWLearningRate learningRate state
    let startedAt ← IO.monoMsNow
    let (inputs, dataInputs) := trainSample step
    let (nextState, trainLossTensor) ← LeanProfiler.span "training.update"
      (module.step optimizer state (Arguments.Internal.fromTensorPack inputs)
        (Arguments.Internal.fromTensorPack dataInputs) (loss := true))
      (metadata :=
        { phase := some "training"
          activity := some "forward-backward-update"
          stepIndex := some (step + 1)
          backend := some (backendProfileName opts)
          dtype := some "Float"
          device := some opts.deviceName })
    let finishedAt ← IO.monoMsNow
    let durationMs := finishedAt - startedAt
    let tokensPerSecond :=
      if durationMs = 0 then
        0.0
      else
        Float.ofNat (runCfg.batch * cfg.sequenceLength) * 1000.0 / Float.ofNat durationMs
    optimizerStateRef.set nextState
    let trainLoss := Tensor.item trainLossTensor
    metricPoints.modify (·.push
      { phase := "train"
        step := step + 1
        loss := trainLoss
        learningRate := some learningRate
        durationMs := some durationMs
        tokensPerSecond := some tokensPerSecond })
    let completed := step + 1
    if runCfg.evalEvery != 0 &&
        (completed % runCfg.evalEvery == 0 || completed == runCfg.steps) then
      let validationLoss ← LeanProfiler.span "evaluation.validation"
        (evaluate completed)
        (metadata :=
          { phase := some "evaluation"
            activity := some "validation"
            stepIndex := some completed
            device := some opts.deviceName })
      metricPoints.modify
        (·.push { phase := "validation", step := completed, loss := validationLoss })
      IO.println <|
        s!"step={completed} train_loss={trainLoss} validation_loss={validationLoss} " ++
        s!"learning_rate={learningRate} " ++
        s!"step_ms={durationMs} tokens_per_second={tokensPerSecond}"
    else
      IO.println <|
        s!"step={completed} train_loss={trainLoss} learning_rate={learningRate} " ++
        s!"step_ms={durationMs} tokens_per_second={tokensPerSecond}"
    match runCfg.checkpointDir? with
    | some root =>
        if runCfg.checkpointEvery != 0 &&
            (completed % runCfg.checkpointEvery == 0 || completed == runCfg.steps) then
          let points ← metricPoints.get
          let checkpoint ← LeanProfiler.span "checkpoint.save-resumable"
            (saveTrainingCheckpoint runCfg opts module data root completed points)
            (metadata :=
              { phase := some "checkpoint"
                activity := some "save-resumable"
                stepIndex := some completed
                device := some opts.deviceName })
          IO.println s!"wrote resumable checkpoint: {checkpoint}"
    | none => pure ()

  -- A tail-recursive driver keeps the long-running update loop independent of collection syntax.
  let rec trainLoop (step : Nat) : Nat → IO Unit
    | 0 => pure ()
    | remaining + 1 => do
        trainOneStep step
        trainLoop (step + 1) remaining
  trainLoop startStep (runCfg.steps - startStep)

  match runCfg.saveParams? with
  | none => pure ()
  | some checkpoint =>
      ensureParent checkpoint
      LeanProfiler.span "checkpoint.save"
        (Checkpoint.save module checkpoint)
        (metadata := { phase := some "checkpoint", activity := some "save" })
      IO.println s!"wrote checkpoint: {checkpoint}"

  let points ← metricPoints.get
  writeMetrics runCfg.metricsPath runCfg points
  writePassport runCfg.passportPath runCfg opts data actualParameterCount
  IO.println s!"wrote metrics: {runCfg.metricsPath}"
  IO.println s!"wrote passport: {runCfg.passportPath}"
  pure parameters

/-- One training run: load and check the data, train, and print an optional completion. -/
def train (opts : Runtime.Config) (args : List String) : IO Unit := do
  let runCfg ← RunConfig.parse opts.seed args
  let trainTokens ← LeanProfiler.span "dataset.train.read"
    (readTokenShard runCfg.trainBin runCfg.model.vocab)
    (metadata := { phase := some "data", activity := some "read-train" })
  let valTokens ← LeanProfiler.span "dataset.validation.read"
    (readTokenShard runCfg.valBin runCfg.model.vocab)
    (metadata := { phase := some "data", activity := some "read-validation" })
  if trainTokens.size <= runCfg.model.context then
    throw <| IO.userError
      s!"{exeName}: training shard needs more than {runCfg.model.context} tokens"
  if valTokens.size <= runCfg.model.context then
    throw <| IO.userError
      s!"{exeName}: validation shard needs more than {runCfg.model.context} tokens"
  let trainMask? ← match runCfg.trainMask? with
    | none => pure none
    | some path => do
        let mask ← LeanProfiler.span "dataset.train-mask.read"
          (readTargetMask path)
          (metadata := { phase := some "data", activity := some "read-train-mask" })
        pure (some mask)
  let valMask? ← match runCfg.valMask? with
    | none => pure none
    | some path => do
        let mask ← LeanProfiler.span "dataset.validation-mask.read"
          (readTargetMask path)
          (metadata := { phase := some "data", activity := some "read-validation-mask" })
        pure (some mask)
  match trainMask? with
  | some mask =>
      if mask.size != trainTokens.size then
        throw <| IO.userError <|
          s!"{exeName}: training mask has {mask.size} entries for {trainTokens.size} tokens"
  | none => pure ()
  match valMask? with
  | some mask =>
      if mask.size != valTokens.size then
        throw <| IO.userError <|
          s!"{exeName}: validation mask has {mask.size} entries for {valTokens.size} tokens"
  | none => pure ()

  let trainRecords? ← match runCfg.trainRecords? with
    | none => pure none
    | some path => do
        let records ← LeanProfiler.span "dataset.train-records.read"
          (readDialogueRecords path trainTokens.size runCfg.model.context)
          (metadata := { phase := some "data", activity := some "read-train-records" })
        match trainMask? with
        | some mask => validateDialogueRecordTargets path records mask
        | none => throw <| IO.userError s!"{exeName}: training records require a target mask"
        pure (some records)
  let valRecords? ← match runCfg.valRecords? with
    | none => pure none
    | some path => do
        let records ← LeanProfiler.span "dataset.validation-records.read"
          (readDialogueRecords path valTokens.size runCfg.model.context)
          (metadata := { phase := some "data", activity := some "read-validation-records" })
        match valMask? with
        | some mask => validateDialogueRecordTargets path records mask
        | none => throw <| IO.userError s!"{exeName}: validation records require a target mask"
        pure (some records)

  let dataIdentity : DatasetIdentity :=
    { trainTokens := trainTokens.size
      validationTokens := valTokens.size
      trainHash := trainTokens.contentHash
      validationHash := valTokens.contentHash
      trainMaskHash? := trainMask?.map (·.contentHash)
      validationMaskHash? := valMask?.map (·.contentHash)
      trainRecordsHash? := trainRecords?.map (·.contentHash)
      validationRecordsHash? := valRecords?.map (·.contentHash) }

  let cfg := runCfg.model.toTorchLean
  if hSeq : cfg.sequenceLength = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero context after validation"
  else if cfg.modelWidth = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero model width after validation"
  else if hVocab : cfg.vocabularySize = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero vocabulary after validation"
  else
    letI : NeZero cfg.vocabularySize := ⟨hVocab⟩
    do
      _root_.TorchLean.rand.manualSeed runCfg.seed
      let model := nn.build runCfg.seed (buildModel cfg runCfg.batch opts)
      do
        letI : NeZero cfg.vocabularySize := ⟨hVocab⟩
        do
          let actualParameterCount :=
            model.stateShapes.foldl
              (fun total shape => total + Shape.size shape) 0

          IO.println s!"preset={runCfg.presetName}"
          IO.println <|
            s!"model=context {cfg.sequenceLength}, vocab {cfg.vocabularySize}, width {cfg.modelWidth}, " ++
            s!"heads {cfg.headCount}, layers {cfg.layerCount}"
          IO.println s!"parameters={actualParameterCount}"
          IO.println s!"device={opts.deviceName} profile={backendProfileName opts}"
          IO.println s!"tokens=train {trainTokens.size}, validation {valTokens.size}"
          IO.println <|
            s!"schedule=steps {runCfg.steps}, tokens/update {runCfg.batch * cfg.sequenceLength}, " ++
              s!"total {runCfg.steps * runCfg.batch * cfg.sequenceLength}"

          let trainedParams ← match trainMask?, valMask?, trainRecords?, valRecords? with
            | none, none, none, none =>
                let trainDef := nn.models.CausalTransformer.objective cfg model
                let evalDef :=
                  nn.models.CausalTransformer.objective cfg model (mode := .eval)
                let trainSample (step : Nat) :=
                  let (tokens, targets) := causalLmTokenBatchFromShard
                    cfg.vocabularySize runCfg.batch cfg.sequenceLength trainTokens runCfg.seed step (padId := 0)
                  (.nil, .cons tokens (.cons targets .nil))
                let validationSample (step : Nat) :=
                  let (tokens, targets) := causalLmTokenBatchFromShard
                    cfg.vocabularySize runCfg.batch cfg.sequenceLength valTokens (runCfg.seed + 1000003) step (padId := 0)
                  (.nil, .cons tokens (.cons targets .nil))
                optimizeObjective runCfg opts cfg model
                  dataIdentity actualParameterCount
                  trainDef evalDef trainSample validationSample
            | some trainMask, some valMask, some trainRecords, some valRecords =>
                IO.println <|
                  s!"objective=dialogue-bounded weighted next-token loss, " ++
                    s!"records=train {trainRecords.entries.size}, validation {valRecords.entries.size}"
                let trainDef := weightedObjective cfg runCfg.batch model
                let evalDef :=
                  weightedObjective cfg runCfg.batch model (mode := .eval)
                let trainSample (step : Nat) :=
                  let (tokens, targets, rowWeights) := causalLmMaskedTokenBatchFromRecords
                    (α := Float) cfg.vocabularySize runCfg.batch cfg.sequenceLength trainTokens trainMask trainRecords
                    runCfg.seed step (padId := 0)
                  (.cons rowWeights .nil, .cons tokens (.cons targets .nil))
                let validationSample (step : Nat) :=
                  let (tokens, targets, rowWeights) := causalLmMaskedTokenBatchFromRecords
                    (α := Float) cfg.vocabularySize runCfg.batch cfg.sequenceLength valTokens valMask valRecords
                    (runCfg.seed + 1000003) step (padId := 0)
                  (.cons rowWeights .nil, .cons tokens (.cons targets .nil))
                optimizeObjective runCfg opts cfg model
                  dataIdentity actualParameterCount
                  trainDef evalDef trainSample validationSample
            | _, _, _, _ =>
                throw <| IO.userError
                  s!"{exeName}: training masks and dialogue records must be supplied as paired sets"

          let forwardProgram : _root_.Runtime.Autograd.Model.Program.Data
              Float (Fin cfg.vocabularySize)
              (model.stateShapes ++ [])
              [cfg.tokenShape [runCfg.batch]] (cfg.vocabularyShape [runCfg.batch]) := by
            exact fun {m} _ _ =>
              by
                simpa using
                  nn.IndexedModel.Internal.program model .eval (α := Float) (m := m)
          let evaluator ← Runtime.Autograd.Model.Module.Evaluator.new
            forwardProgram opts trainedParams
          let predict : Predictor cfg runCfg.batch := fun tokens =>
            Runtime.Autograd.Model.Module.Evaluator.run evaluator .nil (.cons tokens .nil)
          let tokenizer? ← loadTokenizer? runCfg
          match tokenizer? with
          | none =>
              if runCfg.generate != 0 then
                IO.println "generation skipped: pass both --tokenizer-vocab and --tokenizer-merges"
          | some tokenizer =>
              if runCfg.generate != 0 then
                printCompletion runCfg cfg predict tokenizer

/--
Parse the shared runtime flags for commands that execute the host `Float` model.

TorchLean's generic command runner announces `native binary32`, which describes its `Float32`
examples. These commands compute with Lean's `Float` (IEEE-754 binary64) on CPU and CUDA, so they
print their own runtime line and otherwise follow the same steps.
-/
def runFloatCommand
    (commandName : String) (args : List String) (banner : String)
    (command : Runtime.Config → List String → IO Unit)
    (defaultSeed : Nat := 0) : IO UInt32 := do
  let (seed, args) ← IO.ofExcept <|
    (CLI.takeSeed (CLI.dropDashDash args) (default := defaultSeed)).mapError
      (fun message => s!"{commandName}: {message}")
  let (selection, args) ← IO.ofExcept <|
    (Module.RuntimeSelection.parse args).mapError (fun message => s!"{commandName}: {message}")
  if selection.arithmetic != .native then
    throw <| IO.userError s!"{commandName}: this program only supports `--arithmetic native`"
  IO.println "[TorchLean] arithmetic: Lean Float (IEEE-754 binary64)"
  IO.println s!"[TorchLean] device: {selection.device.cliName}"
  let runtime ← IO.ofExcept (Module.RuntimeSelection.toConfig selection seed)
  runtime.validateForExecution
  _root_.TorchLean.rand.manualSeed seed
  IO.println banner
  (← IO.getStdout).flush
  command runtime args
  pure 0

/-- Program entrypoint. -/
def main (args : List String) : IO UInt32 := do
  if CLI.hasHelp args then
    IO.println usage
    return 0
  LeanProfiler.profileFromEnvironment "torchlean-gpt.run" <|
    runFloatCommand exeName args "TorchLean GPT-style training" train

end Run
end TorchLeanGPT
