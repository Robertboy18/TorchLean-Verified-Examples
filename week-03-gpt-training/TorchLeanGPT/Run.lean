/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.CausalTraining
public import TorchLeanGPT.DialogueRecords
public import LeanProfiler
public import NN.Spec.Core.Tensor.Constructors

/-!
# Training and completion runner

This executable consumes the little-endian `uint16` token shards written by
`tools/prepare_dataset.py`. The format is plain: every GPT-2 token id fits in 16 bits,
and a shard contains no executable code or Python object serialization.

The run writes:

* a shape-checked TorchLean parameter checkpoint, when `--save-params` is supplied;
* a metric file containing the observed training and validation losses;
* a passport containing the model dimensions, parameter count, backend profile, data paths, and
  relevant Lean theorem names.

Set `LEAN_PROFILE=1` to add a LeanProfiler trace and summary. Without it, the executable does not
retain profiling events.
-/

@[expose] public section

open TorchLean

namespace TorchLeanGPT
namespace Run

/-- Name used in diagnostics. -/
def exeName : String := "train_torchlean_gpt"

/-- Repository-relative default locations written by the dataset preparation script. -/
def defaultDataDir : System.FilePath :=
  "week-03-gpt-training/data/tinyshakespeare"

/-- One sampled loss value in the run's metric artifact. -/
structure MetricPoint where
  phase : String
  step : Nat
  loss : Float
  learningRate : Option Float := none
  durationMs : Option Nat := none
  tokensPerSecond : Option Float := none
  deriving Repr

/-- Command-line settings after applying a preset and explicit overrides. -/
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
    , ""
    , "Set LEAN_PROFILE=1 to write LeanProfiler trace and summary JSON files."
    ]

/-- Lift a pure parser result into an `IO.userError` carrying a command name. -/
def orThrowFor {α : Type} (command : String) (result : Except String α) : IO α :=
  match result with
  | .ok value => pure value
  | .error message => throw <| IO.userError s!"{command}: {message}"

/-- Parser error helper specialized to the training executable. -/
def orThrow {α : Type} (result : Except String α) : IO α :=
  orThrowFor exeName result

/-- Reject non-finite command-line values before they reach training or sampling. -/
def requireFinite (flag : String) (value : Float) : Except String Unit :=
  if value.isNaN || value.isInf then
    .error s!"--{flag} must be finite"
  else
    .ok ()

/-- Override selected model dimensions while preserving the chosen preset. -/
def parseModelConfigFor (command : String) (args : List String) (base : ModelConfig) :
    IO (ModelConfig × List String) := do
  let (context, args) ← orThrowFor command <|
    CLI.takePositiveNatFlag args command "context" base.context
  let (vocab, args) ← orThrowFor command <|
    CLI.takePositiveNatFlag args command "vocab" base.vocab
  let (width, args) ← orThrowFor command <|
    CLI.takePositiveNatFlag args command "width" base.width
  let (heads, args) ← orThrowFor command <|
    CLI.takePositiveNatFlag args command "heads" base.heads
  let (layers, args) ← orThrowFor command <|
    CLI.takePositiveNatFlag args command "layers" base.layers
  let (dropout, args) ← orThrowFor command <|
    CLI.takeNonnegativeFloatFlag args command "dropout" base.dropout
  let model : ModelConfig :=
    { context, vocab, width, heads, layers, dropout }
  orThrowFor command model.validate
  pure (model, args)

/-- Model parser specialized to the training executable's diagnostics. -/
def parseModelConfig (args : List String) (base : ModelConfig) :
    IO (ModelConfig × List String) :=
  parseModelConfigFor exeName args base

/-- Parse training, artifact, tokenizer, and generation options. -/
def RunConfig.parse (seed : Nat) (args : List String) : IO RunConfig := do
  let (presetName, args) ← orThrow <|
    CLI.takeFlagValueDefault args "preset" "quick"
  let preset ← orThrow <| ModelConfig.ofName presetName
  let (model, args) ← parseModelConfig args preset
  let defaultSteps := if presetName.toLower == "gpt2-small" then 1000 else 2
  let defaultBatch := if presetName.toLower == "gpt2-small" then 4 else 2
  let (steps?, args) ← orThrow <| CLI.takeNatFlagOnce args "steps"
  let (tokenBudget?, args) ← orThrow <| CLI.takeNatFlagOnce args "token-budget"
  let (batch, args) ← orThrow <|
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
  let (learningRate, args) ← orThrow <|
    CLI.takePositiveFloatFlag args exeName "lr" defaultLearningRate
  let (minLearningRate, args) ← orThrow <|
    CLI.takeNonnegativeFloatFlag args exeName "min-lr" (learningRate * 0.1)
  let defaultWarmupSteps :=
    if presetName.toLower == "gpt2-small" then
      min 2000 (steps / 10)
    else
      0
  let (warmupSteps, args) ← orThrow <|
    CLI.takeNatFlagDefault args "warmup-steps" defaultWarmupSteps
  let (weightDecay, args) ← orThrow <|
    CLI.takeNonnegativeFloatFlag args exeName "weight-decay" 0.1
  let (evalEvery, args) ← orThrow <|
    CLI.takeNatFlagDefault args "eval-every" (if steps <= 10 then 1 else 100)
  let (evalBatches, args) ← orThrow <|
    CLI.takePositiveNatFlag args exeName "eval-batches" 4
  let (trainBin, args) ← orThrow <|
    CLI.takePathFlagDefault args "train-bin" (defaultDataDir / "train.bin")
  let (valBin, args) ← orThrow <|
    CLI.takePathFlagDefault args "val-bin" (defaultDataDir / "val.bin")
  let ((trainMask?, valMask?), args) ← orThrow <|
    CLI.takePairedPathFlags args "train-mask" "val-mask"
  let ((trainRecords?, valRecords?), args) ← orThrow <|
    CLI.takePairedPathFlags args "train-records" "val-records"
  if trainMask?.isSome != trainRecords?.isSome then
    throw <| IO.userError <|
      s!"{exeName}: target masks and dialogue records must be supplied together"
  let (loadParams?, args) ← orThrow <| CLI.takePathFlagOnce args "load-params"
  let (saveParams?, args) ← orThrow <| CLI.takePathFlagOnce args "save-params"
  let (resume?, args) ← orThrow <| CLI.takePathFlagOnce args "resume"
  let (checkpointDir?, args) ← orThrow <| CLI.takePathFlagOnce args "checkpoint-dir"
  let (checkpointEvery, args) ← orThrow <| CLI.takeNatFlagDefault args "checkpoint-every" 0
  if loadParams?.isSome && resume?.isSome then
    throw <| IO.userError s!"{exeName}: --load-params and --resume are mutually exclusive"
  if checkpointEvery != 0 && checkpointDir?.isNone then
    throw <| IO.userError s!"{exeName}: --checkpoint-every requires --checkpoint-dir"
  let (metricsPath, args) ← orThrow <|
    CLI.takePathFlagDefault args "metrics"
      "week-03-gpt-training/artifacts/training-metrics.json"
  let (passportPath, args) ← orThrow <|
    CLI.takePathFlagDefault args "passport"
      "week-03-gpt-training/artifacts/run-passport.json"
  let ((tokenizerVocab?, tokenizerMerges?), args) ← orThrow <|
    CLI.takePairedPathFlags args "tokenizer-vocab" "tokenizer-merges"
  let (prompt, args) ← orThrow <|
    CLI.takeFlagValueDefault args "prompt" "The meaning of verification is"
  let (generate, args) ← orThrow <| CLI.takeNatFlagDefault args "generate" 0
  let (temperature, args) ← orThrow <|
    CLI.takePositiveFloatFlag args exeName "temperature" 0.8
  let (topK, args) ← orThrow <| CLI.takeNatFlagDefault args "top-k" 40
  orThrow <| CLI.checkNoArgs args
  orThrow <| requireFinite "lr" learningRate
  orThrow <| requireFinite "min-lr" minLearningRate
  if minLearningRate > learningRate then
    throw <| IO.userError s!"{exeName}: --min-lr cannot exceed --lr"
  orThrow <| requireFinite "weight-decay" weightDecay
  orThrow <| requireFinite "temperature" temperature
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
      topK }

/-- Read one little-endian `uint16`, returning zero for an incomplete trailing word. -/
def readUInt16LE (bytes : ByteArray) (offset : Nat) : Nat :=
  if hHigh : offset + 1 < bytes.size then
    have hLow : offset < bytes.size := Nat.lt_trans (Nat.lt_succ_self offset) hHigh
    (bytes.get offset hLow).toNat + 256 * (bytes.get (offset + 1) hHigh).toNat
  else
    0

/-- A token shard kept in its compact on-disk representation. -/
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

/-- A byte-per-token target mask retained as bytes rather than boxed booleans. -/
structure TargetMask where
  bytes : ByteArray
  contentHash : UInt64

/-- Number of mask entries. -/
def TargetMask.size (mask : TargetMask) : Nat :=
  mask.bytes.size

/-- Read one target-mask bit, defaulting to `false` beyond the mask. -/
def TargetMask.getD (mask : TargetMask) (index : Nat) : Bool :=
  if h : index < mask.size then mask.bytes.get index h != 0 else false

/-- One contiguous training example inside a token shard. -/
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
Load dialogue-window records and check that every example stays inside one token shard.

The writer limits records to `context + 1` tokens: one input window plus its final next-token
target. Rejecting larger records here prevents a caller from silently truncating an example under
a different model context.
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
Check that ordered record targets partition the active target mask.

Every recorded target must be active, target intervals may not overlap, and their total length must
equal the number of active mask bytes. Together these checks prevent omitted, duplicated, or
invented assistant targets from entering the weighted objective.
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

/-- Content identity for the token, target-mask, and dialogue-record shards used by one run. -/
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

The loader rejects truncated files and ids outside the selected model vocabulary before any tensor
indexing occurs.
-/
def readTokenShard (path : System.FilePath) (vocab : Nat) : IO TokenShard := do
  let bytes ← IO.FS.readBinFile path
  if bytes.size % 2 != 0 then
    throw <| IO.userError s!"{exeName}: {path} has an odd byte count"
  for i in [0:bytes.size / 2] do
    let token := readUInt16LE bytes (2 * i)
    if token >= vocab then
      throw <| IO.userError
        s!"{exeName}: token id {token} in {path} is outside vocabulary [0, {vocab})"
  pure { bytes, contentHash := hash bytes }

/--
Load a byte-per-token target mask.

Only `0` and `1` are accepted. Keeping the mask format this small makes it easy to inspect and
ensures that the Lean runner, rather than a Python preprocessing convention, determines the loss
normalization for each sampled batch.
-/
def readTargetMask (path : System.FilePath) : IO TargetMask := do
  let bytes ← IO.FS.readBinFile path
  for i in [0:bytes.size] do
    match bytes[i]!.toNat with
    | 0 | 1 => pure ()
    | value =>
        throw <| IO.userError
          s!"{exeName}: target mask {path} contains byte {value} at offset {i}; expected 0 or 1"
  pure { bytes, contentHash := hash bytes }

/--
Build one deterministic next-token batch without expanding the complete corpus into `Array Nat`.
-/
def causalLmTokenBatchFromShard
    (vocab batch seqLen : Nat) [NeZero vocab]
    (shard : TokenShard) (seed step : Nat) (padId : Nat := 0) :
    Tensor (Fin vocab) [batch, seqLen] × Tensor (Fin vocab) [batch, seqLen] :=
  let offsetAt := text.Corpus.randomBatchOffsets shard.size seqLen batch seed step
  let offsets := Array.ofFn offsetAt
  (Spec.Tensor.generate [batch, seqLen] fun
      | [bi, i] => Fin.ofNat vocab (shard.getD (offsets.getD bi 0 + i) padId)
      | _ => Fin.ofNat vocab padId,
    Spec.Tensor.generate [batch, seqLen] fun
      | [bi, i] => Fin.ofNat vocab (shard.getD (offsets.getD bi 0 + i + 1) padId)
      | _ => Fin.ofNat vocab padId)

/-- Build a deterministic dialogue-bounded batch from compact token, mask, and record shards. -/
def causalLmMaskedTokenBatchFromRecords
    {α : Type} [_root_.Context α] [Runtime.FromFloat α]
    (vocab batch seqLen : Nat) [NeZero vocab]
    (shard : TokenShard) (targetMask : TargetMask)
    (records : DialogueRecords)
    (seed step : Nat) (padId : Nat := 0) :
    Tensor (Fin vocab) [batch, seqLen] ×
      Tensor (Fin vocab) [batch, seqLen] × Tensor α [batch, seqLen] :=
  let key := Runtime.Autograd.TorchLean.Random.keyOf seed step
  let recordAt (batchIndex : Nat) : DialogueRecord :=
    records.entries[Runtime.Autograd.TorchLean.Random.sampleNat
      key batchIndex records.entries.size]!
  let (xTokens, yTokens, enabledTargets) := Id.run do
    let mut xs := Array.mkEmpty (batch * seqLen)
    let mut ys := Array.mkEmpty (batch * seqLen)
    let mut enabled := Array.mkEmpty (batch * seqLen)
    for bi in List.finRange batch do
      let record := recordAt bi
      for i in [0:seqLen] do
        if i + 1 < record.length then
          xs := xs.push (shard.getD (record.offset + i) padId)
          ys := ys.push (shard.getD (record.offset + i + 1) padId)
          let targetIndex := record.offset + i + 1
          let inRecordTargets :=
            TorchLeanGPT.DialogueRecords.targetRowEnabled
              record.offset record.targetOffset record.targetLength i
          enabled := enabled.push (inRecordTargets && targetMask.getD targetIndex)
        else
          xs := xs.push padId
          ys := ys.push padId
          enabled := enabled.push false
    return (xs, ys, enabled)
  let activeCount := enabledTargets.foldl (fun count active =>
    if active then count + 1 else count) 0
  let activeWeight :=
    if activeCount = 0 then 0 else 1.0 / Float.ofNat activeCount
  let rowWeights := enabledTargets.map (fun active => if active then activeWeight else 0)
  (Spec.Tensor.generate [batch, seqLen] fun
      | [bi, i] => Fin.ofNat vocab (xTokens.getD (bi * seqLen + i) padId)
      | _ => Fin.ofNat vocab padId,
    Spec.Tensor.generate [batch, seqLen] fun
      | [bi, i] => Fin.ofNat vocab (yTokens.getD (bi * seqLen + i) padId)
      | _ => Fin.ofNat vocab padId,
    Spec.Tensor.generate [batch, seqLen] fun
      | [bi, i] => Runtime.ofFloat (rowWeights.getD (bi * seqLen + i) 0.0)
      | _ => Runtime.ofFloat 0.0)

/-- Ensure a generated artifact's parent directory exists. -/
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
def backendProfileName (opts : Options) : String :=
  match opts.resolveBackendProfile with
  | .ok profile => profile.name
  | .error _ => "unresolved"

/-- Serialize observed losses in a small stable JSON schema. -/
def writeMetrics (path : System.FilePath) (points : Array MetricPoint) : IO Unit := do
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
    , ("points", .arr rows)
    ]
  IO.FS.writeFile path artifact.pretty

/-! ## Resumable training checkpoints -/

/-- Stable JSON identity for the parts of a run that determine its optimizer trajectory. -/
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

/-- Resume identity used by newly written checkpoints. -/
def checkpointConfigJson
    (cfg : RunConfig) (opts : Options) (data : DatasetIdentity) : Lean.Json :=
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
    (cfg : RunConfig) (opts : Options) (data : DatasetIdentity)
    (completedStep : Nat) : Lean.Json :=
  Lean.Json.mkObj
    [ ("schema", .str "torchlean.gpt-training.resume.v2")
    , ("completed_step", .num (Lean.JsonNumber.fromNat completedStep))
    , ("config", checkpointConfigJson cfg opts data)
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
    (cfg : RunConfig) (opts : Options) (data : DatasetIdentity)
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
Write one exact CUDA AdamW training checkpoint and publish it through `LATEST`.

Parameters and optimizer moments are streamed independently into a temporary directory. The
directory becomes visible only after both payloads, the metric snapshot, and the manifest have
been written successfully.
-/
def saveTrainingCheckpoint
    {β : Type} {stateShapes inputShapes dataInputShapes : List Shape}
    (cfg : RunConfig)
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
    IO.FS.removeDirAll temporary
  IO.FS.createDirAll temporary
  Checkpoint.saveModule module (temporary / "parameters.tlf32")
  Checkpoint.saveOptimizerState module (temporary / "optimizer.tladam")
  writeMetrics (temporary / "metrics.json") points
  IO.FS.writeFile (temporary / "manifest.json")
    (checkpointManifest cfg module.opts data completedStep).pretty
  IO.FS.rename temporary destination
  let latestTemporary := root / ".LATEST.tmp"
  IO.FS.writeFile latestTemporary (name ++ "\n")
  IO.FS.rename latestTemporary (root / "LATEST")
  pure destination

/-- Restore model parameters, CUDA AdamW moments, and the completed global step. -/
def loadTrainingCheckpoint
    {β : Type} {stateShapes inputShapes dataInputShapes : List Shape}
    (cfg : RunConfig)
    (module : Module.Objective Float β stateShapes inputShapes dataInputShapes)
    (data : DatasetIdentity)
    (path : System.FilePath) : IO (Nat × System.FilePath) := do
  let directory ← resolveCheckpointDirectory path
  let completed ← readCheckpointStep cfg module.opts data directory
  Checkpoint.loadModule module (directory / "parameters.tlf32")
  Checkpoint.loadOptimizerState module (directory / "optimizer.tladam")
  pure (completed, directory)

/-- Write the architectural and runtime boundary associated with one run. -/
def writePassport
    (path : System.FilePath) (cfg : RunConfig) (opts : Options)
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
  Tensor (Fin cfg.vocab) (tokenShape cfg batch) →
    IO (Tensor Float (logitShape cfg batch))

/-- Repeat one padded token row across the configured batch. -/
def tokenBatchTensor
    (cfg : nn.models.CausalTransformer.Config) [NeZero cfg.vocab]
    (batch : Nat) (tokens : List Nat) :
    Tensor (Fin cfg.vocab) (tokenShape cfg batch) :=
  let row := (tokens.take cfg.seqLen ++
    List.replicate (cfg.seqLen - Nat.min tokens.length cfg.seqLen) 0).toArray
  Spec.Tensor.generate [batch, cfg.seqLen] fun
    | [_batch, position] => Fin.ofNat cfg.vocab (row.getD position 0)
    | _ => Fin.ofNat cfg.vocab 0

/-- Sample a continuation without restarting the model's learned absolute positions. -/
partial def generateIds
    (cfg : nn.models.CausalTransformer.Config) [NeZero cfg.vocab]
    (batch : Nat) (predict : Predictor cfg batch)
    (promptIds : List Nat)
    (steps : Nat) (temperature : Float) (topK seed : Nat)
    (stopToken? : Option Nat := none) : IO (List Nat) := do
  if promptIds.length + steps > cfg.seqLen then
    throw <| IO.userError <|
      s!"generation: prompt ({promptIds.length}) plus requested continuation ({steps}) " ++
        s!"exceeds context length {cfg.seqLen}"
  else if hBatch : batch = 0 then
    pure promptIds
  else if hSeq : cfg.seqLen = 0 then
    pure promptIds
  else
    letI : NeZero cfg.seqLen := ⟨hSeq⟩
    let generation : text.GenerationOptions :=
      { prompt := ""
        generate := steps
        temperature
        topK
        repeatPenalty := 1.05
        repeatWindow := 64
        seed
        asciiOnly := false }
    let firstBatch : Fin batch := ⟨0, Nat.pos_of_ne_zero hBatch⟩
    let rec loop (ids : List Nat) : Nat → IO (List Nat)
      | 0 => pure ids
      | remaining + 1 => do
          let generatedSoFar := generation.generate - (remaining + 1)
          let predictionPosition := if ids.isEmpty then 0 else ids.length - 1
          let x := tokenBatchTensor cfg batch ids
          let logits ← predict x
          let scores := text.batchLogitScoresAt logits firstBatch
            (Fin.ofNat cfg.seqLen predictionPosition)
          let recent :=
            if generation.repeatWindow = 0 then
              #[]
            else
              (ids.drop (ids.length - Nat.min ids.length generation.repeatWindow)).toArray
          let nextToken ← orThrow <| text.chooseNextToken scores generation generatedSoFar recent
          if stopToken? = some nextToken.val then
            pure ids
          else
            loop (ids ++ [nextToken.val]) remaining
    loop promptIds generation.generate

/-- Load the optional GPT-2 BPE tokenizer pair. -/
def loadTokenizer? (cfg : RunConfig) : IO (Option text.GPT2BPE.Tokenizer) :=
  match cfg.tokenizerVocab?, cfg.tokenizerMerges? with
  | some vocab, some merges =>
      some <$> LeanProfiler.span "tokenizer.load"
        (text.GPT2BPE.loadWithProgress exeName vocab merges)
        (metadata := { phase := some "data", activity := some "tokenizer" })
  | _, _ => pure none

/-- Generate and print one completion when tokenizer assets were supplied. -/
def printCompletion
    (runCfg : RunConfig) (modelCfg : nn.models.CausalTransformer.Config)
    [NeZero modelCfg.vocab]
    (predict : Predictor modelCfg runCfg.batch) (tokenizer : text.GPT2BPE.Tokenizer) : IO Unit := do
  let promptIds ← orThrow <| text.GPT2BPE.encode tokenizer runCfg.prompt
  let outputIds ← LeanProfiler.span "generation.complete"
    (generateIds modelCfg runCfg.batch predict promptIds.toList runCfg.generate
      runCfg.temperature runCfg.topK runCfg.seed)
    (metadata :=
      { phase := some "generation"
        activity := some "autoregressive"
        stepIndex := some runCfg.generate })
  let output := text.GPT2BPE.decodeOrEmpty tokenizer outputIds.toArray
  IO.println "\n--- completion ---"
  IO.println output
  IO.println "------------------"

/--
Run optimization for any scalar objective over this causal Transformer.

The input pack is abstract: ordinary next-token training uses `(tokens, targets)`, while masked
training uses `(tokens, targets, rowWeights)`. Initialization, AdamW state, checkpointing,
evaluation, metrics, and profiling are shared.
-/
def setAdamWLearningRate {shapes : List Shape} (learningRate : Float) :
    _root_.Runtime.Autograd.TorchLean.Optim.StateList
      _root_.Optim.AdamW.State Float shapes →
    _root_.Runtime.Autograd.TorchLean.Optim.StateList
      _root_.Optim.AdamW.State Float shapes
  | .nil => .nil
  | .cons state rest =>
      .cons { state with lr := learningRate } (setAdamWLearningRate learningRate rest)

def optimizeObjective
    {inputShapes dataInputShapes : List Shape}
    (runCfg : RunConfig) (opts : Options)
    (cfg : nn.models.CausalTransformer.Config) [NeZero cfg.vocab]
    (model : nn.Sequential
      (nn.models.CausalTransformer.embeddingShape cfg [runCfg.batch])
      (nn.models.CausalTransformer.embeddingShape cfg [runCfg.batch]))
    (data : DatasetIdentity) (actualParameterCount : Nat)
    (trainDef : Module.ObjectiveDef (Fin cfg.vocab)
      (nn.models.CausalTransformer.Tied.stateShapes cfg model) inputShapes dataInputShapes)
    (evalDef : Module.ObjectiveDef (Fin cfg.vocab)
      (nn.models.CausalTransformer.Tied.stateShapes cfg model) inputShapes dataInputShapes)
    (trainSample validationSample : Nat →
      TensorPack Float inputShapes × TensorPack (Fin cfg.vocab) dataInputShapes) :
    IO (_root_.Runtime.Autograd.Torch.ParamList Float
      (nn.models.CausalTransformer.Tied.stateShapes cfg model)) := do
  let module ← LeanProfiler.span "model.initialize"
    (TorchLean.Module.instantiateAs (α := Float) trainDef id opts)
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
        (Checkpoint.loadModule module checkpoint)
        (metadata := { phase := some "checkpoint", activity := some "load" })

  let optimizer := _root_.Runtime.Autograd.TorchLean.Optim.adamw
    (α := Float) (paramShapes := nn.models.CausalTransformer.Tied.stateShapes cfg model)
    runCfg.learningRate runCfg.weightDecay 0.9 0.999 1e-8
  let optimizerState ← LeanProfiler.span "optimizer.initialize"
    (TorchLean.Module.initOptimizer module optimizer)
    (metadata :=
      { phase := some "initialization"
        activity := some "adamw"
        device := some opts.deviceName })
  let optimizerStateRef ← IO.mkRef optimizerState
  let metricPoints ← IO.mkRef (#[] : Array MetricPoint)
  let schedule := Trainer.Scheduler.warmupCosine
    runCfg.learningRate runCfg.minLearningRate runCfg.warmupSteps runCfg.steps
  let evaluator ← TorchLean.Module.evaluatorWithState
    evalDef opts module.trainer.state

  let startStep ← match runCfg.resume? with
    | none => pure 0
    | some checkpoint => do
        let (completed, directory) ← LeanProfiler.span "checkpoint.resume"
          (loadTrainingCheckpoint runCfg module data checkpoint)
          (metadata := { phase := some "checkpoint", activity := some "resume" })
        IO.println s!"resumed checkpoint: {directory} (completed_step={completed})"
        pure completed

  let evaluate (_step : Nat) : IO Float := do
    let losses ← (List.range runCfg.evalBatches).mapM fun batchIndex => do
      let (inputs, dataInputs) := validationSample batchIndex
      let loss ← TorchLean.Module.Evaluator.run evaluator inputs dataInputs
      pure (Spec.Tensor.item loss)
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
    let learningRate := Trainer.Scheduler.lrAt schedule step
    let state := setAdamWLearningRate learningRate state
    let startedAt ← IO.monoMsNow
    let (inputs, dataInputs) := trainSample step
    let (nextState, trainLossTensor) ← LeanProfiler.span "training.update"
      (TorchLean.Module.optimizerStepWithLoss module optimizer state inputs dataInputs)
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
        Float.ofNat (runCfg.batch * cfg.seqLen) * 1000.0 / Float.ofNat durationMs
    optimizerStateRef.set nextState
    let trainLoss := Spec.Tensor.item trainLossTensor
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
            (saveTrainingCheckpoint runCfg module data root completed points)
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
        (Checkpoint.saveModule module checkpoint)
        (metadata := { phase := some "checkpoint", activity := some "save" })
      IO.println s!"wrote checkpoint: {checkpoint}"

  let points ← metricPoints.get
  writeMetrics runCfg.metricsPath points
  writePassport runCfg.passportPath runCfg opts data actualParameterCount
  IO.println s!"wrote metrics: {runCfg.metricsPath}"
  IO.println s!"wrote passport: {runCfg.passportPath}"
  pure module.trainer.state

/-- Complete one training run after TorchLean has selected the runtime profile. -/
def train (opts : Options) (args : List String) : IO Unit := do
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
  if hSeq : cfg.seqLen = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero context after validation"
  else if hModel : cfg.dModel = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero model width after validation"
  else if hVocab : cfg.vocab = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero vocabulary after validation"
  else
    letI : NeZero cfg.vocab := ⟨hVocab⟩
    do
      _root_.TorchLean.rand.manualSeed runCfg.seed
      nn.withModel (buildModel cfg runCfg.batch hSeq hModel) fun model =>
        letI : NeZero cfg.vocab := ⟨hVocab⟩
        do
          let actualParameterCount :=
            (nn.models.CausalTransformer.Tied.stateShapes cfg model).foldl
              (fun total shape => total + Shape.size shape) 0

          IO.println s!"preset={runCfg.presetName}"
          IO.println <|
            s!"model=context {cfg.seqLen}, vocab {cfg.vocab}, width {cfg.dModel}, " ++
            s!"heads {cfg.numHeads}, layers {cfg.layers}"
          IO.println s!"parameters={actualParameterCount}"
          IO.println s!"device={opts.deviceName} profile={backendProfileName opts}"
          IO.println s!"tokens=train {trainTokens.size}, validation {valTokens.size}"
          IO.println <|
            s!"schedule=steps {runCfg.steps}, tokens/update {runCfg.batch * cfg.seqLen}, " ++
              s!"total {runCfg.steps * runCfg.batch * cfg.seqLen}"

          let trainedParams ← match trainMask?, valMask?, trainRecords?, valRecords? with
            | none, none, none, none =>
                let trainDef := nn.models.CausalTransformer.Tied.objective cfg model
                let evalDef :=
                  nn.models.CausalTransformer.Tied.objectiveWithMode .eval cfg model
                let trainSample (step : Nat) :=
                  let (tokens, targets) := causalLmTokenBatchFromShard
                    cfg.vocab runCfg.batch cfg.seqLen trainTokens runCfg.seed step (padId := 0)
                  (.nil, .cons tokens (.cons targets .nil))
                let validationSample (step : Nat) :=
                  let (tokens, targets) := causalLmTokenBatchFromShard
                    cfg.vocab runCfg.batch cfg.seqLen valTokens (runCfg.seed + 1000003) step (padId := 0)
                  (.nil, .cons tokens (.cons targets .nil))
                optimizeObjective runCfg opts cfg model
                  dataIdentity actualParameterCount
                  trainDef evalDef trainSample validationSample
            | some trainMask, some valMask, some trainRecords, some valRecords =>
                IO.println <|
                  s!"objective=dialogue-bounded weighted next-token loss, " ++
                    s!"records=train {trainRecords.entries.size}, validation {valRecords.entries.size}"
                let trainDef := weightedTiedTokenScalarModuleDef cfg runCfg.batch model
                let evalDef :=
                  weightedTiedTokenScalarModuleDefWithMode .eval cfg runCfg.batch model
                let trainSample (step : Nat) :=
                  let (tokens, targets, rowWeights) := causalLmMaskedTokenBatchFromRecords
                    (α := Float) cfg.vocab runCfg.batch cfg.seqLen trainTokens trainMask trainRecords
                    runCfg.seed step (padId := 0)
                  (.cons rowWeights .nil, .cons tokens (.cons targets .nil))
                let validationSample (step : Nat) :=
                  let (tokens, targets, rowWeights) := causalLmMaskedTokenBatchFromRecords
                    (α := Float) cfg.vocab runCfg.batch cfg.seqLen valTokens valMask valRecords
                    (runCfg.seed + 1000003) step (padId := 0)
                  (.cons rowWeights .nil, .cons tokens (.cons targets .nil))
                optimizeObjective runCfg opts cfg model
                  dataIdentity actualParameterCount
                  trainDef evalDef trainSample validationSample
            | _, _, _, _ =>
                throw <| IO.userError
                  s!"{exeName}: training masks and dialogue records must be supplied as paired sets"

          let forwardProgram : _root_.Runtime.Autograd.TorchLean.ProgramWithDataInputs
              Float (Fin cfg.vocab)
              (nn.models.CausalTransformer.Tied.stateShapes cfg model ++ [])
              [tokenShape cfg runCfg.batch] (logitShape cfg runCfg.batch) := by
            exact fun {m} _ _ =>
              by
                simpa [tokenShape, logitShape] using
                  nn.models.CausalTransformer.Tied.program cfg model (α := Float) (m := m)
          let evaluator ← TorchLean.Module.withState
            forwardProgram opts trainedParams
          let predict : Predictor cfg runCfg.batch := fun tokens =>
            TorchLean.Module.Evaluator.run evaluator .nil (.cons tokens .nil)
          let tokenizer? ← loadTokenizer? runCfg
          match tokenizer? with
          | none =>
              if runCfg.generate != 0 then
                IO.println "generation skipped: pass both --tokenizer-vocab and --tokenizer-merges"
          | some tokenizer =>
              if runCfg.generate != 0 then
                printCompletion runCfg cfg predict tokenizer

/-- Parse the shared runtime flags for commands that execute the host `Float` model. -/
def runFloatCommand
    (commandName : String) (args : List String) (banner : String)
    (command : Options → List String → IO Unit) : IO UInt32 := do
  let (seed, args) ← CLI.seed commandName args
  let (execConfig, rest) ← orThrowFor commandName <|
    TorchLean.Module.ExecConfig.parseWithScalar args .float32
  let opts ← orThrowFor commandName <| execConfig.toOptions seed
  opts.validateForExecution
  IO.println banner
  command opts rest
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
