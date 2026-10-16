/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.Chat
public import TorchLeanGPT.CachedDecode.Generation

/-!
# Cached/full-prefix differential check

`check_torchlean_gpt_cache` loads one checkpoint into one TorchLean module, then evaluates selected
prompt prefixes twice:

1. with TorchLean's ordinary padded tied-token forward program
2. with the incremental decoder and its key/value cache

Both score vectors first get the repeat penalty the generators apply before choosing a token
(`--repeat-penalty`, default 1.05 over the last 64 ids). The penalty subtracts the same amounts from
both vectors, so it leaves their difference unchanged but can move the winner and its margin. The
reported argmax and margin are those of the scores the decoder actually ranks, and with
`--repeat-penalty 0` the check is on raw logits.

Because CUDA reductions may run in a different order, the comparison allows a small floating-point
tolerance. A cache-layout, parameter-order, or position-index bug would produce discrepancies far
larger than that. The reported maximum difference is a measurement on the checked checkpoint and
prefixes, so it bounds nothing else and proves nothing about the native code.
-/

@[expose] public section

open TorchLean

namespace TorchLeanGPT
namespace CachedDecode
namespace Check

def exeName : String := "check_torchlean_gpt_cache"

def usage : String :=
  String.intercalate "\n"
    [ "Compare cached GPT logits with TorchLean's ordinary full-prefix forward pass."
    , ""
    , "Usage:"
    , "  lake -R -K cuda=true exe check_torchlean_gpt_cache --device cuda \\"
    , "    --preset gpt2-small --load-params FILE \\"
    , "    --tokenizer-vocab FILE --tokenizer-merges FILE --message TEXT"
    , ""
    , "The command checks prefix lengths 1, 4, and the complete encoded prompt when available."
    , "Scores are compared after the decoder's repeat penalty (--repeat-penalty X,"
    , "--repeat-window N; default 1.05 over 64 ids). Pass --repeat-penalty 0 for raw logits."
    ]

/--
Summary of one pair of vocabulary-score vectors.

`marginStable` is the sufficient condition `2 * maxAbs < fullMargin` of `Generation.StepCertified`,
evaluated on this observed pair. It certifies nothing about how either array was produced.
-/
structure Difference where
  maxAbs : Float
  meanAbs : Float
  fullArgmax : Nat
  cachedArgmax : Nat
  fullMargin : Float
  marginStable : Bool
  deriving Repr

/-- Index of the largest score, choosing the first index on ties. -/
def argmax (scores : Array Float) : Nat :=
  if scores.isEmpty then
    0
  else
    (List.range scores.size).foldl
      (fun best index =>
        if scores[index]! > scores[best]! then
          index
        else
          best)
      0

/-- Gap between the largest score and the runner-up, or zero for fewer than two scores. -/
def argmaxMargin (scores : Array Float) : Float :=
  if scores.size < 2 then
    0.0
  else
    let winner := argmax scores
    let firstOther := if winner = 0 then 1 else 0
    let runnerUpIndex := (List.range scores.size).foldl
      (fun bestIndex index =>
        if index = winner then
          bestIndex
        else
          if scores[index]! > scores[bestIndex]! then index else bestIndex)
      firstOther
    scores[winner]! - scores[runnerUpIndex]!

/--
Compare two nonempty score arrays of equal length.

Any NaN or infinity is an error. Otherwise the result holds the observed infinity-norm and mean
coordinate differences, both greedy choices, and the full-prefix winner margin.
-/
def compareScores (full cached : Array Float) : Except String Difference := do
  if full.size != cached.size then
    throw s!"score length mismatch (full={full.size}, cached={cached.size})"
  if full.isEmpty then
    throw "cannot compare empty score arrays"
  let mut maximum := 0.0
  let mut total := 0.0
  for index in [0:full.size] do
    let left := full[index]!
    let right := cached[index]!
    if left.isNaN || left.isInf || right.isNaN || right.isInf then
      throw s!"non-finite score at vocabulary index {index}"
    let difference := Float.abs (left - right)
    if difference > maximum then
      maximum := difference
    total := total + difference
  let fullArgmax := argmax full
  let fullMargin := argmaxMargin full
  pure
    { maxAbs := maximum
      meanAbs := total / Float.ofNat full.size
      fullArgmax
      cachedArgmax := argmax cached
      fullMargin
      marginStable := 2.0 * maximum < fullMargin }

/--
Apply the generators' repeat penalty for the history `tokens`, using TorchLean's own
`penalizeRepeats` so the scores match the ones `text.next` ranks.
-/
def penalizedScores (config : Chat.Config) (tokens : List Nat) (scores : Array Float) :
    Array Float :=
  let recent := (Generation.recentTokens tokens config.repeatWindow).toArray
  let recentTensor : Tensor Nat [recent.size] := Tensor.ofFn fun i => recent[i]
  let scoreTensor : Tensor Float [scores.size] :=
    Tensor.from (⟨scores, rfl⟩ : Vector Float scores.size)
  (text.Internal.penalizeRepeats scoreTensor recentTensor config.repeatPenalty).to (Array Float)

/--
Render a small difference in scientific notation. `Float`'s `toString` keeps six decimals, which
prints every binary64 difference of this check as `0.000000`.
-/
def scientific (x : Float) : String :=
  if x == 0.0 || x.isNaN || x.isInf then toString x
  else
    let exponent := (Float.log10 (Float.abs x)).floor
    s!"{x / Float.pow 10.0 exponent}e{exponent.toInt64}"

/-- Zero-pad a nonempty prefix to the context length, as the ordinary generator does. -/
def paddedPrefix (context : Nat) (tokens : List Nat) : Except String (List Nat) := do
  if tokens.isEmpty then
    throw "prompt prefix is empty"
  if tokens.length > context then
    throw s!"prompt prefix length {tokens.length} exceeds context {context}"
  pure <| tokens ++ List.replicate (context - tokens.length) 0

/-- Evaluate one prefix through the ordinary full-context model. -/
def fullPrefixScores
    (cfg : nn.models.CausalTransformer.Config) [NeZero cfg.vocabularySize] [NeZero cfg.sequenceLength]
    (predict : Run.Predictor cfg 1)
    (tokens : List Nat) : IO (Array Float) := do
  let padded ← CLI.orThrow exeName <| paddedPrefix cfg.sequenceLength tokens
  let input := Run.tokenBatchTensor cfg 1 padded
  let logits ← predict input
  let first : Fin 1 := ⟨0, by decide⟩
  pure <| (Tensor.get (Tensor.get logits first)
    (Fin.ofNat cfg.sequenceLength (tokens.length - 1))).to (Array Float)

/-- Prefix lengths that exercise the cache at the beginning and at the complete prompt. -/
def checkLengths (promptLength : Nat) : List Nat :=
  [1, Nat.min 4 promptLength, promptLength].eraseDups.filter (fun n => n != 0)

/-- Run the differential check on one live module and decoder. -/
def checkPrefixes
    (config : Chat.Config)
    (cfg : nn.models.CausalTransformer.Config) [NeZero cfg.vocabularySize] [NeZero cfg.sequenceLength]
    (predict : Run.Predictor cfg 1)
    (decoder : Runtime.Decoder)
    (prompt : List Nat) : IO Unit := do
  let tolerance := 0.001
  IO.println <|
    s!"repeat_penalty={config.repeatPenalty} repeat_window={config.repeatWindow} " ++
      "(argmax and margin are for the penalized scores the decoder ranks)"
  for length in checkLengths prompt.length do
    let tokens := prompt.take length
    let fullStart ← IO.monoNanosNow
    let full ← fullPrefixScores cfg predict tokens
    let fullFinish ← IO.monoNanosNow
    let cachedStart ← IO.monoNanosNow
    let cached ← decoder.prefill tokens
    let cachedFinish ← IO.monoNanosNow
    let difference ← CLI.orThrow exeName <|
      compareScores (penalizedScores config tokens full) (penalizedScores config tokens cached)
    let fullMs := Float.ofNat (fullFinish - fullStart) / 1.0e6
    let cachedMs := Float.ofNat (cachedFinish - cachedStart) / 1.0e6
    IO.println <|
      s!"prefix={length} max_abs={scientific difference.maxAbs} " ++
        s!"mean_abs={scientific difference.meanAbs} " ++
        s!"argmax={difference.fullArgmax}/{difference.cachedArgmax} " ++
        s!"margin={difference.fullMargin} margin_stable={difference.marginStable} " ++
        s!"full_ms={fullMs} cached_ms={cachedMs}"
    if difference.maxAbs > tolerance then
      throw <| IO.userError <|
        s!"{exeName}: cached logits exceeded absolute tolerance {tolerance} at prefix {length}"
    if difference.fullArgmax != difference.cachedArgmax then
      throw <| IO.userError <|
        s!"{exeName}: cached logits changed the greedy token at prefix {length}"

/-- Load one checkpoint and compare both execution paths without copying its parameters. -/
def run (opts : Runtime.Config) (args : List String) : IO Unit := do
  let config ← Chat.Config.parse exeName opts.seed args
  let cfg := config.model.toTorchLean
  if !opts.usesCuda then
    throw <| IO.userError s!"{exeName}: the incremental cache currently requires --device cuda"
  else if hSeq : cfg.sequenceLength = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero context after validation"
  else if cfg.modelWidth = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero model width after validation"
  else if hVocab : cfg.vocabularySize = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero vocabulary after validation"
  else
    letI : NeZero cfg.vocabularySize := ⟨hVocab⟩
    letI : NeZero cfg.sequenceLength := ⟨hSeq⟩
    do
      _root_.TorchLean.rand.manualSeed config.seed
      let model := nn.build config.seed (buildModel cfg 1 opts)
      let tokenizer ←
        text.GPT2BPE.load config.tokenizerVocab config.tokenizerMerges
          (progress := true) (label := exeName)
      let evalDef := nn.models.CausalTransformer.objective cfg model (mode := .eval)
      let runtimeModule ← Module.instantiate evalDef opts (α := Float)
      Checkpoint.load runtimeModule config.checkpoint
      let predict ← runtimeModule.indexedPredictor model
      let decoder ← Runtime.Decoder.initialize cfg
        (Module.Objective.Internal.runtime runtimeModule).trainer.state
      let message := config.message?.getD "What is two plus two?"
      let prompt ← CLI.orThrow exeName <|
        Chat.encodeDialoguePrompt tokenizer config.systemPrompt [] message
      try
        checkPrefixes config cfg predict decoder prompt
        IO.println "cached/full-prefix comparison passed"
      finally
        decoder.close

/-- Program entrypoint. -/
def main (args : List String) : IO UInt32 := do
  if CLI.hasHelp args then
    IO.println usage
    return 0
  Run.runFloatCommand exeName args "TorchLean GPT cached-decoder check" run (defaultSeed := 1337)

end Check
end CachedDecode
end TorchLeanGPT

def main (args : List String) : IO UInt32 :=
  TorchLeanGPT.CachedDecode.Check.main args
