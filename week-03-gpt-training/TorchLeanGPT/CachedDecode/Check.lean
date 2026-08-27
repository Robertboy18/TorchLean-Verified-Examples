/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.Chat
public import TorchLeanGPT.CachedDecode.Runtime

/-!
# Cached/full-prefix differential check

This executable loads one checkpoint into one TorchLean module, then evaluates selected prompt
prefixes twice:

1. with TorchLean's ordinary padded tied-token forward program;
2. with the incremental decoder and its key/value cache.

The comparison is numerical rather than definitional. CUDA reductions may use a different order,
but a cache-layout, parameter-order, or position-index bug produces discrepancies far larger than
the small floating-point tolerance used here. The reported maximum difference is an observation on
the selected checkpoint and prefixes; it is not an upper bound for untested prefixes and is not, by
itself, a Lean proof of the native implementation.
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
    ]

/--
Summary of one pair of vocabulary-logit vectors.

`marginStable` records the sufficient condition `2 * maxAbs < fullMargin` used by
`Generation.StepCertified`. It is computed from this observed pair of arrays. The structure does
not certify how either array was produced.
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

/-- Difference between the largest reference score and the best competing score. -/
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
Compare finite arrays of equal length.

The function rejects NaNs and infinities before computing an observed infinity-norm difference,
the mean coordinate difference, both greedy choices, and the full-prefix winner margin.
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

/-- Pad one prefix exactly as the ordinary generator does. -/
def paddedPrefix (context : Nat) (tokens : List Nat) : Except String (List Nat) := do
  if tokens.isEmpty then
    throw "prompt prefix is empty"
  if tokens.length > context then
    throw s!"prompt prefix length {tokens.length} exceeds context {context}"
  pure <| tokens ++ List.replicate (context - tokens.length) 0

/-- Evaluate one prefix through the ordinary full-context model. -/
def fullPrefixScores
    (cfg : nn.models.CausalTransformer.Config) [NeZero cfg.vocab] [NeZero cfg.seqLen]
    (predict : Run.Predictor cfg 1)
    (tokens : List Nat) : IO (Array Float) := do
  let padded ← Chat.orThrow exeName <| paddedPrefix cfg.seqLen tokens
  let input := Run.tokenBatchTensor cfg 1 padded
  let logits ← predict input
  let first : Fin 1 := ⟨0, by decide⟩
  pure <| (text.batchLogitScoresAt logits first
    (Fin.ofNat cfg.seqLen (tokens.length - 1))).toArray

/-- Prefix lengths that exercise the cache at the beginning and at the complete prompt. -/
def checkLengths (promptLength : Nat) : List Nat :=
  [1, Nat.min 4 promptLength, promptLength].eraseDups.filter (fun n => n != 0)

/-- Run the differential check on one live module and decoder. -/
def checkPrefixes
    (cfg : nn.models.CausalTransformer.Config) [NeZero cfg.vocab] [NeZero cfg.seqLen]
    (predict : Run.Predictor cfg 1)
    (decoder : Runtime.Decoder)
    (prompt : List Nat) : IO Unit := do
  let tolerance := 0.001
  for length in checkLengths prompt.length do
    let tokens := prompt.take length
    let fullStart ← IO.monoNanosNow
    let full ← fullPrefixScores cfg predict tokens
    let fullFinish ← IO.monoNanosNow
    let cachedStart ← IO.monoNanosNow
    let cached ← decoder.prefill tokens
    let cachedFinish ← IO.monoNanosNow
    let difference ← Chat.orThrow exeName <| compareScores full cached
    let fullMs := Float.ofNat (fullFinish - fullStart) / 1.0e6
    let cachedMs := Float.ofNat (cachedFinish - cachedStart) / 1.0e6
    IO.println <|
      s!"prefix={length} max_abs={difference.maxAbs} mean_abs={difference.meanAbs} " ++
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
def run (opts : Options) (args : List String) : IO Unit := do
  let config ← Chat.Config.parse exeName args
  let cfg := config.model.toTorchLean
  if !opts.usesCuda then
    throw <| IO.userError s!"{exeName}: the incremental cache currently requires --device cuda"
  else if hSeq : cfg.seqLen = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero context after validation"
  else if hModel : cfg.dModel = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero model width after validation"
  else if hVocab : cfg.vocab = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero vocabulary after validation"
  else
    letI : NeZero cfg.vocab := ⟨hVocab⟩
    letI : NeZero cfg.seqLen := ⟨hSeq⟩
    do
      _root_.TorchLean.rand.manualSeed config.seed
      nn.withModel (buildModel cfg 1 hSeq hModel) fun model =>
        letI : NeZero cfg.vocab := ⟨hVocab⟩
        letI : NeZero cfg.seqLen := ⟨hSeq⟩
        do
          let tokenizer ←
            text.GPT2BPE.loadWithProgress exeName config.tokenizerVocab config.tokenizerMerges
          let evalDef := nn.models.CausalTransformer.Tied.objectiveWithMode .eval cfg model
          let runtimeModule ← TorchLean.Module.instantiateAs (α := Float) evalDef id opts
          Checkpoint.loadModule runtimeModule config.checkpoint
          let forwardProgram : _root_.Runtime.Autograd.TorchLean.ProgramWithDataInputs
              Float (Fin cfg.vocab)
              (nn.models.CausalTransformer.Tied.stateShapes cfg model ++ [])
              [tokenShape cfg 1] (logitShape cfg 1) := by
            exact fun {m} _ _ => by
              simpa [tokenShape, logitShape] using
                nn.models.CausalTransformer.Tied.program cfg model (α := Float) (m := m)
          let evaluator ← TorchLean.Module.withState
            forwardProgram opts runtimeModule.trainer.state
          let predict : Run.Predictor cfg 1 := fun tokens =>
            TorchLean.Module.Evaluator.run evaluator .nil (.cons tokens .nil)
          let decoder ← Runtime.Decoder.initialize cfg runtimeModule.trainer.state
          let message := config.message?.getD "What is two plus two?"
          let prompt ← Chat.orThrow exeName <|
            Chat.encodeDialoguePrompt tokenizer config.systemPrompt [] message
          try
            checkPrefixes cfg predict decoder prompt
            IO.println "cached/full-prefix comparison passed"
          finally
            decoder.close

/-- Program entrypoint. -/
def main (args : List String) : IO UInt32 := do
  if CLI.hasHelp args then
    IO.println usage
    return 0
  Run.runFloatCommand exeName args "TorchLean GPT cached-decoder check" run

end Check
end CachedDecode
end TorchLeanGPT

def main (args : List String) : IO UInt32 :=
  TorchLeanGPT.CachedDecode.Check.main args
