/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.CachedDecode.Generation

/-!
# End-to-end decoding benchmark

The benchmark loads one model and one checkpoint, then generates the same number of tokens through
the ordinary full-context evaluator and the incremental cache. Tokenization and checkpoint loading
are outside the timed regions. The result reports complete autoregressive latency rather than
isolated kernel timing.
-/

@[expose] public section

open TorchLean

namespace TorchLeanGPT
namespace CachedDecode
namespace Benchmark

def exeName : String := "benchmark_torchlean_gpt_cache"

def usage : String :=
  String.intercalate "\n"
    [ "Benchmark full-prefix and cached TorchLean GPT decoding."
    , ""
    , "Usage:"
    , "  lake -R -K cuda=true exe benchmark_torchlean_gpt_cache --device cuda \\"
    , "    --preset gpt2-small --load-params FILE \\"
    , "    --tokenizer-vocab FILE --tokenizer-merges FILE \\"
    , "    --message TEXT --generate 16 --top-k 1"
    , ""
    , "Use --top-k 1 when exact generated-token agreement is part of the check."
    ]

/-- Convert nanoseconds to milliseconds for the report. -/
def milliseconds (nanos : Nat) : Float :=
  Float.ofNat nanos / 1.0e6

/-- Load one checkpoint, then time both autoregressive evaluators. -/
def run (opts : Options) (args : List String) : IO Unit := do
  let config ← Chat.Config.parse exeName args
  if config.generate = 0 then
    throw <| IO.userError s!"{exeName}: --generate must be positive"
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
          let message := config.message?.getD "Hello!"
          let prompt ← Chat.orThrow exeName <|
            Chat.encodeDialoguePrompt tokenizer config.systemPrompt [] message
          let stopToken ← Chat.orThrow exeName <| Chat.endOfTextToken tokenizer
          if prompt.length + config.generate > cfg.seqLen then
            decoder.close
            throw <| IO.userError <|
              s!"{exeName}: prompt and continuation exceed context length {cfg.seqLen}"
          try
            -- Warm both execution paths before measuring allocation and kernel latency.
            let _ ← decoder.prefill prompt
            let warmPadded := prompt ++ List.replicate (cfg.seqLen - prompt.length) 0
            let warmInput := Run.tokenBatchTensor cfg 1 warmPadded
            let _ ← predict warmInput

            let fullStart ← IO.monoNanosNow
            let fullIds ← Run.generateIds cfg 1 predict prompt config.generate
              config.temperature config.topK config.seed (some stopToken)
            let fullFinish ← IO.monoNanosNow
            let fullNanos := fullFinish - fullStart
            let fullSuffix :=
              Chat.beforeToken stopToken (fullIds.drop prompt.length)

            let cached ← Generation.generateIds config decoder prompt stopToken
            let cachedNanos := cached.prefillNanos + cached.generationNanos
            let speedup :=
              if cachedNanos = 0 then 0.0
              else Float.ofNat fullNanos / Float.ofNat cachedNanos
            let fullRate := Generation.tokensPerSecond fullSuffix.length fullNanos
            let cachedRate := Generation.tokensPerSecond cached.tokenIds.length cachedNanos

            IO.println <|
              s!"prompt_tokens={prompt.length} requested_tokens={config.generate} " ++
                s!"generated_tokens={cached.tokenIds.length}"
            IO.println s!"full_prefix_ms={milliseconds fullNanos} full_prefix_tok_s={fullRate}"
            IO.println <|
              s!"cached_prefill_ms={milliseconds cached.prefillNanos} " ++
                s!"cached_generation_ms={milliseconds cached.generationNanos} " ++
                s!"cached_total_tok_s={cachedRate}"
            IO.println s!"end_to_end_speedup={speedup}x"
            IO.println s!"generated_tokens_equal={fullSuffix == cached.tokenIds}"
            if config.topK = 1 && fullSuffix != cached.tokenIds then
              throw <| IO.userError
                s!"{exeName}: greedy full-prefix and cached continuations differ"
          finally
            decoder.close

/-- Program entrypoint. -/
def main (args : List String) : IO UInt32 := do
  if CLI.hasHelp args then
    IO.println usage
    return 0
  Run.runFloatCommand exeName args "TorchLean GPT decoding benchmark" run

end Benchmark
end CachedDecode
end TorchLeanGPT

def main (args : List String) : IO UInt32 :=
  TorchLeanGPT.CachedDecode.Benchmark.main args
