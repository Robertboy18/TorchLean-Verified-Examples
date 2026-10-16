/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.CachedDecode.Generation

/-!
# Cached interactive generation

`generate_torchlean_gpt_cached` has the same checkpoint, tokenizer, transcript, and sampling
interface as `generate_torchlean_gpt`. Only execution differs. Each prompt token is evaluated once
with its keys and values kept, and each generated token adds one row to the cache.

The model uses learned absolute positions, so a conversation that outgrows the context is rejected
rather than silently cropped with positions restarted at zero.
-/

@[expose] public section

open TorchLean

namespace TorchLeanGPT
namespace CachedGenerate

def exeName : String := "generate_torchlean_gpt_cached"

def usage : String :=
  String.intercalate "\n"
    [ "Generate text with the Week 3 incremental key/value cache."
    , ""
    , "Usage:"
    , "  lake -R -K cuda=true exe generate_torchlean_gpt_cached --device cuda \\"
    , "    --preset gpt2-small --load-params FILE \\"
    , "    --tokenizer-vocab FILE --tokenizer-merges FILE"
    , ""
    , "Generation options:"
    , "  --generate N --temperature X --top-k N --seed N"
    , "  --repeat-penalty X --repeat-window N"
    , "                       subtract X per occurrence among the last N ids, prompt"
    , "                       included (default 1.05 over 64; 0 disables)"
    , "  --system TEXT        optional System: message; pass an empty string to omit it"
    , "  --message TEXT       answer once and exit; otherwise open a prompt loop"
    , ""
    , "Model overrides:"
    , "  --context N --vocab N --width N --heads N --layers N --dropout P"
    ]

/-- Decoded assistant text paired with the underlying generated suffix and timings. -/
structure Answer where
  text : String
  generation : CachedDecode.Generation.Result
  deriving Repr

/-- Generate one assistant turn with the cache, then decode and trim it. -/
def answer
    (config : Chat.Config)
    (decoder : CachedDecode.Runtime.Decoder)
    (tokenizer : text.GPT2BPE.Tokenizer)
    (deviceName : String)
    (promptIds : List Nat) : IO Answer := do
  let stopToken ← CLI.orThrow exeName <| Chat.endOfTextToken tokenizer
  let result ← LeanProfiler.span "chat.generate.cached"
    (CachedDecode.Generation.generateIds config decoder promptIds stopToken)
    (metadata :=
        { phase := some "generation"
          activity := some "kv-cache"
          stepIndex := config.generate?
          device := some deviceName })
  let decoded ← CLI.orThrow exeName <| text.GPT2BPE.decode tokenizer result.tokenIds.toArray
  pure { text := Chat.trimGeneratedTurn decoded, generation := result }

/-- Print one timing line after a response. -/
def printTiming (promptTokens : Nat) (answer : Answer) : IO Unit := do
  let result := answer.generation
  let prefillMs := Float.ofNat result.prefillNanos / 1.0e6
  let generationMs := Float.ofNat result.generationNanos / 1.0e6
  let rate :=
    CachedDecode.Generation.tokensPerSecond result.tokenIds.length result.generationNanos
  IO.println <|
    s!"[{promptTokens} prompt tokens in {prefillMs} ms; " ++
      s!"{result.tokenIds.length} generated tokens in {generationMs} ms; {rate} tok/s]"

/-- Re-encode the whole transcript at each user turn and prefill a fresh cache from it. -/
partial def loop
    (config : Chat.Config)
    (decoder : CachedDecode.Runtime.Decoder)
    (tokenizer : text.GPT2BPE.Tokenizer)
    (deviceName : String) : IO Unit := do
  IO.println "Enter :q or an empty line to stop."
  let stdin ← IO.getStdin
  let rec go (turns : List Chat.Turn) : IO Unit := do
    IO.print "you> "
    let line ← stdin.getLine
    let message := line.trimAscii.toString
    if message.isEmpty || message == ":q" || message == ":quit" then
      pure ()
    else
      let turnConfig := { config with seed := config.seed + turns.length }
      let promptIds ← CLI.orThrow exeName <|
        Chat.encodeDialoguePrompt tokenizer config.systemPrompt turns message
      let response ← answer turnConfig decoder tokenizer deviceName promptIds
      IO.println s!"model> {response.text}"
      printTiming promptIds.length response
      go (turns ++ [{ user := message, assistant := response.text }])
  go []

/-- Load the ordinary TorchLean model and borrow its checked checkpoint buffers for decoding. -/
def run (opts : Runtime.Config) (args : List String) : IO Unit := do
  let config ← Chat.Config.parse exeName opts.seed args
  let cfg := config.model.toTorchLean
  if !opts.usesCuda then
    throw <| IO.userError s!"{exeName}: the incremental cache currently requires --device cuda"
  else if cfg.sequenceLength = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero context after validation"
  else if cfg.modelWidth = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero model width after validation"
  else if hVocab : cfg.vocabularySize = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero vocabulary after validation"
  else
    letI : NeZero cfg.vocabularySize := ⟨hVocab⟩
    do
      _root_.TorchLean.rand.manualSeed config.seed
      let model := nn.build config.seed (buildModel cfg 1 opts)
      let actualParameterCount :=
        (model.stateShapes).foldl
          (fun total shape => total + Shape.size shape) 0
      let tokenizer ← LeanProfiler.span "tokenizer.load"
        (text.GPT2BPE.load config.tokenizerVocab config.tokenizerMerges
          (progress := true) (label := exeName))
        (metadata := { phase := some "data", activity := some "tokenizer" })
      let evalDef := nn.models.CausalTransformer.objective cfg model (mode := .eval)
      let runtimeModule ← LeanProfiler.span "model.initialize"
        (Module.instantiate evalDef opts (α := Float))
        (metadata :=
          { phase := some "initialization"
            activity := some "parameters"
            backend := some (Run.backendProfileName opts)
            dtype := some "Float"
            device := some opts.deviceName })
      LeanProfiler.span "checkpoint.load"
        (Checkpoint.load runtimeModule config.checkpoint)
        (metadata := { phase := some "checkpoint", activity := some "load" })
      let decoder ← LeanProfiler.span "decoder.initialize"
        (CachedDecode.Runtime.Decoder.initialize cfg
          (Module.Objective.Internal.runtime runtimeModule).trainer.state)
        (metadata :=
          { phase := some "initialization"
            activity := some "kv-cache"
            device := some opts.deviceName })
      IO.println s!"loaded {actualParameterCount} parameters on {opts.deviceName}"
      IO.println s!"cache capacity={cfg.sequenceLength}, layers={cfg.layerCount}, heads={cfg.headCount}"
      try
        match config.message? with
        | some message =>
            let promptIds ← CLI.orThrow exeName <|
              Chat.encodeDialoguePrompt tokenizer config.systemPrompt [] message
            let response ← answer config decoder tokenizer opts.deviceName promptIds
            IO.println response.text
            printTiming promptIds.length response
        | none =>
            loop config decoder tokenizer opts.deviceName
      finally
        decoder.close

/-- Program entrypoint. -/
def main (args : List String) : IO UInt32 := do
  if CLI.hasHelp args then
    IO.println usage
    return 0
  LeanProfiler.profileFromEnvironment "torchlean-gpt.cached-generate" <|
    Run.runFloatCommand exeName args "TorchLean GPT cached text generation" run (defaultSeed := 1337)

end CachedGenerate
end TorchLeanGPT

def main (args : List String) : IO UInt32 :=
  TorchLeanGPT.CachedGenerate.main args
