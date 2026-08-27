/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.Chat

/-!
# Interactive text interface

This executable loads a shape-checked TorchLean checkpoint and runs autoregressive decoding with
the same causal Transformer used by `train_torchlean_gpt`.

A pretrained language model completes text. It becomes a useful conversational model only after
instruction or dialogue fine-tuning. The interface below supplies a conventional transcript
format, but it does not mislabel a base Tiny Shakespeare checkpoint as a chat model.
-/

@[expose] public section

open TorchLean

namespace TorchLeanGPT
namespace Generate

def exeName : String := "generate_torchlean_gpt"

def usage : String :=
  String.intercalate "\n"
    [ "Load a TorchLean GPT checkpoint and generate text interactively."
    , ""
    , "Usage:"
    , "  lake -R -K cuda=true exe generate_torchlean_gpt --device cuda \\"
    , "    --load-params FILE --tokenizer-vocab FILE --tokenizer-merges FILE"
    , ""
    , "Options:"
    , "  --preset quick|gpt2-small"
    , "  --context N --vocab N --width N --heads N --layers N --dropout P"
    , "  --generate N --temperature X --top-k N --seed N"
    , "  --system TEXT        optional System: message; pass an empty string to omit it"
    , "  --message TEXT       answer once and exit; otherwise open a prompt loop"
    , ""
    , "A base checkpoint performs completion. Dialogue quality requires a checkpoint"
    , "fine-tuned on the same User:/Assistant: transcript format."
    ]

/-- Sample only the newly generated suffix, not the transcript supplied as context. -/
def answer
    (config : Chat.Config)
    (modelCfg : nn.models.CausalTransformer.Config) [NeZero modelCfg.vocab]
    (predict : Run.Predictor modelCfg 1)
    (tokenizer : text.GPT2BPE.Tokenizer)
    (promptIds : List Nat) : IO String := do
  let stopToken ← Chat.orThrow exeName <| Chat.endOfTextToken tokenizer
  let allIds ← LeanProfiler.span "chat.generate"
    (Run.generateIds modelCfg 1 predict promptIds config.generate
      config.temperature config.topK config.seed (some stopToken))
    (metadata :=
      { phase := some "generation"
        activity := some "interactive-chat"
        stepIndex := some config.generate })
  let suffix := Chat.beforeToken stopToken (allIds.drop promptIds.length)
  pure <| Chat.trimGeneratedTurn (text.GPT2BPE.decodeOrEmpty tokenizer suffix.toArray)

/-- Maintain a transcript across turns while keeping model execution purely autoregressive. -/
partial def loop
    (config : Chat.Config)
    (modelCfg : nn.models.CausalTransformer.Config) [NeZero modelCfg.vocab]
    (predict : Run.Predictor modelCfg 1)
    (tokenizer : text.GPT2BPE.Tokenizer) : IO Unit := do
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
      let promptIds ← Chat.orThrow exeName <|
        Chat.encodeDialoguePrompt tokenizer config.systemPrompt turns message
      let response ← answer turnConfig modelCfg predict tokenizer promptIds
      IO.println s!"model> {response}"
      go (turns ++ [{ user := message, assistant := response }])
  go []

/-- Load the model and checkpoint, then run one response or enter the interactive loop. -/
def run (opts : Options) (args : List String) : IO Unit := do
  let config ← Chat.Config.parse exeName args
  let cfg := config.model.toTorchLean
  if hSeq : cfg.seqLen = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero context after validation"
  else if hModel : cfg.dModel = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero model width after validation"
  else if hVocab : cfg.vocab = 0 then
    throw <| IO.userError s!"{exeName}: impossible zero vocabulary after validation"
  else
    letI : NeZero cfg.vocab := ⟨hVocab⟩
    do
      _root_.TorchLean.rand.manualSeed config.seed
      nn.withModel (buildModel cfg 1 hSeq hModel) fun model =>
        letI : NeZero cfg.vocab := ⟨hVocab⟩
        do
          let actualParameterCount :=
            (nn.models.CausalTransformer.Tied.stateShapes cfg model).foldl
              (fun total shape => total + Shape.size shape) 0
          let tokenizer ← LeanProfiler.span "tokenizer.load"
            (text.GPT2BPE.loadWithProgress exeName config.tokenizerVocab config.tokenizerMerges)
            (metadata := { phase := some "data", activity := some "tokenizer" })
          let evalDef := nn.models.CausalTransformer.Tied.objectiveWithMode .eval cfg model
          let runtimeModule ← LeanProfiler.span "model.initialize"
            (TorchLean.Module.instantiateAs (α := Float) evalDef id opts)
            (metadata :=
              { phase := some "initialization"
                activity := some "parameters"
                backend := some (Run.backendProfileName opts)
                dtype := some "Float"
                device := some opts.deviceName })
          LeanProfiler.span "checkpoint.load"
            (Checkpoint.loadModule runtimeModule config.checkpoint)
            (metadata := { phase := some "checkpoint", activity := some "load" })
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
          IO.println s!"loaded {actualParameterCount} parameters on {opts.deviceName}"
          match config.message? with
          | some message =>
              let promptIds ← Chat.orThrow exeName <|
                Chat.encodeDialoguePrompt tokenizer config.systemPrompt [] message
              let response ← answer config cfg predict tokenizer promptIds
              IO.println response
          | none =>
              loop config cfg predict tokenizer

/-- Program entrypoint. -/
def main (args : List String) : IO UInt32 := do
  if CLI.hasHelp args then
    IO.println usage
    return 0
  LeanProfiler.profileFromEnvironment "torchlean-gpt.generate" <|
    Run.runFloatCommand exeName args "TorchLean GPT text generation" run

end Generate
end TorchLeanGPT

def main (args : List String) : IO UInt32 :=
  TorchLeanGPT.Generate.main args
