/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.Run
import NN.API.Text.Bpe

/-!
# Shared dialogue interface

The full-prefix and cached generators use the same model options, GPT-2 tokenizer, transcript
format, and stopping rule. Keeping those choices here makes execution strategy the only difference
between the two commands.
-/

@[expose] public section

open TorchLean

namespace TorchLeanGPT
namespace Chat

/-- Inference settings. No training shard is needed. -/
structure Config where
  presetName : String
  model : ModelConfig
  seed : Nat
  checkpoint : System.FilePath
  tokenizerVocab : System.FilePath
  tokenizerMerges : System.FilePath
  /-- Continuation length. `none` fills the remaining context, up to `defaultGenerate` tokens. -/
  generate? : Option Nat
  temperature : Float
  topK : Nat
  repeatPenalty : Float
  repeatWindow : Nat
  systemPrompt : String
  message? : Option String
  deriving Repr

/-- Longest continuation chosen when `--generate` is omitted. -/
def defaultGenerate : Nat := 96

/-- Parse the options shared by both generation executables. -/
def Config.parse (command : String) (seed : Nat) (args : List String) : IO Config := do
  let (presetName, args) ← CLI.orThrow command <|
    CLI.takeFlagValue args "preset" "quick"
  let preset ← CLI.orThrow command <| ModelConfig.ofName presetName
  let (model, args) ← Run.parseModelConfigFor command args preset
  let (checkpoint, args) ← CLI.orThrow command <|
    CLI.requirePathFlag args "load-params" command
  let (tokenizerVocab, args) ← CLI.orThrow command <|
    CLI.requirePathFlag args "tokenizer-vocab" command
  let (tokenizerMerges, args) ← CLI.orThrow command <|
    CLI.requirePathFlag args "tokenizer-merges" command
  let (generate?, args) ← CLI.orThrow command <| CLI.takeNatFlag? args "generate"
  let (temperature, args) ← CLI.orThrow command <|
    CLI.takePositiveFloatFlag args command "temperature" 0.8
  let (topK, args) ← CLI.orThrow command <| CLI.takeNatFlag args "top-k" 40
  let (repeatPenalty, args) ← CLI.orThrow command <|
    CLI.takeNonnegativeFloatFlag args command "repeat-penalty" Run.defaultRepeatPenalty
  let (repeatWindow, args) ← CLI.orThrow command <|
    CLI.takeNatFlag args "repeat-window" Run.defaultRepeatWindow
  let (systemPrompt, args) ← CLI.orThrow command <|
    CLI.takeFlagValue args "system"
      "A conversation between a user and a helpful assistant."
  let (message?, args) ← CLI.orThrow command <| CLI.takeFlagValue? args "message"
  CLI.orThrow command <| CLI.checkNoArgs args
  CLI.orThrow command <| Run.requireFinite "temperature" temperature
  pure
    { presetName
      model
      seed
      checkpoint
      tokenizerVocab
      tokenizerMerges
      generate?
      temperature
      topK
      repeatPenalty
      repeatWindow
      systemPrompt
      message? }

/--
Number of tokens to generate after a prompt of `promptLength` tokens.

A `--generate` value is returned unchanged (the generators reject it if it does not fit). Without
the flag, the continuation fills the remaining context, up to `defaultGenerate` tokens, so the
default also works for the 16-token quick preset.
-/
def Config.continuationLength (config : Config) (context promptLength : Nat) :
    Except String Nat :=
  match config.generate? with
  | some steps => pure steps
  | none =>
      if promptLength < context then
        pure (min defaultGenerate (context - promptLength))
      else
        throw <|
          s!"prompt ({promptLength} tokens) leaves no room in the {context}-token context; " ++
            "shorten the message or pass --system \"\""

/-- One completed user/assistant exchange retained as dialogue context. -/
structure Turn where
  user : String
  assistant : String
  deriving Repr

/--
Encode each text segment separately and concatenate the token ids.

The instruction-data writer (`tools/prepare_dataset.py`) splits role headers, message bodies, and
newlines at the same boundaries, so BPE merges at a boundary cannot differ between fine-tuning and
interactive generation.
-/
def encodeSegments
    (tokenizer : text.GPT2BPE.Tokenizer) (segments : List String) :
    Except String (List Nat) := do
  let pieces ← segments.mapM (text.GPT2BPE.encode tokenizer)
  pure (pieces.flatMap Array.toList)

/-- Segments for one message in the transcript format used by the dataset writer. -/
def messageSegments (role content : String) : List String :=
  [role ++ ": ", content.trimAscii.toString, "\n"]

/--
Encode a dialogue prefix ending right after the next `Assistant: ` header.

A blank `systemPrompt` drops the `System:` line, so the transcript starts with `User:`, as in
instruction datasets without a system message.
-/
def encodeDialoguePrompt
    (tokenizer : text.GPT2BPE.Tokenizer)
    (systemPrompt : String) (turns : List Turn) (userMessage : String) :
    Except String (List Nat) := do
  let prior :=
    turns.flatMap fun turn =>
      messageSegments "User" turn.user ++
        messageSegments "Assistant" turn.assistant
  let system :=
    if systemPrompt.trimAscii.isEmpty then
      []
    else
      messageSegments "System" systemPrompt
  encodeSegments tokenizer <|
    system ++
      prior ++
      messageSegments "User" userMessage ++
      ["Assistant: "]

/--
Id of the end-of-text marker, looked up in the loaded vocabulary. Encoding its spelling as
ordinary text would split it into several tokens instead of selecting the special entry.
-/
def endOfTextToken (tokenizer : text.GPT2BPE.Tokenizer) : Except String Nat := do
  match tokenizer.tokenId? "<|endoftext|>" with
  | some token => pure token
  | none => throw "GPT-2 vocabulary has no <|endoftext|> token"

/-- Keep the tokens strictly before the first occurrence of `stopToken`. -/
def beforeToken (stopToken : Nat) (tokens : List Nat) : List Nat :=
  tokens.takeWhile (· != stopToken)

/--
Cut a sampled assistant response before an end-of-text marker or an invented next user turn.

Token-level generation usually stops at the end-of-text token already. The text check also keeps a
generated `User:` delimiter out of the displayed answer.
-/
def trimGeneratedTurn (text : String) : String :=
  let beforeEnd := (text.splitOn "<|endoftext|>").headD text
  (beforeEnd.splitOn "\nUser:").headD beforeEnd |>.trimAscii.toString

end Chat
end TorchLeanGPT
