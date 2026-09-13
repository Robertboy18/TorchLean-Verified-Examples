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

/-- Settings needed for inference; no training shard is required. -/
structure Config where
  presetName : String
  model : ModelConfig
  seed : Nat
  checkpoint : System.FilePath
  tokenizerVocab : System.FilePath
  tokenizerMerges : System.FilePath
  generate : Nat
  temperature : Float
  topK : Nat
  systemPrompt : String
  message? : Option String
  deriving Repr

/-- Lift a pure parser error into a diagnostic carrying the active command name. -/
def orThrow {α : Type} (command : String) (result : Except String α) : IO α :=
  match result with
  | .ok value => pure value
  | .error message => throw <| IO.userError s!"{command}: {message}"

/--
Parse the options shared by both generation executables.

The runtime parser has already consumed `--seed`; use its selected value for model initialization
and sampling as well.
-/
def Config.parse (command : String) (seed : Nat) (args : List String) : IO Config := do
  let (presetName, args) ← orThrow command <|
    CLI.takeFlagValue args "preset" "quick"
  let preset ← orThrow command <| ModelConfig.ofName presetName
  let (model, args) ← Run.parseModelConfigFor command args preset
  let (checkpoint, args) ← orThrow command <|
    CLI.requirePathFlag args "load-params" command
  let (tokenizerVocab, args) ← orThrow command <|
    CLI.requirePathFlag args "tokenizer-vocab" command
  let (tokenizerMerges, args) ← orThrow command <|
    CLI.requirePathFlag args "tokenizer-merges" command
  let (generate, args) ← orThrow command <| CLI.takeNatFlag args "generate" 96
  let (temperature, args) ← orThrow command <|
    CLI.takePositiveFloatFlag args command "temperature" 0.8
  let (topK, args) ← orThrow command <| CLI.takeNatFlag args "top-k" 40
  let (systemPrompt, args) ← orThrow command <|
    CLI.takeFlagValue args "system"
      "A conversation between a user and a helpful assistant."
  let (message?, args) ← orThrow command <| CLI.takeFlagValue? args "message"
  orThrow command <| CLI.checkNoArgs args
  orThrow command <| Run.requireFinite "temperature" temperature
  pure
    { presetName
      model
      seed
      checkpoint
      tokenizerVocab
      tokenizerMerges
      generate
      temperature
      topK
      systemPrompt
      message? }

/-- One completed user/assistant exchange retained as dialogue context. -/
structure Turn where
  user : String
  assistant : String
  deriving Repr

/--
Encode text segments independently and concatenate their token ids.

The instruction-data writer uses the same boundaries around role headers, message bodies, and
newlines. Keeping those boundaries explicit prevents tokenizer behavior at a concatenation point
from drifting between fine-tuning and interactive generation.
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
Encode a dialogue prefix ending immediately after the next `Assistant: ` header.

When `systemPrompt` is empty, the transcript begins directly with `User:`. This matches instruction
datasets that do not carry a system message. The response itself is generated autoregressively.
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
Find the end-of-text marker in the vocabulary loaded by the executable.

The marker names a special vocabulary entry. Encoding its spelling as ordinary text runs the BPE
pre-tokenizer, which can split the punctuation and letters into several tokens. Instead, inspect
the text represented by each individual token and select the exact marker. This also supports
tokenizers whose end-of-text id differs from the usual GPT-2 id of 50256.

Some byte-level tokens contain only part of a UTF-8 character and cannot be decoded on their own.
Those entries cannot be this ASCII marker, so they can be skipped during the search.
-/
def endOfTextToken (tokenizer : text.GPT2BPE.Tokenizer) : Except String Nat := do
  for token in [:tokenizer.vocabularySize] do
    match text.GPT2BPE.decode tokenizer #[token] with
    | .ok "<|endoftext|>" => return token
    | _ => pure ()
  throw "GPT-2 tokenizer vocabulary does not contain <|endoftext|>"

/-- Keep the tokens strictly before the first occurrence of `stopToken`. -/
def beforeToken (stopToken : Nat) : List Nat → List Nat
  | List.nil => []
  | List.cons token rest =>
      if token = stopToken then
        []
      else
        token :: beforeToken stopToken rest

/--
Stop a sampled assistant response before an end-of-text marker or an invented next user turn.

Token-level generation normally removes the end-of-text token first. The text check also prevents
a generated transcript delimiter from being displayed as part of the assistant's answer.
-/
def trimGeneratedTurn (text : String) : String :=
  let beforeEnd := (text.splitOn "<|endoftext|>").headD text
  (beforeEnd.splitOn "\nUser:").headD beforeEnd |>.trimAscii.toString

end Chat
end TorchLeanGPT
