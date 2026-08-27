/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.Chat
public import TorchLeanGPT.CachedDecode.Runtime

/-!
# Sampling from a cached decoder

This module contains the execution-independent part of cached generation. The interactive command
and the benchmark both use it, so they cannot drift in repeat penalty, top-k sampling, context
handling, or timing boundaries.
-/

@[expose] public section

open TorchLean

namespace TorchLeanGPT
namespace CachedDecode
namespace Generation

/-- Generated suffix and wall-clock timing for one prompt. -/
structure Result where
  tokenIds : List Nat
  prefillNanos : Nat
  generationNanos : Nat
  deriving Repr

/-- Render tokens per second without claiming kernel-only timing. -/
def tokensPerSecond (tokens nanos : Nat) : Float :=
  if nanos = 0 then
    0.0
  else
    Float.ofNat tokens * 1.0e9 / Float.ofNat nanos

/-- Repeat-penalty window used by the ordinary TorchLean generator. -/
def recentTokens (ids : List Nat) (window : Nat) : List Nat :=
  if window = 0 then
    []
  else
    ids.drop (ids.length - Nat.min ids.length window)

/--
Generate a continuation from a prompt while preserving TorchLean's ordinary sampling policy.

The learned position table is never restarted. A prompt and continuation that exceed the model's
context are rejected.
-/
def generateIds
    (config : Chat.Config)
    (decoder : Runtime.Decoder)
    (promptIds : List Nat)
    (stopToken : Nat) : IO Result := do
  if promptIds.length + config.generate > decoder.config.seqLen then
    throw <| IO.userError <|
      s!"cached generation: prompt ({promptIds.length}) plus requested continuation " ++
        s!"({config.generate}) exceeds context length {decoder.config.seqLen}"

  let prefillStart ← IO.monoNanosNow
  let mut logits ← decoder.prefill promptIds
  let prefillFinish ← IO.monoNanosNow
  let options : text.GenerationOptions :=
    { prompt := ""
      generate := config.generate
      temperature := config.temperature
      topK := config.topK
      repeatPenalty := 1.05
      repeatWindow := 64
      seed := config.seed
      asciiOnly := false }

  let generationStart ← IO.monoNanosNow
  let rec loop
      (counter remaining : Nat) (scores : Array Float)
      (ids suffix : List Nat) : IO (List Nat) := do
    match remaining with
    | 0 => pure suffix
    | remaining + 1 =>
        let recent := (recentTokens ids options.repeatWindow).toArray
        let scoreTensor : Tensor Float [decoder.config.vocab] ←
          Chat.orThrow "cached generation" <|
            TorchLean.Tensor.ofArray [decoder.config.vocab] scores
        let token ← Chat.orThrow "cached generation" <|
          text.chooseNextToken scoreTensor options counter recent
        if token.val = stopToken then
          pure suffix
        else
          let nextIds := ids ++ [token.val]
          let nextSuffix := suffix ++ [token.val]
          if remaining = 0 then
            pure nextSuffix
          else
            match ← decoder.push token.val true with
            | some nextScores =>
                loop (counter + 1) remaining nextScores nextIds nextSuffix
            | none =>
                throw <| IO.userError "cached generation: sampled token produced no logits"
  let suffix ← loop 0 config.generate logits promptIds []
  let generationFinish ← IO.monoNanosNow
  pure
    { tokenIds := suffix
      prefillNanos := prefillFinish - prefillStart
      generationNanos := generationFinish - generationStart }

end Generation
end CachedDecode
end TorchLeanGPT
