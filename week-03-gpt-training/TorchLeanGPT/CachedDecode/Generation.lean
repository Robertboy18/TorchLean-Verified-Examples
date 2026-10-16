/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.Chat
public import TorchLeanGPT.CachedDecode.Runtime

/-!
# Sampling from a cached decoder

`generateIds` is the sampling loop on top of the cached decoder. The interactive command and the
benchmark both use it, so they cannot drift in repeat penalty, top-k sampling, context handling, or
timing boundaries.
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

/-- Tokens per second over a wall-clock interval in nanoseconds, or zero for an empty interval. -/
def tokensPerSecond (tokens nanos : Nat) : Float :=
  if nanos = 0 then
    0.0
  else
    Float.ofNat tokens * 1.0e9 / Float.ofNat nanos

/-- The last `window` ids of the history, prompt included, as seen by the repeat penalty. -/
def recentTokens (ids : List Nat) (window : Nat) : List Nat :=
  if window = 0 then
    []
  else
    ids.drop (ids.length - Nat.min ids.length window)

/--
Generate a continuation from a prompt with TorchLean's ordinary sampling policy.

Like the full-prefix generator, it calls `text.next`, which applies the repeat penalty
(`config.repeatPenalty` over `config.repeatWindow` ids) before greedy or top-k selection. The
learned position table is never restarted, so a prompt and continuation that exceed the model's
context are rejected. It returns the new tokens, never `stopToken`.
-/
def generateIds
    (config : Chat.Config)
    (decoder : Runtime.Decoder)
    (promptIds : List Nat)
    (stopToken : Nat) : IO Result := do
  let steps ← CLI.orThrow "cached generation" <|
    config.continuationLength decoder.config.sequenceLength promptIds.length
  if promptIds.length + steps > decoder.config.sequenceLength then
    throw <| IO.userError <|
      s!"cached generation: prompt ({promptIds.length}) plus requested continuation " ++
        s!"({steps}) exceeds context length {decoder.config.sequenceLength}"

  let prefillStart ← IO.monoNanosNow
  let mut logits ← decoder.prefill promptIds
  let prefillFinish ← IO.monoNanosNow
  let options : text.GenerationOptions :=
    { prompt := ""
      newTokenCount := steps
      temperature := config.temperature
      topK := config.topK
      repeatPenalty := config.repeatPenalty
      repeatWindow := config.repeatWindow
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
        let recentTensor : Tensor Nat [recent.size] := Tensor.ofFn fun i => recent[i]
        let scoreTensor : Tensor Float [decoder.config.vocabularySize] ←
          if h : scores.size = decoder.config.vocabularySize then
            pure <| Tensor.from (⟨scores, h⟩ : Vector Float decoder.config.vocabularySize)
          else
            throw <| IO.userError
              s!"cached generation: expected {decoder.config.vocabularySize} logits, got {scores.size}"
        let token ← CLI.orThrow "cached generation" <|
          text.next scoreTensor options counter recentTensor
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
  let suffix ← loop 0 steps logits promptIds []
  let generationFinish ← IO.monoNanosNow
  pure
    { tokenIds := suffix
      prefillNanos := prefillFinish - prefillStart
      generationNanos := generationFinish - generationStart }

end Generation
end CachedDecode
end TorchLeanGPT
