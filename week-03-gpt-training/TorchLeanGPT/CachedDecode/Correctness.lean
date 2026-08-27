/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.CachedDecode.Semantics

/-!
# Correctness of incremental causal decoding

The main result, `run_eq_recompute`, states that cached decoding produces the same output at every
position as rebuilding the complete visible prefix. The proof does not assume real arithmetic or a
particular softmax implementation: both paths use the same `Kernel.attend`, while the cache proof
establishes that it receives the same ordered keys and values.

This is the semantic guarantee needed by the Week 3 CUDA decoder. It does not verify CUDA memory
management or floating-point kernel code; those remain an explicit runtime boundary and are checked
against the reference generator on concrete checkpoints.
-/

@[expose] public section

namespace TorchLeanGPT
namespace CachedDecode

variable
  {Token Query Key Value Output : Type}
  (kernel : Kernel Token Query Key Value Output)

namespace Cache

/-- The empty cache represents the empty token prefix. -/
theorem empty_represents :
    (empty : Cache Key Value).Represents kernel [] := by
  simp [empty, Represents]

/-- A represented cache is well formed and has one key/value pair per source token. -/
theorem represents_lengths
    {history : List Token} {cache : Cache Key Value}
    (h : cache.Represents kernel history) :
    cache.WellFormed ∧ cache.length = history.length := by
  rcases h with ⟨hKeys, hValues⟩
  constructor
  · simp [WellFormed, hKeys, hValues]
  · simp [length, hKeys]

end Cache

/--
One cached step preserves the representation invariant and returns the reference output for the
extended prefix.
-/
theorem step_correct
    {history : List Token} {cache : Cache Key Value} {token : Token}
    (h : cache.Represents kernel history) :
    (step kernel cache token).cache.Represents kernel (history ++ [token]) ∧
      (step kernel cache token).output = recomputeOne kernel history token := by
  rcases h with ⟨hKeys, hValues⟩
  constructor
  · constructor <;> simp [step, hKeys, hValues]
  · simp [step, recomputeOne, hKeys, hValues]

/--
Running a cached suffix agrees position-by-position with full-prefix recomputation, and the final
cache represents the original prefix followed by the entire suffix.
-/
theorem run_correct
    {history suffix : List Token} {cache : Cache Key Value}
    (h : cache.Represents kernel history) :
    (run kernel cache suffix).2 = recompute kernel history suffix ∧
      (run kernel cache suffix).1.Represents kernel (history ++ suffix) := by
  induction suffix generalizing history cache with
  | nil =>
      constructor
      · simp [run, recompute]
      · simpa [run, recompute] using h
  | cons token rest ih =>
      have hStep := step_correct kernel (history := history) (cache := cache) (token := token) h
      have hTail :=
        ih (history := history ++ [token]) (cache := (step kernel cache token).cache) hStep.1
      constructor
      · simp only [run, recompute]
        rw [hStep.2, hTail.1]
      · simpa [run, List.append_assoc] using hTail.2

/-- Cached decoding from an empty cache equals the ordinary full-prefix reference decoder. -/
theorem run_eq_recompute (tokens : List Token) :
    (run kernel (Cache.empty : Cache Key Value) tokens).2 =
      recompute kernel [] tokens :=
  (run_correct kernel (Cache.empty_represents kernel)).1

/-- After decoding `tokens`, the cache contains exactly their projected keys and values. -/
theorem run_final_cache_represents (tokens : List Token) :
    (run kernel (Cache.empty : Cache Key Value) tokens).1.Represents kernel tokens := by
  simpa using (run_correct kernel (Cache.empty_represents kernel) (suffix := tokens)).2

/-- Reference decoding returns one output for each newly supplied token. -/
theorem recompute_length (history tokens : List Token) :
    (recompute kernel history tokens).length = tokens.length := by
  induction tokens generalizing history with
  | nil => simp [recompute]
  | cons token rest ih =>
      simp [recompute, ih]

/--
Reference decoding composes over adjacent token segments.

This lemma makes causal isolation explicit: outputs for `left` are computed before any token from
`right` enters the visible prefix.
-/
theorem recompute_append
    (history left right : List Token) :
    recompute kernel history (left ++ right) =
      recompute kernel history left ++
        recompute kernel (history ++ left) right := by
  induction left generalizing history with
  | nil => simp [recompute]
  | cons token rest ih =>
      simp [recompute, ih, List.append_assoc]

/-- The first `n` outputs depend only on the first `n` tokens. -/
theorem recompute_take
    (tokens : List Token) (n : Nat) (hn : n ≤ tokens.length) :
    (recompute kernel [] tokens).take n =
      recompute kernel [] (tokens.take n) := by
  conv_lhs =>
    rw [← List.take_append_drop n tokens]
  rw [recompute_append]
  have hLength : (recompute kernel [] (tokens.take n)).length = n := by
    rw [recompute_length, List.length_take_of_le hn]
  rw [List.take_append_of_le_length (by simp [hLength])]
  have hTake := List.take_length (l := recompute kernel [] (tokens.take n))
  rw [hLength] at hTake
  exact hTake

/-- Two token sequences with the same length-`n` prefix have the same first `n` outputs. -/
theorem recompute_prefix_eq_of_take_eq
    (left right : List Token) (n : Nat)
    (hLeft : n ≤ left.length) (hRight : n ≤ right.length)
    (hPrefix : left.take n = right.take n) :
    (recompute kernel [] left).take n =
      (recompute kernel [] right).take n := by
  rw [recompute_take kernel left n hLeft, recompute_take kernel right n hRight, hPrefix]

/--
Appending future tokens cannot change any output already produced for the current prefix.
-/
theorem causal_prefix_invariant (history future : List Token) :
    (recompute kernel [] (history ++ future)).take history.length =
      recompute kernel [] history := by
  rw [recompute_append]
  have hLength :
      (recompute kernel [] history).length = history.length :=
    recompute_length kernel [] history
  rw [← hLength]
  simp

/--
Cached outputs already returned for a prefix are independent of every later token.

Unlike `causal_prefix_invariant`, this statement is phrased entirely in terms of the incremental
decoder that an application runs. Its proof passes through `run_eq_recompute`, so the result rests
on the cache representation invariant rather than on an additional assumption about `attend`.
-/
theorem cached_causal_prefix_invariant (history future : List Token) :
    (run kernel (Cache.empty : Cache Key Value) (history ++ future)).2.take history.length =
      (run kernel (Cache.empty : Cache Key Value) history).2 := by
  rw [run_eq_recompute, run_eq_recompute]
  exact causal_prefix_invariant kernel history future

/-- Cached decoding is noninterfering on any pair of token sequences with a shared prefix. -/
theorem cached_prefix_eq_of_take_eq
    (left right : List Token) (n : Nat)
    (hLeft : n ≤ left.length) (hRight : n ≤ right.length)
    (hPrefix : left.take n = right.take n) :
    (run kernel (Cache.empty : Cache Key Value) left).2.take n =
      (run kernel (Cache.empty : Cache Key Value) right).2.take n := by
  rw [run_eq_recompute, run_eq_recompute]
  exact recompute_prefix_eq_of_take_eq kernel left right n hLeft hRight hPrefix

end CachedDecode
end TorchLeanGPT
