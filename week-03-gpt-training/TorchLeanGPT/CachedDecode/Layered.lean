/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.CachedDecode.Correctness

/-!
# Correctness of a layered key/value cache

A transformer decoder stores a separate key/value history at every layer. The state entering layer
`i + 1` is the causal output of layer `i`, so proving one isolated attention cache correct is not
enough. This file defines the stacked invariant and proves that incremental execution agrees at
every position with full-prefix recomputation through the complete layer list.
-/

@[expose] public section

namespace TorchLeanGPT
namespace CachedDecode
namespace Layered

variable {State Query Key Value : Type}

/-- One homogeneous decoder stack, with one attention kernel per layer. -/
abbrev Stack := List (Kernel State Query Key Value State)

/-- One key/value cache per decoder layer. -/
abbrev StackCache := List (Cache Key Value)

/-- One empty key/value cache for every layer. -/
def empty : Stack (State := State) (Query := Query) (Key := Key) (Value := Value) →
    StackCache (Key := Key) (Value := Value)
  | [] => []
  | _layer :: layers => Cache.empty :: empty layers

/-- Full-prefix output of the newest token after passing through every layer. -/
def recomputeOne : Stack (State := State) (Query := Query) (Key := Key) (Value := Value) →
    List State → State → State
  | [], _history, token => token
  | layer :: layers, history, token =>
      recomputeOne layers (recompute layer [] history)
        (CachedDecode.recomputeOne layer history token)

/-- Full-prefix reference outputs for a suffix passing through the complete layer stack. -/
def recompute : Stack (State := State) (Query := Query) (Key := Key) (Value := Value) →
    List State → List State → List State
  | _layers, _history, [] => []
  | layers, history, token :: rest =>
      recomputeOne layers history token :: recompute layers (history ++ [token]) rest

/-- Stacked reference decoding returns one output for every newly supplied state. -/
theorem recompute_length
    (layers : Stack (State := State) (Query := Query) (Key := Key) (Value := Value))
    (history suffix : List State) :
    (recompute layers history suffix).length = suffix.length := by
  induction suffix generalizing history with
  | nil => simp [recompute]
  | cons token rest ih => simp [recompute, ih]

/-- Stacked reference decoding composes over adjacent suffixes. -/
theorem recompute_append
    (layers : Stack (State := State) (Query := Query) (Key := Key) (Value := Value))
    (history left right : List State) :
    recompute layers history (left ++ right) =
      recompute layers history left ++ recompute layers (history ++ left) right := by
  induction left generalizing history with
  | nil => simp [recompute]
  | cons token rest ih => simp [recompute, ih, List.append_assoc]

/-- The first `n` stacked outputs depend only on the first `n` input states. -/
theorem recompute_take
    (layers : Stack (State := State) (Query := Query) (Key := Key) (Value := Value))
    (tokens : List State) (n : Nat) (hn : n ≤ tokens.length) :
    (recompute layers [] tokens).take n = recompute layers [] (tokens.take n) := by
  conv_lhs => rw [← List.take_append_drop n tokens]
  rw [recompute_append]
  have hLength : (recompute layers [] (tokens.take n)).length = n := by
    rw [recompute_length, List.length_take_of_le hn]
  rw [List.take_append_of_le_length (by simp [hLength])]
  have hTake := List.take_length (l := recompute layers [] (tokens.take n))
  rw [hLength] at hTake
  exact hTake

/-- Two inputs with a shared length-`n` prefix have equal first `n` stacked outputs. -/
theorem recompute_prefix_eq_of_take_eq
    (layers : Stack (State := State) (Query := Query) (Key := Key) (Value := Value))
    (left right : List State) (n : Nat)
    (hLeft : n ≤ left.length) (hRight : n ≤ right.length)
    (hPrefix : left.take n = right.take n) :
    (recompute layers [] left).take n = (recompute layers [] right).take n := by
  rw [recompute_take layers left n hLeft, recompute_take layers right n hRight, hPrefix]

/-- Increment one token through every layer and update every layer cache. -/
def step : Stack (State := State) (Query := Query) (Key := Key) (Value := Value) →
    StackCache (Key := Key) (Value := Value) → State →
      Option (StackCache (Key := Key) (Value := Value) × State)
  | [], [], token => some ([], token)
  | layer :: layers, cache :: caches, token =>
      let current := CachedDecode.step layer cache token
      match step layers caches current.output with
      | some (nextCaches, output) => some (current.cache :: nextCaches, output)
      | none => none
  | _, _, _ => none

/-- Run an incremental suffix through the complete cached stack. -/
def run : Stack (State := State) (Query := Query) (Key := Key) (Value := Value) →
    StackCache (Key := Key) (Value := Value) → List State →
      Option (StackCache (Key := Key) (Value := Value) × List State)
  | _layers, caches, [] => some (caches, [])
  | layers, caches, token :: rest => do
      let (nextCaches, output) ← step layers caches token
      let (finalCaches, outputs) ← run layers nextCaches rest
      pure (finalCaches, output :: outputs)

/--
Every layer cache represents the history seen at that layer.

The input history of the next layer is the complete causal output history of the previous layer,
not the original token history. This is the invariant missing from a single-layer cache model.
-/
def Represents : Stack (State := State) (Query := Query) (Key := Key) (Value := Value) →
    StackCache (Key := Key) (Value := Value) → List State → Prop
  | [], caches, _history => caches = []
  | _layer :: _layers, [], _history => False
  | layer :: layers, cache :: caches, history =>
      cache.Represents layer history ∧
        Represents layers caches (CachedDecode.recompute layer [] history)

/-- The empty cache stack represents the empty token history. -/
theorem empty_represents (layers : Stack (State := State) (Query := Query) (Key := Key)
    (Value := Value)) :
    Represents layers (empty layers) [] := by
  induction layers with
  | nil => rfl
  | cons layer layers ih =>
      exact ⟨Cache.empty_represents layer, by simpa [empty, CachedDecode.recompute] using ih⟩

/--
A represented stack has one cache per layer, and every cache contains one row per consumed state.

The histories stored by deeper layers contain different values, but causal evaluation preserves
their lengths. Consequently all layer caches advance in lockstep with the input history.
-/
theorem represents_cache_lengths
    {layers : Stack (State := State) (Query := Query) (Key := Key) (Value := Value)}
    {caches : StackCache (Key := Key) (Value := Value)} {history : List State}
    (h : Represents layers caches history) :
    caches.map Cache.length = List.replicate layers.length history.length := by
  induction layers generalizing caches history with
  | nil =>
      simp [Represents] at h
      subst caches
      rfl
  | cons layer layers ih =>
      cases caches with
      | nil => simp [Represents] at h
      | cons cache caches =>
          rcases h with ⟨hHead, hTail⟩
          have hHeadLength := (Cache.represents_lengths layer hHead).2
          have hTailLengths := ih hTail
          have hHistoryLength :
              (CachedDecode.recompute layer [] history).length = history.length :=
            CachedDecode.recompute_length layer [] history
          simp only [List.map_cons, List.length_cons]
          rw [hHeadLength, hTailLengths, hHistoryLength, List.replicate_succ]

/-- One stacked cache step agrees with full-prefix recomputation and preserves every layer cache. -/
theorem step_correct
    (layers : Stack (State := State) (Query := Query) (Key := Key) (Value := Value))
    (caches : StackCache (Key := Key) (Value := Value))
    (history : List State) (token : State)
    (h : Represents layers caches history) :
    ∃ nextCaches output,
      step layers caches token = some (nextCaches, output) ∧
      Represents layers nextCaches (history ++ [token]) ∧
      output = recomputeOne layers history token := by
  induction layers generalizing caches history token with
  | nil =>
      simp [Represents] at h
      subst caches
      exact ⟨[], token, rfl, rfl, rfl⟩
  | cons layer layers ih =>
      cases caches with
      | nil => simp [Represents] at h
      | cons cache caches =>
          rcases h with ⟨hHead, hTail⟩
          have hCurrent := CachedDecode.step_correct layer (token := token) hHead
          obtain ⟨nextCaches, output, hStep, hRepresents, hOutput⟩ :=
            ih caches (CachedDecode.recompute layer [] history)
              (CachedDecode.recomputeOne layer history token) hTail
          refine ⟨(CachedDecode.step layer cache token).cache :: nextCaches, output, ?_, ?_, ?_⟩
          · simp [step, hCurrent.2, hStep]
          · constructor
            · exact hCurrent.1
            · have hAppend :
                  CachedDecode.recompute layer [] (history ++ [token]) =
                    CachedDecode.recompute layer [] history ++
                      [CachedDecode.recomputeOne layer history token] := by
                simpa [CachedDecode.recompute] using
                  CachedDecode.recompute_append layer [] history [token]
              rw [hAppend]
              exact hRepresents
          · simpa [recomputeOne] using hOutput

/--
Incremental decoding through every layer agrees position-by-position with rebuilding every visible
prefix, and the final collection of caches represents the complete consumed history.
-/
theorem run_correct
    (layers : Stack (State := State) (Query := Query) (Key := Key) (Value := Value))
    (caches : StackCache (Key := Key) (Value := Value))
    (history suffix : List State)
    (h : Represents layers caches history) :
    ∃ finalCaches outputs,
      run layers caches suffix = some (finalCaches, outputs) ∧
      Represents layers finalCaches (history ++ suffix) ∧
      outputs = recompute layers history suffix := by
  induction suffix generalizing caches history with
  | nil =>
      exact ⟨caches, [], rfl, by simpa, rfl⟩
  | cons token rest ih =>
      obtain ⟨nextCaches, output, hStep, hNext, hOutput⟩ :=
        step_correct layers caches history token h
      obtain ⟨finalCaches, outputs, hRun, hFinal, hOutputs⟩ :=
        ih nextCaches (history ++ [token]) hNext
      refine ⟨finalCaches, output :: outputs, ?_, ?_, ?_⟩
      · simp [run, hStep, hRun]
      · simpa [List.append_assoc] using hFinal
      · simp [recompute, hOutput, hOutputs]

/-- Cached decoding from empty layer caches equals complete stacked-prefix recomputation. -/
theorem run_from_empty_eq_recompute
    (layers : Stack (State := State) (Query := Query) (Key := Key) (Value := Value))
    (tokens : List State) :
    ∃ finalCaches,
      run layers (empty layers) tokens = some (finalCaches, recompute layers [] tokens) ∧
      Represents layers finalCaches tokens := by
  obtain ⟨finalCaches, outputs, hRun, hFinal, hOutputs⟩ :=
    run_correct layers (empty layers) [] tokens (empty_represents layers)
  subst outputs
  exact ⟨finalCaches, hRun, by simpa using hFinal⟩

/--
After decoding from empty caches, every layer contains exactly one key/value row per input state.
-/
theorem run_from_empty_cache_lengths
    (layers : Stack (State := State) (Query := Query) (Key := Key) (Value := Value))
    (tokens : List State) :
    ∃ finalCaches,
      run layers (empty layers) tokens = some (finalCaches, recompute layers [] tokens) ∧
      finalCaches.map Cache.length = List.replicate layers.length tokens.length := by
  obtain ⟨finalCaches, hRun, hRepresents⟩ := run_from_empty_eq_recompute layers tokens
  exact ⟨finalCaches, hRun, represents_cache_lengths hRepresents⟩

/-- A complete cached decoder stack is noninterfering on any shared input prefix. -/
theorem run_from_empty_prefix_eq_of_take_eq
    (layers : Stack (State := State) (Query := Query) (Key := Key) (Value := Value))
    (left right : List State) (n : Nat)
    (hLeft : n ≤ left.length) (hRight : n ≤ right.length)
    (hPrefix : left.take n = right.take n) :
    ∃ leftCaches rightCaches leftOutputs rightOutputs,
      run layers (empty layers) left = some (leftCaches, leftOutputs) ∧
      run layers (empty layers) right = some (rightCaches, rightOutputs) ∧
      leftOutputs.take n = rightOutputs.take n := by
  obtain ⟨leftCaches, hLeftRun, _hLeftRepresents⟩ :=
    run_from_empty_eq_recompute layers left
  obtain ⟨rightCaches, hRightRun, _hRightRepresents⟩ :=
    run_from_empty_eq_recompute layers right
  exact ⟨leftCaches, rightCaches, recompute layers [] left, recompute layers [] right,
    hLeftRun, hRightRun, recompute_prefix_eq_of_take_eq layers left right n hLeft hRight hPrefix⟩

end Layered
end CachedDecode
end TorchLeanGPT
