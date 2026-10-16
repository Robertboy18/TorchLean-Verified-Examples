/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import Std.WP
import Std.Tactic.Do

/-!
# Packing aligned training rows

The masked GPT loader appends an input, a next-token target, and a loss-mask bit for each row.
The loop invariant records the whole processed prefix rather than the three array lengths, so it
rules out dropped, duplicated, or reordered entries in any of the arrays.

Lean's `vcgen` tactic generates the loop obligations and ordinary list lemmas discharge them,
while the loop itself still pushes into arrays and never allocates a list of row indices.
-/

@[expose] public section

namespace TorchLeanGPT.Batching

open Std.WP

/-- Append aligned inputs, targets, and mask bits in increasing row order. -/
def appendRows {Input Target : Type} (count : Nat) (row : Nat → Input × Target × Bool)
    (initial : Array Input × Array Target × Array Bool) :
    Array Input × Array Target × Array Bool := Id.run do
  let mut inputs := initial.1
  let mut targets := initial.2.1
  let mut enabled := initial.2.2
  for i in [0:count] do
    let value := row i
    inputs := inputs.push value.1
    targets := targets.push value.2.1
    enabled := enabled.push value.2.2
  return (inputs, targets, enabled)

/-- All three arrays contain their initial entries followed by the rows in index order. -/
theorem appendRows_toList {Input Target : Type} (count : Nat)
    (row : Nat → Input × Target × Bool) (initial : Array Input × Array Target × Array Bool) :
    (appendRows count row initial).1.toList =
        initial.1.toList ++ (List.range count).map (fun i => (row i).1) ∧
      (appendRows count row initial).2.1.toList =
        initial.2.1.toList ++ (List.range count).map (fun i => (row i).2.1) ∧
      (appendRows count row initial).2.2.toList =
        initial.2.2.toList ++ (List.range count).map (fun i => (row i).2.2) := by
  apply Id.of_run_eq_wp (prog := appendRows count row initial) rfl
    (fun result => result.1.toList =
        initial.1.toList ++ (List.range count).map (fun i => (row i).1) ∧
      result.2.1.toList =
        initial.2.1.toList ++ (List.range count).map (fun i => (row i).2.1) ∧
      result.2.2.toList =
        initial.2.2.toList ++ (List.range count).map (fun i => (row i).2.2))
  unfold appendRows
  simp only [Id.run]
  set_option experimental.vcgen true in
  vcgen invariants
  · fun pref _ result =>
      result.1.toList = initial.1.toList ++ pref.map (fun i => (row i).1) ∧
      result.2.1.toList = initial.2.1.toList ++ pref.map (fun i => (row i).2.1) ∧
      result.2.2.toList = initial.2.2.toList ++ pref.map (fun i => (row i).2.2)
  all_goals simp_all [Array.toList_push, List.map_append, List.append_assoc,
    Std.Internal.ForIn.toList_range, Std.Legacy.Range.size, ← List.range_eq_range']

/-- Starting from empty arrays, entry `k` of each packed array is that component of `row k`. -/
theorem appendRows_getElem? {Input Target : Type} (count : Nat)
    (row : Nat → Input × Target × Bool) (k : Nat) (hk : k < count) :
    (appendRows count row (#[], #[], #[])).1[k]? = some (row k).1 ∧
      (appendRows count row (#[], #[], #[])).2.1[k]? = some (row k).2.1 ∧
      (appendRows count row (#[], #[], #[])).2.2[k]? = some (row k).2.2 := by
  obtain ⟨hInputs, hTargets, hEnabled⟩ := appendRows_toList count row (#[], #[], #[])
  simp only [← Array.getElem?_toList, hInputs, hTargets, hEnabled]
  simp [hk]

end TorchLeanGPT.Batching
