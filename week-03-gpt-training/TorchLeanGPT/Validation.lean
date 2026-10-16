/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import Std.WP
import Std.Tactic.Do

/-!
# Finding invalid shard entries

The token and target-mask loaders both report the first invalid entry. The loop invariant says
that every earlier index has passed, and breaking out of the loop keeps the offending index along
with that fact. Only the pure scan is proved, not the file reads or the decoding of the bytes.
-/

@[expose] public section

namespace TorchLeanGPT.Validation

open Std.WP

/-- Scan indices in order, stopping at the first failure. An empty scan succeeds. -/
def firstInvalid (count : Nat) (valid : Nat → Bool) : Option Nat := Id.run do
  let mut result : Option Nat := none
  for i in [0:count] do
    if !valid i then
      result := some i
      break
  return result

/-- Specification of `firstInvalid`: `none` if every index passes, else the first failing index. -/
theorem firstInvalid.spec (count : Nat) (valid : Nat → Bool) :
    match firstInvalid count valid with
    | none => ∀ i, i < count → valid i = true
    | some i => i < count ∧ valid i = false ∧ ∀ j, j < i → valid j = true := by
  apply Id.of_run_eq_wp (prog := firstInvalid count valid) rfl
    (fun result => match result with
      | none => ∀ i, i < count → valid i = true
      | some i => i < count ∧ valid i = false ∧ ∀ j, j < i → valid j = true)
  unfold firstInvalid
  simp only [Id.run]
  set_option experimental.vcgen true in
  vcgen invariants
  · fun pref suff result => match result with
    | none => ∀ j, j < pref.length → valid j = true
    | some i => suff = [] ∧ i < count ∧ valid i = false ∧ ∀ j, j < i → valid j = true
  all_goals simp_all [Std.Internal.ForIn.toList_range, Std.Legacy.Range.size,
    ← List.range_eq_range']
  all_goals
    next pref cur suff result hrange ih hvalid =>
      -- The next range index is the length of the already-checked prefix.
      have hlen : pref.length < count := by
        have := congrArg List.length hrange
        simp only [List.length_range, List.length_append, List.length_cons] at this
        omega
      have hcur : cur = pref.length := by
        have := congrArg (fun xs : List Nat => xs[pref.length]?) hrange
        simpa [hlen] using this.symm
      cases result <;> simp_all <;> grind

/-- `firstInvalid` returns `none` iff every index below `count` passes `valid`. -/
theorem firstInvalid_eq_none_iff (count : Nat) (valid : Nat → Bool) :
    firstInvalid count valid = none ↔ ∀ i, i < count → valid i = true := by
  have h := firstInvalid.spec count valid
  cases heq : firstInvalid count valid with
  | none => simpa [heq] using h
  | some i =>
    simp only [heq] at h
    constructor
    · intro hnone
      cases hnone
    · intro hall
      have := hall i h.1
      simp_all

end TorchLeanGPT.Validation
