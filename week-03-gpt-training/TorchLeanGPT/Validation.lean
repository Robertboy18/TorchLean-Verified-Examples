/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import Velvet.Core.Loop.Gadgets
-- Registers the method and proof-command elaborators, not only their syntax.
import Velvet.Frontend.ProveCorrect -- shake: keep

/-!
# Finding invalid shard entries

Both token and target-mask loaders reject the first invalid entry. The loop invariant says that
every earlier index has passed; breaking the loop preserves the offending index and this prefix.
This proves the pure scan, not filesystem reads or the interpretation of the input bytes.
-/

@[expose] public section

namespace TorchLeanGPT.Validation

open Std.Internal.Do

/-- Scan indices in order, stopping at the first failure. An empty scan succeeds. -/
method firstInvalid (count : Nat) (valid : Nat → Bool) returns (result : Option Nat) in Id
  ensures match result with
    | none => ∀ i, i < count → valid i = true
    | some i => i < count ∧ valid i = false ∧ ∀ j, j < i → valid j = true
do
  let mut result : Option Nat := none
  for i in [0:count]
    invariant result = none ∧ ∀ j, j < i → valid j = true
    done_with match result with
      | none => ∀ j, j < count → valid j = true
      | some i => i < count ∧ valid i = false ∧ ∀ j, j < i → valid j = true
  do
    if !valid i then
      result := some i
      break
  return result

prove_correct firstInvalid by
  velvet_vcgen [firstInvalid]
  all_goals simp_all [← List.range_eq_range']
  all_goals grind

/-- Acceptance is equivalent to every entry passing the supplied check. -/
theorem firstInvalid_eq_none_iff (count : Nat) (valid : Nat → Bool) :
    firstInvalid count valid = none ↔ ∀ i, i < count → valid i = true := by
  have h := Id.of_wp_run_eq rfl _ ((firstInvalid.spec count valid).le_wp trivial)
  simp only [Id.run, Named.mk, Lean.Order.ofProp_prop_eq] at h
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
