/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import Velvet.Core.Loop.Gadgets
-- Registers the method and proof-command elaborators, not only their syntax.
import Velvet.Frontend.ProveCorrect -- shake: keep

/-!
# Packing aligned training rows

The masked GPT loader appends an input, a next-token target, and a loss-mask bit together.
The loop contract records the entire processed prefix, not just the three array lengths.
Consequently it rules out dropped, duplicated, or reordered entries in any of the arrays.

Velvet generates the loop obligations; ordinary list lemmas discharge them. Its frontend is
imported only here, so callers use `appendRows` as an ordinary function.
-/

@[expose] public section

namespace TorchLeanGPT.Batching

open Std.Internal.Do

/-- Append aligned inputs, targets, and mask bits in increasing row order. -/
method appendRows {Input Target : Type} (count : Nat) (row : Nat → Input × Target × Bool)
    (initial : Array Input × Array Target × Array Bool)
    returns (result : Array Input × Array Target × Array Bool) in Id
  ensures inputs : result.1.toList =
    initial.1.toList ++ (List.range count).map (fun i => (row i).1)
  ensures targets : result.2.1.toList =
    initial.2.1.toList ++ (List.range count).map (fun i => (row i).2.1)
  ensures enabled : result.2.2.toList =
    initial.2.2.toList ++ (List.range count).map (fun i => (row i).2.2)
do
  let mut inputs := initial.1
  let mut targets := initial.2.1
  let mut enabled := initial.2.2
  for i in [0:count]
    invariant inputs.toList = initial.1.toList ++ __pref.map (fun j => (row j).1)
    invariant targets.toList = initial.2.1.toList ++ __pref.map (fun j => (row j).2.1)
    invariant enabled.toList = initial.2.2.toList ++ __pref.map (fun j => (row j).2.2)
    done_with inputs.toList =
        initial.1.toList ++ (List.range count).map (fun j => (row j).1) ∧
      targets.toList = initial.2.1.toList ++ (List.range count).map (fun j => (row j).2.1) ∧
      enabled.toList = initial.2.2.toList ++ (List.range count).map (fun j => (row j).2.2)
  do
    let value := row i
    inputs := inputs.push value.1
    targets := targets.push value.2.1
    enabled := enabled.push value.2.2
  return (inputs, targets, enabled)

prove_correct appendRows by
  velvet_vcgen [appendRows]
  all_goals simp_all [Array.toList_push, List.map_append, List.append_assoc, List.range_eq_range']

/-- The loop contract as ordinary equalities, for proofs that do not use Hoare notation. -/
theorem appendRows_toList {Input Target : Type} (count : Nat)
    (row : Nat → Input × Target × Bool) (initial : Array Input × Array Target × Array Bool) :
    (appendRows count row initial).1.toList =
        initial.1.toList ++ (List.range count).map (fun i => (row i).1) ∧
      (appendRows count row initial).2.1.toList =
        initial.2.1.toList ++ (List.range count).map (fun i => (row i).2.1) ∧
      (appendRows count row initial).2.2.toList =
        initial.2.2.toList ++ (List.range count).map (fun i => (row i).2.2) := by
  have h := (appendRows.spec count row initial).le_wp trivial
  change Lean.Order.meet _ (Lean.Order.meet _ _) at h
  simpa [Named.mk] using h

end TorchLeanGPT.Batching
