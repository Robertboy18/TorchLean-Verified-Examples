/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import Mathlib.Data.Nat.Basic

/-!
# Correctness of resumable training

The Week 3 runner chooses its batch and learning rate from the global optimizer-step index. This
file gives that control flow a small pure semantics and proves that splitting a run preserves the
index seen by every update.

`resume_eq_uninterrupted` takes exact restoration of the model, optimizer, and deterministic data
state as a hypothesis. Neither the checkpoint parser nor the copy into device buffers is verified.
-/

@[expose] public section

namespace TorchLeanGPT
namespace Training

/-- Run `count` training updates from global step `start`, passing each update its global index. -/
def runSteps (update : Nat → State → State) : Nat → Nat → State → State
  | _start, 0, state => state
  | start, count + 1, state =>
      runSteps update (start + 1) count (update start state)

/-- Splitting an indexed training run preserves the global step seen by every update. -/
theorem runSteps_add
    (update : Nat → State → State) (start first second : Nat) (state : State) :
    runSteps update start (first + second) state =
      runSteps update (start + first) second (runSteps update start first state) := by
  induction first generalizing start state with
  | zero => simp [runSteps]
  | succ first ih =>
      -- The resumed suffix must retain the global index, not restart its schedule at zero.
      simp only [Nat.succ_add, runSteps]
      rw [ih]
      simp [Nat.add_comm, Nat.add_left_comm]

/--
Resuming at global step `start + first` from the state reached after `first` updates gives the same
final state as an uninterrupted run.
-/
theorem resume_eq_uninterrupted
    (update : Nat → State → State) (start first second : Nat)
    (initial restored : State)
    (hRestore : restored = runSteps update start first initial) :
    runSteps update (start + first) second restored =
      runSteps update start (first + second) initial := by
  rw [hRestore, runSteps_add]

end Training
end TorchLeanGPT
