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
index seen by every update. The result applies once checkpoint loading restores the complete model,
optimizer, and deterministic data state exactly.

Binary parsing and device-buffer restoration remain runtime boundaries; this theorem does not
pretend that an IO round trip has been proved correct.
-/

@[expose] public section

namespace TorchLeanGPT
namespace Training

/-- Execute `count` deterministic training updates beginning at global step `start`. -/
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
      simp only [Nat.succ_add, runSteps]
      rw [ih]
      simp [Nat.add_comm, Nat.add_left_comm]

/--
Restoring the exact state after `first` updates and resuming at global step `start + first` is
semantically identical to an uninterrupted run.
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
