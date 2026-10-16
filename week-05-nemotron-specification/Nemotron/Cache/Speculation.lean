/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Nemotron.SSM
public import Mathlib.Data.List.TakeDrop

/-!
# Restoring speculative recurrent state

Section 5.1 of the Ultra report (p. 48) retains an SSM snapshot at each draft step. After a
rejection, the recurrent state must belong to the accepted prefix, not the complete draft.
`snapshots` keeps every post-step state of the shared scan. Its laws hold for any state type and
deterministic transition (a stochastic one must carry its entropy in the state or the prepared
inputs). No draft acceptance rule or its probability law is modeled.
-/

@[expose] public section

namespace Nemotron.Cache.Speculation

open KimiK3

variable {State Input Output : Type}

/-- Initial state, then the state after each draft input, in chronological order. -/
def snapshots (step : State → Input → State × Output) (initial : State) (draft : List Input) :
    List State :=
  initial :: (CausalScan.run (fun state input =>
    let next := step state input
    (next.1, next.1)) initial draft).2

/-- One snapshot is available for every accepted length, including zero and the full draft. -/
@[simp] theorem snapshots_length (step : State → Input → State × Output)
    (initial : State) (draft : List Input) :
    (snapshots step initial draft).length = draft.length + 1 := by
  simp [snapshots, CausalScan.outputs_length]

/-- Stored state after the accepted prefix, found by a bounded index with no default. -/
def restore (step : State → Input → State × Output) (initial : State) (draft : List Input)
    (accepted : Fin (draft.length + 1)) : State :=
  (snapshots step initial draft)[accepted.val]'(by
    rw [snapshots_length]
    exact accepted.isLt)

/-- Restoring gives the state after running the accepted prefix. -/
theorem restore_eq_prefix (step : State → Input → State × Output)
    (initial : State) (draft : List Input) (accepted : Fin (draft.length + 1)) :
    restore step initial draft accepted =
      (CausalScan.run step initial (draft.take accepted.val)).1 := by
  induction draft generalizing initial with
  | nil =>
      simp [restore, snapshots, CausalScan.run]
  | cons input rest ih =>
      cases accepted using Fin.cases with
      | zero => simp [restore, snapshots, CausalScan.run]
      | succ accepted =>
          simpa [restore, snapshots, CausalScan.run] using ih (step initial input).1 accepted

/-- Continuing from the restored snapshot gives the same final state and committed outputs as
scanning the accepted prefix and the continuation. Rejected draft inputs leave no trace. -/
theorem run_after_restore (step : State → Input → State × Output)
    (initial : State) (draft : List Input) (accepted : Fin (draft.length + 1))
    (continuation : List Input) :
    CausalScan.run step initial (draft.take accepted.val ++ continuation) =
      let prior := CausalScan.run step initial (draft.take accepted.val)
      let next := CausalScan.run step (restore step initial draft accepted) continuation
      (next.1, prior.2 ++ next.2) := by
  rw [restore_eq_prefix]
  exact CausalScan.run_append step initial (draft.take accepted.val) continuation

end Nemotron.Cache.Speculation

namespace Nemotron.SSM.Sequence

open Spec TorchLean KimiK3

/-- Every draft state packed along a leading axis, with the initial state in row zero. -/
def snapshots {α : Type} [Storage α] [Add α] [Mul α] [Zero α]
    {tokens valueDim stateDim : Nat} (inputs : Sequence α tokens valueDim stateDim)
    (initial : State α valueDim stateDim) : Tensor α [tokens + 1, valueDim, stateDim] :=
  Tensor.stackLeading fun accepted =>
    Cache.Speculation.restore step initial inputs.toList
      ⟨accepted.val, by simpa using accepted.isLt⟩

/-- Row `accepted` is the state after that many tokens. -/
theorem snapshots_unstack {α : Type} [Storage α] [Add α] [Mul α] [Zero α]
    {tokens valueDim stateDim : Nat} (inputs : Sequence α tokens valueDim stateDim)
    (initial : State α valueDim stateDim) (accepted : Fin (tokens + 1)) :
    Tensor.unstack (inputs.snapshots initial) accepted =
      (CausalScan.run step initial (inputs.toList.take accepted.val)).1 := by
  simp only [snapshots, Tensor.unstack_stackLeading, Cache.Speculation.restore_eq_prefix]

end Nemotron.SSM.Sequence
