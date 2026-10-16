/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Sequence.Scan

/-!
# SSM checkpoint replay

Section 4.5 of the NVIDIA Nemotron 3 Ultra technical report (pp. 44–46) stores an older SSM
checkpoint together with the activations needed to replay the transitions after it. The laws
reuse the shared causal scan and hold for any deterministic transition (a rounded one included),
as long as replay uses the same transition and the same prepared inputs.
-/

@[expose] public section

namespace Nemotron.Cache

open KimiK3

/-- A stored state and the prepared inputs since that state was stored, in chronological order. -/
structure Replay (State Input : Type) where
  checkpoint : State
  pending : List Input

namespace Replay

variable {State Input Output : Type}

/-- Reconstruct the current state and the outputs since the checkpoint. -/
def observe (step : State → Input → State × Output) (cache : Replay State Input) :
    State × List Output :=
  CausalScan.run step cache.checkpoint cache.pending

/-- Retain an additional prepared input without writing the recurrent state. -/
def push (cache : Replay State Input) (input : Input) : Replay State Input :=
  ⟨cache.checkpoint, cache.pending ++ [input]⟩

/-- Commit the reconstructed state through the supplied storage quantizer. -/
def flush (step : State → Input → State × Output) (quantize : State → State)
    (cache : Replay State Input) : Replay State Input :=
  -- Clearing pending inputs prevents applying the already committed transitions twice.
  ⟨quantize (cache.observe step).1, []⟩

/-- Replaying one more input gives the same state and output as stepping the reconstructed state.
Earlier replay outputs stay at the front of the list. -/
theorem observe_push (step : State → Input → State × Output) (cache : Replay State Input)
    (input : Input) :
    (cache.push input).observe step =
      let prior := cache.observe step
      let next := step prior.1 input
      (next.1, prior.2 ++ [next.2]) := by
  simpa [observe, push, CausalScan.run] using
    CausalScan.run_append step cache.checkpoint cache.pending [input]

/-- An exact checkpoint write preserves the reconstructed current state. -/
@[simp] theorem observe_flush_id_state (step : State → Input → State × Output)
    (cache : Replay State Input) :
    ((cache.flush step id).observe step).1 = (cache.observe step).1 := by
  rfl

/-- After a flush the reconstructed state is the quantized state, so a checkpoint adds only the
quantizer's error at that state. -/
@[simp] theorem observe_flush_state (step : State → Input → State × Output)
    (quantize : State → State) (cache : Replay State Input) :
    ((cache.flush step quantize).observe step).1 = quantize (cache.observe step).1 := by
  rfl

/-- Scanning the pending inputs and a suffix from the checkpoint equals replaying, then
continuing from the reconstructed state. Every output agrees, not just the final state. -/
theorem run_from_replay (step : State → Input → State × Output)
    (cache : Replay State Input) (suffix : List Input) :
    CausalScan.run step cache.checkpoint (cache.pending ++ suffix) =
      let prior := cache.observe step
      let next := CausalScan.run step prior.1 suffix
      (next.1, prior.2 ++ next.2) :=
  CausalScan.run_append step cache.checkpoint cache.pending suffix

end Replay

/-- Run each chunk, then quantize the state once. An empty chunk still writes, so callers that
count token periods should pass nonempty chunks. -/
def quantizedChunks {State Input Output : Type}
    (step : State → Input → State × Output) (quantize : State → State) :
    State → List (List Input) → State
  | state, [] => state
  | state, chunk :: rest =>
      quantizedChunks step quantize (quantize (CausalScan.run step state chunk).1) rest

/-- Without storage loss, every chunk partition has the same final state as one full scan. -/
theorem quantizedChunks_id {State Input Output : Type}
    (step : State → Input → State × Output) (state : State) (chunks : List (List Input)) :
    quantizedChunks step id state chunks = (CausalScan.run step state chunks.flatten).1 := by
  induction chunks generalizing state with
  | nil => rfl
  | cons chunk rest ih =>
      simp only [quantizedChunks, id_eq, List.flatten_cons, ih]
      rw [CausalScan.run_append]

end Nemotron.Cache
