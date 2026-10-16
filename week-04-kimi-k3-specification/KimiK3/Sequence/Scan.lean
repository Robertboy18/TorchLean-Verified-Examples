/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.List.Basic

/-!
# Causal scans and composition

These scan laws connect layer-wise sequence evaluation with causal token steps. Lists carry the
recursion in the proofs, and the public sequence interfaces pack inputs and outputs into
shape-indexed tensors. The transition itself may use rounded arithmetic, but every compared
execution must use the same transition and inputs.
-/

@[expose] public section

namespace KimiK3

namespace CausalScan

/-- Run a state transition over a token list, returning the final state and one output per token. -/
def run {State Input Output : Type} (step : State → Input → State × Output) :
    State → List Input → State × List Output
  | state, [] => (state, [])
  | state, token :: rest =>
      let (next, output) := step state token
      let (final, outputs) := run step next rest
      (final, output :: outputs)

/-- A causal scan emits one output per input. -/
theorem outputs_length {State Input Output : Type} (step : State → Input → State × Output)
    (state : State) (tokens : List Input) :
    (run step state tokens).2.length = tokens.length := by
  induction tokens generalizing state with
  | nil => simp [run]
  | cons token rest ih =>
      simp [run, ih]

/--
Splitting a causal scan into chunks changes neither its final state nor its outputs, provided the
second chunk resumes from the state left by the first. It also holds for a rounded transition,
whose evaluation order is fixed, but does not justify regrouping the arithmetic inside a
different implementation.
-/
theorem run_append {State Input Output : Type} (step : State → Input → State × Output)
    (state : State) (left right : List Input) :
    run step state (left ++ right) =
      let first := run step state left
      let second := run step first.1 right
      (second.1, first.2 ++ second.2) := by
  induction left generalizing state with
  | nil => simp [run]
  | cons token rest ih =>
      -- Resume the rest of the chunk from the state produced by this token.
      simp [run, ih]

/-- Preparing each input and finishing each output around a causal transition gives the same
state and outputs as running that transition on the prepared sequence. Finishing may also use the
original input, as a decoder layer does for its residual and depth state. -/
theorem run_map {State Input Prepared Output Finished : Type}
    (step : State → Prepared → State × Output) (prepare : Input → Prepared)
    (finish : Input → Output → Finished) (state : State) (inputs : List Input) :
    run (fun current input =>
      let result := step current (prepare input)
      (result.1, finish input result.2)) state inputs =
      let result := run step state (inputs.map prepare)
      (result.1, inputs.zipWith finish result.2) := by
  induction inputs generalizing state with
  | nil => rfl
  | cons input rest ih => simp [run, ih]

/-- Scanning two causal stages in series is equivalent to scanning the first stage and feeding its
whole output sequence to the second. Each stage retains its own state. -/
theorem run_compose {FirstState SecondState Input Intermediate Output : Type}
    (first : FirstState → Input → FirstState × Intermediate)
    (second : SecondState → Intermediate → SecondState × Output)
    (firstState : FirstState) (secondState : SecondState) (inputs : List Input) :
    run (fun state input =>
      let firstResult := first state.1 input
      let secondResult := second state.2 firstResult.2
      ((firstResult.1, secondResult.1), secondResult.2)) (firstState, secondState) inputs =
      let firstResult := run first firstState inputs
      let secondResult := run second secondState firstResult.2
      ((firstResult.1, secondResult.1), secondResult.2) := by
  induction inputs generalizing firstState secondState with
  | nil => rfl
  -- Keep both stage states arbitrary: each token advances both before the next induction step.
  | cons input rest ih => simp [run, ih]

/-- Mapping the inputs and outputs of a causal transition commutes with the scan: the final state
is unchanged and the outputs are mapped. -/
theorem run_map_output {State Input Prepared Output Finished : Type}
    (step : State → Prepared → State × Output) (prepare : Input → Prepared)
    (finish : Output → Finished) (state : State) (inputs : List Input) :
    run (fun current input =>
      let result := step current (prepare input)
      (result.1, finish result.2)) state inputs =
      let result := run step state (inputs.map prepare)
      (result.1, result.2.map finish) := by
  induction inputs generalizing state with
  | nil => rfl
  | cons input rest ih => simp [run, ih]

/-- Running no inputs leaves the state unchanged and emits no outputs. -/
@[simp] theorem run_nil {State Input Output : Type} (step : State → Input → State × Output)
    (state : State) : run step state ([] : List Input) = (state, []) := by
  rfl

end CausalScan

end KimiK3
