/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Nemotron.Cache.Replay
public import NN.Spec.Core.Context.Real
public import NN.Spec.Core.Tensor
public import Mathlib.Algebra.Order.Ring.Abs
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Errors at the checkpoint sites

Coordinate error bounds that separate recurrence sensitivity from state-storage error. Both
runs see the same prepared inputs, so a change in logits or in the sampled tokens is outside the
comparison. Rounding inside a transition belongs to `step`, and comparing against exact-real
arithmetic also needs local arithmetic-error bounds.
-/

@[expose] public section

namespace Nemotron.Cache

open Spec TorchLean KimiK3

/-- Every coordinate of `x` is within `error` of `y` (vacuous when a dimension is empty). -/
def Within {shape : Shape} (error : ℝ) (x y : Tensor ℝ shape) : Prop :=
  ∀ coordinate, |x coordinate - y coordinate| ≤ error

/-- A tensor is within `0` of itself. -/
theorem within_self {shape : Shape} (x : Tensor ℝ shape) : Within 0 x x := by
  intro coordinate
  simp

/-- Triangle inequality for `Within`: the allowances add. -/
theorem within_triangle {shape : Shape} {x y z : Tensor ℝ shape} {a b : ℝ}
    (hxy : Within a x y) (hyz : Within b y z) : Within (a + b) x z := by
  intro coordinate
  calc
    |x coordinate - z coordinate| =
        |(x coordinate - y coordinate) + (y coordinate - z coordinate)| := by
      congr 1
      ring
    _ ≤ |x coordinate - y coordinate| + |y coordinate - z coordinate| := abs_add_le _ _
    _ ≤ a + b := add_le_add (hxy coordinate) (hyz coordinate)

/-- A per-token sensitivity `L` becomes `L ^ inputs.length` over a whole scan. Only the
recurrent state is bounded, so the outputs can have any type. -/
theorem scan_state_within {shape : Shape} {Input Output : Type}
    (step : Tensor ℝ shape → Input → Tensor ℝ shape × Output) (L : ℝ)
    (hstep : ∀ input e x y, Within e x y → Within (L * e) (step x input).1 (step y input).1)
    (inputs : List Input) (x y : Tensor ℝ shape) (e : ℝ) (hxy : Within e x y) :
    Within (L ^ inputs.length * e)
      (CausalScan.run step x inputs).1 (CausalScan.run step y inputs).1 := by
  induction inputs generalizing x y e with
  | nil => simpa [CausalScan.run] using hxy
  | cons input rest ih =>
      have hnext := ih (step x input).1 (step y input).1 (L * e) (hstep input e x y hxy)
      simpa [CausalScan.run, pow_succ, mul_assoc, mul_left_comm, mul_comm] using hnext

/-- Error budget of a chunked run: each chunk scales the error by `L ^ chunk.length`, then its
write adds `δ`. -/
def checkpointBudget {Input : Type} (L δ : ℝ) : ℝ → List (List Input) → ℝ
  | e, [] => e
  | e, chunk :: rest => checkpointBudget L δ (L ^ chunk.length * e + δ) rest

/-- Each state this execution writes is stored within `δ`. A floating format has no single
absolute bound over every tensor value, so the bound is asked only of the written states. -/
def WritesWithin {shape : Shape} {Input Output : Type}
    (step : Tensor ℝ shape → Input → Tensor ℝ shape × Output)
    (quantize : Tensor ℝ shape → Tensor ℝ shape) (δ : ℝ) :
    Tensor ℝ shape → List (List Input) → Prop
  | _, [] => True
  | state, chunk :: rest =>
      let next := (CausalScan.run step state chunk).1
      Within δ (quantize next) next ∧ WritesWithin step quantize δ (quantize next) rest

/-- Storage bounds at the written states give an error bound for any checkpoint partition.
Sensitivity is paid per token and `δ` once per checkpoint. -/
theorem quantizedChunks_within_of_writes {shape : Shape} {Input Output : Type}
    (step : Tensor ℝ shape → Input → Tensor ℝ shape × Output)
    (quantize : Tensor ℝ shape → Tensor ℝ shape) (L δ : ℝ)
    (hstep : ∀ input e x y, Within e x y → Within (L * e) (step x input).1 (step y input).1)
    (chunks : List (List Input)) (x y : Tensor ℝ shape) (e : ℝ) (hxy : Within e x y)
    (hwrites : WritesWithin step quantize δ x chunks) :
    Within (checkpointBudget L δ e chunks)
      (quantizedChunks step quantize x chunks) (CausalScan.run step y chunks.flatten).1 := by
  induction chunks generalizing x y e with
  | nil => simpa [quantizedChunks, checkpointBudget, CausalScan.run] using hxy
  | cons chunk rest ih =>
      simp only [WritesWithin] at hwrites
      -- First propagate the old error through the chunk, then pay for its one storage write.
      have hscan := scan_state_within step L hstep chunk x y e hxy
      have hwrite := within_triangle hwrites.1 hscan
      have hwrite' : Within (L ^ chunk.length * e + δ)
          (quantize (CausalScan.run step x chunk).1) (CausalScan.run step y chunk).1 := by
        simpa [add_comm] using hwrite
      -- Both executions resume from their own resulting state with the same remaining inputs.
      have hrest := ih (quantize (CausalScan.run step x chunk).1)
        (CausalScan.run step y chunk).1 (L ^ chunk.length * e + δ) hwrite' hwrites.2
      simpa only [quantizedChunks, checkpointBudget, List.flatten_cons,
        CausalScan.run_append] using hrest

/-- Same bound, from a quantizer that is within `δ` everywhere. Fixed grids satisfy this.
Floating grids usually need a cap on the reachable states instead. -/
theorem quantizedChunks_within {shape : Shape} {Input Output : Type}
    (step : Tensor ℝ shape → Input → Tensor ℝ shape × Output)
    (quantize : Tensor ℝ shape → Tensor ℝ shape) (L δ : ℝ)
    (hstep : ∀ input e x y, Within e x y → Within (L * e) (step x input).1 (step y input).1)
    (hquantize : ∀ x, Within δ (quantize x) x)
    (chunks : List (List Input)) (x y : Tensor ℝ shape) (e : ℝ) (hxy : Within e x y) :
    Within (checkpointBudget L δ e chunks)
      (quantizedChunks step quantize x chunks) (CausalScan.run step y chunks.flatten).1 := by
  have hwrites : ∀ state parts, WritesWithin step quantize δ state parts := by
    intro state parts
    induction parts generalizing state with
    | nil => trivial
    | cons chunk rest ih => exact ⟨hquantize _, ih _⟩
  exact quantizedChunks_within_of_writes step quantize L δ hstep chunks x y e hxy
    (hwrites x chunks)

/-- With `0 ≤ L ≤ 1` the budget grows at most linearly in the number of writes. This is a
worst-case bound and does not compare the actual errors of two schedules. -/
theorem checkpointBudget_le_count {Input : Type} (L δ e : ℝ)
    (hL : 0 ≤ L) (hLone : L ≤ 1) (hδ : 0 ≤ δ) (he : 0 ≤ e)
    (chunks : List (List Input)) :
    checkpointBudget L δ e chunks ≤ e + chunks.length * δ := by
  induction chunks generalizing e with
  | nil => simp [checkpointBudget]
  | cons chunk rest ih =>
      have hpower : 0 ≤ L ^ chunk.length := pow_nonneg hL _
      have hpowerOne : L ^ chunk.length ≤ 1 := pow_le_one₀ hL hLone
      have hnext : 0 ≤ L ^ chunk.length * e + δ :=
        add_nonneg (mul_nonneg hpower he) hδ
      have hbound := ih (L ^ chunk.length * e + δ) hnext
      simp only [checkpointBudget, List.length_cons, Nat.cast_add, Nat.cast_one]
      nlinarith [mul_le_mul_of_nonneg_right hpowerOne he]

end Nemotron.Cache
