/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Nemotron.Cache.Error
public import NN.Spec.Core.TensorOps
public import FloatLib.Numerics.Reduction.Error
public import FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.Runtime

/-!
# Gradient accumulation and lost contributions

The report's §2.7 (pp. 11–13) attributes its first training divergence to lowering the output
layer's local gradient accumulation from FP32 to BF16. Small MTP contributions were lost.
Here the gradient tensors are leaves of a FloatLib `ReductionTree`, which fixes the caller's
reduction order, and the bound pays for each rounded addition in that tree. Nothing assumes a
native backend used the same order or that the leaves are exact gradients.
-/

@[expose] public section

namespace Nemotron.Training

open Spec TorchLean FloatLib.Numerics

/-- Combine gradient tensors in the supplied order, with the caller's scalar arithmetic. -/
def accumulate {α : Type} [Storage α] {shape : Shape} (combine : α → α → α)
    (schedule : ReductionTree (Tensor α shape)) : Tensor α shape :=
  schedule.eval (Tensor.map2Spec combine) id

/-- A tensor reduction follows the same scalar schedule at every coordinate. -/
theorem accumulate_apply {α : Type} [Storage α] {shape : Shape} (combine : α → α → α)
    (schedule : ReductionTree (Tensor α shape)) (coordinate : Shape.Coord shape) :
    accumulate combine schedule coordinate = schedule.eval combine (fun x => x coordinate) := by
  induction schedule with
  | leaf x => rfl
  | node left right hl hr =>
      simpa [accumulate, ReductionTree.eval, Tensor.map2Spec] using congrArg₂ combine hl hr

/-- Accumulation error is at most FloatLib's sum of per-addition allowances, for any tensor shape
and scalar type. Errors in producing the leaves are not counted, since the reference adds the
interpreted leaves in exact real arithmetic.
-/
theorem accumulate_within {α : Type} [Storage α] {shape : Shape}
    (combine : α → α → α) (interpret : α → ℝ)
    (schedule : ReductionTree (Tensor α shape))
    (allowance : Tensor α shape → Tensor α shape → ℝ)
    (h : schedule.AllNodes (Tensor.map2Spec combine) id (fun x y =>
      Cache.Within (allowance x y) (Tensor.map interpret (Tensor.map2Spec combine x y))
        (Tensor.addSpec (Tensor.map interpret x) (Tensor.map interpret y)))) :
    Cache.Within (schedule.errorBudget (Tensor.map2Spec combine) id allowance)
      (Tensor.map interpret (accumulate combine schedule))
      (accumulate (· + ·) (schedule.map (Tensor.map interpret))) := by
  intro coordinate
  have hError := schedule.abs_eval_sub_exact_le_errorBudget (Tensor.map2Spec combine) id
    (fun x => interpret (x coordinate)) allowance
    (by
      induction schedule with
      | leaf x => trivial
      | node left right hl hr =>
          -- Project each node's tensor bound to this coordinate.
          exact ⟨hl h.1, hr h.2.1, by
            simpa [Tensor.map, Tensor.addSpec, Tensor.map2Spec] using h.2.2 coordinate⟩)
  change |interpret (accumulate combine schedule coordinate) -
    schedule.eval (· + ·) (fun x => interpret (x coordinate))| ≤ _ at hError
  rw [accumulate_apply] at hError
  simpa [accumulate_apply, ReductionTree.eval_map, Function.comp_def, Tensor.map] using hError

/-- With one allowance `δ` per addition the error is at most `nodeCount * δ`, in any order. -/
theorem accumulate_within_nodeCount {α : Type} [Storage α] {shape : Shape}
    (combine : α → α → α) (interpret : α → ℝ)
    (schedule : ReductionTree (Tensor α shape)) (δ : ℝ)
    (h : schedule.AllNodes (Tensor.map2Spec combine) id (fun x y =>
      Cache.Within δ (Tensor.map interpret (Tensor.map2Spec combine x y))
        (Tensor.addSpec (Tensor.map interpret x) (Tensor.map interpret y)))) :
    Cache.Within (schedule.nodeCount * δ)
      (Tensor.map interpret (accumulate combine schedule))
      (accumulate (· + ·) (schedule.map (Tensor.map interpret))) := by
  simpa only [ReductionTree.errorBudget_const] using
    accumulate_within combine interpret schedule (fun _ _ => δ) h

namespace Counterexample

open FloatLib.Floats.Formats.BinaryInterchange

/-- In BF16, `256 + 1` is a tie and rounds back to the even neighbor `256`. These numbers
illustrate the mechanism and are not recorded gradients from NVIDIA's run. -/
theorem bf16_add_one_eq_self :
    Model.add (Model.roundRat .bfloat16 false 256 1) (Model.roundRat .bfloat16 false 1 1) =
      Model.roundRat .bfloat16 false 256 1 := by
  decide +kernel

/-- In binary32 the same addition is exact. -/
theorem binary32_add_one_eq_sum :
    Model.add (Model.roundRat .binary32 false 256 1) (Model.roundRat .binary32 false 1 1) =
      Model.roundRat .binary32 false 257 1 := by
  decide +kernel

end Counterexample

end Nemotron.Training
