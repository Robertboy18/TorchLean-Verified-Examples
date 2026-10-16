/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Primitives

/-!
# AttnRes depth schedule

Track the completed blocks and the current partial block, including an incomplete final block.
The typed graph and the reference update use the same boundary condition.

Reference: https://arxiv.org/abs/2607.24653, Section 2.2, Eq. 10.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec

open Spec TorchLean NN.GraphSpec.DAG

namespace DepthSchedule

open Spec TorchLean
open TorchLean.Tensor
open NN.GraphSpec.DAG
open Runtime.Autograd.Torch

/-- Number of completed AttnRes sources after `processedLayers` decoder layers.

The token embedding is always the first source and every full block adds one more. The current
incomplete block is carried separately.
-/
def completedCount (blockSize processedLayers : Nat) : Nat :=
  1 + processedLayers / blockSize

/-- Whether the current AttnRes block already contains an earlier decoder layer. -/
def partialPresent (blockSize processedLayers : Nat) : Bool :=
  decide (processedLayers % blockSize ≠ 0)

/-- Every packed depth state contains at least the embedding row. -/
theorem completedCount_pos (blockSize processedLayers : Nat) :
    0 < completedCount blockSize processedLayers := by
  simp [completedCount]

/-- Completing a depth block increases the number of packed sources by one. -/
theorem completedCount_succ_of_boundary {blockSize processedLayers : Nat}
    (_hBlockSize : 0 < blockSize) (hBoundary : (processedLayers + 1) % blockSize = 0) :
    completedCount blockSize (processedLayers + 1) =
      completedCount blockSize processedLayers + 1 := by
  have hDvd : blockSize ∣ processedLayers + 1 :=
    Nat.dvd_iff_mod_eq_zero.mpr hBoundary
  simp only [completedCount, Nat.succ_div_of_dvd hDvd]
  omega

/-- Away from a depth-block boundary, the completed-source tensor keeps the same row count. -/
theorem completedCount_succ_of_not_boundary {blockSize processedLayers : Nat}
    (_hBlockSize : 0 < blockSize) (hBoundary : (processedLayers + 1) % blockSize ≠ 0) :
    completedCount blockSize (processedLayers + 1) =
      completedCount blockSize processedLayers := by
  have hNotDvd : ¬blockSize ∣ processedLayers + 1 := by
    intro hDvd
    exact hBoundary (Nat.mod_eq_zero_of_dvd hDvd)
  simp [completedCount, Nat.succ_div_of_not_dvd hNotDvd]

/-- Shapes of the packed AttnRes state after a fixed number of decoder layers. -/
abbrev Shapes (blockSize processedLayers modelDim : Nat) : List Shape :=
  [ [completedCount blockSize processedLayers, modelDim],
    [modelDim] ]

/-- Update the mathematical packed depth state after one decoder layer.

At a block boundary the accumulated residual becomes a new completed source and the partial sum
resets to zero. At every other layer only the partial sum changes. The new source count is in the
result type, so a wrong boundary update cannot reach the next graph.
-/
def advance (blockSize processedLayers modelDim : Nat) (hBlockSize : 0 < blockSize)
    (completed : Tensor ℝ
      [completedCount blockSize processedLayers, modelDim])
    (nextPartial : Tensor ℝ [modelDim]) :
    TorchLean.TensorPack ℝ (Shapes blockSize (processedLayers + 1) modelDim) :=
  if hBoundary : (processedLayers + 1) % blockSize = 0 then
    let row : Tensor ℝ [1, modelDim] :=
      Tensor.reshapeSpec nextPartial (by simp [Shape.size])
    let appended := Tensor.concat completed row
    have hCount := completedCount_succ_of_boundary hBlockSize hBoundary
    have hShape :
        ([completedCount blockSize processedLayers + 1, modelDim] : Shape) =
          [completedCount blockSize (processedLayers + 1), modelDim] :=
      congrArg (fun rows => ([rows, modelDim] : Shape)) hCount.symm
    let nextCompleted : Tensor ℝ
        [completedCount blockSize (processedLayers + 1), modelDim] :=
      Tensor.castShape appended hShape
    .cons nextCompleted <| .cons (Tensor.zeros [modelDim]) .nil
  else
    have hCount := completedCount_succ_of_not_boundary hBlockSize hBoundary
    have hShape :
        ([completedCount blockSize processedLayers, modelDim] : Shape) =
          [completedCount blockSize (processedLayers + 1), modelDim] :=
      congrArg (fun rows => ([rows, modelDim] : Shape)) hCount.symm
    let nextCompleted : Tensor ℝ
        [completedCount blockSize (processedLayers + 1), modelDim] :=
      Tensor.castShape completed hShape
    .cons nextCompleted (.cons nextPartial .nil)

/-- Graph block implementing `advance` from the current completed tensor and partial sum. -/
def advanceBlock {Γ : List Shape} (blockSize processedLayers modelDim : Nat)
    (hBlockSize : 0 < blockSize)
    (completed : Term Γ
      [completedCount blockSize processedLayers, modelDim])
    (nextPartial : Term Γ [modelDim]) :
    Block Γ (Shapes blockSize (processedLayers + 1) modelDim) :=
  if hBoundary : (processedLayers + 1) % blockSize = 0 then
    let row : Term Γ [1, modelDim] :=
      Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
        (.cons nextPartial .nil)
    let appended : Term Γ
        [completedCount blockSize processedLayers + 1, modelDim] :=
      GraphSpec.concatLeadingTerm (completedCount blockSize processedLayers) 1
        [modelDim] completed row
    have hCount := completedCount_succ_of_boundary hBlockSize hBoundary
    let nextCompleted : Term Γ
        [completedCount blockSize (processedLayers + 1), modelDim] :=
      Term.cast appended
        (congrArg (fun rows => [rows, modelDim]) hCount.symm)
    let zeroPartial : Term Γ [modelDim] :=
      Term.op (NN.GraphSpec.DAG.PrimOp.zero [modelDim]) .nil
    .ret (.cons nextCompleted (.cons zeroPartial .nil))
  else
    have hCount := completedCount_succ_of_not_boundary hBlockSize hBoundary
    let nextCompleted : Term Γ
        [completedCount blockSize (processedLayers + 1), modelDim] :=
      Term.cast completed
        (congrArg (fun rows => [rows, modelDim]) hCount.symm)
    .ret (.cons nextCompleted (.cons nextPartial .nil))

/-- Evaluating `advanceBlock` gives `advance`. -/
theorem eval_advanceBlock {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (blockSize processedLayers modelDim : Nat) (hBlockSize : 0 < blockSize)
    (completed : Term Γ
      [completedCount blockSize processedLayers, modelDim])
    (nextPartial : Term Γ [modelDim]) :
    Block.eval env
        (advanceBlock blockSize processedLayers modelDim hBlockSize completed nextPartial) =
      advance blockSize processedLayers modelDim hBlockSize (Term.eval env completed)
        (Term.eval env nextPartial) := by
  by_cases hBoundary : (processedLayers + 1) % blockSize = 0
  · rw [advanceBlock.eq_def, dite_eq_left hBoundary]
    simp only [Block.eval, Term.evalArgs, Term.eval_cast]
    rw [GraphSpec.eval_concatLeadingTerm]
    simp only [Term.eval_op, Term.evalArgs, NN.GraphSpec.DAG.PrimOp.reshape_specFwd,
      NN.GraphSpec.DAG.PrimOp.zero_specFwd]
    rw [advance.eq_def, dite_eq_left hBoundary]
    rw [Tensor.eqRec_eq_cast_shape]
    rfl
  · rw [advanceBlock.eq_def, dite_eq_right hBoundary]
    simp only [Block.eval, Term.evalArgs, Term.eval_cast]
    rw [advance.eq_def, dite_eq_right hBoundary]
    rw [Tensor.eqRec_eq_cast_shape]

end DepthSchedule

end GraphSpec
end KimiK3
