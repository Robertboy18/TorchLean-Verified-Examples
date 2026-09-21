/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.Vision
public import KimiK3.GraphSpec.Primitives
import NN.Proofs.Tensor.Basic.Folds

/-!
# MoonViT graphs

This module lowers MoonViT-V2 patch embedding, one divided-attention block, and the final
merge/project stage to TorchLean's ordinary typed DAG primitives. Spatial attention is batched over
frames; temporal attention is batched over spatial locations after an explicit axis swap. No
Kimi-specific opaque graph operation is introduced.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec
namespace Vision

open Spec TorchLean
open TorchLean.Tensor
open NN.GraphSpec.DAG

-- Simplifying reflected plans must preserve the types of their kind certificates.
attribute [local implicit_reducible] TorchLean.Tensor.Internal.Check.TransformPlan.normalized

section

open TorchLean.Tensor.Internal

-- Each output coordinate contracts only the feature axis. Identifying that axis with
-- `Fin features` connects the checked einsum plan to the graph's ordinary matrix product.
private theorem einsum_features_apply {frames patches features hidden : Nat}
    (input : Tensor ℝ [frames, patches, features])
    (weight : Tensor ℝ [features, hidden])
    (frame : Fin frames) (patch : Fin patches) (out : Fin hidden) :
    (einsum input, weight "frame patch input, input output -> frame patch output")
      (frame, patch, out, PUnit.unit) =
      ∑ feature : Fin features,
        input (frame, patch, feature, PUnit.unit) * weight (feature, out, PUnit.unit) := by
  rw [Lowering.einsumTensorKernel_correct]
  let contractedCoordinateEquiv :
      AxisTuple
          (fun axis =>
            if axis = Check.EinsumAxis.named "frame" then frames
            else if axis = Check.EinsumAxis.named "patch" then patches
            else if axis = Check.EinsumAxis.named "input" then features
            else if axis = Check.EinsumAxis.named "output" then hidden
            else 1)
          [Check.EinsumAxis.named "input"] ≃ Fin features :=
    (Equiv.piCongrRight fun index : Fin 1 => by
      have hIndex : index = 0 := Subsingleton.elim _ _
      subst index
      apply finCongr
      simp).trans <| Equiv.piUnique (fun _ : Fin 1 => Fin features)
  refine (Semantics.denoteEinsum_apply_reconstructed _ _ _ ?_ ?_ ?_).trans ?_
  · intro contractedCoordinate
    exact (frame, patch, contractedCoordinateEquiv contractedCoordinate, out, PUnit.unit)
  · intro contractedCoordinate
    rfl
  · intro contractedCoordinate
    apply contractedCoordinateEquiv.injective
    rfl
  · refine Fintype.sum_equiv contractedCoordinateEquiv _ _ ?_
    intro contractedCoordinate
    with_unfolding_all
      grw (transparency := all) [Semantics.einsumProductTensor_get]
      simp only [List.ofFn_succ, List.ofFn_zero, List.prod_cons,
        List.prod_nil, Fin.cases_zero, Fin.cases_succ, mul_one]
      congr 1
      · congr 1
        unfold Check.CheckedEinsum.inputCoordinateOfGlobal
        change Coord.broadcast [frames, patches, features] [frames, patches, features] _ _ = _
        grw (transparency := all) [Coord.broadcast_self]
        rfl
      · congr 1
        unfold Check.CheckedEinsum.inputCoordinateOfGlobal
        change Coord.broadcast [features, hidden] [features, hidden] _ _ = _
        grw (transparency := all) [Coord.broadcast_self]
        rfl

private theorem einsum_features_eq_matMul {frames patches features hidden : Nat}
    (input : Tensor ℝ [frames, patches, features])
    (weight : Tensor ℝ [features, hidden]) :
    (einsum input, weight "frame patch input, input output -> frame patch output") =
      Tensor.dim (fun frame => Spec.matMulSpec (input.unstack frame) weight) := by
  apply Rep.ext
  rintro ⟨frame, patch, out, ⟨⟩⟩
  rw [einsum_features_apply]
  simpa [Spec.get2_eq_apply, Tensor.unstack, Tensor.dim] using
    (Spec.get2_mat_mul_spec (A := input.unstack frame) (B := weight) (i := patch) (j := out)).symm

private theorem linearize_triple {a b c : Nat} (i : Fin a) (j : Fin b) (k : Fin c) :
    (Coord.linearize (s := [a, b, c]) (i, j, k, PUnit.unit)).val =
      k.val + c * j.val + b * c * i.val := by
  have outer := Coord.linearize_cons_val (s := [b, c]) i (j, k, PUnit.unit)
  have middle := Coord.linearize_cons_val (s := [c]) j (k, PUnit.unit)
  have inner := Coord.linearize_cons_val (s := []) k PUnit.unit
  have scalar : (Coord.linearize (s := []) PUnit.unit).val = 0 :=
    Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ (Coord.linearize (s := []) PUnit.unit).isLt)
  simpa only [TorchLean.Tensor.Internal.Shape.size, Nat.mul_one, scalar,
    Nat.one_mul, Nat.zero_add, middle, inner]
    using outer

private theorem rearrange_frame_patch {frames patches features : Nat}
    (input : Tensor ℝ [frames, patches, features]) :
    (rearrange input "frame patch feature -> patch frame feature") =
      Tensor.swapAdjacentAxes input 0 := by
  apply TorchLean.Tensor.Internal.Rep.ext
  rintro ⟨patch, frame, feature, ⟨⟩⟩
  simp only [Tensor.swapAdjacentAxes_zero, Tensor.dim, Spec.get, Tensor.unstack,
    Rep.stack_apply, Rep.unstack_apply]
  change (rearrange input "frame patch feature -> patch frame feature")
    (patch, frame, feature, PUnit.unit) = input (frame, patch, feature, PUnit.unit)
  apply Lowering.rearrangeTensor_apply_eq_of_linearIndex_eq _ (by rfl)
  change rearrangeLinearIndex _
    [Check.AxisId.named "frame", .named "patch", .named "feature"]
    [Check.AxisId.named "patch", .named "frame", .named "feature"]
    (Coord.linearize (s := [patches, frames, features]) (patch, frame, feature, PUnit.unit)).val =
    (Coord.linearize (s := [frames, patches, features]) (frame, patch, feature, PUnit.unit)).val
  simp [Check.PartialAxisLengths.set, Check.PartialAxisLengths.seed,
    Check.SupplementaryLengths.lookup?, rearrangeLinearIndex,
    Rearrangement.Impl.rowMajorCoordinates, Rearrangement.Impl.rowMajorIndex]
  grw (transparency := all) [linearize_triple, linearize_triple]
  have hFeatures : 0 < features := Nat.zero_lt_of_lt feature.isLt
  have hFrame : 0 < frames := Nat.zero_lt_of_lt frame.isLt
  have hTail : feature.val + features * frame.val < frames * features := by
    calc
      feature.val + features * frame.val < features + features * frame.val :=
        Nat.add_lt_add_right feature.isLt _
      _ = (frame.val + 1) * features := by ring
      _ ≤ frames * features := Nat.mul_le_mul_right features frame.isLt
  have hMod :
      (feature.val + features * frame.val + frames * features * patch.val) % features =
        feature.val := by
    simp [Nat.add_mod, Nat.mul_mod, Nat.mod_eq_of_lt feature.isLt]
  rw [hMod, Nat.add_mul_div_left _ _ (Nat.mul_pos hFrame hFeatures),
    Nat.div_eq_of_lt hTail, Nat.zero_add, Nat.add_mul_mod_self_left,
    Nat.mod_eq_of_lt hTail, Nat.add_mul_div_left _ _ hFeatures,
    Nat.div_eq_of_lt feature.isLt, Nat.zero_add]

private theorem rearrange_patch_frame {patches frames features : Nat}
    (input : Tensor ℝ [patches, frames, features]) :
    (rearrange input "patch frame feature -> frame patch feature") =
      Tensor.swapAdjacentAxes input 0 := by
  apply TorchLean.Tensor.Internal.Rep.ext
  rintro ⟨frame, patch, feature, ⟨⟩⟩
  simp only [Tensor.swapAdjacentAxes_zero, Tensor.dim, Spec.get, Tensor.unstack,
    Rep.stack_apply, Rep.unstack_apply]
  change (rearrange input "patch frame feature -> frame patch feature")
    (frame, patch, feature, PUnit.unit) = input (patch, frame, feature, PUnit.unit)
  apply Lowering.rearrangeTensor_apply_eq_of_linearIndex_eq _ (by rfl)
  change rearrangeLinearIndex _
    [Check.AxisId.named "patch", .named "frame", .named "feature"]
    [Check.AxisId.named "frame", .named "patch", .named "feature"]
    (Coord.linearize (s := [frames, patches, features]) (frame, patch, feature, PUnit.unit)).val =
    (Coord.linearize (s := [patches, frames, features]) (patch, frame, feature, PUnit.unit)).val
  simp [Check.PartialAxisLengths.set, Check.PartialAxisLengths.seed,
    Check.SupplementaryLengths.lookup?, rearrangeLinearIndex,
    Rearrangement.Impl.rowMajorCoordinates, Rearrangement.Impl.rowMajorIndex]
  grw (transparency := all) [linearize_triple, linearize_triple]
  have hFeatures : 0 < features := Nat.zero_lt_of_lt feature.isLt
  have hFrame : 0 < patches := Nat.zero_lt_of_lt patch.isLt
  have hTail : feature.val + features * patch.val < patches * features := by
    calc
      feature.val + features * patch.val < features + features * patch.val :=
        Nat.add_lt_add_right feature.isLt _
      _ = (patch.val + 1) * features := by ring
      _ ≤ patches * features := Nat.mul_le_mul_right features patch.isLt
  have hMod :
      (feature.val + features * patch.val + patches * features * frame.val) % features =
        feature.val := by
    simp [Nat.add_mod, Nat.mul_mod, Nat.mod_eq_of_lt feature.isLt]
  rw [hMod, Nat.add_mul_div_left _ _ (Nat.mul_pos hFrame hFeatures),
    Nat.div_eq_of_lt hTail, Nat.zero_add, Nat.add_mul_mod_self_left,
    Nat.mod_eq_of_lt hTail, Nat.add_mul_div_left _ _ hFeatures,
    Nat.div_eq_of_lt feature.isLt, Nat.zero_add]

private theorem einsum_patches_apply {frames rows columns pixels features : Nat}
    (input : Tensor ℝ [frames, rows, columns, pixels])
    (weight : Tensor ℝ [pixels, features])
    (frame : Fin frames) (row : Fin rows) (column : Fin columns) (out : Fin features) :
    (einsum input, weight "frame row column pixel, pixel feature -> frame row column feature")
      (frame, row, column, out, PUnit.unit) =
      ∑ pixel : Fin pixels,
        input (frame, row, column, pixel, PUnit.unit) * weight (pixel, out, PUnit.unit) := by
  rw [Lowering.einsumTensorKernel_correct]
  let pixelCoordinateEquiv :
      AxisTuple
          (fun axis =>
            if axis = Check.EinsumAxis.named "frame" then frames
            else if axis = Check.EinsumAxis.named "row" then rows
            else if axis = Check.EinsumAxis.named "column" then columns
            else if axis = Check.EinsumAxis.named "pixel" then pixels
            else if axis = Check.EinsumAxis.named "feature" then features
            else 1)
          [Check.EinsumAxis.named "pixel"] ≃ Fin pixels :=
    (Equiv.piCongrRight fun index : Fin 1 => by
      have hIndex : index = 0 := Subsingleton.elim _ _
      subst index
      apply finCongr
      simp).trans <| Equiv.piUnique (fun _ : Fin 1 => Fin pixels)
  refine (Semantics.denoteEinsum_apply_reconstructed _ _ _ ?_ ?_ ?_).trans ?_
  · intro pixelCoordinate
    exact (frame, row, column, pixelCoordinateEquiv pixelCoordinate, out, PUnit.unit)
  · intro pixelCoordinate
    rfl
  · intro pixelCoordinate
    apply pixelCoordinateEquiv.injective
    rfl
  · refine Fintype.sum_equiv pixelCoordinateEquiv _ _ ?_
    intro pixelCoordinate
    with_unfolding_all
      grw (transparency := all) [Semantics.einsumProductTensor_get]
      simp only [List.ofFn_succ, List.ofFn_zero, List.prod_cons,
        List.prod_nil, Fin.cases_zero, Fin.cases_succ, mul_one]
      congr 1
      · congr 1
        unfold Check.CheckedEinsum.inputCoordinateOfGlobal
        change Coord.broadcast [frames, rows, columns, pixels]
          [frames, rows, columns, pixels] _ _ = _
        grw (transparency := all) [Coord.broadcast_self]
        rfl
      · congr 1
        unfold Check.CheckedEinsum.inputCoordinateOfGlobal
        change Coord.broadcast [pixels, features] [pixels, features] _ _ = _
        grw (transparency := all) [Coord.broadcast_self]
        rfl

private theorem einsum_patches_eq_matMul {frames rows columns pixels features : Nat}
    (input : Tensor ℝ [frames, rows, columns, pixels])
    (weight : Tensor ℝ [pixels, features]) :
    (einsum input, weight "frame row column pixel, pixel feature -> frame row column feature") =
      Tensor.dim (fun frame => Tensor.dim (fun row =>
        Spec.matMulSpec ((input.unstack frame).unstack row) weight)) := by
  apply Rep.ext
  rintro ⟨frame, row, column, out, ⟨⟩⟩
  rw [einsum_patches_apply]
  simpa [Spec.get2_eq_apply, Tensor.unstack, Tensor.dim] using
    (Spec.get2_mat_mul_spec (A := (input.unstack frame).unstack row)
      (B := weight) (i := column) (j := out)).symm

end

private theorem rmsNormSemantics_batch_tokens {frames patches width : Nat}
    (hWidth : 0 < width) (input : Tensor ℝ [frames, patches, width])
    (gamma : Tensor ℝ [width]) :
    PrimOp.rmsNormSemantics [frames, patches] hWidth gamma input =
      Tensor.mapLeading [frames, patches]
        (fun token => RMSNorm.scalePositive hWidth token gamma) input := by
  rfl

@[simp] private theorem rmsNormSemantics_rows_eq {rowCount width : ℕ}
    (hWidth : 0 < width) (input : Tensor ℝ [rowCount, width])
    (gamma : Tensor ℝ [width]) :
    NN.GraphSpec.DAG.PrimOp.rmsNormSemantics (.dim rowCount .scalar) hWidth gamma input =
      RMSNorm.rows hWidth input gamma := by
  rfl

/-- Patch projection plus broadcast spatial and temporal position embeddings. -/
def embedTerm {Γ : List Shape} (frames rows columns patchFeatures hiddenDim : ℕ)
    (patchWeight : Term Γ [patchFeatures, hiddenDim])
    (spatialPosition : Term Γ [rows, columns, hiddenDim])
    (temporalPosition : Term Γ [frames, hiddenDim])
    (patches : Term Γ [frames, rows, columns, patchFeatures]) :
    Term Γ [frames, rows, columns, hiddenDim] :=
  let gridShape : Shape := [frames, rows, columns, hiddenDim]
  let projected := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul [frames, rows] .scalar [frames, rows]
      columns patchFeatures hiddenDim
      (Shape.CanBroadcastTo.refl _) (Shape.CanBroadcastTo.scalarTo _))
    (.cons patches (.cons patchWeight .nil))
  let spatial := Term.op
    (NN.GraphSpec.DAG.PrimOp.broadcast
      (Shape.CanBroadcastTo.expand_dims (Shape.CanBroadcastTo.refl _)))
    (.cons spatialPosition .nil)
  let temporalShape : Shape := [frames, 1, 1, hiddenDim]
  have hTemporal : Shape.size [frames, hiddenDim] = temporalShape.size := by
    simp [temporalShape, Shape.size]
  let temporalSource := Term.op
    (NN.GraphSpec.DAG.PrimOp.reshape [frames, hiddenDim] temporalShape hTemporal)
    (.cons temporalPosition .nil)
  let temporal := Term.op
    (NN.GraphSpec.DAG.PrimOp.broadcast
      (Shape.CanBroadcastTo.dim_eq <|
        Shape.CanBroadcastTo.dim_1_to_n <|
          Shape.CanBroadcastTo.dim_1_to_n (Shape.CanBroadcastTo.refl _)))
    (.cons temporalSource .nil)
  let positioned := Term.op (NN.GraphSpec.DAG.PrimOp.add gridShape)
    (.cons projected (.cons spatial .nil))
  Term.op (NN.GraphSpec.DAG.PrimOp.add gridShape)
    (.cons positioned (.cons temporal .nil))

/-- Evaluation of `embedTerm` is exactly the MoonViT patch-embedding equation. -/
theorem eval_embedTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (frames rows columns patchFeatures hiddenDim : ℕ)
    (patchWeight : Term Γ [patchFeatures, hiddenDim])
    (spatialPosition : Term Γ [rows, columns, hiddenDim])
    (temporalPosition : Term Γ [frames, hiddenDim])
    (patches : Term Γ [frames, rows, columns, patchFeatures]) :
    Term.eval env (embedTerm frames rows columns patchFeatures hiddenDim patchWeight
      spatialPosition temporalPosition patches) =
      (let projected := einsum (Term.eval env patches), (Term.eval env patchWeight)
          "frame row column pixel, pixel feature -> frame row column feature"
       let spatial := TorchLean.Tensor.broadcastTo
         (Shape.CanBroadcastTo.expand_dims (Shape.CanBroadcastTo.refl _))
         (Term.eval env spatialPosition)
       let temporalSource : Tensor ℝ [frames, 1, 1, hiddenDim] :=
         TorchLean.Tensor.reshapeSpec (Term.eval env temporalPosition) (by simp [Shape.size])
       let temporal := TorchLean.Tensor.broadcastTo
         (Shape.CanBroadcastTo.dim_eq <|
           Shape.CanBroadcastTo.dim_1_to_n <|
             Shape.CanBroadcastTo.dim_1_to_n (Shape.CanBroadcastTo.refl _)) temporalSource
       TorchLean.Tensor.addSpec (TorchLean.Tensor.addSpec projected spatial) temporal) := by
  simp [embedTerm, Term.eval, Term.evalArgs, NN.GraphSpec.DAG.PrimOp.broadcast]
  have contraction := @einsum_patches_eq_matMul
  simp only [TorchLean.Tensor.Internal.Lowering.einsumTensorKernel_correct] at contraction ⊢
  dsimp only [id, TorchLean.Tensor.Internal.Rep.castShape] at contraction ⊢
  grw (transparency := all) [contraction]
  simp [Tensor.matmulSpec, Tensor.LinearAlgebra.Internal.matmulCommonBatchSpec,
    Tensor.broadcastTo_expand, Shape.rank]

/-- Temporal pooling, `2 × 2`-style spatial grouping, and projection to text width. -/
def mergeAndProjectTerm {Γ : List Shape}
    (frames rows columns mergeHeight mergeWidth hiddenDim textDim : ℕ)
    (hFrames : 0 < frames)
    (hSpatial : 0 < (rows * mergeHeight) * (columns * mergeWidth))
    (hHidden : 0 < hiddenDim) (hText : 0 < textDim)
    (firstWeight : Term Γ [mergeHeight * mergeWidth * hiddenDim,
      mergeHeight * mergeWidth * hiddenDim])
    (secondWeight : Term Γ [mergeHeight * mergeWidth * hiddenDim, textDim])
    (outputNormScale : Term Γ [textDim])
    (grid : Term Γ [frames, rows * mergeHeight, columns * mergeWidth, hiddenDim]) :
    Term Γ [rows * columns, textDim] :=
  let wideShape : Shape :=
    [rows * mergeHeight, columns * mergeWidth, hiddenDim]
  let gridShape : Shape :=
    [frames, rows * mergeHeight, columns * mergeWidth, hiddenDim]
  let interleavedShape : Shape :=
    [rows, mergeHeight, columns, mergeWidth, hiddenDim]
  let groupedShape : Shape :=
    [rows, columns, mergeHeight, mergeWidth, hiddenDim]
  let mergedDim := mergeHeight * mergeWidth * hiddenDim
  let mergedShape : Shape := [rows * columns, mergedDim]
  have hInterleaved : wideShape.size = interleavedShape.size := by
    simp [wideShape, interleavedShape, Shape.size, Nat.mul_assoc]
  have hGrouped : interleavedShape.swapAdjacentAtDepth 1 = groupedShape := by
    simp [interleavedShape, groupedShape, Shape.swapAdjacentAtDepth]
  have hMerged : groupedShape.size = mergedShape.size := by
    simp [groupedShape, mergedShape, mergedDim, Shape.size, Nat.mul_assoc]
  have hWideRows : 0 < rows * mergeHeight := Nat.pos_of_mul_pos_right hSpatial
  have hWideColumns : 0 < columns * mergeWidth := Nat.pos_of_mul_pos_left hSpatial
  letI : Shape.HasNonemptyAxis 0 gridShape :=
    Shape.hasNonemptyAxisZeroOfPos hFrames
  letI : Shape.WellFormed gridShape := ⟨by
    simp [gridShape, Shape.wellFormed, hFrames, hWideRows, hWideColumns, hHidden]⟩
  let pooled := Term.op (NN.GraphSpec.DAG.PrimOp.reduceMean gridShape 0) (.cons grid .nil)
  let interleaved := Term.op
    (NN.GraphSpec.DAG.PrimOp.reshape wideShape interleavedShape hInterleaved)
    (.cons pooled .nil)
  let groupedRaw := Term.op
    (NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth interleavedShape 1)
    (.cons interleaved .nil)
  let grouped := Term.cast groupedRaw hGrouped
  let merged := Term.op
    (NN.GraphSpec.DAG.PrimOp.reshape groupedShape mergedShape hMerged) (.cons grouped .nil)
  let hidden := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul .scalar .scalar .scalar (rows * columns) mergedDim mergedDim
      (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar))
    (.cons merged (.cons firstWeight .nil))
  let activated := Term.op
    (NN.GraphSpec.DAG.PrimOp.gelu [rows * columns, mergedDim]) (.cons hidden .nil)
  let projected := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul .scalar .scalar .scalar (rows * columns) mergedDim textDim
      (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar))
    (.cons activated (.cons secondWeight .nil))
  Term.op (NN.GraphSpec.DAG.PrimOp.rmsNorm (.dim (rows * columns) .scalar) textDim hText)
    (.cons projected (.cons outputNormScale .nil))

/-- Evaluating `mergeAndProjectTerm` gives the reference merge/project calculation. -/
theorem eval_mergeAndProjectTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (frames rows columns mergeHeight mergeWidth hiddenDim textDim : ℕ)
    (hFrames : 0 < frames)
    (hSpatial : 0 < (rows * mergeHeight) * (columns * mergeWidth))
    (hHidden : 0 < hiddenDim) (hText : 0 < textDim)
    (firstWeight : Term Γ [mergeHeight * mergeWidth * hiddenDim,
      mergeHeight * mergeWidth * hiddenDim])
    (secondWeight : Term Γ [mergeHeight * mergeWidth * hiddenDim, textDim])
    (outputNormScale : Term Γ [textDim])
    (grid : Term Γ [frames, rows * mergeHeight, columns * mergeWidth, hiddenDim]) :
    Term.eval env (mergeAndProjectTerm frames rows columns mergeHeight mergeWidth hiddenDim textDim
      hFrames hSpatial hHidden hText firstWeight secondWeight outputNormScale grid) =
      (let wideShape : Shape :=
         [rows * mergeHeight, columns * mergeWidth, hiddenDim]
       let interleavedShape : Shape :=
         [rows, mergeHeight, columns, mergeWidth, hiddenDim]
       let groupedShape : Shape :=
         [rows, columns, mergeHeight, mergeWidth, hiddenDim]
       let mergedDim := mergeHeight * mergeWidth * hiddenDim
       let mergedShape : Shape := [rows * columns, mergedDim]
       have hInterleaved : wideShape.size = interleavedShape.size := by
         simp [wideShape, interleavedShape, Shape.size, Nat.mul_assoc]
       have hGrouped : interleavedShape.swapAdjacentAtDepth 1 = groupedShape := by
         simp [interleavedShape, groupedShape, Shape.swapAdjacentAtDepth]
       have hMerged : groupedShape.size = mergedShape.size := by
         simp [groupedShape, mergedShape, mergedDim, Shape.size, Nat.mul_assoc]
       let pooled : Tensor ℝ wideShape :=
         TorchLean.Tensor.reduceMean 0 (Term.eval env grid)
           (Shape.hasNonemptyAxisZeroOfPos hFrames).proof
       let interleaved := TorchLean.Tensor.reshapeSpec pooled hInterleaved
       let grouped : Tensor ℝ groupedShape :=
         hGrouped ▸ TorchLean.Tensor.swapAdjacentAxes interleaved 1
       let merged := TorchLean.Tensor.reshapeSpec grouped hMerged
       let hidden := Activation.geluSpec
         (Tensor.matmulSpec (Shape.CanBroadcastTo.refl .scalar)
           (Shape.CanBroadcastTo.refl .scalar) merged (Term.eval env firstWeight))
       let projected := Tensor.matmulSpec (Shape.CanBroadcastTo.refl .scalar)
         (Shape.CanBroadcastTo.refl .scalar) hidden (Term.eval env secondWeight)
       RMSNorm.rows hText projected (Term.eval env outputNormScale)) := by
  simp only [mergeAndProjectTerm, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.reduceMean_specFwd,
    NN.GraphSpec.DAG.PrimOp.reshape_specFwd,
    NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth_specFwd,
    Term.eval_cast, NN.GraphSpec.DAG.PrimOp.matmul_specFwd,
    NN.GraphSpec.DAG.PrimOp.gelu_specFwd,
    NN.GraphSpec.DAG.PrimOp.rmsNorm_specFwd,
    rmsNormSemantics_rows_eq]

/-- Primitive DAG for one MoonViT-V2 divided-attention block. -/
def blockTerm {Γ : List Shape} (frames rows columns heads hiddenDim headDim intermediateDim : ℕ)
    (hFrames : 0 < frames) (hSpatial : 0 < rows * columns) (hHidden : 0 < hiddenDim)
    (spatialQuery spatialKey spatialValue : Term Γ [hiddenDim, heads * headDim])
    (spatialOutput : Term Γ [heads * headDim, hiddenDim])
    (temporalQuery temporalKey temporalValue : Term Γ [hiddenDim, heads * headDim])
    (temporalOutput : Term Γ [heads * headDim, hiddenDim])
    (feedForwardInput : Term Γ [hiddenDim, intermediateDim])
    (feedForwardOutput : Term Γ [intermediateDim, hiddenDim])
    (spatialNorm temporalNorm feedForwardNorm : Term Γ [hiddenDim])
    (grid : Term Γ [frames, rows, columns, hiddenDim]) :
    Term Γ [frames, rows, columns, hiddenDim] :=
  let spatialTokens := rows * columns
  let gridShape : Shape := [frames, rows, columns, hiddenDim]
  let spatialShape : Shape := [frames, spatialTokens, hiddenDim]
  have hGridSpatial : gridShape.size = spatialShape.size := by
    simp [gridShape, spatialShape, spatialTokens, Shape.size, Nat.mul_assoc]
  let spatialInput := Term.op
    (NN.GraphSpec.DAG.PrimOp.reshape gridShape spatialShape hGridSpatial) (.cons grid .nil)
  let normalizedSpatial := Term.op
    (NN.GraphSpec.DAG.PrimOp.rmsNorm [frames, spatialTokens] hiddenDim hHidden)
    (.cons spatialInput (.cons spatialNorm .nil))
  let spatialDelta := Term.op
    (NN.GraphSpec.DAG.PrimOp.multiHeadAttention
      (.dim frames .scalar) spatialTokens heads hiddenDim headDim hSpatial)
    (.cons spatialQuery <| .cons spatialKey <| .cons spatialValue <|
      .cons spatialOutput <| .cons normalizedSpatial .nil)
  let spatial := Term.op (NN.GraphSpec.DAG.PrimOp.add spatialShape)
    (.cons spatialInput (.cons spatialDelta .nil))
  let temporalShape : Shape := [spatialTokens, frames, hiddenDim]
  let temporalInput := Term.op
    (NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth spatialShape 0) (.cons spatial .nil)
  let normalizedTemporal := Term.op
    (NN.GraphSpec.DAG.PrimOp.rmsNorm [spatialTokens, frames] hiddenDim hHidden)
    (.cons temporalInput (.cons temporalNorm .nil))
  let temporalDelta := Term.op
    (NN.GraphSpec.DAG.PrimOp.multiHeadAttention
      (.dim spatialTokens .scalar) frames heads hiddenDim headDim hFrames)
    (.cons temporalQuery <| .cons temporalKey <| .cons temporalValue <|
      .cons temporalOutput <| .cons normalizedTemporal .nil)
  let temporalResidual := Term.op (NN.GraphSpec.DAG.PrimOp.add temporalShape)
    (.cons temporalInput (.cons temporalDelta .nil))
  let temporal := Term.op
    (NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth temporalShape 0) (.cons temporalResidual .nil)
  let normalizedFeedForward := Term.op
    (NN.GraphSpec.DAG.PrimOp.rmsNorm [frames, spatialTokens] hiddenDim hHidden)
    (.cons temporal (.cons feedForwardNorm .nil))
  let hidden := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul [frames] .scalar [frames]
      spatialTokens hiddenDim intermediateDim
      (Shape.CanBroadcastTo.refl _) (Shape.CanBroadcastTo.scalarTo _))
    (.cons normalizedFeedForward (.cons feedForwardInput .nil))
  let activated := Term.op (NN.GraphSpec.DAG.PrimOp.gelu [frames, spatialTokens, intermediateDim])
    (.cons hidden .nil)
  let delta := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul [frames] .scalar [frames]
      spatialTokens intermediateDim hiddenDim
      (Shape.CanBroadcastTo.refl _) (Shape.CanBroadcastTo.scalarTo _))
    (.cons activated (.cons feedForwardOutput .nil))
  let output := Term.op (NN.GraphSpec.DAG.PrimOp.add spatialShape)
    (.cons temporal (.cons delta .nil))
  Term.op (NN.GraphSpec.DAG.PrimOp.reshape spatialShape gridShape hGridSpatial.symm)
    (.cons output .nil)

/-- Evaluating the primitive block term gives the theorem-level MoonViT block. -/
theorem eval_blockTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (frames rows columns heads hiddenDim headDim intermediateDim : ℕ)
    (hFrames : 0 < frames) (hSpatial : 0 < rows * columns) (hHidden : 0 < hiddenDim)
    (spatialQuery spatialKey spatialValue : Term Γ [hiddenDim, heads * headDim])
    (spatialOutput : Term Γ [heads * headDim, hiddenDim])
    (temporalQuery temporalKey temporalValue : Term Γ [hiddenDim, heads * headDim])
    (temporalOutput : Term Γ [heads * headDim, hiddenDim])
    (feedForwardInput : Term Γ [hiddenDim, intermediateDim])
    (feedForwardOutput : Term Γ [intermediateDim, hiddenDim])
    (spatialNorm temporalNorm feedForwardNorm : Term Γ [hiddenDim])
    (grid : Term Γ [frames, rows, columns, hiddenDim]) :
    Term.eval env (blockTerm frames rows columns heads hiddenDim headDim intermediateDim
      hFrames hSpatial hHidden
      spatialQuery spatialKey spatialValue spatialOutput
      temporalQuery temporalKey temporalValue temporalOutput
      feedForwardInput feedForwardOutput spatialNorm temporalNorm feedForwardNorm grid) =
      (let block : MoonViT.Block ℝ heads hiddenDim headDim intermediateDim :=
        { spatialAttention :=
            { queryWeight := Term.eval env spatialQuery
              keyWeight := Term.eval env spatialKey
              valueWeight := Term.eval env spatialValue
              outputWeight := Term.eval env spatialOutput }
          temporalAttention :=
            { queryWeight := Term.eval env temporalQuery
              keyWeight := Term.eval env temporalKey
              valueWeight := Term.eval env temporalValue
              outputWeight := Term.eval env temporalOutput }
          feedForward :=
            { inputWeight := Term.eval env feedForwardInput
              outputWeight := Term.eval env feedForwardOutput }
          spatialNormScale := Term.eval env spatialNorm
          temporalNormScale := Term.eval env temporalNorm
          feedForwardNormScale := Term.eval env feedForwardNorm }
       block.forward (Term.eval env grid) hFrames hSpatial hHidden) := by
  simp only [blockTerm, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.reshape_specFwd,
    NN.GraphSpec.DAG.PrimOp.rmsNorm_specFwd,
    NN.GraphSpec.DAG.PrimOp.multiHeadAttention_specFwd,
    NN.GraphSpec.DAG.PrimOp.add_specFwd,
    NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth_specFwd,
    NN.GraphSpec.DAG.PrimOp.matmul_specFwd,
    NN.GraphSpec.DAG.PrimOp.gelu_specFwd,
    MoonViT.Block.forward, MoonViT.Block.spatialPass, MoonViT.Block.temporalPass,
    MoonViT.Block.feedForwardPass, MoonViT.Block.attendBatch]
  have contraction := @einsum_features_eq_matMul
  simp only [TorchLean.Tensor.Internal.Lowering.einsumTensorKernel_correct] at contraction ⊢
  dsimp only [id, TorchLean.Tensor.Internal.Rep.castShape] at contraction ⊢
  grw (transparency := all) [contraction, contraction]
  have spatialSwap := @rearrange_frame_patch
  have temporalSwap := @rearrange_patch_frame
  dsimp only [id, TorchLean.Tensor.Internal.Rep.castShape] at spatialSwap temporalSwap
  simp only [spatialSwap, temporalSwap]
  simp only [rmsNormSemantics_batch_tokens,
    RMSNorm.rows, Tensor.mapLeading]
  simp only [Tensor.matmulSpec, Tensor.LinearAlgebra.Internal.matmulCommonBatchSpec]
  simp

end Vision
end GraphSpec
end KimiK3
