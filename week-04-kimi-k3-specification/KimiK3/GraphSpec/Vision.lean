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

This module lowers MoonViT-V2 patch embedding, one encoder block, and the patch merger with its
projector to TorchLean's typed DAG primitives. The block attends over all tokens of a media item:
the packed `wqkv` product is split into queries, keys and values, the 2-D rotary embedding is
`x ⊙ cos + (x R) ⊙ sin` with the fixed pair rotation `R`, and every head runs scaled dot-product
attention as a batched matrix product, a scale, a softmax and a second batched product.

Two operations need care. The encoder's RMSNorm uses PyTorch's default epsilon `2⁻²³`, while
TorchLean's DAG RMSNorm node fixes `10⁻⁵`, so `rmsNormRows` is the same node with the epsilon
passed to TorchLean's runtime `Norm.rmsNorm`. The projector uses the exact erf GELU, and TorchLean
has no `erf`, so the projector graph is cut at that activation: `mergeTerm` ends at the first
projector matrix and `projectTerm` starts at the second.

The rotary tables, the pair rotation, the resize matrices, the frame rows and the attention scale
are fixed tensors. They enter the graphs as inputs, and the theorems say which values they must
hold.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec
namespace Vision

open Spec TorchLean
attribute [local simp] Tensor.stackLeading Tensor.concat
open TorchLean.Tensor
open NN.GraphSpec.DAG

-- Simplifying reflected plans must preserve the types of their kind certificates.
attribute [local implicit_reducible] TorchLean.Tensor.Internal.Check.TransformPlan.normalized

section

open TorchLean.Tensor.Internal

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
      Tensor.stackLeading (fun frame => Tensor.stackLeading (fun row =>
        Spec.matMulSpec ((input.unstack frame).unstack row) weight)) := by
  apply Rep.ext
  rintro ⟨frame, row, column, out, ⟨⟩⟩
  rw [einsum_patches_apply]
  simpa [Spec.get2_eq_apply, Tensor.unstack, Tensor.dim] using
    (Spec.get2_mat_mul_spec (A := (input.unstack frame).unstack row)
      (B := weight) (i := column) (j := out)).symm

end

/-! ## Small bridges between graph nodes and reference operations -/

/-- A matmul node without batch axes is the reference matrix product. -/
private theorem batchMatmulSpec_plain {α : Type} [Storage α] [Context α] {m n p : Nat}
    (left : Tensor α [m, n]) (right : Tensor α [n, p]) :
    Tensor.batchMatmulSpec (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl [])
      left right = Spec.matMulSpec left right := by
  simp [Tensor.batchMatmulSpec]

/-- A matmul node with one shared batch axis multiplies slice by slice. -/
private theorem batchMatmulSpec_batched {α : Type} [Storage α] [Context α] {batch m n p : Nat}
    (left : Tensor α [batch, m, n]) (right : Tensor α [batch, n, p]) :
    Tensor.batchMatmulSpec (Shape.CanBroadcastTo.refl [batch]) (Shape.CanBroadcastTo.refl [batch])
      left right =
      Tensor.dim fun index =>
        Spec.matMulSpec (Tensor.unstack left index) (Tensor.unstack right index) := by
  simp [Tensor.batchMatmulSpec, Tensor.LinearAlgebra.Internal.matmulCommonBatchSpec]

/-- A matmul node whose left matrix is shared across a batch axis. -/
private theorem batchMatmulSpec_sharedLeft {α : Type} [Storage α] [Context α] {batch m n p : Nat}
    (left : Tensor α [m, n]) (right : Tensor α [batch, n, p]) :
    Tensor.batchMatmulSpec (Shape.CanBroadcastTo.scalarTo [batch]) (Shape.CanBroadcastTo.refl [batch])
      left right =
      Tensor.dim fun index => Spec.matMulSpec left (Tensor.unstack right index) := by
  simp [Tensor.batchMatmulSpec, Tensor.LinearAlgebra.Internal.matmulCommonBatchSpec]

/-- Multiplying by a broadcast scalar is `scaleSpec`. -/
private theorem mulSpec_broadcast_scalar {α : Type} [Storage α] [Context α] {s : Shape}
    (input : Tensor α s) (scale : Tensor α []) (h : Shape.CanBroadcastTo [] s) :
    Tensor.mulSpec input (Tensor.broadcastTo h scale) = Tensor.scaleSpec input scale.item := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp only [Tensor.mulSpec, Tensor.scaleSpec, Tensor.map2Spec_apply]
  have hBroadcast :
      TorchLean.Tensor.Internal.Rep.get (Tensor.broadcastTo h scale) coordinate = scale.item := by
    unfold Tensor.broadcastTo Broadcasting.Internal.broadcastPadded
    rw [TorchLean.Tensor.Internal.Rep.broadcast, TorchLean.Tensor.Internal.Rep.pull_apply]
    unfold Tensor.padLeft
    rw [TorchLean.Tensor.Internal.Rep.reshape_apply]
    rfl
  rw [hBroadcast]
  simp [Tensor.map]

/-- Scaling acts on every slice of the leading axis. -/
private theorem scaleSpec_slices {α : Type} [Storage α] [Context α] {count : Nat} {s : Shape}
    (input : Tensor α (.dim count s)) (c : α) :
    Tensor.scaleSpec input c =
      Tensor.dim fun index => Tensor.scaleSpec (Tensor.unstack input index) c := by
  apply TorchLean.Tensor.Internal.Rep.ext
  rintro ⟨index, rest⟩
  simp [Tensor.scaleSpec, Tensor.dim, Tensor.unstack, Tensor.map,
    TorchLean.Tensor.Internal.Rep.unstack_apply]

/-- Softmax over the last axis of a stack of score matrices is softmax of each matrix. -/
private theorem softmaxSpec_slices {α : Type} [Storage α] [Context α] {heads rows columns : Nat}
    (scores : Tensor α [heads, rows, columns]) :
    Activation.softmaxSpec 2 scores =
      Tensor.dim fun head => Activation.softmaxSpec 1 (Tensor.unstack scores head) := by
  rw [← Tensor.dim_unstack scores]
  rfl

/-- Swapping deeper axes commutes with taking a leading slice. -/
private theorem unstack_swapAdjacentAxes_succ {α : Type} [Storage α] {n : Nat} {s : Shape}
    (t : Tensor α (.dim n s)) (depth : Nat) (index : Fin n) :
    Tensor.unstack (Tensor.swapAdjacentAxes t (depth + 1)) index =
      Tensor.swapAdjacentAxes (Tensor.unstack t index) depth := by
  show Tensor.unstack
    (Tensor.dim fun i => Tensor.swapAdjacentAxes (Tensor.unstack t i) depth) index = _
  exact Tensor.unstack_dim _ _

@[simp] private theorem softmax_specFwd {α : Type} [Storage α] [Context α] {s : Shape}
    (axis : Nat) [Spec.Shape.AxisInBounds axis s] (input : Tensor α s) :
    (NN.GraphSpec.DAG.PrimOp.softmax s axis).specFwd (.cons input .nil) =
      Activation.softmaxSpec axis input := by
  rfl

@[simp] private theorem select_zero_specFwd {α : Type} [Storage α] [Context α] {count : Nat}
    {s : Shape} (index : Fin count) (input : Tensor α (.dim count s)) :
    (NN.GraphSpec.DAG.PrimOp.select (.dim count s) 0 index).specFwd (.cons input .nil) =
      Tensor.unstack input index := by
  rfl

@[simp] private theorem broadcast_specFwd {α : Type} [Storage α] [Context α] {source target : Shape}
    (h : source.CanBroadcastTo target) (input : Tensor α source) :
    (NN.GraphSpec.DAG.PrimOp.broadcast h).specFwd (.cons input .nil) =
      Tensor.broadcastTo h input := by
  rfl

/-- Batched scores, a broadcast scale, softmax, and a batched value product are scaled dot-product
attention in every head. -/
private theorem attention_heads {tokens heads headDim : Nat} (hTokens : tokens ≠ 0)
    (queries keys values : Tensor ℝ [heads, tokens, headDim]) (scale : Tensor ℝ [])
    (hScale : scale = Tensor.scalar (1 / Spec.attentionScaleDenom headDim))
    (hBroadcast : Shape.CanBroadcastTo [] [heads, tokens, tokens]) :
    Tensor.batchMatmulSpec (Shape.CanBroadcastTo.refl [heads]) (Shape.CanBroadcastTo.refl [heads])
        (Activation.softmaxSpec 2
          (Tensor.mulSpec
            (Tensor.batchMatmulSpec (Shape.CanBroadcastTo.refl [heads])
              (Shape.CanBroadcastTo.refl [heads]) queries (Tensor.swapAdjacentAxes keys 1))
            (Tensor.broadcastTo hBroadcast scale)))
        values =
      Tensor.dim fun head =>
        Spec.scaledDotProductAttention
          ({ Q := Tensor.unstack queries head
             K := Tensor.unstack keys head
             V := Tensor.unstack values head
             mask := none } : Spec.AttentionContext ℝ tokens tokens headDim hTokens hTokens) := by
  rw [mulSpec_broadcast_scalar, batchMatmulSpec_batched, scaleSpec_slices, softmaxSpec_slices,
    batchMatmulSpec_batched]
  subst hScale
  simp only [Tensor.unstack_dim, Tensor.item_scalar, Spec.scaledDotProductAttention,
    unstack_swapAdjacentAxes_succ]

/-! ## Encoder RMSNorm with an explicit epsilon -/

/-- Row RMSNorm with the epsilon as a parameter.

MoonViT's encoder norms leave `eps` unset, so PyTorch uses `2⁻²³` instead of the DAG node's
`ε = 10⁻⁵`. The reference meaning is `MoonViT.normRows`, and the program is TorchLean's runtime
`Norm.rmsNorm` with the epsilon passed through. -/
def rmsNormRows (rowCount width : Nat) (hWidth : 0 < width) (epsilon : ℚ) :
    PrimOp [[rowCount, width], [width]] [rowCount, width] :=
  { name := s!"rmsNorm({width},eps={epsilon})"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons input (.cons gamma .nil) =>
          MoonViT.normRows hWidth (Rat.cast epsilon : α) input gamma
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input gamma =>
        (Runtime.Autograd.Model.Norm.rmsNorm (m := m) (α := α)
          (leading := [rowCount]) (width := width) hWidth input gamma (Rat.cast epsilon) :
          m (Runtime.Autograd.Model.Ref (m := m) (α := α) [rowCount, width])) }

/-- Pure evaluation of `rmsNormRows`. -/
@[simp] theorem rmsNormRows_specFwd {rowCount width : Nat} (hWidth : 0 < width) (epsilon : ℚ)
    {α : Type} [Storage α] [Context α]
    (input : Tensor α [rowCount, width]) (gamma : Tensor α [width]) :
    (rmsNormRows rowCount width hWidth epsilon).specFwd (.cons input (.cons gamma .nil)) =
      MoonViT.normRows hWidth (Rat.cast epsilon : α) input gamma := by
  rfl

/-- Encoder epsilon `2⁻²³` as a rational number. -/
def encoderEpsilon : ℚ := 1 / 8388608

/-- Cast to `ℝ`, the graph epsilon is the reference encoder epsilon. -/
theorem encoderEpsilon_cast : (Rat.cast encoderEpsilon : ℝ) = MoonViT.encoderNormEpsilon := by
  rfl

/-! ## Patch embedding -/

/-- Patch projection, the separable bilinear resize of the learned table, and the frame rows. -/
def embedTerm {Γ : List Shape}
    (frames rows columns patchFeatures sourceRows sourceColumns hiddenDim : ℕ)
    (patchWeight : Term Γ [patchFeatures, hiddenDim])
    (positionWeight : Term Γ [sourceRows, sourceColumns, hiddenDim])
    (rowWeights : Term Γ [rows, sourceRows])
    (columnWeights : Term Γ [columns, sourceColumns])
    (frameRows : Term Γ [frames, hiddenDim])
    (patches : Term Γ [frames, rows, columns, patchFeatures]) :
    Term Γ [frames, rows, columns, hiddenDim] :=
  let gridShape : Shape := [frames, rows, columns, hiddenDim]
  let projected := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul [frames, rows] [] [frames, rows]
      columns patchFeatures hiddenDim
      (Shape.CanBroadcastTo.refl _) (Shape.CanBroadcastTo.scalarTo _))
    (.cons patches (.cons patchWeight .nil))
  have hFlat : Shape.size [sourceRows, sourceColumns, hiddenDim] =
      Shape.size [sourceRows, sourceColumns * hiddenDim] := by
    simp [Shape.size]
  have hUnflat : Shape.size [rows, sourceColumns * hiddenDim] =
      Shape.size [rows, sourceColumns, hiddenDim] := by
    simp [Shape.size]
  let flat := Term.op
    (NN.GraphSpec.DAG.PrimOp.reshape [sourceRows, sourceColumns, hiddenDim]
      [sourceRows, sourceColumns * hiddenDim] hFlat)
    (.cons positionWeight .nil)
  let alongRowsFlat := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul [] [] [] rows sourceRows (sourceColumns * hiddenDim)
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
    (.cons rowWeights (.cons flat .nil))
  let alongRows := Term.op
    (NN.GraphSpec.DAG.PrimOp.reshape [rows, sourceColumns * hiddenDim]
      [rows, sourceColumns, hiddenDim] hUnflat)
    (.cons alongRowsFlat .nil)
  let resized := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul [] [rows] [rows] columns sourceColumns hiddenDim
      (Shape.CanBroadcastTo.scalarTo [rows]) (Shape.CanBroadcastTo.refl [rows]))
    (.cons columnWeights (.cons alongRows .nil))
  let spatial := Term.op
    (NN.GraphSpec.DAG.PrimOp.broadcast
      (Shape.CanBroadcastTo.expand_dims (Shape.CanBroadcastTo.refl _)))
    (.cons resized .nil)
  let temporalShape : Shape := [frames, 1, 1, hiddenDim]
  have hTemporal : Shape.size [frames, hiddenDim] = temporalShape.size := by
    simp [temporalShape, Shape.size]
  let temporalSource := Term.op
    (NN.GraphSpec.DAG.PrimOp.reshape [frames, hiddenDim] temporalShape hTemporal)
    (.cons frameRows .nil)
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

/-- Evaluating `embedTerm` gives the MoonViT patch embedding `MoonViT.embedPatches`. -/
theorem eval_embedTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (frames rows columns patchFeatures sourceRows sourceColumns hiddenDim : ℕ)
    (patchWeight : Term Γ [patchFeatures, hiddenDim])
    (positionWeight : Term Γ [sourceRows, sourceColumns, hiddenDim])
    (rowWeights : Term Γ [rows, sourceRows])
    (columnWeights : Term Γ [columns, sourceColumns])
    (frameRows : Term Γ [frames, hiddenDim])
    (patches : Term Γ [frames, rows, columns, patchFeatures]) :
    Term.eval env (embedTerm frames rows columns patchFeatures sourceRows sourceColumns hiddenDim
      patchWeight positionWeight rowWeights columnWeights frameRows patches) =
      MoonViT.embedPatches (Term.eval env patchWeight) (Term.eval env positionWeight)
        (Term.eval env rowWeights) (Term.eval env columnWeights) (Term.eval env frameRows)
        (Term.eval env patches) := by
  simp [embedTerm, Term.eval, Term.evalArgs, NN.GraphSpec.DAG.PrimOp.broadcast,
    MoonViT.embedPatches, MoonViT.resizeTable, batchMatmulSpec_plain, batchMatmulSpec_sharedLeft]
  have contraction := @einsum_patches_eq_matMul
  simp only [TorchLean.Tensor.Internal.Lowering.einsumTensorKernel_correct] at contraction ⊢
  dsimp only [id, TorchLean.Tensor.Internal.Rep.castShape] at contraction ⊢
  grw (transparency := all) [contraction]
  simp [Tensor.batchMatmulSpec, Tensor.LinearAlgebra.Internal.matmulCommonBatchSpec,
    Tensor.broadcastTo_expand, Shape.rank]

/-- With the released resize matrices and frame rows bound, the embedding graph is `Model.embed`. -/
theorem eval_embedTerm_model {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    {cfg : VisionConfig} {patchFeatures frames rows columns : ℕ}
    (model : MoonViT.Model ℝ cfg patchFeatures) (hFrames : frames ≤ MoonViT.maxFrames)
    (patchWeight : Term Γ [patchFeatures, cfg.hiddenDim])
    (positionWeight : Term Γ [MoonViT.positionRows, MoonViT.positionColumns, cfg.hiddenDim])
    (rowWeights : Term Γ [rows, MoonViT.positionRows])
    (columnWeights : Term Γ [columns, MoonViT.positionColumns])
    (frameRows : Term Γ [frames, cfg.hiddenDim])
    (patches : Term Γ [frames, rows, columns, patchFeatures])
    (hPatch : Term.eval env patchWeight = model.patchWeight)
    (hPosition : Term.eval env positionWeight = model.positionWeight)
    (hRows : Term.eval env rowWeights = MoonViT.interpolationMatrix MoonViT.positionRows rows)
    (hColumns : Term.eval env columnWeights =
      MoonViT.interpolationMatrix MoonViT.positionColumns columns)
    (hFrameRows : Term.eval env frameRows =
      MoonViT.frameEmbedding frames cfg.hiddenDim hFrames) :
    Term.eval env (embedTerm frames rows columns patchFeatures MoonViT.positionRows
      MoonViT.positionColumns cfg.hiddenDim
      patchWeight positionWeight rowWeights columnWeights frameRows patches) =
      model.embed (Term.eval env patches) hFrames := by
  rw [eval_embedTerm, hPatch, hPosition, hRows, hColumns, hFrameRows]
  rfl

/-! ## Encoder block -/

/-- Primitive DAG for one `MoonViTEncoderLayer` over the tokens of one media item. -/
def blockTerm {Γ : List Shape} (tokens heads hiddenDim headDim intermediateDim : ℕ)
    (hHidden : 0 < hiddenDim)
    (attentionNorm : Term Γ [hiddenDim])
    (qkvWeight : Term Γ [hiddenDim, 3 * (heads * headDim)])
    (outputWeight : Term Γ [heads * headDim, hiddenDim])
    (feedForwardNorm : Term Γ [hiddenDim])
    (feedForwardInput : Term Γ [hiddenDim, intermediateDim])
    (feedForwardOutput : Term Γ [intermediateDim, hiddenDim])
    (cosTable sinTable : Term Γ [tokens, heads * headDim])
    (rotation : Term Γ [heads * headDim, heads * headDim])
    (scale : Term Γ [])
    (input : Term Γ [tokens, hiddenDim]) :
    Term Γ [tokens, hiddenDim] :=
  let width := heads * headDim
  let rope (x : Term Γ [tokens, width]) : Term Γ [tokens, width] :=
    let direct := Term.op (NN.GraphSpec.DAG.PrimOp.mul [tokens, width])
      (.cons x (.cons cosTable .nil))
    let rotated := Term.op
      (NN.GraphSpec.DAG.PrimOp.matmul [] [] [] tokens width width
        (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
      (.cons x (.cons rotation .nil))
    let turned := Term.op (NN.GraphSpec.DAG.PrimOp.mul [tokens, width])
      (.cons rotated (.cons sinTable .nil))
    Term.op (NN.GraphSpec.DAG.PrimOp.add [tokens, width]) (.cons direct (.cons turned .nil))
  have hHeads : Shape.size [tokens, width] = Shape.size [tokens, heads, headDim] := by
    simp [width, Shape.size]
  let splitHeads (x : Term Γ [tokens, width]) : Term Γ [heads, tokens, headDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth [tokens, heads, headDim] 0)
      (.cons (Term.op (NN.GraphSpec.DAG.PrimOp.reshape [tokens, width]
        [tokens, heads, headDim] hHeads) (.cons x .nil)) .nil)
  let normalized := Term.op (rmsNormRows tokens hiddenDim hHidden encoderEpsilon)
    (.cons input (.cons attentionNorm .nil))
  let packed := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul [] [] [] tokens hiddenDim (3 * width)
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
    (.cons normalized (.cons qkvWeight .nil))
  have hSplit : Shape.size [tokens, 3 * width] = Shape.size [tokens, 3, width] := by
    simp [Shape.size]
  let split : Term Γ [3, tokens, width] := Term.op
    (NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth [tokens, 3, width] 0)
    (.cons (Term.op (NN.GraphSpec.DAG.PrimOp.reshape [tokens, 3 * width] [tokens, 3, width]
      hSplit) (.cons packed .nil)) .nil)
  let pick (index : Fin 3) : Term Γ [tokens, width] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.select [3, tokens, width] 0 index) (.cons split .nil)
  let queries := splitHeads (rope (pick 0))
  let keys := splitHeads (rope (pick 1))
  let values := splitHeads (pick 2)
  let transposedKeys : Term Γ [heads, headDim, tokens] := Term.op
    (NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth [heads, tokens, headDim] 1) (.cons keys .nil)
  let scores := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads] tokens headDim tokens
      (Shape.CanBroadcastTo.refl _) (Shape.CanBroadcastTo.refl _))
    (.cons queries (.cons transposedKeys .nil))
  let broadcastScale := Term.op
    (NN.GraphSpec.DAG.PrimOp.broadcast (Shape.CanBroadcastTo.scalarTo [heads, tokens, tokens]))
    (.cons scale .nil)
  let scaled := Term.op (NN.GraphSpec.DAG.PrimOp.mul [heads, tokens, tokens])
    (.cons scores (.cons broadcastScale .nil))
  let weights := Term.op (NN.GraphSpec.DAG.PrimOp.softmax [heads, tokens, tokens] 2)
    (.cons scaled .nil)
  let attended := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads] tokens tokens headDim
      (Shape.CanBroadcastTo.refl _) (Shape.CanBroadcastTo.refl _))
    (.cons weights (.cons values .nil))
  have hCombine : Shape.size [tokens, heads, headDim] = Shape.size [tokens, width] := by
    simp [width, Shape.size]
  let combined := Term.op
    (NN.GraphSpec.DAG.PrimOp.reshape [tokens, heads, headDim] [tokens, width] hCombine)
    (.cons (Term.op (NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth [heads, tokens, headDim] 0)
      (.cons attended .nil)) .nil)
  let projected := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul [] [] [] tokens width hiddenDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
    (.cons combined (.cons outputWeight .nil))
  let residual := Term.op (NN.GraphSpec.DAG.PrimOp.add [tokens, hiddenDim])
    (.cons input (.cons projected .nil))
  let normalizedResidual := Term.op (rmsNormRows tokens hiddenDim hHidden encoderEpsilon)
    (.cons residual (.cons feedForwardNorm .nil))
  let hidden := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul [] [] [] tokens hiddenDim intermediateDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
    (.cons normalizedResidual (.cons feedForwardInput .nil))
  let activated := Term.op (NN.GraphSpec.DAG.PrimOp.gelu [tokens, intermediateDim])
    (.cons hidden .nil)
  let delta := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul [] [] [] tokens intermediateDim hiddenDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
    (.cons activated (.cons feedForwardOutput .nil))
  Term.op (NN.GraphSpec.DAG.PrimOp.add [tokens, hiddenDim]) (.cons residual (.cons delta .nil))

/-- Evaluating the primitive block term gives the reference `MoonViTEncoderLayer`. -/
theorem eval_blockTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (tokens heads hiddenDim headDim intermediateDim : ℕ)
    (hTokens : 0 < tokens) (hHidden : 0 < hiddenDim)
    (attentionNorm : Term Γ [hiddenDim])
    (qkvWeight : Term Γ [hiddenDim, 3 * (heads * headDim)])
    (outputWeight : Term Γ [heads * headDim, hiddenDim])
    (feedForwardNorm : Term Γ [hiddenDim])
    (feedForwardInput : Term Γ [hiddenDim, intermediateDim])
    (feedForwardOutput : Term Γ [intermediateDim, hiddenDim])
    (cosTable sinTable : Term Γ [tokens, heads * headDim])
    (rotation : Term Γ [heads * headDim, heads * headDim])
    (scale : Term Γ [])
    (input : Term Γ [tokens, hiddenDim])
    (hRotation : Term.eval env rotation = MoonViT.pairRotation (heads * headDim))
    (hScale : Term.eval env scale = Tensor.scalar (1 / Spec.attentionScaleDenom headDim)) :
    Term.eval env (blockTerm tokens heads hiddenDim headDim intermediateDim hHidden
      attentionNorm qkvWeight outputWeight feedForwardNorm feedForwardInput feedForwardOutput
      cosTable sinTable rotation scale input) =
      (let block : MoonViT.Block ℝ heads hiddenDim headDim intermediateDim :=
        { attentionNormScale := Term.eval env attentionNorm
          qkvWeight := Term.eval env qkvWeight
          outputWeight := Term.eval env outputWeight
          feedForwardNormScale := Term.eval env feedForwardNorm
          feedForward :=
            { inputWeight := Term.eval env feedForwardInput
              outputWeight := Term.eval env feedForwardOutput } }
       block.forward MoonViT.encoderNormEpsilon (Term.eval env cosTable) (Term.eval env sinTable)
         (Term.eval env input) (Nat.ne_of_gt hTokens) hHidden) := by
  simp only [blockTerm, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.reshape_specFwd,
    rmsNormRows_specFwd, encoderEpsilon_cast,
    NN.GraphSpec.DAG.PrimOp.add_specFwd,
    NN.GraphSpec.DAG.PrimOp.mul_specFwd,
    NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth_specFwd,
    NN.GraphSpec.DAG.PrimOp.matmul_specFwd,
    NN.GraphSpec.DAG.PrimOp.gelu_specFwd,
    softmax_specFwd, select_zero_specFwd, broadcast_specFwd,
    batchMatmulSpec_plain, hRotation]
  rw [attention_heads (Nat.ne_of_gt hTokens) _ _ _ _ hScale]
  simp only [MoonViT.Block.forward, MoonViT.Block.attention, MoonViT.Block.feedForwardPass,
    MoonViT.Block.splitQKV, MoonViT.applyRope, Spec.splitHeadsSpec, Spec.combineHeadsSpec]

/-! ## Patch merger and projector -/

/-- Temporal pooling, `2 × 2`-style grouping, and the first projector matrix. The exact GELU that
follows has no TorchLean graph node, so the graph stops here. -/
def mergeTerm {Γ : List Shape}
    (frames rows columns mergeHeight mergeWidth hiddenDim : ℕ)
    (hFrames : 0 < frames)
    (hSpatial : 0 < (rows * mergeHeight) * (columns * mergeWidth))
    (hHidden : 0 < hiddenDim)
    (firstWeight : Term Γ [mergeHeight * mergeWidth * hiddenDim,
      mergeHeight * mergeWidth * hiddenDim])
    (grid : Term Γ [frames, rows * mergeHeight, columns * mergeWidth, hiddenDim]) :
    Term Γ [rows * columns, mergeHeight * mergeWidth * hiddenDim] :=
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
  Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul [] [] [] (rows * columns) mergedDim mergedDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
    (.cons merged (.cons firstWeight .nil))

/-- Evaluating `mergeTerm` pools frames, groups patches, and applies the first projector matrix. -/
theorem eval_mergeTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (frames rows columns mergeHeight mergeWidth hiddenDim : ℕ)
    (hFrames : 0 < frames)
    (hSpatial : 0 < (rows * mergeHeight) * (columns * mergeWidth))
    (hHidden : 0 < hiddenDim)
    (firstWeight : Term Γ [mergeHeight * mergeWidth * hiddenDim,
      mergeHeight * mergeWidth * hiddenDim])
    (grid : Term Γ [frames, rows * mergeHeight, columns * mergeWidth, hiddenDim]) :
    Term.eval env (mergeTerm frames rows columns mergeHeight mergeWidth hiddenDim
      hFrames hSpatial hHidden firstWeight grid) =
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
       matMulSpec merged (Term.eval env firstWeight)) := by
  simp only [mergeTerm, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.reduceMean_specFwd,
    NN.GraphSpec.DAG.PrimOp.reshape_specFwd,
    NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth_specFwd,
    Term.eval_cast, NN.GraphSpec.DAG.PrimOp.matmul_specFwd, batchMatmulSpec_plain]

/-- Second projector matrix and final RMSNorm (`ε = 10⁻⁵`, TorchLean's standard node). -/
def projectTerm {Γ : List Shape} (tokens mergedDim textDim : ℕ) (hText : 0 < textDim)
    (secondWeight : Term Γ [mergedDim, textDim])
    (outputNormScale : Term Γ [textDim])
    (hidden : Term Γ [tokens, mergedDim]) :
    Term Γ [tokens, textDim] :=
  let projected := Term.op
    (NN.GraphSpec.DAG.PrimOp.matmul [] [] [] tokens mergedDim textDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
    (.cons hidden (.cons secondWeight .nil))
  Term.op (NN.GraphSpec.DAG.PrimOp.rmsNorm [tokens] textDim hText)
    (.cons projected (.cons outputNormScale .nil))

/-- Evaluating `projectTerm` gives the projector's second matrix and post-norm. -/
theorem eval_projectTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (tokens mergedDim textDim : ℕ) (hText : 0 < textDim)
    (secondWeight : Term Γ [mergedDim, textDim])
    (outputNormScale : Term Γ [textDim])
    (hidden : Term Γ [tokens, mergedDim]) :
    Term.eval env (projectTerm tokens mergedDim textDim hText secondWeight outputNormScale hidden) =
      RMSNorm.rows hText (matMulSpec (Term.eval env hidden) (Term.eval env secondWeight))
        (Term.eval env outputNormScale) := by
  simp only [projectTerm, Term.eval_op, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.matmul_specFwd,
    NN.GraphSpec.DAG.PrimOp.rmsNorm_specFwd, batchMatmulSpec_plain]
  rfl

/-- The two projector graphs with the exact GELU between them compute `Model.mergeAndProject`. -/
theorem mergeAndProject_eq_graphs {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    {cfg : VisionConfig} {patchFeatures frames rows columns : ℕ}
    (model : MoonViT.Model ℝ cfg patchFeatures)
    (hFrames : 0 < frames)
    (hSpatial : 0 < (rows * cfg.mergeHeight) * (columns * cfg.mergeWidth))
    (hHidden : 0 < cfg.hiddenDim) (hText : 0 < cfg.textHiddenDim)
    (firstWeight : Term Γ [cfg.mergeHeight * cfg.mergeWidth * cfg.hiddenDim,
      cfg.mergeHeight * cfg.mergeWidth * cfg.hiddenDim])
    (secondWeight : Term Γ [cfg.mergeHeight * cfg.mergeWidth * cfg.hiddenDim, cfg.textHiddenDim])
    (outputNormScale : Term Γ [cfg.textHiddenDim])
    (grid : Term Γ [frames, rows * cfg.mergeHeight, columns * cfg.mergeWidth, cfg.hiddenDim])
    (hidden : Term Γ [rows * columns, cfg.mergeHeight * cfg.mergeWidth * cfg.hiddenDim])
    (hFirst : Term.eval env firstWeight = model.projector.firstWeight)
    (hSecond : Term.eval env secondWeight = model.projector.secondWeight)
    (hNorm : Term.eval env outputNormScale = model.projector.outputNormScale)
    (hHiddenValue : Term.eval env hidden = MoonViT.geluExactSpec
      (Term.eval env (mergeTerm frames rows columns cfg.mergeHeight cfg.mergeWidth cfg.hiddenDim
        hFrames hSpatial hHidden firstWeight grid))) :
    Term.eval env (projectTerm (rows * columns) _ cfg.textHiddenDim hText
      secondWeight outputNormScale hidden) =
      model.mergeAndProject (Term.eval env grid) hFrames hText := by
  rw [eval_projectTerm, hHiddenValue, eval_mergeTerm]
  simp only [hFirst, hSecond, hNorm]
  rfl

end Vision
end GraphSpec
end KimiK3
