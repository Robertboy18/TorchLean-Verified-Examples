/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.Vision
public import KimiK3.GraphSpec.Core

/-!
# MoonViT tensor compositions

These terms compose the patch projection, encoder and patch merger with TorchLean tensor
operations. The encoder uses the report's RMSNorm epsilon and TorchLean's batched attention.
The merger ends before the exact GELU, which is supplied to the projection as a tensor.
-/

@[expose] public section

namespace KimiK3.GraphSpec.Vision

open TorchLean TorchLean.Tensor

private def matmulTerm {Γ : List Shape} {m n p : Nat}
    (left : Term Γ [m, n]) (right : Term Γ [n, p]) : Term Γ [m, p] :=
  Term.op (ins := [[m, n], [n, p]])
    ⟨fun | .cons a (.cons b .nil) => Tensor.matmul a b⟩ (.cons left (.cons right .nil))

private def zipTerm {Γ : List Shape} {s : Shape}
    (operation : ∀ {α : Type}, [Context α] → α → α → α)
    (left right : Term Γ s) : Term Γ s :=
  Term.op (ins := [s, s]) ⟨fun | .cons a (.cons b .nil) => Tensor.zipWith operation a b⟩
    (.cons left (.cons right .nil))

/-- Row RMSNorm with an explicit epsilon. MoonViT uses the binary32 machine epsilon. -/
def rmsNormRows (rowCount width : Nat) (hWidth : 0 < width) (epsilon : ℚ) :
    PrimOp [[rowCount, width], [width]] [rowCount, width] :=
  ⟨fun | .cons input (.cons gamma .nil) => MoonViT.normRows hWidth (Rat.cast epsilon) input gamma⟩

/-- Pure evaluation of row RMSNorm. -/
@[simp] theorem rmsNormRows_specFwd {rowCount width : Nat} (hWidth : 0 < width) (epsilon : ℚ)
    {α : Type} [Storage α] [Context α]
    (input : Tensor α [rowCount, width]) (gamma : Tensor α [width]) :
    (rmsNormRows rowCount width hWidth epsilon).specFwd (.cons input (.cons gamma .nil)) =
      MoonViT.normRows hWidth (Rat.cast epsilon : α) input gamma :=
  rfl

/-- Encoder epsilon `2⁻²³` as a rational number. -/
def encoderEpsilon : ℚ := 1 / 8388608

/-- Cast to `ℝ`, the graph epsilon is the reference encoder epsilon. -/
theorem encoderEpsilon_cast : (Rat.cast encoderEpsilon : ℝ) = MoonViT.encoderNormEpsilon :=
  rfl

/-! ## Patch embedding -/

/-- Patch projection, the separable bilinear resize of the learned table, and the frame rows. -/
@[no_expose] def embedTerm {Γ : List Shape}
    (frames rows columns patchFeatures sourceRows sourceColumns hiddenDim : ℕ)
    (patchWeight : Term Γ [patchFeatures, hiddenDim])
    (positionWeight : Term Γ [sourceRows, sourceColumns, hiddenDim])
    (rowWeights : Term Γ [rows, sourceRows])
    (columnWeights : Term Γ [columns, sourceColumns])
    (frameRows : Term Γ [frames, hiddenDim])
    (patches : Term Γ [frames, rows, columns, patchFeatures]) :
    Term Γ [frames, rows, columns, hiddenDim] :=
  let projected : Term Γ [frames, rows, columns, hiddenDim] :=
    Term.op (ins := [[frames, rows, columns, patchFeatures], [patchFeatures, hiddenDim]])
      ⟨fun input => match input with
        | .cons pixels (.cons weight .nil) => (einsum pixels, weight
            "frame row column pixel, pixel feature -> frame row column feature")⟩
      (.cons patches (.cons patchWeight .nil))
  let spatial : Term Γ [frames, rows, columns, hiddenDim] :=
    Term.op (ins := [[rows, sourceRows], [columns, sourceColumns],
        [sourceRows, sourceColumns, hiddenDim]])
      ⟨fun input => match input with
        | .cons row (.cons column (.cons positions .nil)) =>
            Tensor.repeatLeading frames (MoonViT.resizeTable row column positions)⟩
      (.cons rowWeights (.cons columnWeights (.cons positionWeight .nil)))
  let temporal : Term Γ [frames, rows, columns, hiddenDim] :=
    Term.op (ins := [[frames, hiddenDim]])
      ⟨fun input => match input with
        | .cons rowsAtFrame .nil => Tensor.dim fun frame =>
            Tensor.repeatLeading rows (Tensor.repeatLeading columns (rowsAtFrame.unstack frame))⟩
      (.cons frameRows .nil)
  zipTerm (· + ·) (zipTerm (· + ·) projected spatial) temporal

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
  unfold embedTerm MoonViT.embedPatches
  rfl

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

/-- Tensor composition for one `MoonViTEncoderLayer` over the tokens of one media item. -/
@[no_expose] def blockTerm {Γ : List Shape} (tokens heads hiddenDim headDim intermediateDim : ℕ)
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
    zipTerm (· + ·) (zipTerm (· * ·) x cosTable)
      (zipTerm (· * ·) (matmulTerm x rotation) sinTable)
  let splitHeads (x : Term Γ [tokens, width]) : Term Γ [heads, tokens, headDim] :=
    Term.op (ins := [[tokens, width]])
      ⟨fun | .cons value .nil => MultiHeadAttention.splitHeads value heads headDim rfl⟩
      (.cons x .nil)
  let normalized := Term.op (rmsNormRows tokens hiddenDim hHidden encoderEpsilon)
    (.cons input (.cons attentionNorm .nil))
  let packed := matmulTerm normalized qkvWeight
  let split : Term Γ [3, tokens, width] := Term.op (ins := [[tokens, 3 * width]])
    ⟨fun | .cons value .nil => MoonViT.Block.splitQKV value⟩ (.cons packed .nil)
  let pick (index : Fin 3) : Term Γ [tokens, width] :=
    Term.op (ins := [[3, tokens, width]])
      ⟨fun | .cons value .nil => Tensor.unstack value index⟩ (.cons split .nil)
  let queries := splitHeads (rope (pick 0))
  let keys := splitHeads (rope (pick 1))
  let values := splitHeads (pick 2)
  let attended : Term Γ [heads, tokens, headDim] :=
    Term.op (ins := [[heads, tokens, headDim], [heads, tokens, headDim],
        [heads, tokens, headDim], []])
      ⟨fun input => match input with
        | .cons q (.cons k (.cons v (.cons factor .nil))) =>
            Tensor.attention (batch := [heads]) q k v factor.item⟩
      (.cons queries (.cons keys (.cons values (.cons scale .nil))))
  let combined : Term Γ [tokens, width] := Term.op (ins := [[heads, tokens, headDim]])
    ⟨fun | .cons value .nil => MultiHeadAttention.combineHeads value⟩
    (.cons attended .nil)
  let residual := zipTerm (· + ·) input (matmulTerm combined outputWeight)
  let normalizedResidual := Term.op (rmsNormRows tokens hiddenDim hHidden encoderEpsilon)
    (.cons residual (.cons feedForwardNorm .nil))
  let hidden := matmulTerm normalizedResidual feedForwardInput
  let activated : Term Γ [tokens, intermediateDim] := Term.op (ins := [[tokens, intermediateDim]])
    ⟨fun | .cons value .nil => Tensor.map Activation.geluTanh value⟩
    (.cons hidden .nil)
  zipTerm (· + ·) residual (matmulTerm activated feedForwardOutput)

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
    (hScale : Term.eval env scale = Tensor.scalar (1 / (if headDim = 0 then 1 else Context.sqrtNat headDim))) :
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
  simp only [blockTerm, matmulTerm, zipTerm, Term.eval_op, Term.evalArgs,
    rmsNormRows_specFwd, encoderEpsilon_cast, hRotation, hScale, Tensor.item_scalar]
  rfl

/-! ## Patch merger and projector -/

/-- Temporal pooling, `2 × 2`-style grouping, and the first projector matrix. The projection term below takes the result after exact GELU. -/
@[no_expose] def mergeTerm {Γ : List Shape}
    (frames rows columns mergeHeight mergeWidth hiddenDim : ℕ)
    (_hFrames : 0 < frames)
    (_hSpatial : 0 < (rows * mergeHeight) * (columns * mergeWidth))
    (_hHidden : 0 < hiddenDim)
    (firstWeight : Term Γ [mergeHeight * mergeWidth * hiddenDim,
      mergeHeight * mergeWidth * hiddenDim])
    (grid : Term Γ [frames, rows * mergeHeight, columns * mergeWidth, hiddenDim]) :
    Term Γ [rows * columns, mergeHeight * mergeWidth * hiddenDim] :=
  let wideShape : Shape :=
    [rows * mergeHeight, columns * mergeWidth, hiddenDim]
  let interleavedShape : Shape := [rows, mergeHeight, columns, mergeWidth, hiddenDim]
  let groupedShape : Shape := [rows, columns, mergeHeight, mergeWidth, hiddenDim]
  let mergedDim := mergeHeight * mergeWidth * hiddenDim
  let mergedShape : Shape := [rows * columns, mergedDim]
  have hInterleaved : wideShape.size = interleavedShape.size := by
    simp [wideShape, interleavedShape, Shape.size, Nat.mul_assoc]
  have hMerged : groupedShape.size = mergedShape.size := by
    simp [groupedShape, mergedShape, mergedDim, Shape.size, Nat.mul_assoc]
  let pooled : Term Γ wideShape := Term.op (ins := [[frames, rows * mergeHeight,
      columns * mergeWidth, hiddenDim]])
    ⟨fun | .cons value .nil => Tensor.mean value [⟨0, Nat.zero_lt_succ _⟩] false⟩ (.cons grid .nil)
  let interleaved : Term Γ interleavedShape := Term.op (ins := [wideShape])
    ⟨fun | .cons value .nil => Tensor.reshape value interleavedShape hInterleaved⟩
    (.cons pooled .nil)
  let grouped : Term Γ groupedShape := Term.op (ins := [interleavedShape])
    ⟨fun input => match input with
      | .cons value .nil => value.permute (Shape.adjacentSwap interleavedShape 1) groupedShape
          (by rw [Shape.permute_adjacentSwap])⟩ (.cons interleaved .nil)
  let merged : Term Γ mergedShape := Term.op (ins := [groupedShape])
    ⟨fun | .cons value .nil => Tensor.reshape value mergedShape hMerged⟩ (.cons grouped .nil)
  matmulTerm merged firstWeight

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
       have hMerged : groupedShape.size = mergedShape.size := by
         simp [groupedShape, mergedShape, mergedDim, Shape.size, Nat.mul_assoc]
       let pooled : Tensor ℝ wideShape :=
         Tensor.mean (Term.eval env grid) [⟨0, Nat.zero_lt_succ _⟩] false
       let interleaved := Tensor.reshape pooled interleavedShape hInterleaved
       let grouped := interleaved.permute (Shape.adjacentSwap interleavedShape 1) groupedShape (by
         rw [Shape.permute_adjacentSwap])
       let merged := Tensor.reshape grouped mergedShape hMerged
       Tensor.matmul merged (Term.eval env firstWeight)) := by
  unfold mergeTerm
  rfl

/-- Second projector matrix and final RMSNorm (`ε = 10⁻⁵`, the default for RMSNorm). -/
@[no_expose] def projectTerm {Γ : List Shape} (tokens mergedDim textDim : ℕ) (hText : 0 < textDim)
    (secondWeight : Term Γ [mergedDim, textDim])
    (outputNormScale : Term Γ [textDim])
    (hidden : Term Γ [tokens, mergedDim]) :
    Term Γ [tokens, textDim] :=
  let projected := matmulTerm hidden secondWeight
  Term.op (ins := [[tokens, textDim], [textDim]])
    ⟨fun | .cons value (.cons gamma .nil) => RMSNorm.rows hText value gamma⟩
    (.cons projected (.cons outputNormScale .nil))

/-- Evaluating `projectTerm` gives the projector's second matrix and post-norm. -/
theorem eval_projectTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (tokens mergedDim textDim : ℕ) (hText : 0 < textDim)
    (secondWeight : Term Γ [mergedDim, textDim])
    (outputNormScale : Term Γ [textDim])
    (hidden : Term Γ [tokens, mergedDim]) :
    Term.eval env (projectTerm tokens mergedDim textDim hText secondWeight outputNormScale hidden) =
      RMSNorm.rows hText (Tensor.matmul (Term.eval env hidden) (Term.eval env secondWeight))
        (Term.eval env outputNormScale) := by
  unfold projectTerm
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
    (hHiddenValue : Term.eval env hidden = Tensor.gelu
      (Term.eval env (mergeTerm frames rows columns cfg.mergeHeight cfg.mergeWidth cfg.hiddenDim
        hFrames hSpatial hHidden firstWeight grid))) :
    Term.eval env (projectTerm (rows * columns) _ cfg.textHiddenDim hText
      secondWeight outputNormScale hidden) =
      model.mergeAndProject (Term.eval env grid) hFrames hText := by
  rw [eval_projectTerm, hHiddenValue, eval_mergeTerm]
  simp only [hFirst, hSecond, hNorm]
  rfl

end KimiK3.GraphSpec.Vision
