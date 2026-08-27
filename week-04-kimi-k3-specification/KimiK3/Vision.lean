/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Sequence
public import NN.Spec.Models.Transformer

/-!
# MoonViT-V2 specification

Kimi K3 uses one vision encoder for images and videos.  MoonViT-V2 applies shared, bias-free
parameters to both modalities, factorizes attention into spatial and temporal passes, merges each
`2 × 2` group of spatial tokens, and projects the result into the language-model width.

This file describes that dataflow for arbitrary frame and patch-grid dimensions.  The input is an
already extracted grid of flattened patches; pixel decoding, resizing, and patch extraction are
data-pipeline concerns and are intentionally outside the model equation.

Reference: Kimi Team, "Kimi K3: Open Frontier Intelligence", 2026, Section 2.4,
https://arxiv.org/abs/2607.24653.  Exact released dimensions are recorded in `KimiK3.paperConfig`.
-/

@[expose] public section

namespace KimiK3

open Spec
open Tensor

namespace MoonViT

/-- A video/image patch grid with axes `(frame, row, column, feature)`. -/
abbrev Grid (α : Type) (frames rows columns features : Nat) :=
  Tensor α (.dim frames (.dim rows (.dim columns (.dim features .scalar))))

/-- Bias-free MLP used in MoonViT-V2 blocks. -/
structure MLP (α : Type) (inputDim hiddenDim outputDim : Nat) where
  inputWeight : Tensor α (.dim inputDim (.dim hiddenDim .scalar))
  outputWeight : Tensor α (.dim hiddenDim (.dim outputDim .scalar))

namespace MLP

variable {α : Type} [Context α]
variable {inputDim hiddenDim outputDim : Nat}

/-- MoonViT uses the tanh approximation of GELU in its hidden branch. -/
def forward (mlp : MLP α inputDim hiddenDim outputDim)
    (x : Tensor α (.dim inputDim .scalar)) : Tensor α (.dim outputDim .scalar) :=
  vecMatMulSpec (Tensor.mapSpec Activation.Math.geluSpec
    (vecMatMulSpec x mlp.inputWeight)) mlp.outputWeight

end MLP

/-- One divided-attention MoonViT-V2 block. -/
structure Block (α : Type) (heads hiddenDim qkvHeadDim intermediateDim : Nat) where
  spatialAttention : Spec.MultiHeadAttention α heads hiddenDim qkvHeadDim
  temporalAttention : Spec.MultiHeadAttention α heads hiddenDim qkvHeadDim
  feedForward : MLP α hiddenDim intermediateDim hiddenDim
  spatialNormScale : Tensor α (.dim hiddenDim .scalar)
  temporalNormScale : Tensor α (.dim hiddenDim .scalar)
  feedForwardNormScale : Tensor α (.dim hiddenDim .scalar)

namespace Block

variable {α : Type} [Context α]
variable {heads hiddenDim qkvHeadDim intermediateDim frames rows columns : Nat}

/-- Apply one shared attention module independently along a leading batch axis. -/
def attendBatch {batch tokens : Nat}
    (attention : Spec.MultiHeadAttention α heads hiddenDim qkvHeadDim)
    (input : Tensor α (.dim batch (.dim tokens (.dim hiddenDim .scalar))))
    (hTokens : 0 < tokens) :
    Tensor α (.dim batch (.dim tokens (.dim hiddenDim .scalar))) :=
  .dim fun i => attention.forward tokens (Nat.ne_of_gt hTokens) (Spec.get input i) none

/-- Normalize and attend independently within every frame. -/
def spatialPass
    (attention : Spec.MultiHeadAttention α heads hiddenDim qkvHeadDim)
    (normScale : Tensor α (.dim hiddenDim .scalar))
    (grid : Grid α frames rows columns hiddenDim)
    (hFrames : 0 < frames) (hSpatial : 0 < rows * columns) (hHidden : 0 < hiddenDim) :
    Tensor α (.dim frames (.dim (rows * columns) (.dim hiddenDim .scalar))) :=
  let spatialTokens := rows * columns
  let allTokens := frames * spatialTokens
  let gridShape : Shape := .dim frames (.dim rows (.dim columns (.dim hiddenDim .scalar)))
  let spatialShape : Shape := .dim frames (.dim spatialTokens (.dim hiddenDim .scalar))
  let rowShape : Shape := .dim allTokens (.dim hiddenDim .scalar)
  have hGridSpatial : Shape.size gridShape = Shape.size spatialShape := by
    simp [gridShape, spatialShape, spatialTokens, Shape.size, Nat.mul_assoc]
  have hSpatialRows : Shape.size spatialShape = Shape.size rowShape := by
    simp [spatialShape, rowShape, allTokens, Shape.size, Nat.mul_assoc]
  let spatialInput := Spec.Tensor.reshapeSpec grid hGridSpatial
  let spatialRows := Spec.Tensor.reshapeSpec spatialInput hSpatialRows
  let normalizedRows := Spec.rmsNorm spatialRows normScale
    (Nat.mul_pos hFrames hSpatial) hHidden
  let normalized := Spec.Tensor.reshapeSpec normalizedRows hSpatialRows.symm
  Spec.Tensor.addSpec spatialInput (attendBatch attention normalized hSpatial)

/-- Normalize and attend along the frame axis at every spatial location. -/
def temporalPass
    (attention : Spec.MultiHeadAttention α heads hiddenDim qkvHeadDim)
    (normScale : Tensor α (.dim hiddenDim .scalar))
    (spatial : Tensor α (.dim frames (.dim (rows * columns) (.dim hiddenDim .scalar))))
    (hFrames : 0 < frames) (hSpatial : 0 < rows * columns) (hHidden : 0 < hiddenDim) :
    Tensor α (.dim frames (.dim (rows * columns) (.dim hiddenDim .scalar))) :=
  let spatialTokens := rows * columns
  let temporalShape : Shape := .dim spatialTokens (.dim frames (.dim hiddenDim .scalar))
  let temporalRowShape : Shape := .dim (spatialTokens * frames) (.dim hiddenDim .scalar)
  let temporalInput : Tensor α temporalShape := Spec.Tensor.swapAdjacentAxes spatial 0
  have hTemporalRows : Shape.size temporalShape = Shape.size temporalRowShape := by
    simp [temporalShape, temporalRowShape, Shape.size, Nat.mul_assoc]
  let temporalRows := Spec.Tensor.reshapeSpec temporalInput hTemporalRows
  let normalizedRows := Spec.rmsNorm temporalRows normScale
    (Nat.mul_pos hSpatial hFrames) hHidden
  let normalized := Spec.Tensor.reshapeSpec normalizedRows hTemporalRows.symm
  let attended := attendBatch attention normalized hFrames
  let residual := Spec.Tensor.addSpec temporalInput attended
  Spec.Tensor.swapAdjacentAxes residual 0

/-- Apply the normalized residual MLP after divided attention. -/
def feedForwardPass
    (feedForward : MLP α hiddenDim intermediateDim hiddenDim)
    (normScale : Tensor α (.dim hiddenDim .scalar))
    (spatial : Tensor α (.dim frames (.dim (rows * columns) (.dim hiddenDim .scalar))))
    (hFrames : 0 < frames) (hSpatial : 0 < rows * columns) (hHidden : 0 < hiddenDim) :
    Tensor α (.dim frames (.dim (rows * columns) (.dim hiddenDim .scalar))) :=
  let spatialTokens := rows * columns
  let allTokens := frames * spatialTokens
  let spatialShape : Shape := .dim frames (.dim spatialTokens (.dim hiddenDim .scalar))
  let rowShape : Shape := .dim allTokens (.dim hiddenDim .scalar)
  have hSpatialRows : Shape.size spatialShape = Shape.size rowShape := by
    simp [spatialShape, rowShape, allTokens, Shape.size, Nat.mul_assoc]
  let rows := Spec.Tensor.reshapeSpec spatial hSpatialRows
  let normalized := Spec.rmsNorm rows normScale (Nat.mul_pos hFrames hSpatial) hHidden
  let hidden := Spec.matMulSpec normalized feedForward.inputWeight
  let activated := Activation.geluSpec hidden
  let delta := Spec.matMulSpec activated feedForward.outputWeight
  Spec.Tensor.reshapeSpec (Spec.Tensor.addSpec rows delta) hSpatialRows.symm

/--
Spatial attention, temporal attention, and a bias-free residual MLP.

The first reshape presents each frame as a sequence of spatial tokens. Swapping the frame and
spatial axes then presents each spatial location as a sequence over time. These are views of the
same row-major tensor; the explicit swap is the only permutation of values.
-/
def forward
    (block : Block α heads hiddenDim qkvHeadDim intermediateDim)
    (grid : Grid α frames rows columns hiddenDim)
    (hFrames : 0 < frames) (hSpatial : 0 < rows * columns) (hHidden : 0 < hiddenDim) :
    Grid α frames rows columns hiddenDim :=
  let spatialTokens := rows * columns
  let gridShape : Shape := .dim frames (.dim rows (.dim columns (.dim hiddenDim .scalar)))
  let spatialShape : Shape := .dim frames (.dim spatialTokens (.dim hiddenDim .scalar))
  have hGridSpatial : Shape.size gridShape = Shape.size spatialShape := by
    simp [gridShape, spatialShape, spatialTokens, Shape.size, Nat.mul_assoc]
  let spatial := spatialPass block.spatialAttention block.spatialNormScale grid
    hFrames hSpatial hHidden
  let temporal := temporalPass block.temporalAttention block.temporalNormScale spatial
    hFrames hSpatial hHidden
  let output := feedForwardPass block.feedForward block.feedForwardNormScale temporal
    hFrames hSpatial hHidden
  Spec.Tensor.reshapeSpec output hGridSpatial.symm

end Block

/-- Lightweight MLP that maps merged MoonViT features into the text hidden width. -/
structure Projector (α : Type) (mergedDim textDim : Nat) where
  firstWeight : Tensor α (.dim mergedDim (.dim mergedDim .scalar))
  secondWeight : Tensor α (.dim mergedDim (.dim textDim .scalar))
  outputNormScale : Tensor α (.dim textDim .scalar)

namespace Projector

variable {α : Type} [Context α]
variable {mergedDim textDim : Nat}

def forward (projector : Projector α mergedDim textDim)
    (x : Tensor α (.dim mergedDim .scalar)) : Tensor α (.dim textDim .scalar) :=
  let hidden := Tensor.mapSpec Activation.Math.geluSpec (vecMatMulSpec x projector.firstWeight)
  let projected := vecMatMulSpec hidden projector.secondWeight
  RMSNorm.scale projected projector.outputNormScale

end Projector

/-- Parameters for MoonViT-V2 at arbitrary patch-grid dimensions. -/
structure Model (α : Type) (cfg : VisionConfig)
    (frames rows columns patchFeatures : Nat) where
  patchWeight : Tensor α (.dim patchFeatures (.dim cfg.hiddenDim .scalar))
  spatialPosition : Tensor α (.dim rows (.dim columns (.dim cfg.hiddenDim .scalar)))
  temporalPosition : Tensor α (.dim frames (.dim cfg.hiddenDim .scalar))
  blocks : List (Block α cfg.numHeads cfg.hiddenDim
    (cfg.qkvHiddenDim / cfg.numHeads) cfg.intermediateDim)
  blocks_length : blocks.length = cfg.numLayers
  projector : Projector α (cfg.mergeHeight * cfg.mergeWidth * cfg.hiddenDim) cfg.textHiddenDim

namespace Model

variable {α : Type} [Context α]
variable {cfg : VisionConfig}
variable {frames rows columns patchFeatures : Nat}

/-- Patch projection plus divided spatial and temporal position embeddings. -/
def embed (model : Model α cfg frames rows columns patchFeatures)
    (patches : Grid α frames rows columns patchFeatures) :
    Grid α frames rows columns cfg.hiddenDim :=
  let patchCount := frames * rows * columns
  let patchShape : Shape := .dim frames (.dim rows (.dim columns (.dim patchFeatures .scalar)))
  let patchRows : Shape := .dim patchCount (.dim patchFeatures .scalar)
  let hiddenRows : Shape := .dim patchCount (.dim cfg.hiddenDim .scalar)
  let gridShape : Shape := .dim frames (.dim rows (.dim columns (.dim cfg.hiddenDim .scalar)))
  have hPatchRows : Shape.size patchShape = Shape.size patchRows := by
    simp [patchShape, patchRows, patchCount, Shape.size, Nat.mul_assoc]
  have hHiddenRows : Shape.size hiddenRows = Shape.size gridShape := by
    simp [hiddenRows, gridShape, patchCount, Shape.size, Nat.mul_assoc]
  let patchMatrix := Spec.Tensor.reshapeSpec patches hPatchRows
  let projectedRows := Spec.matMulSpec patchMatrix model.patchWeight
  let projected := Spec.Tensor.reshapeSpec projectedRows hHiddenRows
  let spatial := Spec.Tensor.broadcastTo
    (Shape.CanBroadcastTo.expand_dims (Shape.CanBroadcastTo.refl _)) model.spatialPosition
  let temporalSource : Tensor α
      (.dim frames (.dim 1 (.dim 1 (.dim cfg.hiddenDim .scalar)))) :=
    Spec.Tensor.reshapeSpec model.temporalPosition (by simp [Shape.size])
  let temporal := Spec.Tensor.broadcastTo
    (Shape.CanBroadcastTo.dim_eq <|
      Shape.CanBroadcastTo.dim_1_to_n <|
        Shape.CanBroadcastTo.dim_1_to_n (Shape.CanBroadcastTo.refl _))
    temporalSource
  Spec.Tensor.addSpec (Spec.Tensor.addSpec projected spatial) temporal

/-- Apply all 27 paper-sized blocks, or the configured number in a smaller instance. -/
def encode (model : Model α cfg frames rows columns patchFeatures)
    (grid : Grid α frames rows columns cfg.hiddenDim)
    (hFrames : 0 < frames) (hSpatial : 0 < rows * columns) (hHidden : 0 < cfg.hiddenDim) :
    Grid α frames rows columns cfg.hiddenDim :=
  model.blocks.foldl (fun hidden block => block.forward hidden hFrames hSpatial hHidden) grid

/--
Temporal pooling, pixel-shuffle merge, and projection into the language width. The output sequence
has one token for each merged spatial location; the frame axis has been pooled away.
-/
def mergeAndProject
    (model : Model α cfg frames (rows * cfg.mergeHeight)
      (columns * cfg.mergeWidth) patchFeatures)
    (grid : Grid α frames (rows * cfg.mergeHeight) (columns * cfg.mergeWidth) cfg.hiddenDim)
    (hFrames : 0 < frames)
    (hSpatial : 0 < (rows * cfg.mergeHeight) * (columns * cfg.mergeWidth))
    (hText : 0 < cfg.textHiddenDim) :
    Tensor α (.dim (rows * columns) (.dim cfg.textHiddenDim .scalar)) :=
  have hWideRows : 0 < rows * cfg.mergeHeight := Nat.pos_of_mul_pos_right hSpatial
  have hWideColumns : 0 < columns * cfg.mergeWidth := Nat.pos_of_mul_pos_left hSpatial
  have hRows : 0 < rows := Nat.pos_of_mul_pos_right hWideRows
  have hColumns : 0 < columns := Nat.pos_of_mul_pos_right hWideColumns
  let pooled : Tensor α (.dim (rows * cfg.mergeHeight)
      (.dim (columns * cfg.mergeWidth) (.dim cfg.hiddenDim .scalar))) :=
    Spec.Tensor.reduceMean 0 grid (Shape.hasNonemptyAxisZeroOfPos hFrames).proof
  let interleavedShape : Shape :=
    .dim rows (.dim cfg.mergeHeight
      (.dim columns (.dim cfg.mergeWidth (.dim cfg.hiddenDim .scalar))))
  let groupedShape : Shape :=
    .dim rows (.dim columns
      (.dim cfg.mergeHeight (.dim cfg.mergeWidth (.dim cfg.hiddenDim .scalar))))
  let mergedDim := cfg.mergeHeight * cfg.mergeWidth * cfg.hiddenDim
  let mergedShape : Shape := .dim (rows * columns) (.dim mergedDim .scalar)
  have hInterleaved :
      Shape.size (.dim (rows * cfg.mergeHeight)
        (.dim (columns * cfg.mergeWidth) (.dim cfg.hiddenDim .scalar))) =
        Shape.size interleavedShape := by
    simp [interleavedShape, Shape.size, Nat.mul_assoc]
  have hMerged : Shape.size groupedShape = Shape.size mergedShape := by
    simp [groupedShape, mergedShape, mergedDim, Shape.size, Nat.mul_assoc]
  let interleaved := Spec.Tensor.reshapeSpec pooled hInterleaved
  have hGrouped : interleavedShape.swapAdjacentAtDepth 1 = groupedShape := by
    simp [interleavedShape, groupedShape, Shape.swapAdjacentAtDepth]
  let grouped : Tensor α groupedShape :=
    hGrouped ▸ Spec.Tensor.swapAdjacentAxes interleaved 1
  let merged := Spec.Tensor.reshapeSpec grouped hMerged
  let hidden := Activation.geluSpec (Spec.matMulSpec merged model.projector.firstWeight)
  let projected := Spec.matMulSpec hidden model.projector.secondWeight
  Spec.rmsNorm projected model.projector.outputNormScale
    (Nat.mul_pos hRows hColumns) hText

/-- Full visual path from flattened patches to language-width visual tokens. -/
def forward
    (model : Model α cfg frames (rows * cfg.mergeHeight)
      (columns * cfg.mergeWidth) patchFeatures)
    (patches : Grid α frames (rows * cfg.mergeHeight)
      (columns * cfg.mergeWidth) patchFeatures)
    (hFrames : 0 < frames)
    (hSpatial : 0 < (rows * cfg.mergeHeight) * (columns * cfg.mergeWidth))
    (hHidden : 0 < cfg.hiddenDim) (hText : 0 < cfg.textHiddenDim) :
    Tensor α (.dim (rows * columns) (.dim cfg.textHiddenDim .scalar)) :=
  model.mergeAndProject (model.encode (model.embed patches) hFrames hSpatial hHidden)
    hFrames hSpatial hText

end Model

/-- Spatial pixel shuffle preserves scalar count after the temporal axis has been pooled. -/
theorem spatial_merge_preserves_scalar_count
    (rows columns mergeHeight mergeWidth hiddenDim : Nat) :
    (rows * mergeHeight) * (columns * mergeWidth) * hiddenDim =
      rows * columns * (mergeHeight * mergeWidth * hiddenDim) := by
  ring

end MoonViT

end KimiK3
