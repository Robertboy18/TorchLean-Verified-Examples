/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Sequence
public import NN.Spec.Models.Transformer
public import NN.Tensor

/-!
# MoonViT-V2 specification

Kimi K3 uses one vision encoder for images and videos.  MoonViT-V2 applies shared, bias-free
parameters to both modalities, factorizes attention into spatial and temporal passes, merges each
`2 × 2` group of spatial tokens, and projects the result into the language-model width.

This file describes that dataflow for arbitrary frame and patch-grid dimensions.  The input is an
already extracted grid of flattened patches; pixel decoding, resizing, and patch extraction are
data-pipeline concerns and are intentionally outside the model equation.

Patch projection and the residual MLP use `einsum` so their frame and patch axes remain visible.
`rearrange` exchanges the frame and spatial-token axes for temporal attention; `mapLeading`
applies normalization and attention independently across the corresponding batches.

Reference: Kimi Team, "Kimi K3: Open Frontier Intelligence", 2026, Section 2.4, pp. 9--10,
https://arxiv.org/abs/2607.24653. Exact released dimensions are recorded in `KimiK3.paperConfig`.
-/

@[expose] public section

namespace KimiK3

open Spec TorchLean
open Tensor

namespace MoonViT

/-- A video/image patch grid with axes `(frame, row, column, feature)`. -/
abbrev Grid (α : Type) [Storage α] (frames rows columns features : Nat) :=
  Tensor α (.dim frames (.dim rows (.dim columns (.dim features .scalar))))

/-- Bias-free MLP used in MoonViT-V2 blocks. -/
structure MLP (α : Type) [Storage α] (inputDim hiddenDim outputDim : Nat) where
  /-- Input-to-hidden projection. -/
  inputWeight : Tensor α (.dim inputDim (.dim hiddenDim .scalar))
  /-- Hidden-to-output projection. -/
  outputWeight : Tensor α (.dim hiddenDim (.dim outputDim .scalar))

/-- One divided-attention MoonViT-V2 block. -/
structure Block (α : Type) [Storage α] (heads hiddenDim qkvHeadDim intermediateDim : Nat) where
  /-- Attention shared across frame-local spatial sequences. -/
  spatialAttention : Spec.MultiHeadAttention α heads hiddenDim qkvHeadDim
  /-- Attention shared across spatial-location temporal sequences. -/
  temporalAttention : Spec.MultiHeadAttention α heads hiddenDim qkvHeadDim
  /-- Bias-free GELU feed-forward parameters. -/
  feedForward : MLP α hiddenDim intermediateDim hiddenDim
  /-- RMSNorm scale before spatial attention. -/
  spatialNormScale : Tensor α (.dim hiddenDim .scalar)
  /-- RMSNorm scale before temporal attention. -/
  temporalNormScale : Tensor α (.dim hiddenDim .scalar)
  /-- RMSNorm scale before the feed-forward branch. -/
  feedForwardNormScale : Tensor α (.dim hiddenDim .scalar)

namespace Block

variable {α : Type} [Storage α] [Context α]
variable {heads hiddenDim qkvHeadDim intermediateDim frames rows columns : Nat}

/-- Apply one shared attention module independently along a leading batch axis. -/
def attendBatch {batch tokens : Nat}
    (attention : Spec.MultiHeadAttention α heads hiddenDim qkvHeadDim)
    (input : Tensor α (.dim batch (.dim tokens (.dim hiddenDim .scalar))))
    (hTokens : 0 < tokens) :
    Tensor α (.dim batch (.dim tokens (.dim hiddenDim .scalar))) :=
  Tensor.mapLeading [batch]
    (fun sequence => attention.forward tokens (Nat.ne_of_gt hTokens) sequence none) input

/-- Batched attention applies the same TorchLean attention module to every batch row. -/
@[simp] theorem get_attendBatch {batch tokens : Nat}
    (attention : Spec.MultiHeadAttention α heads hiddenDim qkvHeadDim)
    (input : Tensor α [batch, tokens, hiddenDim]) (hTokens : 0 < tokens)
    (index : Fin batch) :
    Spec.get (attendBatch attention input hTokens) index =
      attention.forward tokens (Nat.ne_of_gt hTokens) (Spec.get input index) none := by
  simp [attendBatch, Tensor.mapLeading, Spec.get]

/-- Normalize and attend independently within every frame. -/
def spatialPass
    (attention : Spec.MultiHeadAttention α heads hiddenDim qkvHeadDim)
    (normScale : Tensor α (.dim hiddenDim .scalar))
    (grid : Grid α frames rows columns hiddenDim)
    (hSpatial : 0 < rows * columns) (hHidden : 0 < hiddenDim) :
    Tensor α (.dim frames (.dim (rows * columns) (.dim hiddenDim .scalar))) :=
  let spatialTokens := rows * columns
  let gridShape : Shape := .dim frames (.dim rows (.dim columns (.dim hiddenDim .scalar)))
  let spatialShape : Shape := .dim frames (.dim spatialTokens (.dim hiddenDim .scalar))
  have hGridSpatial : Shape.size gridShape = Shape.size spatialShape := by
    simp [gridShape, spatialShape, spatialTokens, Shape.size, Nat.mul_assoc]
  let spatialInput := Tensor.reshapeSpec grid hGridSpatial
  let normalized := Tensor.mapLeading [frames]
    (fun frame => RMSNorm.rows hHidden frame normScale) spatialInput
  Tensor.addSpec spatialInput (attendBatch attention normalized hSpatial)

/-- Normalize and attend along the frame axis at every spatial location. -/
def temporalPass
    (attention : Spec.MultiHeadAttention α heads hiddenDim qkvHeadDim)
    (normScale : Tensor α (.dim hiddenDim .scalar))
    (spatial : Tensor α (.dim frames (.dim (rows * columns) (.dim hiddenDim .scalar))))
    (hFrames : 0 < frames) (hHidden : 0 < hiddenDim) :
    Tensor α (.dim frames (.dim (rows * columns) (.dim hiddenDim .scalar))) :=
  let spatialTokens := rows * columns
  let temporalShape : Shape := .dim spatialTokens (.dim frames (.dim hiddenDim .scalar))
  let temporalInput : Tensor α temporalShape :=
    rearrange spatial "frame patch feature -> patch frame feature"
  let normalized := Tensor.mapLeading [spatialTokens]
    (fun patch => RMSNorm.rows hHidden patch normScale) temporalInput
  let attended := attendBatch attention normalized hFrames
  let residual := Tensor.addSpec temporalInput attended
  rearrange residual "patch frame feature -> frame patch feature"

/-- Apply the normalized residual MLP after divided attention. -/
def feedForwardPass
    (feedForward : MLP α hiddenDim intermediateDim hiddenDim)
    (normScale : Tensor α (.dim hiddenDim .scalar))
    (spatial : Tensor α (.dim frames (.dim (rows * columns) (.dim hiddenDim .scalar))))
    (hHidden : 0 < hiddenDim) :
    Tensor α (.dim frames (.dim (rows * columns) (.dim hiddenDim .scalar))) :=
  let normalized := Tensor.mapLeading [frames, rows * columns]
    (fun token => RMSNorm.scalePositive hHidden token normScale) spatial
  let hidden := einsum normalized, feedForward.inputWeight
    "frame patch input, input output -> frame patch output"
  let activated := Activation.geluSpec hidden
  let delta := einsum activated, feedForward.outputWeight
    "frame patch input, input output -> frame patch output"
  Tensor.addSpec spatial delta

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
    hSpatial hHidden
  let temporal := temporalPass block.temporalAttention block.temporalNormScale spatial
    hFrames hHidden
  let output := feedForwardPass block.feedForward block.feedForwardNormScale temporal
    hHidden
  Tensor.reshapeSpec output hGridSpatial.symm

end Block

/-- Lightweight MLP that maps merged MoonViT features into the text hidden width. -/
structure Projector (α : Type) [Storage α] (mergedDim textDim : Nat) where
  /-- Square hidden projection after spatial merging. -/
  firstWeight : Tensor α (.dim mergedDim (.dim mergedDim .scalar))
  /-- Projection into the language-model hidden width. -/
  secondWeight : Tensor α (.dim mergedDim (.dim textDim .scalar))
  /-- Final language-width RMSNorm scale. -/
  outputNormScale : Tensor α (.dim textDim .scalar)

/-- Parameters for MoonViT-V2 at arbitrary patch-grid dimensions. -/
structure Model (α : Type) [Storage α] (cfg : VisionConfig)
    (frames rows columns patchFeatures : Nat) where
  /-- Linear projection from flattened patches to MoonViT width. -/
  patchWeight : Tensor α (.dim patchFeatures (.dim cfg.hiddenDim .scalar))
  /-- Learned row-column positions, shared across frames. -/
  spatialPosition : Tensor α (.dim rows (.dim columns (.dim cfg.hiddenDim .scalar)))
  /-- Learned frame positions, shared across spatial locations. -/
  temporalPosition : Tensor α (.dim frames (.dim cfg.hiddenDim .scalar))
  /-- Ordered divided-attention blocks. -/
  blocks : List (Block α cfg.numHeads cfg.hiddenDim
    (cfg.qkvHiddenDim / cfg.numHeads) cfg.intermediateDim)
  /-- The parameter list contains exactly the configured number of blocks. -/
  blocks_length : blocks.length = cfg.numLayers
  /-- Temporal-pooling and spatial-merging projector into text width. -/
  projector : Projector α (cfg.mergeHeight * cfg.mergeWidth * cfg.hiddenDim) cfg.textHiddenDim

namespace Model

variable {α : Type} [Storage α] [Context α]
variable {cfg : VisionConfig}
variable {frames rows columns patchFeatures : Nat}

/-- Patch projection plus divided spatial and temporal position embeddings. -/
def embed (model : Model α cfg frames rows columns patchFeatures)
    (patches : Grid α frames rows columns patchFeatures) :
    Grid α frames rows columns cfg.hiddenDim :=
  let projected := einsum patches, model.patchWeight
    "frame row column pixel, pixel feature -> frame row column feature"
  let spatial := Tensor.broadcastTo
    (Shape.CanBroadcastTo.expand_dims (Shape.CanBroadcastTo.refl _)) model.spatialPosition
  let temporalSource : Tensor α
      (.dim frames (.dim 1 (.dim 1 (.dim cfg.hiddenDim .scalar)))) :=
    Tensor.reshapeSpec model.temporalPosition (by simp [Shape.size])
  let temporal := Tensor.broadcastTo
    (Shape.CanBroadcastTo.dim_eq <|
      Shape.CanBroadcastTo.dim_1_to_n <|
        Shape.CanBroadcastTo.dim_1_to_n (Shape.CanBroadcastTo.refl _))
    temporalSource
  Tensor.addSpec (Tensor.addSpec projected spatial) temporal

/-- Apply all 27 paper-sized blocks, or the configured number in a smaller instance. -/
def encode (model : Model α cfg frames rows columns patchFeatures)
    (grid : Grid α frames rows columns cfg.hiddenDim)
    (hFrames : 0 < frames) (hSpatial : 0 < rows * columns) (hHidden : 0 < cfg.hiddenDim) :
    Grid α frames rows columns cfg.hiddenDim :=
  model.blocks.foldl (fun hidden block => block.forward hidden hFrames hSpatial hHidden) grid

/-- A zero-layer vision configuration leaves the embedded grid unchanged. -/
theorem encode_of_numLayers_eq_zero
    (model : Model α cfg frames rows columns patchFeatures)
    (grid : Grid α frames rows columns cfg.hiddenDim)
    (hFrames : 0 < frames) (hSpatial : 0 < rows * columns) (hHidden : 0 < cfg.hiddenDim)
    (hLayers : cfg.numLayers = 0) :
    model.encode grid hFrames hSpatial hHidden = grid := by
  have hLength : model.blocks.length = 0 := by simpa [hLayers] using model.blocks_length
  have hBlocks : model.blocks = [] := List.length_eq_zero_iff.mp hLength
  simp [encode, hBlocks]

omit [Context α] in
/-- A positive configured depth forces the model to contain at least one MoonViT block. -/
theorem blocks_ne_nil (model : Model α cfg frames rows columns patchFeatures)
    (hLayers : 0 < cfg.numLayers) : model.blocks ≠ [] := by
  intro hBlocks
  have hLength := model.blocks_length
  simp [hBlocks] at hLength
  omega

/--
Temporal pooling, pixel-shuffle merge, and projection into the language width. The output sequence
has one token for each merged spatial location; the frame axis has been pooled away.
-/
def mergeAndProject
    (model : Model α cfg frames (rows * cfg.mergeHeight)
      (columns * cfg.mergeWidth) patchFeatures)
    (grid : Grid α frames (rows * cfg.mergeHeight) (columns * cfg.mergeWidth) cfg.hiddenDim)
    (hFrames : 0 < frames)
    (hText : 0 < cfg.textHiddenDim) :
    Tensor α (.dim (rows * columns) (.dim cfg.textHiddenDim .scalar)) :=
  let pooled : Tensor α (.dim (rows * cfg.mergeHeight)
      (.dim (columns * cfg.mergeWidth) (.dim cfg.hiddenDim .scalar))) :=
    Tensor.reduceMean 0 grid (Shape.hasNonemptyAxisZeroOfPos hFrames).proof
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
  let interleaved := Tensor.reshapeSpec pooled hInterleaved
  have hGrouped : interleavedShape.swapAdjacentAtDepth 1 = groupedShape := by
    simp [interleavedShape, groupedShape, Shape.swapAdjacentAtDepth]
  let grouped : Tensor α groupedShape :=
    hGrouped ▸ Tensor.swapAdjacentAxes interleaved 1
  let merged := Tensor.reshapeSpec grouped hMerged
  let hidden := Activation.geluSpec (Tensor.matmulSpec (Shape.CanBroadcastTo.refl .scalar)
    (Shape.CanBroadcastTo.refl .scalar) merged model.projector.firstWeight)
  let projected := Tensor.matmulSpec (Shape.CanBroadcastTo.refl .scalar)
    (Shape.CanBroadcastTo.refl .scalar) hidden model.projector.secondWeight
  RMSNorm.rows hText projected model.projector.outputNormScale

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
    hFrames hText

end Model

end MoonViT

end KimiK3
