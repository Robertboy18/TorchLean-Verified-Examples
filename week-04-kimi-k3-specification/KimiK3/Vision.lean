/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Common
public import KimiK3.Config
public import NN.Spec.Models.Transformer
public import NN.Tensor
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import NN.Proofs.Tensor.Basic.Folds
import NN.Proofs.Tensor.Basic.LinearAlgebra

/-!
# MoonViT-V2 specification

Kimi K3 uses one vision encoder for images and videos. This file follows the released
`modeling_kimi_k3.py` (Hugging Face `moonshotai/Kimi-K3`, revision `f831ab6`) rather than the
report's prose. Section 2.4 of the report says attention is factorized into intra-frame spatial and
inter-frame temporal passes. The released `MoonViTEncoderLayer` has one attention module, and it
attends jointly over all `t * h * w` patch tokens of a media item. The code is what runs, so the
code is what we specify.

One media item is a grid of `t × h × w` flattened patches with `t ≤ 4`. The released tower then
1. projects each patch with a bias-free `14 × 14` stride-`14` convolution, i.e. a linear map;
2. adds a learned `64 × 64` position table, bilinearly resized to `h × w`, and, when `t > 1`, a
   fixed sine-cosine frame embedding (`Learnable2DInterpPosEmbDivided_fixed`);
3. runs 27 pre-norm blocks. Each block applies RMSNorm, one packed `wqkv` projection, 2-D rotary
   position embedding on queries and keys (`Rope2DPosEmbRepeated`), non-causal scaled dot-product
   attention over the whole item, the output projection `wo`, and a residual; then RMSNorm, a
   bias-free MLP with the tanh GELU, and a residual (`MoonViTEncoderLayer`);
4. applies a final RMSNorm (`MoonViT3dEncoder.final_layernorm`);
5. averages the `t` frames and groups each `2 × 2` patch block into one token (`tpool_patch_merger`);
6. projects with two bias-free linear maps around the exact erf GELU, then RMSNorm with
   `ε = 10⁻⁵` (`PatchMergerMLPV2`).

Different media items never attend to each other (`cu_seqlens`), so a video longer than four frames
is a list of separate items, each pooled on its own. `forwardChunks` states this for a video cut
into equal chunks.

The input is an already extracted grid of flattened patches. Pixel decoding, resizing and patch
extraction belong to the data pipeline, outside the model equation.

Reference: Kimi Team, "Kimi K3: Open Frontier Intelligence", 2026, Section 2.4, pp. 9--10,
https://arxiv.org/abs/2607.24653, and the released code above. Dimensions are recorded in
`KimiK3.paperConfig`.
-/

@[expose] public section

namespace KimiK3

open Spec TorchLean
open Tensor

namespace MoonViT

/-! ## Constants of the released vision tower -/

/-- Rows of the learned position table (`init_pos_emb_height`). -/
def positionRows : Nat := 64

/-- Columns of the learned position table (`init_pos_emb_width`). -/
def positionColumns : Nat := 64

/-- Frames per media item (`init_pos_emb_time`). The position embedding asserts `t ≤ 4`. -/
def maxFrames : Nat := 4

/-- Base of the sine-cosine frame embedding and of the 2-D rotary embedding. -/
def positionBase : Nat := 10000

/-- `nn.RMSNorm(hidden_dim)` inside the encoder leaves `eps` unset. PyTorch then uses the float32
machine epsilon `2⁻²³` for float32, bfloat16 and float16 inputs (`aten/src/ATen/native/
layer_norm.cpp`, `rms_norm_composite`, and the fused CUDA kernel in `layer_norm_kernel.cu`). -/
def encoderNormEpsilon {α : Type} [RatCast α] : α :=
  Rat.cast (1 / 8388608 : Rat)

/-! ## Exact GELU -/

/-- A scalar error function.

PyTorch's default `nn.GELU()` is `x * (1 + erf (x / √2)) / 2`. TorchLean's `Context` has no `erf`,
so code that needs the exact GELU asks for this capability separately. -/
class Erf (α : Type) where
  /-- The error function. -/
  erf : α → α

/-- Real error function `erf x = 2 / √π * ∫₀ˣ exp (-t²) dt`. -/
noncomputable def realErf (x : ℝ) : ℝ :=
  2 / Real.sqrt Real.pi * ∫ t in (0 : ℝ)..x, Real.exp (-(t ^ 2))

noncomputable instance : Erf ℝ := ⟨realErf⟩

/-- `erf 0 = 0`. -/
@[simp] theorem realErf_zero : realErf 0 = 0 := by
  simp [realErf]

/-- PyTorch's default GELU, `nn.GELU()` with `approximate='none'`. -/
def geluExact {α : Type} [Context α] [Erf α] (x : α) : α :=
  x * ((1 + Erf.erf (x / MathFunctions.sqrt ((1 : α) + 1))) / ((1 : α) + 1))

/-- Coordinatewise exact GELU. -/
def geluExactSpec {α : Type} [Storage α] [Context α] [Erf α] {s : Shape} (t : Tensor α s) :
    Tensor α s :=
  Tensor.map geluExact t

/-- The exact real GELU vanishes at zero. -/
@[simp] theorem geluExact_zero : geluExact (0 : ℝ) = 0 := by
  simp [geluExact]

/-! ## Patch grids, position tables, and rotary tables -/

/-- A video/image patch grid with axes `(frame, row, column, feature)`. -/
abbrev Grid (α : Type) [Storage α] (frames rows columns features : Nat) :=
  Tensor α [frames, rows, columns, features]

section Tables

variable {α : Type} [Storage α] [Context α]

/-- Weight of source coordinate `a` in output coordinate `i` for PyTorch's
`F.interpolate(mode="bilinear", align_corners=False)` along one axis of length `source`, resized to
`target`.

PyTorch reads the source coordinate `(i + 1/2) * source / target - 1/2`, clamped below at `0`,
interpolates between its floor and the next index, and repeats the last index at the border.
Here the source coordinate is `numerator / denominator`, and natural subtraction performs the
clamp. -/
def interpolationWeight (source target i a : Nat) : α :=
  let numerator := (2 * i + 1) * source - target
  let denominator := 2 * target
  let lower := numerator / denominator
  let upper := if lower + 1 < source then lower + 1 else lower
  let fraction : α := ((numerator % denominator : Nat) : α) / ((denominator : Nat) : α)
  (if a = lower then 1 - fraction else 0) + (if a = upper then fraction else 0)

/-- One-axis bilinear resize matrix, from `source` positions to `target` positions. -/
def interpolationMatrix (source target : Nat) : Tensor α [target, source] :=
  Tensor.dim fun i => Tensor.dim fun a => Tensor.scalar (interpolationWeight source target i a)

/-- Bilinear resize of a `sourceRows × sourceColumns` table to `rows × columns`.

Bilinear interpolation is separable: the row matrix acts on the first axis and the column matrix on
the second. This is what `get_rope_shape` computes with `F.interpolate` after moving the feature axis
to the front. -/
def resizeTable {sourceRows sourceColumns rows columns width : Nat}
    (rowWeights : Tensor α [rows, sourceRows])
    (columnWeights : Tensor α [columns, sourceColumns])
    (table : Tensor α [sourceRows, sourceColumns, width]) :
    Tensor α [rows, columns, width] :=
  have hFlat : Shape.size [sourceRows, sourceColumns, width] =
      Shape.size [sourceRows, sourceColumns * width] := by
    simp [Shape.size]
  have hUnflat : Shape.size [rows, sourceColumns * width] =
      Shape.size [rows, sourceColumns, width] := by
    simp [Shape.size]
  let alongRows : Tensor α [rows, sourceColumns, width] :=
    Tensor.reshapeSpec (matMulSpec rowWeights (Tensor.reshapeSpec table hFlat)) hUnflat
  Tensor.dim fun row => matMulSpec columnWeights (Tensor.unstack alongRows row)

/-- Frequency `1 / base ^ (index / half)` of the sine-cosine frame embedding. -/
def frameFrequency (width index : Nat) : α :=
  1 / (positionBase : α) ^ ((index : α) / ((width / 2 : Nat) : α))

/-- The fixed buffer `time_weight`: `get_1d_sincos_pos_embed(width, maxFrames)`.

Frame `f` stores `sin (f * ω_k)` in its first `width / 2` coordinates and `cos (f * ω_k)` in the
rest. -/
def frameTable (width : Nat) : Tensor α [maxFrames, width] :=
  Tensor.dim fun frame => Tensor.dim fun k => Tensor.scalar <|
    if k.val < width / 2 then
      MathFunctions.sin ((frame.val : α) * frameFrequency width k.val)
    else
      MathFunctions.cos ((frame.val : α) * frameFrequency width (k.val - width / 2))

/-- Frame embedding added to a `frames`-frame item: nothing for one frame, otherwise the first
`frames` rows of `frameTable`. The released code takes `time_weight[0:t]` only when `t > 1`. -/
def frameEmbedding (frames width : Nat) (hFrames : frames ≤ maxFrames) :
    Tensor α [frames, width] :=
  if frames = 1 then
    Tensor.full [frames, width] 0
  else
    Tensor.dim fun frame =>
      Tensor.unstack (frameTable width) ⟨frame.val, lt_of_lt_of_le frame.isLt hFrames⟩

/-- Rotary angle of head coordinate `coordinate` at patch position `(row, column)`.

`Rope2DPosEmbRepeated` stores `headDim / 2` complex rotations per position. Rotation `j` acts on
the coordinate pair `(2j, 2j + 1)`. Even rotations follow the column, odd rotations the row, and
both use the frequency `base ^ (-4i / headDim)` with `i = j / 2`. Frames share their rotations
(`repeat(t, 1)`). -/
def ropeAngle (headDim row column coordinate : Nat) : α :=
  let rotation := coordinate / 2
  let position : Nat := if rotation % 2 = 0 then column else row
  (position : α) / (positionBase : α) ^ (((4 * (rotation / 2) : Nat) : α) / (headDim : α))

/-- Per-token rotary table `f (angle)` for a `frames × rows × columns` item, laid out over all
heads. Token `p` sits at row `(p % (rows * columns)) / columns` and column `p % columns`. The
released code tabulates these angles for rows and columns up to `512` and asserts that bound. -/
def ropeTable (f : α → α) (frames rows columns heads headDim : Nat) :
    Tensor α [frames * rows * columns, heads * headDim] :=
  Tensor.dim fun token => Tensor.dim fun k => Tensor.scalar <|
    f (ropeAngle headDim ((token.val % (rows * columns)) / columns) (token.val % columns)
      (k.val % headDim))

/-- Fixed matrix sending each coordinate pair `(x₀, x₁)` to `(-x₁, x₀)`. -/
def pairRotation (width : Nat) : Tensor α [width, width] :=
  Tensor.dim fun m => Tensor.dim fun k => Tensor.scalar <|
    if k.val % 2 = 0 ∧ m.val = k.val + 1 then -1
    else if m.val % 2 = 0 ∧ k.val = m.val + 1 then 1
    else 0

/-- Rotary embedding `x ⊙ cos + (x R) ⊙ sin`.

On every pair this is the complex product `(x₀ + i x₁) * (cos θ + i sin θ)` computed by
`apply_rope` (see `applyRope_even` and `applyRope_odd`). -/
def applyRope {tokens width : Nat} (cosTable sinTable input : Tensor α [tokens, width]) :
    Tensor α [tokens, width] :=
  Tensor.addSpec (Tensor.mulSpec input cosTable)
    (Tensor.mulSpec (matMulSpec input (pairRotation width)) sinTable)

/-- Patch embedding with its resize matrices and frame rows supplied: the projected patches, plus
the resized learned table at every frame, plus the frame row at every patch. -/
def embedPatches {frames rows columns patchFeatures sourceRows sourceColumns hidden : Nat}
    (patchWeight : Tensor α [patchFeatures, hidden])
    (positionWeight : Tensor α [sourceRows, sourceColumns, hidden])
    (rowWeights : Tensor α [rows, sourceRows])
    (columnWeights : Tensor α [columns, sourceColumns])
    (frameRows : Tensor α [frames, hidden])
    (patches : Grid α frames rows columns patchFeatures) :
    Grid α frames rows columns hidden :=
  let projected := einsum patches, patchWeight
    "frame row column pixel, pixel feature -> frame row column feature"
  let spatial := Tensor.broadcastTo
    (Shape.CanBroadcastTo.expand_dims (Shape.CanBroadcastTo.refl _))
    (resizeTable rowWeights columnWeights positionWeight)
  let temporalSource : Tensor α [frames, 1, 1, hidden] :=
    Tensor.reshapeSpec frameRows (by simp [Shape.size])
  let temporal := Tensor.broadcastTo
    (Shape.CanBroadcastTo.dim_eq <|
      Shape.CanBroadcastTo.dim_1_to_n <|
        Shape.CanBroadcastTo.dim_1_to_n (Shape.CanBroadcastTo.refl _))
    temporalSource
  Tensor.addSpec (Tensor.addSpec projected spatial) temporal

/-- RMSNorm of every row with a supplied epsilon. -/
def normRows {rowCount width : Nat} (hWidth : 0 < width) (epsilon : α)
    (input : Tensor α [rowCount, width]) (gamma : Tensor α [width]) :
    Tensor α [rowCount, width] :=
  Tensor.mapLeading [rowCount] (RMSNorm.scalePositive hWidth · gamma epsilon) input

end Tables

/-! ## What the tables compute -/

section TableTheorems

private theorem get2_mulSpec {m n : Nat} (a b : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (Tensor.mulSpec a b) i j = Spec.get2 a i j * Spec.get2 b i j := by
  simp [Tensor.mulSpec]

/-- At an even head coordinate, `x R` reads minus the partner coordinate. -/
theorem get2_matMul_pairRotation_even {tokens width : Nat} (x : Tensor ℝ [tokens, width])
    (token : Fin tokens) (j : Nat) (hj : 2 * j + 1 < width) :
    Spec.get2 (matMulSpec x (pairRotation width)) token ⟨2 * j, by omega⟩ =
      -Spec.get2 x token ⟨2 * j + 1, hj⟩ := by
  rw [Spec.get2_mat_mul_spec, Finset.sum_eq_single ⟨2 * j + 1, hj⟩]
  · simp [pairRotation]
  · intro m _ hm
    have hm' : m.val ≠ 2 * j + 1 := fun h => hm (Fin.ext h)
    simp only [pairRotation, Spec.get2_dim]
    split_ifs with h₁ h₂ <;> first | omega | simp
  · simp

/-- At an odd head coordinate, `x R` reads the partner coordinate. -/
theorem get2_matMul_pairRotation_odd {tokens width : Nat} (x : Tensor ℝ [tokens, width])
    (token : Fin tokens) (j : Nat) (hj : 2 * j + 1 < width) :
    Spec.get2 (matMulSpec x (pairRotation width)) token ⟨2 * j + 1, hj⟩ =
      Spec.get2 x token ⟨2 * j, by omega⟩ := by
  rw [Spec.get2_mat_mul_spec, Finset.sum_eq_single ⟨2 * j, by omega⟩]
  · simp [pairRotation]
  · intro m _ hm
    have hm' : m.val ≠ 2 * j := fun h => hm (Fin.ext h)
    simp only [pairRotation, Spec.get2_dim]
    split_ifs with h₁ h₂ <;> first | omega | simp
  · simp

/-- First coordinate of a rotated pair: the real part of `(x₀ + i x₁)(cos θ + i sin θ)`. -/
theorem applyRope_even {tokens width : Nat} (cosTable sinTable x : Tensor ℝ [tokens, width])
    (token : Fin tokens) (j : Nat) (hj : 2 * j + 1 < width) :
    Spec.get2 (applyRope cosTable sinTable x) token ⟨2 * j, by omega⟩ =
      Spec.get2 x token ⟨2 * j, by omega⟩ * Spec.get2 cosTable token ⟨2 * j, by omega⟩ -
        Spec.get2 x token ⟨2 * j + 1, hj⟩ * Spec.get2 sinTable token ⟨2 * j, by omega⟩ := by
  rw [applyRope, Spec.get2_addSpec, get2_mulSpec, get2_mulSpec,
    get2_matMul_pairRotation_even x token j hj]
  ring

/-- Second coordinate of a rotated pair: the imaginary part of `(x₀ + i x₁)(cos θ + i sin θ)`. -/
theorem applyRope_odd {tokens width : Nat} (cosTable sinTable x : Tensor ℝ [tokens, width])
    (token : Fin tokens) (j : Nat) (hj : 2 * j + 1 < width) :
    Spec.get2 (applyRope cosTable sinTable x) token ⟨2 * j + 1, hj⟩ =
      Spec.get2 x token ⟨2 * j + 1, hj⟩ * Spec.get2 cosTable token ⟨2 * j + 1, hj⟩ +
        Spec.get2 x token ⟨2 * j, by omega⟩ * Spec.get2 sinTable token ⟨2 * j + 1, hj⟩ := by
  rw [applyRope, Spec.get2_addSpec, get2_mulSpec, get2_mulSpec,
    get2_matMul_pairRotation_odd x token j hj]

/-- Both coordinates of a pair inside one head share their rotary angle. -/
theorem ropeAngle_pair (headDim row column j : Nat) (hEven : 2 ∣ headDim) :
    (ropeAngle headDim row column ((2 * j) % headDim) : ℝ) =
      ropeAngle headDim row column ((2 * j + 1) % headDim) := by
  have hPair : (2 * j) % headDim / 2 = (2 * j + 1) % headDim / 2 := by
    obtain ⟨half, rfl⟩ := hEven
    rcases Nat.eq_zero_or_pos half with h | h
    · subst h
      simp only [Nat.mul_zero, Nat.mod_zero]
      omega
    · have hLow : j % half < half := Nat.mod_lt j h
      have hSplit : 2 * j + 1 = (2 * (j % half) + 1) + 2 * half * (j / half) := by
        have := Nat.mod_add_div j half
        rw [Nat.mul_assoc]
        omega
      have hOdd : (2 * j + 1) % (2 * half) = 2 * (j % half) + 1 := by
        rw [hSplit, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt (by omega)]
      rw [Nat.mul_mod_mul_left 2 j half, hOdd]
      omega
  simp only [ropeAngle, hPair]

/-- The patch at row `0`, column `0` is not rotated. -/
@[simp] theorem ropeAngle_origin (headDim coordinate : Nat) :
    (ropeAngle headDim 0 0 coordinate : ℝ) = 0 := by
  simp [ropeAngle]

/-- A single frame receives no frame embedding, as in the released `if t == 1` branch. -/
theorem frameEmbedding_one {α : Type} [Storage α] [Context α] (width : Nat)
    (hFrames : 1 ≤ maxFrames) :
    (frameEmbedding 1 width hFrames : Tensor α [1, width]) = Tensor.full [1, width] 0 := by
  simp [frameEmbedding]

/-- Resizing an axis to its own length is the identity. This is why the released code may skip
`F.interpolate` when the patch grid is already `64 × 64`. -/
theorem interpolationWeight_self (n i a : Nat) (hi : i < n) :
    (interpolationWeight n n i a : ℝ) = if a = i then 1 else 0 := by
  have hn : 0 < 2 * n := by omega
  have hNumerator : (2 * i + 1) * n - n = 2 * n * i := by
    rw [Nat.add_mul, one_mul, Nat.add_sub_cancel]
    ring
  have hLower : 2 * n * i / (2 * n) = i := Nat.mul_div_cancel_left i hn
  have hRemainder : 2 * n * i % (2 * n) = 0 := Nat.mul_mod_right (2 * n) i
  simp only [interpolationWeight, hNumerator, hLower, hRemainder]
  split_ifs <;> simp_all

/-- Entries of the resize matrix for an axis kept at its own length. -/
theorem get2_interpolationMatrix_self (n : Nat) (i a : Fin n) :
    Spec.get2 (interpolationMatrix n n : Tensor ℝ [n, n]) i a = if a = i then 1 else 0 := by
  simp only [interpolationMatrix, Spec.get2_dim]
  rw [interpolationWeight_self n i a i.isLt]
  simp [Fin.ext_iff]

end TableTheorems

/-! ## Encoder blocks -/

/-- Bias-free two-layer MLP (`MLP2` with `linear_bias = false`). -/
structure MLP (α : Type) [Storage α] (inputDim hiddenDim outputDim : Nat) where
  /-- `fc0.weight`, transposed. -/
  inputWeight : Tensor α [inputDim, hiddenDim]
  /-- `fc1.weight`, transposed. -/
  outputWeight : Tensor α [hiddenDim, outputDim]

/-- Parameters of one `MoonViTEncoderLayer`: the six released tensors of a block. -/
structure Block (α : Type) [Storage α] (heads hiddenDim headDim intermediateDim : Nat) where
  /-- `norm0.weight`, the RMSNorm scale before attention. -/
  attentionNormScale : Tensor α [hiddenDim]
  /-- `wqkv.weight`, transposed. Output columns are `(q | k | v)`, each laid out head-major. -/
  qkvWeight : Tensor α [hiddenDim, 3 * (heads * headDim)]
  /-- `wo.weight`, transposed. -/
  outputWeight : Tensor α [heads * headDim, hiddenDim]
  /-- `norm1.weight`, the RMSNorm scale before the MLP. -/
  feedForwardNormScale : Tensor α [hiddenDim]
  /-- `mlp.fc0.weight` and `mlp.fc1.weight`. -/
  feedForward : MLP α hiddenDim intermediateDim hiddenDim

namespace Block

variable {α : Type} [Storage α] [Context α]
variable {heads hiddenDim headDim intermediateDim tokens : Nat}

/-- Split the packed projection `x · wqkv` into queries, keys and values, as
`view(..., 3, heads, headDim)` followed by `unbind(dim=-3)`. -/
def splitQKV (packed : Tensor α [tokens, 3 * (heads * headDim)]) :
    Tensor α [3, tokens, heads * headDim] :=
  have hSize : Shape.size [tokens, 3 * (heads * headDim)] =
      Shape.size [tokens, 3, heads * headDim] := by
    simp [Shape.size]
  Tensor.swapAdjacentAxes (Tensor.reshapeSpec packed hSize) 0

/-- Self-attention of one media item: packed projection, 2-D RoPE on queries and keys, non-causal
scaled dot-product attention in every head, and the output projection. -/
def attention (block : Block α heads hiddenDim headDim intermediateDim)
    (cosTable sinTable : Tensor α [tokens, heads * headDim])
    (input : Tensor α [tokens, hiddenDim]) (hTokens : tokens ≠ 0) :
    Tensor α [tokens, hiddenDim] :=
  let packed := splitQKV (matMulSpec input block.qkvWeight)
  let query := applyRope cosTable sinTable (Tensor.unstack packed 0)
  let key := applyRope cosTable sinTable (Tensor.unstack packed 1)
  let value := Tensor.unstack packed 2
  let queryHeads := Spec.splitHeadsSpec query heads headDim rfl
  let keyHeads := Spec.splitHeadsSpec key heads headDim rfl
  let valueHeads := Spec.splitHeadsSpec value heads headDim rfl
  let attended : Tensor α [heads, tokens, headDim] :=
    Tensor.dim fun head =>
      let context : Spec.AttentionContext α tokens tokens headDim hTokens hTokens :=
        { Q := Tensor.unstack queryHeads head
          K := Tensor.unstack keyHeads head
          V := Tensor.unstack valueHeads head
          mask := none }
      Spec.scaledDotProductAttention context
  matMulSpec (Spec.combineHeadsSpec attended) block.outputWeight

/-- Bias-free MLP with PyTorch's tanh GELU (`PytorchGELUTanh`). -/
def feedForwardPass (feedForward : MLP α hiddenDim intermediateDim hiddenDim)
    (input : Tensor α [tokens, hiddenDim]) : Tensor α [tokens, hiddenDim] :=
  matMulSpec (Activation.geluSpec (matMulSpec input feedForward.inputWeight))
    feedForward.outputWeight

/-- `MoonViTEncoderLayer.forward`: two pre-norm residual branches. -/
def forward (block : Block α heads hiddenDim headDim intermediateDim)
    (epsilon : α) (cosTable sinTable : Tensor α [tokens, heads * headDim])
    (input : Tensor α [tokens, hiddenDim]) (hTokens : tokens ≠ 0) (hHidden : 0 < hiddenDim) :
    Tensor α [tokens, hiddenDim] :=
  let attended := Tensor.addSpec input
    (block.attention cosTable sinTable (normRows hHidden epsilon input block.attentionNormScale)
      hTokens)
  Tensor.addSpec attended
    (feedForwardPass block.feedForward
      (normRows hHidden epsilon attended block.feedForwardNormScale))

end Block

/-- `PatchMergerMLPV2`: two bias-free linear maps around the exact GELU, then RMSNorm. -/
structure Projector (α : Type) [Storage α] (mergedDim textDim : Nat) where
  /-- `mm_projector.proj.0.weight`, transposed. -/
  firstWeight : Tensor α [mergedDim, mergedDim]
  /-- `mm_projector.proj.2.weight`, transposed. -/
  secondWeight : Tensor α [mergedDim, textDim]
  /-- `mm_projector.post_norm.weight`. -/
  outputNormScale : Tensor α [textDim]

/-- Parameters of MoonViT-V2 and its projector: the released vision tensors. -/
structure Model (α : Type) [Storage α] (cfg : VisionConfig) (patchFeatures : Nat) where
  /-- `patch_embed.proj.weight` flattened over `(channel, y, x)` and transposed. -/
  patchWeight : Tensor α [patchFeatures, cfg.hiddenDim]
  /-- `patch_embed.pos_emb.weight`. -/
  positionWeight : Tensor α [positionRows, positionColumns, cfg.hiddenDim]
  /-- Ordered encoder blocks. -/
  blocks : List (Block α cfg.numHeads cfg.hiddenDim
    (cfg.qkvHiddenDim / cfg.numHeads) cfg.intermediateDim)
  /-- The parameter list contains exactly the configured number of blocks. -/
  blocks_length : blocks.length = cfg.numLayers
  /-- `encoder.final_layernorm.weight`. -/
  finalNormScale : Tensor α [cfg.hiddenDim]
  /-- Projector into the language width. -/
  projector : Projector α (cfg.mergeHeight * cfg.mergeWidth * cfg.hiddenDim) cfg.textHiddenDim

namespace Model

variable {α : Type} [Storage α] [Context α]
variable {cfg : VisionConfig}
variable {patchFeatures frames rows columns : Nat}

/-- Patch projection plus the resized position table and the frame embedding
(`MoonVision3dPatchEmbed.forward`). -/
def embed (model : Model α cfg patchFeatures)
    (patches : Grid α frames rows columns patchFeatures) (hFrames : frames ≤ maxFrames) :
    Grid α frames rows columns cfg.hiddenDim :=
  embedPatches model.patchWeight model.positionWeight
    (interpolationMatrix positionRows rows) (interpolationMatrix positionColumns columns)
    (frameEmbedding frames cfg.hiddenDim hFrames) patches

/-- Run all configured blocks over one item's token sequence, then the final RMSNorm. -/
def encode (model : Model α cfg patchFeatures) {tokens : Nat}
    (cosTable sinTable : Tensor α [tokens, cfg.numHeads * (cfg.qkvHiddenDim / cfg.numHeads)])
    (input : Tensor α [tokens, cfg.hiddenDim])
    (hTokens : tokens ≠ 0) (hHidden : 0 < cfg.hiddenDim) :
    Tensor α [tokens, cfg.hiddenDim] :=
  let hidden := model.blocks.foldl
    (fun hidden block => block.forward encoderNormEpsilon cosTable sinTable hidden hTokens hHidden)
    input
  normRows hHidden encoderNormEpsilon hidden model.finalNormScale

/-- With zero configured blocks, the encoder is only its final RMSNorm. -/
theorem encode_of_numLayers_eq_zero (model : Model α cfg patchFeatures) {tokens : Nat}
    (cosTable sinTable : Tensor α [tokens, cfg.numHeads * (cfg.qkvHiddenDim / cfg.numHeads)])
    (input : Tensor α [tokens, cfg.hiddenDim])
    (hTokens : tokens ≠ 0) (hHidden : 0 < cfg.hiddenDim) (hLayers : cfg.numLayers = 0) :
    model.encode cosTable sinTable input hTokens hHidden =
      normRows hHidden encoderNormEpsilon input model.finalNormScale := by
  have hLength : model.blocks.length = 0 := by simpa [hLayers] using model.blocks_length
  have hBlocks : model.blocks = [] := List.length_eq_zero_iff.mp hLength
  simp [encode, hBlocks]

omit [Context α] in
/-- A positive configured depth forces the model to contain at least one MoonViT block. -/
theorem blocks_ne_nil (model : Model α cfg patchFeatures)
    (hLayers : 0 < cfg.numLayers) : model.blocks ≠ [] := by
  intro hBlocks
  have hLength := model.blocks_length
  simp [hBlocks] at hLength
  omega

/--
Temporal pooling, pixel-shuffle merge, and projection into the language width
(`tpool_patch_merger` followed by `PatchMergerMLPV2`). The output sequence has one token for each
merged spatial location, and the frame axis is pooled away.
-/
def mergeAndProject [Erf α] (model : Model α cfg patchFeatures)
    (grid : Grid α frames (rows * cfg.mergeHeight) (columns * cfg.mergeWidth) cfg.hiddenDim)
    (hFrames : 0 < frames)
    (hText : 0 < cfg.textHiddenDim) :
    Tensor α [rows * columns, cfg.textHiddenDim] :=
  let pooled : Tensor α [rows * cfg.mergeHeight, columns * cfg.mergeWidth, cfg.hiddenDim] :=
    Tensor.reduceMean 0 grid (Shape.hasNonemptyAxisZeroOfPos hFrames).proof
  let interleavedShape : Shape :=
    [rows, cfg.mergeHeight, columns, cfg.mergeWidth, cfg.hiddenDim]
  let groupedShape : Shape :=
    [rows, columns, cfg.mergeHeight, cfg.mergeWidth, cfg.hiddenDim]
  let mergedDim := cfg.mergeHeight * cfg.mergeWidth * cfg.hiddenDim
  let mergedShape : Shape := [rows * columns, mergedDim]
  have hInterleaved :
      Shape.size [rows * cfg.mergeHeight, columns * cfg.mergeWidth, cfg.hiddenDim] =
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
  let hidden := geluExactSpec (matMulSpec merged model.projector.firstWeight)
  let projected := matMulSpec hidden model.projector.secondWeight
  RMSNorm.rows hText projected model.projector.outputNormScale

/-- Complete reference path for one media item: from flattened patches of at most four frames to
language-width visual tokens. -/
def forward [Erf α] (model : Model α cfg patchFeatures)
    (patches : Grid α frames (rows * cfg.mergeHeight) (columns * cfg.mergeWidth) patchFeatures)
    (hFrames : 0 < frames) (hMaxFrames : frames ≤ maxFrames)
    (hSpatial : 0 < (rows * cfg.mergeHeight) * (columns * cfg.mergeWidth))
    (hHidden : 0 < cfg.hiddenDim) (hText : 0 < cfg.textHiddenDim) :
    Tensor α [rows * columns, cfg.textHiddenDim] :=
  let height := rows * cfg.mergeHeight
  let width := columns * cfg.mergeWidth
  let tokens := frames * height * width
  have hGridTokens : Shape.size [frames, height, width, cfg.hiddenDim] =
      Shape.size [tokens, cfg.hiddenDim] := by
    simp [tokens, Shape.size, Nat.mul_assoc]
  have hTokens : tokens ≠ 0 := by
    have hPositive : 0 < frames * (height * width) := Nat.mul_pos hFrames hSpatial
    simpa [tokens, height, width, Nat.mul_assoc] using Nat.ne_of_gt hPositive
  let heads := cfg.numHeads
  let headDim := cfg.qkvHiddenDim / cfg.numHeads
  let sequence := Tensor.reshapeSpec (model.embed patches hMaxFrames) hGridTokens
  let encoded := model.encode
    (ropeTable MathFunctions.cos frames height width heads headDim)
    (ropeTable MathFunctions.sin frames height width heads headDim)
    sequence hTokens hHidden
  model.mergeAndProject (Tensor.reshapeSpec encoded hGridTokens.symm) hFrames hText

/-- A video given as `chunks` items of `frames` frames each. Items never attend to each other, so
each chunk is encoded and pooled on its own. Its visual tokens form one row of the result. -/
def forwardChunks [Erf α] (model : Model α cfg patchFeatures) {chunks : Nat}
    (video : Tensor α
      [chunks, frames, rows * cfg.mergeHeight, columns * cfg.mergeWidth, patchFeatures])
    (hFrames : 0 < frames) (hMaxFrames : frames ≤ maxFrames)
    (hSpatial : 0 < (rows * cfg.mergeHeight) * (columns * cfg.mergeWidth))
    (hHidden : 0 < cfg.hiddenDim) (hText : 0 < cfg.textHiddenDim) :
    Tensor α [chunks, rows * columns, cfg.textHiddenDim] :=
  Tensor.dim fun chunk =>
    model.forward (Tensor.unstack video chunk) hFrames hMaxFrames hSpatial hHidden hText

/-- Each chunk's tokens depend only on that chunk. -/
@[simp] theorem get_forwardChunks [Erf α] (model : Model α cfg patchFeatures) {chunks : Nat}
    (video : Tensor α
      [chunks, frames, rows * cfg.mergeHeight, columns * cfg.mergeWidth, patchFeatures])
    (hFrames : 0 < frames) (hMaxFrames : frames ≤ maxFrames)
    (hSpatial : 0 < (rows * cfg.mergeHeight) * (columns * cfg.mergeWidth))
    (hHidden : 0 < cfg.hiddenDim) (hText : 0 < cfg.textHiddenDim) (chunk : Fin chunks) :
    Spec.get (model.forwardChunks video hFrames hMaxFrames hSpatial hHidden hText) chunk =
      model.forward (Spec.get video chunk) hFrames hMaxFrames hSpatial hHidden hText := by
  simp [forwardChunks, Spec.get]

end Model

end MoonViT

end KimiK3
