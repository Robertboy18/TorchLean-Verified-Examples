/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.OCP.MX.Standard.Core
public import NN.Proofs.Tensor.Basic.LinearAlgebra
public import NN.Spec.Core.TensorReductionShape
public import NN.Tensor.Operations

/-!
# OCP microscaling formats used by Kimi K3

Kimi K3 stores routed-expert weights in MXFP4 and supplies their matrix multiplications with
MXFP8 activations.  Both formats use the Open Compute Project's microscaling layout, in which 32
private floating-point values share one E8M0 scale.  MXFP4 uses E2M1 private values.  MXFP8 can use
E4M3 or E5M2, and the K3 report does not say which, so the format is an argument.

Scalar and block definitions do not depend on K3's layer dimensions.  `BlockVector` and
`BlockMatrix` lift the 32-value encoding to the tensors of the expert matrix multiplications.
Matrices follow TorchLean's convention (first axis input, second axis output), so a block holds 32
consecutive input coordinates for one output coordinate.

References:

* Kimi Team, "Kimi K3: Open Frontier Intelligence", 2026, Section 4.1.4, p. 14:
  https://arxiv.org/abs/2607.24653
* Open Compute Project, "Microscaling Formats (MX) Specification", version 1.0,
  Sections 5.2--5.4, September 2023.
-/

@[expose] public section

namespace KimiK3
namespace Microscaling

open TorchLean
open Spec
open Tensor

open FloatLib.Floats.Formats.OCP
attribute [local simp] Tensor.stackLeading

/-- A finite E8M0 scale code. OCP reserves the encoding `255` for NaN, so finite scales are
represented by the remaining 255 codes. -/
abbrev E8M0 := Fin 255

/-- Real value of an E8M0 scale. Code `e` denotes the binary power `2^(e - 127)`. -/
noncomputable def scaleValue (scale : E8M0) : ℝ :=
  (2 : ℝ) ^ (MX.E8M0.exponent? (BitVec.ofNat 8 scale.val)).getD 0

/-- Every finite E8M0 scale is strictly positive. -/
theorem scaleValue_pos (scale : E8M0) : 0 < scaleValue scale :=
  zpow_pos (by norm_num) _

/-- Decode an E2M1 word with FloatLib and cast its rational value to ℝ.
Both signed-zero encodings decode to zero. -/
noncomputable def decodeE2M1 (code : Fin 16) : ℝ :=
  ((MX.Standard.Element.toRat? (profile := .e2m1) (BitVec.ofFin code)).getD 0 : ℚ)

-- The finite value table lets real-arithmetic proofs simplify concrete element codes.
private theorem e2m1_values (code : Fin 16) :
    (MX.Standard.Element.toRat? (profile := .e2m1) (BitVec.ofFin code)).getD 0 =
      ([0, 1/2, 1, 3/2, 2, 3, 4, 6, 0, -(1/2), -1, -(3/2), -2, -3, -4, -6] :
        List ℚ)[code.val]! := by
  revert code
  decide +kernel

/-- Every E2M1 value has magnitude at most six (the format has no infinities or NaNs). -/
theorem abs_decodeE2M1_le_six (code : Fin 16) : |decodeE2M1 code| ≤ 6 := by
  fin_cases code <;> norm_num [decodeE2M1, e2m1_values]

/-- Saturating nearest-even conversion of a real number to an E2M1 word.
Magnitude is rounded first and the sign bit kept afterwards, so a small negative input
rounds to negative zero. -/
noncomputable def quantizeE2M1 (value : ℝ) : Fin 16 :=
  let magnitude : Fin 16 :=
    if |value| ≤ 1 / 4 then 0
    else if |value| < 3 / 4 then 1
    else if |value| ≤ 5 / 4 then 2
    else if |value| < 7 / 4 then 3
    else if |value| ≤ 5 / 2 then 4
    else if |value| < 7 / 2 then 5
    else if |value| ≤ 5 then 6
    else 7
  if value < 0 then ⟨(magnitude.val + 8) % 16, Nat.mod_lt _ (by omega)⟩
  else magnitude

/-- In-range E2M1 rounding has absolute error at most one, attained between four and six. -/
theorem quantizeE2M1_error_le_one {value : ℝ} (hvalue : |value| ≤ 6) :
    |decodeE2M1 (quantizeE2M1 value) - value| ≤ 1 := by
  rw [abs_le] at hvalue
  unfold quantizeE2M1
  by_cases hnegative : value < 0
  · simp only [abs_of_neg hnegative, ite_eq_left hnegative]
    split_ifs <;> norm_num [decodeE2M1, e2m1_values] <;>
      rw [abs_le] <;> constructor <;> linarith
  · simp only [abs_of_nonneg (le_of_not_gt hnegative), ite_eq_right hnegative]
    split_ifs <;> norm_num [decodeE2M1, e2m1_values] <;>
      rw [abs_le] <;> constructor <;> linarith

/-- Quantizing a decoded E2M1 value returns the same real value, signed zeros included. -/
@[simp] theorem decode_quantizeE2M1 (code : Fin 16) :
    decodeE2M1 (quantizeE2M1 (decodeE2M1 code)) = decodeE2M1 code := by
  fin_cases code <;> norm_num [quantizeE2M1, decodeE2M1, e2m1_values]

/-- The two OCP private-element encodings available to MXFP8. -/
inductive FP8Format where
  | e4m3
  | e5m2
  deriving Repr, DecidableEq

namespace FP8Format

/-- FloatLib profile used to decode the selected MXFP8 element format. -/
def profile : FP8Format → MX.Standard.Profile
  | .e4m3 => .e4m3
  | .e5m2 => .e5m2

/-- A word is finite when FloatLib decodes it to a rational, which rules out the exceptional
encodings. -/
abbrev IsFinite (format : FP8Format) (code : Fin 256) : Prop :=
  (MX.Standard.Element.toRat? (profile := format.profile)
    (BitVec.ofNat format.profile.width code.val)).isSome = true

end FP8Format

/-- A finite FP8 element code in the selected OCP format. -/
abbrev FP8Code (format : FP8Format) := { code : Fin 256 // format.IsFinite code }

/-- Decode a finite MXFP8 element through FloatLib, including subnormals. -/
noncomputable def decodeFP8 (format : FP8Format) (code : FP8Code format) : ℝ :=
  ((MX.Standard.Element.toRat? (profile := format.profile)
    (BitVec.ofNat format.profile.width code.1.val)).getD 0 : ℚ)

/-- A microscaling block with the OCP-mandated block size of 32. -/
structure Block (Code : Type) where
  /-- Shared positive power-of-two scale for the block. -/
  scale : E8M0
  /-- Private element code at each of the 32 block positions. -/
  element : Fin 32 → Code

namespace Block

/-- Decode one coordinate by multiplying its private value by the shared block scale. -/
noncomputable def decode {Code : Type} (decodeElement : Code → ℝ) (block : Block Code)
    (index : Fin 32) : ℝ :=
  scaleValue block.scale * decodeElement (block.element index)

/-- Dot product of two decoded blocks. -/
noncomputable def dot {CodeA CodeB : Type} (decodeA : CodeA → ℝ) (decodeB : CodeB → ℝ)
    (a : Block CodeA) (b : Block CodeB) : ℝ :=
  ∑ index, a.decode decodeA index * b.decode decodeB index

/-- MX dot products factor into the product of block scales and a dot product of private values. -/
theorem dot_eq_scale_mul_sum {CodeA CodeB : Type} (decodeA : CodeA → ℝ)
    (decodeB : CodeB → ℝ) (a : Block CodeA) (b : Block CodeB) :
    a.dot decodeA decodeB b =
      scaleValue a.scale * scaleValue b.scale *
        ∑ index, decodeA (a.element index) * decodeB (b.element index) := by
  rw [dot, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro index _
  simp only [decode]
  ring_nf

end Block

/-- An MXFP4 block: 32 E2M1 elements with one finite E8M0 scale. -/
abbrev MXFP4Block := Block (Fin 16)

/-- An MXFP8 block in the chosen OCP element format. -/
abbrev MXFP8Block (format : FP8Format) := Block (FP8Code format)

/-- Quantize 32 real values with a given finite E8M0 scale. Scale selection is left to the caller
because the K3 report fixes the format but not its amax tie and overflow policy. -/
noncomputable def quantizeMXFP4Block (scale : E8M0) (values : Fin 32 → ℝ) : MXFP4Block :=
  { scale
    element := fun index => quantizeE2M1 (values index / scaleValue scale) }

/-- Every decoded MXFP4 coordinate is bounded by six times its block scale. -/
theorem abs_mxfp4_decode_le (block : MXFP4Block) (index : Fin 32) :
    |block.decode decodeE2M1 index| ≤ 6 * scaleValue block.scale := by
  rw [Block.decode, abs_mul, abs_of_pos (scaleValue_pos block.scale)]
  simpa [mul_comm] using
    mul_le_mul_of_nonneg_left (abs_decodeE2M1_le_six (block.element index))
      (le_of_lt (scaleValue_pos block.scale))

/-- If every source value fits the range selected by the shared scale, MXFP4 quantization has
coordinate error at most that scale. -/
theorem quantizeMXFP4Block_error_le (scale : E8M0) (values : Fin 32 → ℝ)
    (hRange : ∀ index, |values index| ≤ 6 * scaleValue scale) (index : Fin 32) :
    |(quantizeMXFP4Block scale values).decode decodeE2M1 index - values index| ≤
      scaleValue scale := by
  let scaleReal := scaleValue scale
  have hscale : 0 < scaleReal := scaleValue_pos scale
  have hscaled : |values index / scaleReal| ≤ 6 := by
    rw [abs_div, abs_of_pos hscale]
    exact (div_le_iff₀ hscale).2 (by simpa [scaleReal, mul_comm] using hRange index)
  have hround := quantizeE2M1_error_le_one hscaled
  change |scaleReal * decodeE2M1 (quantizeE2M1 (values index / scaleReal)) -
    values index| ≤ scaleReal
  have hidentity :
      scaleReal * decodeE2M1 (quantizeE2M1 (values index / scaleReal)) - values index =
        scaleReal *
          (decodeE2M1 (quantizeE2M1 (values index / scaleReal)) -
            values index / scaleReal) := by
    field_simp
  calc
    |scaleReal * decodeE2M1 (quantizeE2M1 (values index / scaleReal)) - values index| =
        scaleReal *
          |decodeE2M1 (quantizeE2M1 (values index / scaleReal)) -
            values index / scaleReal| := by
      rw [hidentity, abs_mul, abs_of_pos hscale]
    _ ≤ scaleReal * 1 := mul_le_mul_of_nonneg_left hround hscale.le
    _ = scaleReal := mul_one _

/-! ## Blocked vectors and matrices -/

/-- A vector represented by consecutive 32-value MX blocks. -/
abbrev BlockVector (Code : Type) (blocks : Nat) := Fin blocks → Block Code

/-- Decode a blocked vector to the shape used by TorchLean's tensor specifications. -/
noncomputable def decodeVector {Code : Type} {blocks : Nat} (decodeElement : Code → ℝ)
    (vector : BlockVector Code blocks) : Tensor ℝ [blocks * 32] :=
  Tensor.stackLeading fun index =>
    let position := finProdFinEquiv.symm index
    Tensor.scalar ((vector position.1).decode decodeElement position.2)

/-- An MXFP8 vector in one chosen private-element format. -/
abbrev MXFP8Vector (format : FP8Format) (blocks : Nat) :=
  BlockVector (FP8Code format) blocks

/-- A checked MXFP8 encoding of a real tensor: encoded blocks plus a coordinatewise error bound.

The K3 report fixes MXFP8 as the activation format but states neither the E4M3/E5M2 choice nor
the scale-selection and tie-breaking policy. Instead of guessing them, the runtime encoder supplies
its blocks and proves the error bound, and the expert analysis works from that certificate alone.
-/
structure MXFP8Encoding (format : FP8Format) (blocks : Nat)
    (source : Tensor ℝ [blocks * 32]) where
  /-- Encoded MXFP8 blocks. -/
  encoded : MXFP8Vector format blocks
  /-- Coordinatewise upper bounds on absolute decoding error. -/
  errorBound : Tensor ℝ [blocks * 32]
  decode_error_le : ∀ index,
    |Tensor.getScalar (decodeVector (decodeFP8 format) encoded) index -
        Tensor.getScalar source index| ≤
      Tensor.getScalar errorBound index

namespace MXFP8Encoding

/-- Decode the certified blocks carried by an MXFP8 encoding. -/
noncomputable def decode {format : FP8Format} {blocks : Nat}
    {source : Tensor ℝ [blocks * 32]}
    (encoding : MXFP8Encoding format blocks source) :
    Tensor ℝ [blocks * 32] :=
  decodeVector (decodeFP8 format) encoding.encoded

/-- Bounds in an absolute-error certificate are nonnegative. -/
theorem errorBound_nonneg {format : FP8Format} {blocks : Nat}
    {source : Tensor ℝ [blocks * 32]}
    (encoding : MXFP8Encoding format blocks source) (index : Fin (blocks * 32)) :
    0 ≤ Tensor.getScalar encoding.errorBound index :=
  (abs_nonneg _).trans (encoding.decode_error_le index)

end MXFP8Encoding

/-- A matrix whose input axis is partitioned into blocks of 32. Each output coordinate has its own
scale for each consecutive group of 32 inputs. -/
abbrev BlockMatrix (Code : Type) (inputBlocks outputDim : Nat) :=
  Fin outputDim → Fin inputBlocks → Block Code

namespace BlockMatrix

/-- Decode a blocked matrix into TorchLean's input-by-output matrix convention. -/
noncomputable def decode {Code : Type} {inputBlocks outputDim : Nat}
    (decodeElement : Code → ℝ) (matrix : BlockMatrix Code inputBlocks outputDim) :
    Tensor ℝ [inputBlocks * 32, outputDim] :=
  Tensor.stackLeading fun input =>
    let position := finProdFinEquiv.symm input
    Tensor.stackLeading fun output =>
      Tensor.scalar ((matrix output position.1).decode decodeElement position.2)

/-- A decoded matrix coordinate reads its output column and consecutive 32-value input block. -/
@[simp] theorem get2_decode {Code : Type} {inputBlocks outputDim : Nat}
    (decodeElement : Code → ℝ) (matrix : BlockMatrix Code inputBlocks outputDim)
    (input : Fin (inputBlocks * 32)) (output : Fin outputDim) :
    get2 (matrix.decode decodeElement) input output =
      (matrix output (finProdFinEquiv.symm input).1).decode decodeElement
        (finProdFinEquiv.symm input).2 := by
  simp [decode, Spec.get2, Spec.get]

/-- Every decoded MXFP4 matrix coordinate is bounded by six times the scale of its block. -/
theorem abs_decode_mxfp4_le {inputBlocks outputDim : Nat}
    (matrix : BlockMatrix (Fin 16) inputBlocks outputDim)
    (input : Fin (inputBlocks * 32)) (output : Fin outputDim) :
    |get2 (matrix.decode decodeE2M1) input output| ≤
      6 * scaleValue (matrix output (finProdFinEquiv.symm input).1).scale := by
  rw [get2_decode]
  exact abs_mxfp4_decode_le
    (matrix output (finProdFinEquiv.symm input).1) (finProdFinEquiv.symm input).2

end BlockMatrix

/-- An MXFP4 matrix with block scaling along its input axis. -/
abbrev MXFP4Matrix (inputBlocks outputDim : Nat) := BlockMatrix (Fin 16) inputBlocks outputDim

/-- Quantize a matrix with a given E8M0 scale for every output/block pair. -/
noncomputable def quantizeMXFP4Matrix {inputBlocks outputDim : Nat}
    (scale : Fin outputDim → Fin inputBlocks → E8M0)
    (values : Tensor ℝ [inputBlocks * 32, outputDim]) :
    MXFP4Matrix inputBlocks outputDim :=
  fun output inputBlock => quantizeMXFP4Block (scale output inputBlock) fun offset =>
    get2 values (finProdFinEquiv (inputBlock, offset)) output

/-- Matrix quantization inherits the block bound at the corresponding output and input block.
This per-weight bound feeds the matrix-product bounds below. -/
theorem quantizeMXFP4Matrix_error_le {inputBlocks outputDim : Nat}
    (scale : Fin outputDim → Fin inputBlocks → E8M0)
    (values : Tensor ℝ [inputBlocks * 32, outputDim])
    (hRange : ∀ output inputBlock offset,
      |get2 values (finProdFinEquiv (inputBlock, offset)) output| ≤
        6 * scaleValue (scale output inputBlock))
    (input : Fin (inputBlocks * 32)) (output : Fin outputDim) :
    |get2 ((quantizeMXFP4Matrix scale values).decode decodeE2M1) input output -
        get2 values input output| ≤
      scaleValue (scale output (finProdFinEquiv.symm input).1) := by
  rw [BlockMatrix.get2_decode]
  simpa only [quantizeMXFP4Matrix, Prod.mk.eta, Equiv.apply_symm_apply] using
    quantizeMXFP4Block_error_le (scale output (finProdFinEquiv.symm input).1)
      (fun coordinate => get2 values
        (finProdFinEquiv ((finProdFinEquiv.symm input).1, coordinate)) output)
      (hRange output (finProdFinEquiv.symm input).1) (finProdFinEquiv.symm input).2

/-- Quantizing a matrix perturbs one output coordinate by at most the input-weighted sum of the
shared scales of the blocks contributing to that coordinate. It keeps the actual activation
magnitudes and each block's own scale instead of a dimension-only estimate. -/
theorem vecMatMul_quantizeMXFP4Matrix_error_le {inputBlocks outputDim : Nat}
    (scale : Fin outputDim → Fin inputBlocks → E8M0)
    (weights : Tensor ℝ [inputBlocks * 32, outputDim])
    (hRange : ∀ output inputBlock offset,
      |get2 weights (finProdFinEquiv (inputBlock, offset)) output| ≤
        6 * scaleValue (scale output inputBlock))
    (input : Tensor ℝ [inputBlocks * 32]) (output : Fin outputDim) :
    |Tensor.getScalar
          (vecMatMulSpec input ((quantizeMXFP4Matrix scale weights).decode decodeE2M1)) output -
        Tensor.getScalar (vecMatMulSpec input weights) output| ≤
      ∑ index : Fin (inputBlocks * 32),
        |Tensor.getScalar input index| *
          scaleValue (scale output (finProdFinEquiv.symm index).1) := by
  rw [Spec.getScalar_vec_mat_mul_spec, Spec.getScalar_vec_mat_mul_spec,
    ← Finset.sum_sub_distrib]
  calc
    |∑ index : Fin (inputBlocks * 32),
        (Tensor.getScalar input index *
          get2 ((quantizeMXFP4Matrix scale weights).decode decodeE2M1) index output -
          Tensor.getScalar input index * get2 weights index output)| =
        |∑ index : Fin (inputBlocks * 32),
          Tensor.getScalar input index *
            (get2 ((quantizeMXFP4Matrix scale weights).decode decodeE2M1) index output -
              get2 weights index output)| := by
      congr 1
      apply Finset.sum_congr rfl
      intro index _
      ring
    _ ≤ ∑ index : Fin (inputBlocks * 32),
        |Tensor.getScalar input index *
          (get2 ((quantizeMXFP4Matrix scale weights).decode decodeE2M1) index output -
            get2 weights index output)| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ index : Fin (inputBlocks * 32),
        |Tensor.getScalar input index| *
          |get2 ((quantizeMXFP4Matrix scale weights).decode decodeE2M1) index output -
            get2 weights index output| := by
      apply Finset.sum_congr rfl
      intro index _
      rw [abs_mul]
    _ ≤ ∑ index : Fin (inputBlocks * 32),
        |Tensor.getScalar input index| *
          scaleValue (scale output (finProdFinEquiv.symm index).1) := by
      apply Finset.sum_le_sum
      intro index _
      exact mul_le_mul_of_nonneg_left
        (quantizeMXFP4Matrix_error_le scale weights hRange index output) (abs_nonneg _)

/-- Combined MXFP8-activation and MXFP4-weight error for one matrix-product coordinate.

In the sum, the first term comes from activation encoding and uses the magnitude bound on the
decoded MXFP4 weight. The second comes from E2M1 weight rounding and uses the original activation.
Keeping them apart helps in QAT, where different scale policies control the two errors.
-/
theorem vecMatMul_mxfp8_mxfp4_error_le {format : FP8Format}
    {inputBlocks outputDim : Nat}
    {sourceInput : Tensor ℝ [inputBlocks * 32]}
    (input : MXFP8Encoding format inputBlocks sourceInput)
    (scale : Fin outputDim → Fin inputBlocks → E8M0)
    (weights : Tensor ℝ [inputBlocks * 32, outputDim])
    (hRange : ∀ output inputBlock offset,
      |get2 weights (finProdFinEquiv (inputBlock, offset)) output| ≤
        6 * scaleValue (scale output inputBlock))
    (output : Fin outputDim) :
    |Tensor.getScalar (vecMatMulSpec input.decode
          ((quantizeMXFP4Matrix scale weights).decode decodeE2M1)) output -
        Tensor.getScalar (vecMatMulSpec sourceInput weights) output| ≤
      ∑ index : Fin (inputBlocks * 32),
        (Tensor.getScalar input.errorBound index *
            (6 * scaleValue (scale output (finProdFinEquiv.symm index).1)) +
          |Tensor.getScalar sourceInput index| *
            scaleValue (scale output (finProdFinEquiv.symm index).1)) := by
  rw [Spec.getScalar_vec_mat_mul_spec, Spec.getScalar_vec_mat_mul_spec,
    ← Finset.sum_sub_distrib]
  calc
    |∑ index : Fin (inputBlocks * 32),
        (Tensor.getScalar input.decode index *
          get2 ((quantizeMXFP4Matrix scale weights).decode decodeE2M1) index output -
          Tensor.getScalar sourceInput index * get2 weights index output)| =
        |∑ index : Fin (inputBlocks * 32),
          ((Tensor.getScalar input.decode index - Tensor.getScalar sourceInput index) *
              get2 ((quantizeMXFP4Matrix scale weights).decode decodeE2M1) index output +
            Tensor.getScalar sourceInput index *
              (get2 ((quantizeMXFP4Matrix scale weights).decode decodeE2M1) index output -
                get2 weights index output))| := by
      congr 1
      apply Finset.sum_congr rfl
      intro index _
      ring
    _ ≤ ∑ index : Fin (inputBlocks * 32),
        |(Tensor.getScalar input.decode index - Tensor.getScalar sourceInput index) *
              get2 ((quantizeMXFP4Matrix scale weights).decode decodeE2M1) index output +
            Tensor.getScalar sourceInput index *
              (get2 ((quantizeMXFP4Matrix scale weights).decode decodeE2M1) index output -
                get2 weights index output)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ index : Fin (inputBlocks * 32),
        (Tensor.getScalar input.errorBound index *
            (6 * scaleValue (scale output (finProdFinEquiv.symm index).1)) +
          |Tensor.getScalar sourceInput index| *
            scaleValue (scale output (finProdFinEquiv.symm index).1)) := by
      apply Finset.sum_le_sum
      intro index _
      calc
        |(Tensor.getScalar input.decode index - Tensor.getScalar sourceInput index) *
              get2 ((quantizeMXFP4Matrix scale weights).decode decodeE2M1) index output +
            Tensor.getScalar sourceInput index *
              (get2 ((quantizeMXFP4Matrix scale weights).decode decodeE2M1) index output -
                get2 weights index output)| ≤
            |Tensor.getScalar input.decode index - Tensor.getScalar sourceInput index| *
                |get2 ((quantizeMXFP4Matrix scale weights).decode decodeE2M1) index output| +
              |Tensor.getScalar sourceInput index| *
                |get2 ((quantizeMXFP4Matrix scale weights).decode decodeE2M1) index output -
                  get2 weights index output| := by
            simpa only [abs_mul] using abs_add_le
              ((Tensor.getScalar input.decode index - Tensor.getScalar sourceInput index) *
                get2 ((quantizeMXFP4Matrix scale weights).decode decodeE2M1) index output)
              (Tensor.getScalar sourceInput index *
                (get2 ((quantizeMXFP4Matrix scale weights).decode decodeE2M1) index output -
                  get2 weights index output))
        _ ≤ Tensor.getScalar input.errorBound index *
              (6 * scaleValue (scale output (finProdFinEquiv.symm index).1)) +
            |Tensor.getScalar sourceInput index| *
              scaleValue (scale output (finProdFinEquiv.symm index).1) := by
          apply add_le_add
          · exact mul_le_mul
              (input.decode_error_le index)
              (BlockMatrix.abs_decode_mxfp4_le
                (quantizeMXFP4Matrix scale weights) index output)
              (abs_nonneg _)
              (input.errorBound_nonneg index)
          · exact mul_le_mul_of_nonneg_left
              (quantizeMXFP4Matrix_error_le scale weights hRange index output) (abs_nonneg _)

end Microscaling
end KimiK3
