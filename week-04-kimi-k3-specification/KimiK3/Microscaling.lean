/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Core
public import NN.Proofs.Tensor.Basic.LinearAlgebra
public import NN.Spec.Core.TensorReductionShape

/-!
# OCP microscaling formats used by Kimi K3

Kimi K3 stores routed-expert weights in MXFP4 and supplies their matrix multiplications with
MXFP8 activations.  Both formats use the Open Compute Project's microscaling layout: 32 private
floating-point values share one E8M0 scale.  MXFP4 uses E2M1 private values.  MXFP8 can use E4M3
or E5M2; the K3 report does not identify which of the two encodings is used, so the format remains
an explicit argument.

The scalar and block definitions below are independent of K3's layer dimensions.  `BlockVector`
and `BlockMatrix` then lift the 32-value encoding to the tensors used by expert matrix
multiplications.  The matrix layout follows TorchLean's convention: the first tensor axis is the
input axis and the second is the output axis.  A block therefore contains 32 consecutive input
coordinates for one output coordinate.

Reference: Open Compute Project, "Microscaling Formats (MX) Specification", version 1.0,
Sections 5.2--5.4, September 2023.
-/

@[expose] public section

namespace KimiK3
namespace Microscaling

open TorchLean.Floats
open Spec
open Tensor

/-- A finite E8M0 scale code. The OCP encoding `255` is reserved for NaN, so finite scales are
represented by the remaining 255 codes. -/
abbrev E8M0 := Fin 255

/-- Real value of an E8M0 scale. Code `e` denotes the binary power `2^(e - 127)`. -/
noncomputable def scaleValue (scale : E8M0) : ℝ :=
  neuralBpow binaryRadix ((scale.val : ℤ) - 127)

/-- Every finite E8M0 scale is strictly positive. -/
theorem scaleValue_pos (scale : E8M0) : 0 < scaleValue scale :=
  neuralBpow.pos binaryRadix _

/-- Decode one four-bit E2M1 payload. Codes `0`--`7` are nonnegative and codes `8`--`15`
carry the sign bit. Both signed-zero encodings denote real zero. -/
noncomputable def decodeE2M1 (code : Fin 16) : ℝ :=
  match code.val with
  | 0 | 8 => 0
  | 1 => 1 / 2
  | 2 => 1
  | 3 => 3 / 2
  | 4 => 2
  | 5 => 3
  | 6 => 4
  | 7 => 6
  | 9 => -(1 / 2)
  | 10 => -1
  | 11 => -(3 / 2)
  | 12 => -2
  | 13 => -3
  | 14 => -4
  | _ => -6

/-- E2M1 has no infinities or NaNs; its largest finite magnitude is six. -/
theorem abs_decodeE2M1_le_six (code : Fin 16) : |decodeE2M1 code| ≤ 6 := by
  fin_cases code <;> norm_num [decodeE2M1]

/-- Flip the sign bit of an E2M1 payload. -/
def negateE2M1Code (code : Fin 16) : Fin 16 :=
  ⟨(code.val + 8) % 16, Nat.mod_lt _ (by omega)⟩

/-- Flipping the E2M1 sign bit negates its decoded real value. -/
theorem decodeE2M1_negate (code : Fin 16) :
    decodeE2M1 (negateE2M1Code code) = -decodeE2M1 code := by
  fin_cases code <;> norm_num [negateE2M1Code, decodeE2M1]

/-- Round a nonnegative real to E2M1 with saturation and ties to an even significand. -/
noncomputable def quantizeE2M1Nonnegative (value : ℝ) : Fin 16 :=
  if value ≤ 1 / 4 then 0
  else if value < 3 / 4 then 1
  else if value ≤ 5 / 4 then 2
  else if value < 7 / 4 then 3
  else if value ≤ 5 / 2 then 4
  else if value < 7 / 2 then 5
  else if value ≤ 5 then 6
  else 7

/-- Saturating round-to-nearest-even conversion from real values to E2M1 payloads. -/
noncomputable def quantizeE2M1 (value : ℝ) : Fin 16 :=
  if value < 0 then negateE2M1Code (quantizeE2M1Nonnegative (-value))
  else quantizeE2M1Nonnegative value

/-- On the nonnegative representable interval, E2M1 conversion has absolute error at most one.
The constant is sharp at the midpoint between the two largest finite values, four and six. -/
private theorem quantizeE2M1Nonnegative_error_le_one {value : ℝ}
    (hzero : 0 ≤ value) (hsix : value ≤ 6) :
    |decodeE2M1 (quantizeE2M1Nonnegative value) - value| ≤ 1 := by
  simp only [quantizeE2M1Nonnegative]
  split_ifs <;> norm_num [decodeE2M1] <;> rw [abs_le] <;> constructor <;> linarith

/-- Every in-range scalar converted to E2M1 and decoded again differs by at most one. -/
theorem quantizeE2M1_error_le_one {value : ℝ} (hvalue : |value| ≤ 6) :
    |decodeE2M1 (quantizeE2M1 value) - value| ≤ 1 := by
  rw [abs_le] at hvalue
  by_cases hnegative : value < 0
  · rw [quantizeE2M1, if_pos hnegative, decodeE2M1_negate]
    have hround := quantizeE2M1Nonnegative_error_le_one
      (value := -value) (by linarith) (by linarith [hvalue.1])
    calc
      |-decodeE2M1 (quantizeE2M1Nonnegative (-value)) - value| =
          |decodeE2M1 (quantizeE2M1Nonnegative (-value)) + value| := by
        rw [show -decodeE2M1 (quantizeE2M1Nonnegative (-value)) - value =
          -(decodeE2M1 (quantizeE2M1Nonnegative (-value)) + value) by ring_nf, abs_neg]
      _ = |decodeE2M1 (quantizeE2M1Nonnegative (-value)) - (-value)| := by ring_nf
      _ ≤ 1 := hround
  · rw [quantizeE2M1, if_neg hnegative]
    exact quantizeE2M1Nonnegative_error_le_one (le_of_not_gt hnegative) hvalue.2

/-- The two OCP private-element encodings available to MXFP8. -/
inductive FP8Format where
  | e4m3
  | e5m2
  deriving Repr, DecidableEq

namespace FP8Format

/-- Number of trailing significand bits in an FP8 encoding. -/
def mantissaBits : FP8Format → Nat
  | .e4m3 => 3
  | .e5m2 => 2

/-- Exponent bias of an FP8 encoding. -/
def bias : FP8Format → ℤ
  | .e4m3 => 7
  | .e5m2 => 15

/-- Raw biased exponent contained in an FP8 byte. -/
def rawExponent (format : FP8Format) (code : Fin 256) : Nat :=
  code.val / 2 ^ format.mantissaBits %
    (match format with | .e4m3 => 16 | .e5m2 => 32)

/-- Raw trailing significand contained in an FP8 byte. -/
def rawMantissa (format : FP8Format) (code : Fin 256) : Nat :=
  code.val % 2 ^ format.mantissaBits

/-- Whether an FP8 byte denotes a finite value in the OCP format. E4M3 reserves exponent 15,
mantissa 7 for NaN; E5M2 reserves exponent 31 for infinities and NaNs. -/
def IsFinite (format : FP8Format) (code : Fin 256) : Prop :=
  match format with
  | .e4m3 => format.rawExponent code ≠ 15 ∨ format.rawMantissa code ≠ 7
  | .e5m2 => format.rawExponent code ≠ 31

instance (format : FP8Format) (code : Fin 256) : Decidable (format.IsFinite code) := by
  cases format <;> simp only [IsFinite] <;> infer_instance

end FP8Format

/-- A finite FP8 element code in the selected OCP format. -/
abbrev FP8Code (format : FP8Format) := { code : Fin 256 // format.IsFinite code }

/-- Decode one finite OCP FP8 element, including subnormals. -/
noncomputable def decodeFP8 (format : FP8Format) (code : FP8Code format) : ℝ :=
  let rawExponent := format.rawExponent code.1
  let rawMantissa := format.rawMantissa code.1
  let denominator : ℝ := (2 ^ format.mantissaBits : Nat)
  let magnitude :=
    if rawExponent = 0 then
      (rawMantissa : ℝ) / denominator * neuralBpow binaryRadix (1 - format.bias)
    else
      (1 + (rawMantissa : ℝ) / denominator) *
        neuralBpow binaryRadix ((rawExponent : ℤ) - format.bias)
  if code.1.val < 128 then magnitude else -magnitude

/-- A microscaling block with the OCP-mandated block size of 32. -/
structure Block (Code : Type) where
  scale : E8M0
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

/-- An MXFP8 block in the explicitly selected OCP element format. -/
abbrev MXFP8Block (format : FP8Format) := Block (FP8Code format)

/-- Quantize 32 real values using a supplied finite E8M0 scale. Scale selection is separate
because the K3 report fixes the format but does not specify its amax tie and overflow policy. -/
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
    (vector : BlockVector Code blocks) : Tensor ℝ (.dim (blocks * 32) .scalar) :=
  Tensor.dim fun index =>
    let position := finProdFinEquiv.symm index
    Tensor.scalar ((vector position.1).decode decodeElement position.2)

/-- An MXFP8 vector in one explicitly selected private-element format. -/
abbrev MXFP8Vector (format : FP8Format) (blocks : Nat) :=
  BlockVector (FP8Code format) blocks

/-- A checked MXFP8 encoding of a real tensor.

The K3 report fixes MXFP8 as the activation format but does not state the E4M3/E5M2 choice or the
scale-selection and tie-breaking policy. Rather than silently invent those choices, a runtime
encoder supplies its concrete blocks together with a coordinatewise error certificate. This
structure is the proof boundary between that policy and the format-independent expert analysis.
-/
structure MXFP8Encoding (format : FP8Format) (blocks : Nat)
    (source : Tensor ℝ (.dim (blocks * 32) .scalar)) where
  encoded : MXFP8Vector format blocks
  errorBound : Tensor ℝ (.dim (blocks * 32) .scalar)
  errorBound_nonneg : ∀ index, 0 ≤ Tensor.getScalar errorBound index
  decode_error_le : ∀ index,
    |Tensor.getScalar (decodeVector (decodeFP8 format) encoded) index - Tensor.getScalar source index| ≤
      Tensor.getScalar errorBound index

namespace MXFP8Encoding

/-- Decode the certified blocks carried by an MXFP8 encoding. -/
noncomputable def decode {format : FP8Format} {blocks : Nat}
    {source : Tensor ℝ (.dim (blocks * 32) .scalar)}
    (encoding : MXFP8Encoding format blocks source) :
    Tensor ℝ (.dim (blocks * 32) .scalar) :=
  decodeVector (decodeFP8 format) encoding.encoded

end MXFP8Encoding

/-- A matrix whose input axis is partitioned into blocks of 32. Each output coordinate has one
independent block scale for each consecutive group on the input axis. -/
structure BlockMatrix (Code : Type) (inputBlocks outputDim : Nat) where
  block : Fin outputDim → Fin inputBlocks → Block Code

namespace BlockMatrix

/-- Decode a blocked matrix into TorchLean's input-by-output matrix convention. -/
noncomputable def decode {Code : Type} {inputBlocks outputDim : Nat}
    (decodeElement : Code → ℝ) (matrix : BlockMatrix Code inputBlocks outputDim) :
    Tensor ℝ (.dim (inputBlocks * 32) (.dim outputDim .scalar)) :=
  Tensor.dim fun input =>
    let position := finProdFinEquiv.symm input
    Tensor.dim fun output =>
      Tensor.scalar ((matrix.block output position.1).decode decodeElement position.2)

/-- Every decoded MXFP4 matrix coordinate is bounded by six times the scale of its block. -/
theorem abs_decode_mxfp4_le {inputBlocks outputDim : Nat}
    (matrix : BlockMatrix (Fin 16) inputBlocks outputDim)
    (input : Fin (inputBlocks * 32)) (output : Fin outputDim) :
    |get2 (matrix.decode decodeE2M1) input output| ≤
      6 * scaleValue (matrix.block output (finProdFinEquiv.symm input).1).scale := by
  obtain ⟨⟨inputBlock, offset⟩, rfl⟩ := finProdFinEquiv.surjective input
  have hsplit := Equiv.symm_apply_apply finProdFinEquiv (inputBlock, offset)
  have hblock : (finProdFinEquiv (inputBlock, offset)).divNat = inputBlock :=
    congrArg Prod.fst hsplit
  have hoffset : (finProdFinEquiv (inputBlock, offset)).modNat = offset :=
    congrArg Prod.snd hsplit
  simpa [decode, Spec.get2, Spec.get, Spec.get, hblock, hoffset] using
    abs_mxfp4_decode_le (matrix.block output inputBlock) offset

end BlockMatrix

/-- An MXFP4 matrix with block scaling along its input axis. -/
abbrev MXFP4Matrix (inputBlocks outputDim : Nat) := BlockMatrix (Fin 16) inputBlocks outputDim

/-- Quantize a matrix using an explicitly supplied E8M0 scale for every output/block pair. -/
noncomputable def quantizeMXFP4Matrix {inputBlocks outputDim : Nat}
    (scale : Fin outputDim → Fin inputBlocks → E8M0)
    (values : Tensor ℝ (.dim (inputBlocks * 32) (.dim outputDim .scalar))) :
    MXFP4Matrix inputBlocks outputDim where
  block output inputBlock := quantizeMXFP4Block (scale output inputBlock) fun offset =>
    get2 values (finProdFinEquiv (inputBlock, offset)) output

/-- Matrix quantization inherits the scalar block bound at the corresponding output and input
block. This theorem is the local weight-error contract used by quantized expert layers. -/
theorem quantizeMXFP4Matrix_error_le {inputBlocks outputDim : Nat}
    (scale : Fin outputDim → Fin inputBlocks → E8M0)
    (values : Tensor ℝ (.dim (inputBlocks * 32) (.dim outputDim .scalar)))
    (hRange : ∀ output inputBlock offset,
      |get2 values (finProdFinEquiv (inputBlock, offset)) output| ≤
        6 * scaleValue (scale output inputBlock))
    (input : Fin (inputBlocks * 32)) (output : Fin outputDim) :
    |get2 ((quantizeMXFP4Matrix scale values).decode decodeE2M1) input output -
        get2 values input output| ≤
      scaleValue (scale output (finProdFinEquiv.symm input).1) := by
  obtain ⟨⟨inputBlock, offset⟩, rfl⟩ := finProdFinEquiv.surjective input
  have hsplit := Equiv.symm_apply_apply finProdFinEquiv (inputBlock, offset)
  have hblock : (finProdFinEquiv (inputBlock, offset)).divNat = inputBlock :=
    congrArg Prod.fst hsplit
  have hoffset : (finProdFinEquiv (inputBlock, offset)).modNat = offset :=
    congrArg Prod.snd hsplit
  simpa [BlockMatrix.decode, quantizeMXFP4Matrix, Spec.get2, Spec.get,
    Spec.get, hblock, hoffset] using
    quantizeMXFP4Block_error_le (scale output inputBlock)
      (fun coordinate => get2 values (finProdFinEquiv (inputBlock, coordinate)) output)
      (hRange output inputBlock) offset

/-- Quantizing a matrix perturbs one output coordinate by at most the input-weighted sum of the
shared scales of the blocks contributing to that coordinate. Unlike a dimension-only estimate,
this bound retains the actual activation magnitudes and the scale chosen for every matrix block. -/
theorem vecMatMul_quantizeMXFP4Matrix_error_le {inputBlocks outputDim : Nat}
    (scale : Fin outputDim → Fin inputBlocks → E8M0)
    (weights : Tensor ℝ (.dim (inputBlocks * 32) (.dim outputDim .scalar)))
    (hRange : ∀ output inputBlock offset,
      |get2 weights (finProdFinEquiv (inputBlock, offset)) output| ≤
        6 * scaleValue (scale output inputBlock))
    (input : Tensor ℝ (.dim (inputBlocks * 32) .scalar)) (output : Fin outputDim) :
    |Tensor.getScalar (vecMatMulSpec input ((quantizeMXFP4Matrix scale weights).decode decodeE2M1)) output -
        Tensor.getScalar (vecMatMulSpec input weights) output| ≤
      ∑ index : Fin (inputBlocks * 32),
        |Tensor.getScalar input index| * scaleValue (scale output (finProdFinEquiv.symm index).1) := by
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
        |Tensor.getScalar input index| * scaleValue (scale output (finProdFinEquiv.symm index).1) := by
      apply Finset.sum_le_sum
      intro index _
      exact mul_le_mul_of_nonneg_left
        (quantizeMXFP4Matrix_error_le scale weights hRange index output) (abs_nonneg _)

/-- Combined MXFP8-activation and MXFP4-weight error for one matrix-product coordinate.

The first sum term is caused by activation encoding and uses the magnitude bound on the decoded
MXFP4 weight. The second is caused by E2M1 weight rounding and uses the original activation. This
separation is useful in QAT because the two errors are controlled by different scale policies.
-/
theorem vecMatMul_mxfp8_mxfp4_error_le {format : FP8Format}
    {inputBlocks outputDim : Nat}
    {sourceInput : Tensor ℝ (.dim (inputBlocks * 32) .scalar)}
    (input : MXFP8Encoding format inputBlocks sourceInput)
    (scale : Fin outputDim → Fin inputBlocks → E8M0)
    (weights : Tensor ℝ (.dim (inputBlocks * 32) (.dim outputDim .scalar)))
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
