/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.FeedForward.Activation
public import KimiK3.Microscaling

/-!
# Shared and routed experts

One SiTU expert equation serves both the full-width shared experts and the latent-width routed
experts. Quantized experts decode to that equation, and their activation encodings carry error
bounds.

Reference: https://arxiv.org/abs/2607.24653, Sections 2.3 and 4.1.4.
-/

@[expose] public section

namespace KimiK3

open Spec TorchLean
open Tensor

/-- A SiTU-GLU feed-forward expert, with its input, hidden, and output widths in the type. -/
structure Expert (α : Type) [Storage α] (inputDim hiddenDim outputDim : Nat) where
  /-- Projection from the input into the SiTU gate branch. -/
  gateWeight : Tensor α [inputDim, hiddenDim]
  /-- Projection from the input into the SiTU up branch. -/
  upWeight : Tensor α [inputDim, hiddenDim]
  /-- Projection from the gated hidden activation to the output. -/
  downWeight : Tensor α [hiddenDim, outputDim]

namespace Expert

variable {α : Type} [Storage α] [Context α]
variable {inputDim hiddenDim outputDim : Nat}

/-- Evaluate one SiTU-GLU expert. -/
def forward (expert : Expert α inputDim hiddenDim outputDim) (gateCap upCap : α)
    (x : Tensor α [inputDim]) : Tensor α [outputDim] :=
  let gate := vecMatMulSpec x expert.gateWeight
  let up := vecMatMulSpec x expert.upWeight
  vecMatMulSpec (SiTU.vector gateCap upCap gate up) expert.downWeight

end Expert

/-! ## Microscaled routed experts -/

/-- MXFP4 gate, up, and down matrices of one routed SiTU-GLU expert.

The input and hidden widths are counted in 32-coordinate MX blocks, so every matrix axis an MX
kernel consumes splits into whole OCP blocks. K3's latent width `3584` is `112 * 32` and its routed
hidden width `3072` is `96 * 32`.
-/
structure MXFP4Expert (inputBlocks hiddenBlocks outputDim : Nat) where
  /-- MXFP4-packed gate projection. -/
  gateWeight : Microscaling.MXFP4Matrix inputBlocks (hiddenBlocks * 32)
  /-- MXFP4-packed up projection. -/
  upWeight : Microscaling.MXFP4Matrix inputBlocks (hiddenBlocks * 32)
  /-- MXFP4-packed down projection. -/
  downWeight : Microscaling.MXFP4Matrix hiddenBlocks outputDim

namespace MXFP4Expert

open Microscaling

variable {inputBlocks hiddenBlocks outputDim : Nat}

/-- Decode the packed matrices into the real-valued expert specification used by TorchLean. -/
noncomputable def decode (expert : MXFP4Expert inputBlocks hiddenBlocks outputDim) :
    Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim where
  gateWeight := expert.gateWeight.decode decodeE2M1
  upWeight := expert.upWeight.decode decodeE2M1
  downWeight := expert.downWeight.decode decodeE2M1

/-- Real denotation of an MXFP4 expert: decode, then run `Expert.forward`, so model proofs see the
same SiTU-GLU equation as for an unquantized routed expert. -/
noncomputable def forward (expert : MXFP4Expert inputBlocks hiddenBlocks outputDim)
    (gateCap upCap : ℝ) (input : Tensor ℝ [inputBlocks * 32]) :
    Tensor ℝ [outputDim] :=
  expert.decode.forward gateCap upCap input

/-- E8M0 block scales for packing the three matrices of an expert. -/
structure Scales (inputBlocks hiddenBlocks outputDim : Nat) where
  /-- Block scale for each gate-projection output and input block. -/
  gate : Fin (hiddenBlocks * 32) → Fin inputBlocks → E8M0
  /-- Block scale for each up-projection output and input block. -/
  up : Fin (hiddenBlocks * 32) → Fin inputBlocks → E8M0
  /-- Block scale for each down-projection output and hidden block. -/
  down : Fin outputDim → Fin hiddenBlocks → E8M0

/-- Pack a real-valued expert into MXFP4 with the given block scales. -/
noncomputable def quantize
    (scales : Scales inputBlocks hiddenBlocks outputDim)
    (expert : Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim) :
    MXFP4Expert inputBlocks hiddenBlocks outputDim where
  gateWeight := quantizeMXFP4Matrix scales.gate expert.gateWeight
  upWeight := quantizeMXFP4Matrix scales.up expert.upWeight
  downWeight := quantizeMXFP4Matrix scales.down expert.downWeight

/-- Error of the quantized gate projection at one hidden coordinate, when every weight satisfies
`|w| ≤ 6 * scale` for its block scale. -/
theorem gateProjection_error_le
    (scales : Scales inputBlocks hiddenBlocks outputDim)
    (expert : Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim)
    (hCover : ∀ output inputBlock offset,
      |get2 expert.gateWeight (finProdFinEquiv (inputBlock, offset)) output| ≤
        6 * scaleValue (scales.gate output inputBlock))
    (input : Tensor ℝ [inputBlocks * 32])
    (output : Fin (hiddenBlocks * 32)) :
    |Tensor.getScalar (vecMatMulSpec input (quantize scales expert).decode.gateWeight) output -
        Tensor.getScalar (vecMatMulSpec input expert.gateWeight) output| ≤
      ∑ index : Fin (inputBlocks * 32),
        |Tensor.getScalar input index| *
          scaleValue (scales.gate output (finProdFinEquiv.symm index).1) := by
  exact vecMatMul_quantizeMXFP4Matrix_error_le scales.gate expert.gateWeight
    hCover input output

/-- Same bound for the up projection. -/
theorem upProjection_error_le
    (scales : Scales inputBlocks hiddenBlocks outputDim)
    (expert : Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim)
    (hCover : ∀ output inputBlock offset,
      |get2 expert.upWeight (finProdFinEquiv (inputBlock, offset)) output| ≤
        6 * scaleValue (scales.up output inputBlock))
    (input : Tensor ℝ [inputBlocks * 32])
    (output : Fin (hiddenBlocks * 32)) :
    |Tensor.getScalar (vecMatMulSpec input (quantize scales expert).decode.upWeight) output -
        Tensor.getScalar (vecMatMulSpec input expert.upWeight) output| ≤
      ∑ index : Fin (inputBlocks * 32),
        |Tensor.getScalar input index| *
          scaleValue (scales.up output (finProdFinEquiv.symm index).1) := by
  exact vecMatMul_quantizeMXFP4Matrix_error_le scales.up expert.upWeight
    hCover input output

/-- Same bound for the down projection, at one output coordinate. Its input is the post-SiTU
activation, so this is also the contract for the second routed-expert matrix multiplication. -/
theorem downProjection_error_le
    (scales : Scales inputBlocks hiddenBlocks outputDim)
    (expert : Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim)
    (hCover : ∀ output hiddenBlock offset,
      |get2 expert.downWeight (finProdFinEquiv (hiddenBlock, offset)) output| ≤
        6 * scaleValue (scales.down output hiddenBlock))
    (hidden : Tensor ℝ [hiddenBlocks * 32])
    (output : Fin outputDim) :
    |Tensor.getScalar (vecMatMulSpec hidden (quantize scales expert).decode.downWeight) output -
        Tensor.getScalar (vecMatMulSpec hidden expert.downWeight) output| ≤
      ∑ index : Fin (hiddenBlocks * 32),
        |Tensor.getScalar hidden index| *
          scaleValue (scales.down output (finProdFinEquiv.symm index).1) := by
  exact vecMatMul_quantizeMXFP4Matrix_error_le scales.down expert.downWeight
    hCover hidden output

/-- Hidden SiTU activation when the gate and up projections take MXFP8 inputs and MXFP4 weights.
The result is real-valued, so the down projection needs a separately certified MXFP8 encoding of
it. -/
noncomputable def hiddenActivation {format : FP8Format}
    (scales : Scales inputBlocks hiddenBlocks outputDim)
    (expert : Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim)
    (gateCap upCap : ℝ)
    {sourceInput : Tensor ℝ [inputBlocks * 32]}
    (input : MXFP8Encoding format inputBlocks sourceInput) :
    Tensor ℝ [hiddenBlocks * 32] :=
  let packed := quantize scales expert
  let gate := vecMatMulSpec input.decode packed.decode.gateWeight
  let up := vecMatMulSpec input.decode packed.decode.upWeight
  SiTU.vector gateCap upCap gate up

-- Keep this activation folded inside certificate types (callers are unaffected).
attribute [local irreducible] hiddenActivation

/-- Microscaled execution witness for one routed expert.

`input` certifies the MXFP8 values entering the gate and up projections, and `hidden` certifies the
separately encoded SiTU result entering the down projection. All three routed-expert GEMMs then run
MXFP8 by MXFP4, the assignment stated in the K3 report.
-/
structure Execution (format : FP8Format)
    (scales : Scales inputBlocks hiddenBlocks outputDim)
    (expert : Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim)
    (gateCap upCap : ℝ)
    (sourceInput : Tensor ℝ [inputBlocks * 32]) where
  /-- Certified MXFP8 encoding consumed by the gate and up projections. -/
  input : MXFP8Encoding format inputBlocks sourceInput
  /-- Certified MXFP8 encoding of the SiTU activation consumed by the down projection. -/
  hidden : MXFP8Encoding format hiddenBlocks
    (hiddenActivation (format := format) (sourceInput := sourceInput)
      scales expert gateCap upCap input)

namespace Execution

/-- Output of all three microscaled expert matrix multiplications. -/
noncomputable def output {format : FP8Format}
    {scales : Scales inputBlocks hiddenBlocks outputDim}
    {expert : Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim}
    {gateCap upCap : ℝ}
    {sourceInput : Tensor ℝ [inputBlocks * 32]}
    (execution : Execution format scales expert gateCap upCap sourceInput) :
    Tensor ℝ [outputDim] :=
  vecMatMulSpec execution.hidden.decode (quantize scales expert).decode.downWeight

/-- Coordinatewise error of the final MXFP8-by-MXFP4 multiplication against the real down
projection of the unencoded hidden activation. -/
theorem output_error_le {format : FP8Format}
    {scales : Scales inputBlocks hiddenBlocks outputDim}
    {expert : Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim}
    (hCover : ∀ output hiddenBlock offset,
      |get2 expert.downWeight (finProdFinEquiv (hiddenBlock, offset)) output| ≤
        6 * scaleValue (scales.down output hiddenBlock))
    {gateCap upCap : ℝ}
    {sourceInput : Tensor ℝ [inputBlocks * 32]}
    (execution : Execution format scales expert gateCap upCap sourceInput)
    (coordinate : Fin outputDim) :
    |Tensor.getScalar execution.output coordinate -
        Tensor.getScalar (vecMatMulSpec
          (hiddenActivation scales expert gateCap upCap execution.input)
          expert.downWeight) coordinate| ≤
      ∑ index : Fin (hiddenBlocks * 32),
        (Tensor.getScalar execution.hidden.errorBound index *
            (6 * scaleValue (scales.down coordinate (finProdFinEquiv.symm index).1)) +
          |Tensor.getScalar (hiddenActivation scales expert gateCap upCap execution.input) index| *
            scaleValue (scales.down coordinate (finProdFinEquiv.symm index).1)) := by
  exact vecMatMul_mxfp8_mxfp4_error_le execution.hidden scales.down expert.downWeight
    hCover coordinate

end Execution

end MXFP4Expert

end KimiK3
