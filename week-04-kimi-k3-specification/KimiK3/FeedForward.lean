/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Common
public import KimiK3.Microscaling
public import NN.Spec.Layers.Activation
public import Mathlib.Analysis.SpecialFunctions.Sigmoid
import NN.Proofs.Gradients.Activation

/-!
# Stable LatentMoE feed-forward semantics

This module formalizes the channel-mixing path in Section 2.3 (pp. 6--8) and Appendix B of the
Kimi K3 technical report. A `Route` reuses `Function.Embedding` to represent a fixed-cardinality
selection without duplicate experts. For real-valued model semantics, `Route.chooseTopK` computes
that selection from adjusted router scores and breaks exact ties by expert index.

The main equations represented here are:

* SiTU-GLU with independently capped gate and up branches (Eq. 12 and Appendix B);
* latent down-projection, routed expert aggregation, RMS normalization, and up-projection (Eq. 11);
* sigmoid router scores, bias-adjusted top-k selection, and unbiased routing weights (Eq. 13).

Reference: Kimi Team, "Kimi K3: Open Frontier Intelligence", 2026, Section 2.3, pp. 6--8,
Eqs. 11--13, and Appendix B, p. 43,
https://arxiv.org/abs/2607.24653.
-/

@[expose] public section

namespace KimiK3

open Spec TorchLean
open Tensor

namespace SiTU

/-- Smoothly cap a scalar at magnitude `cap` using `cap * tanh (x / cap)`. -/
def softCap {α : Type} [Context α] (cap x : α) : α :=
  cap * MathFunctions.tanh (x / cap)

/--
The SiTU-GLU scalar response.  The sigmoid is applied to the uncapped gate preactivation, while
the linear factors in both GLU branches are smoothly capped.
-/
def scalar {α : Type} [Context α] (gateCap upCap gate up : α) : α :=
  (softCap gateCap gate * Activation.Math.sigmoidSpec gate) * softCap upCap up

/-- SiTU-GLU applied coordinatewise to gate and up projections. -/
def vector {α : Type} [Storage α] [Context α] {n : Nat} (gateCap upCap : α)
    (gate up : Tensor α (.dim n .scalar)) : Tensor α (.dim n .scalar) :=
  Tensor.dim (fun i => Tensor.scalar (scalar gateCap upCap (Tensor.getScalar gate i)
    (Tensor.getScalar up i)))

/-- The generic TorchLean sigmoid agrees with Mathlib's real sigmoid. -/
private theorem sigmoidSpec_real (x : ℝ) : Activation.Math.sigmoidSpec x = Real.sigmoid x := by
  simpa [Real.sigmoid] using Proofs.sigmoid_eq_inv_exp x

/-- The generic `MathFunctions` projection is definitionally real `tanh` at scalar type `ℝ`. -/
private theorem tanh_real (x : ℝ) : MathFunctions.tanh x = Real.tanh x := by
  rfl

/--
SiTU-GLU has the paper's advertised uniform bound.  Unlike a numerical test, this statement holds
for every pair of real preactivations.
-/
theorem abs_scalar_le_mul_caps {gateCap upCap : ℝ}
    (hGate : 0 ≤ gateCap) (hUp : 0 ≤ upCap) (gate up : ℝ) :
    |scalar gateCap upCap gate up| ≤ gateCap * upCap := by
  have hTanhGate : |Real.tanh (gate / gateCap)| ≤ 1 := (Real.abs_tanh_lt_one _).le
  have hTanhUp : |Real.tanh (up / upCap)| ≤ 1 := (Real.abs_tanh_lt_one _).le
  have hSigmoid : |Real.sigmoid gate| ≤ 1 := by
    rw [abs_of_nonneg (Real.sigmoid_nonneg gate)]
    exact Real.sigmoid_le_one gate
  have hSoftGate : |softCap gateCap gate| ≤ gateCap := by
    rw [softCap, tanh_real, abs_mul, abs_of_nonneg hGate]
    exact mul_le_of_le_one_right hGate hTanhGate
  have hSoftUp : |softCap upCap up| ≤ upCap := by
    rw [softCap, tanh_real, abs_mul, abs_of_nonneg hUp]
    exact mul_le_of_le_one_right hUp hTanhUp
  rw [scalar, sigmoidSpec_real, abs_mul, abs_mul]
  calc
    |softCap gateCap gate| * |Real.sigmoid gate| * |softCap upCap up|
        ≤ gateCap * 1 * upCap := by gcongr
    _ = gateCap * upCap := by ring

/-- Kimi K3's caps imply the concrete coordinate bound `100`. -/
theorem paper_caps_bound (gate up : ℝ) : |scalar 4 25 gate up| ≤ 100 := by
  have h := abs_scalar_le_mul_caps (gateCap := (4 : ℝ)) (upCap := (25 : ℝ))
    (by positivity) (by positivity) gate up
  norm_num at h ⊢
  exact h

/-- Expanding the capped gate into elementary tensor operations preserves the SiTU equation.

The executable graph represents division by a cap as multiplication by its reciprocal. The
identity remains valid when a cap is zero because division in a field is total and satisfies
`x / c = x * c⁻¹`.
-/
theorem expanded_eq_vector {n : Nat} (gateCap upCap : ℝ)
    (gate up : Tensor ℝ (.dim n .scalar)) :
    Tensor.mulSpec
      (Tensor.mulSpec
        (Tensor.mulSpec
          (Activation.tanhSpec (Tensor.mulSpec gate (Tensor.full [n] gateCap⁻¹)))
          (Tensor.full [n] gateCap))
        (Activation.sigmoidSpec gate))
      (Tensor.mulSpec
        (Activation.tanhSpec (Tensor.mulSpec up (Tensor.full [n] upCap⁻¹)))
        (Tensor.full [n] upCap)) =
      vector gateCap upCap gate up := by
  apply Tensor.ext_vector
  intro index
  simp [Tensor.mulSpec, Activation.tanhSpec, Activation.sigmoidSpec,
    Activation.Math.tanhSpec, vector, scalar, softCap, div_eq_mul_inv]
  ring

end SiTU

/-- A SiTU-GLU feed-forward expert with explicit input, hidden, and output widths. -/
structure Expert (α : Type) [Storage α] (inputDim hiddenDim outputDim : Nat) where
  /-- Projection from the input into the SiTU gate branch. -/
  gateWeight : Tensor α (.dim inputDim (.dim hiddenDim .scalar))
  /-- Projection from the input into the SiTU up branch. -/
  upWeight : Tensor α (.dim inputDim (.dim hiddenDim .scalar))
  /-- Projection from the gated hidden activation to the output. -/
  downWeight : Tensor α (.dim hiddenDim (.dim outputDim .scalar))

namespace Expert

variable {α : Type} [Storage α] [Context α]
variable {inputDim hiddenDim outputDim : Nat}

/-- Evaluate one SiTU-GLU expert. -/
def forward (expert : Expert α inputDim hiddenDim outputDim) (gateCap upCap : α)
    (x : Tensor α (.dim inputDim .scalar)) : Tensor α (.dim outputDim .scalar) :=
  let gate := vecMatMulSpec x expert.gateWeight
  let up := vecMatMulSpec x expert.upWeight
  vecMatMulSpec (SiTU.vector gateCap upCap gate up) expert.downWeight

end Expert

/-! ## Microscaled routed experts -/

/-- The three MXFP4 matrices of one routed SiTU-GLU expert.

The input and hidden dimensions are expressed as numbers of 32-coordinate MX blocks. This is not
padding metadata: it states that every matrix axis consumed by an MX kernel is exactly partitioned
into complete OCP blocks. K3's latent width `3584` and routed hidden width `3072` are respectively
`112 * 32` and `96 * 32`.
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

/-- Real denotation of an MXFP4 expert. The definition deliberately goes through `Expert.forward`,
so subsequent model proofs use the same SiTU-GLU equation as an unquantized routed expert. -/
noncomputable def forward (expert : MXFP4Expert inputBlocks hiddenBlocks outputDim)
    (gateCap upCap : ℝ) (input : Tensor ℝ (.dim (inputBlocks * 32) .scalar)) :
    Tensor ℝ (.dim outputDim .scalar) :=
  expert.decode.forward gateCap upCap input

/-- The E8M0 scales used when packing the three matrices of an expert. -/
structure Scales (inputBlocks hiddenBlocks outputDim : Nat) where
  /-- Block scale for each gate-projection output and input block. -/
  gate : Fin (hiddenBlocks * 32) → Fin inputBlocks → E8M0
  /-- Block scale for each up-projection output and input block. -/
  up : Fin (hiddenBlocks * 32) → Fin inputBlocks → E8M0
  /-- Block scale for each down-projection output and hidden block. -/
  down : Fin outputDim → Fin hiddenBlocks → E8M0

/-- Pack a real-valued expert into MXFP4 using explicitly supplied block scales. -/
noncomputable def quantize
    (scales : Scales inputBlocks hiddenBlocks outputDim)
    (expert : Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim) :
    MXFP4Expert inputBlocks hiddenBlocks outputDim where
  gateWeight := quantizeMXFP4Matrix scales.gate expert.gateWeight
  upWeight := quantizeMXFP4Matrix scales.up expert.upWeight
  downWeight := quantizeMXFP4Matrix scales.down expert.downWeight

/-- Error of the quantized gate projection at one hidden coordinate. -/
theorem gateProjection_error_le
    (scales : Scales inputBlocks hiddenBlocks outputDim)
    (expert : Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim)
    (hCover : ∀ output inputBlock offset,
      |get2 expert.gateWeight (finProdFinEquiv (inputBlock, offset)) output| ≤
        6 * scaleValue (scales.gate output inputBlock))
    (input : Tensor ℝ (.dim (inputBlocks * 32) .scalar))
    (output : Fin (hiddenBlocks * 32)) :
    |Tensor.getScalar (vecMatMulSpec input (quantize scales expert).decode.gateWeight) output -
        Tensor.getScalar (vecMatMulSpec input expert.gateWeight) output| ≤
      ∑ index : Fin (inputBlocks * 32),
        |Tensor.getScalar input index| *
          scaleValue (scales.gate output (finProdFinEquiv.symm index).1) := by
  exact vecMatMul_quantizeMXFP4Matrix_error_le scales.gate expert.gateWeight
    hCover input output

/-- Error of the quantized up projection at one hidden coordinate. -/
theorem upProjection_error_le
    (scales : Scales inputBlocks hiddenBlocks outputDim)
    (expert : Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim)
    (hCover : ∀ output inputBlock offset,
      |get2 expert.upWeight (finProdFinEquiv (inputBlock, offset)) output| ≤
        6 * scaleValue (scales.up output inputBlock))
    (input : Tensor ℝ (.dim (inputBlocks * 32) .scalar))
    (output : Fin (hiddenBlocks * 32)) :
    |Tensor.getScalar (vecMatMulSpec input (quantize scales expert).decode.upWeight) output -
        Tensor.getScalar (vecMatMulSpec input expert.upWeight) output| ≤
      ∑ index : Fin (inputBlocks * 32),
        |Tensor.getScalar input index| *
          scaleValue (scales.up output (finProdFinEquiv.symm index).1) := by
  exact vecMatMul_quantizeMXFP4Matrix_error_le scales.up expert.upWeight
    hCover input output

/-- Error of the quantized down projection at one output coordinate. The input here is the
post-SiTU activation, so the theorem is also the contract needed for the second routed-expert
matrix multiplication. -/
theorem downProjection_error_le
    (scales : Scales inputBlocks hiddenBlocks outputDim)
    (expert : Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim)
    (hCover : ∀ output hiddenBlock offset,
      |get2 expert.downWeight (finProdFinEquiv (hiddenBlock, offset)) output| ≤
        6 * scaleValue (scales.down output hiddenBlock))
    (hidden : Tensor ℝ (.dim (hiddenBlocks * 32) .scalar))
    (output : Fin outputDim) :
    |Tensor.getScalar (vecMatMulSpec hidden (quantize scales expert).decode.downWeight) output -
        Tensor.getScalar (vecMatMulSpec hidden expert.downWeight) output| ≤
      ∑ index : Fin (hiddenBlocks * 32),
        |Tensor.getScalar hidden index| *
          scaleValue (scales.down output (finProdFinEquiv.symm index).1) := by
  exact vecMatMul_quantizeMXFP4Matrix_error_le scales.down expert.downWeight
    hCover hidden output

/-- Hidden SiTU activation produced when the first two expert matrices use MXFP8 inputs and MXFP4
weights. The result is still real-valued; an MXFP8 encoder must certify the representation supplied
to the final down projection.

The local irreducibility attribute below keeps this expression folded when constructing dependent
encoding witnesses, without changing its reducibility for callers. -/
noncomputable def hiddenActivation {format : FP8Format}
    (scales : Scales inputBlocks hiddenBlocks outputDim)
    (expert : Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim)
    (gateCap upCap : ℝ)
    {sourceInput : Tensor ℝ (.dim (inputBlocks * 32) .scalar)}
    (input : MXFP8Encoding format inputBlocks sourceInput) :
    Tensor ℝ (.dim (hiddenBlocks * 32) .scalar) :=
  let packed := quantize scales expert
  let gate := vecMatMulSpec input.decode packed.decode.gateWeight
  let up := vecMatMulSpec input.decode packed.decode.upWeight
  SiTU.vector gateCap upCap gate up

-- Certificate types refer to this activation without evaluating its quantized matrix products.
attribute [local irreducible] hiddenActivation

/-- A complete microscaled execution witness for one routed expert.

`input` certifies the MXFP8 values consumed by the gate and up matrix multiplications. `hidden`
certifies the separately encoded SiTU result consumed by the down multiplication. Consequently all
three routed-expert GEMMs use the MXFP8-by-MXFP4 arithmetic assignment stated in the K3 report.
-/
structure Execution (format : FP8Format)
    (scales : Scales inputBlocks hiddenBlocks outputDim)
    (expert : Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim)
    (gateCap upCap : ℝ)
    (sourceInput : Tensor ℝ (.dim (inputBlocks * 32) .scalar)) where
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
    {sourceInput : Tensor ℝ (.dim (inputBlocks * 32) .scalar)}
    (execution : Execution format scales expert gateCap upCap sourceInput) :
    Tensor ℝ (.dim outputDim .scalar) :=
  vecMatMulSpec execution.hidden.decode (quantize scales expert).decode.downWeight

/-- The final MXFP8-by-MXFP4 multiplication has a concrete coordinatewise error bound relative to
the real down projection of the pre-encoding hidden activation. -/
theorem output_error_le {format : FP8Format}
    {scales : Scales inputBlocks hiddenBlocks outputDim}
    {expert : Expert ℝ (inputBlocks * 32) (hiddenBlocks * 32) outputDim}
    (hCover : ∀ output hiddenBlock offset,
      |get2 expert.downWeight (finProdFinEquiv (hiddenBlock, offset)) output| ≤
        6 * scaleValue (scales.down output hiddenBlock))
    {gateCap upCap : ℝ}
    {sourceInput : Tensor ℝ (.dim (inputBlocks * 32) .scalar)}
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

/-- A route is an embedding of active slots into experts, so no expert can be selected twice. -/
abbrev Route (numExperts activeExperts : Nat) := Fin activeExperts ↪ Fin numExperts

namespace Route

variable {numExperts activeExperts : Nat}

/-- Expert selected for a route slot. -/
abbrev expert (route : Route numExperts activeExperts) : Fin activeExperts → Fin numExperts :=
  route

/-- A selected expert has a score at least as large as every unselected expert. -/
def IsTopK (route : Route numExperts activeExperts) (score : Fin numExperts → ℝ) : Prop :=
  ∀ selected candidate,
    (∀ slot, route.expert slot ≠ candidate) →
      score (route.expert selected) ≥ score candidate

/-- Adding the same constant to every score does not change which routes satisfy top-k. -/
theorem isTopK_add_common_iff (route : Route numExperts activeExperts)
    (score : Fin numExperts → ℝ) (offset : ℝ) :
    route.IsTopK (fun expert => score expert + offset) ↔ route.IsTopK score := by
  constructor <;> intro h selected candidate hCandidate
  · have := h selected candidate hCandidate
    linarith
  · have := h selected candidate hCandidate
    linarith

namespace Internal

/-- Descending score order with the lower expert index breaking ties. -/
def betterEq (score : Fin numExperts → ℝ)
    (left right : Fin numExperts) : Prop :=
  score right < score left ∨ (score left = score right ∧ left.val ≤ right.val)

theorem betterEq_trans (score : Fin numExperts → ℝ) {a b c : Fin numExperts}
    (hab : betterEq score a b) (hbc : betterEq score b c) : betterEq score a c := by
  rcases hab with hab | ⟨habScore, habIndex⟩
  · rcases hbc with hbc | ⟨hbcScore, _⟩
    · left; linarith
    · left; linarith
  · rcases hbc with hbc | ⟨hbcScore, hbcIndex⟩
    · left; linarith
    · right
      constructor
      · linarith
      · omega

theorem betterEq_total (score : Fin numExperts → ℝ) (a b : Fin numExperts) :
    betterEq score a b ∨ betterEq score b a := by
  rcases lt_trichotomy (score a) (score b) with hab | hab | hab
  · exact Or.inr (Or.inl hab)
  · by_cases hIndex : a.val ≤ b.val
    · exact Or.inl (Or.inr ⟨hab, hIndex⟩)
    · exact Or.inr (Or.inr ⟨hab.symm, Nat.le_of_lt (Nat.lt_of_not_ge hIndex)⟩)
  · exact Or.inl (Or.inl hab)

theorem betterEq_score_ge (score : Fin numExperts → ℝ) {a b : Fin numExperts}
    (h : betterEq score a b) : score a ≥ score b := by
  rcases h with h | ⟨h, _⟩
  · exact h.le
  · exact h.symm.le

/-- All expert indices ordered by decreasing score, with lower indices winning exact ties. -/
noncomputable def orderedExperts (score : Fin numExperts → ℝ) : List (Fin numExperts) := by
  classical
  exact (List.finRange numExperts).mergeSort
    (fun left right => decide (betterEq score left right))

theorem orderedExperts_pairwise (score : Fin numExperts → ℝ) :
    (orderedExperts score).Pairwise (betterEq score) := by
  classical
  unfold orderedExperts
  simpa only [decide_eq_true_eq] using
    (List.pairwise_mergeSort
      (le := fun left right => decide (betterEq score left right))
      (fun a b c hab hbc => by
        simpa only [decide_eq_true_eq] using betterEq_trans score
          (by simpa only [decide_eq_true_eq] using hab)
          (by simpa only [decide_eq_true_eq] using hbc))
      (fun a b => by
        simp only [Bool.or_eq_true, decide_eq_true_eq]
        exact betterEq_total score a b)
      (List.finRange numExperts))

theorem orderedExperts_nodup (score : Fin numExperts → ℝ) :
    (orderedExperts score).Nodup := by
  classical
  unfold orderedExperts
  exact List.nodup_mergeSort.mpr (List.nodup_finRange numExperts)

theorem orderedExperts_length (score : Fin numExperts → ℝ) :
    (orderedExperts score).length = numExperts := by
  exact (List.mergeSort_perm (List.finRange numExperts) _).length_eq.trans
    List.length_finRange

theorem orderedExperts_mem (score : Fin numExperts → ℝ) (expert : Fin numExperts) :
    expert ∈ orderedExperts score := by
  exact (List.mergeSort_perm (List.finRange numExperts) _).mem_iff.mpr
    (List.mem_finRange expert)

end Internal

/-- Choose exactly `activeExperts` experts by adjusted score. The inequality makes impossible
routes unrepresentable, while index tie-breaking makes the mathematical result deterministic. -/
noncomputable def chooseTopK (score : Fin numExperts → ℝ)
    (hActive : activeExperts ≤ numExperts) : Route numExperts activeExperts := by
  let ordered := Internal.orderedExperts score
  let selected := ordered.take activeExperts
  have hLength : selected.length = activeExperts := by
    simp only [selected, List.length_take, ordered, Internal.orderedExperts_length]
    exact Nat.min_eq_left hActive
  have hNodup : selected.Nodup := (Internal.orderedExperts_nodup score).take
  refine
    { toFun := fun slot => selected.get (Fin.cast hLength.symm slot)
      inj' := ?_ }
  intro left right hEqual
  apply Fin.cast_inj hLength.symm |>.mp
  exact hNodup.get_inj_iff.mp hEqual

namespace Internal

theorem chooseTopK_expert_mem_take (score : Fin numExperts → ℝ)
    (hActive : activeExperts ≤ numExperts) (slot : Fin activeExperts) :
    (chooseTopK score hActive).expert slot ∈ (orderedExperts score).take activeExperts := by
  simp only [chooseTopK]
  exact List.get_mem _ _

theorem mem_drop_of_not_selected (score : Fin numExperts → ℝ)
    (hActive : activeExperts ≤ numExperts) (candidate : Fin numExperts)
    (hCandidate : ∀ slot, (chooseTopK score hActive).expert slot ≠ candidate) :
    candidate ∈ (orderedExperts score).drop activeExperts := by
  have hMem := orderedExperts_mem score candidate
  rw [← List.take_append_drop activeExperts (orderedExperts score)] at hMem
  rcases List.mem_append.mp hMem with hTake | hDrop
  · obtain ⟨index, hIndex⟩ := List.get_of_mem hTake
    have hLength : ((orderedExperts score).take activeExperts).length = activeExperts := by
      simp [orderedExperts_length, Nat.min_eq_left hActive]
    let slot : Fin activeExperts := Fin.cast hLength index
    exfalso
    apply hCandidate slot
    simp only [chooseTopK]
    change ((orderedExperts score).take activeExperts).get
      (Fin.cast hLength.symm slot) = candidate
    simpa [slot] using hIndex
  · exact hDrop

end Internal

/-- The deterministic selector satisfies the mathematical top-k contract. -/
theorem chooseTopK_isTopK (score : Fin numExperts → ℝ)
    (hActive : activeExperts ≤ numExperts) :
    (chooseTopK score hActive).IsTopK score := by
  intro selectedSlot candidate hCandidate
  let ordered := Internal.orderedExperts score
  have hPairwise : ordered.Pairwise (Internal.betterEq score) :=
    Internal.orderedExperts_pairwise score
  have hAppend : ordered.take activeExperts ++ ordered.drop activeExperts = ordered :=
    List.take_append_drop activeExperts ordered
  have hCross : ∀ a ∈ ordered.take activeExperts, ∀ b ∈ ordered.drop activeExperts,
      Internal.betterEq score a b := by
    rw [← hAppend] at hPairwise
    exact (List.pairwise_append.mp hPairwise).2.2
  apply Internal.betterEq_score_ge score
  exact hCross ((chooseTopK score hActive).expert selectedSlot)
    (Internal.chooseTopK_expert_mem_take score hActive selectedSlot) candidate
    (Internal.mem_drop_of_not_selected score hActive candidate hCandidate)
end Route

/-- Parameters for Kimi K3's Stable LatentMoE layer. -/
structure StableLatentMoE (α : Type) [Storage α]
    (modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat) where
  /-- Projection from model width into the routed-expert latent space. -/
  downProject : Tensor α (.dim modelDim (.dim latentDim .scalar))
  /-- Projection from routed-expert latent space back to model width. -/
  upProject : Tensor α (.dim latentDim (.dim modelDim .scalar))
  /-- RMSNorm scale applied to the routed latent aggregate. -/
  routedNormScale : Tensor α (.dim latentDim .scalar)
  /-- Projection producing one raw score per routed expert. -/
  routerWeight : Tensor α (.dim modelDim (.dim numRouted .scalar))
  /-- Bias used to select experts without changing their mixture weights. -/
  routerBias : Tensor α (.dim numRouted .scalar)
  /-- Full-width shared experts evaluated for every token. -/
  shared : Fin numShared → Expert α modelDim sharedHidden modelDim
  /-- Latent-width experts selected by the router. -/
  routed : Fin numRouted → Expert α latentDim routedHidden latentDim

namespace StableLatentMoE

variable {α : Type} [Storage α] [Context α]
variable {modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat}

/-- Raw router scores are sigmoid outputs (Eq. 13). -/
def rawRouterScores
    (moe : StableLatentMoE α modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (x : Tensor α (.dim modelDim .scalar)) : Tensor α (.dim numRouted .scalar) :=
  Activation.sigmoidSpec (vecMatMulSpec x moe.routerWeight)

/-- Every raw routing score is strictly positive. K3 applies the logistic sigmoid before expert
selection, so the normalization denominator cannot vanish as soon as at least one expert is
selected. -/
theorem rawRouterScores_pos
    (moe : StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (x : Tensor ℝ (.dim modelDim .scalar)) (expert : Fin numRouted) :
    0 < Tensor.getScalar (moe.rawRouterScores x) expert := by
  rw [rawRouterScores, Activation.sigmoidSpec, Normalize.getScalar_mapSpec,
    Proofs.sigmoid_eq_inv_exp]
  positivity

/-- Bias is used for top-k selection but not for the mixture weights (Eq. 13). -/
def adjustedRouterScores
    (moe : StableLatentMoE α modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (x : Tensor α (.dim modelDim .scalar)) : Tensor α (.dim numRouted .scalar) :=
  moe.rawRouterScores x + moe.routerBias

/-- Extract the raw scores of the selected experts and normalize them to obtain `p_i`. -/
def routeWeights
    (moe : StableLatentMoE α modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (route : Route numRouted activeExperts)
    (x : Tensor α (.dim modelDim .scalar)) : Tensor α (.dim activeExperts .scalar) :=
  let raw := moe.rawRouterScores x
  Normalize.probabilities <|
    Tensor.dim (fun slot => Tensor.scalar (Tensor.getScalar raw (route.expert slot)))

/-- The selected raw routing scores have positive total mass for every nonempty route. -/
theorem selectedRouterScoreTotal_pos
    (moe : StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (route : Route numRouted activeExperts) (x : Tensor ℝ (.dim modelDim .scalar))
    (hActive : 0 < activeExperts) :
    0 < Tensor.sumSpec
    (Tensor.dim fun slot => Tensor.scalar
        (Tensor.getScalar (moe.rawRouterScores x) (route.expert slot))) := by
  let _ : Nonempty (Fin activeExperts) := Fin.pos_iff_nonempty.mp hActive
  rw [Spec.sum_spec_vec]
  simp only [Tensor.getScalar_dim]
  exact Finset.sum_pos
    (fun slot _ => moe.rawRouterScores_pos x (route.expert slot))
    Finset.univ_nonempty

/-- For a nonempty route, K3's guarded probability normalization is ordinary division by the
strictly positive sum of the selected raw scores. -/
theorem routeWeights_apply
    (moe : StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (route : Route numRouted activeExperts) (x : Tensor ℝ (.dim modelDim .scalar))
    (hActive : 0 < activeExperts) (slot : Fin activeExperts) :
    Tensor.getScalar (moe.routeWeights route x) slot =
      Tensor.getScalar (moe.rawRouterScores x) (route.expert slot) /
        Tensor.sumSpec
          (Tensor.dim fun selected => Tensor.scalar
            (Tensor.getScalar (moe.rawRouterScores x) (route.expert selected))) := by
  rw [routeWeights, Normalize.probabilities,
    ite_eq_left (moe.selectedRouterScoreTotal_pos route x hActive)]
  simp only [Tensor.getScalar_dim]

/-- Routing weights of a nonempty route sum to one. -/
theorem routeWeights_sum_eq_one
    (moe : StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (route : Route numRouted activeExperts) (x : Tensor ℝ (.dim modelDim .scalar))
    (hActive : 0 < activeExperts) :
    Tensor.sumSpec (moe.routeWeights route x) = 1 := by
  rw [Spec.sum_spec_vec]
  simp_rw [moe.routeWeights_apply route x hActive]
  rw [← Finset.sum_div]
  have hTotal := moe.selectedRouterScoreTotal_pos route x hActive
  have hNumerator :
      (∑ slot, Tensor.getScalar (moe.rawRouterScores x) (route.expert slot)) =
        Tensor.sumSpec
          (Tensor.dim fun slot => Tensor.scalar
            (Tensor.getScalar (moe.rawRouterScores x) (route.expert slot))) := by
    rw [Spec.sum_spec_vec]
    simp only [Tensor.getScalar_dim]
  rw [hNumerator]
  exact div_self hTotal.ne'

/-- Sum all full-width shared experts. -/
def sharedOutput
    (moe : StableLatentMoE α modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (gateCap upCap : α) (x : Tensor α (.dim modelDim .scalar)) :
    Tensor α (.dim modelDim .scalar) :=
  (List.finRange numShared).foldl
    (fun total i => total + (moe.shared i).forward gateCap upCap x)
    (Tensor.zeros [modelDim])

/-- Weighted aggregate of the selected latent experts, the vector `u` in Eq. 11. -/
def routedAggregate
    (moe : StableLatentMoE α modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (route : Route numRouted activeExperts) (gateCap upCap : α)
    (x : Tensor α (.dim modelDim .scalar)) : Tensor α (.dim latentDim .scalar) :=
  let latent := vecMatMulSpec x moe.downProject
  let weights := moe.routeWeights route x
  (List.finRange activeExperts).foldl
    (fun total slot =>
      let expertOutput := (moe.routed (route.expert slot)).forward gateCap upCap latent
      total + Tensor.mapSpec (fun value => Tensor.getScalar weights slot * value) expertOutput)
    (Tensor.zeros [latentDim])

/--
Stable LatentMoE forward pass (Eq. 11): shared experts operate at model width, while the selected
routed experts operate in latent width and are normalized before the up-projection.
-/
def forward
    (moe : StableLatentMoE α modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (route : Route numRouted activeExperts) (gateCap upCap : α)
    (x : Tensor α (.dim modelDim .scalar)) : Tensor α (.dim modelDim .scalar) :=
  let shared := moe.sharedOutput gateCap upCap x
  let routed := moe.routedAggregate route gateCap upCap x
  shared + vecMatMulSpec (RMSNorm.scale routed moe.routedNormScale) moe.upProject

/-- Route a real-valued token using Eq. 13's bias-adjusted scores. -/
noncomputable def route
    (moe : StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (hActive : activeExperts ≤ numRouted)
    (x : Tensor ℝ (.dim modelDim .scalar)) : Route numRouted activeExperts :=
  Route.chooseTopK (fun expert => Tensor.getScalar (moe.adjustedRouterScores x) expert) hActive

/-- The route computed by a Stable LatentMoE layer is top-k for its adjusted router scores. -/
theorem route_isTopK
    (moe : StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (hActive : activeExperts ≤ numRouted)
    (x : Tensor ℝ (.dim modelDim .scalar)) :
    (moe.route hActive x).IsTopK
      (fun expert => Tensor.getScalar (moe.adjustedRouterScores x) expert) :=
  Route.chooseTopK_isTopK _ hActive

/-- Real-valued Stable LatentMoE semantics with routing computed from the layer's own scores. -/
noncomputable def forwardReal
    (moe : StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (hActive : activeExperts ≤ numRouted) (gateCap upCap : ℝ)
    (x : Tensor ℝ (.dim modelDim .scalar)) : Tensor ℝ (.dim modelDim .scalar) :=
  moe.forward (moe.route hActive x) gateCap upCap x

/-- Replace the real-valued routed experts by the denotations of packed MXFP4 experts. Shared
experts, router projections, latent projections, and normalization parameters remain unchanged,
matching the precision boundary stated in the K3 report. -/
noncomputable def withMXFP4Routed {latentBlocks hiddenBlocks : Nat}
    (moe : StableLatentMoE ℝ modelDim (latentBlocks * 32) sharedHidden (hiddenBlocks * 32)
      numShared numRouted activeExperts)
    (routed : Fin numRouted → MXFP4Expert latentBlocks hiddenBlocks (latentBlocks * 32)) :
    StableLatentMoE ℝ modelDim (latentBlocks * 32) sharedHidden (hiddenBlocks * 32)
      numShared numRouted activeExperts :=
  { moe with routed := fun expert => (routed expert).decode }

/-- Stable LatentMoE semantics whose selected routed experts are decoded from packed MXFP4
weights. Routing and mixture weights are still computed by the original high-precision fields. -/
noncomputable def forwardMXFP4 {latentBlocks hiddenBlocks : Nat}
    (moe : StableLatentMoE ℝ modelDim (latentBlocks * 32) sharedHidden (hiddenBlocks * 32)
      numShared numRouted activeExperts)
    (routed : Fin numRouted → MXFP4Expert latentBlocks hiddenBlocks (latentBlocks * 32))
    (hActive : activeExperts ≤ numRouted) (gateCap upCap : ℝ)
    (input : Tensor ℝ (.dim modelDim .scalar)) : Tensor ℝ (.dim modelDim .scalar) :=
  (moe.withMXFP4Routed routed).forwardReal hActive gateCap upCap input

/-- Certified microscaled executions of the experts selected for one token.

The router, low-rank latent projection, and mixture weights are evaluated by `moe` before this
witness is constructed. Each selected expert then certifies its own MXFP8 activation encodings and
MXFP4 block scales. The three coverage fields state directly that packing cannot saturate any
source weight.
-/
structure MXExecution {latentBlocks hiddenBlocks : Nat} (format : Microscaling.FP8Format)
    (moe : StableLatentMoE ℝ modelDim (latentBlocks * 32) sharedHidden (hiddenBlocks * 32)
      numShared numRouted activeExperts)
    (route : Route numRouted activeExperts) (gateCap upCap : ℝ)
    (input : Tensor ℝ (.dim modelDim .scalar)) where
  /-- Per-expert MXFP4 block scales. -/
  scales : Fin numRouted → MXFP4Expert.Scales latentBlocks hiddenBlocks (latentBlocks * 32)
  gateScaleCovers : ∀ expert output inputBlock offset,
    |get2 (moe.routed expert).gateWeight (finProdFinEquiv (inputBlock, offset)) output| ≤
      6 * Microscaling.scaleValue ((scales expert).gate output inputBlock)
  upScaleCovers : ∀ expert output inputBlock offset,
    |get2 (moe.routed expert).upWeight (finProdFinEquiv (inputBlock, offset)) output| ≤
      6 * Microscaling.scaleValue ((scales expert).up output inputBlock)
  downScaleCovers : ∀ expert output hiddenBlock offset,
    |get2 (moe.routed expert).downWeight (finProdFinEquiv (hiddenBlock, offset)) output| ≤
      6 * Microscaling.scaleValue ((scales expert).down output hiddenBlock)
  /-- Certified microscaled execution for each selected expert. -/
  selected : ∀ slot,
    MXFP4Expert.Execution format (scales (route.expert slot)) (moe.routed (route.expert slot))
      gateCap upCap (vecMatMulSpec input moe.downProject)

/-- Weighted latent output of the selected microscaled experts. -/
noncomputable def routedAggregateMX {latentBlocks hiddenBlocks : Nat}
    {format : Microscaling.FP8Format}
    (moe : StableLatentMoE ℝ modelDim (latentBlocks * 32) sharedHidden (hiddenBlocks * 32)
      numShared numRouted activeExperts)
    (route : Route numRouted activeExperts) (gateCap upCap : ℝ)
    (input : Tensor ℝ (.dim modelDim .scalar))
    (execution : MXExecution format moe route gateCap upCap input) :
    Tensor ℝ (.dim (latentBlocks * 32) .scalar) :=
  let weights := moe.routeWeights route input
  (List.finRange activeExperts).foldl
    (fun total slot =>
      let expertOutput := (execution.selected slot).output
      total + Tensor.mapSpec (fun value => Tensor.getScalar weights slot * value) expertOutput)
    (Tensor.zeros [latentBlocks * 32])

/-- Stable LatentMoE with the precision split used by K3 deployment: shared experts, routing,
latent projections, normalization, and the final up-projection remain high precision, while every
selected routed expert is evaluated through its certified MXFP8/MXFP4 execution witness. -/
noncomputable def forwardMX {latentBlocks hiddenBlocks : Nat}
    {format : Microscaling.FP8Format}
    (moe : StableLatentMoE ℝ modelDim (latentBlocks * 32) sharedHidden (hiddenBlocks * 32)
      numShared numRouted activeExperts)
    (route : Route numRouted activeExperts) (gateCap upCap : ℝ)
    (input : Tensor ℝ (.dim modelDim .scalar))
    (execution : MXExecution format moe route gateCap upCap input) :
    Tensor ℝ (.dim modelDim .scalar) :=
  let shared := moe.sharedOutput gateCap upCap input
  let routed := moe.routedAggregateMX route gateCap upCap input execution
  shared + vecMatMulSpec (RMSNorm.scale routed moe.routedNormScale) moe.upProject

end StableLatentMoE

end KimiK3
