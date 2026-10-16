/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Nemotron.Cache.Error
public import NN.Spec.Core.Sequence
public import NN.Spec.Core.TensorOps
public import NN.Spec.Core.Tensor.Linalg
public import NN.Tensor.Operations
public import Mathlib.Data.List.GetD

import NN.Proofs.Tensor.Algebra

/-!
# A prepared single-head SSM core

The Ultra report names Mamba-2 in §2.1 and Table 1 (pp. 3–4), but §4.5 does not give the full
selective SSM equations. Here `retention`, `write`, `value`, and `read` arrive already projected
and discretized, and the recurrence is `S' = a S + x Bᵀ`, `y = S' C`. This is the
scalar-transition core of one head, with the matrix transposed so each row belongs to a value
channel. Convolution, groups, gating, normalization, and the output skip are left out, and
nothing here builds the coefficients from a Nemotron checkpoint (`Nemotron.Architecture.Mamba2`
adds grouping and the D skip).
-/

@[expose] public section

namespace Nemotron.SSM

open Spec TorchLean KimiK3

/-- State of a single head, with one row per value channel and one column per SSM coordinate. -/
abbrev State (α : Type) [Storage α] (valueDim stateDim : Nat) :=
  Tensor α [valueDim, stateDim]

/-- Prepared coefficients for one token, after projection and discretization. -/
structure Input (α : Type) [Storage α] (valueDim stateDim : Nat) where
  retention : α
  value : Tensor α [valueDim]
  write : Tensor α [stateDim]
  read : Tensor α [stateDim]

/-- One recurrent update `S' = a S + x Bᵀ`: scalar retention plus a rank-one write. -/
def update {α : Type} [Storage α] [Add α] [Mul α]
    {valueDim stateDim : Nat} (state : State α valueDim stateDim)
    (input : Input α valueDim stateDim) : State α valueDim stateDim :=
  Tensor.addSpec (Tensor.scaleSpec state input.retention)
    (Tensor.outerProductSpec input.value input.write)

/-- Update the state, then read the new state against the token's read vector. -/
def step {α : Type} [Storage α] [Add α] [Mul α] [Zero α]
    {valueDim stateDim : Nat} (state : State α valueDim stateDim)
    (input : Input α valueDim stateDim) :
    State α valueDim stateDim × Tensor α [valueDim] :=
  let next := update state input
  (next, Tensor.matVecMulSpec next input.read)

/-- Applying the same prepared input to two states cancels the rank-one write. -/
theorem update_sub_apply {valueDim stateDim : Nat}
    (x y : State ℝ valueDim stateDim) (input : Input ℝ valueDim stateDim)
    (coordinate : Shape.Coord [valueDim, stateDim]) :
    update x input coordinate - update y input coordinate =
      input.retention * (x coordinate - y coordinate) := by
  simp [update, Tensor.addSpec, Tensor.scaleSpec, Tensor.map2Spec, Tensor.map]
  ring

/-- If `|retention| ≤ L`, one update scales a coordinate error bound by `L`, for any dimensions. -/
theorem update_within {valueDim stateDim : Nat}
    (x y : State ℝ valueDim stateDim) (input : Input ℝ valueDim stateDim)
    (L e : ℝ) (hL : 0 ≤ L) (hretention : |input.retention| ≤ L)
    (hxy : Cache.Within e x y) : Cache.Within (L * e) (update x input) (update y input) := by
  intro coordinate
  rw [update_sub_apply, abs_mul]
  exact mul_le_mul hretention (hxy coordinate) (abs_nonneg _) hL

/-- A fixed read vector turns a coordinatewise state error into an output error, scaled by the
read vector's absolute sum. So a stable state alone does not give a small readout error. -/
theorem read_within {valueDim stateDim : Nat}
    (x y : State ℝ valueDim stateDim) (read : Tensor ℝ [stateDim])
    (e : ℝ) (hxy : Cache.Within e x y) :
    Cache.Within (e * ∑ j : Fin stateDim, |read.getScalar j|)
      (Tensor.matVecMulSpec x read) (Tensor.matVecMulSpec y read) := by
  rintro ⟨i, rest⟩
  cases rest
  change |(Tensor.matVecMulSpec x read).getScalar i -
    (Tensor.matVecMulSpec y read).getScalar i| ≤ _
  rw [Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec,
    Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec, ← Finset.sum_sub_distrib,
    Finset.mul_sum]
  calc
    |∑ j : Fin stateDim, (get2 x i j * read.getScalar j - get2 y i j * read.getScalar j)|
        ≤ ∑ j : Fin stateDim,
            |get2 x i j * read.getScalar j - get2 y i j * read.getScalar j| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j : Fin stateDim, e * |read.getScalar j| := by
      apply Finset.sum_le_sum
      intro j _
      rw [← sub_mul, abs_mul]
      apply mul_le_mul_of_nonneg_right _ (abs_nonneg _)
      simpa [get2, Spec.get, Tensor.getScalar, Tensor.unstack, Tensor.item] using
        hxy ⟨i, j, ()⟩

/-- Output error after one step: the state bound `L * e` times the read vector's absolute sum. -/
theorem step_output_within {valueDim stateDim : Nat}
    (x y : State ℝ valueDim stateDim) (input : Input ℝ valueDim stateDim)
    (L e : ℝ) (hL : 0 ≤ L) (hretention : |input.retention| ≤ L)
    (hxy : Cache.Within e x y) :
    Cache.Within ((L * e) * ∑ j : Fin stateDim, |input.read.getScalar j|)
      (step x input).2 (step y input).2 :=
  read_within _ _ input.read (L * e) (update_within x y input L e hL hretention hxy)

/-- Prepared inputs for a whole token sequence, one row per token. -/
structure Sequence (α : Type) [Storage α] (tokens valueDim stateDim : Nat) where
  retention : Tensor α [tokens]
  value : Tensor α [tokens, valueDim]
  write : Tensor α [tokens, stateDim]
  read : Tensor α [tokens, stateDim]

namespace Sequence

/-- Prepared input at token `i`. -/
def token {α : Type} [Storage α] {tokens valueDim stateDim : Nat}
    (inputs : Sequence α tokens valueDim stateDim) (i : Fin tokens) :
    Input α valueDim stateDim :=
  ⟨inputs.retention.getScalar i, Tensor.unstack inputs.value i,
    Tensor.unstack inputs.write i, Tensor.unstack inputs.read i⟩

/-- Tokens in chronological order, as a list for the shared causal scan. -/
def toList {α : Type} [Storage α] {tokens valueDim stateDim : Nat}
    (inputs : Sequence α tokens valueDim stateDim) : List (Input α valueDim stateDim) :=
  (List.finRange tokens).map inputs.token

@[simp] theorem toList_length {α : Type} [Storage α] {tokens valueDim stateDim : Nat}
    (inputs : Sequence α tokens valueDim stateDim) : inputs.toList.length = tokens := by
  simp [toList]

/-- Run the scan over a whole sequence and pack its outputs into a `[tokens, valueDim]` tensor. -/
def run {α : Type} [Storage α] [Add α] [Mul α] [Zero α]
    {tokens valueDim stateDim : Nat} (inputs : Sequence α tokens valueDim stateDim)
    (initial : State α valueDim stateDim) :
    State α valueDim stateDim × Tensor α [tokens, valueDim] :=
  let result := CausalScan.run step initial inputs.toList
  (result.1, Tensor.stackLeading fun i =>
    result.2.getD i.val (Tensor.full [valueDim] 0))

/-- Every bounded output index is present, so packing does not use the default zero vector. -/
theorem output_index_lt {α : Type} [Storage α] [Add α] [Mul α] [Zero α]
    {tokens valueDim stateDim : Nat} (inputs : Sequence α tokens valueDim stateDim)
    (initial : State α valueDim stateDim) (i : Fin tokens) :
    i.val < (CausalScan.run step initial inputs.toList).2.length := by
  rw [CausalScan.outputs_length, inputs.toList_length]
  exact i.isLt

/-- Row `i` of the packed output is the scan's `i`-th output. -/
theorem run_unstack {α : Type} [Storage α] [Add α] [Mul α] [Zero α]
    {tokens valueDim stateDim : Nat} (inputs : Sequence α tokens valueDim stateDim)
    (initial : State α valueDim stateDim) (i : Fin tokens) :
    Tensor.unstack (inputs.run initial).2 i =
      (CausalScan.run step initial inputs.toList).2[i.val]'(inputs.output_index_lt initial i) := by
  simp only [run, Tensor.unstack_stackLeading]
  exact List.getD_eq_getElem _ _ (inputs.output_index_lt initial i)

end Sequence

/-- Prepared inputs that carry a proof of `|retention| ≤ L`, as an error certificate needs. -/
abbrev BoundedInput (L : ℝ) (valueDim stateDim : Nat) :=
  { input : Input ℝ valueDim stateDim // |input.retention| ≤ L }

/-- `step` restricted to inputs with a retention bound. -/
noncomputable def boundedStep {valueDim stateDim : Nat} (L : ℝ)
    (state : State ℝ valueDim stateDim)
    (input : BoundedInput L valueDim stateDim) :
    State ℝ valueDim stateDim × Tensor ℝ [valueDim] :=
  step state input.val

/-- `Cache.quantizedChunks_within_of_writes` for the tensor SSM core. Retention and
storage-error bounds are hypotheses, not inferred from a precision format or trained weights. -/
theorem checkpoint_error {valueDim stateDim : Nat}
    (quantize : State ℝ valueDim stateDim → State ℝ valueDim stateDim)
    (L δ : ℝ) (hL : 0 ≤ L)
    (initial : State ℝ valueDim stateDim)
    (chunks : List (List (BoundedInput L valueDim stateDim)))
    (hwrites : Cache.WritesWithin (boundedStep L) quantize δ initial chunks) :
    Cache.Within (Cache.checkpointBudget L δ 0 chunks)
      (Cache.quantizedChunks (boundedStep L) quantize initial chunks)
      (CausalScan.run (boundedStep L) initial chunks.flatten).1 := by
  apply Cache.quantizedChunks_within_of_writes (boundedStep L) quantize L δ
    (fun input e x y hxy => update_within x y input.val L e hL input.property hxy)
    chunks initial initial 0 (Cache.within_self initial) hwrites

end Nemotron.SSM
