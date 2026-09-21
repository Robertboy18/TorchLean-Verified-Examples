/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import NN.Spec.Core.Sequence
public import NN.Spec.Core.Context.Real
public import NN.Spec.Core.Tensor.Numerics
public import NN.Spec.Core.TensorReductionShape
public import NN.Spec.Layers.Normalization

/-!
# Shared Kimi K3 operations

This module contains small mathematical operations used by several parts of Kimi K3. Keeping them
here ensures that the language backbone, routed experts, and vision encoder refer to one
specification rather than carrying locally equivalent copies.
-/

@[expose] public section

namespace KimiK3

open Spec TorchLean
open Tensor

namespace RMSNorm

/-- Scale-only RMS normalization when the vector width is known to be positive. -/
def scalePositive {α : Type} [Storage α] [Context α] {n : Nat} (h : 0 < n)
    (x gamma : Tensor α [n])
    (epsilon : α := normalizationEpsilon) : Tensor α [n] :=
  let rows : Tensor α [1, n] := .dim fun _ => x
  Spec.get (Spec.rmsNorm rows gamma (by positivity) h epsilon)
    ⟨0, by positivity⟩

/-- Scale-only RMS normalization of a vector.

For `x, gamma : R^n`, coordinate `i` of the result is

`x_i / sqrt((sum_j x_j^2) / n + epsilon) * gamma_i`.

This is a vector-shaped adapter around TorchLean's canonical matrix `Spec.rmsNorm`. For a nonempty
vector it inserts a singleton row, applies the library operation along the final axis, and removes
the row again. The empty-vector branch returns the unique tensor of that shape.
-/
def scale {α : Type} [Storage α] [Context α] {n : Nat}
    (x gamma : Tensor α (.dim n .scalar))
    (epsilon : α := normalizationEpsilon) :
    Tensor α (.dim n .scalar) :=
  match n with
  | 0 => x
  | n + 1 => scalePositive (by omega) x gamma epsilon

/-- On a positive-width vector, the total adapter is the direct positive-width definition. -/
theorem scale_eq_scalePositive {α : Type} [Storage α] [Context α] {n : Nat} (h : 0 < n)
    (x gamma : Tensor α [n]) : scale x gamma = scalePositive h x gamma := by
  cases n with
  | zero => omega
  | succ n => rfl

/-- Apply the shared vector RMSNorm independently to every matrix row. -/
def rows {α : Type} [Storage α] [Context α] {rowCount width : Nat}
    (hWidth : 0 < width) (input : Tensor α [rowCount, width])
    (gamma : Tensor α [width]) : Tensor α [rowCount, width] :=
  Tensor.mapLeading [rowCount] (scalePositive hWidth · gamma) input

/-- Row-wise RMS normalization agrees with vector normalization at every row. -/
@[simp] theorem get_rows {α : Type} [Storage α] [Context α] {rowCount width : Nat}
    (hWidth : 0 < width) (input : Tensor α [rowCount, width])
    (gamma : Tensor α [width]) (row : Fin rowCount) :
    Spec.get (rows hWidth input gamma) row = scalePositive hWidth (Spec.get input row) gamma := by
  simp [rows, Tensor.mapLeading, Spec.get]

/-- RMS normalization without a learned scale.

AttnRes uses this form to normalize keys before computing attention over depth.
-/
def unit {α : Type} [Storage α] [Context α] {n : Nat}
    (x : Tensor α (.dim n .scalar))
    (epsilon : α := normalizationEpsilon) :
    Tensor α (.dim n .scalar) :=
  scale x (Tensor.ones [n]) epsilon

end RMSNorm

namespace Normalize

/-- KDA's additive L2 regularizer is `10⁻⁶`, distinct from RMSNorm's `10⁻⁵` stabilizer. -/
def l2Epsilon {α : Type} [Context α] : α :=
  1 / 1000000

/-- Elementwise specification maps commute with vector indexing. -/
@[simp] theorem getScalar_mapSpec {α : Type} [Storage α] {n : Nat}
    (f : α → α) (vector : Tensor α [n]) (index : Fin n) :
    Tensor.getScalar (Tensor.mapSpec f vector) index = f (Tensor.getScalar vector index) := by
  change Tensor.getScalar (Tensor.map f vector) index = f (Tensor.getScalar vector index)
  exact Tensor.getScalar_map f vector index

/-- Positive total mass selects ordinary normalization. -/
theorem normalizeByPositiveSumSpec_of_sum_pos {α : Type} [Storage α] [Context α] {n : Nat}
    (weights : Tensor α [n]) (hTotal : 0 < Tensor.sumSpec weights) :
    Spec.normalizeByPositiveSumSpec weights =
      Tensor.mapSpec (fun weight => weight / Tensor.sumSpec weights) weights := by
  unfold Spec.normalizeByPositiveSumSpec
  simp only [hTotal, ite_eq_left]
  apply Tensor.ext_vector
  intro index
  rw [getScalar_mapSpec]
  cases hEntry : Spec.get weights index
  simp [Tensor.getScalar, hEntry]

/-- Nonpositive total mass selects the uniform fallback. -/
theorem normalizeByPositiveSumSpec_of_sum_nonpos {α : Type} [Storage α] [Context α] {n : Nat}
    (weights : Tensor α [n]) (hTotal : ¬ 0 < Tensor.sumSpec weights) :
    Spec.normalizeByPositiveSumSpec weights =
      Tensor.full [n] ((1 : α) / ((n : Nat) : α)) := by
  unfold Spec.normalizeByPositiveSumSpec
  apply Tensor.ext_vector
  intro index
  simp [hTotal, Tensor.getScalar_eq_apply, Tensor.dim, Tensor.scalar]

end Normalize

end KimiK3
