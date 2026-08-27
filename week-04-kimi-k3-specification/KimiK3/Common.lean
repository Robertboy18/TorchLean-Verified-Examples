/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import NN.Spec.Layers.Normalization
public import NN.Spec.Core.TensorReductionShape

/-!
# Shared Kimi K3 operations

This module contains small mathematical operations used by several parts of Kimi K3. Keeping them
here ensures that the language backbone, routed experts, and vision encoder refer to one
specification rather than carrying locally equivalent copies.
-/

@[expose] public section

namespace KimiK3

open Spec
open Tensor

namespace RMSNorm

/-- Scale-only RMS normalization when the vector width is known to be positive. -/
def scalePositive {α : Type} [Context α] {n : Nat} (h : 0 < n)
    (x gamma : Tensor α [n])
    (epsilon : α := Numbers.normalizationEpsilon) : Tensor α [n] :=
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
def scale {α : Type} [Context α] {n : Nat}
    (x gamma : Tensor α (.dim n .scalar))
    (epsilon : α := Numbers.normalizationEpsilon) :
    Tensor α (.dim n .scalar) :=
  match n with
  | 0 => x
  | n + 1 => scalePositive (by omega) x gamma epsilon

/-- On a positive-width vector, the total adapter is the direct positive-width definition. -/
theorem scale_eq_scalePositive {α : Type} [Context α] {n : Nat} (h : 0 < n)
    (x gamma : Tensor α [n]) : scale x gamma = scalePositive h x gamma := by
  cases n with
  | zero => omega
  | succ n => rfl

/-- RMS normalization without a learned scale.

AttnRes uses this form to normalize keys before computing attention over depth.
-/
def unit {α : Type} [Context α] {n : Nat}
    (x : Tensor α (.dim n .scalar))
    (epsilon : α := Numbers.normalizationEpsilon) :
    Tensor α (.dim n .scalar) :=
  scale x (Spec.fill 1 (.dim n .scalar)) epsilon

end RMSNorm

namespace Normalize

/-- Elementwise specification maps commute with vector indexing. -/
@[simp] theorem getScalar_mapSpec {α : Type} [Context α] {n : Nat}
    (f : α → α) (vector : Tensor α [n]) (index : Fin n) :
    Tensor.getScalar (Tensor.mapSpec f vector) index = f (Tensor.getScalar vector index) := by
  cases vector with
  | dim values =>
      cases h : values index with
      | scalar value =>
          simp [Tensor.mapSpec, Tensor.getScalar, Spec.get, h]

/-- Normalize nonnegative weights, using the uniform distribution when their sum is zero. -/
def probabilities {α : Type} [Context α] {n : Nat}
    (weights : Tensor α [n]) : Tensor α [n] :=
  let total := Tensor.sumSpec weights
  if total > 0 then
    Tensor.mapSpec (fun weight => weight / total) weights
  else
    Spec.fill ((1 : α) / ((n : Nat) : α)) [n]

/-- L2-normalize a vector with an additive term under the square root. -/
def regularizedL2 {α : Type} [Context α] {n : Nat}
    (vector : Tensor α [n]) (regularizer : α) : Tensor α [n] :=
  let norm := MathFunctions.sqrt (Tensor.sumSpec (Tensor.squareSpec vector) + regularizer)
  Tensor.mapSpec (fun value => value / norm) vector

end Normalize

end KimiK3
