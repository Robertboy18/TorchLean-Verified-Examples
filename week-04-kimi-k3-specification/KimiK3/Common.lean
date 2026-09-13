/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import NN.Spec.Core.Context.Real
public import NN.Spec.Layers.Normalization
public import NN.Spec.Core.TensorReductionShape
import Mathlib.Tactic.Positivity

/-!
# Shared Kimi K3 operations

This module contains small mathematical operations used by several parts of Kimi K3. Keeping them
here ensures that the language backbone, routed experts, and vision encoder refer to one
specification rather than carrying locally equivalent copies.
-/

@[expose] public section

namespace KimiK3

open TorchLean

open Spec
open Tensor

namespace RMSNorm

/-- Scale-only RMS normalization when the vector width is known to be positive. -/
def scalePositive {α : Type} [Storage α] [Context α] {n : Nat} (h : 0 < n)
    (x gamma : Tensor α [n])
    (epsilon : α := TorchLean.normalizationEpsilon) : Tensor α [n] :=
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
    (epsilon : α := TorchLean.normalizationEpsilon) :
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

/-- RMS normalization without a learned scale.

AttnRes uses this form to normalize keys before computing attention over depth.
-/
def unit {α : Type} [Storage α] [Context α] {n : Nat}
    (x : Tensor α (.dim n .scalar))
    (epsilon : α := TorchLean.normalizationEpsilon) :
    Tensor α (.dim n .scalar) :=
  scale x (Tensor.full (.dim n .scalar) 1) epsilon

end RMSNorm

namespace Normalize

/-- Elementwise specification maps commute with vector indexing. -/
@[simp] theorem getScalar_mapSpec {α : Type} [Storage α] [Context α] {n : Nat}
    (f : α → α) (vector : Tensor α [n]) (index : Fin n) :
    Tensor.getScalar (Tensor.mapSpec f vector) index = f (Tensor.getScalar vector index) := by
  exact Tensor.getScalar_mapSpec f vector index

/-- Additive L2 regularizer used by KDA's query and key normalization.

This is `10⁻⁶`, independently of the `10⁻⁵` stabilizer used by RMS normalization. Keeping the
constant with the KDA normalization convention makes both the tensor specification and its graph
lowering use the same denominator.
-/
def l2Epsilon {α : Type} [Context α] : α :=
  1 / 1000000

/-- Normalize nonnegative weights, using the uniform distribution when their sum is zero. -/
def probabilities {α : Type} [Storage α] [Context α] {n : Nat}
    (weights : Tensor α [n]) : Tensor α [n] :=
  let total := Tensor.sumSpec weights
  if total > 0 then
    Tensor.mapSpec (fun weight => weight / total) weights
  else
    Tensor.full [n] ((1 : α) / ((n : Nat) : α))

/-- L2-normalize a vector with an additive term under the square root. -/
def regularizedL2 {α : Type} [Storage α] [Context α] {n : Nat}
    (vector : Tensor α [n]) (regularizer : α) : Tensor α [n] :=
  let norm := MathFunctions.sqrt (Tensor.sumSpec (Tensor.squareSpec vector) + regularizer)
  Tensor.mapSpec (fun value => value / norm) vector

end Normalize

end KimiK3
