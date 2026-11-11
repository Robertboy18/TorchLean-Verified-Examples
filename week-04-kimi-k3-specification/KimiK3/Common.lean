/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Layers.Normalization
public import TorchLean.Tensor.LinearAlgebra.Algorithms
public import TorchLean.Tensor.Operations

/-!
# Shared Kimi K3 operations

Small mathematical operations used by several parts of Kimi K3. The language backbone, routed
experts and vision encoder all refer to these definitions instead of carrying local copies.
-/

@[expose] public section

namespace KimiK3

open TorchLean
open Tensor

namespace RMSNorm

/-- Scale-only RMS normalization when the vector width is known to be positive. -/
def scalePositive {α : Type} [Storage α] [Context α] {n : Nat} (_h : 0 < n)
    (x gamma : Tensor α [n])
    (epsilon : α := normalizationEpsilon) : Tensor α [n] :=
  Tensor.rmsNorm x [⟨0, Nat.zero_lt_one⟩] gamma epsilon

/-- Scale-only RMS normalization of a vector.

For `x, gamma : R^n`, coordinate `i` of the result is

`x_i / sqrt((sum_j x_j^2) / n + epsilon) * gamma_i`.

A nonempty vector is normalized along its sole axis. An empty vector is returned as is.
-/
def scale {α : Type} [Storage α] [Context α] {n : Nat}
    (x gamma : Tensor α [n])
    (epsilon : α := normalizationEpsilon) :
    Tensor α [n] :=
  match n with
  | 0 => x
  | n + 1 => scalePositive (by omega) x gamma epsilon

/-- For positive width, `scale` unfolds to `scalePositive`. -/
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
    Tensor.unstack (rows hWidth input gamma) row =
      scalePositive hWidth (Tensor.unstack input row) gamma := by
  simp [rows]

/-- RMS normalization without a learned scale.

AttnRes uses this form to normalize keys before computing attention over depth.
-/
def unit {α : Type} [Storage α] [Context α] {n : Nat}
    (x : Tensor α [n])
    (epsilon : α := normalizationEpsilon) :
    Tensor α [n] :=
  scale x (Tensor.ones [n]) epsilon

end RMSNorm

namespace Normalize

/-- Divide routing weights by their total when it is positive, and return the uniform distribution
otherwise. The fallback also covers empty routes and totals that are not positive after rounding. -/
def probabilities {α : Type} [Storage α] [Context α] {n : Nat}
    (weights : Tensor α [n]) : Tensor α [n] :=
  let total := Tensor.sum weights
  if 0 < total then
    Tensor.map (fun weight => weight / total) weights
  else
    Tensor.full [n] (1 / (n : α))

/-- Additive stabilizer `10⁻⁶` in KDA's L2 normalization (RMSNorm uses `10⁻⁵`). -/
def l2Epsilon {α : Type} [Context α] : α :=
  1 / 1000000

/-- Positive total mass selects ordinary normalization. -/
theorem probabilities_of_sum_pos {α : Type} [Storage α] [Context α] {n : Nat}
    (weights : Tensor α [n]) (hTotal : 0 < Tensor.sum weights) :
    probabilities weights =
      Tensor.map (fun weight => weight / Tensor.sum weights) weights := by
  simp [probabilities, hTotal]

/-- Nonpositive total mass selects the uniform fallback. -/
theorem probabilities_of_sum_nonpos {α : Type} [Storage α] [Context α] {n : Nat}
    (weights : Tensor α [n]) (hTotal : ¬ 0 < Tensor.sum weights) :
    probabilities weights =
      Tensor.full [n] ((1 : α) / ((n : Nat) : α)) := by
  simp [probabilities, hTotal]

end Normalize

end KimiK3
