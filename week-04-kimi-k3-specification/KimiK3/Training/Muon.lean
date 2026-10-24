/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Common
public import NN.MLTheory.Optimization.Muon
public import NN.Proofs.Tensor.Basic.Algebra

/-!
# Per-head Muon

Split a projection's output columns into head blocks, apply a supplied orthogonalizer to each
block, and reassemble the original layout. Split and merge laws prove this is a reversible
reindexing. Properties of the orthogonalized blocks come from the supplied orthogonalizer's
contract, so no particular iterative algorithm is proved correct here.

Reference: Kimi K3 technical report, Section 2.5 and Sections 3--4, pp. 10--14.
https://arxiv.org/abs/2607.24653.
-/

@[expose] public section

namespace KimiK3

open Spec TorchLean
open Tensor
open scoped BigOperators

namespace PerHeadMuon

open Optim.Muon

/-- One momentum matrix for every attention head. -/
abbrev HeadMatrices (α : Type) [Storage α] (heads rows columns : Nat) :=
  Fin heads → MatrixTensor α rows columns

/-- Partition a projection matrix along its output-head axis.

For a matrix of shape `rows × (heads * columns)`, output column `(head, column)` belongs to the
`head`-th `rows × columns` block. Per-Head Muon uses this layout to treat the Q, K, and V momentum
buffers head by head.
-/
def split {α : Type} [Storage α] {heads rows columns : Nat}
    (momentum : MatrixTensor α rows (heads * columns)) :
    HeadMatrices α heads rows columns :=
  fun head => Tensor.stackLeading fun row => Tensor.stackLeading fun column =>
    Tensor.scalar (get2 momentum row (finProdFinEquiv (head, column)))

/-- Reassemble per-head matrices into the original projection layout. -/
def merge {α : Type} [Storage α] {heads rows columns : Nat}
    (blocks : HeadMatrices α heads rows columns) :
    MatrixTensor α rows (heads * columns) :=
  Tensor.stackLeading fun row => Tensor.stackLeading fun outputColumn =>
    let headColumn := finProdFinEquiv.symm outputColumn
    Tensor.scalar (get2 (blocks headColumn.1) row headColumn.2)

/-- Splitting and reassembling a projection momentum matrix preserves every entry. -/
theorem merge_split {α : Type} [Storage α] {heads rows columns : Nat}
    (momentum : MatrixTensor α rows (heads * columns)) :
    merge (split momentum) = momentum := by
  apply TorchLean.Tensor.Internal.Rep.ext
  rintro ⟨row, column, ⟨⟩⟩
  have h : get2 (merge (split momentum)) row column = get2 momentum row column := by
    simp only [merge, split, Tensor.stackLeading, Spec.get2_dim, Prod.mk.eta,
      Equiv.apply_symm_apply]
  simpa only [Spec.get2_eq_apply] using h

/-- Reassembling and splitting an indexed head family returns the same family. -/
theorem split_merge {α : Type} [Storage α] {heads rows columns : Nat}
    (blocks : HeadMatrices α heads rows columns) :
    split (merge blocks) = blocks := by
  funext head
  apply TorchLean.Tensor.Internal.Rep.ext
  rintro ⟨row, column, ⟨⟩⟩
  have h : get2 (split (merge blocks) head) row column = get2 (blocks head) row column := by
    simp only [merge, split, Tensor.stackLeading, Spec.get2_dim, Equiv.symm_apply_apply]
  simpa only [Spec.get2_eq_apply] using h

/-- Orthogonalize every output-head block independently and restore the projection layout. -/
def orthogonalize {α : Type} [Storage α] {heads rows columns : Nat}
    (orthogonalizer : Orthogonalizer α [rows, columns])
    (momentum : MatrixTensor α rows (heads * columns)) :
    MatrixTensor α rows (heads * columns) :=
  merge fun head => orthogonalizer.apply (split momentum head)

/-- Splitting a Per-Head Muon direction recovers the independently orthogonalized head blocks. -/
theorem split_orthogonalize {α : Type} [Storage α] {heads rows columns : Nat}
    (orthogonalizer : Orthogonalizer α [rows, columns])
    (momentum : MatrixTensor α rows (heads * columns)) :
    split (orthogonalize orthogonalizer momentum) =
      fun head => orthogonalizer.apply (split momentum head) := by
  exact split_merge _

/-- With an exact orthogonalizer, every head block of the reassembled update has orthonormal
columns. -/
theorem orthogonalize_hasExactColumnGram {α : Type} [Storage α] [Context α]
    {heads rows columns : Nat}
    (orthogonalizer : Orthogonalizer α [rows, columns])
    (hExact : ∀ buffer : MatrixTensor α rows columns,
      ExactOrthogonalizesBuffer orthogonalizer buffer)
    (momentum : MatrixTensor α rows (heads * columns)) :
    ∀ head, HasExactColumnGram (split (orthogonalize orthogonalizer momentum) head) := by
  intro head
  rw [split_orthogonalize]
  exact hExact (split momentum head)

/-- An approximate orthogonalizer's Gram-residual bound carries over to every reassembled head. -/
theorem orthogonalize_hasApproxColumnGram {α : Type} [Storage α] [Context α]
    {heads rows columns : Nat} (epsilon : α)
    (orthogonalizer : Orthogonalizer α [rows, columns])
    (hApprox : ∀ buffer : MatrixTensor α rows columns,
      ApproxOrthogonalizesBuffer epsilon orthogonalizer buffer)
    (momentum : MatrixTensor α rows (heads * columns)) :
    ∀ head, HasApproxColumnGram epsilon
      (split (orthogonalize orthogonalizer momentum) head) := by
  intro head
  rw [split_orthogonalize]
  exact hApprox (split momentum head)

end PerHeadMuon

end KimiK3
