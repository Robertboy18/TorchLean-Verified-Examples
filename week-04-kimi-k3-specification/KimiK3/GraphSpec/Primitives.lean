/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.Common
public import NN.GraphSpec.DAG
public import NN.GraphSpec.DAG.Model

/-!
# Kimi K3 graph compositions

TorchLean's DAG language already supplies the architecture-independent matrix, elementwise,
reshape, and activation operations used below. The SiTU cap is therefore a term builder made from
ordinary TorchLean nodes, not a new opaque primitive. The other declarations bridge K3's reference
semantics to existing GraphSpec operations.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec

open Spec TorchLean
open TorchLean.Tensor
open NN.GraphSpec.DAG

private theorem reshape_dim {α : Type} [Storage α] {n : Nat} {source target : Shape}
    (h : source.size = target.size) (input : Tensor α (.dim n source)) :
    Tensor.reshapeSpec input (show (Shape.dim n source).size = (Shape.dim n target).size by
      simp [Shape.size, h]) =
      Tensor.dim (fun i => Tensor.reshapeSpec (input.unstack i) h) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  rintro ⟨i, j⟩
  simp only [Tensor.reshapeSpec, TorchLean.Tensor.Internal.Rep.reshape_apply,
    Tensor.dim, TorchLean.Tensor.Internal.Rep.stack_apply, Tensor.unstack,
    TorchLean.Tensor.Internal.Rep.unstack_apply]
  congr 1
  apply TorchLean.Tensor.Internal.Coord.linearize_injective
  apply Fin.ext
  simp only [TorchLean.Tensor.Internal.Coord.linearize_unlinearize,
    finCongr_apply, Fin.val_cast, TorchLean.Tensor.Internal.Coord.linearize_cons_val]
  simp [Shape.internalSize_congr h]

private theorem singletonShape_rank (leading shape : Shape) :
    (leading.concat (Shape.padLeft shape.rank .scalar)).rank =
      (leading.concat shape).rank := by
  simp [Tensor.LinearAlgebra.Internal.rank_concat, Shape.rank]

private theorem singletonShape_broadcast (leading shape : Shape) :
    (leading.concat (Shape.padLeft shape.rank .scalar)).CanBroadcastTo
      (leading.concat shape) := by
  induction leading with
  | scalar =>
      induction shape with
      | scalar => trivial
      | dim count rest ih =>
          exact (Shape.canBroadcastTo_dim_dim_of_rank_eq
            (singletonShape_rank .scalar rest)).mpr ⟨Or.inr rfl, ih⟩
  | dim count rest ih =>
      exact (Shape.canBroadcastTo_dim_dim_of_rank_eq
        (singletonShape_rank rest shape)).mpr ⟨Or.inl rfl, ih⟩

/-- Repeat each leading coordinate across a suffix of any rank, without adding arithmetic. -/
def expandTerm {Γ : List Shape} (leading shape : Shape)
    (input : Term Γ (leading.concat .scalar)) : Term Γ (leading.concat shape) :=
  let singleton := Term.op
    (PrimOp.reshape _ (leading.concat (Shape.padLeft shape.rank .scalar)) (by
      simp [Shape.size_concat, Shape.size])) (.cons input .nil)
  Term.op (PrimOp.broadcast (by exact singletonShape_broadcast leading shape))
    (.cons singleton .nil)

private theorem reshape_scalar_full {α : Type} [Storage α] {shape : Shape}
    (h : Shape.scalar.size = shape.size) (input : Tensor α []) :
    Tensor.reshapeSpec input h = Tensor.full shape input.item := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [Tensor.reshapeSpec, Tensor.full, Tensor.item]

private theorem broadcast_singletons {α : Type} [Storage α] (shape : Shape) (value : α) :
    Tensor.broadcastTo (singletonShape_broadcast .scalar shape)
      (Tensor.full (Shape.padLeft shape.rank .scalar) value) = Tensor.full shape value := by
  induction shape with
  | scalar => exact Tensor.broadcastTo_scalar _ _
  | dim count rest ih =>
      rw [show Tensor.broadcastTo (singletonShape_broadcast .scalar (.dim count rest))
          (Tensor.full (Shape.padLeft (Shape.dim count rest).rank .scalar) value) =
          Tensor.dim (fun _ : Fin count =>
            Tensor.broadcastTo (singletonShape_broadcast .scalar rest)
              ((Tensor.full (.dim 1 (Shape.padLeft rest.rank .scalar)) value).unstack 0)) from
        Tensor.broadcastTo_dim_one (by simp [Shape.rank]) _ _]
      have slice : (Tensor.full (.dim 1 (Shape.padLeft rest.rank .scalar)) value).unstack 0 =
          Tensor.full (Shape.padLeft rest.rank .scalar) value := by
        apply TorchLean.Tensor.Internal.Rep.ext
        intro coordinate
        simp [Tensor.full, Tensor.unstack]
      simp only [slice]
      simp_rw [ih]
      apply TorchLean.Tensor.Internal.Rep.ext
      rintro ⟨i, coordinate⟩
      simp [Tensor.dim, Tensor.full]

/-- Expansion copies the scalar at each leading coordinate into its entire suffix. -/
@[simp] theorem eval_expandTerm {α : Type} [Storage α] [Context α] {Γ : List Shape}
    (env : TorchLean.TensorPack α Γ) (leading shape : Shape)
    (input : Term Γ (leading.concat .scalar)) :
    Term.eval env (expandTerm leading shape input) =
      Tensor.mapLeading leading (fun x => Tensor.full shape x.item) (Term.eval env input) := by
  simp only [expandTerm, Term.eval_op, Term.evalArgs, PrimOp.reshape_specFwd,
    PrimOp.broadcast]
  generalize Term.eval env input = values
  clear input env
  induction leading with
  | scalar => rw [reshape_scalar_full]; exact broadcast_singletons shape values.item
  | dim count rest ih =>
      rw [reshape_dim (by simp [Shape.size_concat, Shape.size]),
        Tensor.broadcastTo_dim_eq (singletonShape_rank rest shape)]
      simp only [Tensor.unstack_dim, Tensor.mapLeading, ih]

/-- Broadcasting a scalar through every leading axis is the constant tensor of that shape. -/
@[simp] private theorem broadcast_scalarTo_eq_full {α : Type} [Storage α]
    (shape : Shape) (value : Tensor α []) :
    Tensor.broadcastTo (Shape.CanBroadcastTo.scalarTo shape) value =
      Tensor.full shape value.item := by
  induction shape with
  | scalar =>
      apply Tensor.ext_scalar
      simp
  | dim count rest ih =>
      rw [Tensor.broadcastTo_expand (by simp [Shape.rank])]
      simp_rw [ih]
      apply TorchLean.Tensor.Internal.Rep.ext
      rintro ⟨index, coordinate⟩
      simp [Tensor.dim, Tensor.full]

private theorem broadcast_leading_proof (leading : Shape) {source target : Shape}
    (hRank : source.rank = target.rank) (h : source.CanBroadcastTo target) :
    (leading.concat source).CanBroadcastTo (leading.concat target) := by
  induction leading with
  | scalar => exact h
  | dim count rest ih =>
      exact (Shape.canBroadcastTo_dim_dim_of_rank_eq (by
        simp only [Tensor.LinearAlgebra.Internal.rank_concat, hRank])).mpr ⟨Or.inl rfl, ih⟩

/-- Outer products over an arbitrary leading shape, built from reshape, broadcast, and multiply. -/
def outerTerm {Γ : List Shape} (leading : Shape) (rows columns : Nat)
    (left : Term Γ (leading.concat [rows]))
    (right : Term Γ (leading.concat [columns])) :
    Term Γ (leading.concat [rows, columns]) :=
  let expandedLeft : Term Γ (leading.concat [rows, columns]) :=
    Term.cast (expandTerm (leading.concat [rows]) [columns]
        (Term.cast left (Shape.concat_scalar _).symm))
      (by simp [Shape.concat_assoc, Shape.concat])
  let rightRow := Term.op (PrimOp.reshape _ (leading.concat [1, columns]) (by
      simp [Shape.size_concat, Shape.size])) (.cons right .nil)
  let expandedRight := Term.op (PrimOp.broadcast (by
      exact broadcast_leading_proof leading rfl (.dim_1_to_n (.refl [columns]))))
    (.cons rightRow .nil)
  Term.op (PrimOp.mul _) (.cons expandedLeft (.cons expandedRight .nil))

@[simp] private theorem cast_dim {α : Type} [Storage α] {n : Nat} {source target : Shape}
    (h : source = target) (values : Fin n → Tensor α source) :
    (congrArg (Shape.dim n) h ▸ Tensor.dim values) = Tensor.dim (fun i => h ▸ values i) := by
  cases h
  rfl

@[simp] private theorem unstack_cast {α : Type} [Storage α] {n : Nat} {source target : Shape}
    (h : source = target) (input : Tensor α (.dim n source)) (i : Fin n) :
    (congrArg (Shape.dim n) h ▸ input).unstack i = h ▸ input.unstack i := by
  cases h
  rfl

/-- The graph computes one outer product at each leading coordinate. -/
@[simp] theorem eval_outerTerm {α : Type} [Storage α] [Context α] {Γ : List Shape}
    (env : TorchLean.TensorPack α Γ) (leading : Shape) (rows columns : Nat)
    (left : Term Γ (leading.concat [rows]))
    (right : Term Γ (leading.concat [columns])) :
    Term.eval env (outerTerm leading rows columns left right) =
      Tensor.zipEach leading [rows, columns] Spec.outerProductSpec
        (Term.eval env left) (Term.eval env right) := by
  simp only [outerTerm, Term.eval_op, Term.evalArgs, Term.eval_cast, eval_expandTerm,
    PrimOp.reshape_specFwd, PrimOp.broadcast, PrimOp.mul_specFwd]
  generalize Term.eval env left = a
  generalize Term.eval env right = b
  clear left right env
  induction leading with
  | scalar =>
      simp only [Shape.concat, Tensor.mapLeading, Tensor.zipEach, Tensor.reshapeSpec,
        TorchLean.Tensor.Internal.Rep.reshape_one_cons]
      rw [show Tensor.broadcastTo _ (TorchLean.Tensor.Internal.Rep.stack fun _ : Fin 1 => b) =
          Tensor.dim (fun _ : Fin rows => b) from by
        calc
          _ = Tensor.dim (fun _ : Fin rows =>
              Tensor.broadcastTo (.refl [columns]) ((Tensor.dim fun _ : Fin 1 => b).unstack 0)) :=
            Tensor.broadcastTo_dim_one rfl _ _
          _ = _ := by simp]
      apply TorchLean.Tensor.Internal.Rep.ext
      rintro ⟨i, j, ⟨⟩⟩
      simp [Spec.outerProductSpec, Tensor.mulSpec, Tensor.map2Spec,
        Tensor.dim, Tensor.full, Tensor.getScalar, Tensor.item, Tensor.unstack, Spec.get]
  | dim count rest ih =>
      rw [reshape_dim (by simp [Shape.size_concat, Shape.size])]
      have ranks : (rest.concat [1, columns]).rank = (rest.concat [rows, columns]).rank := by
        simp only [Tensor.LinearAlgebra.Internal.rank_concat, Shape.rank]
      rw [Tensor.broadcastTo_dim_eq ranks]
      simp only [Shape.concat, Tensor.zipEach, Tensor.mapLeading, Tensor.unstack_dim]
      rw [cast_dim (n := count) (Shape.concat_assoc rest [rows] [columns])]
      simp_rw [unstack_cast (n := count) (Shape.concat_scalar (rest.concat [rows])).symm]
      simp_rw [← ih]
      simp [Tensor.mulSpec, Tensor.map2Spec, Tensor.dim,
        TorchLean.Tensor.Internal.Rep.zipWith_stack]

/-- Multiply every coordinate by a scalar, using ordinary broadcast and multiplication nodes. -/
def scaleTerm {Γ : List Shape} (shape : Shape)
    (coefficient : Term Γ []) (input : Term Γ shape) : Term Γ shape :=
  let coefficients := Term.op (PrimOp.broadcast (Shape.CanBroadcastTo.scalarTo shape))
    (.cons coefficient .nil)
  Term.op (PrimOp.mul shape) (.cons coefficients (.cons input .nil))

/-- Scaling preserves the scalar-first multiplication order of the reference calculation. -/
@[simp] theorem eval_scaleTerm {α : Type} [Storage α] [Context α] {Γ : List Shape}
    (env : TorchLean.TensorPack α Γ) (shape : Shape)
    (coefficient : Term Γ []) (input : Term Γ shape) :
    Term.eval env (scaleTerm shape coefficient input) =
      Tensor.mapSpec ((Term.eval env coefficient).item * ·) (Term.eval env input) := by
  simp [scaleTerm, Term.eval, Term.evalArgs, PrimOp.broadcast, PrimOp.mul]

/-- Smoothly cap every coordinate of a tensor by a scalar graph input.

The returned term contains reciprocal, scalar-broadcast, multiplication, and `tanh` nodes computing
`cap * tanh(x / cap)`. Keeping the cap as an input records the numerical convention in the graph
ABI without hiding the calculation behind a K3-specific operation.
-/
def softCapTerm {Γ : List Shape} (shape : Shape)
    (cap : Term Γ .scalar) (input : Term Γ shape) : Term Γ shape :=
  let inverse := Term.op (NN.GraphSpec.DAG.PrimOp.inv .scalar) (.cons cap .nil)
  let inverseTensor := Term.op
    (NN.GraphSpec.DAG.PrimOp.broadcast (Shape.CanBroadcastTo.scalarTo shape))
    (.cons inverse .nil)
  let scaled := Term.op (NN.GraphSpec.DAG.PrimOp.mul shape)
    (.cons input (.cons inverseTensor .nil))
  let cappedUnit := Term.op (NN.GraphSpec.DAG.PrimOp.tanh shape) (.cons scaled .nil)
  let capTensor := Term.op
    (NN.GraphSpec.DAG.PrimOp.broadcast (Shape.CanBroadcastTo.scalarTo shape))
    (.cons cap .nil)
  Term.op (NN.GraphSpec.DAG.PrimOp.mul shape)
    (.cons cappedUnit (.cons capTensor .nil))

/-- Evaluating `softCapTerm` gives the coordinatewise cap `cap * tanh (input / cap)`. -/
@[simp] theorem eval_softCapTerm {α : Type} [Storage α] [Context α] {Γ : List Shape}
    (env : TorchLean.TensorPack α Γ) (shape : Shape)
    (cap : Term Γ .scalar) (input : Term Γ shape) :
    Term.eval env (softCapTerm shape cap input) =
      Tensor.mulSpec
        (Activation.tanhSpec
          (Tensor.mulSpec (Term.eval env input)
            (Tensor.full shape (1 / (Term.eval env cap).item))))
        (Tensor.full shape (Term.eval env cap).item) := by
  simp [softCapTerm, Term.eval, Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.inv, NN.GraphSpec.DAG.PrimOp.broadcast,
    NN.GraphSpec.DAG.PrimOp.mul, NN.GraphSpec.DAG.PrimOp.tanh]

/-- With no batch axes, the broadcast primitive is the ordinary vector-matrix product. -/
@[simp] theorem broadcastVecMat_scalar_specFwd {α : Type} [Storage α] [Context α]
    {rows columns : Nat} (vector : Tensor α [rows]) (matrix : Tensor α [rows, columns]) :
    (NN.GraphSpec.DAG.PrimOp.broadcastVecMat .scalar .scalar .scalar
      rows columns .scalar .scalar).specFwd (.cons vector (.cons matrix .nil)) =
      Spec.vecMatMulSpec vector matrix := by
  simp [NN.GraphSpec.DAG.PrimOp.broadcastVecMat,
    NN.GraphSpec.DAG.PrimOp.Internal.vecMatCommonBatchSpec,
    Tensor.broadcastTo_self, Tensor.zipEach]

/-- The K3 vector adapter and TorchLean's DAG vector primitive use the same RMSNorm semantics. -/
@[simp] theorem rmsNormVectorSpec_eq_scale {α : Type} [Storage α] [Context α] {width : Nat}
    (hWidth : 0 < width)
    (input gamma : TorchLean.Tensor α (.dim width .scalar)) :
    NN.GraphSpec.DAG.PrimOp.Internal.rmsNormVectorSpec hWidth input gamma =
      KimiK3.RMSNorm.scale input gamma := by
  rw [RMSNorm.scale_eq_scalePositive hWidth]
  rfl

/-- Generalized RMS normalization specializes to the K3 vector adapter with no leading axes. -/
theorem rmsNormSpec_scalar_eq_scale {α : Type} [Storage α] [Context α] {width : Nat}
    (hWidth : 0 < width)
    (input gamma : TorchLean.Tensor α (.dim width .scalar)) :
    NN.GraphSpec.DAG.PrimOp.rmsNormSpec .scalar hWidth gamma input =
      KimiK3.RMSNorm.scale input gamma := by
  exact rmsNormVectorSpec_eq_scale hWidth input gamma

/-- Generalized axis concatenation reduces to ordinary leading-axis concatenation at axis zero. -/
private theorem concatAxisSpec_zero {α : Type} [Storage α] {left right : Nat}
    {rest : _root_.Spec.Shape}
    (a : TorchLean.Tensor α (.dim left rest))
    (b : TorchLean.Tensor α (.dim right rest)) :
    NN.GraphSpec.DAG.PrimOp.concatAxisSpec (.dim left rest) 0 left right a b =
      TorchLean.Tensor.concatAxisSpec .scalar a b := by
  rfl

/-- Concatenate two graph terms along their leading axis. -/
def concatLeadingTerm {Γ : List _root_.Spec.Shape}
    (left right : Nat) (rest : _root_.Spec.Shape)
    (a : NN.GraphSpec.DAG.Term Γ (.dim left rest))
    (b : NN.GraphSpec.DAG.Term Γ (.dim right rest)) :
    NN.GraphSpec.DAG.Term Γ (.dim (left + right) rest) :=
  NN.GraphSpec.DAG.Term.cast
    (NN.GraphSpec.DAG.Term.op
      (NN.GraphSpec.DAG.PrimOp.concatAxis (.dim left rest) 0 left right)
      (.cons a (.cons b .nil)))
    (by rfl)

/-- Evaluation of leading-axis term concatenation is the canonical tensor concatenation. -/
@[simp] theorem eval_concatLeadingTerm {α : Type} [Storage α] [Context α]
    {Γ : List _root_.Spec.Shape} (env : TorchLean.TensorPack α Γ)
    (left right : Nat) (rest : _root_.Spec.Shape)
    (a : NN.GraphSpec.DAG.Term Γ (.dim left rest))
    (b : NN.GraphSpec.DAG.Term Γ (.dim right rest)) :
    NN.GraphSpec.DAG.Term.eval env (concatLeadingTerm left right rest a b) =
      TorchLean.Tensor.concatAxisSpec .scalar
        (NN.GraphSpec.DAG.Term.eval env a) (NN.GraphSpec.DAG.Term.eval env b) := by
  unfold concatLeadingTerm
  rw [NN.GraphSpec.DAG.Term.eval_cast]
  rw [NN.GraphSpec.DAG.Term.eval_op]
  simp only [NN.GraphSpec.DAG.Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.concatAxis_specFwd]
  exact concatAxisSpec_zero (α := α) _ _

end GraphSpec
end KimiK3
