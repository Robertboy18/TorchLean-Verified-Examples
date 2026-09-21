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
    Tensor.broadcastTo_self]

/-- The K3 vector adapter and TorchLean's DAG vector primitive use the same RMSNorm semantics. -/
@[simp] theorem rmsNormVectorSemantics_eq_scale {α : Type} [Storage α] [Context α] {width : Nat}
    (hWidth : 0 < width)
    (input gamma : TorchLean.Tensor α (.dim width .scalar)) :
    NN.GraphSpec.DAG.PrimOp.Internal.rmsNormVectorSemantics hWidth input gamma =
      KimiK3.RMSNorm.scale input gamma := by
  rw [RMSNorm.scale_eq_scalePositive hWidth]
  rfl

/-- Generalized RMS normalization specializes to the K3 vector adapter with no leading axes. -/
theorem rmsNormSemantics_scalar_eq_scale {α : Type} [Storage α] [Context α] {width : Nat}
    (hWidth : 0 < width)
    (input gamma : TorchLean.Tensor α (.dim width .scalar)) :
    NN.GraphSpec.DAG.PrimOp.rmsNormSemantics .scalar hWidth gamma input =
      KimiK3.RMSNorm.scale input gamma := by
  exact rmsNormVectorSemantics_eq_scale hWidth input gamma

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
