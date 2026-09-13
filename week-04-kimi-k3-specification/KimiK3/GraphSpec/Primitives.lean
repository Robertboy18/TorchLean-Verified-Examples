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
# Graph primitives used by Kimi K3

TorchLean's DAG language already supplies the architecture-independent matrix, elementwise,
reshape, and activation operations used below. This module contains only the capped activation
that is specific to K3's SiTU expert. Its mathematical and executable meanings are both expressed
with ordinary TorchLean operations; it does not call an expert implementation as an opaque step.
-/

@[expose] public section

namespace KimiK3

open TorchLean
namespace GraphSpec
namespace PrimOp

open Spec
open TorchLean.Tensor
open NN.GraphSpec.DAG
open Runtime.Autograd.Torch

/-- Smoothly cap every coordinate of a vector by a scalar graph input.

The executable program uses reciprocal, scalar broadcasting, multiplication, and `tanh` to
compute `cap * tanh(x / cap)`. Keeping the cap as an input records the scalar convention in the
graph ABI rather than freezing a paper-specific value into the operation.
-/
def softCap (n : Nat) : PrimOp [.scalar, .dim n .scalar] (.dim n .scalar) :=
  { name := "softCap"
    specFwd := fun {_α} _storage _context xs =>
      match xs with
      | .cons cap (.cons input .nil) =>
          Tensor.mulSpec
            (Activation.tanhSpec
              (Tensor.mulSpec input (Tensor.full (.dim n .scalar) (1 / cap.item))))
            (Tensor.full (.dim n .scalar) cap.item)
    program := fun {α} _ _ =>
      fun {m} _ _ => fun cap input =>
        (do
          let inverse ← Runtime.Autograd.Model.inv (m := m) (α := α) cap
          let inverseVector ← Runtime.Autograd.Model.broadcastTo (m := m) (α := α)
            (Shape.CanBroadcastTo.scalarTo (.dim n .scalar)) inverse
          let scaled ← Runtime.Autograd.Model.mul (m := m) (α := α) input inverseVector
          let cappedUnit ← Runtime.Autograd.Model.tanh (m := m) (α := α) scaled
          let capVector ← Runtime.Autograd.Model.broadcastTo (m := m) (α := α)
            (Shape.CanBroadcastTo.scalarTo (.dim n .scalar)) cap
          Runtime.Autograd.Model.mul (m := m) (α := α) cappedUnit capVector :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α) (.dim n .scalar))) }

end PrimOp

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
    (input gamma : _root_.TorchLean.Tensor α (.dim width .scalar)) :
    NN.GraphSpec.DAG.PrimOp.Internal.rmsNormVectorSemantics hWidth input gamma =
      KimiK3.RMSNorm.scale input gamma := by
  rw [RMSNorm.scale_eq_scalePositive hWidth]
  rfl

/-- Generalized RMS normalization specializes to the K3 vector adapter with no leading axes. -/
theorem rmsNormSemantics_scalar_eq_scale {α : Type} [Storage α] [Context α] {width : Nat}
    (hWidth : 0 < width)
    (input gamma : _root_.TorchLean.Tensor α (.dim width .scalar)) :
    NN.GraphSpec.DAG.PrimOp.rmsNormSemantics .scalar hWidth gamma input =
      KimiK3.RMSNorm.scale input gamma := by
  exact rmsNormVectorSemantics_eq_scale hWidth input gamma

/-- Generalized axis concatenation reduces to ordinary leading-axis concatenation at axis zero. -/
private theorem concatAxisSpec_zero {α : Type} [Storage α] [Context α] {left right : Nat}
    {rest : _root_.Spec.Shape}
    (a : _root_.TorchLean.Tensor α (.dim left rest))
    (b : _root_.TorchLean.Tensor α (.dim right rest)) :
    NN.GraphSpec.DAG.PrimOp.concatAxisSpec (.dim left rest) 0 left right a b =
      _root_.TorchLean.Tensor.concatAxisSpec .scalar a b := by
  rfl

/-- Concatenate two graph terms along their first axis through the generalized axis primitive. -/
def concatAxisZeroTerm {Γ : List _root_.Spec.Shape}
    (left right : Nat) (rest : _root_.Spec.Shape)
    (a : NN.GraphSpec.DAG.Term Γ (.dim left rest))
    (b : NN.GraphSpec.DAG.Term Γ (.dim right rest)) :
    NN.GraphSpec.DAG.Term Γ (.dim (left + right) rest) :=
  NN.GraphSpec.DAG.Term.cast
    (NN.GraphSpec.DAG.Term.op
      (NN.GraphSpec.DAG.PrimOp.concatAxis (.dim left rest) 0 left right)
      (.cons a (.cons b .nil)))
    (by rfl)

/-- Evaluation of first-axis term concatenation is the canonical tensor concatenation. -/
@[simp] theorem eval_concatAxisZeroTerm {Γ : List _root_.Spec.Shape}
    (env : TorchLean.TensorPack ℝ Γ) (left right : Nat) (rest : _root_.Spec.Shape)
    (a : NN.GraphSpec.DAG.Term Γ (.dim left rest))
    (b : NN.GraphSpec.DAG.Term Γ (.dim right rest)) :
    NN.GraphSpec.DAG.Term.eval env (concatAxisZeroTerm left right rest a b) =
      _root_.TorchLean.Tensor.concatAxisSpec .scalar
        (NN.GraphSpec.DAG.Term.eval env a) (NN.GraphSpec.DAG.Term.eval env b) := by
  unfold concatAxisZeroTerm
  rw [NN.GraphSpec.DAG.Term.eval_cast]
  rw [NN.GraphSpec.DAG.Term.eval_op]
  simp only [NN.GraphSpec.DAG.Term.evalArgs,
    NN.GraphSpec.DAG.PrimOp.concatAxis_specFwd]
  exact concatAxisSpec_zero (α := ℝ) _ _

end GraphSpec
end KimiK3
