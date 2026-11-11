/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Operators

/-!
# Shared tensor expressions in K3

The recurrent cells reuse scalar expansion, outer products and normalization.
The SiTU cap composes reciprocal, multiplication and tanh, with the cap
supplied alongside the input tensor.
-/

@[expose] public section

namespace KimiK3.GraphSpec

open TorchLean

/-- Fill the suffix at each leading coordinate from that coordinate's scalar. -/
def expandTerm {Γ : List Shape} (leading shape : Shape)
    (input : Term Γ (leading ++ [])) : Term Γ (leading ++ shape) :=
  Term.op (PrimOp.unary fun x => Tensor.mapLeading leading
    (fun v => Tensor.full shape v.item) x) (.cons input .nil)

@[simp] theorem eval_expandTerm {α : Type} [Storage α] [Context α] {Γ : List Shape}
    (env : TensorPack α Γ) (leading shape : Shape) (input : Term Γ (leading ++ [])) :
    Term.eval env (expandTerm leading shape input) =
      Tensor.mapLeading leading (fun x => Tensor.full shape x.item) (Term.eval env input) := rfl

/-- Form one outer product at each leading coordinate. -/
def outerTerm {Γ : List Shape} (leading : Shape) (rows columns : Nat)
    (left : Term Γ (leading ++ [rows])) (right : Term Γ (leading ++ [columns])) :
    Term Γ (leading ++ [rows, columns]) :=
  Term.op (PrimOp.binary fun a b => Tensor.zipEach leading [rows, columns]
    (fun x y => Tensor.matrix fun i j => x.getScalar i * y.getScalar j) a b)
    (.cons left (.cons right .nil))

@[simp] theorem eval_outerTerm {α : Type} [Storage α] [Context α] {Γ : List Shape}
    (env : TensorPack α Γ) (leading : Shape) (rows columns : Nat)
    (left : Term Γ (leading ++ [rows])) (right : Term Γ (leading ++ [columns])) :
    Term.eval env (outerTerm leading rows columns left right) =
      Tensor.zipEach leading [rows, columns]
        (fun x y => Tensor.matrix fun i j => x.getScalar i * y.getScalar j)
        (Term.eval env left) (Term.eval env right) := rfl

/-- Multiply every coordinate with the scalar coefficient on the left. -/
def scaleTerm {Γ : List Shape} (shape : Shape)
    (coefficient : Term Γ []) (input : Term Γ shape) : Term Γ shape :=
  Term.op (PrimOp.binary fun c x => Tensor.map (c.item * ·) x)
    (.cons coefficient (.cons input .nil))

@[simp] theorem eval_scaleTerm {α : Type} [Storage α] [Context α] {Γ : List Shape}
    (env : TensorPack α Γ) (shape : Shape)
    (coefficient : Term Γ []) (input : Term Γ shape) :
    Term.eval env (scaleTerm shape coefficient input) =
      Tensor.map ((Term.eval env coefficient).item * ·) (Term.eval env input) := rfl

def fillTerm {Γ : List Shape} (shape : Shape) (value : Term Γ []) : Term Γ shape :=
  Term.op (PrimOp.unary fun x => Tensor.full shape x.item) (.cons value .nil)

@[simp] private theorem eval_fillTerm {α : Type} [Storage α] [Context α] {Γ : List Shape}
    (env : TensorPack α Γ) (shape : Shape) (value : Term Γ []) :
    Term.eval env (fillTerm shape value) = Tensor.full shape (Term.eval env value).item := rfl

/-- Compute the SiTU soft cap using reciprocal, pointwise multiplication and tanh. -/
def softCapTerm {Γ : List Shape} (shape : Shape) (cap : Term Γ [])
    (input : Term Γ shape) : Term Γ shape :=
  let inverse := Term.op (PrimOp.inv []) (.cons cap .nil)
  let scaled := Term.op (PrimOp.mul shape) (.cons input (.cons (fillTerm shape inverse) .nil))
  let cappedUnit := Term.op (PrimOp.tanh shape) (.cons scaled .nil)
  Term.op (PrimOp.mul shape) (.cons cappedUnit (.cons (fillTerm shape cap) .nil))

@[simp] theorem eval_softCapTerm {α : Type} [Storage α] [Context α] {Γ : List Shape}
    (env : TensorPack α Γ) (shape : Shape) (cap : Term Γ []) (input : Term Γ shape) :
    Term.eval env (softCapTerm shape cap input) =
      Tensor.tanh (Term.eval env input * Tensor.full shape (1 / (Term.eval env cap).item)) *
        Tensor.full shape (Term.eval env cap).item := by
  simp only [softCapTerm, Term.eval_op, Term.evalArgs, eval_fillTerm,
    PrimOp.mul_specFwd, PrimOp.inv_specFwd]
  change Tensor.tanh (Term.eval env input * Tensor.full shape
      (Tensor.reciprocal (Term.eval env cap)).item) * _ = _
  apply congrArg (· * Tensor.full shape (Term.eval env cap).item)
  congr 2
  simp [Tensor.reciprocal, Context.reciprocal, Tensor.item]

/-- Without leading axes, broadcasting has no effect on a vector-matrix product. -/
@[simp] theorem broadcastVecMat_scalar_specFwd {α : Type} [Storage α] [Context α]
    {rows columns : Nat} (vector : Tensor α [rows]) (matrix : Tensor α [rows, columns]) :
    (PrimOp.broadcastVecMat [] [] [] rows columns
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl [])).specFwd
        (.cons vector (.cons matrix .nil)) = Tensor.vecmat vector matrix := by
  simp [PrimOp.broadcastVecMat_specFwd, PrimOp.Internal.vecMatCommonBatchSpec, Tensor.zipEach]

@[simp] theorem rmsNormVectorSpec_eq_scale {α : Type} [Storage α] [Context α] {width : Nat}
    (hWidth : 0 < width) (input gamma : Tensor α [width]) :
    PrimOp.Internal.rmsNormVectorSpec hWidth input gamma = KimiK3.RMSNorm.scale input gamma :=
  (RMSNorm.scale_eq_scalePositive hWidth input gamma).symm

theorem rmsNormSpec_scalar_eq_scale {α : Type} [Storage α] [Context α] {width : Nat}
    (hWidth : 0 < width) (input gamma : Tensor α [width]) :
    PrimOp.rmsNormSpec [] hWidth gamma input = KimiK3.RMSNorm.scale input gamma :=
  rmsNormVectorSpec_eq_scale hWidth input gamma

/-- Append two tensors along their leading axis, retaining the other axes. -/
def concatLeadingTerm {Γ : List Shape} (left right : Nat) (rest : Shape)
    (a : Term Γ (left :: rest)) (b : Term Γ (right :: rest)) :
    Term Γ ((left + right) :: rest) :=
  Term.op (PrimOp.binary fun x y => Tensor.concat x y) (.cons a (.cons b .nil))

@[simp] theorem eval_concatLeadingTerm {α : Type} [Storage α] [Context α] {Γ : List Shape}
    (env : TensorPack α Γ) (left right : Nat) (rest : Shape)
    (a : Term Γ (left :: rest)) (b : Term Γ (right :: rest)) :
    Term.eval env (concatLeadingTerm left right rest a b) =
      Tensor.concat (Term.eval env a) (Term.eval env b) := rfl

end KimiK3.GraphSpec
