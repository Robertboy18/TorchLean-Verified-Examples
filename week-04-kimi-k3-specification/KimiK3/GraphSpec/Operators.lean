/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.Common
public import KimiK3.GraphSpec.Core
public import TorchLean.Tensor.LinearAlgebra.Algorithms

/-!
# Tensor operations used by the K3 equations

The graph's local adapters package tensors for TorchLean's operators. Matrix
products, activations, movement and reductions use the same tensor functions
as the rest of the example.
-/

@[expose] public section

namespace KimiK3.GraphSpec.PrimOp

open TorchLean

def unary {s t : Shape}
    (f : ∀ {α : Type}, [Storage α] → [Context α] → Tensor α s → Tensor α t) :
    PrimOp [s] t :=
  ⟨fun {_} _ _ args => f args.head⟩

def binary {s t u : Shape}
    (f : ∀ {α : Type}, [Storage α] → [Context α] → Tensor α s → Tensor α t → Tensor α u) :
    PrimOp [s, t] u :=
  ⟨fun {_} _ _ args => f args.head args.tail.head⟩

def zero (s : Shape) : PrimOp [] s := ⟨fun _ => Tensor.full s 0⟩

def add (s : Shape) : PrimOp [s, s] s :=
  binary fun a b => Ops.pureApply (.add .scalar) inferInstance (a, b, ())

def sub (s : Shape) : PrimOp [s, s] s :=
  binary fun a b => Ops.pureApply (.sub .scalar) inferInstance (a, b, ())

def mul (s : Shape) : PrimOp [s, s] s :=
  binary fun a b => Ops.pureApply (.mul .scalar) inferInstance (a, b, ())

@[simp] theorem zero_specFwd {α : Type} [Storage α] [Context α] (s : Shape) :
    (zero s).specFwd (.nil : TensorPack α []) = Tensor.full s 0 := rfl

@[simp] theorem add_specFwd {α : Type} [Storage α] [Context α] {s : Shape} (a b : Tensor α s) :
    (add s).specFwd (.cons a (.cons b .nil)) = a + b := rfl

@[simp] theorem sub_specFwd {α : Type} [Storage α] [Context α] {s : Shape} (a b : Tensor α s) :
    (sub s).specFwd (.cons a (.cons b .nil)) = a - b := rfl

@[simp] theorem mul_specFwd {α : Type} [Storage α] [Context α] {s : Shape} (a b : Tensor α s) :
    (mul s).specFwd (.cons a (.cons b .nil)) = a * b := rfl

def exp (s : Shape) : PrimOp [s] s := unary fun x => Tensor.exp x
def inv (s : Shape) : PrimOp [s] s := unary fun x => Tensor.reciprocal x
def tanh (s : Shape) : PrimOp [s] s := unary fun x => Tensor.tanh x
def sigmoid (s : Shape) : PrimOp [s] s := unary fun x => Tensor.sigmoid x
def silu (s : Shape) : PrimOp [s] s := unary fun x => Tensor.map Activation.silu x

@[simp] theorem inv_specFwd {α : Type} [Storage α] [Context α] {s : Shape} (x : Tensor α s) :
    (inv s).specFwd (.cons x .nil) = Tensor.reciprocal x := rfl

@[simp] theorem sigmoid_specFwd {α : Type} [Storage α] [Context α] {s : Shape} (x : Tensor α s) :
    (sigmoid s).specFwd (.cons x .nil) = Tensor.sigmoid x := rfl

@[simp] theorem silu_specFwd {α : Type} [Storage α] [Context α] {s : Shape} (x : Tensor α s) :
    (silu s).specFwd (.cons x .nil) = Tensor.map Activation.silu x := rfl

def reshape (source target : Shape) (h : source.size = target.size) : PrimOp [source] target :=
  unary fun x => Tensor.reshape x target h

@[simp] theorem reshape_specFwd {α : Type} [Storage α] [Context α] {s t : Shape}
    (h : s.size = t.size) (x : Tensor α s) :
    (reshape s t h).specFwd (.cons x .nil) = Tensor.reshape x t h := rfl

def swapAdjacentAtDepth (s : Shape) (depth : Nat) : PrimOp [s] (s.swapAdjacentAtDepth depth) :=
  unary fun x =>
    Tensor.permute x (s.adjacentSwap depth) _ (s.permute_adjacentSwap depth).symm

@[simp] theorem swapAdjacentAtDepth_specFwd {α : Type} [Storage α] [Context α]
    (s : Shape) (depth : Nat) (x : Tensor α s) :
    (swapAdjacentAtDepth s depth).specFwd (.cons x .nil) =
      Tensor.permute x (s.adjacentSwap depth) _ (s.permute_adjacentSwap depth).symm := rfl

def reduceSum (s : Shape) (axis : Nat) : PrimOp [s] (s.eraseIdx axis) :=
  unary fun x => Tensor.reduceDimWith Tensor.sum s axis x

namespace Internal

def vecMatCommonBatchSpec {α : Type} [Storage α] [Context α] {rows columns : Nat}
    (batch : Shape) (vectors : Tensor α (batch ++ [rows]))
    (matrices : Tensor α (batch ++ [rows, columns])) : Tensor α (batch ++ [columns]) :=
  Tensor.zipEach batch [columns] Tensor.vecmat vectors matrices

def rmsNormVectorSpec {α : Type} [Storage α] [Context α] {width : Nat}
    (hWidth : 0 < width) (input gamma : Tensor α [width]) : Tensor α [width] :=
  RMSNorm.scalePositive hWidth input gamma

end Internal

def broadcastVecMat (vectorBatch matrixBatch batch : Shape) (rows columns : Nat)
    (a : vectorBatch.CanBroadcastTo batch) (b : matrixBatch.CanBroadcastTo batch) :
    PrimOp [vectorBatch ++ [rows], matrixBatch ++ [rows, columns]] (batch ++ [columns]) :=
  binary fun x y =>
    let a' := Shape.CanBroadcastTo.concat_right [rows] a
    let b' := Shape.CanBroadcastTo.concat_right [rows, columns] b
    Internal.vecMatCommonBatchSpec batch
      (Tensor.broadcast x _ a'.dims a'.broadcastDims)
      (Tensor.broadcast y _ b'.dims b'.broadcastDims)

@[simp] theorem broadcastVecMat_specFwd {α : Type} [Storage α] [Context α]
    {vectorBatch matrixBatch batch : Shape} {rows columns : Nat}
    (a : vectorBatch.CanBroadcastTo batch) (b : matrixBatch.CanBroadcastTo batch)
    (x : Tensor α (vectorBatch ++ [rows])) (y : Tensor α (matrixBatch ++ [rows, columns])) :
    (broadcastVecMat vectorBatch matrixBatch batch rows columns a b).specFwd
        (.cons x (.cons y .nil)) =
      Internal.vecMatCommonBatchSpec batch
        (Tensor.broadcast x _ (Shape.CanBroadcastTo.concat_right [rows] a).dims
          (Shape.CanBroadcastTo.concat_right [rows] a).broadcastDims)
        (Tensor.broadcast y _ (Shape.CanBroadcastTo.concat_right [rows, columns] b).dims
          (Shape.CanBroadcastTo.concat_right [rows, columns] b).broadcastDims) := rfl

def rmsNormSpec {α : Type} [Storage α] [Context α] (leading : Shape) {width : Nat}
    (hWidth : 0 < width) (gamma : Tensor α [width]) :
    Tensor α (leading ++ [width]) → Tensor α (leading ++ [width]) :=
  Tensor.mapLeading leading (fun input => Internal.rmsNormVectorSpec hWidth input gamma)

def rmsNorm (leading : Shape) (width : Nat) (hWidth : 0 < width) :
    PrimOp [leading ++ [width], [width]] (leading ++ [width]) :=
  binary fun input gamma => rmsNormSpec leading hWidth gamma input

@[simp] theorem rmsNorm_specFwd {α : Type} [Storage α] [Context α]
    {leading : Shape} {width : Nat} (hWidth : 0 < width)
    (input : Tensor α (leading ++ [width])) (gamma : Tensor α [width]) :
    (rmsNorm leading width hWidth).specFwd (.cons input (.cons gamma .nil)) =
      rmsNormSpec leading hWidth gamma input := rfl

def normalizeL2Regularized (leading : Shape) (width : Nat) (_hWidth : 0 < width) :
    PrimOp [leading ++ [width], []] (leading ++ [width]) :=
  binary fun input epsilon => Tensor.mapLeading leading
    (fun row => Tensor.normalizeL2 row epsilon.item) input

end KimiK3.GraphSpec.PrimOp

namespace KimiK3.GraphSpec.Term

open TorchLean

/-- Add expressions of the same shape in their given order. -/
def sum {Γ : List Shape} (s : Shape) (terms : List (Term Γ s)) : Term Γ s :=
  terms.foldl (fun total term => Term.op (PrimOp.add s) (.cons total (.cons term .nil)))
    (Term.op (PrimOp.zero s) .nil)

theorem eval_sum {Γ : List Shape} {s : Shape} {α : Type} [Storage α] [Context α]
    (env : TensorPack α Γ) (terms : List (Term Γ s)) :
    Term.eval env (sum s terms) =
      terms.foldl (fun total term => total + Term.eval env term) (Tensor.full s 0) := by
  exact (List.foldl_hom (Term.eval env)
    (g₁ := fun total term => Term.op (PrimOp.add s) (.cons total (.cons term .nil)))
    (g₂ := fun total term => total + Term.eval env term)
    (l := terms) (init := Term.op (PrimOp.zero s) .nil) (fun _ _ => rfl)).symm

end KimiK3.GraphSpec.Term
