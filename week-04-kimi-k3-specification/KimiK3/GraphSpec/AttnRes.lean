/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.Sequence.AttnRes
public import KimiK3.GraphSpec.Primitives

/-!
# Block attention residual graph

This module lowers the fixed-shape form of K3's AttnRes retrieval to TorchLean's typed DAG. The
architecture specification keeps the decoder state as a list, and at an execution boundary its
visible sources are packed as the rows of a tensor. There is no AttnRes primitive: the graph is
built from normalization, transpose, exponential, reduction, and matrix multiplication nodes that
TorchLean already has.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec
namespace AttnRes

open Spec TorchLean
open TorchLean.Tensor
open NN.GraphSpec.DAG

/-- A query vector and a packed matrix of visible depth representations. -/
abbrev Inputs (sources modelDim : Nat) : List Shape :=
  [[modelDim], [sources, modelDim]]

/-- Package a query and packed source matrix in the graph's input order. -/
def inputs {α : Type} [Storage α] {sources modelDim : Nat}
    (query : Tensor α [modelDim])
    (values : Tensor α [sources, modelDim]) :
    TorchLean.TensorPack α (Inputs sources modelDim) :=
  .cons query (.cons values .nil)

/-- AttnRes retrieval as a reusable term in an arbitrary surrounding graph context. -/
def term {Γ : List Shape} (sources modelDim : Nat)
    (hModel : 0 < modelDim) (query : Term Γ [modelDim])
    (values : Term Γ [sources, modelDim]) :
    Term Γ [modelDim] :=
  let unitScale := Term.op (NN.GraphSpec.DAG.PrimOp.one [modelDim]) .nil
  let normalized := Term.op
    (NN.GraphSpec.DAG.PrimOp.rmsNorm [sources] modelDim hModel)
    (.cons values (.cons unitScale .nil))
  let transposed := Term.op (NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth
      [sources, modelDim] 0)
    (.cons normalized .nil)
  let scores := Term.op (PrimOp.broadcastVecMat [] [] [] modelDim sources
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
    (.cons query (.cons transposed .nil))
  let kernels := Term.op (NN.GraphSpec.DAG.PrimOp.exp [sources])
    (.cons scores .nil)
  -- Bind the exponentials once: the denominator and weighted values use the same scores.
  Term.let1 kernels <|
    let boundKernels : Term (Γ ++ [[sources]]) [sources] :=
      Term.var (Var.last Γ)
    let denominator := Term.op (NN.GraphSpec.DAG.PrimOp.sum [sources])
      (.cons boundKernels .nil)
    let inverse := Term.op (NN.GraphSpec.DAG.PrimOp.inv []) (.cons denominator .nil)
    let weights := scaleTerm [sources] inverse boundKernels
    Term.op (PrimOp.broadcastVecMat [] [] [] sources modelDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
      (.cons weights (.cons (Term.weakenRight values) .nil))

private theorem eval_let1 {Γ : List Shape} {σ τ : Shape} {α : Type} [Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (value : Term Γ σ) (body : Term (Γ ++ [σ]) τ) :
    Term.eval env (.let1 value body) =
      Term.eval (TorchLean.TensorPack.append env (.cons (Term.eval env value) .nil)) body := by
  rfl

/-- Evaluating the term gives the packed AttnRes semantics `attendPacked`. -/
theorem eval_term {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (sources modelDim : Nat) (hModel : 0 < modelDim)
    (query : Term Γ [modelDim])
    (values : Term Γ [sources, modelDim]) :
    Term.eval env (term sources modelDim hModel query values) =
      KimiK3.AttnRes.attendPacked hModel (Term.eval env query) (Term.eval env values) := by
  -- The let-bound exponentials extend the environment, and the original values are weakened.
  unfold term
  rw [eval_let1]
  simp only [Term.eval_op, Term.evalArgs, eval_scaleTerm,
    Term.eval_var_last_append, Term.eval_weakenRight]
  simp only [NN.GraphSpec.DAG.PrimOp.rmsNorm_specFwd,
    NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth_specFwd,
    NN.GraphSpec.DAG.PrimOp.inv_specFwd,
    PrimOp.broadcastVecMat, NN.GraphSpec.DAG.PrimOp.exp,
    NN.GraphSpec.DAG.PrimOp.sum, NN.GraphSpec.DAG.PrimOp.one]
  -- Match normalized scoring and the weighted sum of the original, unnormalized values.
  simp [KimiK3.AttnRes.attendPacked, NN.GraphSpec.DAG.PrimOp.rmsNormSpec,
    NN.GraphSpec.DAG.PrimOp.Internal.rmsNormVectorSpec, RMSNorm.scalePositive,
    PrimOp.Internal.mapFinalAxis, Tensor.mapLeading, Tensor.stackLeading, Tensor.ones,
    Tensor.invSpec, div_eq_mul_inv, mul_comm, Spec.get]

/-- Standalone graph wrapper for one packed AttnRes retrieval. -/
def model (sources modelDim : Nat) (hModel : 0 < modelDim) :
    NN.GraphSpec.DAG.Model [] (Inputs sources modelDim) [modelDim] :=
  match Args.vars (Inputs sources modelDim) with
  | .cons query (.cons values .nil) =>
      { initParams := .nil
        body := term sources modelDim hModel query values }

end AttnRes
end GraphSpec
end KimiK3
