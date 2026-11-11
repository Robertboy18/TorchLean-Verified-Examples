/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.Sequence.AttnRes
public import KimiK3.GraphSpec.Core

/-!
# Block attention residual graph

The decoder keeps its state as a list. When the visible sources are packed into tensor rows,
AttnRes composes TorchLean normalization, transpose, exponential, reduction, and matrix products.
The shared exponential scores supply both the denominator and the weighted output.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec
namespace AttnRes

open TorchLean
open TorchLean.Tensor

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
  let kernels : Term Γ [sources] := ⟨fun env =>
    let normalized : Tensor _ [sources, modelDim] := Tensor.dim fun row =>
      RMSNorm.scalePositive hModel (Tensor.unstack (values.eval env) row) (Tensor.ones [modelDim])
    let transposed := normalized.permute (Shape.adjacentSwap [sources, modelDim] 0)
      [modelDim, sources]
    Tensor.exp (Tensor.vecmat (query.eval env) transposed)⟩
  Term.let1 kernels <|
    let boundKernels : Term (Γ ++ [[sources]]) [sources] := Term.var (Var.last Γ)
    let boundValues := Term.weakenRight (t := [sources]) values
    ⟨fun env =>
      let kernels := boundKernels.eval env
      let denominator := Tensor.sum kernels
      let weights := Tensor.map (fun weight => weight / denominator) kernels
      Tensor.vecmat weights (boundValues.eval env)⟩

/-- Evaluating the term gives the packed AttnRes semantics `attendPacked`. -/
theorem eval_term {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (sources modelDim : Nat) (hModel : 0 < modelDim)
    (query : Term Γ [modelDim])
    (values : Term Γ [sources, modelDim]) :
    Term.eval env (term sources modelDim hModel query values) =
      KimiK3.AttnRes.attendPacked hModel (Term.eval env query) (Term.eval env values) := by
  unfold term
  rw [Term.eval_let1]
  change Tensor.vecmat (Tensor.map (fun weight => weight / Tensor.sum _) _) _ = _
  simp only [Term.eval_var_last_append, Term.eval_weakenRight]
  rfl

/-- Standalone graph wrapper for one packed AttnRes retrieval. -/
def model (sources modelDim : Nat) (hModel : 0 < modelDim) :
    Model [] (Inputs sources modelDim) [modelDim] :=
  match Args.vars (Inputs sources modelDim) with
  | .cons query (.cons values .nil) =>
      { initParams := .nil
        body := term sources modelDim hModel query values }

end AttnRes
end GraphSpec
end KimiK3
