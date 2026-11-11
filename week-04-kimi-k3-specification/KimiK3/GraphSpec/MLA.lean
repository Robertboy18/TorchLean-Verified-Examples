/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Core
public import KimiK3.Sequence.MLA

/-!
# Fixed-cache Gated MLA graph

This module composes a complete Kimi K3 Gated MLA token step from shaped tensor operations. It
includes the query and KV down-projections, normalization, cache append, every head's key/value
reconstruction and attention, the output gate, and the final projection.

The cache length is part of every tensor shape. The composition returns the same three outputs as
`GatedMLA.stepFixed`. `KimiK3.GatedMLA.Cache` represents the list-based streaming step.

K3 uses NoPE attention. Its two score terms come from the content key and the shared unrotated key.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec
namespace MLA

open TorchLean
open TorchLean.Tensor

private theorem eval_term {Γ : List Shape} {s : Shape} {α : Type}
    [Storage α] [Context α] (env : TensorPack α Γ)
    (f : ∀ {β : Type}, [Storage β] → [Context β] → TensorPack β Γ → Tensor β s) :
    Term.eval env (⟨f⟩ : Term Γ s) = f env := rfl

/-! ## Complete Gated MLA step -/

/-- Parameters of one Gated MLA layer, with per-head matrices packed on a leading head axis. -/
abbrev LayerParams
    (modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim : Nat) :
    List Shape :=
  [ [modelDim, queryLatentDim],
    [queryLatentDim],
    [modelDim, kvLatentDim],
    [kvLatentDim],
    [modelDim, sharedKeyDim],
    [heads, queryLatentDim, contentKeyDim],
    [heads, queryLatentDim, sharedKeyDim],
    [heads, kvLatentDim, contentKeyDim],
    [heads, kvLatentDim, valueDim],
    [modelDim, heads, valueDim],
    [heads, valueDim, modelDim] ]

/-- Previous latent cache, previous shared-key cache, current token, and score scale. -/
abbrev StepInputs (pastTokens modelDim kvLatentDim sharedKeyDim : Nat) : List Shape :=
  [ [pastTokens, kvLatentDim],
    [pastTokens, sharedKeyDim],
    [modelDim],
    [] ]

/-- Updated latent cache, updated shared-key cache, and the current MLA output. -/
abbrev StepOutputs (pastTokens modelDim kvLatentDim sharedKeyDim : Nat) : List Shape :=
  [ [pastTokens + 1, kvLatentDim],
    [pastTokens + 1, sharedKeyDim],
    [modelDim] ]

/-- Zero-filled defaults for the standalone GraphSpec package. -/
def initialLayerParams
    (modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim : Nat) :
    TorchLean.TensorPack Float
      (LayerParams modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim) :=
  TorchLean.TensorPack.zero

/-- Package a theorem-level Gated MLA layer in the graph parameter ABI. -/
def layerParameters {α : Type} [Storage α]
    {modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim : Nat}
    (layer :
      GatedMLA α modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim) :
    TorchLean.TensorPack α
      (LayerParams modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim) :=
  .cons layer.queryDown <| .cons layer.queryNormScale <| .cons layer.kvDown <|
    .cons layer.kvNormScale <| .cons layer.sharedKeyDown <|
      .cons layer.queryContentUp <| .cons layer.querySharedUp <|
        .cons layer.keyUp <| .cons layer.valueUp <| .cons layer.gateWeight <|
          .cons layer.outputWeight .nil

/-- Package one fixed-context causal step in the graph input ABI. -/
def stepInputs {α : Type} [Storage α] {pastTokens modelDim kvLatentDim sharedKeyDim : Nat}
    (pastLatentCache : Tensor α [pastTokens, kvLatentDim])
    (pastSharedKeyCache : Tensor α [pastTokens, sharedKeyDim])
    (x : Tensor α [modelDim]) (scoreScale : α) :
    TorchLean.TensorPack α (StepInputs pastTokens modelDim kvLatentDim sharedKeyDim) :=
  .cons pastLatentCache <| .cons pastSharedKeyCache <| .cons x <|
    .cons (Tensor.scalar scoreScale) .nil

/-- Compose one Gated MLA token step with a fixed cache length. -/
def stepModel (pastTokens modelDim heads queryLatentDim kvLatentDim contentKeyDim
    sharedKeyDim valueDim : Nat) (hQueryLatent : 0 < queryLatentDim)
    (hKVLatent : 0 < kvLatentDim) :
    MultiModel
      (LayerParams modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
      (StepInputs pastTokens modelDim kvLatentDim sharedKeyDim)
      (StepOutputs pastTokens modelDim kvLatentDim sharedKeyDim) :=
  let params :=
    LayerParams modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim
  let inputs := StepInputs pastTokens modelDim kvLatentDim sharedKeyDim
  let Γ := params ++ inputs
  let parameterTerms : Args Γ params :=
    Args.rename (Var.inLeft inputs) (Args.vars params)
  let inputTerms : Args Γ inputs :=
    Args.rename (Var.inRight params) (Args.vars inputs)
  let queryDown := Args.get parameterTerms .head
  let queryNormScale := Args.get parameterTerms (.tail .head)
  let kvDown := Args.get parameterTerms (.tail (.tail .head))
  let kvNormScale := Args.get parameterTerms (.tail (.tail (.tail .head)))
  let sharedKeyDown := Args.get parameterTerms (.tail (.tail (.tail (.tail .head))))
  let queryContentUp := Args.get parameterTerms
    (.tail (.tail (.tail (.tail (.tail .head)))))
  let querySharedUp := Args.get parameterTerms
    (.tail (.tail (.tail (.tail (.tail (.tail .head))))))
  let keyUp := Args.get parameterTerms
    (.tail (.tail (.tail (.tail (.tail (.tail (.tail .head)))))))
  let valueUp := Args.get parameterTerms
    (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail .head))))))))
  let gateWeight := Args.get parameterTerms
    (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail .head)))))))))
  let outputWeight := Args.get parameterTerms
    (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail .head))))))))))
  let pastLatentCache := Args.get inputTerms .head
  let pastSharedKeyCache := Args.get inputTerms (.tail .head)
  let x := Args.get inputTerms (.tail (.tail .head))
  let scoreScale := Args.get inputTerms (.tail (.tail (.tail .head)))
  let vecmat {m n : Nat} (vector : Term Γ [m]) (matrix : Term Γ [m, n]) : Term Γ [n] :=
    ⟨fun env => Tensor.vecmat (vector.eval env) (matrix.eval env)⟩
  let rmsNorm {n : Nat} (h : 0 < n) (x gamma : Term Γ [n]) : Term Γ [n] :=
    ⟨fun env => RMSNorm.scalePositive h (x.eval env) (gamma.eval env)⟩
  let reshape {source target : Shape} (x : Term Γ source)
      (h : source.size = target.size) : Term Γ target :=
    ⟨fun env => Tensor.reshape (x.eval env) target h⟩
  let concat {n m : Nat} {s : Shape} (past : Term Γ (n :: s))
      (current : Term Γ (m :: s)) : Term Γ ((n + m) :: s) :=
    ⟨fun env => Tensor.concat (past.eval env) (current.eval env)⟩
  let repeatHead {s : Shape} (x : Term Γ s) : Term Γ (heads :: s) :=
    ⟨fun env => Tensor.repeatLeading heads (x.eval env)⟩
  let matmul {m n p : Nat} (a : Term Γ [heads, m, n]) (b : Term Γ [heads, n, p]) :
      Term Γ [heads, m, p] :=
    ⟨fun env => Tensor.batchMatmul (.refl [heads]) (.refl [heads]) (a.eval env) (b.eval env)⟩
  let transpose {m n : Nat} (a : Term Γ [heads, m, n]) : Term Γ [heads, n, m] :=
    ⟨fun env => Tensor.permute (a.eval env) (Shape.adjacentSwap [heads, m, n] 1) [heads, n, m]
      (Shape.permute_adjacentSwap [heads, m, n] 1).symm⟩
  let queryLatent := rmsNorm hQueryLatent (vecmat x queryDown) queryNormScale
  let currentKV := rmsNorm hKVLatent (vecmat x kvDown) kvNormScale
  let currentShared := vecmat x sharedKeyDown
  let currentKVRow : Term Γ [1, kvLatentDim] := reshape currentKV (by simp [Shape.size])
  let currentSharedRow : Term Γ [1, sharedKeyDim] := reshape currentShared (by simp [Shape.size])
  let latentCache := concat pastLatentCache currentKVRow
  let sharedCache := concat pastSharedKeyCache currentSharedRow
  let queryRow : Term Γ [1, queryLatentDim] := ⟨fun env => Tensor.dim fun _ => queryLatent.eval env⟩
  let queryBatch := repeatHead queryRow
  let latentBatch := repeatHead latentCache
  let sharedBatch := repeatHead sharedCache
  let queryContent := matmul queryBatch queryContentUp
  let queryShared := matmul queryBatch querySharedUp
  let keys := matmul latentBatch keyUp
  let values := matmul latentBatch valueUp
  let contentScores := matmul queryContent (transpose keys)
  let sharedScores := matmul queryShared (transpose sharedBatch)
  let scoreShape : Shape := [heads, 1, pastTokens + 1]
  let weights : Term Γ scoreShape := ⟨fun env =>
    let scores := Tensor.zipWith (· + ·) (contentScores.eval env) (sharedScores.eval env)
    let scaled := Tensor.zipWith (· * ·) (Tensor.full scoreShape (scoreScale.eval env).item) scores
    Tensor.softmax scaled ⟨2, by simp [scoreShape]⟩⟩
  let headOutput3 := matmul weights values
  let headOutput : Term Γ [heads, valueDim] := reshape headOutput3 (by simp [Shape.size])
  let gateMatrix : Term Γ [modelDim, heads * valueDim] := reshape gateWeight (by simp [Shape.size])
  let gateFlat := vecmat x gateMatrix
  let gateUnactivated : Term Γ [heads, valueDim] := reshape gateFlat (by simp [Shape.size])
  let gatedHeads : Term Γ [heads, valueDim] := ⟨fun env =>
    Tensor.zipWith (· * ·) (Tensor.sigmoid (gateUnactivated.eval env)) (headOutput.eval env)⟩
  let gatedFlat : Term Γ [heads * valueDim] := reshape gatedHeads (by simp [Shape.size])
  let outputMatrix : Term Γ [heads * valueDim, modelDim] :=
    reshape outputWeight (by simp [Shape.size, Nat.mul_assoc])
  let output := vecmat gatedFlat outputMatrix
  { initParams :=
      initialLayerParams modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim
        valueDim
    body := Block.ret (.cons latentCache (.cons sharedCache (.cons output .nil))) }

/-- Evaluate the typed MLA token-step graph on concrete parameters and inputs. -/
noncomputable def stepGraphOutputs
    (pastTokens modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim : Nat)
    (hQueryLatent : 0 < queryLatentDim) (hKVLatent : 0 < kvLatentDim)
    (layer :
      GatedMLA ℝ modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (pastLatentCache : Tensor ℝ [pastTokens, kvLatentDim])
    (pastSharedKeyCache : Tensor ℝ [pastTokens, sharedKeyDim])
    (x : Tensor ℝ [modelDim]) (scoreScale : ℝ) :
    TorchLean.TensorPack ℝ (StepOutputs pastTokens modelDim kvLatentDim sharedKeyDim) :=
  (stepModel pastTokens modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim
    valueDim hQueryLatent hKVLatent).specFwd (layerParameters layer)
      (stepInputs pastLatentCache pastSharedKeyCache x scoreScale)

/-- Package the reference MLA token-step result in the graph's output shape list. -/
noncomputable def stepFixedOutputs
    (pastTokens modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim : Nat)
    (layer :
      GatedMLA ℝ modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (pastLatentCache : Tensor ℝ [pastTokens, kvLatentDim])
    (pastSharedKeyCache : Tensor ℝ [pastTokens, sharedKeyDim])
    (x : Tensor ℝ [modelDim]) (scoreScale : ℝ) :
    TorchLean.TensorPack ℝ (StepOutputs pastTokens modelDim kvLatentDim sharedKeyDim) :=
  let result := layer.stepFixed pastTokens pastLatentCache
    pastSharedKeyCache x scoreScale
  .cons result.1 (.cons result.2.1 (.cons result.2.2 .nil))

/-- Graph and reference token step return the same extended latent cache, extended shared-key
cache, and output. Both paths append the current token before attending. -/
theorem stepModel_specFwd_eq_stepFixed
    {pastTokens modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim : Nat}
    (hQueryLatent : 0 < queryLatentDim) (hKVLatent : 0 < kvLatentDim)
    (layer :
      GatedMLA ℝ modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (pastLatentCache : Tensor ℝ [pastTokens, kvLatentDim])
    (pastSharedKeyCache : Tensor ℝ [pastTokens, sharedKeyDim])
    (x : Tensor ℝ [modelDim]) (scoreScale : ℝ) :
    stepGraphOutputs pastTokens modelDim heads queryLatentDim kvLatentDim contentKeyDim
      sharedKeyDim valueDim hQueryLatent hKVLatent layer pastLatentCache pastSharedKeyCache x
        scoreScale =
      stepFixedOutputs pastTokens modelDim heads queryLatentDim kvLatentDim contentKeyDim
        sharedKeyDim valueDim layer pastLatentCache pastSharedKeyCache x scoreScale := by
  simp only [stepGraphOutputs, stepFixedOutputs, MultiModel.specFwd, stepModel,
    Block.eval_ret, Term.evalArgs, eval_term, Term.eval_get, Term.evalArgs_rename_inLeft,
    Term.evalArgs_rename_inRight, Term.evalArgs_vars, Env.tget, layerParameters, stepInputs,
    Tensor.item_scalar, GatedMLA.stepFixed, RMSNorm.scale_eq_scalePositive hQueryLatent,
    RMSNorm.scale_eq_scalePositive hKVLatent]

end MLA
end GraphSpec
end KimiK3
