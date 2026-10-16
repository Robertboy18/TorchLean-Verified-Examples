/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Primitives
public import KimiK3.Sequence.MLA

/-!
# Fixed-cache Gated MLA graph

This module lowers a complete Kimi K3 Gated MLA token step to TorchLean's typed DAG. The graph
includes the query and KV down-projections, normalization, cache append, every head's key/value
reconstruction and attention, the output gate, and the final projection.

The cache length is part of every tensor shape, which is what compilation, autograd, and backend
planning need. `KimiK3.GatedMLA.Cache` is a separate list-based streaming view. The graph is proved
equal to `GatedMLA.stepFixed`, and no theorem yet says that packing a list cache and running
`GatedMLA.step` gives the same result.

K3 uses NoPE attention, so the graph has no rotary-position operation and its two score terms come
from the content key and the shared unrotated key.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec
namespace MLA

open Spec TorchLean
open TorchLean.Tensor
open NN.GraphSpec.DAG
open Runtime.Autograd.Torch

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

/-- One fixed-context Gated MLA token step as a typed multi-output DAG. -/
def stepModel (pastTokens modelDim heads queryLatentDim kvLatentDim contentKeyDim
    sharedKeyDim valueDim : Nat) (hQueryLatent : 0 < queryLatentDim)
    (hKVLatent : 0 < kvLatentDim) :
    NN.GraphSpec.DAG.MultiModel
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
  let queryProjected : Term Γ [queryLatentDim] :=
    Term.op (PrimOp.broadcastVecMat [] [] [] modelDim queryLatentDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
      (.cons x (.cons queryDown .nil))
  let queryLatent : Term Γ [queryLatentDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.rmsNorm [] queryLatentDim hQueryLatent)
      (.cons queryProjected (.cons queryNormScale .nil))
  let kvProjected : Term Γ [kvLatentDim] :=
    Term.op (PrimOp.broadcastVecMat [] [] [] modelDim kvLatentDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
      (.cons x (.cons kvDown .nil))
  let currentKV : Term Γ [kvLatentDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.rmsNorm [] kvLatentDim hKVLatent)
      (.cons kvProjected (.cons kvNormScale .nil))
  let currentShared : Term Γ [sharedKeyDim] :=
    Term.op (PrimOp.broadcastVecMat [] [] [] modelDim sharedKeyDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
      (.cons x (.cons sharedKeyDown .nil))
  let currentKVRow : Term Γ [1, kvLatentDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons currentKV .nil)
  let currentSharedRow : Term Γ [1, sharedKeyDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons currentShared .nil)
  let latentCache : Term Γ [pastTokens + 1, kvLatentDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.concatAxis
      [pastTokens, kvLatentDim] 0 pastTokens 1)
      (.cons pastLatentCache (.cons currentKVRow .nil))
  let sharedKeyCache : Term Γ [pastTokens + 1, sharedKeyDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.concatAxis
      [pastTokens, sharedKeyDim] 0 pastTokens 1)
      (.cons pastSharedKeyCache (.cons currentSharedRow .nil))
  let queryBatchSource : Term Γ [1, 1, queryLatentDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons queryLatent .nil)
  let queryBatch : Term Γ [heads, 1, queryLatentDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.broadcast
      (Shape.CanBroadcastTo.dim_1_to_n
        (Shape.CanBroadcastTo.dim_eq
          (Shape.CanBroadcastTo.dim_eq
            (Shape.CanBroadcastTo.scalarTo []))))) (.cons queryBatchSource .nil)
  let latentBatchSource :
      Term Γ [1, pastTokens + 1, kvLatentDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons latentCache .nil)
  let latentBatch :
      Term Γ [heads, pastTokens + 1, kvLatentDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.broadcast
      (Shape.CanBroadcastTo.dim_1_to_n
        (Shape.CanBroadcastTo.dim_eq
          (Shape.CanBroadcastTo.dim_eq
            (Shape.CanBroadcastTo.scalarTo []))))) (.cons latentBatchSource .nil)
  let sharedBatchSource :
      Term Γ [1, pastTokens + 1, sharedKeyDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons sharedKeyCache .nil)
  let sharedBatch :
      Term Γ [heads, pastTokens + 1, sharedKeyDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.broadcast
      (Shape.CanBroadcastTo.dim_1_to_n
        (Shape.CanBroadcastTo.dim_eq
          (Shape.CanBroadcastTo.dim_eq
            (Shape.CanBroadcastTo.scalarTo []))))) (.cons sharedBatchSource .nil)
  let queryContent : Term Γ [heads, 1, contentKeyDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads]
      1 queryLatentDim contentKeyDim (.refl [heads]) (.refl [heads]))
      (.cons queryBatch (.cons queryContentUp .nil))
  let queryShared : Term Γ [heads, 1, sharedKeyDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads]
      1 queryLatentDim sharedKeyDim (.refl [heads]) (.refl [heads]))
      (.cons queryBatch (.cons querySharedUp .nil))
  let keys :
      Term Γ [heads, pastTokens + 1, contentKeyDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads]
      (pastTokens + 1) kvLatentDim contentKeyDim (.refl [heads]) (.refl [heads]))
      (.cons latentBatch (.cons keyUp .nil))
  let values : Term Γ [heads, pastTokens + 1, valueDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads]
      (pastTokens + 1) kvLatentDim valueDim (.refl [heads]) (.refl [heads]))
      (.cons latentBatch (.cons valueUp .nil))
  let keysTranspose :
      Term Γ [heads, contentKeyDim, pastTokens + 1] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth
      [heads, pastTokens + 1, contentKeyDim] 1)
      (.cons keys .nil)
  let sharedTranspose :
      Term Γ [heads, sharedKeyDim, pastTokens + 1] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth
      [heads, pastTokens + 1, sharedKeyDim] 1)
      (.cons sharedBatch .nil)
  let contentScores : Term Γ [heads, 1, pastTokens + 1] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads]
      1 contentKeyDim (pastTokens + 1) (.refl [heads]) (.refl [heads]))
      (.cons queryContent (.cons keysTranspose .nil))
  let sharedScores : Term Γ [heads, 1, pastTokens + 1] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads]
      1 sharedKeyDim (pastTokens + 1) (.refl [heads]) (.refl [heads]))
      (.cons queryShared (.cons sharedTranspose .nil))
  let scoreShape : Shape := [heads, 1, pastTokens + 1]
  let scores : Term Γ scoreShape :=
    Term.op (NN.GraphSpec.DAG.PrimOp.add scoreShape)
      (.cons contentScores (.cons sharedScores .nil))
  let scaledScores : Term Γ scoreShape :=
    scaleTerm scoreShape scoreScale scores
  let weights : Term Γ scoreShape :=
    Term.op (NN.GraphSpec.DAG.PrimOp.softmax scoreShape 2) (.cons scaledScores .nil)
  let headOutput3 : Term Γ [heads, 1, valueDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads]
      1 (pastTokens + 1) valueDim (.refl [heads]) (.refl [heads]))
      (.cons weights (.cons values .nil))
  let headOutput : Term Γ [heads, valueDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons headOutput3 .nil)
  let gateMatrix : Term Γ [modelDim, heads * valueDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons gateWeight .nil)
  let gateFlat : Term Γ [heads * valueDim] :=
    Term.op (PrimOp.broadcastVecMat [] [] [] modelDim (heads * valueDim)
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
      (.cons x (.cons gateMatrix .nil))
  let gateUnactivated : Term Γ [heads, valueDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons gateFlat .nil)
  let gate : Term Γ [heads, valueDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.sigmoid [heads, valueDim])
      (.cons gateUnactivated .nil)
  let gatedHeads : Term Γ [heads, valueDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.mul [heads, valueDim])
      (.cons gate (.cons headOutput .nil))
  let gatedFlat : Term Γ [heads * valueDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons gatedHeads .nil)
  let outputMatrix : Term Γ [heads * valueDim, modelDim] :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size, Nat.mul_assoc]))
      (.cons outputWeight .nil)
  let output : Term Γ [modelDim] :=
    Term.op (PrimOp.broadcastVecMat [] [] [] (heads * valueDim) modelDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
      (.cons gatedFlat (.cons outputMatrix .nil))
  { initParams :=
      initialLayerParams modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim
        valueDim
    body := Block.ret (.cons latentCache (.cons sharedKeyCache (.cons output .nil))) }

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
  -- Interpret the packed graph operations as the projections, cache appends, and attention
  -- in the reference step. The three output tensors must agree separately.
  simp only [stepGraphOutputs, stepFixedOutputs, NN.GraphSpec.DAG.MultiModel.specFwd,
    stepModel, layerParameters, stepInputs,
    Args.vars, Args.weakenLeft, Args.rename,
    Term.rename, Block.eval, Term.evalArgs, Term.eval,
    TorchLean.TensorPack.append, NN.GraphSpec.DAG.PrimOp.concatAxis,
    NN.GraphSpec.DAG.PrimOp.concatAxisSpec, Shape.replaceAxis,
    NN.GraphSpec.DAG.PrimOp.reshape_specFwd,
    NN.GraphSpec.DAG.PrimOp.rmsNorm_specFwd, NN.GraphSpec.DAG.PrimOp.rmsNormSpec,
    NN.GraphSpec.DAG.PrimOp.matmul_specFwd,
    NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth_specFwd,
    NN.GraphSpec.DAG.PrimOp.add_specFwd, eval_scaleTerm,
    NN.GraphSpec.DAG.PrimOp.sigmoid_specFwd, NN.GraphSpec.DAG.PrimOp.mul_specFwd,
    PrimOp.broadcastVecMat, NN.GraphSpec.DAG.PrimOp.broadcast,
    NN.GraphSpec.DAG.PrimOp.softmax, GatedMLA.stepFixed,
    GraphSpec.rmsNormVectorSpec_eq_scale]
  simp [Tensor.permuteByAdjacentSwaps]
  constructor
  · with_unfolding_all rfl
  · constructor
    · with_unfolding_all rfl
    · with_unfolding_all rfl

end MLA
end GraphSpec
end KimiK3
