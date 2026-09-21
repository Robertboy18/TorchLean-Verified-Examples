/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Primitives
public import KimiK3.Sequence

/-!
# Fixed-cache Gated MLA graph

This module lowers a complete Kimi K3 Gated MLA token step to TorchLean's typed DAG. The graph
includes the query and KV down-projections, normalization, cache append, every head's key/value
reconstruction and attention, the output gate, and the final projection.

The cache length is part of every tensor shape.  That is the representation needed by compilation,
autograd, and backend planning. `KimiK3.GatedMLA.Cache` remains a separate list-based streaming
view. This file proves the graph against `GatedMLA.stepFixed`; it does not yet prove that packing a
list cache and running `GatedMLA.step` gives the same result.

K3 intentionally uses NoPE attention.  Consequently this graph contains no rotary-position
operation: its two score contributions are the content key and the shared unrotated key.
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
  [ .dim modelDim (.dim queryLatentDim .scalar),
    .dim queryLatentDim .scalar,
    .dim modelDim (.dim kvLatentDim .scalar),
    .dim kvLatentDim .scalar,
    .dim modelDim (.dim sharedKeyDim .scalar),
    .dim heads (.dim queryLatentDim (.dim contentKeyDim .scalar)),
    .dim heads (.dim queryLatentDim (.dim sharedKeyDim .scalar)),
    .dim heads (.dim kvLatentDim (.dim contentKeyDim .scalar)),
    .dim heads (.dim kvLatentDim (.dim valueDim .scalar)),
    .dim modelDim (.dim heads (.dim valueDim .scalar)),
    .dim heads (.dim valueDim (.dim modelDim .scalar)) ]

/-- Previous latent cache, previous shared-key cache, current token, and score scale. -/
abbrev StepInputs (pastTokens modelDim kvLatentDim sharedKeyDim : Nat) : List Shape :=
  [ .dim pastTokens (.dim kvLatentDim .scalar),
    .dim pastTokens (.dim sharedKeyDim .scalar),
    .dim modelDim .scalar,
    .scalar ]

/-- Updated latent cache, updated shared-key cache, and the current MLA output. -/
abbrev StepOutputs (pastTokens modelDim kvLatentDim sharedKeyDim : Nat) : List Shape :=
  [ .dim (pastTokens + 1) (.dim kvLatentDim .scalar),
    .dim (pastTokens + 1) (.dim sharedKeyDim .scalar),
    .dim modelDim .scalar ]

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
      .cons layer.queryContentUpPacked <| .cons layer.querySharedUpPacked <|
        .cons layer.keyUpPacked <| .cons layer.valueUpPacked <| .cons layer.gateWeight <|
          .cons layer.outputWeight .nil

/-- Package one fixed-context causal step in the graph input ABI. -/
def stepInputs {α : Type} [Storage α] {pastTokens modelDim kvLatentDim sharedKeyDim : Nat}
    (pastLatentCache : Tensor α (.dim pastTokens (.dim kvLatentDim .scalar)))
    (pastSharedKeyCache : Tensor α (.dim pastTokens (.dim sharedKeyDim .scalar)))
    (x : Tensor α (.dim modelDim .scalar)) (scoreScale : α) :
    TorchLean.TensorPack α (StepInputs pastTokens modelDim kvLatentDim sharedKeyDim) :=
  .cons pastLatentCache <| .cons pastSharedKeyCache <| .cons x <|
    .cons (.scalar scoreScale) .nil

/-- The complete fixed-context Gated MLA token step as a typed, multi-output DAG. -/
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
  let queryProjected : Term Γ (.dim queryLatentDim .scalar) :=
    Term.op (PrimOp.broadcastVecMat .scalar .scalar .scalar modelDim queryLatentDim
      (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar))
      (.cons x (.cons queryDown .nil))
  let queryLatent : Term Γ (.dim queryLatentDim .scalar) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.rmsNorm .scalar queryLatentDim hQueryLatent)
      (.cons queryProjected (.cons queryNormScale .nil))
  let kvProjected : Term Γ (.dim kvLatentDim .scalar) :=
    Term.op (PrimOp.broadcastVecMat .scalar .scalar .scalar modelDim kvLatentDim
      (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar))
      (.cons x (.cons kvDown .nil))
  let currentKV : Term Γ (.dim kvLatentDim .scalar) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.rmsNorm .scalar kvLatentDim hKVLatent)
      (.cons kvProjected (.cons kvNormScale .nil))
  let currentShared : Term Γ (.dim sharedKeyDim .scalar) :=
    Term.op (PrimOp.broadcastVecMat .scalar .scalar .scalar modelDim sharedKeyDim
      (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar))
      (.cons x (.cons sharedKeyDown .nil))
  let currentKVRow : Term Γ (.dim 1 (.dim kvLatentDim .scalar)) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons currentKV .nil)
  let currentSharedRow : Term Γ (.dim 1 (.dim sharedKeyDim .scalar)) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons currentShared .nil)
  let latentCache : Term Γ (.dim (pastTokens + 1) (.dim kvLatentDim .scalar)) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.concatAxis
      (.dim pastTokens (.dim kvLatentDim .scalar)) 0 pastTokens 1)
      (.cons pastLatentCache (.cons currentKVRow .nil))
  let sharedKeyCache : Term Γ (.dim (pastTokens + 1) (.dim sharedKeyDim .scalar)) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.concatAxis
      (.dim pastTokens (.dim sharedKeyDim .scalar)) 0 pastTokens 1)
      (.cons pastSharedKeyCache (.cons currentSharedRow .nil))
  let queryBatchSource : Term Γ (.dim 1 (.dim 1 (.dim queryLatentDim .scalar))) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons queryLatent .nil)
  let queryBatch : Term Γ (.dim heads (.dim 1 (.dim queryLatentDim .scalar))) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.broadcast
      (Shape.CanBroadcastTo.dim_1_to_n
        (Shape.CanBroadcastTo.dim_eq
          (Shape.CanBroadcastTo.dim_eq
            (Shape.CanBroadcastTo.scalarTo .scalar))))) (.cons queryBatchSource .nil)
  let latentBatchSource :
      Term Γ (.dim 1 (.dim (pastTokens + 1) (.dim kvLatentDim .scalar))) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons latentCache .nil)
  let latentBatch :
      Term Γ (.dim heads (.dim (pastTokens + 1) (.dim kvLatentDim .scalar))) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.broadcast
      (Shape.CanBroadcastTo.dim_1_to_n
        (Shape.CanBroadcastTo.dim_eq
          (Shape.CanBroadcastTo.dim_eq
            (Shape.CanBroadcastTo.scalarTo .scalar))))) (.cons latentBatchSource .nil)
  let sharedBatchSource :
      Term Γ (.dim 1 (.dim (pastTokens + 1) (.dim sharedKeyDim .scalar))) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons sharedKeyCache .nil)
  let sharedBatch :
      Term Γ (.dim heads (.dim (pastTokens + 1) (.dim sharedKeyDim .scalar))) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.broadcast
      (Shape.CanBroadcastTo.dim_1_to_n
        (Shape.CanBroadcastTo.dim_eq
          (Shape.CanBroadcastTo.dim_eq
            (Shape.CanBroadcastTo.scalarTo .scalar))))) (.cons sharedBatchSource .nil)
  let queryContent : Term Γ (.dim heads (.dim 1 (.dim contentKeyDim .scalar))) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads]
      1 queryLatentDim contentKeyDim (.refl [heads]) (.refl [heads]))
      (.cons queryBatch (.cons queryContentUp .nil))
  let queryShared : Term Γ (.dim heads (.dim 1 (.dim sharedKeyDim .scalar))) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads]
      1 queryLatentDim sharedKeyDim (.refl [heads]) (.refl [heads]))
      (.cons queryBatch (.cons querySharedUp .nil))
  let keys :
      Term Γ (.dim heads (.dim (pastTokens + 1) (.dim contentKeyDim .scalar))) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads]
      (pastTokens + 1) kvLatentDim contentKeyDim (.refl [heads]) (.refl [heads]))
      (.cons latentBatch (.cons keyUp .nil))
  let values : Term Γ (.dim heads (.dim (pastTokens + 1) (.dim valueDim .scalar))) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads]
      (pastTokens + 1) kvLatentDim valueDim (.refl [heads]) (.refl [heads]))
      (.cons latentBatch (.cons valueUp .nil))
  let keysTranspose :
      Term Γ (.dim heads (.dim contentKeyDim (.dim (pastTokens + 1) .scalar))) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth
      (.dim heads (.dim (pastTokens + 1) (.dim contentKeyDim .scalar))) 1)
      (.cons keys .nil)
  let sharedTranspose :
      Term Γ (.dim heads (.dim sharedKeyDim (.dim (pastTokens + 1) .scalar))) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth
      (.dim heads (.dim (pastTokens + 1) (.dim sharedKeyDim .scalar))) 1)
      (.cons sharedBatch .nil)
  let contentScores : Term Γ (.dim heads (.dim 1 (.dim (pastTokens + 1) .scalar))) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads]
      1 contentKeyDim (pastTokens + 1) (.refl [heads]) (.refl [heads]))
      (.cons queryContent (.cons keysTranspose .nil))
  let sharedScores : Term Γ (.dim heads (.dim 1 (.dim (pastTokens + 1) .scalar))) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads]
      1 sharedKeyDim (pastTokens + 1) (.refl [heads]) (.refl [heads]))
      (.cons queryShared (.cons sharedTranspose .nil))
  let scoreShape := .dim heads (.dim 1 (.dim (pastTokens + 1) .scalar))
  let scores : Term Γ scoreShape :=
    Term.op (NN.GraphSpec.DAG.PrimOp.add scoreShape)
      (.cons contentScores (.cons sharedScores .nil))
  let scaledScores : Term Γ scoreShape :=
    Term.op (NN.GraphSpec.DAG.PrimOp.scalarMul scoreShape)
      (.cons scoreScale (.cons scores .nil))
  let weights : Term Γ scoreShape :=
    Term.op (NN.GraphSpec.DAG.PrimOp.softmax scoreShape 2) (.cons scaledScores .nil)
  let headOutput3 : Term Γ (.dim heads (.dim 1 (.dim valueDim .scalar))) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.matmul [heads] [heads] [heads]
      1 (pastTokens + 1) valueDim (.refl [heads]) (.refl [heads]))
      (.cons weights (.cons values .nil))
  let headOutput : Term Γ (.dim heads (.dim valueDim .scalar)) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons headOutput3 .nil)
  let gateMatrix : Term Γ (.dim modelDim (.dim (heads * valueDim) .scalar)) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons gateWeight .nil)
  let gateFlat : Term Γ (.dim (heads * valueDim) .scalar) :=
    Term.op (PrimOp.broadcastVecMat .scalar .scalar .scalar modelDim (heads * valueDim)
      (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar))
      (.cons x (.cons gateMatrix .nil))
  let gateUnactivated : Term Γ (.dim heads (.dim valueDim .scalar)) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons gateFlat .nil)
  let gate : Term Γ (.dim heads (.dim valueDim .scalar)) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.sigmoid (.dim heads (.dim valueDim .scalar)))
      (.cons gateUnactivated .nil)
  let gatedHeads : Term Γ (.dim heads (.dim valueDim .scalar)) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.mul (.dim heads (.dim valueDim .scalar)))
      (.cons gate (.cons headOutput .nil))
  let gatedFlat : Term Γ (.dim (heads * valueDim) .scalar) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size]))
      (.cons gatedHeads .nil)
  let outputMatrix : Term Γ (.dim (heads * valueDim) (.dim modelDim .scalar)) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.reshape _ _ (by simp [Shape.size, Nat.mul_assoc]))
      (.cons outputWeight .nil)
  let output : Term Γ (.dim modelDim .scalar) :=
    Term.op (PrimOp.broadcastVecMat .scalar .scalar .scalar (heads * valueDim) modelDim
      (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar))
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
    (pastLatentCache : Tensor ℝ (.dim pastTokens (.dim kvLatentDim .scalar)))
    (pastSharedKeyCache : Tensor ℝ (.dim pastTokens (.dim sharedKeyDim .scalar)))
    (x : Tensor ℝ (.dim modelDim .scalar)) (scoreScale : ℝ) :
    TorchLean.TensorPack ℝ (StepOutputs pastTokens modelDim kvLatentDim sharedKeyDim) :=
  (stepModel pastTokens modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim
    valueDim hQueryLatent hKVLatent).specFwd (layerParameters layer)
      (stepInputs pastLatentCache pastSharedKeyCache x scoreScale)

/-- Package the reference MLA token-step result in the graph's output shape list. -/
noncomputable def stepFixedOutputs
    (pastTokens modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim : Nat)
    (layer :
      GatedMLA ℝ modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (pastLatentCache : Tensor ℝ (.dim pastTokens (.dim kvLatentDim .scalar)))
    (pastSharedKeyCache : Tensor ℝ (.dim pastTokens (.dim sharedKeyDim .scalar)))
    (x : Tensor ℝ (.dim modelDim .scalar)) (scoreScale : ℝ) :
    TorchLean.TensorPack ℝ (StepOutputs pastTokens modelDim kvLatentDim sharedKeyDim) :=
  let result := layer.stepFixed pastTokens pastLatentCache
    pastSharedKeyCache x scoreScale
  .cons result.1 (.cons result.2.1 (.cons result.2.2 .nil))

-- The complete Gated MLA DAG denotes the fixed-context mathematical token step.
theorem stepModel_specFwd_eq_stepFixed
    {pastTokens modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim : Nat}
    (hQueryLatent : 0 < queryLatentDim) (hKVLatent : 0 < kvLatentDim)
    (layer :
      GatedMLA ℝ modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim valueDim)
    (pastLatentCache : Tensor ℝ (.dim pastTokens (.dim kvLatentDim .scalar)))
    (pastSharedKeyCache : Tensor ℝ (.dim pastTokens (.dim sharedKeyDim .scalar)))
    (x : Tensor ℝ (.dim modelDim .scalar)) (scoreScale : ℝ) :
    stepGraphOutputs pastTokens modelDim heads queryLatentDim kvLatentDim contentKeyDim
      sharedKeyDim valueDim hQueryLatent hKVLatent layer pastLatentCache pastSharedKeyCache x
        scoreScale =
      stepFixedOutputs pastTokens modelDim heads queryLatentDim kvLatentDim contentKeyDim
        sharedKeyDim valueDim layer pastLatentCache pastSharedKeyCache x scoreScale := by
  simp only [stepGraphOutputs, stepFixedOutputs, NN.GraphSpec.DAG.MultiModel.specFwd,
    stepModel, layerParameters, stepInputs, Args.get_rename,
    Args.vars, Args.weakenLeft,
    Term.weakenLeft, Term.rename, Block.eval, Term.evalArgs, Term.eval,
    TorchLean.TensorPack.append, NN.GraphSpec.DAG.PrimOp.concatAxis,
    NN.GraphSpec.DAG.PrimOp.concatAxisSpec, Shape.replaceAxis,
    NN.GraphSpec.DAG.PrimOp.reshape_specFwd,
    NN.GraphSpec.DAG.PrimOp.rmsNorm_specFwd, NN.GraphSpec.DAG.PrimOp.rmsNormSemantics,
    NN.GraphSpec.DAG.PrimOp.matmul_specFwd,
    NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth_specFwd,
    NN.GraphSpec.DAG.PrimOp.add_specFwd, NN.GraphSpec.DAG.PrimOp.scalarMul_specFwd,
    NN.GraphSpec.DAG.PrimOp.sigmoid_specFwd, NN.GraphSpec.DAG.PrimOp.mul_specFwd,
    PrimOp.broadcastVecMat, NN.GraphSpec.DAG.PrimOp.broadcast,
    NN.GraphSpec.DAG.PrimOp.softmax, GatedMLA.stepFixed,
    GraphSpec.rmsNormVectorSemantics_eq_scale]
  simp [Tensor.permuteByAdjacentSwaps]
  constructor
  · with_unfolding_all rfl
  · constructor
    · with_unfolding_all rfl
    · with_unfolding_all rfl

end MLA
end GraphSpec
end KimiK3
