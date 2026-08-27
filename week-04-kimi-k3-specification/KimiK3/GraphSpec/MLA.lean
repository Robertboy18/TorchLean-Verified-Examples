/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Primitives
public import KimiK3.Sequence

/-!
# Fixed-cache latent attention graph

This module lowers one Kimi K3 MLA head to TorchLean's typed DAG.  The graph starts after the
layer-level query and KV down-projections: it receives one normalized query latent and two
fixed-length cache tensors, then reconstructs the per-head keys and values and performs scaled
softmax attention.

The cache length is part of every tensor shape.  That is the representation needed by compilation,
autograd, and backend planning.  `KimiK3.GatedMLA.Cache` remains the convenient streaming view; a
later equivalence theorem relates its first `tokens` entries to the tensors accepted here.

K3 intentionally uses NoPE attention.  Consequently this graph contains no rotary-position
operation: its two score contributions are the content key and the shared unrotated key.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec
namespace MLA

open Spec
open Spec.Tensor
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
  .cons (Spec.fill 0 (.dim modelDim (.dim queryLatentDim .scalar))) <|
    .cons (Spec.fill 0 (.dim queryLatentDim .scalar)) <|
      .cons (Spec.fill 0 (.dim modelDim (.dim kvLatentDim .scalar))) <|
        .cons (Spec.fill 0 (.dim kvLatentDim .scalar)) <|
          .cons (Spec.fill 0 (.dim modelDim (.dim sharedKeyDim .scalar))) <|
            .cons (Spec.fill 0
              (.dim heads (.dim queryLatentDim (.dim contentKeyDim .scalar)))) <|
              .cons (Spec.fill 0
                (.dim heads (.dim queryLatentDim (.dim sharedKeyDim .scalar)))) <|
                .cons (Spec.fill 0
                  (.dim heads (.dim kvLatentDim (.dim contentKeyDim .scalar)))) <|
                  .cons (Spec.fill 0
                    (.dim heads (.dim kvLatentDim (.dim valueDim .scalar)))) <|
                    .cons (Spec.fill 0
                      (.dim modelDim (.dim heads (.dim valueDim .scalar)))) <|
                      .cons (Spec.fill 0
                        (.dim heads (.dim valueDim (.dim modelDim .scalar))) ) .nil

/-- Package a theorem-level Gated MLA layer in the graph parameter ABI. -/
def layerParameters {α : Type}
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
def stepInputs {α : Type} {pastTokens modelDim kvLatentDim sharedKeyDim : Nat}
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
  let queryDown : Term Γ (.dim modelDim (.dim queryLatentDim .scalar)) :=
    Term.var .head
  let queryNormScale : Term Γ (.dim queryLatentDim .scalar) :=
    Term.var (.tail .head)
  let kvDown : Term Γ (.dim modelDim (.dim kvLatentDim .scalar)) :=
    Term.var (.tail (.tail .head))
  let kvNormScale : Term Γ (.dim kvLatentDim .scalar) :=
    Term.var (.tail (.tail (.tail .head)))
  let sharedKeyDown : Term Γ (.dim modelDim (.dim sharedKeyDim .scalar)) :=
    Term.var (.tail (.tail (.tail (.tail .head))))
  let queryContentUp :
      Term Γ (.dim heads (.dim queryLatentDim (.dim contentKeyDim .scalar))) :=
    Term.var (.tail (.tail (.tail (.tail (.tail .head)))))
  let querySharedUp :
      Term Γ (.dim heads (.dim queryLatentDim (.dim sharedKeyDim .scalar))) :=
    Term.var (.tail (.tail (.tail (.tail (.tail (.tail .head))))))
  let keyUp : Term Γ (.dim heads (.dim kvLatentDim (.dim contentKeyDim .scalar))) :=
    Term.var (.tail (.tail (.tail (.tail (.tail (.tail (.tail .head)))))))
  let valueUp : Term Γ (.dim heads (.dim kvLatentDim (.dim valueDim .scalar))) :=
    Term.var (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail .head))))))))
  let gateWeight : Term Γ (.dim modelDim (.dim heads (.dim valueDim .scalar))) :=
    Term.var (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail .head)))))))))
  let outputWeight : Term Γ (.dim heads (.dim valueDim (.dim modelDim .scalar))) :=
    Term.var
      (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail .head))))))))))
  let pastLatentCache : Term Γ (.dim pastTokens (.dim kvLatentDim .scalar)) :=
    Term.var
      (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail
        (.tail .head)))))))))))
  let pastSharedKeyCache : Term Γ (.dim pastTokens (.dim sharedKeyDim .scalar)) :=
    Term.var
      (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail
        (.tail (.tail .head))))))))))))
  let x : Term Γ (.dim modelDim .scalar) :=
    Term.var
      (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail
        (.tail (.tail (.tail .head)))))))))))))
  let scoreScale : Term Γ .scalar :=
    Term.var
      (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail
        (.tail (.tail (.tail (.tail .head))))))))))))))
  let queryProjected : Term Γ (.dim queryLatentDim .scalar) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.vecMat modelDim queryLatentDim)
      (.cons x (.cons queryDown .nil))
  let queryLatent : Term Γ (.dim queryLatentDim .scalar) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.rmsNorm .scalar queryLatentDim hQueryLatent)
      (.cons queryProjected (.cons queryNormScale .nil))
  let kvProjected : Term Γ (.dim kvLatentDim .scalar) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.vecMat modelDim kvLatentDim)
      (.cons x (.cons kvDown .nil))
  let currentKV : Term Γ (.dim kvLatentDim .scalar) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.rmsNorm .scalar kvLatentDim hKVLatent)
      (.cons kvProjected (.cons kvNormScale .nil))
  let currentShared : Term Γ (.dim sharedKeyDim .scalar) :=
    Term.op (NN.GraphSpec.DAG.PrimOp.vecMat modelDim sharedKeyDim)
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
    Term.op (NN.GraphSpec.DAG.PrimOp.vecMat modelDim (heads * valueDim))
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
    Term.op (NN.GraphSpec.DAG.PrimOp.vecMat (heads * valueDim) modelDim)
      (.cons gatedFlat (.cons outputMatrix .nil))
  { initParams :=
      initialLayerParams modelDim heads queryLatentDim kvLatentDim contentKeyDim sharedKeyDim
        valueDim
    body := Block.ret (.cons latentCache (.cons sharedKeyCache (.cons output .nil))) }

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
    stepModel, layerParameters, stepInputs, Block.eval, Term.evalArgs, Term.eval, Env.tget,
    TorchLean.TensorPack.append, NN.GraphSpec.DAG.PrimOp.concatAxis,
    NN.GraphSpec.DAG.PrimOp.concatAxisSpec, Shape.replaceAxis,
    NN.GraphSpec.DAG.PrimOp.reshape_specFwd,
    NN.GraphSpec.DAG.PrimOp.rmsNorm_specFwd, NN.GraphSpec.DAG.PrimOp.rmsNormSemantics,
    NN.GraphSpec.DAG.PrimOp.matmul_specFwd,
    NN.GraphSpec.DAG.PrimOp.swapAdjacentAtDepth_specFwd,
    NN.GraphSpec.DAG.PrimOp.add_specFwd, NN.GraphSpec.DAG.PrimOp.scalarMul_specFwd,
    NN.GraphSpec.DAG.PrimOp.sigmoid_specFwd, NN.GraphSpec.DAG.PrimOp.mul_specFwd,
    NN.GraphSpec.DAG.PrimOp.vecMat, NN.GraphSpec.DAG.PrimOp.broadcast,
    NN.GraphSpec.DAG.PrimOp.softmax, GatedMLA.stepFixed, Tensor.item_scalar,
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
