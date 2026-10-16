/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.FeedForward.Expert
public import KimiK3.FeedForward.Routing
import NN.Proofs.Gradients.Activation

/-!
# Stable LatentMoE

Shared experts process the full input. Routed experts process its latent projection, and RMSNorm
comes before the projection back to model width. Routing bias changes which experts are selected,
while the mixture weights use the unbiased sigmoid scores.

Reference: https://arxiv.org/abs/2607.24653, Section 2.3, Eqs. 11--13.
-/

@[expose] public section

namespace KimiK3

open Spec TorchLean
open Tensor

/-- Parameters for Kimi K3's Stable LatentMoE layer. -/
structure StableLatentMoE (α : Type) [Storage α]
    (modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat) where
  /-- Projection from model width into the routed-expert latent space. -/
  downProject : Tensor α [modelDim, latentDim]
  /-- Projection from routed-expert latent space back to model width. -/
  upProject : Tensor α [latentDim, modelDim]
  /-- RMSNorm scale applied to the routed latent aggregate. -/
  routedNormScale : Tensor α [latentDim]
  /-- Projection producing one raw score per routed expert. -/
  routerWeight : Tensor α [modelDim, numRouted]
  /-- Bias added to the scores for expert selection only. -/
  routerBias : Tensor α [numRouted]
  /-- Full-width shared experts evaluated for every token. -/
  shared : Fin numShared → Expert α modelDim sharedHidden modelDim
  /-- Latent-width experts selected by the router. -/
  routed : Fin numRouted → Expert α latentDim routedHidden latentDim

namespace StableLatentMoE

variable {α : Type} [Storage α] [Context α]
variable {modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat}

/-- Raw router scores, the sigmoid of the router projection (Eq. 13). -/
def rawRouterScores
    (moe : StableLatentMoE α modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (x : Tensor α [modelDim]) : Tensor α [numRouted] :=
  Activation.sigmoidSpec (vecMatMulSpec x moe.routerWeight)

/-- Every raw routing score is strictly positive, since it is a sigmoid output. -/
theorem rawRouterScores_pos
    (moe : StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (x : Tensor ℝ [modelDim]) (expert : Fin numRouted) :
    0 < Tensor.getScalar (moe.rawRouterScores x) expert := by
  rw [rawRouterScores, Activation.sigmoidSpec, Tensor.getScalar_map,
    Proofs.sigmoid_eq_inv_exp]
  positivity

/-- Bias-adjusted scores used for top-k selection (Eq. 13). -/
def adjustedRouterScores
    (moe : StableLatentMoE α modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (x : Tensor α [modelDim]) : Tensor α [numRouted] :=
  moe.rawRouterScores x + moe.routerBias

/-- Mixture weights `p_i`: the raw scores of the selected experts, normalized. -/
def routeWeights
    (moe : StableLatentMoE α modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (route : Route numRouted activeExperts)
    (x : Tensor α [modelDim]) : Tensor α [activeExperts] :=
  let raw := moe.rawRouterScores x
  Normalize.probabilities <|
    Tensor.stackLeading (fun slot => Tensor.scalar (Tensor.getScalar raw (route slot)))

/-- For a nonempty route, the selected raw routing scores have positive total. -/
theorem selectedRouterScoreTotal_pos
    (moe : StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (route : Route numRouted activeExperts) (x : Tensor ℝ [modelDim])
    (hActive : 0 < activeExperts) :
    0 < Tensor.sumSpec
    (Tensor.stackLeading fun slot => Tensor.scalar
        (Tensor.getScalar (moe.rawRouterScores x) (route slot))) := by
  let _ : Nonempty (Fin activeExperts) := Fin.pos_iff_nonempty.mp hActive
  rw [Spec.sum_spec_vec]
  simp only [Tensor.stackLeading, Tensor.getScalar_dim]
  exact Finset.sum_pos
    (fun slot _ => moe.rawRouterScores_pos x (route slot))
    Finset.univ_nonempty

/-- For a nonempty route, the guarded normalization is plain division by the (positive) sum of the
selected raw scores. -/
theorem routeWeights_apply
    (moe : StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (route : Route numRouted activeExperts) (x : Tensor ℝ [modelDim])
    (hActive : 0 < activeExperts) (slot : Fin activeExperts) :
    Tensor.getScalar (moe.routeWeights route x) slot =
      Tensor.getScalar (moe.rawRouterScores x) (route slot) /
        Tensor.sumSpec
          (Tensor.stackLeading fun selected => Tensor.scalar
            (Tensor.getScalar (moe.rawRouterScores x) (route selected))) := by
  rw [routeWeights, Normalize.probabilities_of_sum_pos _
    (moe.selectedRouterScoreTotal_pos route x hActive)]
  simp only [Tensor.getScalar_map, Tensor.stackLeading, Tensor.getScalar_dim]

/-- Routing weights of a nonempty route sum to one. -/
theorem routeWeights_sum_eq_one
    (moe : StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (route : Route numRouted activeExperts) (x : Tensor ℝ [modelDim])
    (hActive : 0 < activeExperts) :
    Tensor.sumSpec (moe.routeWeights route x) = 1 := by
  rw [Spec.sum_spec_vec]
  simp_rw [moe.routeWeights_apply route x hActive]
  rw [← Finset.sum_div]
  have hTotal := moe.selectedRouterScoreTotal_pos route x hActive
  have hNumerator :
      (∑ slot, Tensor.getScalar (moe.rawRouterScores x) (route slot)) =
        Tensor.sumSpec
          (Tensor.stackLeading fun slot => Tensor.scalar
            (Tensor.getScalar (moe.rawRouterScores x) (route slot))) := by
    rw [Spec.sum_spec_vec]
    simp only [Tensor.stackLeading, Tensor.getScalar_dim]
  rw [hNumerator]
  exact div_self hTotal.ne'

/-- Sum all full-width shared experts. -/
def sharedOutput
    (moe : StableLatentMoE α modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (gateCap upCap : α) (x : Tensor α [modelDim]) :
    Tensor α [modelDim] :=
  (List.finRange numShared).foldl
    (fun total i => total + (moe.shared i).forward gateCap upCap x)
    (Tensor.zeros [modelDim])

/-- Weighted aggregate of the selected latent experts, the vector `u` in Eq. 11. -/
def routedAggregate
    (moe : StableLatentMoE α modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (route : Route numRouted activeExperts) (gateCap upCap : α)
    (x : Tensor α [modelDim]) : Tensor α [latentDim] :=
  let latent := vecMatMulSpec x moe.downProject
  let weights := moe.routeWeights route x
  (List.finRange activeExperts).foldl
    (fun total slot =>
      let expertOutput := (moe.routed (route slot)).forward gateCap upCap latent
      total + Tensor.map (fun value => Tensor.getScalar weights slot * value) expertOutput)
    (Tensor.zeros [latentDim])

/--
Stable LatentMoE forward pass (Eq. 11). Shared experts run at model width, the selected routed
experts run at latent width, and their weighted sum is RMS-normalized before the up-projection.
-/
def forward
    (moe : StableLatentMoE α modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (route : Route numRouted activeExperts) (gateCap upCap : α)
    (x : Tensor α [modelDim]) : Tensor α [modelDim] :=
  let shared := moe.sharedOutput gateCap upCap x
  let routed := moe.routedAggregate route gateCap upCap x
  shared + vecMatMulSpec (RMSNorm.scale routed moe.routedNormScale) moe.upProject

/-- Route a real-valued token using Eq. 13's bias-adjusted scores. -/
noncomputable def route
    (moe : StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (hActive : activeExperts ≤ numRouted)
    (x : Tensor ℝ [modelDim]) : Route numRouted activeExperts :=
  Route.chooseTopK (fun expert => Tensor.getScalar (moe.adjustedRouterScores x) expert) hActive

/-- A Stable LatentMoE layer's computed route is top-k for its adjusted router scores. -/
theorem route_isTopK
    (moe : StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (hActive : activeExperts ≤ numRouted)
    (x : Tensor ℝ [modelDim]) :
    (moe.route hActive x).IsTopK
      (fun expert => Tensor.getScalar (moe.adjustedRouterScores x) expert) :=
  Route.chooseTopK_isTopK _ hActive

/-- Real-valued Stable LatentMoE semantics with routing computed from the layer's own scores. -/
noncomputable def forwardReal
    (moe : StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared numRouted
      activeExperts)
    (hActive : activeExperts ≤ numRouted) (gateCap upCap : ℝ)
    (x : Tensor ℝ [modelDim]) : Tensor ℝ [modelDim] :=
  moe.forward (moe.route hActive x) gateCap upCap x

/-- Replace the real-valued routed experts by the denotations of packed MXFP4 experts. Shared
experts, router and latent projections, and normalization parameters stay unchanged, matching the
precision boundary stated in the K3 report. -/
noncomputable def withMXFP4Routed {latentBlocks hiddenBlocks : Nat}
    (moe : StableLatentMoE ℝ modelDim (latentBlocks * 32) sharedHidden (hiddenBlocks * 32)
      numShared numRouted activeExperts)
    (routed : Fin numRouted → MXFP4Expert latentBlocks hiddenBlocks (latentBlocks * 32)) :
    StableLatentMoE ℝ modelDim (latentBlocks * 32) sharedHidden (hiddenBlocks * 32)
      numShared numRouted activeExperts :=
  { moe with routed := fun expert => (routed expert).decode }

/-- Stable LatentMoE semantics with the routed experts decoded from packed MXFP4 weights. Routing
and mixture weights still come from the original high-precision fields. -/
noncomputable def forwardMXFP4 {latentBlocks hiddenBlocks : Nat}
    (moe : StableLatentMoE ℝ modelDim (latentBlocks * 32) sharedHidden (hiddenBlocks * 32)
      numShared numRouted activeExperts)
    (routed : Fin numRouted → MXFP4Expert latentBlocks hiddenBlocks (latentBlocks * 32))
    (hActive : activeExperts ≤ numRouted) (gateCap upCap : ℝ)
    (input : Tensor ℝ [modelDim]) : Tensor ℝ [modelDim] :=
  (moe.withMXFP4Routed routed).forwardReal hActive gateCap upCap input

/-- Certified microscaled executions of the experts selected for one token.

The router, low-rank latent projection, and mixture weights come from `moe`. Each selected expert
carries its own MXFP8 activation encodings, and the three coverage fields say that the per-expert
MXFP4 block scales saturate no source weight.
-/
structure MXExecution {latentBlocks hiddenBlocks : Nat} (format : Microscaling.FP8Format)
    (moe : StableLatentMoE ℝ modelDim (latentBlocks * 32) sharedHidden (hiddenBlocks * 32)
      numShared numRouted activeExperts)
    (route : Route numRouted activeExperts) (gateCap upCap : ℝ)
    (input : Tensor ℝ [modelDim]) where
  /-- Per-expert MXFP4 block scales. -/
  scales : Fin numRouted → MXFP4Expert.Scales latentBlocks hiddenBlocks (latentBlocks * 32)
  gateScaleCovers : ∀ expert output inputBlock offset,
    |get2 (moe.routed expert).gateWeight (finProdFinEquiv (inputBlock, offset)) output| ≤
      6 * Microscaling.scaleValue ((scales expert).gate output inputBlock)
  upScaleCovers : ∀ expert output inputBlock offset,
    |get2 (moe.routed expert).upWeight (finProdFinEquiv (inputBlock, offset)) output| ≤
      6 * Microscaling.scaleValue ((scales expert).up output inputBlock)
  downScaleCovers : ∀ expert output hiddenBlock offset,
    |get2 (moe.routed expert).downWeight (finProdFinEquiv (hiddenBlock, offset)) output| ≤
      6 * Microscaling.scaleValue ((scales expert).down output hiddenBlock)
  /-- Certified microscaled execution for each selected expert. -/
  selected : ∀ slot,
    MXFP4Expert.Execution format (scales (route slot)) (moe.routed (route slot))
      gateCap upCap (vecMatMulSpec input moe.downProject)

/-- Weighted latent output of the selected microscaled experts. -/
noncomputable def routedAggregateMX {latentBlocks hiddenBlocks : Nat}
    {format : Microscaling.FP8Format}
    (moe : StableLatentMoE ℝ modelDim (latentBlocks * 32) sharedHidden (hiddenBlocks * 32)
      numShared numRouted activeExperts)
    (route : Route numRouted activeExperts) (gateCap upCap : ℝ)
    (input : Tensor ℝ [modelDim])
    (execution : MXExecution format moe route gateCap upCap input) :
    Tensor ℝ [latentBlocks * 32] :=
  let weights := moe.routeWeights route input
  (List.finRange activeExperts).foldl
    (fun total slot =>
      let expertOutput := (execution.selected slot).output
      total + Tensor.map (fun value => Tensor.getScalar weights slot * value) expertOutput)
    (Tensor.zeros [latentBlocks * 32])

/-- Stable LatentMoE with the precision split used by K3 deployment: shared experts, routing,
latent projections, normalization, and the final up-projection remain high precision, while every
selected routed expert is evaluated through its certified MXFP8/MXFP4 execution witness. -/
noncomputable def forwardMX {latentBlocks hiddenBlocks : Nat}
    {format : Microscaling.FP8Format}
    (moe : StableLatentMoE ℝ modelDim (latentBlocks * 32) sharedHidden (hiddenBlocks * 32)
      numShared numRouted activeExperts)
    (route : Route numRouted activeExperts) (gateCap upCap : ℝ)
    (input : Tensor ℝ [modelDim])
    (execution : MXExecution format moe route gateCap upCap input) :
    Tensor ℝ [modelDim] :=
  let shared := moe.sharedOutput gateCap upCap input
  let routed := moe.routedAggregateMX route gateCap upCap input execution
  shared + vecMatMulSpec (RMSNorm.scale routed moe.routedNormScale) moe.upProject

end StableLatentMoE

end KimiK3
