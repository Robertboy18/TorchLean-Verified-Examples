/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Expert
public import KimiK3.FeedForward.MoE

/-!
# Stable LatentMoE graph

This module lowers one routed Stable LatentMoE evaluation to TorchLean's typed DAG. Expert weights
sit in leading-axis banks, a `Route` selects slices from them, and every continuous operation is an
ordinary GraphSpec node.

The route is an argument of the graph rather than a floating-point primitive. Top-k is a discrete
decision whose contract `Route.chooseTopK_isTopK` proves, and the graph runs the differentiable
computation for the resulting route.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec
namespace StableLatentMoE

open Spec TorchLean
open TorchLean.Tensor
open NN.GraphSpec.DAG

/-- Packed parameter shapes for one Stable LatentMoE layer. -/
abbrev Params (modelDim latentDim sharedHidden routedHidden numShared numRouted : Nat) :
    List Shape :=
  [ [modelDim, latentDim],
    [latentDim, modelDim],
    [latentDim],
    [modelDim, numRouted],
    [numRouted],
    [numShared, modelDim, sharedHidden],
    [numShared, modelDim, sharedHidden],
    [numShared, sharedHidden, modelDim],
    [numRouted, latentDim, routedHidden],
    [numRouted, latentDim, routedHidden],
    [numRouted, routedHidden, latentDim] ]

/-- One model-width token and the two SiTU caps are non-parameter graph inputs. -/
abbrev Inputs (modelDim : Nat) : List Shape :=
  [[modelDim], [], []]

/-- Zero initialization used only by GraphSpec's model package. -/
def initialParams (modelDim latentDim sharedHidden routedHidden numShared numRouted : Nat) :
    TorchLean.TensorPack Float
      (Params modelDim latentDim sharedHidden routedHidden numShared numRouted) :=
  TorchLean.TensorPack.zero

/-- Pack the mathematical MoE record into the graph's expert-bank layout. -/
def parameters {α : Type} [Storage α]
    {modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat}
    (moe : KimiK3.StableLatentMoE α modelDim latentDim sharedHidden routedHidden numShared
      numRouted activeExperts) :
    TorchLean.TensorPack α
      (Params modelDim latentDim sharedHidden routedHidden numShared numRouted) :=
  .cons moe.downProject <| .cons moe.upProject <| .cons moe.routedNormScale <|
    .cons moe.routerWeight <| .cons moe.routerBias <|
      .cons (Tensor.stackLeading fun expert => (moe.shared expert).gateWeight) <|
        .cons (Tensor.stackLeading fun expert => (moe.shared expert).upWeight) <|
          .cons (Tensor.stackLeading fun expert => (moe.shared expert).downWeight) <|
            .cons (Tensor.stackLeading fun expert => (moe.routed expert).gateWeight) <|
              .cons (Tensor.stackLeading fun expert => (moe.routed expert).upWeight) <|
                .cons (Tensor.stackLeading fun expert => (moe.routed expert).downWeight) .nil

/-- Package one token and its SiTU caps in the graph input layout. -/
def inputs {α : Type} [Storage α] {modelDim : Nat}
    (input : Tensor α [modelDim]) (gateCap upCap : α) :
    TorchLean.TensorPack α (Inputs modelDim) :=
  .cons input <| .cons (Tensor.scalar gateCap) <| .cons (Tensor.scalar upCap) .nil

/-- Select one entry from a leading parameter bank. -/
def selectLeadingTerm {Γ : List Shape} (count : Nat) (shape : Shape)
    (index : Fin count) (bank : Term Γ (.dim count shape)) : Term Γ shape :=
  Term.cast
    (Term.op (NN.GraphSpec.DAG.PrimOp.select (.dim count shape) 0 index) (.cons bank .nil))
    (by rfl)

@[simp] theorem eval_selectLeadingTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (count : Nat) (shape : Shape) (index : Fin count)
    (bank : Term Γ (.dim count shape)) :
    Term.eval env (selectLeadingTerm count shape index bank) =
      Spec.get (Term.eval env bank) index := by
  rfl

/-- TorchLean graph for a Stable LatentMoE evaluation with a fixed route. -/
def modelGivenRoute
    (modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat)
    (route : Route numRouted activeExperts) (hLatent : 0 < latentDim) :
    NN.GraphSpec.DAG.Model
      (Params modelDim latentDim sharedHidden routedHidden numShared numRouted)
      (Inputs modelDim) [modelDim] :=
  let Γ :=
    Params modelDim latentDim sharedHidden routedHidden numShared numRouted ++ Inputs modelDim
  let envTerms : Args Γ
      (Params modelDim latentDim sharedHidden routedHidden numShared numRouted ++
        Inputs modelDim) := by
    simpa [Γ] using Args.vars Γ
  let .cons downProject <| .cons upProject <| .cons routedNormScale <|
      .cons routerWeight <| .cons _routerBias <| .cons sharedGate <| .cons sharedUp <|
      .cons sharedDown <| .cons routedGate <| .cons routedUp <| .cons routedDown <|
      .cons input <| .cons gateCap <| .cons upCap .nil := envTerms
  let sharedOutput := Term.sum [modelDim] <|
    (List.finRange numShared).map fun expert =>
      let gateWeight := selectLeadingTerm numShared
        [modelDim, sharedHidden] expert sharedGate
      let upWeight := selectLeadingTerm numShared
        [modelDim, sharedHidden] expert sharedUp
      let downWeight := selectLeadingTerm numShared
        [sharedHidden, modelDim] expert sharedDown
      Expert.term modelDim sharedHidden modelDim input gateWeight upWeight downWeight gateCap upCap
  let latent := Term.op (PrimOp.broadcastVecMat [] [] [] modelDim latentDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
    (.cons input (.cons downProject .nil))
  let rawScores := Term.op (NN.GraphSpec.DAG.PrimOp.sigmoid [numRouted])
    (.cons (Term.op (PrimOp.broadcastVecMat [] [] [] modelDim numRouted
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
      (.cons input (.cons routerWeight .nil))) .nil)
  let selectedTotal := Term.sum [] <|
    (List.finRange activeExperts).map fun slot =>
      let score := selectLeadingTerm numRouted [] (route slot) rawScores
      score
  let inverseTotal := Term.op (NN.GraphSpec.DAG.PrimOp.inv []) (.cons selectedTotal .nil)
  let routedOutput := Term.sum [latentDim] <|
    (List.finRange activeExperts).map fun slot =>
      let expertIndex := route slot
      let gateWeight := selectLeadingTerm numRouted
        [latentDim, routedHidden] expertIndex routedGate
      let upWeight := selectLeadingTerm numRouted
        [latentDim, routedHidden] expertIndex routedUp
      let downWeight := selectLeadingTerm numRouted
        [routedHidden, latentDim] expertIndex routedDown
      let expertOutput := Expert.term latentDim routedHidden latentDim latent gateWeight upWeight
        downWeight gateCap upCap
      let score := selectLeadingTerm numRouted [] expertIndex rawScores
      let weight := Term.op (NN.GraphSpec.DAG.PrimOp.mul [])
        (.cons score (.cons inverseTotal .nil))
      scaleTerm [latentDim] weight expertOutput
  let normalized := Term.op (NN.GraphSpec.DAG.PrimOp.rmsNorm [] latentDim hLatent)
    (.cons routedOutput (.cons routedNormScale .nil))
  let projected := Term.op (PrimOp.broadcastVecMat [] [] [] latentDim modelDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
    (.cons normalized (.cons upProject .nil))
  { initParams := initialParams modelDim latentDim sharedHidden routedHidden numShared numRouted
    body := Term.op (NN.GraphSpec.DAG.PrimOp.add [modelDim])
      (.cons sharedOutput (.cons projected .nil)) }

/-- For a nonempty route, the packed graph computes `forward` with that route. -/
theorem modelGivenRoute_specFwd_eq_forward
    {modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat}
    (hLatent : 0 < latentDim) (hActive : 0 < activeExperts)
    (moe : KimiK3.StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared
      numRouted activeExperts)
    (route : Route numRouted activeExperts)
    (input : Tensor ℝ [modelDim]) (gateCap upCap : ℝ) :
    (modelGivenRoute modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts
      route hLatent).specFwd (parameters moe) (inputs input gateCap upCap) =
      moe.forward route gateCap upCap input := by
  -- Match the graph's scalar fold with the selected-score sum used to normalize route weights.
  have selectedTotal_eq :
      (List.finRange activeExperts).foldl
          (fun total slot => Tensor.addSpec total
            (Spec.get (moe.rawRouterScores input) (route slot)))
          (Tensor.zeros []) =
        Tensor.scalar (Tensor.sumSpec (Tensor.stackLeading fun slot => Tensor.scalar
          (Tensor.getScalar (moe.rawRouterScores input) (route slot)))) := by
    rw [show (Tensor.zeros [] : Tensor ℝ []) = Tensor.scalar 0 by
      apply Tensor.ext_scalar
      simp]
    have hFold :
        (List.finRange activeExperts).foldl
          (fun total slot => Tensor.addSpec total
            (Spec.get (moe.rawRouterScores input) (route slot))) (Tensor.scalar 0) =
          Tensor.scalar ((List.finRange activeExperts).foldl
            (fun total slot => total +
              (Spec.get (moe.rawRouterScores input) (route slot)).item) 0) := by
      convert Spec.foldl_add_scalar
        (fun slot => Spec.get (moe.rawRouterScores input) (route slot))
        (List.finRange activeExperts) 0 using 1
      rfl
    rw [hFold]
    congr 1
    rw [List.finRange_foldl_add_eq_finset_sum, Spec.sum_spec_vec]
    apply Finset.sum_congr rfl
    intro slot _
    simp [Tensor.getScalar]
  simp only [KimiK3.StableLatentMoE.rawRouterScores] at selectedTotal_eq
  simp [modelGivenRoute, parameters, inputs,
    NN.GraphSpec.DAG.Model.specFwd, Args.vars, Args.weakenLeft, Args.rename,
    Term.rename, Term.eval_sum, List.foldl_map,
    eval_selectLeadingTerm, Term.eval, Term.evalArgs, Env.tget,
    TorchLean.TensorPack.append,
    NN.GraphSpec.DAG.PrimOp.add, eval_scaleTerm,
    PrimOp.broadcastVecMat, NN.GraphSpec.DAG.PrimOp.sigmoid,
    NN.GraphSpec.DAG.PrimOp.rmsNorm,
    KimiK3.StableLatentMoE.forward, KimiK3.StableLatentMoE.sharedOutput,
    KimiK3.StableLatentMoE.routedAggregate,
    KimiK3.StableLatentMoE.routeWeights_apply, hActive,
    KimiK3.StableLatentMoE.rawRouterScores, KimiK3.Expert.forward,
    div_eq_mul_inv, mul_assoc]
  rw [selectedTotal_eq]
  simp [Spec.get, Spec.get, Tensor.getScalar]
  simp only [KimiK3.RMSNorm.scale_eq_scalePositive hLatent]
  rfl

/--
The sparse graph specialized to the route this model selects for this token.

The DAG still contains the ordinary projections, activations, expert evaluations, and mixture
arithmetic. Only the discrete expert indices are fixed when the graph is assembled. Until the DAG
language carries integer-valued intermediate results this is the right GraphSpec boundary, since
top-k is neither hidden in an opaque floating-point operation nor replaceable by an unrelated route.

Routing depends on both `moe` and `input`, so the graph must be rebuilt whenever a change to either
changes the selected experts. A runtime may instead compute and check a route before running
`modelGivenRoute`, and the theorem below names the route such a checker must produce.
-/
noncomputable def modelAtInput
    {modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat}
    (hActive : activeExperts ≤ numRouted) (hLatent : 0 < latentDim)
    (moe : KimiK3.StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared
      numRouted activeExperts)
    (input : Tensor ℝ [modelDim]) :
    NN.GraphSpec.DAG.Model
      (Params modelDim latentDim sharedHidden routedHidden numShared numRouted)
      (Inputs modelDim) [modelDim] :=
  modelGivenRoute modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts
    (moe.route hActive input) hLatent

/--
The route-specialized graph computes `forwardReal`, so the route comes from the adjusted router
scores rather than from the caller as in `modelGivenRoute_specFwd_eq_forward`.
-/
theorem modelAtInput_specFwd_eq_forwardReal
    {modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat}
    (hActive : activeExperts ≤ numRouted) (hLatent : 0 < latentDim)
    (hActivePos : 0 < activeExperts)
    (moe : KimiK3.StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared
      numRouted activeExperts)
    (input : Tensor ℝ [modelDim]) (gateCap upCap : ℝ) :
    (modelAtInput hActive hLatent moe input).specFwd
        (parameters moe) (inputs input gateCap upCap) =
      moe.forwardReal hActive gateCap upCap input := by
  exact modelGivenRoute_specFwd_eq_forward hLatent hActivePos moe
    (moe.route hActive input) input gateCap upCap

end StableLatentMoE
end GraphSpec
end KimiK3
