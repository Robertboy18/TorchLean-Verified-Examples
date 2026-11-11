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

The expert weights are packed into tensor banks. A `Route` selects rows from those banks, and
the remaining computation uses TorchLean tensor operations. `Route.chooseTopK_isTopK` proves the
contract of the discrete route used by `modelAtInput`.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec
namespace StableLatentMoE

open TorchLean
open TorchLean.Tensor

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

/-- Zero initialization used only by the model package. -/
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
      .cons (Tensor.dim fun expert => (moe.shared expert).gateWeight) <|
        .cons (Tensor.dim fun expert => (moe.shared expert).upWeight) <|
          .cons (Tensor.dim fun expert => (moe.shared expert).downWeight) <|
            .cons (Tensor.dim fun expert => (moe.routed expert).gateWeight) <|
              .cons (Tensor.dim fun expert => (moe.routed expert).upWeight) <|
                .cons (Tensor.dim fun expert => (moe.routed expert).downWeight) .nil

/-- Package one token and its SiTU caps in the graph input layout. -/
def inputs {α : Type} [Storage α] {modelDim : Nat}
    (input : Tensor α [modelDim]) (gateCap upCap : α) :
    TorchLean.TensorPack α (Inputs modelDim) :=
  .cons input <| .cons (Tensor.scalar gateCap) <| .cons (Tensor.scalar upCap) .nil

/-- Select one entry from a leading parameter bank. -/
def selectLeadingTerm {Γ : List Shape} (count : Nat) (shape : Shape)
    (index : Fin count) (bank : Term Γ (.dim count shape)) : Term Γ shape :=
  ⟨fun env => Tensor.unstack (bank.eval env) index⟩

@[simp] theorem eval_selectLeadingTerm {Γ : List Shape} (env : TorchLean.TensorPack ℝ Γ)
    (count : Nat) (shape : Shape) (index : Fin count)
    (bank : Term Γ (.dim count shape)) :
    Term.eval env (selectLeadingTerm count shape index bank) =
      Tensor.unstack (Term.eval env bank) index := by
  rfl

/-- TorchLean graph for a Stable LatentMoE evaluation with a fixed route. -/
def modelGivenRoute
    (modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat)
    (route : Route numRouted activeExperts) (hLatent : 0 < latentDim) :
    Model
      (Params modelDim latentDim sharedHidden routedHidden numShared numRouted)
      (Inputs modelDim) [modelDim] :=
  let Γ :=
    Params modelDim latentDim sharedHidden routedHidden numShared numRouted ++ Inputs modelDim
  let envTerms : Args Γ (Params modelDim latentDim sharedHidden routedHidden numShared numRouted ++
      Inputs modelDim) := Args.vars Γ
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
  let rawScores := Term.op (PrimOp.sigmoid [numRouted])
    (.cons (Term.op (PrimOp.broadcastVecMat [] [] [] modelDim numRouted
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
      (.cons input (.cons routerWeight .nil))) .nil)
  let weights : Term Γ [activeExperts] := ⟨fun env =>
    let scores := rawScores.eval env
    Normalize.probabilities (Tensor.ofFn fun slot => scores.getScalar (route slot))⟩
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
      let weight := selectLeadingTerm activeExperts [] slot weights
      scaleTerm [latentDim] weight expertOutput
  let normalized := Term.op (PrimOp.rmsNorm [] latentDim hLatent)
    (.cons routedOutput (.cons routedNormScale .nil))
  let projected := Term.op (PrimOp.broadcastVecMat [] [] [] latentDim modelDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
    (.cons normalized (.cons upProject .nil))
  { initParams := initialParams modelDim latentDim sharedHidden routedHidden numShared numRouted
    body := Term.op (PrimOp.add [modelDim])
      (.cons sharedOutput (.cons projected .nil)) }

/-- For a nonempty route, the packed graph computes `forward` with that route. -/
theorem modelGivenRoute_specFwd_eq_forward
    {modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat}
    (hLatent : 0 < latentDim) (_hActive : 0 < activeExperts)
    (moe : KimiK3.StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared
      numRouted activeExperts)
    (route : Route numRouted activeExperts)
    (input : Tensor ℝ [modelDim]) (gateCap upCap : ℝ) :
    (modelGivenRoute modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts
      route hLatent).specFwd (parameters moe) (inputs input gateCap upCap) =
      moe.forward route gateCap upCap input := by
  simp only [modelGivenRoute, Model.specFwd, parameters, inputs, Params, Inputs,
    List.cons_append, List.nil_append,
    Args.vars, Args.weakenLeft, Args.rename, Term.rename_var, Term.eval_mk, Term.eval_var,
    Env.tget, Pack.append, Term.eval_op, Term.evalArgs, Term.eval_sum,
    List.foldl_map, eval_selectLeadingTerm, Tensor.unstack_dim, Expert.eval_term, Tensor.item_scalar,
    PrimOp.add_specFwd, broadcastVecMat_scalar_specFwd, PrimOp.sigmoid_specFwd,
    eval_scaleTerm, PrimOp.rmsNorm_specFwd, rmsNormSpec_scalar_eq_scale,
    KimiK3.StableLatentMoE.forward, KimiK3.StableLatentMoE.sharedOutput,
    KimiK3.StableLatentMoE.routedAggregate, KimiK3.StableLatentMoE.routeWeights,
    KimiK3.StableLatentMoE.rawRouterScores, KimiK3.Expert.forward,
    Tensor.getScalar, Tensor.zeros]

/--
The sparse graph specialized to the route this model selects for this token.

The projections, activations, expert evaluations, and mixture arithmetic use the route selected
from `moe` and `input`. A caller that already has this route can pass it to `modelGivenRoute`.
-/
noncomputable def modelAtInput
    {modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat}
    (hActive : activeExperts ≤ numRouted) (hLatent : 0 < latentDim)
    (moe : KimiK3.StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared
      numRouted activeExperts)
    (input : Tensor ℝ [modelDim]) :
    Model
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
