/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Expert

/-!
# Stable LatentMoE graph

This module lowers one routed Stable LatentMoE evaluation to TorchLean's typed DAG. Expert weights
are stored in leading-axis banks. A proved `Route` selects slices from those banks, while every
continuous operation remains visible as an ordinary GraphSpec node.

Routing is deliberately supplied to the graph rather than hidden in a floating-point primitive.
Top-k is a discrete decision: `Route.chooseTopK_isTopK` proves its mathematical contract, and the
graph below executes the differentiable computation for the resulting route.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec
namespace StableLatentMoE

open Spec
open Spec.Tensor
open NN.GraphSpec.DAG

/-- Packed parameter shapes for one Stable LatentMoE layer. -/
abbrev Params (modelDim latentDim sharedHidden routedHidden numShared numRouted : Nat) :
    List Shape :=
  [ .dim modelDim (.dim latentDim .scalar),
    .dim latentDim (.dim modelDim .scalar),
    .dim latentDim .scalar,
    .dim modelDim (.dim numRouted .scalar),
    .dim numRouted .scalar,
    .dim numShared (.dim modelDim (.dim sharedHidden .scalar)),
    .dim numShared (.dim modelDim (.dim sharedHidden .scalar)),
    .dim numShared (.dim sharedHidden (.dim modelDim .scalar)),
    .dim numRouted (.dim latentDim (.dim routedHidden .scalar)),
    .dim numRouted (.dim latentDim (.dim routedHidden .scalar)),
    .dim numRouted (.dim routedHidden (.dim latentDim .scalar)) ]

/-- One model-width token and the two SiTU caps are non-parameter graph inputs. -/
abbrev Inputs (modelDim : Nat) : List Shape :=
  [.dim modelDim .scalar, .scalar, .scalar]

/-- Zero initialization used only by GraphSpec's model package. -/
def initialParams (modelDim latentDim sharedHidden routedHidden numShared numRouted : Nat) :
    TorchLean.TensorPack Float
      (Params modelDim latentDim sharedHidden routedHidden numShared numRouted) :=
  .cons (Spec.fill 0 (.dim modelDim (.dim latentDim .scalar))) <|
    .cons (Spec.fill 0 (.dim latentDim (.dim modelDim .scalar))) <|
      .cons (Spec.fill 0 (.dim latentDim .scalar)) <|
        .cons (Spec.fill 0 (.dim modelDim (.dim numRouted .scalar))) <|
          .cons (Spec.fill 0 (.dim numRouted .scalar)) <|
            .cons (Spec.fill 0 (.dim numShared (.dim modelDim (.dim sharedHidden .scalar)))) <|
              .cons (Spec.fill 0 (.dim numShared (.dim modelDim (.dim sharedHidden .scalar)))) <|
                .cons (Spec.fill 0 (.dim numShared (.dim sharedHidden (.dim modelDim .scalar)))) <|
                  .cons
                    (Spec.fill 0 (.dim numRouted (.dim latentDim (.dim routedHidden .scalar)))) <|
                    .cons
                      (Spec.fill 0 (.dim numRouted (.dim latentDim (.dim routedHidden .scalar)))) <|
                      .cons
                        (Spec.fill 0 (.dim numRouted (.dim routedHidden (.dim latentDim .scalar))))
                        .nil

/-- Pack the mathematical MoE record into the graph's expert-bank layout. -/
def parameters {α : Type}
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
def inputs {α : Type} {modelDim : Nat}
    (input : Tensor α (.dim modelDim .scalar)) (gateCap upCap : α) :
    TorchLean.TensorPack α (Inputs modelDim) :=
  .cons input <| .cons (.scalar gateCap) <| .cons (.scalar upCap) .nil

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

/-- TorchLean graph for a Stable LatentMoE evaluation with a fixed, proved route. -/
def modelGivenRoute
    (modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat)
    (route : Route numRouted activeExperts) (hLatent : 0 < latentDim) :
    NN.GraphSpec.DAG.Model
      (Params modelDim latentDim sharedHidden routedHidden numShared numRouted)
      (Inputs modelDim) (.dim modelDim .scalar) :=
  let Γ := Params modelDim latentDim sharedHidden routedHidden numShared numRouted ++ Inputs modelDim
  let downProject : Term Γ (.dim modelDim (.dim latentDim .scalar)) :=
    Term.var .head
  let upProject : Term Γ (.dim latentDim (.dim modelDim .scalar)) :=
    Term.var (.tail .head)
  let routedNormScale : Term Γ (.dim latentDim .scalar) :=
    Term.var (.tail (.tail .head))
  let routerWeight : Term Γ (.dim modelDim (.dim numRouted .scalar)) :=
    Term.var (.tail (.tail (.tail .head)))
  let _routerBias : Term Γ (.dim numRouted .scalar) :=
    Term.var (.tail (.tail (.tail (.tail .head))))
  let sharedGate : Term Γ (.dim numShared (.dim modelDim (.dim sharedHidden .scalar))) :=
    Term.var (.tail (.tail (.tail (.tail (.tail .head)))))
  let sharedUp : Term Γ (.dim numShared (.dim modelDim (.dim sharedHidden .scalar))) :=
    Term.var (.tail (.tail (.tail (.tail (.tail (.tail .head))))))
  let sharedDown : Term Γ (.dim numShared (.dim sharedHidden (.dim modelDim .scalar))) :=
    Term.var (.tail (.tail (.tail (.tail (.tail (.tail (.tail .head)))))))
  let routedGate : Term Γ (.dim numRouted (.dim latentDim (.dim routedHidden .scalar))) :=
    Term.var (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail .head))))))))
  let routedUp : Term Γ (.dim numRouted (.dim latentDim (.dim routedHidden .scalar))) :=
    Term.var (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail .head)))))))))
  let routedDown : Term Γ (.dim numRouted (.dim routedHidden (.dim latentDim .scalar))) :=
    Term.var
      (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail .head))))))))))
  let input : Term Γ (.dim modelDim .scalar) :=
    Term.var
      (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail
        (.tail .head)))))))))))
  let gateCap : Term Γ .scalar :=
    Term.var
      (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail
        (.tail (.tail .head))))))))))))
  let upCap : Term Γ .scalar :=
    Term.var
      (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail (.tail
        (.tail (.tail (.tail .head)))))))))))))
  let sharedOutput := Term.sum (.dim modelDim .scalar) <|
    (List.finRange numShared).map fun expert =>
      let gateWeight := selectLeadingTerm numShared
        (.dim modelDim (.dim sharedHidden .scalar)) expert sharedGate
      let upWeight := selectLeadingTerm numShared
        (.dim modelDim (.dim sharedHidden .scalar)) expert sharedUp
      let downWeight := selectLeadingTerm numShared
        (.dim sharedHidden (.dim modelDim .scalar)) expert sharedDown
      Expert.term modelDim sharedHidden modelDim input gateWeight upWeight downWeight gateCap upCap
  let latent := Term.op (NN.GraphSpec.DAG.PrimOp.vecMat modelDim latentDim)
    (.cons input (.cons downProject .nil))
  let rawScores := Term.op (NN.GraphSpec.DAG.PrimOp.sigmoid (.dim numRouted .scalar))
    (.cons (Term.op (NN.GraphSpec.DAG.PrimOp.vecMat modelDim numRouted)
      (.cons input (.cons routerWeight .nil))) .nil)
  let selectedTotal := Term.sum .scalar <|
    (List.finRange activeExperts).map fun slot =>
      let score := selectLeadingTerm numRouted .scalar (route.expert slot) rawScores
      score
  let inverseTotal := Term.op (NN.GraphSpec.DAG.PrimOp.inv .scalar) (.cons selectedTotal .nil)
  let routedOutput := Term.sum (.dim latentDim .scalar) <|
    (List.finRange activeExperts).map fun slot =>
      let expertIndex := route.expert slot
      let gateWeight := selectLeadingTerm numRouted
        (.dim latentDim (.dim routedHidden .scalar)) expertIndex routedGate
      let upWeight := selectLeadingTerm numRouted
        (.dim latentDim (.dim routedHidden .scalar)) expertIndex routedUp
      let downWeight := selectLeadingTerm numRouted
        (.dim routedHidden (.dim latentDim .scalar)) expertIndex routedDown
      let expertOutput := Expert.term latentDim routedHidden latentDim latent gateWeight upWeight
        downWeight gateCap upCap
      let score := selectLeadingTerm numRouted .scalar expertIndex rawScores
      let weight := Term.op (NN.GraphSpec.DAG.PrimOp.mul .scalar)
        (.cons score (.cons inverseTotal .nil))
      Term.op (NN.GraphSpec.DAG.PrimOp.scalarMul (.dim latentDim .scalar))
        (.cons weight (.cons expertOutput .nil))
  let normalized := Term.op (NN.GraphSpec.DAG.PrimOp.rmsNorm .scalar latentDim hLatent)
    (.cons routedOutput (.cons routedNormScale .nil))
  let projected := Term.op (NN.GraphSpec.DAG.PrimOp.vecMat latentDim modelDim)
    (.cons normalized (.cons upProject .nil))
  { initParams := initialParams modelDim latentDim sharedHidden routedHidden numShared numRouted
    body := Term.op (NN.GraphSpec.DAG.PrimOp.add (.dim modelDim .scalar))
      (.cons sharedOutput (.cons projected .nil)) }

/-- The packed TorchLean graph denotes the Stable LatentMoE equation for the supplied route. -/
theorem modelGivenRoute_specFwd_eq_forward
    {modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat}
    (hLatent : 0 < latentDim) (hActive : 0 < activeExperts)
    (moe : KimiK3.StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared
      numRouted activeExperts)
    (route : Route numRouted activeExperts)
    (input : Tensor ℝ (.dim modelDim .scalar)) (gateCap upCap : ℝ) :
    (modelGivenRoute modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts
      route hLatent).specFwd (parameters moe) (inputs input gateCap upCap) =
      moe.forward route gateCap upCap input := by
  have selectedTotal_eq :
      (List.finRange activeExperts).foldl
          (fun total slot => Tensor.addSpec total
            (Spec.get (moe.rawRouterScores input) (route.expert slot)))
          (Spec.fill 0 .scalar) =
        Tensor.scalar (Tensor.sumSpec (Tensor.dim fun slot => Tensor.scalar
          (Tensor.getScalar (moe.rawRouterScores input) (route.expert slot)))) := by
    change
      (List.finRange activeExperts).foldl
        (fun total slot => total + Spec.get (moe.rawRouterScores input) (route.expert slot))
        (Tensor.scalar 0) = _
    rw [Spec.foldl_add_scalar]
    congr 1
    rw [List.finRange_foldl_add_eq_finset_sum, Spec.sum_spec_vec]
    apply Finset.sum_congr rfl
    intro slot _
    rfl
  simp only [KimiK3.StableLatentMoE.rawRouterScores] at selectedTotal_eq
  simp [modelGivenRoute, parameters, inputs,
    NN.GraphSpec.DAG.Model.specFwd, Term.eval_sum, List.foldl_map,
    eval_selectLeadingTerm, Term.eval, Term.evalArgs, Env.tget,
    TorchLean.TensorPack.append,
    NN.GraphSpec.DAG.PrimOp.add,
    NN.GraphSpec.DAG.PrimOp.vecMat, NN.GraphSpec.DAG.PrimOp.sigmoid,
    NN.GraphSpec.DAG.PrimOp.rmsNorm,
    KimiK3.StableLatentMoE.forward, KimiK3.StableLatentMoE.sharedOutput,
    KimiK3.StableLatentMoE.routedAggregate,
    KimiK3.StableLatentMoE.routeWeights_apply, hActive,
    KimiK3.StableLatentMoE.rawRouterScores, KimiK3.Expert.forward,
    div_eq_mul_inv, mul_assoc]
  rw [selectedTotal_eq]
  simp [Spec.get, Spec.get, Tensor.getScalar]
  rw [KimiK3.RMSNorm.scale_eq_scalePositive hLatent]
  rfl

/--
Specialize the sparse graph to the route selected by this model on this token.

The resulting DAG still contains the ordinary projections, activations, expert evaluations, and
mixture arithmetic. Only the discrete expert indices are fixed while the graph is assembled. This
is the appropriate GraphSpec boundary until the DAG language carries integer-valued intermediate
results: it does not conceal top-k inside an opaque floating-point operation, and it does not permit
an unrelated route to stand in for K3 routing.

Because routing depends on both `moe` and `input`, this graph must be rebuilt if either changes enough
to change the selected experts. Runtime implementations may instead compute and check a route before
executing `modelGivenRoute`; the theorem below identifies the exact route that such a checker must
produce.
-/
noncomputable def modelAtInput
    {modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat}
    (hActive : activeExperts ≤ numRouted) (hLatent : 0 < latentDim)
    (moe : KimiK3.StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared
      numRouted activeExperts)
    (input : Tensor ℝ (.dim modelDim .scalar)) :
    NN.GraphSpec.DAG.Model
      (Params modelDim latentDim sharedHidden routedHidden numShared numRouted)
      (Inputs modelDim) (.dim modelDim .scalar) :=
  modelGivenRoute modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts
    (moe.route hActive input) hLatent

/--
The route-specialized graph denotes K3's real-valued Stable LatentMoE semantics, including the
deterministic score-based routing decision. This strengthens `modelGivenRoute_specFwd_eq_forward`:
the right-hand side is `forwardReal`, whose route is derived from the adjusted router scores rather
than supplied by the caller.
-/
theorem modelAtInput_specFwd_eq_forwardReal
    {modelDim latentDim sharedHidden routedHidden numShared numRouted activeExperts : Nat}
    (hActive : activeExperts ≤ numRouted) (hLatent : 0 < latentDim)
    (hActivePos : 0 < activeExperts)
    (moe : KimiK3.StableLatentMoE ℝ modelDim latentDim sharedHidden routedHidden numShared
      numRouted activeExperts)
    (input : Tensor ℝ (.dim modelDim .scalar)) (gateCap upCap : ℝ) :
    (modelAtInput hActive hLatent moe input).specFwd
        (parameters moe) (inputs input gateCap upCap) =
      moe.forwardReal hActive gateCap upCap input := by
  exact modelGivenRoute_specFwd_eq_forward hLatent hActivePos moe
    (moe.route hActive input) input gateCap upCap

end StableLatentMoE
end GraphSpec
end KimiK3
