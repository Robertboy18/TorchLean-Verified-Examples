/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.FeedForward.Expert
public import KimiK3.GraphSpec.Primitives
public import NN.Proofs.Tensor.Basic.Algebra

/-!
# Executable SiTU expert graph

A SiTU-GLU expert in TorchLean's typed GraphSpec DAG, built from ordinary linear, elementwise,
`tanh`, and `sigmoid` primitives. The evaluation theorem shows that the graph computes
`Expert.forward` over real-valued tensors.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec

open Spec TorchLean
open TorchLean.Tensor
open NN.GraphSpec.DAG
open Runtime.Autograd.Torch

namespace Expert

/-- Parameter shapes of a SiTU expert in K3's `[input, output]` matrix layout. -/
abbrev Params (inputDim hiddenDim outputDim : Nat) : List Shape :=
  [ [inputDim, hiddenDim],
    [inputDim, hiddenDim],
    [hiddenDim, outputDim] ]

/-- Non-parameter inputs: one feature vector followed by the two SiTU caps. -/
abbrev Inputs (inputDim : Nat) : List Shape :=
  [[inputDim], [], []]

/-- Zero initialization, present because GraphSpec's model package requires one.

Training code replaces it with initialized or checkpoint-loaded parameters, and the semantic
theorem below holds for arbitrary expert weights.
-/
def initialParams (inputDim hiddenDim outputDim : Nat) :
    TorchLean.TensorPack Float (Params inputDim hiddenDim outputDim) :=
  TorchLean.TensorPack.zero

/-- Typed DAG term for one SiTU expert, given input and parameter terms.

Use this form to embed an expert in a larger graph such as Stable LatentMoE.
-/
def term {Γ : List Shape} (inputDim hiddenDim outputDim : Nat)
    (input : Term Γ [inputDim])
    (gateWeight upWeight : Term Γ [inputDim, hiddenDim])
    (downWeight : Term Γ [hiddenDim, outputDim])
    (gateCap upCap : Term Γ []) : Term Γ [outputDim] :=
  let gate := Term.op (PrimOp.broadcastVecMat [] [] [] inputDim hiddenDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
    (.cons input (.cons gateWeight .nil))
  let up := Term.op (PrimOp.broadcastVecMat [] [] [] inputDim hiddenDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
    (.cons input (.cons upWeight .nil))
  let cappedGate := GraphSpec.softCapTerm [hiddenDim] gateCap gate
  let sigmoidGate := Term.op (NN.GraphSpec.DAG.PrimOp.sigmoid [hiddenDim])
    (.cons gate .nil)
  let gated := Term.op (NN.GraphSpec.DAG.PrimOp.mul [hiddenDim])
    (.cons cappedGate (.cons sigmoidGate .nil))
  let cappedUp := GraphSpec.softCapTerm [hiddenDim] upCap up
  let hidden := Term.op (NN.GraphSpec.DAG.PrimOp.mul [hiddenDim])
    (.cons gated (.cons cappedUp .nil))
  Term.op (PrimOp.broadcastVecMat [] [] [] hiddenDim outputDim
      (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []))
    (.cons hidden (.cons downWeight .nil))

/-- Evaluating the expert term gives the SiTU expert equation on the subterms' values. -/
@[simp] theorem eval_term {Γ : List Shape}
    (env : TorchLean.TensorPack ℝ Γ) (inputDim hiddenDim outputDim : Nat)
    (input : Term Γ [inputDim])
    (gateWeight upWeight : Term Γ [inputDim, hiddenDim])
    (downWeight : Term Γ [hiddenDim, outputDim])
    (gateCap upCap : Term Γ []) :
    Term.eval env
        (term inputDim hiddenDim outputDim input gateWeight upWeight downWeight gateCap upCap) =
      vecMatMulSpec
        (SiTU.vector
          (Tensor.item (Term.eval env gateCap))
          (Tensor.item (Term.eval env upCap))
          (vecMatMulSpec (Term.eval env input) (Term.eval env gateWeight))
          (vecMatMulSpec (Term.eval env input) (Term.eval env upWeight)))
        (Term.eval env downWeight) := by
  -- Expand the graph primitives, then identify the two capped branches with SiTU.
  simp [term, Term.eval, Term.evalArgs,
    PrimOp.broadcastVecMat, NN.GraphSpec.DAG.PrimOp.sigmoid,
    NN.GraphSpec.DAG.PrimOp.mul, one_div, SiTU.expanded_eq_vector]

/-- Typed GraphSpec representation of one SiTU-GLU expert. -/
def model (inputDim hiddenDim outputDim : Nat) :
    NN.GraphSpec.DAG.Model
      (Params inputDim hiddenDim outputDim)
      (Inputs inputDim)
      [outputDim] :=
  { initParams := initialParams inputDim hiddenDim outputDim
    body := by
      let Γ := Params inputDim hiddenDim outputDim ++ Inputs inputDim
      let envTerms : Args Γ
          (Params inputDim hiddenDim outputDim ++ Inputs inputDim) := by
        simpa [Γ] using Args.vars Γ
      let .cons gateWeight <| .cons upWeight <| .cons downWeight <| .cons input <|
          .cons gateCap <| .cons upCap .nil := envTerms
      exact term inputDim hiddenDim outputDim input gateWeight upWeight downWeight gateCap upCap }

/-- Convert the theorem-oriented expert record to GraphSpec's parameter ABI. -/
def parameters {α : Type} [Storage α] {inputDim hiddenDim outputDim : Nat}
    (expert : KimiK3.Expert α inputDim hiddenDim outputDim) :
    TorchLean.TensorPack α (Params inputDim hiddenDim outputDim) :=
  .cons expert.gateWeight <| .cons expert.upWeight <| .cons expert.downWeight .nil

/-- Package an expert input and its two caps in GraphSpec's typed input ABI. -/
def inputs {α : Type} [Storage α] {inputDim : Nat}
    (input : Tensor α [inputDim]) (gateCap upCap : α) :
    TorchLean.TensorPack α (Inputs inputDim) :=
  .cons input <| .cons (Tensor.scalar gateCap) <| .cons (Tensor.scalar upCap) .nil

/-- The expert graph computes `Expert.forward`. -/
theorem specFwd_eq_forward {inputDim hiddenDim outputDim : Nat}
    (expert : KimiK3.Expert ℝ inputDim hiddenDim outputDim)
    (input : Tensor ℝ [inputDim]) (gateCap upCap : ℝ) :
    (model inputDim hiddenDim outputDim).specFwd
        (parameters expert) (inputs input gateCap upCap) =
      expert.forward gateCap upCap input := by
  simp [model, parameters, inputs, NN.GraphSpec.DAG.Model.specFwd,
    TorchLean.TensorPack.append, Args.vars, Args.weakenLeft, Args.rename,
    Term.rename, Term.eval, Env.tget,
    KimiK3.Expert.forward, Params, Inputs]

end Expert

end GraphSpec
end KimiK3
