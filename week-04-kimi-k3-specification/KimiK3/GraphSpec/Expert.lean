/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.FeedForward
public import KimiK3.GraphSpec.Primitives
public import NN.Proofs.Tensor.Basic.Algebra

/-!
# Executable SiTU expert graph

This module expresses a SiTU-GLU expert in TorchLean's typed GraphSpec DAG. It is assembled from
ordinary linear, elementwise, `tanh`, and `sigmoid` primitives. No opaque K3 operation is allowed
to call `Expert.forward` behind the graph interface.
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
  [ .dim inputDim (.dim hiddenDim .scalar),
    .dim inputDim (.dim hiddenDim .scalar),
    .dim hiddenDim (.dim outputDim .scalar) ]

/-- Non-parameter inputs: one feature vector followed by the two SiTU caps. -/
abbrev Inputs (inputDim : Nat) : List Shape :=
  [.dim inputDim .scalar, .scalar, .scalar]

/-- A deterministic initialization supplied only to satisfy GraphSpec's model package.

Training code normally replaces this typed list with initialized or checkpoint-loaded parameters.
The semantic theorem below quantifies over arbitrary expert weights.
-/
def initialParams (inputDim hiddenDim outputDim : Nat) :
    TorchLean.TensorPack Float (Params inputDim hiddenDim outputDim) :=
  TorchLean.TensorPack.zero

/-- Build the typed DAG term for one SiTU expert from explicit input and parameter terms.

This is the compositional form used when an expert is embedded in a larger graph such as Stable
LatentMoE. It contains the same ordinary operations as `model`; no expert computation is hidden
behind an opaque node.
-/
def term {Γ : List Shape} (inputDim hiddenDim outputDim : Nat)
    (input : Term Γ (.dim inputDim .scalar))
    (gateWeight upWeight : Term Γ (.dim inputDim (.dim hiddenDim .scalar)))
    (downWeight : Term Γ (.dim hiddenDim (.dim outputDim .scalar)))
    (gateCap upCap : Term Γ .scalar) : Term Γ (.dim outputDim .scalar) :=
  let gate := Term.op (PrimOp.broadcastVecMat .scalar .scalar .scalar inputDim hiddenDim
      (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar))
    (.cons input (.cons gateWeight .nil))
  let up := Term.op (PrimOp.broadcastVecMat .scalar .scalar .scalar inputDim hiddenDim
      (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar))
    (.cons input (.cons upWeight .nil))
  let cappedGate := GraphSpec.softCapTerm [hiddenDim] gateCap gate
  let sigmoidGate := Term.op (NN.GraphSpec.DAG.PrimOp.sigmoid (.dim hiddenDim .scalar))
    (.cons gate .nil)
  let gated := Term.op (NN.GraphSpec.DAG.PrimOp.mul (.dim hiddenDim .scalar))
    (.cons cappedGate (.cons sigmoidGate .nil))
  let cappedUp := GraphSpec.softCapTerm [hiddenDim] upCap up
  let hidden := Term.op (NN.GraphSpec.DAG.PrimOp.mul (.dim hiddenDim .scalar))
    (.cons gated (.cons cappedUp .nil))
  Term.op (PrimOp.broadcastVecMat .scalar .scalar .scalar hiddenDim outputDim
      (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar))
    (.cons hidden (.cons downWeight .nil))

/-- Evaluation of the compositional expert term is the SiTU expert equation on its subterms. -/
@[simp] theorem eval_term {Γ : List Shape}
    (env : TorchLean.TensorPack ℝ Γ) (inputDim hiddenDim outputDim : Nat)
    (input : Term Γ (.dim inputDim .scalar))
    (gateWeight upWeight : Term Γ (.dim inputDim (.dim hiddenDim .scalar)))
    (downWeight : Term Γ (.dim hiddenDim (.dim outputDim .scalar)))
    (gateCap upCap : Term Γ .scalar) :
    Term.eval env
        (term inputDim hiddenDim outputDim input gateWeight upWeight downWeight gateCap upCap) =
      vecMatMulSpec
        (SiTU.vector
          (Tensor.item (Term.eval env gateCap))
          (Tensor.item (Term.eval env upCap))
          (vecMatMulSpec (Term.eval env input) (Term.eval env gateWeight))
          (vecMatMulSpec (Term.eval env input) (Term.eval env upWeight)))
        (Term.eval env downWeight) := by
  simp [term, Term.eval, Term.evalArgs,
    PrimOp.broadcastVecMat, NN.GraphSpec.DAG.PrimOp.sigmoid,
    NN.GraphSpec.DAG.PrimOp.mul, one_div, SiTU.expanded_eq_vector]

/-- Typed GraphSpec representation of one SiTU-GLU expert. -/
def model (inputDim hiddenDim outputDim : Nat) :
    NN.GraphSpec.DAG.Model
      (Params inputDim hiddenDim outputDim)
      (Inputs inputDim)
      (.dim outputDim .scalar) :=
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
    (input : Tensor α (.dim inputDim .scalar)) (gateCap upCap : α) :
    TorchLean.TensorPack α (Inputs inputDim) :=
  .cons input <| .cons (.scalar gateCap) <| .cons (.scalar upCap) .nil

/-- The typed executable graph denotes the original SiTU expert equation. -/
theorem specFwd_eq_forward {inputDim hiddenDim outputDim : Nat}
    (expert : KimiK3.Expert ℝ inputDim hiddenDim outputDim)
    (input : Tensor ℝ (.dim inputDim .scalar)) (gateCap upCap : ℝ) :
    (model inputDim hiddenDim outputDim).specFwd
        (parameters expert) (inputs input gateCap upCap) =
      expert.forward gateCap upCap input := by
  simp [model, parameters, inputs, NN.GraphSpec.DAG.Model.specFwd,
    TorchLean.TensorPack.append, Args.vars, Args.weakenLeft,
    Term.weakenLeft, Term.rename, Term.eval, Env.tget,
    KimiK3.Expert.forward, Params, Inputs]

end Expert

end GraphSpec
end KimiK3
