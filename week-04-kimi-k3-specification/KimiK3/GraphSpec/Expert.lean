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
ordinary
linear, elementwise, `tanh`, and `sigmoid` primitives. No opaque K3 operation is allowed to call
`Expert.forward` behind the graph interface.
-/

@[expose] public section

namespace KimiK3
namespace GraphSpec

open Spec
open Spec.Tensor
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
  .cons (Spec.fill 0 (.dim inputDim (.dim hiddenDim .scalar))) <|
    .cons (Spec.fill 0 (.dim inputDim (.dim hiddenDim .scalar))) <|
      .cons (Spec.fill 0 (.dim hiddenDim (.dim outputDim .scalar))) .nil

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
  let gate := Term.op (NN.GraphSpec.DAG.PrimOp.vecMat inputDim hiddenDim)
    (.cons input (.cons gateWeight .nil))
  let up := Term.op (NN.GraphSpec.DAG.PrimOp.vecMat inputDim hiddenDim)
    (.cons input (.cons upWeight .nil))
  let cappedGate := Term.op (PrimOp.softCap hiddenDim) (.cons gateCap (.cons gate .nil))
  let sigmoidGate := Term.op (NN.GraphSpec.DAG.PrimOp.sigmoid (.dim hiddenDim .scalar))
    (.cons gate .nil)
  let gated := Term.op (NN.GraphSpec.DAG.PrimOp.mul (.dim hiddenDim .scalar))
    (.cons cappedGate (.cons sigmoidGate .nil))
  let cappedUp := Term.op (PrimOp.softCap hiddenDim) (.cons upCap (.cons up .nil))
  let hidden := Term.op (NN.GraphSpec.DAG.PrimOp.mul (.dim hiddenDim .scalar))
    (.cons gated (.cons cappedUp .nil))
  Term.op (NN.GraphSpec.DAG.PrimOp.vecMat hiddenDim outputDim)
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
  generalize hGateCap : Term.eval env gateCap = gateCapValue
  generalize hUpCap : Term.eval env upCap = upCapValue
  cases gateCapValue with
  | scalar gateCapValue =>
      cases upCapValue with
      | scalar upCapValue =>
          simp [term, Term.eval, Term.evalArgs, NN.GraphSpec.DAG.PrimOp.vecMat,
            PrimOp.softCap, NN.GraphSpec.DAG.PrimOp.sigmoid, NN.GraphSpec.DAG.PrimOp.mul,
            hGateCap, hUpCap, one_div]
          rw [SiTU.expanded_eq_vector]

/-- Typed GraphSpec representation of one SiTU-GLU expert. -/
def model (inputDim hiddenDim outputDim : Nat) :
    NN.GraphSpec.DAG.Model
      (Params inputDim hiddenDim outputDim)
      (Inputs inputDim)
      (.dim outputDim .scalar) :=
  let Γ : List Shape :=
    Params inputDim hiddenDim outputDim ++ Inputs inputDim
  let gateWeight : Term Γ (.dim inputDim (.dim hiddenDim .scalar)) :=
    Term.var (Γ := Γ) .head
  let upWeight : Term Γ (.dim inputDim (.dim hiddenDim .scalar)) :=
    Term.var (Γ := Γ) (.tail .head)
  let downWeight : Term Γ (.dim hiddenDim (.dim outputDim .scalar)) :=
    Term.var (Γ := Γ) (.tail (.tail .head))
  let input : Term Γ (.dim inputDim .scalar) :=
    Term.var (Γ := Γ) (.tail (.tail (.tail .head)))
  let gateCap : Term Γ .scalar :=
    Term.var (Γ := Γ) (.tail (.tail (.tail (.tail .head))))
  let upCap : Term Γ .scalar :=
    Term.var (Γ := Γ) (.tail (.tail (.tail (.tail (.tail .head)))))
  { initParams := initialParams inputDim hiddenDim outputDim
    body := term inputDim hiddenDim outputDim input gateWeight upWeight downWeight gateCap upCap }

/-- Convert the theorem-oriented expert record to GraphSpec's parameter ABI. -/
def parameters {α : Type} {inputDim hiddenDim outputDim : Nat}
    (expert : KimiK3.Expert α inputDim hiddenDim outputDim) :
    TorchLean.TensorPack α (Params inputDim hiddenDim outputDim) :=
  .cons expert.gateWeight <| .cons expert.upWeight <| .cons expert.downWeight .nil

/-- Package an expert input and its two caps in GraphSpec's typed input ABI. -/
def inputs {α : Type} {inputDim : Nat}
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
    TorchLean.TensorPack.append, Term.eval, Env.tget,
    KimiK3.Expert.forward, Params, Inputs]

end Expert

end GraphSpec
end KimiK3
