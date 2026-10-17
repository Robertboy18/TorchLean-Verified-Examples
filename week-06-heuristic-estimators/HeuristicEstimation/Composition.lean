/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
Authors: Robert Joseph George
-/

module

public import NN.Proofs.Autograd.Tape.Algebra.Soundness
public import NN.Proofs.Tensor.Basic.Folds
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Ring

/-!
# Output discrepancies weighted by the existing reverse pass

Compare two executions of the same TorchLean graph. At each node, subtract its JVP at the
reference execution from the actual change in its output. The remaining discrepancy contributes
to the final answer through the cotangent computed by TorchLean's existing reverse traversal.

`eval_difference` is an exact identity, not an approximation or a smoothness assertion. Its only
local assumption is the graph's existing JVP/VJP adjoint law. In particular, it also applies to
piecewise-linear nodes with a specified convention at their boundaries. Bounds on the local
remainders and cancellation of the input term in expectation are separate results.
-/

@[expose] public section

namespace HeuristicEstimation.Composition

open Spec (Shape)
open TorchLean Proofs.Autograd.Algebra
open Proofs.TensorAlgebra (dot)
open scoped BigOperators

variable {α : Type} [Storage α] [CommRing α]

/-- Subtracting tensors subtracts their pairings with a fixed output cotangent. -/
theorem dot_sub_left {shape : Shape} (left right seed : Tensor α shape) :
    dot (Tensor.subSpec left right) seed = dot left seed - dot right seed := by
  have cancel : Tensor.addSpec (Tensor.subSpec left right) right = left := by
    apply Tensor.Internal.Rep.ext
    intro coordinate
    simp [Tensor.addSpec, Tensor.subSpec, Tensor.map2Spec]
  have paired := congrArg (fun value => dot value seed) cancel
  rw [Proofs.TensorAlgebra.dot_add_left] at paired
  exact eq_sub_of_add_eq paired

/-- The pairing of a context difference is the difference of its pairings. -/
theorem dotList_sub_left {shapes : List Shape}
    (left right seed : TorchLean.TensorPack α shapes) :
    TensorPack.dotList (TorchLean.TensorPack.sub left right) seed =
      TensorPack.dotList left seed - TensorPack.dotList right seed := by
  induction shapes with
  | nil =>
      cases left; cases right; cases seed
      simp [TorchLean.TensorPack.sub, TensorPack.dotList]
  | cons shape shapes ih =>
      cases left with
      | cons first rest =>
          cases right with
          | cons other remaining =>
              cases seed with
              | cons output outputs =>
                  change dot (Tensor.subSpec first other) output +
                    TensorPack.dotList (TorchLean.TensorPack.sub rest remaining) outputs =
                      (dot first output + TensorPack.dotList rest outputs) -
                        (dot other output + TensorPack.dotList remaining outputs)
                  rw [dot_sub_left, ih]
                  ring

private theorem sub_cast {before after : List Shape} (same : before = after)
    (left right : TorchLean.TensorPack α before) :
    TorchLean.TensorPack.sub (TorchLean.TensorPack.cast same left)
        (TorchLean.TensorPack.cast same right) =
      TorchLean.TensorPack.cast same (TorchLean.TensorPack.sub left right) := by
  cases same
  rfl

/-- The part of a node's output change not explained by its reference JVP. -/
def nodeRemainder {data : Type} {inputs : List Shape} {output : Shape}
    (node : Node (α := α) data inputs output)
    (values reference : TorchLean.TensorPack α inputs) (parameters : data) : Tensor α output :=
  Tensor.subSpec (Tensor.subSpec (node.forward values parameters)
    (node.forward reference parameters))
    (node.jvp reference (TorchLean.TensorPack.sub values reference) parameters)

/-- The local discrepancy splits into a pulled-back input change and its exact remainder. -/
theorem node_difference {data : Type} {inputs : List Shape} {output : Shape}
    (node : Node (α := α) data inputs output)
    (values reference : TorchLean.TensorPack α inputs) (parameters : data)
    (seed : Tensor α output) :
    dot (Tensor.subSpec (node.forward values parameters) (node.forward reference parameters))
        seed =
      TensorPack.dotList (TorchLean.TensorPack.sub values reference)
          (node.vjp reference parameters seed) +
        dot (nodeRemainder node values reference parameters) seed := by
  simp only [nodeRemainder, dot_sub_left, node.correct]
  ring

/-- Sum node remainders against the cotangents of the reference reverse traversal.

The recursion uses the same saved reference values, VJPs, and context addition as
`Graph.backpropCtx`. It does not introduce a second graph representation or differentiation rule.
-/
def remainder {data : Type} {inputs : List Shape} {outputs : List Shape}
    (graph : Graph (α := α) data inputs outputs)
    (values reference : TorchLean.TensorPack α inputs) (parameters : data)
    (seed : TorchLean.TensorPack α (inputs ++ outputs)) : α :=
  match graph with
  | .nil => 0
  | .snoc (ss := previousShapes) (τ := output) previous node =>
      let previousSeed := TorchLean.TensorPack.cast
        (List.append_assoc inputs previousShapes [output]).symm seed
      let split := TorchLean.TensorPack.unsnoc previousSeed
      let referenceContext := previous.eval reference parameters
      let contribution := node.vjp referenceContext parameters split.2
      remainder previous values reference parameters
          (TorchLean.TensorPack.add split.1 contribution) +
        dot (nodeRemainder node (previous.eval values parameters) referenceContext parameters)
          split.2

/-- Every output discrepancy is the input discrepancy pulled back by the existing reverse pass,
plus the cotangent-weighted remainders of all nodes. Shared nodes and intermediate output seeds
are included because the statement uses the graph's complete typed context. -/
theorem eval_difference {data : Type} {inputs : List Shape} {outputs : List Shape}
    (graph : Graph (α := α) data inputs outputs) :
    ∀ (values reference : TorchLean.TensorPack α inputs) (parameters : data)
      (seed : TorchLean.TensorPack α (inputs ++ outputs)),
      TensorPack.dotList
          (TorchLean.TensorPack.sub (graph.eval values parameters)
            (graph.eval reference parameters))
          seed =
        TensorPack.dotList (TorchLean.TensorPack.sub values reference)
            (graph.backpropCtx reference parameters seed) +
          remainder graph values reference parameters seed := by
  induction graph with
  | nil =>
      intro values reference parameters seed
      simp only [Graph.eval, Graph.toData, GraphData.eval, Graph.backpropCtx,
        GraphData.backpropCtx, remainder, sub_cast, TensorPack.dotList_cast_left, add_zero]
  | snoc previous node ih =>
      intro values reference parameters seed
      rename_i previousShapes output
      let fullContext := previous.eval values parameters
      let referenceContext := previous.eval reference parameters
      let same : (inputs ++ previousShapes) ++ [output] =
          inputs ++ (previousShapes ++ [output]) := List.append_assoc inputs previousShapes [output]
      let previousSeed := TorchLean.TensorPack.cast same.symm seed
      let split := TorchLean.TensorPack.unsnoc previousSeed
      let contribution := node.vjp referenceContext parameters split.2
      let accumulated := TorchLean.TensorPack.add split.1 contribution
      have seed_split : TorchLean.TensorPack.snoc split.1 split.2 = previousSeed :=
        TorchLean.TensorPack.snoc_unsnoc previousSeed
      have output_split :
          TensorPack.dotList
              (TorchLean.TensorPack.sub ((Graph.snoc previous node).eval values parameters)
                ((Graph.snoc previous node).eval reference parameters)) seed =
            TensorPack.dotList (TorchLean.TensorPack.sub fullContext referenceContext) split.1 +
              dot (Tensor.subSpec (node.forward fullContext parameters)
                (node.forward referenceContext parameters)) split.2 := by
        change TensorPack.dotList
            (TorchLean.TensorPack.sub
              (TorchLean.TensorPack.cast same
                (TorchLean.TensorPack.snoc fullContext (node.forward fullContext parameters)))
              (TorchLean.TensorPack.cast same
                (TorchLean.TensorPack.snoc referenceContext
                  (node.forward referenceContext parameters)))) seed = _
        rw [sub_cast, TensorPack.dotList_cast_left, dotList_sub_left]
        change TensorPack.dotList
              (TorchLean.TensorPack.snoc fullContext (node.forward fullContext parameters))
              previousSeed -
            TensorPack.dotList
              (TorchLean.TensorPack.snoc referenceContext
                (node.forward referenceContext parameters))
              previousSeed = _
        rw [← seed_split, TensorPack.dotList_snoc, TensorPack.dotList_snoc,
          dotList_sub_left, dot_sub_left]
        ring
      calc
        _ = TensorPack.dotList (TorchLean.TensorPack.sub fullContext referenceContext) split.1 +
            dot (Tensor.subSpec (node.forward fullContext parameters)
              (node.forward referenceContext parameters)) split.2 := output_split
        _ = TensorPack.dotList (TorchLean.TensorPack.sub fullContext referenceContext) accumulated +
            dot (nodeRemainder node fullContext referenceContext parameters) split.2 := by
              rw [node_difference, TensorPack.dotList_add_right]
              ring
        _ = TensorPack.dotList (TorchLean.TensorPack.sub values reference)
              (previous.backpropCtx reference parameters accumulated) +
            (remainder previous values reference parameters accumulated +
              dot (nodeRemainder node fullContext referenceContext parameters) split.2) := by
              rw [ih]
              ring
        _ = _ := rfl

/-! ## Bounds from local tensor certificates -/

/-- Coordinatewise error bounds control the pairing with any output cotangent. -/
theorem dot_abs_le {shape : Shape} (error seed radius : Tensor ℝ shape)
    (within : ∀ coordinate, |error coordinate| ≤ radius coordinate) :
    |dot error seed| ≤ dot (Tensor.map abs seed) radius := by
  classical
  simp only [← Spec.dot_eq_tensorAlgebra_dot, Spec.dot, Spec.sum_spec_eq_coord_sum,
    Tensor.mulSpec, Tensor.map2Spec, Tensor.Internal.Rep.zipWith_apply,
    Tensor.map, Tensor.Internal.Rep.map_apply]
  calc
    |∑ coordinate : shape.Coord, error coordinate * seed coordinate| ≤
        ∑ coordinate : shape.Coord, |error coordinate * seed coordinate| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ coordinate : shape.Coord, |seed coordinate| * radius coordinate := by
      apply Finset.sum_le_sum
      intro coordinate _
      rw [abs_mul, mul_comm]
      exact mul_le_mul_of_nonneg_left (within coordinate) (abs_nonneg _)

/-- Each supplied tensor bounds the remainder of its corresponding graph node.

This is a local certificate condition: no bound on the complete output discrepancy is assumed.
The bound tensors may be constructed from the reference execution and input-region information;
this proposition states the proof obligation for those constructions.
-/
def Within {data : Type} {inputs : List Shape} {outputs : List Shape}
    (graph : Graph (α := ℝ) data inputs outputs)
    (values reference : TorchLean.TensorPack ℝ inputs) (parameters : data)
    (radii : TorchLean.TensorPack ℝ outputs) : Prop :=
  match graph with
  | .nil => True
  | .snoc previous node =>
      let split := TorchLean.TensorPack.unsnoc radii
      Within previous values reference parameters split.1 ∧
        ∀ coordinate,
          |nodeRemainder node (previous.eval values parameters)
            (previous.eval reference parameters) parameters coordinate| ≤ split.2 coordinate

/-- Propagate supplied local radii using only the reference execution and its existing VJPs.

Unlike `remainder`, this expression never evaluates the full input. It adds the local radii
weighted by absolute cotangents, preserving the reverse accumulation at shared nodes.
-/
noncomputable def bound {data : Type} {inputs : List Shape} {outputs : List Shape}
    (graph : Graph (α := ℝ) data inputs outputs)
    (reference : TorchLean.TensorPack ℝ inputs) (parameters : data)
    (seed : TorchLean.TensorPack ℝ (inputs ++ outputs))
    (radii : TorchLean.TensorPack ℝ outputs) : ℝ :=
  match graph with
  | .nil => 0
  | .snoc (ss := previousShapes) (τ := output) previous node =>
      let previousSeed := TorchLean.TensorPack.cast
        (List.append_assoc inputs previousShapes [output]).symm seed
      let splitSeed := TorchLean.TensorPack.unsnoc previousSeed
      let splitRadius := TorchLean.TensorPack.unsnoc radii
      let referenceContext := previous.eval reference parameters
      let contribution := node.vjp referenceContext parameters splitSeed.2
      bound previous reference parameters
          (TorchLean.TensorPack.add splitSeed.1 contribution) splitRadius.1 +
        dot (Tensor.map abs splitSeed.2) splitRadius.2

/-- Local coordinatewise remainder certificates imply a bound on the complete graph remainder.
The right-hand side uses no values from the full execution. -/
theorem remainder_le_bound {data : Type} {inputs : List Shape} {outputs : List Shape}
    (graph : Graph (α := ℝ) data inputs outputs) :
    ∀ (values reference : TorchLean.TensorPack ℝ inputs) (parameters : data)
      (seed : TorchLean.TensorPack ℝ (inputs ++ outputs))
      (radii : TorchLean.TensorPack ℝ outputs),
      Within graph values reference parameters radii →
        |remainder graph values reference parameters seed| ≤
          bound graph reference parameters seed radii := by
  induction graph with
  | nil =>
      intro values reference parameters seed radii _
      simp only [remainder, bound, abs_zero, le_refl]
  | snoc previous node ih =>
      intro values reference parameters seed radii within
      rename_i previousShapes output
      let previousSeed := TorchLean.TensorPack.cast
        (List.append_assoc inputs previousShapes [output]).symm seed
      let splitSeed := TorchLean.TensorPack.unsnoc previousSeed
      let splitRadius := TorchLean.TensorPack.unsnoc radii
      let contribution := node.vjp (previous.eval reference parameters) parameters splitSeed.2
      let accumulated := TorchLean.TensorPack.add splitSeed.1 contribution
      change Within previous values reference parameters splitRadius.1 ∧ _ at within
      obtain ⟨previousWithin, nodeWithin⟩ := within
      change |remainder previous values reference parameters accumulated +
          dot (nodeRemainder node (previous.eval values parameters)
            (previous.eval reference parameters) parameters) splitSeed.2| ≤
        bound previous reference parameters accumulated splitRadius.1 +
          dot (Tensor.map abs splitSeed.2) splitRadius.2
      exact (abs_add_le _ _).trans
        (add_le_add (ih values reference parameters accumulated splitRadius.1 previousWithin)
          (dot_abs_le _ _ _ nodeWithin))

end HeuristicEstimation.Composition
