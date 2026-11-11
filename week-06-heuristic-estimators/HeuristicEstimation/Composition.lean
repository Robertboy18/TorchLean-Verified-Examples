/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
Authors: Robert Joseph George
-/

module

public import TorchLean.NN.Tape.Graph
public import TorchLean.Tensor.Reduce.Real
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Ring

/-!
# Output discrepancies weighted by the existing reverse pass

Two executions of the same TorchLean graph are compared node by node. At each node, the change
in output minus the node's JVP at the reference execution is a local remainder. Each remainder
reaches the final answer through the cotangent of TorchLean's reverse pass.

`eval_difference` is an identity and needs no smoothness. It uses only each node's JVP/VJP
adjoint law, so it covers piecewise-linear nodes with a fixed convention at their kinks.
`remainder_le_bound` bounds the total remainder from local certificates, and `Averaging` cancels
the input term in expectation.
-/

@[expose] public section

namespace HeuristicEstimation.Composition

open TorchLean Tape
open Tensor.Rep (dot)
open scoped BigOperators

variable {α : Type} [Storage α]

/-- Subtract corresponding tensors in two contexts of the same shapes. -/
abbrev subtract [Sub α] {shapes : List Shape} (left right : TensorPack α shapes) :
    TensorPack α shapes :=
  Pack.zipWith (· - ·) left right

variable [CommRing α]

/-- Subtracting tensors subtracts their pairings with a fixed output cotangent. -/
theorem dot_sub_left {shape : Shape} (left right seed : Tensor α shape) :
    dot (left - right) seed = dot left seed - dot right seed := by
  have cancel : left - right + right = left := sub_add_cancel _ _
  have paired := congrArg (fun value => dot value seed) cancel
  rw [Tensor.dot_add_left] at paired
  exact eq_sub_of_add_eq paired

/-- For a typed context of tensors too, the pairing of a difference is the difference of the
pairings. -/
theorem dotList_sub_left {shapes : List Shape}
    (left right seed : TorchLean.TensorPack α shapes) :
    TensorPack.dotList (subtract left right) seed =
      TensorPack.dotList left seed - TensorPack.dotList right seed := by
  induction shapes with
  | nil =>
      cases left; cases right; cases seed
      simp [subtract, TensorPack.dotList]
  | cons shape shapes ih =>
      cases left with
      | cons first rest =>
          cases right with
          | cons other remaining =>
              cases seed with
              | cons output outputs =>
                  change dot (first - other) output +
                    TensorPack.dotList (subtract rest remaining) outputs =
                      (dot first output + TensorPack.dotList rest outputs) -
                        (dot other output + TensorPack.dotList remaining outputs)
                  rw [dot_sub_left, ih]
                  ring

private theorem sub_cast {before after : List Shape} (same : before = after)
    (left right : TorchLean.TensorPack α before) :
    subtract (Pack.cast same left)
        (Pack.cast same right) =
      Pack.cast same (subtract left right) := by
  cases same
  rfl

/-- Part of a node's output change that its JVP at the reference inputs does not explain. -/
def nodeRemainder {data : Type} {inputs : List Shape} {output : Shape}
    (node : NodeData α data inputs output)
    (values reference : TorchLean.TensorPack α inputs) (parameters : data) : Tensor α output :=
  node.forward values parameters - node.forward reference parameters -
    node.jvp reference (subtract values reference) parameters

/-- Paired with `seed`, a node's output change is its input change paired with the VJP of
`seed`, plus the paired remainder. -/
theorem node_difference {data : Type} {inputs : List Shape} {output : Shape}
    (node : NodeData α data inputs output) (correct : node.Correct)
    (values reference : TorchLean.TensorPack α inputs) (parameters : data)
    (seed : Tensor α output) :
    dot (node.forward values parameters - node.forward reference parameters)
        seed =
      TensorPack.dotList (subtract values reference)
          (node.vjp reference parameters seed) +
        dot (nodeRemainder node values reference parameters) seed := by
  simp only [nodeRemainder, dot_sub_left, correct reference (subtract values reference)
    parameters seed]
  ring

/-- Sum of node remainders, each paired with its cotangent from the reference reverse pass.

The recursion follows `GraphData.backpropCtx`, with the same saved reference values, VJPs and context
addition.
-/
def remainder {data : Type} {inputs : List Shape} {outputs : List Shape}
    (graph : GraphData α data inputs outputs)
    (values reference : TorchLean.TensorPack α inputs) (parameters : data)
    (seed : TorchLean.TensorPack α (inputs ++ outputs)) : α :=
  match graph with
  | .nil => 0
  | .snoc (ss := previousShapes) (τ := output) previous node =>
      let previousSeed := Pack.cast
        (List.append_assoc inputs previousShapes [output]).symm seed
      let split := Pack.unsnoc previousSeed
      let referenceContext := previous.eval reference parameters
      let contribution := node.vjp referenceContext parameters split.2
      remainder previous values reference parameters
          (TorchLean.TensorPack.add split.1 contribution) +
        dot (nodeRemainder node (previous.eval values parameters) referenceContext parameters)
          split.2

/-- A graph's paired output change is its input change paired with the reverse-pass cotangent,
plus `remainder`. The seed covers the whole typed context, so shared nodes and seeds on
intermediate values are included. -/
theorem eval_difference {data : Type} {inputs : List Shape} {outputs : List Shape}
    (graph : GraphData α data inputs outputs) (correct : graph.Correct) :
    ∀ (values reference : TorchLean.TensorPack α inputs) (parameters : data)
      (seed : TorchLean.TensorPack α (inputs ++ outputs)),
      TensorPack.dotList
          (subtract (graph.eval values parameters)
            (graph.eval reference parameters))
          seed =
        TensorPack.dotList (subtract values reference)
            (graph.backpropCtx reference parameters seed) +
          remainder graph values reference parameters seed := by
  induction graph with
  | nil =>
      intro values reference parameters seed
      simp only [GraphData.eval, GraphData.backpropCtx, remainder, sub_cast,
        TensorPack.dotList_cast_left, add_zero]
  | snoc previous node ih =>
      intro values reference parameters seed
      rename_i previousShapes output
      obtain ⟨previousCorrect, nodeCorrect⟩ := correct
      let fullContext := previous.eval values parameters
      let referenceContext := previous.eval reference parameters
      let same : (inputs ++ previousShapes) ++ [output] =
          inputs ++ (previousShapes ++ [output]) := List.append_assoc inputs previousShapes [output]
      let previousSeed := Pack.cast same.symm seed
      let split := Pack.unsnoc previousSeed
      let contribution := node.vjp referenceContext parameters split.2
      let accumulated := TorchLean.TensorPack.add split.1 contribution
      have seed_split : Pack.snoc split.1 split.2 = previousSeed :=
        Pack.snoc_unsnoc previousSeed
      have output_split :
          TensorPack.dotList
              (subtract ((GraphData.snoc previous node).eval values parameters)
                ((GraphData.snoc previous node).eval reference parameters)) seed =
            TensorPack.dotList (subtract fullContext referenceContext) split.1 +
              dot (node.forward fullContext parameters -
                node.forward referenceContext parameters) split.2 := by
        change TensorPack.dotList
            (subtract
              (Pack.cast same
                (Pack.snoc fullContext (node.forward fullContext parameters)))
              (Pack.cast same
                (Pack.snoc referenceContext
                  (node.forward referenceContext parameters)))) seed = _
        rw [sub_cast, TensorPack.dotList_cast_left, dotList_sub_left]
        change TensorPack.dotList
              (Pack.snoc fullContext (node.forward fullContext parameters))
              previousSeed -
            TensorPack.dotList
              (Pack.snoc referenceContext
                (node.forward referenceContext parameters))
              previousSeed = _
        rw [← seed_split, TensorPack.dotList_snoc, TensorPack.dotList_snoc,
          dotList_sub_left, dot_sub_left]
        ring
      calc
        _ = TensorPack.dotList (subtract fullContext referenceContext) split.1 +
            dot (node.forward fullContext parameters -
              node.forward referenceContext parameters) split.2 := output_split
        _ = TensorPack.dotList (subtract fullContext referenceContext) accumulated +
            dot (nodeRemainder node fullContext referenceContext parameters) split.2 := by
              rw [node_difference node nodeCorrect, TensorPack.dotList_add_right]
              ring
        _ = TensorPack.dotList (subtract values reference)
              (previous.backpropCtx reference parameters accumulated) +
            (remainder previous values reference parameters accumulated +
              dot (nodeRemainder node fullContext referenceContext parameters) split.2) := by
              rw [ih previousCorrect]
              ring
        _ = _ := rfl

/-! ## Bounds from local tensor certificates -/

/-- Coordinatewise bounds on `error` bound its pairing with any cotangent `seed` by the pairing
of `|seed|` with `radius`. -/
theorem dot_abs_le {shape : Shape} (error seed radius : Tensor ℝ shape)
    (within : ∀ coordinate, |error coordinate| ≤ radius coordinate) :
    |dot error seed| ≤ dot (Tensor.map abs seed) radius := by
  classical
  simp only [Tensor.Rep.dot, Tensor.map_apply]
  calc
    |∑ coordinate : shape.Coord, error coordinate * seed coordinate| ≤
        ∑ coordinate : shape.Coord, |error coordinate * seed coordinate| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ coordinate : shape.Coord, |seed coordinate| * radius coordinate := by
      apply Finset.sum_le_sum
      intro coordinate _
      rw [abs_mul, mul_comm]
      exact mul_le_mul_of_nonneg_left (within coordinate) (abs_nonneg _)

/-- Each radius tensor bounds the remainder of its node, coordinate by coordinate.

This is the local proof obligation for whatever computes the radii, typically from the reference
execution and the input region.
-/
def Within {data : Type} {inputs : List Shape} {outputs : List Shape}
    (graph : GraphData ℝ data inputs outputs)
    (values reference : TorchLean.TensorPack ℝ inputs) (parameters : data)
    (radii : TorchLean.TensorPack ℝ outputs) : Prop :=
  match graph with
  | .nil => True
  | .snoc previous node =>
      let split := Pack.unsnoc radii
      Within previous values reference parameters split.1 ∧
        ∀ coordinate,
          |nodeRemainder node (previous.eval values parameters)
            (previous.eval reference parameters) parameters coordinate| ≤ split.2 coordinate

/-- Budget for `remainder`: local radii paired with absolute cotangents from the reference
reverse pass.

Only the reference execution is evaluated. Cotangents still accumulate at shared nodes as they
do in the reverse pass.
-/
noncomputable def bound {data : Type} {inputs : List Shape} {outputs : List Shape}
    (graph : GraphData ℝ data inputs outputs)
    (reference : TorchLean.TensorPack ℝ inputs) (parameters : data)
    (seed : TorchLean.TensorPack ℝ (inputs ++ outputs))
    (radii : TorchLean.TensorPack ℝ outputs) : ℝ :=
  match graph with
  | .nil => 0
  | .snoc (ss := previousShapes) (τ := output) previous node =>
      let previousSeed := Pack.cast
        (List.append_assoc inputs previousShapes [output]).symm seed
      let splitSeed := Pack.unsnoc previousSeed
      let splitRadius := Pack.unsnoc radii
      let referenceContext := previous.eval reference parameters
      let contribution := node.vjp referenceContext parameters splitSeed.2
      bound previous reference parameters
          (TorchLean.TensorPack.add splitSeed.1 contribution) splitRadius.1 +
        dot (Tensor.map abs splitSeed.2) splitRadius.2

/-- Local remainder certificates (`Within`) bound the whole graph remainder by `bound`, which
uses only the reference execution. -/
theorem remainder_le_bound {data : Type} {inputs : List Shape} {outputs : List Shape}
    (graph : GraphData ℝ data inputs outputs) :
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
      let previousSeed := Pack.cast
        (List.append_assoc inputs previousShapes [output]).symm seed
      let splitSeed := Pack.unsnoc previousSeed
      let splitRadius := Pack.unsnoc radii
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
