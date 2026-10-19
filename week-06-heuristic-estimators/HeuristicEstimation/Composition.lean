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

/-- For a typed context of tensors too, the pairing of a difference is the difference of the
pairings. -/
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

/-- Part of a node's output change that its JVP at the reference inputs does not explain. -/
def nodeRemainder {data : Type} {inputs : List Shape} {output : Shape}
    (node : Node (α := α) data inputs output)
    (values reference : TorchLean.TensorPack α inputs) (parameters : data) : Tensor α output :=
  Tensor.subSpec (Tensor.subSpec (node.forward values parameters)
    (node.forward reference parameters))
    (node.jvp reference (TorchLean.TensorPack.sub values reference) parameters)

/-- Paired with `seed`, a node's output change is its input change paired with the VJP of
`seed`, plus the paired remainder. -/
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

/-- Sum of node remainders, each paired with its cotangent from the reference reverse pass.

The recursion follows `Graph.backpropCtx`, with the same saved reference values, VJPs and context
addition.
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

/-- A graph's paired output change is its input change paired with the reverse-pass cotangent,
plus `remainder`. The seed covers the whole typed context, so shared nodes and seeds on
intermediate values are included. -/
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

/-- Coordinatewise bounds on `error` bound its pairing with any cotangent `seed` by the pairing
of `|seed|` with `radius`. -/
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

/-- Each radius tensor bounds the remainder of its node, coordinate by coordinate.

This is the local proof obligation for whatever computes the radii, typically from the reference
execution and the input region.
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

/-- Budget for `remainder`: local radii paired with absolute cotangents from the reference
reverse pass.

Only the reference execution is evaluated. Cotangents still accumulate at shared nodes as they
do in the reverse pass.
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

/-- Local remainder certificates (`Within`) bound the whole graph remainder by `bound`, which
uses only the reference execution. -/
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
