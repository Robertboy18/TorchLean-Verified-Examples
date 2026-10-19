/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import HeuristicEstimation.Basic
public import HeuristicEstimation.Composition

/-!
# Averaging a graph's discarded input

`Composition.eval_difference` splits a graph's output change into a pulled-back input change
and local nonlinear remainders. If the discarded input is centered for every retained state, the
pulled-back term averages to zero, and local certificates bound what remains.

Each row of `law` is the conditional distribution of the discarded state, so it is nonnegative
and sums to one. Centering is required of the graph inputs, coordinate by coordinate, and
uncorrelated intermediate activations don't imply it. Output seeds may depend on the retained
state but not on the discarded one, which is what lets the linear term cancel. Remainders at
later nodes need not be centered.
-/

@[expose] public section

namespace HeuristicEstimation.Averaging

open TorchLean
open Spec (Shape)
open Proofs.Autograd.Algebra (Graph)
open scoped BigOperators

private theorem sum_dot_eq_zero {n : Nat} {shape : Shape}
    (weights : Fin n → ℝ) (values : Fin n → Tensor ℝ shape) (seed : Tensor ℝ shape)
    (hmean : ∀ coordinate, ∑ i : Fin n, weights i * values i coordinate = 0) :
    (∑ i : Fin n, weights i * Proofs.TensorAlgebra.dot (values i) seed) = 0 := by
  have hdot (i : Fin n) : Proofs.TensorAlgebra.dot (values i) seed =
      ∑ coordinate : shape.Coord, values i coordinate * seed coordinate := by
    rw [← Spec.dot_eq_tensorAlgebra_dot, Spec.dot, Spec.sum_spec_eq_coord_sum]
    simp only [Tensor.mulSpec, Tensor.map2Spec, Tensor.Internal.Rep.zipWith_apply]
  simp_rw [hdot, Finset.mul_sum, ← mul_assoc]
  rw [Finset.sum_comm]
  simp_rw [← Finset.sum_mul, hmean, zero_mul, Finset.sum_const_zero]

private theorem sum_dotList_eq_zero {n : Nat} {shapes : List Shape}
    (weights : Fin n → ℝ) (values : Fin n → TensorPack ℝ shapes)
    (seed : TensorPack ℝ shapes)
    (hmean : ∀ (position : Fin shapes.length) (coordinate : (shapes.get position).Coord),
      ∑ i : Fin n, weights i * (values i).get position coordinate = 0) :
    (∑ i : Fin n,
      weights i * Proofs.Autograd.Algebra.TensorPack.dotList (values i) seed) = 0 := by
  induction shapes with
  | nil =>
      have hnil (i : Fin n) :
          Proofs.Autograd.Algebra.TensorPack.dotList (values i) seed = 0 := by
        cases values i
        cases seed
        rfl
      simp only [hnil, mul_zero, Finset.sum_const_zero]
  | cons shape shapes ih =>
      cases seed with
      | cons first rest =>
          have hhead (coordinate : shape.Coord) :
              ∑ i : Fin n, weights i * (values i).head coordinate = 0 := by
            convert hmean ⟨0, by simp⟩ coordinate using 1
            apply Finset.sum_congr rfl
            intro i _
            cases values i
            rfl
          have htail (position : Fin shapes.length)
              (coordinate : (shapes.get position).Coord) :
              ∑ i : Fin n, weights i * (values i).tail.get position coordinate = 0 := by
            convert hmean position.succ coordinate using 1
            apply Finset.sum_congr rfl
            intro i _
            cases values i
            rfl
          have hsplit (i : Fin n) :
              Proofs.Autograd.Algebra.TensorPack.dotList (values i) (.cons first rest) =
                Proofs.TensorAlgebra.dot (values i).head first +
                  Proofs.Autograd.Algebra.TensorPack.dotList (values i).tail rest := by
            cases values i
            rfl
          simp_rw [hsplit, mul_add, Finset.sum_add_distrib]
          rw [sum_dot_eq_zero weights _ first hhead, ih _ rest htail, zero_add]

private theorem abs_weighted_sum_le {n : Nat} (weights values bounds : Fin n → ℝ)
    (hweights : ∀ i, 0 ≤ weights i) (hbound : ∀ i, |values i| ≤ bounds i) :
    |∑ i : Fin n, weights i * values i| ≤ ∑ i : Fin n, weights i * bounds i := by
  calc
    _ ≤ ∑ i : Fin n, |weights i * values i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ _ := Finset.sum_le_sum fun i _ => by
      rw [abs_mul, abs_of_nonneg (hweights i)]
      exact mul_le_mul_of_nonneg_left (hbound i) (hweights i)

/-- Under conditional centering, the weighted mean output change equals the weighted mean of
the graph remainders. The outer weights need not be normalized. -/
theorem mean_difference {retained discarded : Nat} {data : Type}
    {inputs outputs : List Shape} (graph : Graph (α := ℝ) data inputs outputs)
    (weights : Tensor ℝ [retained]) (law : Tensor ℝ [retained, discarded])
    (values : Fin retained → Fin discarded → TensorPack ℝ inputs)
    (reference : Fin retained → TensorPack ℝ inputs) (parameters : data)
    (seed : Fin retained → TensorPack ℝ (inputs ++ outputs))
    (hmean : ∀ (i : Fin retained) (position : Fin inputs.length)
      (coordinate : (inputs.get position).Coord),
      ∑ j : Fin discarded, law[(i, j)] *
        (TensorPack.sub (values i j) (reference i)).get position coordinate = 0) :
    (∑ i : Fin retained, weights.getScalar i * ∑ j : Fin discarded, law[(i, j)] *
      Proofs.Autograd.Algebra.TensorPack.dotList
        (TensorPack.sub (graph.eval (values i j) parameters)
          (graph.eval (reference i) parameters)) (seed i)) =
      ∑ i : Fin retained, weights.getScalar i * ∑ j : Fin discarded, law[(i, j)] *
        Composition.remainder graph (values i j) (reference i) parameters (seed i) := by
  apply Finset.sum_congr rfl
  intro i _
  congr 1
  simp_rw [Composition.eval_difference, mul_add, Finset.sum_add_distrib]
  rw [sum_dotList_eq_zero (fun j => law[(i, j)])
    (fun j => TensorPack.sub (values i j) (reference i))
    (graph.backpropCtx (reference i) parameters (seed i)) (hmean i), zero_add]

/-- Bounds on the graph remainder at every pair of states bound the expected output error. The
reference term is averaged over retained states only. With normalized outer weights, both sides
are expectations. -/
private theorem mean_error_le_of_bound {retained discarded : Nat} {data : Type}
    {inputs outputs : List Shape} (graph : Graph (α := ℝ) data inputs outputs)
    (weights : Tensor ℝ [retained]) (law budget : Tensor ℝ [retained, discarded])
    (values : Fin retained → Fin discarded → TensorPack ℝ inputs)
    (reference : Fin retained → TensorPack ℝ inputs) (parameters : data)
    (seed : Fin retained → TensorPack ℝ (inputs ++ outputs))
    (hweights : ∀ i : Fin retained, 0 ≤ weights.getScalar i)
    (hlaw : ∀ (i : Fin retained) (j : Fin discarded), 0 ≤ law[(i, j)])
    (hmass : ∀ i : Fin retained, ∑ j : Fin discarded, law[(i, j)] = 1)
    (hmean : ∀ (i : Fin retained) (position : Fin inputs.length)
      (coordinate : (inputs.get position).Coord),
      ∑ j : Fin discarded, law[(i, j)] *
        (TensorPack.sub (values i j) (reference i)).get position coordinate = 0)
    (hbound : ∀ (i : Fin retained) (j : Fin discarded),
      |Composition.remainder graph (values i j) (reference i) parameters (seed i)| ≤
        budget[(i, j)]) :
    |(∑ i : Fin retained, weights.getScalar i * ∑ j : Fin discarded, law[(i, j)] *
      Proofs.Autograd.Algebra.TensorPack.dotList (graph.eval (values i j) parameters) (seed i)) -
      (∑ i : Fin retained, weights.getScalar i *
        Proofs.Autograd.Algebra.TensorPack.dotList
          (graph.eval (reference i) parameters) (seed i))| ≤
      ∑ i : Fin retained, weights.getScalar i * ∑ j : Fin discarded,
        law[(i, j)] * budget[(i, j)] := by
  have hdifference :
      (∑ i : Fin retained, weights.getScalar i * ∑ j : Fin discarded, law[(i, j)] *
        Proofs.Autograd.Algebra.TensorPack.dotList (graph.eval (values i j) parameters) (seed i)) -
        (∑ i : Fin retained, weights.getScalar i *
          Proofs.Autograd.Algebra.TensorPack.dotList
            (graph.eval (reference i) parameters) (seed i)) =
      ∑ i : Fin retained, weights.getScalar i * ∑ j : Fin discarded, law[(i, j)] *
        Proofs.Autograd.Algebra.TensorPack.dotList
          (TensorPack.sub (graph.eval (values i j) parameters)
            (graph.eval (reference i) parameters)) (seed i) := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    simp_rw [Composition.dotList_sub_left, mul_sub, Finset.sum_sub_distrib]
    rw [← Finset.sum_mul, hmass i, one_mul, mul_sub]
  rw [hdifference, mean_difference graph weights law values reference parameters seed hmean]
  apply abs_weighted_sum_le _ _ _ hweights
  intro i
  exact abs_weighted_sum_le _ _ _ (hlaw i) (hbound i)

/-- Local remainder certificates bound the expected output error by a budget computed from
retained states alone. One radius pack per retained state must cover every discarded state in
its row. Producing those radii is a separate, operation-specific job. -/
theorem mean_error_le {retained discarded : Nat} {data : Type}
    {inputs outputs : List Shape} (graph : Graph (α := ℝ) data inputs outputs)
    (weights : Tensor ℝ [retained]) (law : Tensor ℝ [retained, discarded])
    (values : Fin retained → Fin discarded → TensorPack ℝ inputs)
    (reference : Fin retained → TensorPack ℝ inputs) (parameters : data)
    (seed : Fin retained → TensorPack ℝ (inputs ++ outputs))
    (radii : Fin retained → TensorPack ℝ outputs)
    (hweights : ∀ i : Fin retained, 0 ≤ weights.getScalar i)
    (hlaw : ∀ (i : Fin retained) (j : Fin discarded), 0 ≤ law[(i, j)])
    (hmass : ∀ i : Fin retained, ∑ j : Fin discarded, law[(i, j)] = 1)
    (hmean : ∀ (i : Fin retained) (position : Fin inputs.length)
      (coordinate : (inputs.get position).Coord),
      ∑ j : Fin discarded, law[(i, j)] *
        (TensorPack.sub (values i j) (reference i)).get position coordinate = 0)
    (hlocal : ∀ (i : Fin retained) (j : Fin discarded),
      Composition.Within graph (values i j) (reference i) parameters (radii i)) :
    |(∑ i : Fin retained, weights.getScalar i * ∑ j : Fin discarded, law[(i, j)] *
      Proofs.Autograd.Algebra.TensorPack.dotList (graph.eval (values i j) parameters) (seed i)) -
      (∑ i : Fin retained, weights.getScalar i *
        Proofs.Autograd.Algebra.TensorPack.dotList
          (graph.eval (reference i) parameters) (seed i))| ≤
      ∑ i : Fin retained, weights.getScalar i *
        Composition.bound graph (reference i) parameters (seed i) (radii i) := by
  let budget := Tensor.matrix fun i (_ : Fin discarded) =>
    Composition.bound graph (reference i) parameters (seed i) (radii i)
  have hbound (i : Fin retained) (j : Fin discarded) :
      |Composition.remainder graph (values i j) (reference i) parameters (seed i)| ≤
        budget[(i, j)] := by
    dsimp only [budget]
    rw [matrix_entry]
    exact Composition.remainder_le_bound graph _ _ _ _ _ (hlocal i j)
  have h := mean_error_le_of_bound graph weights law budget values reference parameters seed
    hweights hlaw hmass hmean hbound
  simpa only [budget, matrix_entry, ← Finset.sum_mul, hmass, one_mul] using h

end HeuristicEstimation.Averaging
