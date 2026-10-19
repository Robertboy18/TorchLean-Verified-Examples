/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import HeuristicEstimation.Basic
public import Mathlib.Algebra.Order.BigOperators.Ring.Finset

/-!
# Downstream-aware finite sketches

Each row of `tails` is a later linear observation of the intermediate vector, and `weights`
gives each row's context weight. Their weighted Gram matrix is computed once and reused for every
residual. `check` tests absolute row sums in exact rationals. Its soundness theorems turn one
passing check into a quadratic-error bound for all residuals at once.
-/

@[expose] public section

namespace HeuristicEstimation.Sketch

open TorchLean
open scoped BigOperators

/-- Weighted second-moment matrix of the downstream observation vectors. -/
def gram {α : Type} [Storage α] [CommSemiring α] {contexts d : Nat}
    (weights : Tensor α [contexts]) (tails : Tensor α [contexts, d]) : Tensor α [d, d] :=
  Tensor.matrix fun i j => ∑ c : Fin contexts,
    weights.getScalar c * tails[(c, i)] * tails[(c, j)]

/-- Weighted squared error of a residual, measured through the downstream observations. -/
def error {α : Type} [Storage α] [CommSemiring α] {contexts d : Nat}
    (weights : Tensor α [contexts]) (tails : Tensor α [contexts, d])
    (residual : Tensor α [d]) : α :=
  ∑ c : Fin contexts, weights.getScalar c *
    (∑ i : Fin d, tails[(c, i)] * residual.getScalar i) ^ 2

/-- Quadratic form `rᵀ M r` of a square tensor matrix. -/
def quadratic {α : Type} [Storage α] [CommSemiring α] {d : Nat}
    (matrix : Tensor α [d, d]) (residual : Tensor α [d]) : α :=
  ∑ i : Fin d, ∑ j : Fin d,
    matrix[(i, j)] * residual.getScalar i * residual.getScalar j

/-- Entry `(i, j)` of `gram` is the weighted sum of `tails[(c, i)] * tails[(c, j)]`. -/
theorem gram_entry {α : Type} [Storage α] [CommSemiring α] {contexts d : Nat}
    (weights : Tensor α [contexts]) (tails : Tensor α [contexts, d]) (i j : Fin d) :
    (gram weights tails)[(i, j)] = ∑ c : Fin contexts,
      weights.getScalar c * tails[(c, i)] * tails[(c, j)] := by
  exact matrix_entry _ i j

/-- Downstream error equals the Gram matrix's quadratic form at the residual. -/
theorem error_eq_quadratic {α : Type} [Storage α] [CommSemiring α] {contexts d : Nat}
    (weights : Tensor α [contexts]) (tails : Tensor α [contexts, d])
    (residual : Tensor α [d]) :
    error weights tails residual = quadratic (gram weights tails) residual := by
  classical
  simp only [error, quadratic, gram_entry, pow_two, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  apply Finset.sum_congr rfl
  intro c _
  ring

/-- Gram matrices are symmetric, whatever the signs of the weights. -/
theorem gram_symmetric {α : Type} [Storage α] [CommSemiring α] {contexts d : Nat}
    (weights : Tensor α [contexts]) (tails : Tensor α [contexts, d]) (i j : Fin d) :
    (gram weights tails)[(i, j)] = (gram weights tails)[(j, i)] := by
  simp only [gram_entry]
  apply Finset.sum_congr rfl
  intro c _
  ring

/-- Nonnegative weights make the Gram quadratic form nonnegative for every residual. -/
theorem gram_nonneg {α : Type} [Storage α] [CommRing α] [LinearOrder α]
    [IsStrictOrderedRing α] {contexts d : Nat}
    (weights : Tensor α [contexts]) (tails : Tensor α [contexts, d])
    (hw : ∀ c, 0 ≤ weights.getScalar c) (residual : Tensor α [d]) :
    0 ≤ quadratic (gram weights tails) residual := by
  rw [← error_eq_quadratic]
  exact Finset.sum_nonneg fun c _ => mul_nonneg (hw c) (sq_nonneg _)

/-- Squared Euclidean norm of a vector. -/
def energy {α : Type} [Storage α] [CommSemiring α] {d : Nat}
    (residual : Tensor α [d]) : α :=
  ∑ i : Fin d, residual.getScalar i ^ 2

/-- What a coordinate sketch leaves out, `g - g̃`. Retained coordinates (`true`) become zero
and discarded coordinates keep their values. -/
def residual {α : Type} [Storage α] [Zero α] {d : Nat}
    (keep : Tensor Bool [d]) (values : Tensor α [d]) : Tensor α [d] :=
  Tensor.ofFn fun i => if keep.getScalar i then 0 else values.getScalar i

/-- Zero the rows and columns of retained coordinates, keeping the block on discarded ones.
Its quadratic form measures what a coordinate sketch leaves out (`quadratic_restrict`). -/
def restrict {α : Type} [Storage α] [Zero α] {d : Nat}
    (keep : Tensor Bool [d]) (matrix : Tensor α [d, d]) : Tensor α [d, d] :=
  Tensor.matrix fun i j =>
    if keep.getScalar i || keep.getScalar j then 0 else matrix[(i, j)]

/-- Zeroing retained coordinates in the matrix is equivalent to zeroing them in the vector. -/
theorem quadratic_restrict {α : Type} [Storage α] [CommSemiring α] {d : Nat}
    (keep : Tensor Bool [d]) (matrix : Tensor α [d, d]) (values : Tensor α [d]) :
    quadratic (restrict keep matrix) values = quadratic matrix (residual keep values) := by
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  simp only [restrict, matrix_entry, residual, Tensor.getScalar_ofFn]
  cases keep.getScalar i <;> cases keep.getScalar j <;> simp

/-- Retaining more coordinates never increases residual energy. That controls the certified
budget, but actual downstream error can still grow, because its terms can cancel. -/
theorem residual_energy_mono {α : Type} [Storage α] [CommRing α] [LinearOrder α]
    [IsStrictOrderedRing α] {d : Nat} (coarse fine : Tensor Bool [d])
    (hkeep : ∀ i, coarse.getScalar i = true → fine.getScalar i = true)
    (values : Tensor α [d]) : energy (residual fine values) ≤ energy (residual coarse values) := by
  apply Finset.sum_le_sum
  intro i _
  simp only [residual, Tensor.getScalar_ofFn]
  cases hc : coarse.getScalar i
  · cases fine.getScalar i <;> simp [sq_nonneg]
  · simp [hkeep i hc]

/-- If a symmetric matrix has absolute row sums at most `bound`, its quadratic form is at most
`bound * energy` at every residual. No eigensolver is trusted, at the price of a possibly loose
bound. -/
theorem quadratic_le {α : Type} [Storage α] [CommRing α] [LinearOrder α]
    [IsStrictOrderedRing α] {d : Nat} (matrix : Tensor α [d, d]) (bound : α)
    (hsym : ∀ i j : Fin d, matrix[(i, j)] = matrix[(j, i)])
    (hrow : ∀ i : Fin d, ∑ j : Fin d, |matrix[(i, j)]| ≤ bound)
    (residual : Tensor α [d]) :
    quadratic matrix residual ≤ bound * energy residual := by
  classical
  have hentry (i j : Fin d) :
      2 * (matrix[(i, j)] * residual.getScalar i * residual.getScalar j) ≤
        |matrix[(i, j)]| * residual.getScalar i ^ 2 +
          |matrix[(i, j)]| * residual.getScalar j ^ 2 := by
    have hab := le_abs_self (matrix[(i, j)] * residual.getScalar i * residual.getScalar j)
    rw [abs_mul, abs_mul] at hab
    have hsq := mul_le_mul_of_nonneg_left
      (two_mul_le_add_sq |residual.getScalar i| |residual.getScalar j|)
      (abs_nonneg matrix[(i, j)])
    simp only [sq_abs] at hsq
    nlinarith
  have hswap :
      (∑ i : Fin d, ∑ j : Fin d, |matrix[(i, j)]| * residual.getScalar j ^ 2) =
        ∑ i : Fin d, (∑ j : Fin d, |matrix[(i, j)]|) * residual.getScalar i ^ 2 := by
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro i _
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro j _
    rw [hsym j i]
  have hsum : (∑ i : Fin d, (∑ j : Fin d, |matrix[(i, j)]|) *
      residual.getScalar i ^ 2) ≤ bound * energy residual := by
    rw [energy, Finset.mul_sum]
    exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hrow i) (sq_nonneg _)
  have htotal : 2 * quadratic matrix residual ≤
      2 * (∑ i : Fin d, (∑ j : Fin d, |matrix[(i, j)]|) *
        residual.getScalar i ^ 2) := by
    calc
      _ ≤ ∑ i : Fin d, ∑ j : Fin d,
          (|matrix[(i, j)]| * residual.getScalar i ^ 2 +
            |matrix[(i, j)]| * residual.getScalar j ^ 2) := by
        simp only [quadratic, Finset.mul_sum]
        exact Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => hentry i j
      _ = _ := by
        simp_rw [Finset.sum_add_distrib]
        rw [hswap]
        simp only [← Finset.sum_mul, two_mul]
  linarith

/-- Decide in exact rationals whether every absolute row sum is at most `bound`. Off-diagonal
entries count, and a negative bound fails unless the matrix is empty. -/
def check {d : Nat} (matrix : Tensor Rat [d, d]) (bound : Rat) : Bool :=
  decide (∀ i : Fin d, ∑ j : Fin d, |matrix[(i, j)]| ≤ bound)

/-- `check` passes if and only if every absolute row sum is at most `bound`. -/
theorem check_iff {d : Nat} (matrix : Tensor Rat [d, d]) (bound : Rat) :
    check matrix bound = true ↔ ∀ i : Fin d, ∑ j : Fin d, |matrix[(i, j)]| ≤ bound := by
  simp [check]

/-- For a symmetric matrix, a passing check bounds the quadratic form at every rational
residual. Symmetry is necessary: `[[1, 0], [1, 0]]` passes with bound `1`, yet its form at
`(1, 1/2)` is `3/2`, above `5/4`. -/
theorem check_sound {d : Nat} (matrix : Tensor Rat [d, d]) (bound : Rat)
    (hsym : ∀ i j : Fin d, matrix[(i, j)] = matrix[(j, i)])
    (hcheck : check matrix bound = true) (residual : Tensor Rat [d]) :
    quadratic matrix residual ≤ bound * energy residual :=
  quadratic_le matrix bound hsym ((check_iff matrix bound).mp hcheck) residual

/-- For a symmetric rational matrix, a passing check also bounds its real interpretation at every
real residual, rational or not. -/
theorem check_sound_real {d : Nat} (matrix : Tensor Rat [d, d]) (bound : Rat)
    (hsym : ∀ i j : Fin d, matrix[(i, j)] = matrix[(j, i)])
    (hcheck : check matrix bound = true) (residual : Tensor ℝ [d]) :
    quadratic (Tensor.matrix fun i j => ((matrix[(i, j)] : Rat) : ℝ)) residual ≤
      (bound : ℝ) * energy residual := by
  apply quadratic_le
  · intro i j
    simp only [matrix_entry, hsym i j]
  · intro i
    simp only [matrix_entry]
    exact_mod_cast ((check_iff matrix bound).mp hcheck i)

/-- A passing check on the Gram matrix bounds downstream error at every residual. Weights need
not sum to one. When they do, `error` is a mean squared error. -/
theorem error_le_of_check {contexts d : Nat} (weights : Tensor Rat [contexts])
    (tails : Tensor Rat [contexts, d]) (bound : Rat)
    (hcheck : check (gram weights tails) bound = true) (residual : Tensor Rat [d]) :
    error weights tails residual ≤ bound * energy residual := by
  rw [error_eq_quadratic]
  exact check_sound _ bound (gram_symmetric weights tails) hcheck residual

/-- A passing check on the restricted Gram matrix bounds a coordinate sketch's downstream error
at every input. With bound `0` the error is at most zero for all inputs, hence zero when the
weights are nonnegative. -/
theorem masked_error_le_of_check {contexts d : Nat} (weights : Tensor Rat [contexts])
    (tails : Tensor Rat [contexts, d]) (keep : Tensor Bool [d]) (bound : Rat)
    (hcheck : check (restrict keep (gram weights tails)) bound = true)
    (values : Tensor Rat [d]) :
    error weights tails (residual keep values) ≤ bound * energy (residual keep values) := by
  have hid : residual keep (residual keep values) = residual keep values := by
    apply Tensor.ext_vector
    intro i
    simp only [residual, Tensor.getScalar_ofFn]
    cases keep.getScalar i <;> simp
  rw [error_eq_quadratic, ← hid, ← quadratic_restrict, hid]
  apply check_sound _ bound _ hcheck
  intro i j
  simp only [restrict, matrix_entry]
  rw [Bool.or_comm, gram_symmetric weights tails i j]

end HeuristicEstimation.Sketch
