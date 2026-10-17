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

Rows of `tails` describe future linear observations, with nonnegative context weights.
Their weighted Gram matrix can be prepared once and reused for many residuals. The
row-sum checker certifies a uniform quadratic-error bound, rather than checking one vector.
All executable data uses TorchLean tensors; finite sums use mathlib's existing operations.
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

/-- Quadratic form of a tensor matrix. This uses the same coordinates as mathlib matrices. -/
def quadratic {α : Type} [Storage α] [CommSemiring α] {d : Nat}
    (matrix : Tensor α [d, d]) (residual : Tensor α [d]) : α :=
  ∑ i : Fin d, ∑ j : Fin d,
    matrix[(i, j)] * residual.getScalar i * residual.getScalar j

/-- Reading one Gram entry gives its defining weighted coordinate product. -/
theorem gram_entry {α : Type} [Storage α] [CommSemiring α] {contexts d : Nat}
    (weights : Tensor α [contexts]) (tails : Tensor α [contexts, d]) (i j : Fin d) :
    (gram weights tails)[(i, j)] = ∑ c : Fin contexts,
      weights.getScalar c * tails[(c, i)] * tails[(c, j)] := by
  exact matrix_entry _ i j

/-- Computing through the contexts or through their Gram matrix gives the same exact result. -/
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

/-- Gram matrices are symmetric even without a positivity premise on the weights. -/
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

/-- Squared Euclidean size of a tensor vector. -/
def energy {α : Type} [Storage α] [CommSemiring α] {d : Nat}
    (residual : Tensor α [d]) : α :=
  ∑ i : Fin d, residual.getScalar i ^ 2

/-- Discard coordinates outside a retained sketch; `true` marks a coordinate we keep. -/
def residual {α : Type} [Storage α] [Zero α] {d : Nat}
    (keep : Tensor Bool [d]) (values : Tensor α [d]) : Tensor α [d] :=
  Tensor.ofFn fun i => if keep.getScalar i then 0 else values.getScalar i

/-- Restrict a matrix to discarded coordinates. This is the matrix of a coordinate
projection's remaining error, not a new approximation to its entries. -/
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

/-- Retaining more coordinates never increases residual energy. This controls the certified
budget, not necessarily the actual downstream error, whose terms can cancel. -/
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

/-- A symmetric matrix with bounded absolute row sums bounds every residual's quadratic
error. This certificate avoids trusting an eigensolver, though it can be conservative. -/
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

/-- Check a uniform row-sum bound using exact rational arithmetic. A negative bound cannot
pass for a nonempty matrix, and all rows are checked, not only the diagonal. -/
def check {d : Nat} (matrix : Tensor Rat [d, d]) (bound : Rat) : Bool :=
  decide (∀ i : Fin d, ∑ j : Fin d, |matrix[(i, j)]| ≤ bound)

/-- A successful check supplies the row-sum premise for every row. -/
theorem check_iff {d : Nat} (matrix : Tensor Rat [d, d]) (bound : Rat) :
    check matrix bound = true ↔ ∀ i : Fin d, ∑ j : Fin d, |matrix[(i, j)]| ≤ bound := by
  simp [check]

/-- Once the exact row certificate passes, the bound holds for every rational residual,
including vectors that were not supplied when checking the certificate. -/
theorem check_sound {d : Nat} (matrix : Tensor Rat [d, d]) (bound : Rat)
    (hsym : ∀ i j : Fin d, matrix[(i, j)] = matrix[(j, i)])
    (hcheck : check matrix bound = true) (residual : Tensor Rat [d]) :
    quadratic matrix residual ≤ bound * energy residual :=
  quadratic_le matrix bound hsym ((check_iff matrix bound).mp hcheck) residual

/-- Exact rational checking also certifies the interpreted matrix on arbitrary real vectors;
the residual itself does not have to be representable as rational data. -/
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

/-- The checked Gram certificate controls downstream error for every residual. The context
weights need not sum to one; when they do, the error is a mean squared error. -/
theorem error_le_of_check {contexts d : Nat} (weights : Tensor Rat [contexts])
    (tails : Tensor Rat [contexts, d]) (bound : Rat)
    (hcheck : check (gram weights tails) bound = true) (residual : Tensor Rat [d]) :
    error weights tails residual ≤ bound * energy residual := by
  rw [error_eq_quadratic]
  exact check_sound _ bound (gram_symmetric weights tails) hcheck residual

/-- A proposed coordinate sketch is checked against the remaining Gram matrix. In
particular, a zero certificate proves zero error for all inputs, not only the example vector. -/
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
