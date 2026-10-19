/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import HeuristicEstimation.Basic

/-!
# Attention scores from moments

An unscaled query–key score is the ordinary tensor dot product. Its expectation is the score
of the means plus the trace of the query–key cross-covariance, and that trace survives when both
means vanish. Coordinates, queries and keys may depend on each other arbitrarily.
-/

@[expose] public section

namespace HeuristicEstimation.Attention

open TorchLean MeasureTheory ProbabilityTheory
open scoped BigOperators

/-- Unscaled attention score `qᵀ k` of one query and key. -/
def score {α : Type} [Storage α] [Add α] [Mul α] [Zero α] {d : Nat}
    (query key : Tensor α [d]) : α :=
  Tensor.sum (query * key)

/-- Over ℝ, the score unfolds to a sum over coordinates. -/
theorem score_eq_sum {d : Nat} (query key : Tensor ℝ [d]) :
    score query key = ∑ i : Fin d, query.getScalar i * key.getScalar i := by
  change Tensor.sumSpec (Tensor.mulSpec query key) = _
  rw [Spec.sum_spec_vec]
  simp only [Spec.getScalar_mul_spec]

/-- Covariance-aware estimate of a score: the score of the means plus the trace of the supplied
cross-covariance. -/
def estimate {α : Type} [Storage α] [Add α] [Mul α] [Zero α] {d : Nat}
    (queryMean keyMean : Tensor α [d]) (crossCovariance : Tensor α [d, d]) : α :=
  score queryMean keyMean + Tensor.sum (Tensor.ofFn fun (i : Fin d) => crossCovariance[(i, i)])

/-- With the true means and cross-covariance, `estimate` is the expected score. Each coordinate
needs a finite second moment, and no Gaussian or independence assumption is made. -/
theorem score_mean {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {d : Nat} (query key : Ω → Tensor ℝ [d])
    (hq : ∀ i, MemLp (fun ω => (query ω).getScalar i) 2 μ)
    (hk : ∀ i, MemLp (fun ω => (key ω).getScalar i) 2 μ) :
    (∫ ω, score (query ω) (key ω) ∂μ) =
      estimate (Tensor.ofFn fun i => ∫ ω, (query ω).getScalar i ∂μ)
        (Tensor.ofFn fun i => ∫ ω, (key ω).getScalar i ∂μ)
        (Tensor.matrix fun i j => covariance
          (fun ω => (query ω).getScalar i) (fun ω => (key ω).getScalar j) μ) := by
  simp_rw [score_eq_sum]
  rw [integral_finsetSum Finset.univ
    (f := fun i ω => (query ω).getScalar i * (key ω).getScalar i)
    (fun i _ => (hq i).integrable_mul (hk i))]
  simp only [estimate, score_eq_sum, Tensor.getScalar_ofFn, Tensor.sum,
    Spec.sum_spec_vec]
  simp only [matrix_entry, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => product_mean (hq i) (hk i)

/-- Projecting query and key with a symmetric idempotent matrix `P` loses `qᵀ (k - P k)`, the
score against the discarded part of the key. -/
theorem score_projection_error {d : Nat} (projection : Tensor ℝ [d, d])
    (hsym : ∀ i j : Fin d, projection[(i, j)] = projection[(j, i)])
    (hid : Tensor.matmul projection projection = projection) (query key : Tensor ℝ [d]) :
    score query key - score (Tensor.matvec projection query) (Tensor.matvec projection key) =
      score query (key - Tensor.matvec projection key) := by
  have hadjoint (x y : Tensor ℝ [d]) :
      score (Tensor.matvec projection x) y = score x (Tensor.matvec projection y) := by
    simp only [score_eq_sum, Tensor.matvec, Spec.getScalar_mat_vec_mul_spec,
      Finset.sum_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro j _
    change projection[(j, i)] * x.getScalar i * y.getScalar j =
      x.getScalar i * (projection[(i, j)] * y.getScalar j)
    rw [hsym j i]
    ring
  have htwice : Tensor.matvec projection (Tensor.matvec projection key) =
      Tensor.matvec projection key := by
    change Spec.matVecMulSpec projection (Spec.matVecMulSpec projection key) = _
    rw [Spec.mat_vec_assoc]
    change Tensor.matvec (Tensor.matmul projection projection) key = _
    rw [hid]
  rw [hadjoint, htwice]
  change _ = score query (Tensor.subSpec key (Tensor.matvec projection key))
  have hsub (i : Fin d) :
      (Tensor.subSpec key (Tensor.matvec projection key)).getScalar i =
        key.getScalar i - (Tensor.matvec projection key).getScalar i := by
    simp [Tensor.getScalar_eq_apply, Tensor.subSpec, Tensor.map2Spec]
  simp only [score_eq_sum, hsub, mul_sub, Finset.sum_sub_distrib]

end HeuristicEstimation.Attention
