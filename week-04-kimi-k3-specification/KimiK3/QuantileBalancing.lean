/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.FeedForward

/-!
# Exact coordinate minimization for Quantile Balancing

Appendix C derives the QB update by minimizing a one-dimensional hinge objective. This file proves
that claim directly for a finite batch. If exactly `targetLoad` margins strictly exceed a threshold,
then the threshold is a global minimizer of the coordinate objective. The hinge argument permits
equal margins, but the exact-count hypothesis need not have a witness for a tied batch. This file
does not construct the threshold or cover nonintegral target loads.
-/

@[expose] public section

namespace KimiK3
namespace QuantileBalancing

/-- Number of batch elements whose margin lies strictly above a threshold. -/
noncomputable def exceedanceCount {tokens : Nat} (margins : Fin tokens → ℝ)
    (threshold : ℝ) : Nat :=
  (Finset.univ.filter fun token => threshold < margins token).card

/-- An exact target quantile has precisely `targetLoad` strict exceedances.

Values equal to the threshold are deliberately excluded, matching the routing convention in
Appendix C.
-/
def IsExactTargetQuantile {tokens : Nat} (targetLoad : Nat)
    (margins : Fin tokens → ℝ) (threshold : ℝ) : Prop :=
  exceedanceCount margins threshold = targetLoad

/-- Number of tokens routed to an expert after applying a candidate bias. -/
noncomputable def routedTokenCount {tokens : Nat}
    (rawScore cutoff : Fin tokens → ℝ) (bias : ℝ) : Nat :=
  (Finset.univ.filter fun token => cutoff token < rawScore token + bias).card

/-- Negating an exact margin quantile gives the expert exactly its target load. -/
theorem neg_quantile_bias_hits_target {tokens targetLoad : Nat}
    (rawScore cutoff : Fin tokens → ℝ) (threshold : ℝ)
    (hQuantile : IsExactTargetQuantile targetLoad
      (fun token => rawScore token - cutoff token) threshold) :
    routedTokenCount rawScore cutoff (-threshold) = targetLoad := by
  rw [← hQuantile]
  apply congrArg Finset.card
  ext token
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  constructor <;> intro h <;> linarith

/-- Subtract the mean expert bias, as prescribed by Eq. 14. -/
noncomputable def centerBias {experts : Nat} (bias : Fin experts → ℝ) : Fin experts → ℝ :=
  fun expert => bias expert - (∑ index, bias index) / experts

/-- Centering gives zero total bias whenever the expert set is nonempty. -/
theorem sum_centerBias_eq_zero {experts : Nat} (hExperts : 0 < experts)
    (bias : Fin experts → ℝ) :
    ∑ expert, centerBias bias expert = 0 := by
  simp only [centerBias, Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul,
    Finset.card_univ, Fintype.card_fin]
  have hExpertsReal : (experts : ℝ) ≠ 0 := by positivity
  field_simp
  ring

/-- Centering preserves every adjusted-score comparison because it subtracts a common offset. -/
theorem centerBias_preserves_pairwise_order {experts : Nat} (raw bias : Fin experts → ℝ)
    (left right : Fin experts) :
    raw left + centerBias bias left ≥ raw right + centerBias bias right ↔
      raw left + bias left ≥ raw right + bias right := by
  simp only [centerBias]
  constructor <;> intro h <;> linarith

/-- Consequently, centering leaves the set of valid top-k routes unchanged. -/
theorem centerBias_preserves_topK {experts active : Nat} (route : Route experts active)
    (raw bias : Fin experts → ℝ) :
    route.IsTopK (raw + centerBias bias) ↔ route.IsTopK (raw + bias) := by
  constructor <;> intro h selected candidate hCandidate
  · exact (centerBias_preserves_pairwise_order raw bias _ _).mp
      (h selected candidate hCandidate)
  · exact (centerBias_preserves_pairwise_order raw bias _ _).mpr
      (h selected candidate hCandidate)

/-- One coordinate of the convex QB dual objective. -/
noncomputable def coordinateObjective {tokens : ℕ} (targetLoad : ℕ)
    (margins : Fin tokens → ℝ) (threshold : ℝ) : ℝ :=
  targetLoad * threshold + ∑ token, max 0 (margins token - threshold)

private theorem hinge_le_hinge_of_le {margin candidate threshold : ℝ}
    (hCandidate : candidate ≤ threshold) :
    max 0 (margin - threshold) +
        (if threshold < margin then threshold - candidate else 0) ≤
      max 0 (margin - candidate) := by
  split_ifs with hMargin
  · rw [max_eq_right (by linarith), max_eq_right (by linarith)]
    linarith
  · rw [max_eq_left (by linarith)]
    simpa only [add_zero] using le_max_left 0 (margin - candidate)

private theorem hinge_le_hinge_of_ge {margin candidate threshold : ℝ}
    (_hCandidate : threshold ≤ candidate) :
    max 0 (margin - threshold) ≤
      max 0 (margin - candidate) +
        (if threshold < margin then candidate - threshold else 0) := by
  split_ifs with hMargin
  · rw [max_eq_right (by linarith)]
    have hDifference : margin - candidate ≤ max 0 (margin - candidate) :=
      le_max_right _ _
    linarith [_hCandidate]
  · rw [max_eq_left (by linarith)]
    exact add_nonneg (le_max_left _ _) (by linarith [_hCandidate])

private theorem sum_indicator_sub {tokens targetLoad : ℕ} (margins : Fin tokens → ℝ)
    (threshold candidate : ℝ)
    (hQuantile : IsExactTargetQuantile targetLoad margins threshold) :
    (∑ token, if threshold < margins token then threshold - candidate else 0) =
      targetLoad * (threshold - candidate) := by
  classical
  calc
    (∑ token, if threshold < margins token then threshold - candidate else 0) =
        ((Finset.univ.filter fun token => threshold < margins token).card : ℝ) *
          (threshold - candidate) := by
      rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
    _ = targetLoad * (threshold - candidate) := by
      rw [show (Finset.univ.filter fun token => threshold < margins token).card =
          targetLoad from hQuantile]

private theorem sum_indicator_add {tokens targetLoad : ℕ} (margins : Fin tokens → ℝ)
    (threshold candidate : ℝ)
    (hQuantile : IsExactTargetQuantile targetLoad margins threshold) :
    (∑ token, if threshold < margins token then candidate - threshold else 0) =
      targetLoad * (candidate - threshold) := by
  classical
  calc
    (∑ token, if threshold < margins token then candidate - threshold else 0) =
        ((Finset.univ.filter fun token => threshold < margins token).card : ℝ) *
          (candidate - threshold) := by
      rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
    _ = targetLoad * (candidate - threshold) := by
      rw [show (Finset.univ.filter fun token => threshold < margins token).card =
          targetLoad from hQuantile]

/-- An exact QB quantile is a global minimizer of its coordinate dual objective. -/
theorem coordinateObjective_minimized_at_exactQuantile {tokens targetLoad : ℕ}
    (margins : Fin tokens → ℝ) (threshold : ℝ)
    (hQuantile : IsExactTargetQuantile targetLoad margins threshold) (candidate : ℝ) :
    coordinateObjective targetLoad margins threshold ≤
      coordinateObjective targetLoad margins candidate := by
  rcases le_total candidate threshold with hCandidate | hCandidate
  · have hSum :
        (∑ token : Fin tokens, (max 0 (margins token - threshold) +
          (if threshold < margins token then threshold - candidate else 0))) ≤
          (∑ token : Fin tokens, max 0 (margins token - candidate)) :=
      Finset.sum_le_sum fun token _ =>
        hinge_le_hinge_of_le (margin := margins token) hCandidate
    rw [Finset.sum_add_distrib,
      sum_indicator_sub margins threshold candidate hQuantile] at hSum
    simp only [coordinateObjective]
    linarith
  · have hSum :
        (∑ token : Fin tokens, max 0 (margins token - threshold)) ≤
          (∑ token : Fin tokens, (max 0 (margins token - candidate) +
            (if threshold < margins token then candidate - threshold else 0))) :=
      Finset.sum_le_sum fun token _ =>
        hinge_le_hinge_of_ge (margin := margins token) hCandidate
    rw [Finset.sum_add_distrib,
      sum_indicator_add margins threshold candidate hQuantile] at hSum
    simp only [coordinateObjective]
    linarith

end QuantileBalancing
end KimiK3
