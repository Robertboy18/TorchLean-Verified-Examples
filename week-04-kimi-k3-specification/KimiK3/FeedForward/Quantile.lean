/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.FeedForward.Routing
public import Mathlib.Data.Finset.Max

/-!
# Exact coordinate minimization for Quantile Balancing

Appendix C derives the QB update by minimizing a one-dimensional hinge objective, and this file
proves that claim for a finite batch. If exactly `targetLoad` margins strictly exceed a threshold,
the threshold is a global minimizer of the coordinate objective. More generally, the target may lie
anywhere between the count strictly above the threshold and the count at or above it. That covers
ties and the upper endpoint of a minimizing interval, which the strict-count clause after
Appendix C, Eq. 24 misses (the interval between the k-th and (k+1)-th largest margins stated there
is right). We prove that this bracket characterizes every global minimizer, fractional targets
included. The strict routing count stays integral, and a minimizing threshold need not realize it.
The last section checks the endpoint example and the tie example.
-/

@[expose] public section

namespace KimiK3
namespace QuantileBalancing

/-- Number of batch elements whose margin lies strictly above a threshold. -/
noncomputable def exceedanceCount {tokens : Nat} (margins : Fin tokens → ℝ)
    (threshold : ℝ) : Nat :=
  (Finset.univ.filter fun token => threshold < margins token).card

/-- Number of batch elements at or above a threshold, including ties. -/
noncomputable def atLeastCount {tokens : Nat} (margins : Fin tokens → ℝ)
    (threshold : ℝ) : Nat :=
  (Finset.univ.filter fun token => threshold ≤ margins token).card

/-- `threshold` has exactly `targetLoad` strict exceedances.

Margins equal to the threshold do not count, matching the routing convention in Appendix C.
-/
def IsExactTargetQuantile {tokens : Nat} (targetLoad : Nat)
    (margins : Fin tokens → ℝ) (threshold : ℝ) : Prop :=
  exceedanceCount margins threshold = targetLoad

/-- Number of tokens routed to an expert after applying a candidate bias. -/
noncomputable def routedTokenCount {tokens : Nat}
    (rawScore cutoff : Fin tokens → ℝ) (bias : ℝ) : Nat :=
  (Finset.univ.filter fun token => cutoff token < rawScore token + bias).card

/-- Using the negated exact margin quantile as bias routes `targetLoad` tokens to the expert. -/
theorem routedTokenCount_neg_eq {tokens targetLoad : Nat}
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

/-- Centering subtracts a common offset, so it preserves every adjusted-score comparison. -/
theorem centerBias_score_ge_iff {experts : Nat} (raw bias : Fin experts → ℝ)
    (left right : Fin experts) :
    raw left + centerBias bias left ≥ raw right + centerBias bias right ↔
      raw left + bias left ≥ raw right + bias right := by
  simp only [centerBias]
  constructor <;> intro h <;> linarith

/-- Centering leaves the set of valid top-k routes unchanged. -/
theorem centerBias_isTopK_iff {experts active : Nat} (route : Route experts active)
    (raw bias : Fin experts → ℝ) :
    route.IsTopK (raw + centerBias bias) ↔ route.IsTopK (raw + bias) := by
  have hScores : raw + centerBias bias =
      fun expert => (raw + bias) expert + -((∑ index, bias index) / experts) := by
    funext expert
    simp only [Pi.add_apply, centerBias]
    ring
  rw [hScores]
  exact route.isTopK_add_common_iff (raw + bias) _

/-- One coordinate of the convex QB dual objective. -/
noncomputable def coordinateObjective {tokens : ℕ} (targetLoad : ℝ)
    (margins : Fin tokens → ℝ) (threshold : ℝ) : ℝ :=
  targetLoad * threshold + ∑ token, max 0 (margins token - threshold)

private theorem hinge_le_hinge_of_le {margin candidate threshold : ℝ}
    (hCandidate : candidate ≤ threshold) :
    max 0 (margin - threshold) +
        (if threshold ≤ margin then threshold - candidate else 0) ≤
      max 0 (margin - candidate) := by
  split_ifs with hMargin
  · rw [max_eq_right (by linarith [hCandidate]), max_eq_right (by linarith [hCandidate])]
    linarith
  · rw [max_eq_left (by linarith)]
    simpa only [add_zero] using le_max_left 0 (margin - candidate)

private theorem hinge_le_hinge_of_ge {margin candidate threshold : ℝ} :
    max 0 (margin - threshold) ≤
      max 0 (margin - candidate) +
        (if threshold < margin then candidate - threshold else 0) := by
  split_ifs with hMargin
  · rw [max_eq_right (by linarith)]
    have hDifference : margin - candidate ≤ max 0 (margin - candidate) :=
      le_max_right _ _
    linarith
  · rw [max_eq_left (by linarith)]
    exact add_nonneg (le_max_left _ _) (by linarith)

/-- A threshold minimizes the QB objective when its strict and weak counts bracket the target.

This is the corrected sufficient condition for Appendix C, Eq. 24, and it does not assume that an
exact strict-count quantile exists.
-/
theorem coordinateObjective_le_of_counts {tokens : ℕ} {targetLoad : ℝ}
    (margins : Fin tokens → ℝ) (threshold : ℝ)
    (hStrict : (exceedanceCount margins threshold : ℝ) ≤ targetLoad)
    (hWeak : targetLoad ≤ (atLeastCount margins threshold : ℝ)) (candidate : ℝ) :
    coordinateObjective targetLoad margins threshold ≤
      coordinateObjective targetLoad margins candidate := by
  classical
  rcases le_total candidate threshold with hCandidate | hCandidate
  · have hSum := Finset.sum_le_sum (s := Finset.univ) fun token _ =>
      hinge_le_hinge_of_le (margin := margins token) hCandidate
    rw [Finset.sum_add_distrib, ← Finset.sum_filter, Finset.sum_const,
      nsmul_eq_mul] at hSum
    change (∑ token, max 0 (margins token - threshold)) +
      (atLeastCount margins threshold : ℝ) * (threshold - candidate) ≤
      (∑ token, max 0 (margins token - candidate)) at hSum
    have hProduct := mul_le_mul_of_nonneg_right hWeak (sub_nonneg.mpr hCandidate)
    simp only [coordinateObjective]
    linarith
  · have hSum := Finset.sum_le_sum (s := Finset.univ) fun token _ =>
      hinge_le_hinge_of_ge (margin := margins token) (candidate := candidate)
        (threshold := threshold)
    rw [Finset.sum_add_distrib, ← Finset.sum_filter, Finset.sum_const,
      nsmul_eq_mul] at hSum
    change (∑ token, max 0 (margins token - threshold)) ≤
      (∑ token, max 0 (margins token - candidate)) +
      (exceedanceCount margins threshold : ℝ) * (candidate - threshold) at hSum
    have hProduct := mul_le_mul_of_nonneg_right hStrict (sub_nonneg.mpr hCandidate)
    simp only [coordinateObjective]
    linarith

/-- An exact QB quantile is a global minimizer of its coordinate dual objective. -/
theorem coordinateObjective_le_of_exact_quantile {tokens targetLoad : ℕ}
    (margins : Fin tokens → ℝ) (threshold : ℝ)
    (hQuantile : IsExactTargetQuantile targetLoad margins threshold) (candidate : ℝ) :
    coordinateObjective targetLoad margins threshold ≤
      coordinateObjective targetLoad margins candidate := by
  classical
  apply coordinateObjective_le_of_counts margins threshold
  · exact_mod_cast hQuantile.le
  · rw [← hQuantile]
    exact_mod_cast (show exceedanceCount margins threshold ≤
        atLeastCount margins threshold from by
      apply Finset.card_le_card
      intro token hToken
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hToken ⊢
      exact hToken.le)

private theorem positive_step {tokens : ℕ} (gap : Fin tokens → ℝ) :
    ∃ step > 0, ∀ token, 0 < gap token → step ≤ gap token := by
  classical
  let bounds := insert 1 ((Finset.univ.filter fun token => 0 < gap token).image gap)
  have hBounds : bounds.Nonempty := ⟨1, Finset.mem_insert_self _ _⟩
  refine ⟨bounds.min' hBounds, ?_, ?_⟩
  · apply (Finset.lt_min'_iff _ _).2
    intro value hValue
    rcases Finset.mem_insert.mp hValue with rfl | hValue
    · norm_num
    · obtain ⟨token, hToken, rfl⟩ := Finset.mem_image.mp hValue
      exact (Finset.mem_filter.mp hToken).2
  · intro token hToken
    apply Finset.min'_le
    exact Finset.mem_insert_of_mem (Finset.mem_image.mpr
      ⟨token, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hToken⟩, rfl⟩)

private theorem strict_count_le_of_minimum {tokens : ℕ} {targetLoad : ℝ}
    (margins : Fin tokens → ℝ) (threshold : ℝ)
    (hMin : ∀ candidate, coordinateObjective targetLoad margins threshold ≤
      coordinateObjective targetLoad margins candidate) :
    (exceedanceCount margins threshold : ℝ) ≤ targetLoad := by
  classical
  obtain ⟨step, hStep, hGap⟩ := positive_step (fun token => margins token - threshold)
  have hTerm (token : Fin tokens) :
      max 0 (margins token - (threshold + step)) =
        max 0 (margins token - threshold) -
          (if threshold < margins token then step else 0) := by
    by_cases hMargin : threshold < margins token
    · have hBound := hGap token (sub_pos.mpr hMargin)
      rw [ite_eq_left hMargin, max_eq_right (by linarith),
        max_eq_right (by linarith)]
      ring
    · rw [ite_eq_right hMargin, max_eq_left (by linarith),
        max_eq_left (by linarith)]
      ring
  have hChange : coordinateObjective targetLoad margins (threshold + step) =
      coordinateObjective targetLoad margins threshold +
        ((targetLoad : ℝ) - exceedanceCount margins threshold) * step := by
    simp only [coordinateObjective, hTerm, Finset.sum_sub_distrib]
    rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
    unfold exceedanceCount
    ring
  have hNonnegative := hMin (threshold + step)
  rw [hChange] at hNonnegative
  nlinarith

private theorem target_le_weak_count_of_minimum {tokens : ℕ} {targetLoad : ℝ}
    (margins : Fin tokens → ℝ) (threshold : ℝ)
    (hMin : ∀ candidate, coordinateObjective targetLoad margins threshold ≤
      coordinateObjective targetLoad margins candidate) :
    targetLoad ≤ (atLeastCount margins threshold : ℝ) := by
  classical
  obtain ⟨step, hStep, hGap⟩ := positive_step (fun token => threshold - margins token)
  have hTerm (token : Fin tokens) :
      max 0 (margins token - (threshold - step)) =
        max 0 (margins token - threshold) +
          (if threshold ≤ margins token then step else 0) := by
    by_cases hMargin : threshold ≤ margins token
    · rw [ite_eq_left hMargin, max_eq_right (by linarith),
        max_eq_right (by linarith)]
      ring
    · have hBound := hGap token (sub_pos.mpr (lt_of_not_ge hMargin))
      rw [ite_eq_right hMargin, max_eq_left (by linarith),
        max_eq_left (by linarith)]
      ring
  have hChange : coordinateObjective targetLoad margins (threshold - step) =
      coordinateObjective targetLoad margins threshold +
        ((atLeastCount margins threshold : ℝ) - targetLoad) * step := by
    simp only [coordinateObjective, hTerm, Finset.sum_add_distrib]
    rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
    unfold atLeastCount
    ring
  have hNonnegative := hMin (threshold - step)
  rw [hChange] at hNonnegative
  nlinarith

/-- A threshold minimizes the QB hinge objective iff its strict and weak counts bracket the
target, for real target loads and with ties.

The right slope is the target minus the strict count and the left slope is the target minus the
weak count, so a minimum needs the first nonnegative and the second nonpositive. The finite-step
lemmas above test both slopes without assuming differentiability at ties.
-/
theorem coordinateObjective_minimum_iff {tokens : ℕ} (targetLoad : ℝ)
    (margins : Fin tokens → ℝ) (threshold : ℝ) :
    (∀ candidate, coordinateObjective targetLoad margins threshold ≤
      coordinateObjective targetLoad margins candidate) ↔
      (exceedanceCount margins threshold : ℝ) ≤ targetLoad ∧
        targetLoad ≤ (atLeastCount margins threshold : ℝ) := by
  constructor
  · intro hMin
    exact ⟨strict_count_le_of_minimum margins threshold hMin,
      target_le_weak_count_of_minimum margins threshold hMin⟩
  · rintro ⟨hStrict, hWeak⟩ candidate
    exact coordinateObjective_le_of_counts margins threshold hStrict hWeak candidate

/-! ## The endpoint and tie examples

Two margins and target load one are enough to see both edge cases of Appendix C, Eq. 24.
-/

/-- Two distinct margins, `1` and `0`. -/
noncomputable def endpointMargins : Fin 2 → ℝ := ![1, 0]

/-- Strict count for the endpoint example, one indicator per margin. -/
theorem exceedanceCount_endpointMargins (threshold : ℝ) :
    exceedanceCount endpointMargins threshold =
      (if threshold < 1 then 1 else 0) + (if threshold < 0 then 1 else 0) := by
  unfold exceedanceCount
  rw [Finset.card_filter, Fin.sum_univ_two]
  rfl

/-- Weak count for the endpoint example, one indicator per margin. -/
theorem atLeastCount_endpointMargins (threshold : ℝ) :
    atLeastCount endpointMargins threshold =
      (if threshold ≤ 1 then 1 else 0) + (if threshold ≤ 0 then 1 else 0) := by
  unfold atLeastCount
  rw [Finset.card_filter, Fin.sum_univ_two]
  rfl

/-- For margins `1, 0` and target load one, the minimizers form the closed interval `[0, 1]`
between the first and second largest margins. -/
theorem endpointMargins_minimizer_iff (threshold : ℝ) :
    (∀ candidate, coordinateObjective 1 endpointMargins threshold ≤
      coordinateObjective 1 endpointMargins candidate) ↔ threshold ∈ Set.Icc 0 1 := by
  rw [coordinateObjective_minimum_iff, exceedanceCount_endpointMargins,
    atLeastCount_endpointMargins, Set.mem_Icc]
  rcases lt_or_ge threshold 0 with hZero | hZero
  · rw [ite_eq_left (hZero.trans zero_lt_one), ite_eq_left hZero]
    constructor
    · rintro ⟨hStrict, -⟩
      norm_num at hStrict
    · rintro ⟨hLower, -⟩
      exact absurd hLower (not_le.mpr hZero)
  · rcases le_or_gt threshold 1 with hOne | hOne
    · rw [ite_eq_right (not_lt.mpr hZero), ite_eq_left hOne]
      split_ifs <;> norm_num [hZero, hOne]
    · rw [ite_eq_right (not_lt.mpr hZero), ite_eq_right (not_lt.mpr hOne.le),
        ite_eq_right (not_le.mpr hOne), ite_eq_right (not_le.mpr (zero_lt_one.trans hOne))]
      constructor
      · rintro ⟨-, hWeak⟩
        norm_num at hWeak
      · rintro ⟨-, hUpper⟩
        exact absurd hUpper (not_le.mpr hOne)

/-- At the upper endpoint no margin lies strictly above the threshold. -/
theorem exceedanceCount_endpointMargins_one : exceedanceCount endpointMargins 1 = 0 := by
  rw [exceedanceCount_endpointMargins, ite_eq_right (lt_irrefl 1),
    ite_eq_right (not_lt.mpr zero_le_one)]

/-- "Minimized exactly when k margins lie above the threshold" fails for this batch: the
threshold `1` minimizes the objective, but its strict count is zero. -/
theorem not_forall_minimizer_iff_isExactTargetQuantile :
    ¬ ∀ threshold, (∀ candidate, coordinateObjective 1 endpointMargins threshold ≤
      coordinateObjective 1 endpointMargins candidate) ↔
        IsExactTargetQuantile 1 endpointMargins threshold := by
  intro hClaim
  have hMinimizer := (endpointMargins_minimizer_iff 1).2 ⟨zero_le_one, le_rfl⟩
  have hExact := (hClaim 1).1 hMinimizer
  rw [IsExactTargetQuantile, exceedanceCount_endpointMargins_one] at hExact
  exact zero_ne_one hExact

/-- Two tied margins, both `0`. -/
noncomputable def tiedMargins : Fin 2 → ℝ := ![0, 0]

/-- Strict count for the tie example. -/
theorem exceedanceCount_tiedMargins (threshold : ℝ) :
    exceedanceCount tiedMargins threshold =
      (if threshold < 0 then 1 else 0) + (if threshold < 0 then 1 else 0) := by
  unfold exceedanceCount
  rw [Finset.card_filter, Fin.sum_univ_two]
  rfl

/-- A strict threshold selects both tied margins or neither, never exactly one. -/
theorem exceedanceCount_tiedMargins_ne_one (threshold : ℝ) :
    exceedanceCount tiedMargins threshold ≠ 1 := by
  rw [exceedanceCount_tiedMargins]
  split_ifs <;> norm_num

/-- With both margins tied at `0`, the cutoff `0` still minimizes the objective for target load
one. -/
theorem tiedMargins_zero_minimizer (candidate : ℝ) :
    coordinateObjective 1 tiedMargins 0 ≤ coordinateObjective 1 tiedMargins candidate := by
  apply (coordinateObjective_minimum_iff 1 tiedMargins 0).2
  have hWeak : atLeastCount tiedMargins 0 = 2 := by
    unfold atLeastCount
    rw [Finset.card_filter, Fin.sum_univ_two]
    norm_num [tiedMargins]
  rw [exceedanceCount_tiedMargins, ite_eq_right (lt_irrefl 0), hWeak]
  norm_num

end QuantileBalancing
end KimiK3
