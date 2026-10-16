/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.Common

/-!
# Top-k expert routing

Routes are `Function.Embedding`s, which rules out repeated experts. Selection sorts adjusted
scores, breaks ties by expert index, and is proved to satisfy the top-k contract.

Reference: https://arxiv.org/abs/2607.24653, Section 2.3.3, Eq. 13.
-/

@[expose] public section

namespace KimiK3

open Spec TorchLean
open Tensor

/-- A route is an embedding of active slots into experts, so no expert can be selected twice. -/
abbrev Route (numExperts activeExperts : Nat) := Fin activeExperts ↪ Fin numExperts

namespace Route

variable {numExperts activeExperts : Nat}

/-- Every selected expert scores at least as high as every unselected one. -/
def IsTopK (route : Route numExperts activeExperts) (score : Fin numExperts → ℝ) : Prop :=
  ∀ selected candidate,
    (∀ slot, route slot ≠ candidate) →
      score (route selected) ≥ score candidate

/-- Adding the same constant to every score does not change which routes satisfy top-k. -/
theorem isTopK_add_common_iff (route : Route numExperts activeExperts)
    (score : Fin numExperts → ℝ) (offset : ℝ) :
    route.IsTopK (fun expert => score expert + offset) ↔ route.IsTopK score := by
  constructor <;> intro h selected candidate hCandidate
  · have := h selected candidate hCandidate
    linarith
  · have := h selected candidate hCandidate
    linarith

namespace Internal

/-- Descending score order with the lower expert index breaking ties. -/
def betterEq (score : Fin numExperts → ℝ)
    (left right : Fin numExperts) : Prop :=
  score right < score left ∨ (score left = score right ∧ left.val ≤ right.val)

/-- `betterEq`, the tie-broken score order, is transitive. -/
theorem betterEq_trans (score : Fin numExperts → ℝ) {a b c : Fin numExperts}
    (hab : betterEq score a b) (hbc : betterEq score b c) : betterEq score a c := by
  rcases hab with hab | ⟨habScore, habIndex⟩
  · rcases hbc with hbc | ⟨hbcScore, _⟩
    · left; linarith
    · left; linarith
  · rcases hbc with hbc | ⟨hbcScore, hbcIndex⟩
    · left; linarith
    · right
      constructor
      · linarith
      · omega

/-- Every pair is comparable, including equal scores with distinct indices. -/
theorem betterEq_total (score : Fin numExperts → ℝ) (a b : Fin numExperts) :
    betterEq score a b ∨ betterEq score b a := by
  rcases lt_trichotomy (score a) (score b) with hab | hab | hab
  · exact Or.inr (Or.inl hab)
  · by_cases hIndex : a.val ≤ b.val
    · exact Or.inl (Or.inr ⟨hab, hIndex⟩)
    · exact Or.inr (Or.inr ⟨hab.symm, Nat.le_of_lt (Nat.lt_of_not_ge hIndex)⟩)
  · exact Or.inl (Or.inl hab)

/-- Forgetting the tie-breaker recovers the non-strict score inequality. -/
theorem betterEq_score_ge (score : Fin numExperts → ℝ) {a b : Fin numExperts}
    (h : betterEq score a b) : score a ≥ score b := by
  rcases h with h | ⟨h, _⟩
  · exact h.le
  · exact h.symm.le

/-- All expert indices ordered by decreasing score, with lower indices first on ties. -/
noncomputable def orderedExperts (score : Fin numExperts → ℝ) : List (Fin numExperts) := by
  classical
  exact (List.finRange numExperts).mergeSort
    (fun left right => decide (betterEq score left right))

/-- Any earlier sorted expert outranks any later one, not merely its immediate neighbor. -/
theorem orderedExperts_pairwise (score : Fin numExperts → ℝ) :
    (orderedExperts score).Pairwise (betterEq score) := by
  classical
  unfold orderedExperts
  simpa only [decide_eq_true_eq] using
    (List.pairwise_mergeSort
      (le := fun left right => decide (betterEq score left right))
      (fun a b c hab hbc => by
        simpa only [decide_eq_true_eq] using betterEq_trans score
          (by simpa only [decide_eq_true_eq] using hab)
          (by simpa only [decide_eq_true_eq] using hbc))
      (fun a b => by
        simp only [Bool.or_eq_true, decide_eq_true_eq]
        exact betterEq_total score a b)
      (List.finRange numExperts))

/-- Sorting changes order without duplicating expert indices. -/
theorem orderedExperts_nodup (score : Fin numExperts → ℝ) :
    (orderedExperts score).Nodup := by
  classical
  unfold orderedExperts
  exact List.nodup_mergeSort.mpr (List.nodup_finRange numExperts)

/-- Sorting retains the full expert population. -/
theorem orderedExperts_length (score : Fin numExperts → ℝ) :
    (orderedExperts score).length = numExperts := by
  exact (List.mergeSort_perm (List.finRange numExperts) _).length_eq.trans
    List.length_finRange

/-- Every expert remains available in the sorted list. -/
theorem orderedExperts_mem (score : Fin numExperts → ℝ) (expert : Fin numExperts) :
    expert ∈ orderedExperts score := by
  exact (List.mergeSort_perm (List.finRange numExperts) _).mem_iff.mpr
    (List.mem_finRange expert)

end Internal

/-- Choose the `activeExperts` distinct experts with the highest adjusted scores. The hypothesis
`activeExperts ≤ numExperts` rules out impossible routes, and index tie-breaking makes the result
deterministic. -/
noncomputable def chooseTopK (score : Fin numExperts → ℝ)
    (hActive : activeExperts ≤ numExperts) : Route numExperts activeExperts := by
  let ordered := Internal.orderedExperts score
  let selected := ordered.take activeExperts
  -- The length and no-duplicate proofs turn the selected prefix into an embedding.
  have hLength : selected.length = activeExperts := by
    simp only [selected, List.length_take, ordered, Internal.orderedExperts_length]
    exact Nat.min_eq_left hActive
  have hNodup : selected.Nodup := (Internal.orderedExperts_nodup score).take
  refine
    { toFun := fun slot => selected.get (Fin.cast hLength.symm slot)
      inj' := ?_ }
  intro left right hEqual
  apply Fin.cast_inj hLength.symm |>.mp
  exact hNodup.get_inj_iff.mp hEqual

namespace Internal

/-- Every active slot selects an expert from the sorted prefix. -/
theorem chooseTopK_expert_mem_take (score : Fin numExperts → ℝ)
    (hActive : activeExperts ≤ numExperts) (slot : Fin activeExperts) :
    (chooseTopK score hActive) slot ∈ (orderedExperts score).take activeExperts := by
  simp only [chooseTopK]
  exact List.get_mem _ _

/-- An unselected expert must lie in the suffix, since sorting retains every index. -/
theorem mem_drop_of_not_selected (score : Fin numExperts → ℝ)
    (hActive : activeExperts ≤ numExperts) (candidate : Fin numExperts)
    (hCandidate : ∀ slot, (chooseTopK score hActive) slot ≠ candidate) :
    candidate ∈ (orderedExperts score).drop activeExperts := by
  have hMem := orderedExperts_mem score candidate
  rw [← List.take_append_drop activeExperts (orderedExperts score)] at hMem
  rcases List.mem_append.mp hMem with hTake | hDrop
  · obtain ⟨index, hIndex⟩ := List.get_of_mem hTake
    have hLength : ((orderedExperts score).take activeExperts).length = activeExperts := by
      simp [orderedExperts_length, Nat.min_eq_left hActive]
    let slot : Fin activeExperts := Fin.cast hLength index
    exfalso
    apply hCandidate slot
    simp only [chooseTopK]
    change ((orderedExperts score).take activeExperts).get
      (Fin.cast hLength.symm slot) = candidate
    simpa [slot] using hIndex
  · exact hDrop

end Internal

/-- `chooseTopK` satisfies the top-k contract `IsTopK`. -/
theorem chooseTopK_isTopK (score : Fin numExperts → ℝ)
    (hActive : activeExperts ≤ numExperts) :
    (chooseTopK score hActive).IsTopK score := by
  intro selectedSlot candidate hCandidate
  let ordered := Internal.orderedExperts score
  have hPairwise : ordered.Pairwise (Internal.betterEq score) :=
    Internal.orderedExperts_pairwise score
  have hAppend : ordered.take activeExperts ++ ordered.drop activeExperts = ordered :=
    List.take_append_drop activeExperts ordered
  -- Pairwise ordering compares every selected-prefix entry with every unselected-suffix entry.
  have hCross : ∀ a ∈ ordered.take activeExperts, ∀ b ∈ ordered.drop activeExperts,
      Internal.betterEq score a b := by
    rw [← hAppend] at hPairwise
    exact (List.pairwise_append.mp hPairwise).2.2
  apply Internal.betterEq_score_ge score
  exact hCross ((chooseTopK score hActive) selectedSlot)
    (Internal.chooseTopK_expert_mem_take score hActive selectedSlot) candidate
    (Internal.mem_drop_of_not_selected score hActive candidate hCandidate)
end Route

end KimiK3
