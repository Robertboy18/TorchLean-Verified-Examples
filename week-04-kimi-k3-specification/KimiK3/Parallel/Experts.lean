/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.Common
public import Mathlib.Algebra.Order.Floor.Div

/-!
# Combinatorial bounds behind MoonEP

Appendix E of the Kimi K3 report separates MoonEP's mathematical planning claim from its
distributed implementation. This file does the same. Experts have fixed home ranks, and a plan
sends every token-expert assignment to the rank that computes it. No claim is made here about
communication, prefetching, kernels, or the online planner.

The upper bound (Theorem 1, p. 45) rests on a key lemma: some plan gives every rank exactly
`S * K` assignments, and the remote assignments of each rank come from one other rank. Its proof
in the report is a construction. Repeatedly fill an underloaded rank up to `S * K` from one
overloaded rank, then put the overloaded rank back. `exists_fill` is that construction, by
induction on the ranks still unbalanced. `moonEP_upper_bound` then proves the report's bound with
no hypothesis beyond its setting: when each rank hosts at most `E / R` experts, every router output
has a balanced plan with at most `E / R` redundant experts on every rank.
`ceilDiv_card_le_expertCount` is the counting argument used for the lower bound.
-/

@[expose] public section

namespace KimiK3
namespace MoonEP

/-- Experts whose fixed home is a given expert-parallel rank. -/
def homeExperts {experts ranks : ℕ} (home : Fin experts → Fin ranks) (rank : Fin ranks) :
    Finset (Fin experts) :=
  Finset.univ.filter fun expert => home expert = rank

/-- One-source migration limits each destination to the source rank's expert capacity. -/
theorem remote_card_le_of_one_source {experts ranks expertsPerRank : ℕ}
    (home : Fin experts → Fin ranks) (remote : Fin ranks → Finset (Fin experts))
    (source : Fin ranks → Fin ranks)
    (hCapacity : ∀ rank, (homeExperts home rank).card ≤ expertsPerRank)
    (hSource : ∀ destination expert, expert ∈ remote destination →
      home expert = source destination)
    (destination : Fin ranks) :
    (remote destination).card ≤ expertsPerRank := by
  apply (Finset.card_le_card ?_).trans (hCapacity (source destination))
  intro expert hExpert
  simp only [homeExperts, Finset.mem_filter, Finset.mem_univ, true_and]
  exact hSource destination expert hExpert

/-- Maximum redundant-expert count over all destinations. -/
def peakRedundant {experts ranks : ℕ} (remote : Fin ranks → Finset (Fin experts)) : ℕ :=
  Finset.univ.sup fun rank => (remote rank).card

/-- Last step of Theorem 1: once each destination's remote experts come from one source rank,
the peak is at most the per-rank capacity. `exists_fill` constructs such a plan. -/
theorem peakRedundant_le_of_one_source {experts ranks expertsPerRank : ℕ}
    (home : Fin experts → Fin ranks) (remote : Fin ranks → Finset (Fin experts))
    (source : Fin ranks → Fin ranks)
    (hCapacity : ∀ rank, (homeExperts home rank).card ≤ expertsPerRank)
    (hSource : ∀ destination expert, expert ∈ remote destination →
      home expert = source destination) :
    peakRedundant remote ≤ expertsPerRank := by
  exact Finset.sup_le fun destination _ =>
    remote_card_le_of_one_source home remote source hCapacity hSource destination

/-! ## Plans and the key lemma -/

section Plans

variable {Assignment : Type*} [Fintype Assignment] [DecidableEq Assignment]
variable {experts ranks : ℕ}

/-- Redundant experts on `rank` under a plan: the distinct experts of the assignments that `rank`
computes although the expert's home is elsewhere. This is `m_r(P)` in Appendix E. -/
def redundantExperts (expertOf : Assignment → Fin experts) (home : Fin experts → Fin ranks)
    (place : Assignment → Fin ranks) (rank : Fin ranks) : Finset (Fin experts) :=
  (Finset.univ.filter fun assignment =>
    place assignment = rank ∧ home (expertOf assignment) ≠ rank).image expertOf

/-- Every rank computes exactly `balanced` assignments. -/
def IsBalanced (place : Assignment → Fin ranks) (balanced : ℕ) : Prop :=
  ∀ rank, (Finset.univ.filter fun assignment => place assignment = rank).card = balanced

/-- Every rank receives its remote assignments from one other rank. -/
def OneSource (expertOf : Assignment → Fin experts) (home : Fin experts → Fin ranks)
    (place : Assignment → Fin ranks) : Prop :=
  ∀ rank, ∃ source : Fin ranks, ∀ assignment, place assignment = rank →
    home (expertOf assignment) ≠ rank → home (expertOf assignment) = source

/-- If some count differs from the average, one count is below it and another above it. -/
private theorem exists_below_above {ι : Type*} (U : Finset ι) (count : ι → ℕ) (balanced : ℕ)
    (hSum : ∑ rank ∈ U, count rank = U.card * balanced)
    (hNot : ¬ ∀ rank ∈ U, count rank = balanced) :
    (∃ rank ∈ U, count rank < balanced) ∧ (∃ rank ∈ U, balanced < count rank) := by
  have hConst : ∑ _rank ∈ U, balanced = U.card * balanced := by simp
  push Not at hNot
  obtain ⟨r, hr, hne⟩ := hNot
  constructor
  · by_contra hNone
    push Not at hNone
    have hlt : ∑ _rank ∈ U, balanced < ∑ rank ∈ U, count rank :=
      Finset.sum_lt_sum hNone ⟨r, hr, lt_of_le_of_ne (hNone r hr) (Ne.symm hne)⟩
    omega
  · by_contra hNone
    push Not at hNone
    have hlt : ∑ rank ∈ U, count rank < ∑ _rank ∈ U, balanced :=
      Finset.sum_lt_sum hNone ⟨r, hr, lt_of_le_of_ne (hNone r hr) hne⟩
    omega

omit [Fintype Assignment] in
/-- **Key lemma of Appendix E, with its construction.**

Let `T` be assignments whose experts live on the ranks in `U`, with `|T| = |U| * balanced`. Then
some plan computes every assignment of `T` on a rank of `U`, gives every rank of `U` exactly
`balanced` assignments, and lets every rank receive remote assignments from one rank only.

Its proof is the report's filling procedure. If all ranks are balanced, every assignment stays
home. Otherwise pick an underloaded rank `u` and an overloaded rank `o`, move exactly
`balanced - load u` of `o`'s own assignments to `u`, remove `u`, and continue with the rest. A
filled rank never changes again, so each rank is filled once, from one rank. -/
theorem exists_fill (expertOf : Assignment → Fin experts) (home : Fin experts → Fin ranks)
    (balanced : ℕ) :
    ∀ (U : Finset (Fin ranks)) (T : Finset Assignment),
      (∀ assignment ∈ T, home (expertOf assignment) ∈ U) →
      T.card = U.card * balanced →
      ∃ place : Assignment → Fin ranks,
        (∀ assignment ∈ T, place assignment ∈ U) ∧
        (∀ rank ∈ U, (T.filter fun assignment => place assignment = rank).card = balanced) ∧
        (∀ rank ∈ U, ∃ source : Fin ranks, ∀ assignment ∈ T, place assignment = rank →
          home (expertOf assignment) ≠ rank → home (expertOf assignment) = source) := by
  intro U
  induction U using Finset.strongInduction with
  | H U ih =>
  intro T hHome hCard
  -- Assignments of `T` whose expert lives on `rank`.
  let localPart : Fin ranks → Finset Assignment := fun rank =>
    T.filter fun assignment => home (expertOf assignment) = rank
  have hSum : ∑ rank ∈ U, (localPart rank).card = U.card * balanced := by
    rw [← hCard]
    exact (Finset.card_eq_sum_card_fiberwise (fun assignment hAssignment =>
      hHome assignment hAssignment)).symm
  by_cases hAll : ∀ rank ∈ U, (localPart rank).card = balanced
  · -- Every rank is balanced: nothing moves.
    refine ⟨fun assignment => home (expertOf assignment), hHome, ?_, ?_⟩
    · exact hAll
    · intro rank _
      exact ⟨rank, fun assignment _ hPlace hRemote => absurd hPlace hRemote⟩
  · obtain ⟨⟨u, hu, hUnder⟩, ⟨o, ho, hOver⟩⟩ :=
      exists_below_above U (fun rank => (localPart rank).card) balanced hSum hAll
    have huo : u ≠ o := by
      intro h
      subst h
      omega
    -- Fill `u` exactly to `balanced` with assignments that live on `o`.
    obtain ⟨moved, hMovedSub, hMovedCard⟩ :=
      Finset.exists_subset_card_eq
        (s := localPart o) (n := balanced - (localPart u).card) (by omega)
    let filled := localPart u ∪ moved
    have hDisjoint : Disjoint (localPart u) moved := by
      rw [Finset.disjoint_left]
      intro assignment hU hMoved
      have hOHome := (Finset.mem_filter.mp (hMovedSub hMoved)).2
      have hUHome := (Finset.mem_filter.mp hU).2
      exact huo (hUHome.symm.trans hOHome)
    have hFilledCard : filled.card = balanced := by
      rw [Finset.card_union_of_disjoint hDisjoint, hMovedCard]
      omega
    have hFilledSub : filled ⊆ T := by
      intro assignment hAssignment
      rcases Finset.mem_union.mp hAssignment with h | h
      · exact (Finset.mem_filter.mp h).1
      · exact (Finset.mem_filter.mp (hMovedSub h)).1
    -- Continue with `u` removed.
    let rest := T \ filled
    have hSub : U.erase u ⊂ U := Finset.erase_ssubset hu
    have hRestHome : ∀ assignment ∈ rest, home (expertOf assignment) ∈ U.erase u := by
      intro assignment hAssignment
      obtain ⟨hT, hNotFilled⟩ := Finset.mem_sdiff.mp hAssignment
      refine Finset.mem_erase.mpr ⟨?_, hHome assignment hT⟩
      intro hEq
      exact hNotFilled (Finset.mem_union_left _ (Finset.mem_filter.mpr ⟨hT, hEq⟩))
    have hRestCard : rest.card = (U.erase u).card * balanced := by
      rw [Finset.card_sdiff_of_subset hFilledSub, hFilledCard, hCard,
        Finset.card_erase_of_mem hu, Nat.sub_mul, Nat.one_mul]
    obtain ⟨place', hPlaceU, hPlaceCount, hPlaceSource⟩ :=
      ih (U.erase u) hSub rest hRestHome hRestCard
    refine ⟨fun assignment => if assignment ∈ filled then u else place' assignment, ?_, ?_, ?_⟩
    · intro assignment hAssignment
      by_cases hFilled : assignment ∈ filled
      · simp [hFilled, hu]
      · simp only [hFilled, ite_false]
        exact Finset.mem_of_mem_erase
          (hPlaceU assignment (Finset.mem_sdiff.mpr ⟨hAssignment, hFilled⟩))
    · intro rank hRank
      by_cases hRankU : rank = u
      · subst hRankU
        have hEq : (T.filter fun assignment =>
            (if assignment ∈ filled then rank else place' assignment) = rank) = filled := by
          ext assignment
          constructor
          · intro hMem
            obtain ⟨hT, hPlace⟩ := Finset.mem_filter.mp hMem
            by_contra hNotFilled
            simp only [hNotFilled, ite_false] at hPlace
            have := hPlaceU assignment (Finset.mem_sdiff.mpr ⟨hT, hNotFilled⟩)
            rw [hPlace] at this
            exact (Finset.notMem_erase rank U) this
          · intro hMem
            exact Finset.mem_filter.mpr ⟨hFilledSub hMem, by simp [hMem]⟩
        rw [hEq, hFilledCard]
      · have hEq : (T.filter fun assignment =>
            (if assignment ∈ filled then u else place' assignment) = rank) =
            rest.filter fun assignment => place' assignment = rank := by
          ext assignment
          simp only [Finset.mem_filter, Finset.mem_sdiff, rest]
          constructor
          · rintro ⟨hT, hPlace⟩
            by_cases hFilled : assignment ∈ filled
            · simp only [hFilled, ite_true] at hPlace
              exact absurd hPlace.symm hRankU
            · simp only [hFilled, ite_false] at hPlace
              exact ⟨⟨hT, hFilled⟩, hPlace⟩
          · rintro ⟨⟨hT, hFilled⟩, hPlace⟩
            exact ⟨hT, by simp [hFilled, hPlace]⟩
        rw [hEq]
        exact hPlaceCount rank (Finset.mem_erase.mpr ⟨hRankU, hRank⟩)
    · intro rank hRank
      by_cases hRankU : rank = u
      · subst hRankU
        refine ⟨o, fun assignment hT hPlace hRemote => ?_⟩
        by_cases hFilled : assignment ∈ filled
        · rcases Finset.mem_union.mp hFilled with h | h
          · exact absurd (Finset.mem_filter.mp h).2 hRemote
          · exact (Finset.mem_filter.mp (hMovedSub h)).2
        · simp only [hFilled, ite_false] at hPlace
          have := hPlaceU assignment (Finset.mem_sdiff.mpr ⟨hT, hFilled⟩)
          rw [hPlace] at this
          exact absurd this (Finset.notMem_erase rank U)
      · obtain ⟨source, hSource⟩ :=
          hPlaceSource rank (Finset.mem_erase.mpr ⟨hRankU, hRank⟩)
        refine ⟨source, fun assignment hT hPlace hRemote => ?_⟩
        by_cases hFilled : assignment ∈ filled
        · simp only [hFilled, ite_true] at hPlace
          exact absurd hPlace.symm hRankU
        · simp only [hFilled, ite_false] at hPlace
          exact hSource assignment (Finset.mem_sdiff.mpr ⟨hT, hFilled⟩) hPlace hRemote

/-- Key lemma over all assignments: if there are `ranks * balanced` of them, a balanced
one-source plan exists. -/
theorem exists_balanced_oneSource (expertOf : Assignment → Fin experts)
    (home : Fin experts → Fin ranks) (balanced : ℕ)
    (hCard : Fintype.card Assignment = ranks * balanced) :
    ∃ place : Assignment → Fin ranks,
      IsBalanced place balanced ∧ OneSource expertOf home place := by
  obtain ⟨place, -, hCount, hSource⟩ :=
    exists_fill expertOf home balanced Finset.univ Finset.univ (by simp)
      (by simpa using hCard)
  refine ⟨place, fun rank => hCount rank (Finset.mem_univ rank), fun rank => ?_⟩
  obtain ⟨source, h⟩ := hSource rank (Finset.mem_univ rank)
  exact ⟨source, fun assignment => h assignment (Finset.mem_univ assignment)⟩

omit [DecidableEq Assignment] in
/-- In a one-source plan, a rank's redundant experts all live on its source rank, so there are at
most `expertsPerRank` of them. -/
theorem redundantExperts_card_le_of_oneSource {expertsPerRank : ℕ}
    (expertOf : Assignment → Fin experts) (home : Fin experts → Fin ranks)
    (place : Assignment → Fin ranks)
    (hCapacity : ∀ rank, (homeExperts home rank).card ≤ expertsPerRank)
    (hOne : OneSource expertOf home place) (rank : Fin ranks) :
    (redundantExperts expertOf home place rank).card ≤ expertsPerRank := by
  obtain ⟨source, hSource⟩ := hOne rank
  apply (Finset.card_le_card ?_).trans (hCapacity source)
  intro expert hExpert
  obtain ⟨assignment, hAssignment, rfl⟩ := Finset.mem_image.mp hExpert
  obtain ⟨hPlace, hRemote⟩ := (Finset.mem_filter.mp hAssignment).2
  simp only [homeExperts, Finset.mem_filter, Finset.mem_univ, true_and]
  exact hSource assignment hPlace hRemote

/-- Every router output admits a balanced plan whose redundant experts per rank are bounded by the
experts hosted per rank. -/
theorem exists_balanced_redundant_le {expertsPerRank : ℕ}
    (expertOf : Assignment → Fin experts) (home : Fin experts → Fin ranks) (balanced : ℕ)
    (hCard : Fintype.card Assignment = ranks * balanced)
    (hCapacity : ∀ rank, (homeExperts home rank).card ≤ expertsPerRank) :
    ∃ place : Assignment → Fin ranks, IsBalanced place balanced ∧
      peakRedundant (redundantExperts expertOf home place) ≤ expertsPerRank := by
  obtain ⟨place, hBalanced, hOne⟩ := exists_balanced_oneSource expertOf home balanced hCard
  exact ⟨place, hBalanced, Finset.sup_le fun rank _ =>
    redundantExperts_card_le_of_oneSource expertOf home place hCapacity hOne rank⟩

end Plans

/-- Per-expert capacities bound the total number of routed tokens. -/
theorem card_le_capacity_mul {Token : Type*}
    {expertCount capacityPerExpert : ℕ} (tokens : Finset Token)
    (assignedExpert : Token → Fin expertCount)
    (hCapacity : ∀ expert,
      (tokens.filter fun token => assignedExpert token = expert).card ≤ capacityPerExpert) :
    tokens.card ≤ capacityPerExpert * expertCount := by
  rw [Finset.card_eq_sum_card_fiberwise
    (t := Finset.univ) (f := assignedExpert) (by simp)]
  calc
    (∑ expert : Fin expertCount,
        (tokens.filter fun token => assignedExpert token = expert).card) ≤
        ∑ _expert : Fin expertCount, capacityPerExpert :=
      Finset.sum_le_sum fun expert _ => hCapacity expert
    _ = capacityPerExpert * expertCount := by
      simp [Nat.mul_comm]

/-- If no expert takes more than `capacityPerExpert` of `tokens`, there are at least
`⌈|tokens| / capacityPerExpert⌉` experts.

This is the pigeonhole bound of the MoonEP lower-bound argument, summed over the expert fibers.
-/
theorem ceilDiv_card_le_expertCount {Token : Type*}
    {expertCount capacityPerExpert : ℕ} (tokens : Finset Token)
    (assignedExpert : Token → Fin expertCount) (hPositive : 0 < capacityPerExpert)
    (hCapacity : ∀ expert,
      (tokens.filter fun token => assignedExpert token = expert).card ≤ capacityPerExpert) :
    tokens.card ⌈/⌉ capacityPerExpert ≤ expertCount := by
  exact (ceilDiv_le_iff_le_mul hPositive).2
    (card_le_capacity_mul tokens assignedExpert hCapacity)

/-- **Theorem 1 of Appendix E (general upper bound), in the report's setting.**

`R = ranks` expert-parallel ranks each hold `S` tokens, and every token selects `K` experts
(`route rank token slot`). Each rank hosts at most `E / R` of the `E = experts` experts. Then a plan
exists in which every rank computes exactly `S * K` assignments and no rank holds more than `E / R`
redundant experts: `M(I) ≤ E / R` for every router output `I`. Nothing is assumed about the plan,
because `exists_fill` constructs the one-source property. -/
theorem moonEP_upper_bound {experts ranks S K : ℕ} (route : Fin ranks → Fin S → Fin K → Fin experts)
    (home : Fin experts → Fin ranks)
    (hCapacity : ∀ rank, (homeExperts home rank).card ≤ experts / ranks) :
    ∃ place : Fin ranks × Fin S × Fin K → Fin ranks,
      IsBalanced place (S * K) ∧
      peakRedundant (redundantExperts (fun assignment =>
        route assignment.1 assignment.2.1 assignment.2.2) home place) ≤ experts / ranks :=
  exists_balanced_redundant_le _ home (S * K) (by simp [Fintype.card_prod]) hCapacity

end MoonEP
end KimiK3
