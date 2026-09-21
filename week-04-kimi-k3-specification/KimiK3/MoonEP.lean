/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.FeedForward
public import Mathlib.Algebra.Order.Floor.Div

/-!
# Combinatorial bounds behind MoonEP

Appendix E of the Kimi K3 report separates MoonEP's mathematical planning claim from its
distributed implementation. This file does the same. Experts have fixed home ranks, and a plan
records only the distinct remote experts made available at each destination. No claim is made here
about communication, prefetching, kernels, or the online planner.

The report's upper bound needs two hypotheses that are implicit in the prose: each rank owns at
most `expertsPerRank` experts, and all remote tokens received at a destination come from one source
rank. Under precisely those hypotheses, the number of redundant experts is at most
`expertsPerRank`. The final theorem is the counting argument used for the lower bound.
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

/-- The report's `E/R` conclusion, stated using an explicit per-rank capacity. -/
theorem peakRedundant_le_of_one_source {experts ranks expertsPerRank : ℕ}
    (home : Fin experts → Fin ranks) (remote : Fin ranks → Finset (Fin experts))
    (source : Fin ranks → Fin ranks)
    (hCapacity : ∀ rank, (homeExperts home rank).card ≤ expertsPerRank)
    (hSource : ∀ destination expert, expert ∈ remote destination →
      home expert = source destination) :
    peakRedundant remote ≤ expertsPerRank := by
  exact Finset.sup_le fun destination _ =>
    remote_card_le_of_one_source home remote source hCapacity hSource destination

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

/-- A destination receiving `tokens` needs at least the ceiling number of distinct experts.

This is the finite pigeonhole lower bound used in the MoonEP argument; unlike the previous
packaging, it derives the aggregate inequality from the individual expert fibers.
-/
theorem ceilDiv_card_le_expertCount {Token : Type*}
    {expertCount capacityPerExpert : ℕ} (tokens : Finset Token)
    (assignedExpert : Token → Fin expertCount) (hPositive : 0 < capacityPerExpert)
    (hCapacity : ∀ expert,
      (tokens.filter fun token => assignedExpert token = expert).card ≤ capacityPerExpert) :
    tokens.card ⌈/⌉ capacityPerExpert ≤ expertCount := by
  exact (ceilDiv_le_iff_le_mul hPositive).2
    (card_le_capacity_mul tokens assignedExpert hCapacity)

end MoonEP
end KimiK3
