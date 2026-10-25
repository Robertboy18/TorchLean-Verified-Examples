/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import ProgramEquilibrium.Agents

/-!
# Reciprocity guards

The strict prefix `□ me` and the delayed prefix `□ (¬ □^k ⊥ → me)` each prevent
eventual exploitation, even after conjoining an arbitrary extra guard. These are
statements about any compatible modal universe, not only a finite test population.
They do not guarantee useful cooperation or safety at every early world.

The delayed prefix is related to Barasz et al.'s WaitFairBot family (Section 5),
whose formula also has an outer `¬ □^k ⊥` conjunct.
-/

@[expose] public section

namespace ProgramEquilibrium.Form

/-- Recognize the cooperation prefix that supplies the guard's safety guarantee. -/
def isGuardedSchema {α : Type} : Form (Target α) → Bool
  | .and (.box (.atom .me)) _ => true
  | _ => false

theorem isGuardedSchema_sound {α : Type} {schema : Form (Target α)}
    (h : schema.isGuardedSchema = true) :
    ∃ guard, schema = and (box (atom .me)) guard := by
  cases schema with
  | and left guard =>
    cases left with
    | box body =>
      cases body with
      | atom target =>
        cases target with
        | me => exact ⟨guard, rfl⟩
        | agent _ => simp [isGuardedSchema] at h
      | _ => simp [isGuardedSchema] at h
    | _ => simp [isGuardedSchema] at h
  | _ => simp [isGuardedSchema] at h

end ProgramEquilibrium.Form

namespace ProgramEquilibrium.Universe

open ProgramEquilibrium Form

variable {U : Universe} {f b : U.Agent} {guard : Form (Target U.Agent)}

/-- At any world where the guarded policy cooperates, its opponent cooperated with it at every
strictly earlier world. The additional guard has no bearing on this necessary condition. -/
theorem guarded_play_requires_counterparty_below
    (hf : U.schema f = and (box (atom .me)) guard) {n : ℕ}
    (h : U.play n f b = true) : ∀ m < n, U.play m b f = true := by
  rw [U.play_eq] at h
  simp only [rule, hf, map] at h
  have hbox := (Bool.and_eq_true_iff.mp h).1
  exact (eval_box (fun k p => U.play k p.1 p.2) n (atom (b, f))).1 hbox

/-- Eventual cooperation by the guarded policy requires its opponent to cooperate with it at
every world, not merely at sufficiently late worlds. -/
theorem guard_cooperation_requires_counterparty_all_worlds
    (hf : U.schema f = and (box (atom .me)) guard)
    (h : U.Cooperates f b) : ∀ m, U.play m b f = true := by
  obtain ⟨K, hK⟩ := h
  intro m
  exact guarded_play_requires_counterparty_below hf
    (hK (max K (m + 1)) (Nat.le_max_left K (m + 1))) m (by omega)

/-- An opponent that eventually defects causes this guarded policy to eventually defect too.
This holds for any extra guard and does not rely on testing a finite list of opponents. -/
theorem guarded_unexploitable (hf : U.schema f = and (box (atom .me)) guard)
    (h : U.Defects b f) : U.Defects f b := by
  obtain ⟨K, hK⟩ := h
  refine ⟨K + 1, fun n hn => Bool.eq_false_iff.2 ?_⟩
  intro hcoop
  have counterparty := guarded_play_requires_counterparty_below hf hcoop K (by omega)
  rw [hK K (Nat.le_refl K)] at counterparty
  contradiction

/-- Guarded cooperation and eventual counterparty defection cannot occur together. -/
theorem guarded_not_exploited (hf : U.schema f = and (box (atom .me)) guard) :
    ¬ (U.Cooperates f b ∧ U.Defects b f) := by
  rintro ⟨hcoop, hdefect⟩
  exact U.not_cooperates_and_defects f b ⟨hcoop, guarded_unexploitable hf hdefect⟩

/-- The executable schema check suffices for non-exploitation against every opponent in any
universe containing the checked agent, whether that universe is finite or infinite. -/
theorem checked_guard_not_exploited (h : (U.schema f).isGuardedSchema = true) :
    ¬ (U.Cooperates f b ∧ U.Defects b f) := by
  obtain ⟨guard, hguard⟩ := Form.isGuardedSchema_sound h
  exact guarded_not_exploited hguard

end ProgramEquilibrium.Universe

namespace ProgramEquilibrium.Form

/-- `k` boxes around falsity, true exactly below world `k`. -/
def cutoff {α : Type} : ℕ → Form α
  | 0 => bot
  | k + 1 => box (cutoff k)

@[simp] theorem map_cutoff {α β : Type} (f : α → β) (k : ℕ) :
    (cutoff k).map f = cutoff k := by
  induction k with
  | zero => rfl
  | succ k ih => simp [cutoff, map, ih]

theorem eval_cutoff {α : Type} (V : ℕ → α → Bool) (k n : ℕ) :
    eval V n (cutoff k) = true ↔ n < k := by
  induction k generalizing n with
  | zero => simp [cutoff, eval]
  | succ k ih =>
    rw [cutoff, eval_box]
    constructor
    · intro h
      by_contra hn
      have hk : k < n := by omega
      have := (ih k).1 (h k hk)
      omega
    · intro hn m hm
      exact (ih m).2 (by omega)

end ProgramEquilibrium.Form

namespace ProgramEquilibrium.Universe

open Form

variable {U : Universe} {f b : U.Agent} {k : ℕ} {guard : Form (Target U.Agent)}

/-- Cooperation requires reciprocity at every earlier world above the cutoff. -/
theorem delayed_guard_requires_reciprocity
    (hf : U.schema f = and (box (imp (not (cutoff k)) (atom .me))) guard)
    {n m : ℕ} (h : U.play n f b = true) (hkm : k ≤ m) (hmn : m < n) :
    U.play m b f = true := by
  rw [U.play_eq] at h
  simp only [rule, hf, map, map_cutoff] at h
  have hbox := (Bool.and_eq_true_iff.mp h).1
  have hbody := (eval_box _ _ _).1 hbox m hmn
  have hcut : eval (fun j (p : U.Agent × U.Agent) => U.play j p.1 p.2)
      m (cutoff k) = false := by
    apply Bool.eq_false_iff.2
    intro htrue
    have := (eval_cutoff _ k m).1 htrue
    omega
  have hnot := (eval_not _ _ _).2 hcut
  exact (eval_imp _ _ _ _).1 hbody hnot

/-- A finite modal cutoff does not allow cooperation with an eventual defector. -/
theorem delayed_guard_unexploitable
    (hf : U.schema f = and (box (imp (not (cutoff k)) (atom .me))) guard)
    (h : U.Defects b f) : U.Defects f b := by
  obtain ⟨K, hK⟩ := h
  refine ⟨max K k + 1, fun n hn => Bool.eq_false_iff.2 ?_⟩
  intro hcoop
  have other := delayed_guard_requires_reciprocity hf hcoop
    (m := max K k) (Nat.le_max_right K k) (by omega)
  rw [hK (max K k) (Nat.le_max_left K k)] at other
  contradiction

/-- The delayed guard prevents eventual exploitation in any compatible universe. -/
theorem delayed_guard_not_exploited
    (hf : U.schema f = and (box (imp (not (cutoff k)) (atom .me))) guard) :
    ¬ (U.Cooperates f b ∧ U.Defects b f) := by
  rintro ⟨hcoop, hdefect⟩
  exact U.not_cooperates_and_defects f b
    ⟨hcoop, delayed_guard_unexploitable hf hdefect⟩

end ProgramEquilibrium.Universe

