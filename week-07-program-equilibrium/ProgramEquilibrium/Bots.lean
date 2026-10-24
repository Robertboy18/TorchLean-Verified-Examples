/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import ProgramEquilibrium.Agents

/-!
# CooperateBot, DefectBot, FairBot and PrudentBot

The agents of Barasz et al. (2014), as schemas over "my opponent cooperates with …":

* CooperateBot always cooperates, and DefectBot always defects.
* FairBot cooperates when it can prove that its opponent cooperates with it: `□ me`.
* `twin` is FairBot written differently, `□ (me ∨ ⊥)`. The two have different formulas, so an
  agent that only cooperates with exact copies of itself would not cooperate with both.
* PrudentBot also wants a proof, assuming arithmetic is consistent, that its opponent defects
  against DefectBot: `□ me ∧ □ (¬□⊥ → ¬ [opponent cooperates with DefectBot])`.
* `cupod` cooperates unless it can prove that its opponent defects against it: `¬□¬ me`. It is
  the modal idealization of Critch, Dennis and Russell's CUPOD(k), with ordinary provability in
  place of a bounded proof search; the same idealization of their DUPOC(k) is FairBot.

The theorems below prove the article's matchups at every world of the chain, and `outcomes`
collects their eventual values. Every box holds at world `0`, so FairBot and PrudentBot cooperate
with DefectBot there, and PrudentBot cooperates with CooperateBot at worlds `0` and `1`. In the
limit FairBot, its twin and PrudentBot cooperate with each other, DefectBot exploits none of
them, and PrudentBot, unlike FairBot, defects against CooperateBot.
-/

@[expose] public section

namespace ProgramEquilibrium

open Form

/-- The agents of the article. -/
inductive Bot where
  | cooperate
  | defect
  | fair
  | twin
  | prudent
  | cupod
  deriving DecidableEq, Repr

namespace Bot

/-- Each agent's formula, the same against every opponent. -/
def schema : Bot → Form (Target Bot)
  | cooperate => top
  | defect => bot
  | fair => box (atom .me)
  | twin => box (or (atom .me) bot)
  | prudent => and (box (atom .me)) (box (imp (not (box bot)) (not (atom (.agent defect)))))
  | cupod => not (box (not (atom .me)))

end Bot

/-- The population of the article's agents. -/
abbrev bots : Universe where
  Agent := Bot
  schema := Bot.schema
  modalized a := by cases a <;> rfl

namespace Bot

theorem cooperate_play (n : ℕ) (b : Bot) : bots.play n cooperate b = true := by
  rw [Universe.play_eq]
  rfl

theorem defect_play (n : ℕ) (b : Bot) : bots.play n defect b = false := by
  rw [Universe.play_eq]
  rfl

theorem fair_play (n : ℕ) (b : Bot) :
    bots.play n fair b = true ↔ ∀ m < n, bots.play m b fair = true := by
  rw [Universe.play_eq]
  exact eval_box (fun m p => bots.play m p.1 p.2) n (atom (b, fair))

theorem twin_play (n : ℕ) (b : Bot) :
    bots.play n twin b = true ↔ ∀ m < n, bots.play m b twin = true := by
  rw [Universe.play_eq]
  change eval (fun m p => bots.play m p.1 p.2) n (box (or (atom (b, twin)) bot)) = true ↔ _
  rw [eval_box]
  exact forall₂_congr fun m _ => by simp [eval]

theorem prudent_play (n : ℕ) (b : Bot) :
    bots.play n prudent b = true ↔
      (∀ m < n, bots.play m b prudent = true) ∧ ∀ m < n, m ≠ 0 → bots.play m b defect = false := by
  rw [Universe.play_eq]
  change (eval (fun m p => bots.play m p.1 p.2) n (box (atom (b, prudent))) &&
      eval (fun m p => bots.play m p.1 p.2) n
        (box (imp (not (box bot)) (not (atom (b, defect)))))) = true ↔ _
  rw [Bool.and_eq_true, eval_box, eval_box]
  refine and_congr Iff.rfl (forall₂_congr fun m _ => ?_)
  rw [eval_imp, eval_not, eval_not, Bool.eq_false_iff, ne_eq, eval_box_bot]
  rfl

/-- FairBot cooperates with FairBot at every world: Löb's theorem in the table. -/
theorem fair_fair (n : ℕ) : bots.play n fair fair = true := by
  induction n using Nat.strongRecOn with
  | _ n ih => exact (fair_play n fair).2 ih

/-- FairBot and its twin cooperate with each other at every world, though their formulas
differ. -/
theorem fair_twin (n : ℕ) : bots.play n fair twin = true ∧ bots.play n twin fair = true := by
  induction n using Nat.strongRecOn with
  | _ n ih =>
    exact ⟨(fair_play n twin).2 fun m hm => (ih m hm).2,
      (twin_play n fair).2 fun m hm => (ih m hm).1⟩

/-- FairBot cooperates with DefectBot only at world `0`. -/
theorem fair_defect (n : ℕ) : bots.play n fair defect = true ↔ n = 0 := by
  rw [fair_play]
  constructor
  · intro h
    by_contra hn
    have := h 0 (Nat.pos_of_ne_zero hn)
    simp [defect_play] at this
  · rintro rfl m hm
    omega

theorem fair_cooperate (n : ℕ) : bots.play n fair cooperate = true :=
  (fair_play n cooperate).2 fun m _ => cooperate_play m fair

/-- PrudentBot cooperates with DefectBot only at world `0`. -/
theorem prudent_defect (n : ℕ) : bots.play n prudent defect = true ↔ n = 0 := by
  rw [prudent_play]
  constructor
  · rintro ⟨h, -⟩
    by_contra hn
    have := h 0 (Nat.pos_of_ne_zero hn)
    simp [defect_play] at this
  · rintro rfl
    exact ⟨fun m hm => by omega, fun m hm => by omega⟩

/-- PrudentBot cooperates with CooperateBot only at worlds `0` and `1`. -/
theorem prudent_cooperate (n : ℕ) : bots.play n prudent cooperate = true ↔ n ≤ 1 := by
  rw [prudent_play]
  constructor
  · rintro ⟨-, h⟩
    by_contra hn
    have := h 1 (by omega) Nat.one_ne_zero
    simp [cooperate_play] at this
  · intro hn
    exact ⟨fun m _ => cooperate_play m prudent, fun m hm h0 => by omega⟩

/-- PrudentBot cooperates with PrudentBot at every world. -/
theorem prudent_prudent (n : ℕ) : bots.play n prudent prudent = true := by
  induction n using Nat.strongRecOn with
  | _ n ih =>
    refine (prudent_play n prudent).2 ⟨ih, fun m _ hm => ?_⟩
    exact Bool.eq_false_iff.2 fun h => hm ((prudent_defect m).1 h)

/-- PrudentBot and FairBot cooperate with each other at every world. -/
theorem prudent_fair (n : ℕ) : bots.play n prudent fair = true ∧ bots.play n fair prudent = true := by
  induction n using Nat.strongRecOn with
  | _ n ih =>
    refine ⟨(prudent_play n fair).2 ⟨fun m hm => (ih m hm).2, fun m _ hm => ?_⟩,
      (fair_play n prudent).2 fun m hm => (ih m hm).1⟩
    exact Bool.eq_false_iff.2 fun h => hm ((fair_defect m).1 h)

theorem cupod_play (n : ℕ) (b : Bot) :
    bots.play n cupod b = true ↔ ∃ m < n, bots.play m b cupod = true := by
  rw [Universe.play_eq]
  change eval (fun m p => bots.play m p.1 p.2) n (not (box (not (atom (b, cupod))))) = true ↔ _
  rw [eval_not, Bool.eq_false_iff, ne_eq, eval_box]
  simp only [eval_not, not_forall, Bool.not_eq_false]
  exact exists_congr fun m => ⟨fun ⟨h, e⟩ => ⟨h, e⟩, fun ⟨h, e⟩ => ⟨h, e⟩⟩

/-- CUPOD defects against CUPOD at every world, as CUPOD(k) does for large `k`
(Critch, Dennis and Russell, Theorem 3.4). -/
theorem cupod_cupod (n : ℕ) : bots.play n cupod cupod = false := by
  induction n using Nat.strongRecOn with
  | _ n ih =>
    apply Bool.eq_false_iff.2
    intro h
    obtain ⟨m, hm, hc⟩ := (cupod_play n cupod).1 h
    simp [ih m hm] at hc

/-- Against CUPOD, FairBot cooperates only at world `0`, and CUPOD cooperates from world `1` on.
Critch, Dennis and Russell conjecture the same outcome, (D, C), for DUPOC(k) against CUPOD(k)
(their Open Problem 3). -/
theorem fair_cupod (n : ℕ) :
    (bots.play n fair cupod = true ↔ n = 0) ∧ (bots.play n cupod fair = true ↔ 1 ≤ n) := by
  induction n using Nat.strongRecOn with
  | _ n ih =>
    constructor
    · rw [fair_play]
      constructor
      · intro h
        by_contra hn
        have := (ih 0 (Nat.pos_of_ne_zero hn)).2.1 (h 0 (Nat.pos_of_ne_zero hn))
        omega
      · rintro rfl m hm
        omega
    · rw [cupod_play]
      constructor
      · rintro ⟨m, hm, -⟩
        omega
      · intro hn
        exact ⟨0, by omega, (ih 0 (by omega)).1.2 rfl⟩

/-- The eventual outcome of every matchup the article discusses. -/
theorem outcomes :
    bots.Cooperates fair fair ∧ bots.Cooperates fair twin ∧ bots.Cooperates twin fair ∧
    bots.Cooperates fair cooperate ∧ bots.Defects fair defect ∧
    bots.Cooperates prudent prudent ∧ bots.Cooperates prudent fair ∧
    bots.Cooperates fair prudent ∧ bots.Defects prudent defect ∧
    bots.Defects prudent cooperate ∧ bots.Defects cupod cupod ∧ bots.Defects fair cupod ∧
    bots.Cooperates cupod fair := by
  refine ⟨⟨0, fun n _ => fair_fair n⟩, ⟨0, fun n _ => (fair_twin n).1⟩,
    ⟨0, fun n _ => (fair_twin n).2⟩, ⟨0, fun n _ => fair_cooperate n⟩, ⟨1, fun n hn => ?_⟩,
    ⟨0, fun n _ => prudent_prudent n⟩, ⟨0, fun n _ => (prudent_fair n).1⟩,
    ⟨0, fun n _ => (prudent_fair n).2⟩, ⟨1, fun n hn => ?_⟩, ⟨2, fun n hn => ?_⟩,
    ⟨0, fun n _ => cupod_cupod n⟩, ⟨1, fun n hn => ?_⟩, ⟨1, fun n hn => (fair_cupod n).2.2 hn⟩⟩
  · exact Bool.eq_false_iff.2 fun h => by have := (fair_defect n).1 h; omega
  · exact Bool.eq_false_iff.2 fun h => by have := (prudent_defect n).1 h; omega
  · exact Bool.eq_false_iff.2 fun h => by have := (prudent_cooperate n).1 h; omega
  · exact Bool.eq_false_iff.2 fun h => by have := (fair_cupod n).1.1 h; omega

/-- FairBot cannot be exploited in the eventual outcomes. -/
theorem fair_unexploitable {b : Bot} (h : bots.Defects b fair) : bots.Defects fair b :=
  Universe.fair_unexploitable (U := bots) (f := fair) rfl h

end Bot

end ProgramEquilibrium
