/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Theory.Lob

/-!
# DUPOC: defect unless proof of cooperation

A `Game` is a set of program sources with a semantics: `run a b` is whether the program with
source `a` cooperates when it reads source `b`. A `Checker` supplies proofs, their sizes, the
sentence "`a` cooperates with `b`" and a decidable proof checker. The opponent is always given as
source data; nothing here compares names or consults the opponent's output.

`decision cands k me opp` is DUPOC's move: it scans the finite list of candidate proofs and
cooperates exactly when one of them has at most `k` characters and checks as a proof that
`opp` cooperates with `me`. A source `dupoc k` *is* DUPOC when running it is this decision with
its own source as `me` (`IsDupoc`); a concrete language obtains such a source from the recursion
theorem.

What is proved here:

* `not_exploited`: if checked proofs are sound, DUPOC never cooperates with an opponent that
  defects against it. No search order or budget assumption is needed.
* `decision_iff_of_complete`: if the candidate list contains every proof of at most `k`
  characters, DUPOC's move is "some affordable proof exists". `template_first` shows that putting
  a heuristic guess in front of a complete enumeration keeps this property, so such an agent
  behaves exactly like exhaustive DUPOC and inherits its theorems; the guess only changes how
  quickly the search stops.
* `template_only_misses`: a scan that only tries a template can return "defect" although an
  affordable checked proof exists, so theorems about exhaustive search do not transfer to it.
* `self_cooperation`: from the bounded Löb theorem with budget `f(k) = k`, a system that proves
  `∀ k > 0, □_k cooperates(k) → cooperates(k)` for DUPOC and is sound for the resulting sentence
  makes DUPOC cooperate with itself at every large budget (Theorem 3.7 of Critch, Dennis and
  Russell).
-/

@[expose] public section

namespace BoundedLob

/-- Program sources and what they do against each other. -/
structure Game where
  /-- Source code. -/
  Source : Type
  /-- `run a b`: the program with source `a`, reading source `b`, cooperates. -/
  run : Source → Source → Bool

/-- Proofs about a game, with sizes and a checker. -/
structure Checker (G : Game) where
  /-- Proofs. -/
  Proof : Type
  /-- The number of characters of a proof. -/
  size : Proof → ℕ
  /-- Sentences. -/
  Sentence : Type
  /-- The sentence "`a` cooperates with `b`". -/
  coop : G.Source → G.Source → Sentence
  /-- The proof checker. -/
  check : Proof → Sentence → Bool

namespace Checker

variable {G : Game} (P : Checker G)

/-- `π` is an affordable, checked proof of `σ` within budget `k`. -/
def accepts (k : ℕ) (σ : P.Sentence) (π : P.Proof) : Bool :=
  decide (P.size π ≤ k) && P.check π σ

/-- DUPOC's move with candidate proofs `cands`: cooperate exactly when some candidate is an
affordable, checked proof that `opp` cooperates with `me`. -/
def decision (cands : List P.Proof) (k : ℕ) (me opp : G.Source) : Bool :=
  cands.any (P.accepts k (P.coop opp me))

/-- `d k` is DUPOC with budget `k` and candidate generator `cands`, run on its own source. -/
def IsDupoc (cands : ℕ → G.Source → G.Source → List P.Proof) (d : ℕ → G.Source) : Prop :=
  ∀ k opp, G.run (d k) opp = P.decision (cands k (d k) opp) k (d k) opp

/-- Checked proofs of cooperation are true. -/
def Sound : Prop := ∀ π a b, P.check π (P.coop a b) = true → G.run a b = true

/-- `cands` lists every proof of at most `k` characters. -/
def Complete (cands : List P.Proof) (k : ℕ) : Prop := ∀ π, P.size π ≤ k → π ∈ cands

variable {P}

/-- DUPOC never cooperates with an opponent that defects against it. -/
theorem not_exploited (sound : P.Sound) {cands : ℕ → G.Source → G.Source → List P.Proof}
    {d : ℕ → G.Source} (hd : P.IsDupoc cands d) {k : ℕ} {opp : G.Source}
    (h : G.run (d k) opp = true) : G.run opp (d k) = true := by
  rw [hd] at h
  obtain ⟨π, -, hπ⟩ := List.any_eq_true.1 h
  simp only [accepts, Bool.and_eq_true, decide_eq_true_eq] at hπ
  exact sound π opp (d k) hπ.2

/-- With a complete candidate list, DUPOC cooperates exactly when an affordable proof exists. -/
theorem decision_iff_of_complete {cands : List P.Proof} {k : ℕ} (hc : P.Complete cands k)
    (me opp : G.Source) :
    P.decision cands k me opp = true ↔ ∃ π, P.size π ≤ k ∧ P.check π (P.coop opp me) = true := by
  simp only [decision, List.any_eq_true, accepts, Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨fun ⟨π, _, h⟩ => ⟨π, h⟩, fun ⟨π, h⟩ => ⟨π, hc π h.1, h⟩⟩

/-- A heuristic guess in front of a complete enumeration is still complete. -/
theorem template_first {template enum : List P.Proof} {k : ℕ} (hc : P.Complete enum k) :
    P.Complete (template ++ enum) k :=
  fun π h => List.mem_append_right template (hc π h)

end Checker

/-! ### A template alone is not enough

One cooperating program, and a checker with two proofs of size zero of which only the second
checks. A scan of the template `false` alone finds nothing, although the proof `true` is
affordable and checks. -/

namespace Example

/-- A game with one source, which always cooperates. -/
def missGame : Game := ⟨Unit, fun _ _ => true⟩

/-- A checker for `missGame` with two proofs, `false` and `true`, both of size zero; only `true`
checks. -/
def missChecker : Checker missGame where
  Proof := Bool
  size _ := 0
  Sentence := Unit
  coop _ _ := ()
  check π _ := π

/-- Scanning only the template `false` returns `false`, although the affordable proof `true`
checks. -/
theorem template_only_misses :
    missChecker.decision [false] 0 () () = false ∧
      ∃ π, missChecker.size π ≤ 0 ∧ missChecker.check π (missChecker.coop () ()) = true :=
  ⟨rfl, true, le_rfl, rfl⟩

end Example

/-- **Self-cooperation.** Let `S` be a proof system with budget symbol `f(k) = k`, and `coop` a
formula expressing "DUPOC with budget `k` cooperates with itself". If `S` proves
`∀ k > 0, □_k coop(k) → coop(k)`, and every sentence `∀ k > k₂, coop(k)` that `S` proves is true of
the actual programs, then DUPOC cooperates with itself at every budget beyond some `k₂`.
This is Theorem 3.7 of Critch, Dennis and Russell. -/
theorem self_cooperation {G : Game} (d : ℕ → G.Source) (S : System id) (coop : S.Pred)
    (hyp : S.Proves (S.above 0 (S.imp (S.box .f coop) coop)))
    (sound : ∀ k₂, S.Proves (S.above k₂ coop) → ∀ k > k₂, G.run (d k) (d k) = true) :
    ∃ k₂, ∀ k > k₂, G.run (d k) (d k) = true := by
  obtain ⟨k₂, h⟩ := S.pblt .f growsFasterThanLog_id hyp
  exact ⟨k₂, sound k₂ h⟩

end BoundedLob
