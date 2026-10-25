/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Theory.Dupoc

/-!
# DUPOC against DIMCID

Open Problem 6 of Critch, Dennis and Russell (arXiv:2208.07006) asks for the outcome of DUPOC(k)
against DIMCID(k). DIMCID ("defect if my cooperation implies defection") defects exactly when it
finds a proof, of at most `k` characters, of its guard

  `G(k)`: if DIMCID(k) cooperates with DUPOC(k), then DUPOC(k) defects against DIMCID(k).

The answer is mutual defection for every large `k`, in any sound system with these properties
that proves the agents' code and the propositional fact `¬A → (A → B)`. The bounded Löb theorem
is applied to DIMCID's own defection `d(k)`, for a budget `f(k)` that may be anything growing
faster than `lg k` (the agents use `f(k) = k`):

* `d(k) → G(k)` is ex falso: "DIMCID cooperates" is false once DIMCID defects, and anything
  follows from it. The system has one proof of it, of some length `N`, for every positive `k`.
* Distributing the box, a proof of `d(k)` within `⌊f(k)/2⌋` characters gives a proof of `G(k)`
  within `⌊f(k)/2⌋ + C + 2N + lg k + c`, which is at most `f(k)` once `k` is large
  (`System.box_imp_of_gap`). DIMCID's code turns any such proof into `d(k)`.
* So the system proves `∀ k > k₁, □_{⌊f(k)/2⌋} d(k) → d(k)`, and `pblt` with that budget proves
  `d(k)` for every large `k` (`System.self_fulfilling`).
* DUPOC cooperates only on a checked proof that DIMCID cooperates (`Checker.not_exploited`), and
  DIMCID defects, so DUPOC defects as well (`dupoc_vs_dimcid`).

Critch, Dennis and Russell prove `outcome(CUPOD(k), DIMCID(k)) = (D, D)` (their Theorem 5.5(b))
with the bounded Löb theorem applied to the whole outcome. Applying it to DIMCID's move alone
avoids needing any short proof of DUPOC's failed search; DUPOC's move then follows from
soundness.
-/

@[expose] public section

namespace BoundedLob

namespace System

variable {f : ℕ → ℕ} (S : System f)

/-- **Self-fulfilling statements.** If `f` grows faster than `lg k` and the system proves `d → G`
and `□_f G → d` for every positive `k`, it proves `d` beyond some `k₂`. For DIMCID, `d` is its
defection and `G` its guard. -/
theorem self_fulfilling (hf : GrowsFasterThanLog f) {d G : S.Pred}
    (exfalso : S.Proves (S.above 0 (S.imp d G)))
    (code : S.Proves (S.above 0 (S.imp (S.box .f G) d))) :
    ∃ k₂, S.Proves (S.above k₂ d) := by
  -- Half the budget leaves room for the overhead of turning a proof of `d` into one of `G`.
  have gap : ∀ c, ∃ K, ∀ k > K,
      (Bound.div .f 2).eval f k + Nat.log 2 k + c ≤ Bound.f.eval f k := by
    intro c
    obtain ⟨K, hK⟩ := hf.eventually_le_div c 2 (by omega)
    refine ⟨K, fun k hk => ?_⟩
    have := hK k hk
    simp only [Bound.eval]
    omega
  obtain ⟨K, hK⟩ := S.box_imp_of_gap gap exfalso code
  exact S.pblt (.div .f 2) (hf.div 2 (by omega)) hK

end System

/-- **Open Problem 6.** DUPOC(k) against DIMCID(k) ends in mutual defection for every large `k`.

`S` is a system with Critch's properties and budget symbol `f(k) = k`, in which `defects` says
"DIMCID(k) defects against DUPOC(k)" and `guard` is DIMCID's guard. The hypotheses are what the
setting of Critch, Dennis and Russell provides: `exfalso` (a program has one output), `code`
(DIMCID defects whenever its guard has a proof of at most `k` characters), soundness of `S` for
the sentences `∀ k > k₂, defects(k)`, and DUPOC as an executable search over a sound checker. -/
theorem dupoc_vs_dimcid {G : Game} {P : Checker G} (sound : P.Sound)
    {cands : ℕ → G.Source → G.Source → List P.Proof} {dupoc : ℕ → G.Source}
    (hd : P.IsDupoc cands dupoc) (dimcid : ℕ → G.Source) (S : System id) (defects guard : S.Pred)
    (exfalso : S.Proves (S.above 0 (S.imp defects guard)))
    (code : S.Proves (S.above 0 (S.imp (S.box .f guard) defects)))
    (true_of : ∀ k₂, S.Proves (S.above k₂ defects) →
      ∀ k > k₂, G.run (dimcid k) (dupoc k) = false) :
    ∃ k₂, ∀ k > k₂, G.run (dupoc k) (dimcid k) = false ∧ G.run (dimcid k) (dupoc k) = false := by
  obtain ⟨k₂, h⟩ := S.self_fulfilling growsFasterThanLog_id exfalso code
  refine ⟨k₂, fun k hk => ⟨?_, true_of k₂ h k hk⟩⟩
  cases hrun : G.run (dupoc k) (dimcid k)
  · rfl
  · have := Checker.not_exploited sound hd hrun
    rw [true_of k₂ h k hk] at this
    exact absurd this Bool.false_ne_true

end BoundedLob
