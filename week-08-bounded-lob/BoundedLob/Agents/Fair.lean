/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Agents.Search
public import BoundedLob.Agents.Sources
public import BoundedLob.Checker.Calculus
public import BoundedLob.Theory.Robust
import Mathlib.Tactic.Set

/-!
# Different implementations cooperate

`System.robust` for the checker's system (`fair_cooperate`). Two agents whose sources, read in
the match, are `□_{t_A} α` and `□_{t_B} β` cooperate with each other at every large budget,
provided:

* the checker proves `α` and `β` from mutual cooperation `coop a b ∧ coop b a`;
* some budget `u(k)`, growing faster than `lg k`, stays below both `t_A` and `t_B` by `lg k`
  and any constant.

The conjunction is `¬(X → ¬Y)`, with the rules `Cert.conjLeft`, `Cert.conjRight` and
`Cert.conjIntro`. Instances: DUPOC with any two linear budgets `p·k + r` and `q·k + s` with
`p, q ≥ 1` (`dupoc_linear`), DUPOC against CIMCIC (`dupoc_cimcic`, Theorem 5.2 of Critch, Dennis
and Russell), and DUPOC against a DUPOC written with a redundant `⊤ →` (`dupoc_top`).
-/

@[expose] public section

namespace BoundedLob

open Form Agent

/-- **Robust cooperation** in the checker's system. Two guarded sources that read, in the match,
as `□_{t_A} α` and `□_{t_B} β` cooperate at every large `k`, when the checker proves `α` and `β`
from mutual cooperation and some budget `u` growing faster than `lg k` stays below `t_A` and
`t_B` by `lg k` and any constant. -/
theorem fair_cooperate {a b α β : Form} {ta tb u : Bound}
    (ha : a.isPlaceholder = false) (hb : b.isPlaceholder = false) (hga : a.guarded = true)
    (hgb : b.guarded = true) (hfa : a.fill b a = box ta α) (hfb : b.fill a b = box tb β)
    (hα : calculus.Proves (calculus.above 0 (imp (Form.conj (coop a b) (coop b a)) α)))
    (hβ : calculus.Proves (calculus.above 0 (imp (Form.conj (coop a b) (coop b a)) β)))
    (hu : GrowsFasterThanLog (u.eval id))
    (gapA : ∀ c, ∃ K, ∀ k > K, u.eval id k + Nat.log 2 k + c ≤ ta.eval id k)
    (gapB : ∀ c, ∃ K, ∀ k > K, u.eval id k + Nat.log 2 k + c ≤ tb.eval id k) :
    ∃ K, ∀ k > K, run k a b = true ∧ run k b a = true := by
  set X := coop a b
  set Y := coop b a
  have fairA := proves_of_check (a := 0) (φ := imp (box ta α) X) (.iffR .prog) (by
    simp [X, check, Sen.withForm, Sen.form, ha, hb, hga, hfa])
  have fairB := proves_of_check (a := 0) (φ := imp (box tb β) Y) (.iffR .prog) (by
    simp [Y, check, Sen.withForm, Sen.form, ha, hb, hgb, hfb])
  have hP := proves_of_check _ (check_conjIntro X Y)
  obtain ⟨K, hK⟩ := calculus.robust hu gapA gapB hα hβ hP fairA fairB
  have hX := calculus.mp (proves_of_check _ (check_conjLeft X Y)) hK
  have hY := calculus.mp (proves_of_check _ (check_conjRight X Y)) hK
  refine ⟨max 0 K, fun k hk => ⟨(run_spec k a b).2 ?_, (run_spec k b a).2 ?_⟩⟩
  · exact calculus_sound hX k hk
  · exact calculus_sound hY k hk

/-- `X ∧ Y → (ψ → X)`, for an unused hypothesis `ψ`. -/
theorem proves_conj_addHyp (X Y ψ : Form) :
    calculus.Proves (calculus.above 0 (imp (Form.conj X Y) (imp ψ X))) :=
  proves_of_check _ (check_addHyp ψ (check_conjLeft X Y))

/-- A linear budget `p·k + r` with `p ≥ 1` is at least `k`. -/
theorem le_linear {p r : ℕ} (hp : 0 < p) (k : ℕ) :
    k ≤ (Bound.add (.scale p .f) (.const r)).eval id k := by
  have := Nat.le_mul_of_pos_left k hp
  simp only [Bound.eval, id]
  omega

/-- **Budgets may differ.** For every fixed `p, q ≥ 1` and `r, s`, DUPOC with budget `p·k + r`
and DUPOC with budget `q·k + s` cooperate with each other at every large `k`. -/
theorem dupoc_linear {p q r s : ℕ} (hp : 0 < p) (hq : 0 < q) :
    ∃ K, ∀ k > K,
      run k (dupoc (.add (.scale p .f) (.const r))) (dupoc (.add (.scale q .f) (.const s))) =
          true ∧
        run k (dupoc (.add (.scale q .f) (.const s))) (dupoc (.add (.scale p .f) (.const r))) =
          true :=
  fair_cooperate (u := .div .f 2) (ha := rfl) (hb := rfl) (hga := rfl) (hgb := rfl)
    (hfa := rfl) (hfb := rfl) (proves_of_check _ (check_conjRight _ _))
    (proves_of_check _ (check_conjLeft _ _)) (growsFasterThanLog_id.div 2 (by decide))
    (gap_half (le_linear hp)) (gap_half (le_linear hq))

/-- **Theorem 5.2** of Critch, Dennis and Russell: DUPOC and CIMCIC cooperate at every large `k`.
CIMCIC's condition, "if I cooperate then you do", follows from mutual cooperation. -/
theorem dupoc_cimcic : ∃ K, ∀ k > K, run k dupoc cimcic = true ∧ run k cimcic dupoc = true :=
  fair_cooperate (α := coop cimcic dupoc) (β := imp (coop cimcic dupoc) (coop dupoc cimcic))
    (u := .div .f 2) (ha := rfl) (hb := rfl) (hga := rfl) (hgb := rfl) (hfa := rfl) (hfb := rfl)
    (proves_of_check _ (check_conjRight _ _)) (proves_conj_addHyp _ _ _)
    (growsFasterThanLog_id.div 2 (by decide)) (gap_half fun _ => le_rfl)
    (gap_half fun _ => le_rfl)

/-- DUPOC and a DUPOC whose source reads "if true, then the opponent cooperates" cooperate at
every large `k`. -/
theorem dupoc_top : ∃ K, ∀ k > K, run k dupoc dupocTop = true ∧ run k dupocTop dupoc = true :=
  fair_cooperate (α := coop dupocTop dupoc) (β := imp top (coop dupoc dupocTop))
    (u := .div .f 2) (ha := rfl) (hb := rfl) (hga := rfl) (hgb := rfl) (hfa := rfl) (hfb := rfl)
    (proves_of_check _ (check_conjRight _ _)) (proves_conj_addHyp _ _ _)
    (growsFasterThanLog_id.div 2 (by decide)) (gap_half fun _ => le_rfl)
    (gap_half fun _ => le_rfl)

end BoundedLob
