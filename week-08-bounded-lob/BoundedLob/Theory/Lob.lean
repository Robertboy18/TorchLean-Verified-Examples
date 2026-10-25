/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Theory.System
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Set

/-!
# The parametric bounded Löb theorem

`System.pblt` is Lemma 3.6 of Critch, Dennis and Russell (arXiv:2208.07006), the case of
Critch's Theorem 3 (arXiv:1602.04184v5) with a linear proof-expansion function. If the system
proves

  `∀ k > k₁, □_{b(k)} p(k) → p(k)`

for a bound `b` such that `b(k) / lg k` tends to infinity, then for some `k₂` it proves
`∀ k > k₂, p(k)`. DUPOC uses `b(k) = k`; other agents need smaller budgets such as `⌊k/2⌋`.

The growth hypothesis is `GrowsFasterThanLog`: for every constant `c`, eventually
`c · (lg k + 1) < b(k)`. Notation 3.5 of Critch, Dennis and Russell states `f(k) ≻ O(lg k)` with
*some* constant `c`; the theorem they cite uses every `c`, and so does this proof, where the
length of the proof built from the diagonal lemma is only known once the lemma has been applied.

The proof follows Critch's. Take `g = ⌊b / D⌋` with `D = 2(e + 1)` and `expanded = e · g`. Inner
necessitation lands exactly on `expanded`, and since `(e + 1) · g ≤ b / 2`, the two uses of the
smaller budget leave half of `b` for the logarithmic overheads. The diagonal lemma gives
`ψ(k) ↔ (□_{g(k)} ψ(k) → p(k))`. Distributing the bounded box over that equivalence and then
over its consequent, and spending the budget `b(k)` through the hypothesis, yields
`□_{g(k)} ψ(k) → p(k)`; that implication is `ψ(k)` itself, and one more quantifier distribution
shows `□_{g(k)} ψ(k)` for large `k`.
-/

@[expose] public section

namespace BoundedLob

/-! ### Budgets that grow faster than the logarithm -/

/-- `f(k) / lg k` tends to infinity: for every `c`, eventually `c · (lg k + 1) < f(k)`. -/
def GrowsFasterThanLog (f : ℕ → ℕ) : Prop :=
  ∀ c : ℕ, ∃ K, ∀ k > K, c * (Nat.log 2 k + 1) < f k

/-- Any fixed overhead plus `lg k` eventually fits in `f(k) / D`. -/
theorem GrowsFasterThanLog.eventually_le_div {f : ℕ → ℕ} (hf : GrowsFasterThanLog f) (A D : ℕ)
    (hD : 0 < D) : ∃ K, ∀ k > K, A + Nat.log 2 k ≤ f k / D := by
  obtain ⟨K, hK⟩ := hf (D * (A + 1))
  refine ⟨K, fun k hk => ?_⟩
  have big := hK k hk
  rw [Nat.le_div_iff_mul_le hD]
  have : (A + Nat.log 2 k) * D ≤ D * (A + 1) * (Nat.log 2 k + 1) := by
    have : A + Nat.log 2 k ≤ (A + 1) * (Nat.log 2 k + 1) := by nlinarith
    calc (A + Nat.log 2 k) * D ≤ (A + 1) * (Nat.log 2 k + 1) * D := Nat.mul_le_mul_right D this
      _ = D * (A + 1) * (Nat.log 2 k + 1) := by ring
  omega

/-- Dividing the budget by a constant keeps it ahead of every multiple of `lg k`. -/
theorem GrowsFasterThanLog.div {f : ℕ → ℕ} (hf : GrowsFasterThanLog f) (D : ℕ) (hD : 0 < D) :
    GrowsFasterThanLog (fun k => f k / D) := by
  intro c
  obtain ⟨K, hK⟩ := hf (D * (c + 1))
  refine ⟨K, fun k hk => ?_⟩
  have big := hK k hk
  set L := Nat.log 2 k + 1
  have hL : 1 ≤ L := by omega
  have : (c * L + 1) * D ≤ f k := by nlinarith
  exact Nat.lt_of_lt_of_le (Nat.lt_succ_self _) ((Nat.le_div_iff_mul_le hD).2 this)

/-- The budget `f(k) = k` grows faster than the logarithm. -/
theorem growsFasterThanLog_id : GrowsFasterThanLog id := by
  intro c
  -- `(m + 1)^2 < 2^m` from `m = 6` on, the first `m` where it holds.
  have sq : ∀ m ≥ 6, (m + 1) * (m + 1) < 2 ^ m := by
    intro m hm
    induction m, hm using Nat.le_induction with
    | base => decide
    | succ m hm ih =>
      have : 2 * m + 3 ≤ 2 ^ m := by
        have : m < 2 ^ m := Nat.lt_two_pow_self
        nlinarith
      rw [pow_succ]
      nlinarith
  set M := max c 6
  refine ⟨2 ^ M, fun k hk => ?_⟩
  have hpos : 0 < k := lt_of_le_of_lt (Nat.zero_le _) hk
  have hM : M ≤ Nat.log 2 k := Nat.le_log_of_pow_le (by norm_num) (le_of_lt hk)
  have hk2 : 2 ^ Nat.log 2 k ≤ k := Nat.pow_log_le_self 2 (Nat.pos_iff_ne_zero.mp hpos)
  have hsq := sq (Nat.log 2 k) (le_trans (le_max_right c 6) hM)
  have hc : c ≤ Nat.log 2 k + 1 := by have := le_max_left c 6; omega
  calc c * (Nat.log 2 k + 1) ≤ (Nat.log 2 k + 1) * (Nat.log 2 k + 1) :=
        Nat.mul_le_mul_right _ hc
    _ < 2 ^ Nat.log 2 k := hsq
    _ ≤ k := hk2

/-- `⌊k/2⌋ + lg k + c ≤ t(k)` from some point on, for any budget with `t(k) ≥ k`. -/
theorem gap_half {t : Bound} (ht : ∀ k, k ≤ t.eval id k) :
    ∀ c, ∃ K, ∀ k > K, (Bound.div .f 2).eval id k + Nat.log 2 k + c ≤ t.eval id k := by
  intro c
  obtain ⟨K, hK⟩ := growsFasterThanLog_id.eventually_le_div c 2 (by decide)
  refine ⟨K, fun k hk => ?_⟩
  have := hK k hk
  have := ht k
  simp only [Bound.eval, id] at *
  omega

/-! ### The theorem -/

namespace System

variable {f : ℕ → ℕ} (S : System f)

/-- **Parametric bounded Löb theorem** (Lemma 3.6 of Critch, Dennis and Russell): if the system
proves `∀ k > k₁, □_{b(k)} p(k) → p(k)` and `b` grows faster than `lg k`, it proves
`∀ k > k₂, p(k)` for some `k₂`. -/
theorem pblt (b : Bound) (grows : GrowsFasterThanLog (b.eval f)) {k₁ : ℕ} {p : S.Pred}
    (h : S.Proves (S.above k₁ (S.imp (S.box b p) p))) : ∃ k₂, S.Proves (S.above k₂ p) := by
  -- The two smaller budgets of Critch's proof.
  set D := 2 * (S.expansion + 1) with hD
  have hDpos : 0 < D := by omega
  set g : Bound := .div b D
  set expanded : Bound := .scale S.expansion g
  -- ψ(k) ↔ (□_{g(k)} ψ(k) → p(k)), proved with `n` characters.
  obtain ⟨ψ, n, hdiag⟩ := S.diag g p
  set G := S.imp (S.box g ψ) p
  -- Distribute the box over the forward implication `ψ → G`.
  set ℓ : Bound := .const (S.instCost + 2 * (n + S.iffCost)) + .lg
  have h1 : S.Proves (S.above 0 (S.box ℓ (S.imp ψ G))) := S.quant_dist (S.iff_forward hdiag)
  set x₁ : Bound := ℓ + g + .const S.distCost
  have h2 : S.Proves (S.above 0 (S.imp (S.box g ψ) (S.box x₁ G))) := by
    simpa only [max_self, Proves] using S.mp (S.imp_dist ℓ g ψ G) h1
  -- Distribute over the consequent of `G = □_g ψ → p`.
  set x₂ : Bound := x₁ + expanded + .const S.distCost
  have h3 : S.Proves (S.above 0
      (S.imp (S.box g ψ) (S.imp (S.box expanded (S.box g ψ)) (S.box x₂ p)))) := by
    simpa only [max_self, Proves] using S.trans h2 (S.imp_dist x₁ expanded (S.box g ψ) p)
  -- The budget `x₂(k)` eventually fits in `b(k)`, with room to spare.
  have fits : ∀ c, ∃ K, ∀ k > K, x₂.eval f k + c ≤ b.eval f k := by
    intro c
    obtain ⟨ka, hka⟩ :=
      grows.eventually_le_div (S.instCost + 2 * (n + S.iffCost) + 2 * S.distCost + c) 2
        (by omega)
    refine ⟨ka, fun k hk => ?_⟩
    have half := hka k hk
    have share : (S.expansion + 1) * (b.eval f k / D) ≤ b.eval f k / 2 := by
      rw [Nat.le_div_iff_mul_le (by omega)]
      calc (S.expansion + 1) * (b.eval f k / D) * 2 = b.eval f k / D * D := by rw [hD]; ring
        _ ≤ b.eval f k := Nat.div_mul_le_self (b.eval f k) D
    have split : (S.expansion + 1) * (b.eval f k / D) =
        b.eval f k / D + S.expansion * (b.eval f k / D) := by ring
    simp only [x₂, x₁, ℓ, expanded, g, Bound.eval]
    omega
  -- Spend the budget through the hypothesis: (5.4) in Critch's proof.
  obtain ⟨ka, hmono⟩ := S.mono p fits
  have h4 : S.Proves (S.above (max ka k₁) (S.imp (S.box x₂ p) p)) := S.trans hmono h
  have h5 : S.Proves (S.above (max ka k₁)
      (S.imp (S.box g ψ) (S.imp (S.box expanded (S.box g ψ)) p))) := by
    simpa only [Nat.zero_max, Proves] using S.trans h3 (S.imp_mono h4)
  -- Inner necessitation lands exactly on `expanded = e · g`: (5.5), then (5.6).
  obtain ⟨a₄, hnec⟩ := S.inner_nec g ψ
  set K := max (max ka k₁) a₄
  have hK : S.Proves (S.above K (S.imp (S.box g ψ) p)) := S.contract h5 hnec
  -- That implication is `ψ(k)` itself.
  obtain ⟨N, hN⟩ : S.Proves (S.above K ψ) := by
    simpa only [Nat.zero_max, Proves] using S.mp (S.iff_backward hdiag) hK
  -- One more distribution, and `instCost + 2N + lg k` eventually fits in `g(k)`: (5.7).
  have fits' : ∀ c, ∃ K, ∀ k > K,
      (Bound.const (S.instCost + 2 * N) + .lg).eval f k + c ≤ g.eval f k := by
    intro c
    obtain ⟨kb, hkb⟩ := grows.eventually_le_div (S.instCost + 2 * N + c) D hDpos
    refine ⟨kb, fun k hk => ?_⟩
    have := hkb k hk
    simp only [g, Bound.eval]
    omega
  obtain ⟨kb, hmono'⟩ := S.mono ψ fits'
  have h7 : S.Proves (S.above (max kb K) (S.box g ψ)) := S.mp hmono' (S.quant_dist hN)
  exact ⟨max K (max kb K), S.mp hK h7⟩

/-- **Moving a proof to a smaller budget.** If `P → α` is provable, `□_t α → X` is provable, and
some budget `u` stays below `t` by `lg k` and any constant, then `□_u P → X` is provable beyond a
guard: a proof of `P` within `u(k)` is, a few characters later, a proof of `α` within `t(k)`. -/
theorem box_imp_of_gap {P α X : S.Pred} {t u : Bound}
    (gap : ∀ c, ∃ K, ∀ k > K, u.eval f k + Nat.log 2 k + c ≤ t.eval f k)
    (hα : S.Proves (S.above 0 (S.imp P α))) (hX : S.Proves (S.above 0 (S.imp (S.box t α) X))) :
    ∃ K, S.Proves (S.above K (S.imp (S.box u P) X)) := by
  obtain ⟨N, hN⟩ := hα
  set ℓ : Bound := .const (S.instCost + 2 * N) + .lg
  have h₁ : S.Proves (S.above 0 (S.imp (S.box u P) (S.box (ℓ + u + .const S.distCost) α))) := by
    simpa only [max_self, Proves] using S.mp (S.imp_dist ℓ u P α) (S.quant_dist hN)
  have fits : ∀ c, ∃ K, ∀ k > K, (ℓ + u + .const S.distCost).eval f k + c ≤ t.eval f k := by
    intro c
    obtain ⟨K, hK⟩ := gap (S.instCost + 2 * N + S.distCost + c)
    refine ⟨K, fun k hk => ?_⟩
    have := hK k hk
    simp only [ℓ, Bound.eval]
    omega
  obtain ⟨K, hmono⟩ := S.mono α fits
  refine ⟨K, ?_⟩
  simpa only [Nat.zero_max, Nat.max_zero, Proves] using S.trans (S.trans h₁ hmono) hX

end System

end BoundedLob
