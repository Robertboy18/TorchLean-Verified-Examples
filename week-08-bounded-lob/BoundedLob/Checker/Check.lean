/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Checker.Encoding
public import BoundedLob.Checker.Compare

/-!
# The proof checker

`check c σ` decides whether the certificate `c` proves the sentence `σ`. The checker only reads
syntax: it never runs an agent and never evaluates truth. Its rules are those of a Hilbert system
(the axioms K, S and double negation, modus ponens), the guards of the quantifier, and the
bounded-provability rules Critch's proof uses, each with the side condition that makes it sound.
The bounded rules, and the exact syntax the checker accepts for them (`check` has every rule):

* `qd`: quantifier distribution. From a proof of `∀ k > a, φ` with `N` characters, the sentence
  `∀ k > a, □_{M + lg k} φ`, written with the bound `const M + lg`, for `N + Cost.inst ≤ M`;
* `nec`: `□_t φ` at `k = n` from a proof of `φ` at `n` with at most `t(n)` characters;
* `dist`: `□_x (φ → ψ) → □_y φ → □_z ψ` with `z` written as `x + y + const 0`;
* `four`: `□_x φ → □_{2x} □_x φ`, where the text of `x` can be paid for: beyond a guard that
  `Bound.fits` accepts, or at an instance `n` with `|x| + 2 ≤ x(n)`;
* `mono`: `□_x φ → □_y φ`, beyond a guard that `Bound.guard` gives, or at an instance where
  `x(n) ≤ y(n)`;
* `diag` and `prog`: the diagonal formula and the source of an agent.
-/

@[expose] public section

namespace BoundedLob

open Form

/-! ### The constants of the rules -/

/-- The slack of quantifier distribution: instantiating an `N`-character proof of `∀ k > a, φ`
at `k` gives a proof of `φ(k)` with `N + lg k + 3` characters, which fits in `N + Cost.inst +
lg k`. -/
def Cost.inst : ℕ := 4

/-- The expansion factor of inner necessitation: boxing a proof adds the text of the bound and
two characters, which fit in a second copy of the budget. -/
def Cost.expansion : ℕ := 2

/-- Beyond `a`, the extra text of inner necessitation can be paid for: either `2^|x| ≤ a`, so that
`|x| ≤ lg k` and the binary numeral for `k` that every proof contains already covers it, or
`Bound.guard` shows `|x| + 2 ≤ x(k)` for `k > a` (`Bound.fits_spec`). -/
def Bound.fits (x : Bound) (a : ℕ) : Bool :=
  decide (2 ^ x.size ≤ a) ||
    match (Bound.const (x.size + 2)).guard x with
    | some K => decide (K ≤ a)
    | none => false

/-- What `Bound.fits` guarantees. -/
theorem Bound.fits_spec {x : Bound} {a : ℕ} (h : x.fits a = true) {k : ℕ} (hk : a < k) :
    x.size + 2 ≤ x.eval id k ∨ x.size ≤ Nat.log 2 k := by
  simp only [Bound.fits, Bool.or_eq_true, decide_eq_true_eq] at h
  rcases h with h | h
  · exact Or.inr (Nat.le_log_of_pow_le (by decide) (by omega))
  · split at h
    · rename_i K hK
      simp only [decide_eq_true_eq] at h
      have := Bound.le_of_guard hK k (by omega)
      simp only [Bound.eval] at this
      exact Or.inl this
    · simp at h

/-- The smaller of the two guards `Bound.fits` tries (not the least guard that works). -/
def Bound.fitGuard (x : Bound) : ℕ :=
  match (Bound.const (x.size + 2)).guard x with
  | some K => min K (2 ^ x.size)
  | none => 2 ^ x.size

/-- `Bound.fits` accepts `Bound.fitGuard`. -/
theorem Bound.fits_fitGuard (x : Bound) : x.fits x.fitGuard = true := by
  unfold Bound.fits Bound.fitGuard
  split
  · simp_all
    omega
  · simp_all

/-! ### The checker -/

/-- `check c σ`: the certificate `c` proves the sentence `σ`. -/
def check : Cert → Sen → Bool
  | .axK, σ =>
    match σ.form with
    | imp φ (imp _ φ') => φ == φ'
    | _ => false
  | .axS, σ =>
    match σ.form with
    | imp (imp φ (imp ψ χ)) (imp (imp φ₁ ψ₁) (imp φ₂ χ₁)) =>
      φ == φ₁ && φ == φ₂ && ψ == ψ₁ && χ == χ₁
    | _ => false
  | .axDN, σ =>
    match σ.form with
    | imp (imp (imp φ bot) bot) φ' => φ == φ'
    | _ => false
  | .iffL c, σ =>
    match σ.form with
    | imp φ ψ => check c (σ.withForm (iff φ ψ))
    | _ => false
  | .iffR c, σ =>
    match σ.form with
    | imp ψ φ => check c (σ.withForm (iff φ ψ))
    | _ => false
  | .iffI c d, σ =>
    match σ.form with
    | iff φ ψ => check c (σ.withForm (imp φ ψ)) && check d (σ.withForm (imp ψ φ))
    | _ => false
  | .mp φ c d, σ => check c (σ.withForm (imp φ σ.form)) && check d (σ.withForm φ)
  | .raise a c, σ =>
    match σ with
    | .all b φ => decide (a ≤ b) && check c (.all a φ)
    | .inst _ _ => false
  | .inst a c, σ =>
    match σ with
    | .inst n φ => decide (a < n) && check c (.all a φ)
    | .all _ _ => false
  | .qd c, σ =>
    match σ with
    | .all a (box (.add (.const M) .lg) φ) =>
      decide (len c (.all a φ) + Cost.inst ≤ M) && check c (.all a φ)
    | _ => false
  | .nec c, σ =>
    match σ with
    | .inst n (box t φ) => decide (len c (.inst n φ) ≤ t.eval id n) && check c (.inst n φ)
    | _ => false
  | .dist, σ =>
    match σ.form with
    | imp (box x (imp φ ψ)) (imp (box y φ₁) (box z ψ₁)) =>
      φ == φ₁ && ψ == ψ₁ && z == x + y + .const 0
    | _ => false
  | .four, σ =>
    match σ with
    | .all a (imp (box x φ) (box y (box x₁ φ₁))) =>
      x == x₁ && φ == φ₁ && y == .scale Cost.expansion x && x.fits a
    | .inst n (imp (box x φ) (box y (box x₁ φ₁))) =>
      x == x₁ && φ == φ₁ && y == .scale Cost.expansion x && decide (x.size + 2 ≤ x.eval id n)
    | _ => false
  | .mono, σ =>
    match σ with
    | .all a (imp (box x φ) (box y φ₁)) =>
      φ == φ₁ && (match x.guard y with
        | some K => decide (K ≤ a)
        | none => false)
    | .inst n (imp (box x φ) (box y φ₁)) => φ == φ₁ && decide (x.eval id n ≤ y.eval id n)
    | _ => false
  | .diag, σ =>
    match σ.form with
    | iff (diag g p) (imp (box g₁ (diag g₂ p₁)) p₂) =>
      g == g₁ && g == g₂ && p == p₁ && p == p₂
    | _ => false
  | .prog, σ =>
    match σ.form with
    | iff (coop a b) φ => !a.isPlaceholder && !b.isPlaceholder && a.guarded && φ == a.fill b a
    | _ => false

/-- Provability within `t` characters. -/
def Provable (t : ℕ) (σ : Sen) : Prop := ∃ c, check c σ = true ∧ len c σ ≤ t

/-- A larger budget allows the same proofs. -/
theorem Provable.mono {s t : ℕ} {σ : Sen} (h : Provable s σ) (hst : s ≤ t) : Provable t σ := by
  obtain ⟨c, hc, hl⟩ := h
  exact ⟨c, hc, le_trans hl hst⟩

end BoundedLob
