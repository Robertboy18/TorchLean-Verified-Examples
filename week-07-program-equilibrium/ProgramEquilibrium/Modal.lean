/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import Mathlib.Order.Basic

/-!
# Modal formulas on the chain of worlds

`Form α` is the language of provability logic with atoms in `α`: the connectives and a box,
read "it is provable that". `eval V n φ` evaluates `φ` at world `n` of the chain `0, 1, 2, …`,
where world `n` sees exactly the worlds below it and `V m a` is the truth value of the atom `a`
at world `m`. Each world sees a finite strict linear order, and such frames validate the
provability logic GL (which also has branching frames).

`loeb` checks Löb's axiom `□(□φ → φ) → □φ` at every world for every valuation. A formula is
`modalized` when every atom sits under a box; `eval_congr_of_modalized` says that such a formula
only looks at worlds strictly below the one where it is evaluated, which is what lets an agent's
decision at world `n` be computed from the worlds before it.
-/

@[expose] public section

namespace ProgramEquilibrium

/-- Formulas of provability logic over atoms of type `α`. -/
inductive Form (α : Type) where
  | atom : α → Form α
  | top : Form α
  | bot : Form α
  | not : Form α → Form α
  | and : Form α → Form α → Form α
  | or : Form α → Form α → Form α
  | imp : Form α → Form α → Form α
  | box : Form α → Form α
  deriving DecidableEq, Repr

namespace Form

variable {α : Type}

/-- Truth of `φ` at world `n` of the chain, where `□ψ` holds at `n` when `ψ` holds at every
world below `n`. -/
def eval (V : ℕ → α → Bool) : ℕ → Form α → Bool
  | n, atom a => V n a
  | _, top => true
  | _, bot => false
  | n, not φ => !eval V n φ
  | n, and φ ψ => eval V n φ && eval V n ψ
  | n, or φ ψ => eval V n φ || eval V n ψ
  | n, imp φ ψ => !eval V n φ || eval V n ψ
  | n, box φ => (List.range n).all fun m => eval V m φ

theorem eval_box (V : ℕ → α → Bool) (n : ℕ) (φ : Form α) :
    eval V n (box φ) = true ↔ ∀ m < n, eval V m φ = true := by
  simp [eval, List.all_eq_true, List.mem_range]

theorem eval_imp (V : ℕ → α → Bool) (n : ℕ) (φ ψ : Form α) :
    eval V n (imp φ ψ) = true ↔ (eval V n φ = true → eval V n ψ = true) := by
  cases h : eval V n φ <;> simp [eval, h]

theorem eval_not (V : ℕ → α → Bool) (n : ℕ) (φ : Form α) :
    eval V n (not φ) = true ↔ eval V n φ = false := by
  simp [eval]

/-- `□⊥` holds exactly at world `0`, the world that sees nothing. -/
theorem eval_box_bot (V : ℕ → α → Bool) (n : ℕ) : eval V n (box bot) = true ↔ n = 0 := by
  rw [eval_box]
  constructor
  · intro h
    by_contra hn
    simpa [eval] using h 0 (Nat.pos_of_ne_zero hn)
  · rintro rfl m hm
    omega

/-- Löb's axiom `□(□φ → φ) → □φ` holds at every world of the chain, for every valuation. -/
theorem loeb (V : ℕ → α → Bool) (n : ℕ) (φ : Form α) :
    eval V n (imp (box (imp (box φ) φ)) (box φ)) = true := by
  rw [eval_imp, eval_box, eval_box]
  intro h m hm
  induction m using Nat.strongRecOn with
  | _ m ih =>
    have step := (eval_imp V m (box φ) φ).1 (h m hm)
    exact step ((eval_box V m φ).2 fun k hk => ih k hk (by omega))

/-- Every atom of `φ` occurs under a box. -/
def modalized : Form α → Bool
  | atom _ => false
  | top => true
  | bot => true
  | not φ => modalized φ
  | and φ ψ => modalized φ && modalized ψ
  | or φ ψ => modalized φ && modalized ψ
  | imp φ ψ => modalized φ && modalized ψ
  | box _ => true

/-- Rename the atoms of a formula. -/
def map {β : Type} (f : α → β) : Form α → Form β
  | atom a => atom (f a)
  | top => top
  | bot => bot
  | not φ => not (map f φ)
  | and φ ψ => and (map f φ) (map f ψ)
  | or φ ψ => or (map f φ) (map f ψ)
  | imp φ ψ => imp (map f φ) (map f ψ)
  | box φ => box (map f φ)

@[simp] theorem modalized_map {β : Type} (f : α → β) (φ : Form α) :
    (map f φ).modalized = φ.modalized := by
  induction φ <;> simp_all [map, modalized]

/-- The value at world `k` depends only on the valuation at worlds up to `k`. -/
theorem eval_congr_le {V W : ℕ → α → Bool} {k : ℕ} (h : ∀ m ≤ k, ∀ a, V m a = W m a)
    (φ : Form α) : eval V k φ = eval W k φ := by
  induction φ generalizing k with
  | atom a => exact h k (Nat.le_refl k) a
  | top | bot => rfl
  | not φ ih => simp [eval, ih h]
  | and φ ψ ihφ ihψ => simp [eval, ihφ h, ihψ h]
  | or φ ψ ihφ ihψ => simp [eval, ihφ h, ihψ h]
  | imp φ ψ ihφ ihψ => simp [eval, ihφ h, ihψ h]
  | box φ ih =>
    have agree : ∀ m < k, eval V m φ = eval W m φ := fun m hm =>
      ih fun j hj a => h j (by omega) a
    apply Bool.eq_iff_iff.2
    rw [eval_box, eval_box]
    exact forall₂_congr fun m hm => by rw [agree m hm]

/-- A modalized formula at world `n` depends only on the valuation at worlds below `n`. -/
theorem eval_congr_of_modalized {V W : ℕ → α → Bool} {n : ℕ}
    (h : ∀ m < n, ∀ a, V m a = W m a) {φ : Form α} (hφ : modalized φ = true) :
    eval V n φ = eval W n φ := by
  induction φ with
  | atom a => simp [modalized] at hφ
  | top | bot => rfl
  | not φ ih =>
    simp only [modalized] at hφ
    simp [eval, ih hφ]
  | and φ ψ ihφ ihψ =>
    simp only [modalized, Bool.and_eq_true] at hφ
    simp [eval, ihφ hφ.1, ihψ hφ.2]
  | or φ ψ ihφ ihψ =>
    simp only [modalized, Bool.and_eq_true] at hφ
    simp [eval, ihφ hφ.1, ihψ hφ.2]
  | imp φ ψ ihφ ihψ =>
    simp only [modalized, Bool.and_eq_true] at hφ
    simp [eval, ihφ hφ.1, ihψ hφ.2]
  | box φ =>
    apply Bool.eq_iff_iff.2
    rw [eval_box, eval_box]
    exact forall₂_congr fun m hm => by
      rw [eval_congr_le (fun j hj a => h j (by omega) a) φ]

/-- Under any valuation, a boxed formula changes value at most once, from true to false: either
its body holds at every world, or a world where it fails falsifies the box above it. -/
theorem box_stabilizes (V : ℕ → α → Bool) (φ : Form α) :
    ∃ K, ∀ n ≥ K, eval V n (box φ) = eval V K (box φ) := by
  by_cases h : ∀ m, eval V m φ = true
  · refine ⟨0, fun n _ => ?_⟩
    rw [(eval_box V n φ).2 fun m _ => h m, (eval_box V 0 φ).2 fun m _ => h m]
  · obtain ⟨m, hm⟩ := Classical.not_forall.1 h
    have low : ∀ k > m, eval V k (box φ) = false := fun k hk =>
      Bool.eq_false_iff.2 fun hb => hm ((eval_box V k φ).1 hb m hk)
    exact ⟨m + 1, fun n hn => by rw [low n (by omega), low (m + 1) (by omega)]⟩

/-- Under any valuation, a modalized formula is eventually constant along the chain. -/
theorem stabilizes_of_modalized (V : ℕ → α → Bool) {φ : Form α} (hφ : modalized φ = true) :
    ∃ K, ∀ n ≥ K, eval V n φ = eval V K φ := by
  have pair : ∀ {φ ψ : Form α} (f : Bool → Bool → Bool),
      (∃ K, ∀ n ≥ K, eval V n φ = eval V K φ) → (∃ K, ∀ n ≥ K, eval V n ψ = eval V K ψ) →
      ∃ K, ∀ n ≥ K, f (eval V n φ) (eval V n ψ) = f (eval V K φ) (eval V K ψ) := by
    rintro φ ψ f ⟨K₁, h₁⟩ ⟨K₂, h₂⟩
    refine ⟨max K₁ K₂, fun n hn => ?_⟩
    rw [h₁ n (by omega), h₂ n (by omega), h₁ (max K₁ K₂) (by omega), h₂ (max K₁ K₂) (by omega)]
  induction φ with
  | atom a => simp [modalized] at hφ
  | top | bot => exact ⟨0, fun _ _ => rfl⟩
  | not φ ih =>
    simp only [modalized] at hφ
    obtain ⟨K, hK⟩ := ih hφ
    exact ⟨K, fun n hn => by simp [eval, hK n hn]⟩
  | and φ ψ ihφ ihψ =>
    simp only [modalized, Bool.and_eq_true] at hφ
    exact pair (· && ·) (ihφ hφ.1) (ihψ hφ.2)
  | or φ ψ ihφ ihψ =>
    simp only [modalized, Bool.and_eq_true] at hφ
    exact pair (· || ·) (ihφ hφ.1) (ihψ hφ.2)
  | imp φ ψ ihφ ihψ =>
    simp only [modalized, Bool.and_eq_true] at hφ
    exact pair (fun x y => !x || y) (ihφ hφ.1) (ihψ hφ.2)
  | box φ => exact box_stabilizes V φ

/-! ### Löb's axiom on other frames

`holdsOn R V w φ` reads `φ` at world `w` of an arbitrary frame, where `□ψ` holds at `w` when `ψ`
holds at every `v` with `R v w` (`w` sees `v`). Löb's axiom holds on every transitive frame whose
relation is well founded, and `loeb_needs_transitivity` shows that transitivity cannot be dropped.
-/

/-- Truth on an arbitrary frame, where `R v w` means that `w` sees `v`. -/
def holdsOn {W : Type} (R : W → W → Prop) (V : W → α → Prop) : W → Form α → Prop
  | w, atom a => V w a
  | _, top => True
  | _, bot => False
  | w, not φ => ¬ holdsOn R V w φ
  | w, and φ ψ => holdsOn R V w φ ∧ holdsOn R V w ψ
  | w, or φ ψ => holdsOn R V w φ ∨ holdsOn R V w ψ
  | w, imp φ ψ => holdsOn R V w φ → holdsOn R V w ψ
  | w, box φ => ∀ v, R v w → holdsOn R V v φ

/-- Löb's axiom holds on every transitive, well-founded frame. -/
theorem loeb_of_wellFounded {W : Type} (R : W → W → Prop) (trans : ∀ u v w, R u v → R v w → R u w)
    (wf : WellFounded R) (V : W → α → Prop) (w : W) (φ : Form α) :
    holdsOn R V w (imp (box (imp (box φ) φ)) (box φ)) := by
  intro h v hv
  induction v using wf.induction with
  | _ v ih => exact h v hv fun u hu => ih u hu (trans u v w hu hv)

/-- Without transitivity Löb's axiom can fail: world `2` sees `1`, `1` sees `0`, but `2` does not
see `0`, and the atom is false at `0` and `1`. -/
theorem loeb_needs_transitivity :
    ¬ holdsOn (fun v w : Fin 3 => w.val = v.val + 1) (fun w (_ : Unit) => w.val = 2) 2
      (imp (box (imp (box (atom ())) (atom ()))) (box (atom ()))) := by
  intro h
  have step : ∀ v : Fin 3, (2 : Fin 3).val = v.val + 1 →
      (∀ u : Fin 3, v.val = u.val + 1 → u.val = 2) → v.val = 2 := by
    intro v hv hbox
    have hv' : v.val = 1 := by
      have h2 : (2 : Fin 3).val = 2 := rfl
      omega
    have hv1 : v = 1 := Fin.ext hv'
    subst hv1
    exact absurd (hbox 0 rfl) (by decide)
  have h1 : (1 : Fin 3).val = 2 := h step 1 rfl
  exact absurd h1 (by decide)

end Form

end ProgramEquilibrium
