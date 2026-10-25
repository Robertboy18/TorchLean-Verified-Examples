/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Checker.Check

/-!
# Derived rules

Certificates built from the axioms K, S and double negation and modus ponens, for the
propositional steps the proofs take: chaining implications, reordering hypotheses, ex falso,
contraposition, and the three rules of the conjunction `X ∧ Y = ¬(X → ¬Y)`. Each takes the
formulas it needs for the cut formulas of modus ponens, and proves its conclusion under whatever
guard or instance the sentence has. The conjunction rules are checked here for all `X` and `Y`;
the agents use the same certificates the theorems do.
-/

@[expose] public section

namespace BoundedLob

open Form

namespace Cert

/-- Chaining implications: from `φ → ψ` and `ψ → χ`, `φ → χ`. -/
def trans (φ ψ χ : Form) (c d : Cert) : Cert :=
  mp (imp φ ψ) (mp (imp φ (imp ψ χ)) axS (mp (imp ψ χ) axK d)) c

/-- Contracting a shared hypothesis: from `φ → ψ → χ` and `φ → ψ`, `φ → χ`. -/
def contract (φ ψ χ : Form) (c d : Cert) : Cert :=
  mp (imp φ ψ) (mp (imp φ (imp ψ χ)) axS c) d

/-- Weakening under a hypothesis: from `ψ → χ`, `(φ → ψ) → (φ → χ)`. -/
def impMono (φ ψ χ : Form) (c : Cert) : Cert :=
  mp (imp φ (imp ψ χ)) axS (mp (imp ψ χ) axK c)

/-- Reflexivity, `φ → φ`, as `S K K`. -/
def refl (φ : Form) : Cert :=
  mp (imp φ (imp φ φ)) (mp (imp φ (imp (imp φ φ) φ)) axS axK) axK

/-- Ex falso, `⊥ → φ`. -/
def efq (φ : Form) : Cert := trans bot (neg (neg φ)) φ axK axDN

/-- Exchanging two hypotheses: from `a → b → c`, `b → a → c`. -/
def swap (a b c : Form) (d : Cert) : Cert :=
  trans b (imp a b) (imp a c) axK (mp (imp a (imp b c)) axS d)

/-- From `P → γ`, `P → (ψ → γ)`: an unused hypothesis. -/
def addHyp (P γ ψ : Form) (c : Cert) : Cert := trans P γ (imp ψ γ) c axK

/-- Applying an implication, `x → (x → z) → z`. -/
def applyImp (x z : Form) : Cert := swap (imp x z) x z (refl (imp x z))

/-- Composing implications, `(q → r) → (p → q) → (p → r)`. -/
def compose (p q r : Form) : Cert :=
  trans (imp q r) (imp p (imp q r)) (imp (imp p q) (imp p r)) axK axS

/-- Contraposition: from `c : a → b`, `¬b → ¬a`. -/
def contra (a b : Form) (c : Cert) : Cert :=
  swap a (imp b bot) bot (trans a b (imp (imp b bot) bot) c (applyImp b bot))

/-- The right conjunct, `X ∧ Y → Y`. -/
def conjRight (X Y : Form) : Cert :=
  trans (Form.conj X Y) (neg (neg Y)) Y (contra (neg Y) (imp X (neg Y)) axK) axDN

/-- The left conjunct, `X ∧ Y → X`. -/
def conjLeft (X Y : Form) : Cert :=
  trans (Form.conj X Y) (neg (neg X)) X
    (contra (neg X) (imp X (neg Y)) (impMono X bot (neg Y) (efq (neg Y)))) axDN

/-- Introducing a conjunction, `X → Y → X ∧ Y`. With `H = X → ¬Y`: from `X`, `H → ¬Y`;
composing with `¬Y → ⊥`, which `Y` gives, yields `H → ⊥`. -/
def conjIntro (X Y : Form) : Cert :=
  let H := imp X (neg Y)
  let Q := imp (neg Y) bot
  let R := imp H bot
  let step₁ := trans X (imp H (neg Y)) (imp Q R) (applyImp X (neg Y))
    (swap (imp (neg Y) bot) (imp H (neg Y)) (imp H bot) (compose H (neg Y) bot))
  let step₂ := trans X (imp Q R) (imp (imp Y Q) (imp Y R)) step₁ (compose Y Q R)
  mp (imp Y Q) (swap X (imp Y Q) (imp Y R) step₂) (applyImp Y bot)

end Cert

/-- `conjRight` proves `X ∧ Y → Y`. -/
theorem check_conjRight (X Y : Form) :
    check (Cert.conjRight X Y) (.all 0 (imp (Form.conj X Y) Y)) = true := by
  simp [Cert.conjRight, Cert.contra, Cert.applyImp, Cert.swap, Cert.trans, Cert.refl, check,
    Sen.withForm, Sen.form]

/-- `conjLeft` proves `X ∧ Y → X`. -/
theorem check_conjLeft (X Y : Form) :
    check (Cert.conjLeft X Y) (.all 0 (imp (Form.conj X Y) X)) = true := by
  simp [Cert.conjLeft, Cert.contra, Cert.applyImp, Cert.swap, Cert.trans, Cert.refl,
    Cert.impMono, Cert.efq, check, Sen.withForm, Sen.form]

/-- `conjIntro` proves `X → Y → X ∧ Y`. -/
theorem check_conjIntro (X Y : Form) :
    check (Cert.conjIntro X Y) (.all 0 (imp X (imp Y (Form.conj X Y)))) = true := by
  simp [Cert.conjIntro, Cert.compose, Cert.applyImp, Cert.swap, Cert.trans, Cert.refl, check,
    Sen.withForm, Sen.form]

/-- `addHyp` adds an unused hypothesis. -/
theorem check_addHyp {a : ℕ} {P γ : Form} (ψ : Form) {c : Cert}
    (h : check c (.all a (imp P γ)) = true) :
    check (Cert.addHyp P γ ψ c) (.all a (imp P (imp ψ γ))) = true := by
  simp [Cert.addHyp, Cert.trans, check, Sen.withForm, Sen.form, h]

end BoundedLob
