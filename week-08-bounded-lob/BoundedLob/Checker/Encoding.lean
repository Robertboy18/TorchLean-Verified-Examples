/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Checker.Form

/-!
# Proofs as text

Formulas, sentences and certificates are printed as strings over a fixed alphabet of 35
characters, with numerals in binary. Every length in the development is the length of that text:
a proof of a sentence has as many characters as the sentence plus its certificate (`len`). A
numeral `n` takes `⌊log₂ n⌋ + 2` characters, so writing `k` costs about `lg k`.
-/

@[expose] public section

namespace BoundedLob

/-! ### Numerals -/

/-- Binary digits of `n`, most significant first, with `fuel` halvings available. -/
def natDigitsAux : ℕ → ℕ → List Char
  | 0, _ => []
  | fuel + 1, n =>
    if n < 2 then [if n = 0 then '0' else '1']
    else natDigitsAux fuel (n / 2) ++ [if n % 2 = 0 then '0' else '1']

/-- Binary digits of `n`, most significant first. Structural recursion on a fuel of `n + 1`, so
that the kernel can compute it. -/
def natDigits (n : ℕ) : List Char := natDigitsAux (n + 1) n

/-- Enough fuel gives the same digits. -/
theorem natDigitsAux_fuel : ∀ (n f : ℕ), n < f → natDigitsAux f n = natDigitsAux (n + 1) n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro f hf
    obtain ⟨g, rfl⟩ : ∃ g, f = g + 1 := ⟨f - 1, by omega⟩
    simp only [natDigitsAux]
    split
    · rfl
    · rw [ih (n / 2) (by omega) g (by omega), ih (n / 2) (by omega) n (by omega)]

/-- The digit of `0` or `1`. -/
theorem natDigits_of_lt {n : ℕ} (h : n < 2) : natDigits n = [if n = 0 then '0' else '1'] := by
  simp [natDigits, natDigitsAux, h]

/-- The digits of `n ≥ 2` are those of `n / 2`, then the last bit. -/
theorem natDigits_of_two_le {n : ℕ} (h : 2 ≤ n) :
    natDigits n = natDigits (n / 2) ++ [if n % 2 = 0 then '0' else '1'] := by
  have h' : ¬ n < 2 := by omega
  conv_lhs => rw [natDigits, natDigitsAux]; simp only [h', ↓reduceIte]
  rw [natDigitsAux_fuel (n / 2) n (by omega)]
  rfl

/-! ### Text -/

/-- A numeral: binary digits, then `.`. -/
def natText (n : ℕ) : List Char := natDigits n ++ ['.']

/-- The text of a bound. -/
def Bound.text : Bound → List Char
  | .const n => 'n' :: natText n
  | .lg => ['l']
  | .f => ['k']
  | .add a b => '+' :: (a.text ++ b.text)
  | .scale c a => '*' :: (natText c ++ a.text)
  | .div a d => '/' :: (a.text ++ natText d)

/-- The text of a formula. -/
def Form.text : Form → List Char
  | .bot => ['F']
  | .imp φ ψ => '>' :: (φ.text ++ ψ.text)
  | .iff φ ψ => '=' :: (φ.text ++ ψ.text)
  | .box t φ => '#' :: (t.text ++ φ.text)
  | .coop a b => 'c' :: (a.text ++ b.text)
  | .opp => ['o']
  | .me => ['m']
  | .diag g p => 'd' :: (g.text ++ p.text)

/-- The text of a sentence. -/
def Sen.text : Sen → List Char
  | .all a φ => 'A' :: (natText a ++ φ.text)
  | .inst n φ => 'I' :: (natText n ++ φ.text)

/-- The text of a certificate. -/
def Cert.text : Cert → List Char
  | .axK => ['K']
  | .axS => ['S']
  | .axDN => ['N']
  | .iffL c => 'L' :: c.text
  | .iffR c => 'R' :: c.text
  | .iffI c d => 'B' :: (c.text ++ d.text)
  | .mp φ c d => 'M' :: (φ.text ++ c.text ++ d.text)
  | .raise a c => 'U' :: (natText a ++ c.text)
  | .inst a c => 'T' :: (natText a ++ c.text)
  | .qd c => 'Q' :: c.text
  | .nec c => 'X' :: c.text
  | .dist => ['P']
  | .four => ['4']
  | .mono => ['O']
  | .diag => ['G']
  | .prog => ['Y']

/-- The characters all texts are written in. -/
def alphabet : List Char :=
  ['F', '>', '=', '#', 'c', 'o', 'm', 'd', 'n', 'l', 'k', '+', '*', '/', '0', '1', '.',
   'A', 'I', 'K', 'S', 'N', 'L', 'R', 'B', 'M', 'U', 'T', 'Q', 'X', 'P', '4', 'O', 'G', 'Y']

/-- Number of characters. -/
def Bound.size (t : Bound) : ℕ := t.text.length

/-- Number of characters. -/
def Form.size (φ : Form) : ℕ := φ.text.length

/-- Number of characters. -/
def Sen.size (σ : Sen) : ℕ := σ.text.length

/-- Number of characters. -/
def Cert.size (c : Cert) : ℕ := c.text.length

/-- The length of the proof of `σ` with certificate `c`: the sentence and the certificate. -/
def len (c : Cert) (σ : Sen) : ℕ := σ.size + c.size

/-! ### Lengths

Every length below is the length of the text. -/

/-- `n` has `⌊log₂ n⌋ + 1` binary digits. -/
theorem natDigits_length (n : ℕ) : (natDigits n).length = Nat.log 2 n + 1 := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    by_cases h : n < 2
    · have : Nat.log 2 n = 0 := Nat.log_of_lt (by omega)
      simp [natDigits_of_lt h, this]
    · have h2 : Nat.log 2 n = Nat.log 2 (n / 2) + 1 :=
        Nat.log_of_one_lt_of_le (by decide) (by omega)
      rw [natDigits_of_two_le (by omega), List.length_append, ih (n / 2) (by omega), h2]
      simp

/-- A numeral takes `⌊log₂ n⌋ + 2` characters. -/
@[simp] theorem natText_length (n : ℕ) : (natText n).length = Nat.log 2 n + 2 := by
  simp [natText, natDigits_length]

/-- The length of the text of `Bound.const`. -/
@[simp] theorem Bound.size_const (n : ℕ) : (Bound.const n).size = Nat.log 2 n + 3 := by
  simp [Bound.size, Bound.text]
/-- The length of the text of `Bound.lg`. -/
@[simp] theorem Bound.size_lg : Bound.lg.size = 1 := rfl
/-- The length of the text of `Bound.f`. -/
@[simp] theorem Bound.size_f : Bound.f.size = 1 := rfl
/-- The length of the text of `Bound.add`. -/
@[simp] theorem Bound.size_add (a b : Bound) : (a + b).size = a.size + b.size + 1 := by
  change (a.add b).size = _
  simp [Bound.size, Bound.text]
/-- The length of the text of `Bound.scale`. -/
@[simp] theorem Bound.size_scale (c : ℕ) (a : Bound) :
    (Bound.scale c a).size = Nat.log 2 c + a.size + 3 := by
  simp [Bound.size, Bound.text]; omega
/-- The length of the text of `Bound.div`. -/
@[simp] theorem Bound.size_div (a : Bound) (d : ℕ) :
    (Bound.div a d).size = a.size + Nat.log 2 d + 3 := by
  simp [Bound.size, Bound.text]; omega

/-- The length of the text of `Form.bot`. -/
@[simp] theorem Form.size_bot : Form.bot.size = 1 := rfl
/-- The length of the text of `Form.opp`. -/
@[simp] theorem Form.size_opp : Form.opp.size = 1 := rfl
/-- The length of the text of `Form.me`. -/
@[simp] theorem Form.size_me : Form.me.size = 1 := rfl
/-- The length of the text of `Form.imp`. -/
@[simp] theorem Form.size_imp (φ ψ : Form) : (Form.imp φ ψ).size = φ.size + ψ.size + 1 := by
  simp [Form.size, Form.text]
/-- The length of the text of `Form.iff`. -/
@[simp] theorem Form.size_iff (φ ψ : Form) : (Form.iff φ ψ).size = φ.size + ψ.size + 1 := by
  simp [Form.size, Form.text]
/-- The length of the text of `Form.box`. -/
@[simp] theorem Form.size_box (t : Bound) (φ : Form) :
    (Form.box t φ).size = t.size + φ.size + 1 := by
  simp [Form.size, Form.text, Bound.size]
/-- The length of the text of `Form.coop`. -/
@[simp] theorem Form.size_coop (a b : Form) : (Form.coop a b).size = a.size + b.size + 1 := by
  simp [Form.size, Form.text]
/-- The length of the text of `Form.diag`. -/
@[simp] theorem Form.size_diag (g : Bound) (p : Form) :
    (Form.diag g p).size = g.size + p.size + 1 := by
  simp [Form.size, Form.text, Bound.size]

/-- The length of the text of `Sen.all`. -/
@[simp] theorem Sen.size_all (a : ℕ) (φ : Form) :
    (Sen.all a φ).size = Nat.log 2 a + φ.size + 3 := by
  simp [Sen.size, Sen.text, Form.size]; omega
/-- The length of the text of `Sen.inst`. -/
@[simp] theorem Sen.size_inst (n : ℕ) (φ : Form) :
    (Sen.inst n φ).size = Nat.log 2 n + φ.size + 3 := by
  simp [Sen.size, Sen.text, Form.size]; omega

/-- The length of the text of `Cert.axK`. -/
@[simp] theorem Cert.size_axK : Cert.axK.size = 1 := rfl
/-- The length of the text of `Cert.axS`. -/
@[simp] theorem Cert.size_axS : Cert.axS.size = 1 := rfl
/-- The length of the text of `Cert.axDN`. -/
@[simp] theorem Cert.size_axDN : Cert.axDN.size = 1 := rfl
/-- The length of the text of `Cert.dist`. -/
@[simp] theorem Cert.size_dist : Cert.dist.size = 1 := rfl
/-- The length of the text of `Cert.four`. -/
@[simp] theorem Cert.size_four : Cert.four.size = 1 := rfl
/-- The length of the text of `Cert.mono`. -/
@[simp] theorem Cert.size_mono : Cert.mono.size = 1 := rfl
/-- The length of the text of `Cert.diag`. -/
@[simp] theorem Cert.size_diag : Cert.diag.size = 1 := rfl
/-- The length of the text of `Cert.prog`. -/
@[simp] theorem Cert.size_prog : Cert.prog.size = 1 := rfl
/-- The length of the text of `Cert.iffL`. -/
@[simp] theorem Cert.size_iffL (c : Cert) : (Cert.iffL c).size = c.size + 1 := by
  simp [Cert.size, Cert.text]
/-- The length of the text of `Cert.iffR`. -/
@[simp] theorem Cert.size_iffR (c : Cert) : (Cert.iffR c).size = c.size + 1 := by
  simp [Cert.size, Cert.text]
/-- The length of the text of `Cert.iffI`. -/
@[simp] theorem Cert.size_iffI (c d : Cert) : (Cert.iffI c d).size = c.size + d.size + 1 := by
  simp [Cert.size, Cert.text]
/-- The length of the text of `Cert.mp`. -/
@[simp] theorem Cert.size_mp (φ : Form) (c d : Cert) :
    (Cert.mp φ c d).size = φ.size + c.size + d.size + 1 := by
  simp [Cert.size, Cert.text, Form.size]; omega
/-- The length of the text of `Cert.raise`. -/
@[simp] theorem Cert.size_raise (a : ℕ) (c : Cert) :
    (Cert.raise a c).size = Nat.log 2 a + c.size + 3 := by
  simp [Cert.size, Cert.text]; omega
/-- The length of the text of `Cert.inst`. -/
@[simp] theorem Cert.size_inst (a : ℕ) (c : Cert) :
    (Cert.inst a c).size = Nat.log 2 a + c.size + 3 := by
  simp [Cert.size, Cert.text]; omega
/-- The length of the text of `Cert.qd`. -/
@[simp] theorem Cert.size_qd (c : Cert) : (Cert.qd c).size = c.size + 1 := by
  simp [Cert.size, Cert.text]
/-- The length of the text of `Cert.nec`. -/
@[simp] theorem Cert.size_nec (c : Cert) : (Cert.nec c).size = c.size + 1 := by
  simp [Cert.size, Cert.text]

/-- Every formula has at least one character. -/
theorem Form.size_pos (φ : Form) : 0 < φ.size := by cases φ <;> simp

/-- Every certificate has at least one character. -/
theorem Cert.size_pos (c : Cert) : 0 < c.size := by cases c <;> simp

/-- The length of a proof of an instance. -/
theorem len_inst (c : Cert) (n : ℕ) (φ : Form) :
    len c (.inst n φ) = Nat.log 2 n + φ.size + 3 + c.size := by
  simp [len]

/-- The length of a proof of a guarded universal. -/
theorem len_all (c : Cert) (a : ℕ) (φ : Form) :
    len c (.all a φ) = Nat.log 2 a + φ.size + 3 + c.size := by
  simp [len]

end BoundedLob
