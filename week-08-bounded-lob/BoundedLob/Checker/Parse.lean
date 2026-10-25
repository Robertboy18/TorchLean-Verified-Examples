/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Checker.Encoding

/-!
# Reading texts back

`Cert.parse` reads a certificate from its complete text, and `Cert.parse_text` says it inverts
`Cert.text`. Bounds and formulas have the prefix parsers the certificate parser needs; sentences
are never parsed. The parsers are recursive descent: each reads one term from the front of the
input and returns the rest. Recursion is on a fuel argument that decreases by one along each
branch (sibling subterms get the same fuel), so the length of the input is always enough.

`strings m` lists every string over `alphabet` with at most `m` characters, so a search over
proofs of bounded length is a search over a finite list. The list is not ordered by length, so a
search through it finds some proof, not necessarily a shortest one.
-/

@[expose] public section

namespace BoundedLob

/-! ### Numerals -/

/-- Binary digits are `0` and `1`. -/
theorem natDigits_mem {n : ℕ} {ch : Char} (h : ch ∈ natDigits n) : ch = '0' ∨ ch = '1' := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    by_cases hn : n < 2
    · rw [natDigits_of_lt hn] at h
      split at h <;> simp_all
    · rw [natDigits_of_two_le (by omega)] at h
      rcases List.mem_append.1 h with h | h
      · exact ih _ (by omega) h
      · split at h <;> simp_all

/-- Read binary digits up to `.`, starting from the value `acc`. -/
def parseDigits (acc : ℕ) : List Char → Option (ℕ × List Char)
  | '0' :: s => parseDigits (acc * 2) s
  | '1' :: s => parseDigits (acc * 2 + 1) s
  | '.' :: s => some (acc, s)
  | _ => none

/-- Read a numeral as `natText` writes it: `0.`, or `1`, more binary digits and `.`. -/
def parseNat : List Char → Option (ℕ × List Char)
  | '0' :: '.' :: s => some (0, s)
  | '1' :: s => parseDigits 1 s
  | _ => none

/-- Reading the digits of a positive `n` is reading the rest after the leading `1`. -/
theorem parseNat_digits {n : ℕ} (hn : 0 < n) (s : List Char) :
    parseNat (natDigits n ++ s) = parseDigits n s := by
  induction n using Nat.strong_induction_on generalizing s with
  | _ n ih =>
    by_cases h2 : n < 2
    · obtain rfl : n = 1 := by omega
      rw [natDigits_of_lt h2]
      rfl
    · rw [natDigits_of_two_le (by omega), List.append_assoc, ih _ (by omega) (by omega)]
      split <;> simp only [List.singleton_append, parseDigits] <;> congr 1 <;> omega

/-- `parseNat` inverts `natText`. -/
theorem parseNat_text (n : ℕ) (s : List Char) : parseNat (natText n ++ s) = some (n, s) := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · rw [natText, natDigits_of_lt (by omega)]
    rfl
  · rw [natText, List.append_assoc, parseNat_digits hn]
    rfl

/-! ### Parsers -/

/-- Read a bound from the front of the input, with one unit of fuel per level of nesting. -/
def parseBound : ℕ → List Char → Option (Bound × List Char)
  | 0, _ => none
  | _ + 1, 'n' :: s => (parseNat s).map fun (k, s) => (.const k, s)
  | _ + 1, 'l' :: s => some (.lg, s)
  | _ + 1, 'k' :: s => some (.f, s)
  | n + 1, '+' :: s => (parseBound n s).bind fun (a, s) =>
      (parseBound n s).map fun (b, s) => (.add a b, s)
  | n + 1, '*' :: s => (parseNat s).bind fun (c, s) =>
      (parseBound n s).map fun (a, s) => (.scale c a, s)
  | n + 1, '/' :: s => (parseBound n s).bind fun (a, s) =>
      (parseNat s).map fun (d, s) => (.div a d, s)
  | _, _ => none

/-- `parseBound` inverts `Bound.text`, with enough fuel. -/
theorem parseBound_text (t : Bound) (s : List Char) {n : ℕ} (hn : t.text.length ≤ n) :
    parseBound n (t.text ++ s) = some (t, s) := by
  induction t generalizing s n <;> rcases n with _ | n <;>
    simp only [Bound.text, List.length_cons, List.length_append] at hn <;>
    first
    | omega
    | simp (disch := omega) [Bound.text, parseBound, parseNat_text, *]

/-- Read a formula from the front of the input, with one unit of fuel per level of nesting. -/
def parseForm : ℕ → List Char → Option (Form × List Char)
  | 0, _ => none
  | _ + 1, 'F' :: s => some (.bot, s)
  | n + 1, '>' :: s => (parseForm n s).bind fun (φ, s) =>
      (parseForm n s).map fun (ψ, s) => (.imp φ ψ, s)
  | n + 1, '=' :: s => (parseForm n s).bind fun (φ, s) =>
      (parseForm n s).map fun (ψ, s) => (.iff φ ψ, s)
  | n + 1, '#' :: s => (parseBound n s).bind fun (t, s) =>
      (parseForm n s).map fun (φ, s) => (.box t φ, s)
  | n + 1, 'c' :: s => (parseForm n s).bind fun (a, s) =>
      (parseForm n s).map fun (b, s) => (.coop a b, s)
  | _ + 1, 'o' :: s => some (.opp, s)
  | _ + 1, 'm' :: s => some (.me, s)
  | n + 1, 'd' :: s => (parseBound n s).bind fun (g, s) =>
      (parseForm n s).map fun (p, s) => (.diag g p, s)
  | _, _ => none

/-- `parseForm` inverts `Form.text`, with enough fuel. -/
theorem parseForm_text (φ : Form) (s : List Char) {n : ℕ} (hn : φ.text.length ≤ n) :
    parseForm n (φ.text ++ s) = some (φ, s) := by
  induction φ generalizing s n <;> rcases n with _ | n <;>
    simp only [Form.text, List.length_cons, List.length_append, List.length_nil] at hn <;>
    first
    | omega
    | simp (disch := omega) [Form.text, parseForm, parseBound_text, *]

/-- Read a certificate from the front of the input, with one unit of fuel per level of
nesting. -/
def parseCert : ℕ → List Char → Option (Cert × List Char)
  | 0, _ => none
  | _ + 1, 'K' :: s => some (.axK, s)
  | _ + 1, 'S' :: s => some (.axS, s)
  | _ + 1, 'N' :: s => some (.axDN, s)
  | n + 1, 'L' :: s => (parseCert n s).map fun (c, s) => (.iffL c, s)
  | n + 1, 'R' :: s => (parseCert n s).map fun (c, s) => (.iffR c, s)
  | n + 1, 'B' :: s => (parseCert n s).bind fun (c, s) =>
      (parseCert n s).map fun (d, s) => (.iffI c d, s)
  | n + 1, 'M' :: s => (parseForm n s).bind fun (φ, s) => (parseCert n s).bind fun (c, s) =>
      (parseCert n s).map fun (d, s) => (.mp φ c d, s)
  | n + 1, 'U' :: s => (parseNat s).bind fun (a, s) =>
      (parseCert n s).map fun (c, s) => (.raise a c, s)
  | n + 1, 'T' :: s => (parseNat s).bind fun (a, s) =>
      (parseCert n s).map fun (c, s) => (.inst a c, s)
  | n + 1, 'Q' :: s => (parseCert n s).map fun (c, s) => (.qd c, s)
  | n + 1, 'X' :: s => (parseCert n s).map fun (c, s) => (.nec c, s)
  | _ + 1, 'P' :: s => some (.dist, s)
  | _ + 1, '4' :: s => some (.four, s)
  | _ + 1, 'O' :: s => some (.mono, s)
  | _ + 1, 'G' :: s => some (.diag, s)
  | _ + 1, 'Y' :: s => some (.prog, s)
  | _, _ => none

/-- `parseCert` inverts `Cert.text`, with enough fuel. -/
theorem parseCert_text (c : Cert) (s : List Char) {n : ℕ} (hn : c.text.length ≤ n) :
    parseCert n (c.text ++ s) = some (c, s) := by
  induction c generalizing s n <;> rcases n with _ | n <;>
    simp only [Cert.text, List.length_cons, List.length_append, List.length_nil] at hn <;>
    first
    | omega
    | simp (disch := omega) [Cert.text, parseCert, parseNat_text, parseForm_text, *]

/-- Parse a certificate from its complete text. -/
def Cert.parse (s : List Char) : Option Cert :=
  match parseCert s.length s with
  | some (c, []) => some c
  | _ => none

/-- `Cert.parse` inverts `Cert.text`. -/
theorem Cert.parse_text (c : Cert) : Cert.parse c.text = some c := by
  have h := parseCert_text c [] le_rfl
  rw [List.append_nil] at h
  simp [Cert.parse, h]

/-! ### The alphabet -/

/-- Numerals are written in the alphabet. -/
theorem natText_subset (n : ℕ) : natText n ⊆ alphabet := by
  intro ch h
  rcases List.mem_append.1 h with h | h
  · rcases natDigits_mem h with rfl | rfl <;> decide
  · rw [List.mem_singleton.1 h]; decide

/-- Bounds are written in the alphabet. -/
theorem Bound.text_subset (t : Bound) : t.text ⊆ alphabet := by
  induction t <;> simp [Bound.text, List.cons_subset, natText_subset, *] <;> decide

/-- Formulas are written in the alphabet. -/
theorem Form.text_subset (φ : Form) : φ.text ⊆ alphabet := by
  induction φ <;> simp [Form.text, List.cons_subset, Bound.text_subset, *] <;> decide

/-- Certificates are written in the alphabet. -/
theorem Cert.text_subset (c : Cert) : c.text ⊆ alphabet := by
  induction c <;>
    simp [Cert.text, List.cons_subset, natText_subset, Form.text_subset, *] <;> decide

/-- Every string over `alphabet` with at most `m` characters. -/
def strings : ℕ → List (List Char)
  | 0 => [[]]
  | m + 1 =>
    let shorter := strings m
    [] :: alphabet.flatMap fun ch => shorter.map (ch :: ·)

/-- `strings m` contains every string over the alphabet with at most `m` characters. -/
theorem mem_strings {s : List Char} {m : ℕ} (hs : s.length ≤ m) (ha : ∀ ch ∈ s, ch ∈ alphabet) :
    s ∈ strings m := by
  induction m generalizing s with
  | zero => simp_all [strings]
  | succ m ih =>
    cases s with
    | nil => simp [strings]
    | cons ch t =>
      simp only [strings, List.mem_cons, List.mem_flatMap, List.mem_map]
      exact .inr ⟨ch, ha ch (by simp), t, ih (by simpa using hs) (by simp_all), rfl⟩

end BoundedLob
