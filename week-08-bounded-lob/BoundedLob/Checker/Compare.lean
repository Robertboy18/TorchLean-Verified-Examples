/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Theory.System
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# When one bound eventually exceeds another

This compares concrete bounds with `f` read as the identity, for the checker's monotonicity rule.
Every bound is `s·k + β·lg k` plus a bounded error, with rational `s, β ≥ 0`: each `div` loses
less than one to the floor. `Bound.enclosure` computes `s`, `β` and an interval for the error,
all multiplied by a common denominator so that everything is an integer. The difference `y - x`
grows without bound exactly when its pair `(s, β)` is positive in the lexicographic order, and
`Bound.guard` then turns the coefficients into a threshold beyond which `x ≤ y`. Only `ℕ` and `ℤ`
arithmetic is involved, so the kernel evaluates `guard`.

`guard` only handles gaps that grow without bound: `x.guard x` is `none`, so `none` is not
evidence that an inequality fails. When `s` is positive the threshold comes from the first dyadic
block `[2^j, 2^(j+1))` from which `s·2^j` stays ahead of the logarithm and the error; when only
`β` is positive the threshold is a power of two. Neither is claimed to be the least threshold.
-/

@[expose] public section

namespace BoundedLob

namespace Bound

/-- `den` times a bound lies between `slope·k + log·lg k + lo` and `slope·k + log·lg k + hi`. -/
structure Enclosure where
  /-- The common denominator, a product of divisors. -/
  den : ℕ
  /-- The coefficient of `k`. -/
  slope : ℕ
  /-- The coefficient of `lg k`. -/
  log : ℕ
  /-- A lower bound on the error term. -/
  lo : ℤ
  /-- An upper bound on the error term. -/
  hi : ℤ

/-- The enclosure of a bound with `f` read as the identity. -/
def enclosure : Bound → Enclosure
  | const n => ⟨1, 0, 0, n, n⟩
  | lg => ⟨1, 0, 1, 0, 0⟩
  | f => ⟨1, 1, 0, 0, 0⟩
  | add a b =>
    let p := a.enclosure
    let q := b.enclosure
    ⟨p.den * q.den, q.den * p.slope + p.den * q.slope, q.den * p.log + p.den * q.log,
      q.den * p.lo + p.den * q.lo, q.den * p.hi + p.den * q.hi⟩
  | scale c a =>
    let p := a.enclosure
    ⟨p.den, c * p.slope, c * p.log, c * p.lo, c * p.hi⟩
  | div a d =>
    -- Division by `0` is `0` in `ℕ`, and so is the enclosure.
    if d = 0 then ⟨1, 0, 0, 0, 0⟩ else
      let p := a.enclosure
      ⟨p.den * d, p.slope, p.log, p.lo - p.den * ((d : ℤ) - 1), p.hi⟩

/-- The enclosure is correct at every `k`. -/
theorem enclosure_spec (x : Bound) (k : ℕ) :
    0 < x.enclosure.den ∧
      (x.enclosure.slope * k + x.enclosure.log * Nat.log 2 k + x.enclosure.lo : ℤ) ≤
        x.enclosure.den * x.eval id k ∧
      (x.enclosure.den * x.eval id k : ℤ) ≤
        x.enclosure.slope * k + x.enclosure.log * Nat.log 2 k + x.enclosure.hi := by
  induction x with
  | const n => simp [enclosure, eval]
  | lg => simp [enclosure, eval]
  | f => simp [enclosure, eval]
  | add a b iha ihb =>
    obtain ⟨ha0, ha1, ha2⟩ := iha
    obtain ⟨hb0, hb1, hb2⟩ := ihb
    simp only [enclosure, eval, Nat.cast_add, Nat.cast_mul]
    have hA : (0 : ℤ) ≤ a.enclosure.den := Nat.cast_nonneg _
    have hB : (0 : ℤ) ≤ b.enclosure.den := Nat.cast_nonneg _
    refine ⟨Nat.mul_pos ha0 hb0, ?_, ?_⟩
    · nlinarith [mul_le_mul_of_nonneg_left ha1 hB, mul_le_mul_of_nonneg_left hb1 hA]
    · nlinarith [mul_le_mul_of_nonneg_left ha2 hB, mul_le_mul_of_nonneg_left hb2 hA]
  | scale c a ih =>
    obtain ⟨h0, h1, h2⟩ := ih
    simp only [enclosure, eval, Nat.cast_mul]
    have hc : (0 : ℤ) ≤ c := Nat.cast_nonneg c
    refine ⟨h0, ?_, ?_⟩
    · nlinarith [mul_le_mul_of_nonneg_left h1 hc]
    · nlinarith [mul_le_mul_of_nonneg_left h2 hc]
  | div a d ih =>
    obtain ⟨h0, h1, h2⟩ := ih
    by_cases hd : d = 0
    · subst hd
      simp [enclosure, eval]
    · simp only [enclosure, eval, hd, ite_false, Nat.cast_mul]
      set v := a.eval id k
      have hq : (d : ℤ) * (v / d : ℕ) + (v % d : ℕ) = v := by exact_mod_cast Nat.div_add_mod v d
      have hr : ((v % d : ℕ) : ℤ) + 1 ≤ d := by exact_mod_cast Nat.mod_lt v (Nat.pos_of_ne_zero hd)
      have hr0 : (0 : ℤ) ≤ (v % d : ℕ) := Nat.cast_nonneg _
      have hD : (0 : ℤ) ≤ a.enclosure.den := Nat.cast_nonneg _
      refine ⟨Nat.mul_pos h0 (Nat.pos_of_ne_zero hd), ?_, ?_⟩
      · nlinarith [mul_le_mul_of_nonneg_left hr hD]
      · nlinarith [mul_nonneg hD hr0]

/-- `n ≤ k·a` once `k` is at least the ceiling `(n + a - 1) / a`. -/
private theorem le_mul_of_ceil_le {n a k : ℕ} (ha : 0 < a) (h : (n + a - 1) / a ≤ k) :
    n ≤ k * a := by
  have := (Nat.div_lt_iff_lt_mul ha).mp (Nat.lt_succ_of_le h)
  rw [Nat.succ_mul] at this
  omega

/-- The first `j` in `[i, i + n]` with `p j`, or `i + n` when there is none; callers make sure
`p (i + n)` holds. -/
def firstFrom (p : ℕ → Bool) : ℕ → ℕ → ℕ
  | 0, i => i
  | n + 1, i => if p i then i else firstFrom p n (i + 1)

private theorem firstFrom_spec (p : ℕ → Bool) (n i : ℕ) (h : p (i + n)) : p (firstFrom p n i) := by
  induction n generalizing i with
  | zero => simpa [firstFrom] using h
  | succ n ih =>
    simp only [firstFrom]
    split_ifs with hp
    · exact hp
    · exact ih (i + 1) (by rwa [show i + 1 + n = i + (n + 1) by omega])

/-- `a·2^j` is at least `b·j + c` and at least the step `b` to the next block. -/
def covers (a b c j : ℕ) : Bool := decide (b * j + c ≤ a * 2 ^ j ∧ b ≤ a * 2 ^ j)

private theorem covers_iff {a b c j : ℕ} :
    covers a b c j ↔ b * j + c ≤ a * 2 ^ j ∧ b ≤ a * 2 ^ j := by
  simp [covers]

private theorem covers_mono {a b c i j : ℕ} (h : covers a b c j) (hij : j ≤ i) :
    covers a b c i := by
  induction i, hij using Nat.le_induction with
  | base => exact h
  | succ i _ ih =>
    rw [covers_iff] at ih ⊢
    rw [pow_succ]
    constructor <;> nlinarith [ih.1, ih.2]

/-- The search for a block can stop at `4 · ⌈(b + c) / a⌉`: squaring `2M + 1 ≤ 2^(2M)` shows
that this block already covers. -/
private theorem covers_fuel {a b c : ℕ} (ha : 0 < a) :
    covers a b c (4 * ((b + c + a - 1) / a)) := by
  set M := (b + c + a - 1) / a
  have hM : b + c ≤ M * a := le_mul_of_ceil_le ha le_rfl
  have h1 : 2 * M + 1 ≤ 2 ^ (2 * M) := Nat.lt_two_pow_self
  have h2 : M * (4 * M + 1) ≤ 2 ^ (4 * M) := by
    rw [show 4 * M = 2 * M + 2 * M by ring, pow_add]
    nlinarith [Nat.mul_le_mul h1 h1]
  have h3 : a * (M * (4 * M + 1)) ≤ a * 2 ^ (4 * M) := Nat.mul_le_mul_left a h2
  have h4 : (b + c) * (4 * M + 1) ≤ M * a * (4 * M + 1) := Nat.mul_le_mul_right _ hM
  rw [covers_iff]
  constructor <;> nlinarith

/-- The first dyadic block from which `a·2^j` stays at least `b·j + c`, for `a > 0`. -/
def block (a b c : ℕ) : ℕ := firstFrom (covers a b c) (4 * ((b + c + a - 1) / a)) 0

private theorem le_of_block {a b c k : ℕ} (ha : 0 < a)
    (hk : (b * block a b c + c + a - 1) / a ≤ k) :
    b * Nat.log 2 k + c ≤ a * k := by
  have hj : covers a b c (block a b c) :=
    firstFrom_spec _ _ 0 (by rw [zero_add]; exact covers_fuel ha)
  have hk' := le_mul_of_ceil_le ha hk
  by_cases hl : Nat.log 2 k ≤ block a b c
  · nlinarith [Nat.mul_le_mul_left b hl]
  · have hk0 : k ≠ 0 := by
      rintro rfl
      simp at hl
    have hp := (covers_iff.mp (covers_mono hj (le_of_not_ge hl))).1
    nlinarith [Nat.mul_le_mul_left a (Nat.pow_log_le_self 2 hk0)]

/-- A `K` beyond which `a·k + b·lg k ≥ c`, when `(a, b)` is lexicographically positive. -/
def threshold (a b c : ℤ) : Option ℕ :=
  if 0 < a then
    let a' := a.toNat
    let b' := (-b).toNat
    let c' := c.toNat
    some ((b' * block a' b' c' + c' + a' - 1) / a')
  else if a = 0 ∧ 0 < b then
    let b' := b.toNat
    some (2 ^ ((c.toNat + b' - 1) / b') - 1)
  else none

/-- Beyond the threshold, `a·k + b·lg k ≥ c`. -/
theorem threshold_spec {a b c : ℤ} {K : ℕ} (h : threshold a b c = some K) :
    ∀ k > K, c ≤ a * k + b * Nat.log 2 k := by
  intro k hk
  have hlg : (0 : ℤ) ≤ Nat.log 2 k := Nat.cast_nonneg _
  have hc : c ≤ (c.toNat : ℤ) := Int.self_le_toNat c
  unfold threshold at h
  split_ifs at h with ha hab
  · cases h
    have hb : -b ≤ ((-b).toNat : ℤ) := Int.self_le_toNat (-b)
    have hk' : ((-b).toNat * Nat.log 2 k + c.toNat : ℤ) ≤ a.toNat * k := by
      exact_mod_cast le_of_block (Int.lt_toNat.mpr ha) hk.le
    rw [Int.toNat_of_nonneg ha.le] at hk'
    nlinarith [mul_nonneg (show (0 : ℤ) ≤ (-b).toNat + b by linarith) hlg]
  · cases h
    obtain ⟨rfl, hb⟩ := hab
    have hb' : 0 < b.toNat := Int.lt_toNat.mpr hb
    have hpow : 2 ^ ((c.toNat + b.toNat - 1) / b.toNat) ≤ k := by
      have := Nat.one_le_two_pow (n := (c.toNat + b.toNat - 1) / b.toNat)
      omega
    have hm := Nat.le_log_of_pow_le (by norm_num) hpow
    have hc' : (c.toNat : ℕ) ≤ Nat.log 2 k * b.toNat :=
      (le_mul_of_ceil_le hb' le_rfl).trans (Nat.mul_le_mul_right _ hm)
    have hc'' : (c.toNat : ℤ) ≤ Nat.log 2 k * b.toNat := by exact_mod_cast hc'
    rw [Int.toNat_of_nonneg hb.le] at hc''
    linarith

/-- A threshold exists exactly for lexicographically positive `(a, b)`. -/
theorem threshold_isSome {a b c : ℤ} :
    (threshold a b c).isSome ↔ 0 < a ∨ a = 0 ∧ 0 < b := by
  unfold threshold
  split_ifs with ha hab <;> simp_all

/-- `some K` when `y` eventually exceeds `x` by every constant; then `x ≤ y` beyond `K`
(`le_of_guard`). Bounded gaps, such as `x` against itself, give `none`. -/
def guard (x y : Bound) : Option ℕ :=
  let p := x.enclosure
  let q := y.enclosure
  threshold (p.den * q.slope - q.den * p.slope) (p.den * q.log - q.den * p.log)
    (q.den * p.hi - p.den * q.lo)

/-- Beyond the guard, `x ≤ y`. -/
theorem le_of_guard {x y : Bound} {K : ℕ} (h : x.guard y = some K) :
    ∀ k > K, x.eval id k ≤ y.eval id k := by
  intro k hk
  have h0 := threshold_spec h k hk
  obtain ⟨hx0, -, hx⟩ := enclosure_spec x k
  obtain ⟨hy0, hy, -⟩ := enclosure_spec y k
  have h1 := mul_le_mul_of_nonneg_left hx (Nat.cast_nonneg (α := ℤ) y.enclosure.den)
  have h2 := mul_le_mul_of_nonneg_left hy (Nat.cast_nonneg (α := ℤ) x.enclosure.den)
  have hpos : (0 : ℤ) < x.enclosure.den * y.enclosure.den := by
    exact_mod_cast Nat.mul_pos hx0 hy0
  have : (x.enclosure.den * y.enclosure.den : ℤ) * x.eval id k ≤
      (x.enclosure.den * y.enclosure.den : ℤ) * y.eval id k := by
    linarith
  exact_mod_cast le_of_mul_le_mul_left this hpos

/-- A gap that grows without bound always has a guard. -/
theorem guard_isSome {x y : Bound}
    (h : ∀ c, ∃ K, ∀ k > K, x.eval id k + c ≤ y.eval id k) : (x.guard y).isSome := by
  by_contra hn
  simp only [guard] at hn
  rw [threshold_isSome] at hn
  set a : ℤ := x.enclosure.den * y.enclosure.slope - y.enclosure.den * x.enclosure.slope
  set b : ℤ := x.enclosure.den * y.enclosure.log - y.enclosure.den * x.enclosure.log
  set E : ℤ := x.enclosure.den * y.enclosure.hi - y.enclosure.den * x.enclosure.lo
  obtain ⟨K, hK⟩ := h (E.toNat + 1)
  have ha : a ≤ 0 := by
    by_contra ha
    exact hn (Or.inl (lt_of_not_ge ha))
  -- a block `2^i` beyond `K` on which `a·k + b·lg k` is not positive
  obtain ⟨i, hiK, hi⟩ : ∃ i : ℕ, K < 2 ^ i ∧ a * ((2 ^ i : ℕ) : ℤ) + b * i ≤ 0 := by
    by_cases hb : b ≤ 0
    · refine ⟨K + 1, by have := @Nat.lt_two_pow_self (K + 1); omega, ?_⟩
      have : a * ((2 ^ (K + 1) : ℕ) : ℤ) ≤ 0 := mul_nonpos_of_nonpos_of_nonneg ha (by positivity)
      have : b * ((K + 1 : ℕ) : ℤ) ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hb (by positivity)
      linarith
    · have hb : 0 < b := lt_of_not_ge hb
      have ha' : a < 0 := lt_of_le_of_ne ha (fun h0 => hn (Or.inr ⟨h0, hb⟩))
      have hJ := covers_fuel (b := b.toNat) (c := 0) (Int.lt_toNat.mpr (neg_pos.mpr ha'))
      set J := 4 * ((b.toNat + 0 + (-a).toNat - 1) / (-a).toNat)
      refine ⟨max J (K + 1), ?_, ?_⟩
      · have := @Nat.lt_two_pow_self (max J (K + 1))
        omega
      · have hc := (covers_iff.mp (covers_mono hJ (le_max_left J (K + 1)))).1
        have hc' : ((b.toNat * max J (K + 1) + 0 : ℕ) : ℤ) ≤
            (((-a).toNat * 2 ^ max J (K + 1) : ℕ) : ℤ) := by exact_mod_cast hc
        rw [Nat.cast_add, Nat.cast_mul, Nat.cast_mul, Int.toNat_of_nonneg hb.le,
          Int.toNat_of_nonneg (by linarith)] at hc'
        push_cast at hc' ⊢
        linarith
  have hk : ((x.eval id (2 ^ i) : ℕ) : ℤ) + E.toNat + 1 ≤ y.eval id (2 ^ i) := by
    exact_mod_cast hK (2 ^ i) hiK
  have hlog : Nat.log 2 (2 ^ i) = i := Nat.log_pow (by norm_num) i
  obtain ⟨hx0, hx, -⟩ := enclosure_spec x (2 ^ i)
  obtain ⟨hy0, -, hy⟩ := enclosure_spec y (2 ^ i)
  rw [hlog] at hx hy
  have hD : (1 : ℤ) ≤ x.enclosure.den * y.enclosure.den := by
    exact_mod_cast Nat.mul_pos hx0 hy0
  have h1 := mul_le_mul_of_nonneg_left hx (Nat.cast_nonneg (α := ℤ) y.enclosure.den)
  have h2 := mul_le_mul_of_nonneg_left hy (Nat.cast_nonneg (α := ℤ) x.enclosure.den)
  have h3 := mul_le_mul_of_nonneg_left hk (show (0 : ℤ) ≤ x.enclosure.den * y.enclosure.den
    by linarith)
  have h4 := mul_le_mul_of_nonneg_right hD (show (0 : ℤ) ≤ E.toNat + 1 by positivity)
  have := Int.self_le_toNat E
  nlinarith

/-! The shapes the bounded Löb proof meets, with a representative overhead of `5000`. -/

example : ((const 5000 + lg) + div f 6 + const 0 + scale 2 (div f 6) + const 0).guard f =
    some 10028 := by decide +kernel

example : ((const 5000 + lg) + div (div f 2) 6 + const 0 + scale 2 (div (div f 2) 6) +
    const 0).guard (div f 2) = some 20062 := by decide +kernel

example : (const 5000 + lg).guard (div f 6) = some 30095 := by decide +kernel

example : (const 5000 + lg).guard (div (div f 2) 6) = some 60203 := by decide +kernel

example : ((const 5000 + lg) + div f 2 + const 0).guard f = some 10028 := by decide +kernel

example : f.guard (div f 2) = none := by decide +kernel

/-! Smaller overheads, and nested divisions. -/

example : ((const 160 + lg) + div f 6 + const 0 + scale 2 (div f 6) + const 0).guard f =
    some 338 := by decide +kernel

example : (const 9).guard (div f 6) = some 59 := by decide +kernel

example : (const 30).guard (div (div (add f (const 100)) 2) 6) = some 271 := by decide +kernel

end Bound

end BoundedLob
