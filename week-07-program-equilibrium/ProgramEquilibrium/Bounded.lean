/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import Mathlib.Data.List.Find

/-!
# Cooperation with a certificate budget

Unlike the modal-chain evaluator, these agents return an actual move after a finite search.
The budget counts UTF-8 bytes in a cooperation certificate. The two certificate rules cover
constant cooperators and pairs of searchers that can afford each other's reciprocal certificate.
The latter rule is justified by `Certificate.check_sound`, not assumed as a reflection axiom.

This is a restricted certificate calculus, not an arithmetization of Lean or Peano arithmetic.
It does not establish the parametric bounded Löb theorem or solve Open Problem 4 of Critch,
Dennis and Russell, <https://arxiv.org/abs/2208.07006>. In particular, a certificate's byte count
is not the length of its expanded Lean proof, and search exhaustion says nothing about proofs
outside these two rules. The program registry is fixed throughout each theorem.
-/

@[expose] public section

namespace ProgramEquilibrium.Bounded

/-- Registered code. A searcher chooses a byte budget and which certificate rule to try first.
Other total Lean functions may be opponents, even when neither rule can certify them. -/
inductive Program where
  | cooperate
  | defect
  | search (certificateBytes : Nat) (reciprocalFirst : Bool)
  | other (action : String → Bool)

/-- Evidence that the named subject cooperates with the named opponent, in a fixed registry. -/
inductive Certificate where
  | constant (subject opponent : String)
  | reciprocal (subject opponent : String)
  deriving Repr

namespace Certificate

/-- The program whose move is certified. -/
def subject : Certificate → String
  | constant a _ | reciprocal a _ => a

/-- The program it is playing against. -/
def opponent : Certificate → String
  | constant _ b | reciprocal _ b => b

/-- Literal certificate text, with quoted names. Names refer to the supplied registry;
they are not hashes of source code. -/
def encode (p : Certificate) : String :=
  -- Use total string operations so concrete certificate sizes can also be kernel-reduced.
  let quoteName (name : String) :=
    "\"" ++ String.join (name.toList.map fun c => c.quoteCore (inString := true)) ++ "\""
  match p with
  | constant a b => "C(" ++ quoteName a ++ "," ++ quoteName b ++ ")"
  | reciprocal a b => "R(" ++ quoteName a ++ "," ++ quoteName b ++ ")"

/-- The budget counts this text, including names and delimiters. -/
def bytes (p : Certificate) : Nat := p.encode.utf8ByteSize

end Certificate

/-- Check both searchers' budgets without running either searcher. -/
def reciprocalCheck (program : String → Program) (a b : String) : Bool :=
  match program a, program b with
  | .search ka _, .search kb _ =>
    decide ((Certificate.reciprocal b a).bytes ≤ ka ∧
      (Certificate.reciprocal a b).bytes ≤ kb)
  | _, _ => false

/-- The constant rule inspects the subject's code. The reciprocal rule inspects both programs
and their budgets. Neither rule accepts an opaque `Program.other` function. -/
def Certificate.check (program : String → Program) : Certificate → Bool
  | constant a _ => match program a with
    | .cooperate => true
    | _ => false
  | reciprocal a b => reciprocalCheck program a b

/-- Two heuristic orders over the same finite certificate calculus. -/
def candidates (a b : String) (reciprocalFirst : Bool) : List Certificate :=
  if reciprocalFirst then [.reciprocal a b, .constant a b]
  else [.constant a b, .reciprocal a b]

/-- Return the first valid certificate that fits the byte budget. The search terminates after
at most two candidates; it does not enumerate all Lean proofs of the goal. -/
def findCertificate (program : String → Program) (certificateBytes : Nat)
    (reciprocalFirst : Bool) (a b : String) : Option Certificate :=
  (candidates a b reciprocalFirst).find? fun p =>
    decide (p.bytes ≤ certificateBytes) && p.check program

/-- A searcher cooperates exactly when it finds an affordable certificate of its opponent's
cooperation. `true` means cooperation; `false` means defection. -/
def run (program : String → Program) (a b : String) : Bool :=
  match program a with
  | .cooperate => true
  | .defect => false
  | .other action => action b
  | .search budget order => (findCertificate program budget order b a).isSome

private theorem findCertificate_success (program : String → Program) (k : Nat) (order : Bool)
    (a b : String) :
    (findCertificate program k order a b).isSome = true ↔
      ((Certificate.constant a b).bytes ≤ k ∧ (Certificate.constant a b).check program = true) ∨
      ((Certificate.reciprocal a b).bytes ≤ k ∧
        (Certificate.reciprocal a b).check program = true) := by
  cases order <;> simp [findCertificate, candidates, List.find?_isSome, or_comm]

/-- Every accepted certificate describes an actual move of the terminating program.
The reciprocal case unfolds both searches: each contains the affordable certificate whose
validity the checker has already established from the program descriptions. -/
theorem Certificate.check_sound (program : String → Program) (p : Certificate)
    (h : p.check program = true) : run program p.subject p.opponent = true := by
  cases p with
  | constant a b =>
    cases ha : program a <;> simp [Certificate.check, run, Certificate.subject, ha] at h ⊢
  | reciprocal a b =>
    cases ha : program a <;> cases hb : program b <;>
      simp only [Certificate.check, reciprocalCheck, ha, hb, Bool.false_eq_true] at h
    rename_i ka oa kb ob
    have limits : (Certificate.reciprocal b a).bytes ≤ ka ∧
        (Certificate.reciprocal a b).bytes ≤ kb := of_decide_eq_true h
    simp only [Certificate.subject, Certificate.opponent, run, ha]
    apply (findCertificate_success program ka oa b a).2
    refine Or.inr ⟨limits.1, ?_⟩
    simp [Certificate.check, reciprocalCheck, ha, hb, limits.1, limits.2]

/-- A successful search never exceeds its declared certificate budget. -/
theorem findCertificate_bytes {program : String → Program} {k : Nat} {order : Bool}
    {a b : String} {p : Certificate} (h : findCertificate program k order a b = some p) :
    p.bytes ≤ k := by
  have hp := List.find?_some h
  exact of_decide_eq_true (Bool.and_eq_true_iff.mp hp).1

/-- A returned certificate proves the requested ordered pair, not merely some cooperation. -/
theorem findCertificate_sound {program : String → Program} {k : Nat} {order : Bool}
    {a b : String} {p : Certificate} (h : findCertificate program k order a b = some p) :
    run program a b = true := by
  have hp := List.find?_some h
  have checked : p.check program = true := (Bool.and_eq_true_iff.mp hp).2
  have hmem := List.mem_of_find?_eq_some h
  have goal : (p.subject, p.opponent) = (a, b) := by
    cases order <;> simp [candidates] at hmem <;>
      rcases hmem with rfl | rfl <;> rfl
  have cooperation := p.check_sound program checked
  have hs : p.subject = a := congrArg Prod.fst goal
  have ho : p.opponent = b := congrArg Prod.snd goal
  simpa only [hs, ho] using cooperation

/-- Against any registered opponent, including arbitrary total code, a searcher never
cooperates unilaterally. No eventual-world or population cutoff is involved. -/
theorem search_not_exploited {program : String → Program} {a b : String}
    {k : Nat} {order : Bool} (ha : program a = .search k order)
    (h : run program a b = true) : run program b a = true := by
  simp only [run, ha] at h
  rcases (findCertificate_success program k order b a).1 h with h | h
  · exact Certificate.check_sound program (.constant b a) h.2
  · exact Certificate.check_sound program (.reciprocal b a) h.2

/-- The exact self-play threshold, for every name, budget, search order and fixed registry.
This is a theorem about the two certificate rules above, not a bounded Löb theorem for PA. -/
theorem self_cooperates_iff {program : String → Program} {a : String}
    {k : Nat} {order : Bool} (ha : program a = .search k order) :
    run program a a = true ↔ (Certificate.reciprocal a a).bytes ≤ k := by
  simp [run, ha, findCertificate_success, Certificate.check, reciprocalCheck]

end ProgramEquilibrium.Bounded
