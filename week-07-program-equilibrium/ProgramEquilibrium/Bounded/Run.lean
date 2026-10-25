/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import ProgramEquilibrium.Bounded
-- These bodies let `decide` unfold the ASCII quoting calculation below.
import all Init.Data.Repr
import all Init.Data.String.Defs
meta import ProgramEquilibrium.Bounded

/-!
# Running bounded self-play

`lake exe play_bounded_agents 16` runs two search orders with the same certificate budget,
then tries a cooperator, a defector and an opaque function. Try `15` to see self-play defect.
-/

@[expose] public section

namespace ProgramEquilibrium.Bounded.Example

/-- The two searchers use different rule orders. The opaque cooperator shows that this proof
search can miss true cooperation: its code is executable, but neither rule proves its move. -/
def programs (budget : Nat) : String → Program
  | "self" => .search budget false
  | "peer" => .search budget true
  | "C" => .cooperate
  | "D" => .defect
  | _ => .other fun _ => true

/-- The threshold is computed from the literal certificate `R("self","self")`.
The parameter is arbitrary; this is not just a check of one chosen budget. -/
theorem self_cooperates_iff (budget : Nat) :
    run (programs budget) "self" "self" = true ↔ 16 ≤ budget := by
  have size : (Certificate.reciprocal "self" "self").bytes = 16 := by
    unfold Certificate.bytes Certificate.encode String.join Char.quoteCore
    decide
  simpa only [size] using Bounded.self_cooperates_iff
    (program := programs budget) (a := "self") rfl

-- These concrete boundary and rejection checks use ordinary kernel reduction.
example : run (programs 15) "self" "self" = false := by
  exact Bool.eq_false_iff.mpr fun h => (by decide : ¬16 ≤ 15) ((self_cooperates_iff 15).mp h)
example : (Certificate.reciprocal "self" "D").check (programs 16) = false := by decide
example : (Certificate.constant "D" "self").check (programs 16) = false := by decide

/-- A single line of the native demo, including the certificate actually found by `self`. -/
def describe (budget : Nat) (opponent : String) : String :=
  let program := programs budget
  let move (value : Bool) := if value then "C" else "D"
  let outcome := s!"{move (run program "self" opponent)}/{move (run program opponent "self")}"
  let evidence := match findCertificate program budget false opponent "self" with
    | none => "no certificate found"
    | some p => s!"{p.encode}, {p.bytes} bytes"
  s!"self vs {opponent}: {outcome}; {evidence}"

end ProgramEquilibrium.Bounded.Example

/-- Run the finite search, with an optional certificate-byte budget (default 16). -/
def main (args : List String) : IO Unit := do
  let budget ← match args with
    | [] => pure 16
    | [raw] => match raw.toNat? with
      | some budget => pure budget
      | none => throw (IO.userError "budget must be a natural number of certificate bytes")
    | _ => throw (IO.userError "usage: play_bounded_agents [certificate-bytes]")
  IO.println s!"Certificate budget: {budget} UTF-8 bytes; at most two candidates per search."
  let start ← IO.monoNanosNow
  for opponent in ["self", "peer", "C", "D", "hidden"] do
    IO.println (ProgramEquilibrium.Bounded.Example.describe budget opponent)
  let elapsed := (← IO.monoNanosNow) - start
  IO.println s!"Search and output: {elapsed / 1000} microseconds (excludes build/startup)."
