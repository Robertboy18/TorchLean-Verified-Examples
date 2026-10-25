/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Agents.Search
public import BoundedLob.Agents.Sources

/-!
# Playing the agents

`lake exe play_dupoc 500000` plays every ordered pair of the agents below at `k = 500000` and
prints each outcome, how each agent's search was decided, and how long the two runs of the match
took. `lake exe play_dupoc 17880 --proof` prints the proof two copies of DUPOC find.

Before a match is played, each agent's searches are decided by the worlds or by a checked
candidate alone. A match that would need the complete search over strings is reported and not
played, since that search is not a practical computation at these budgets. This preflight reads
sources whose searches are boxes outside any `coop` or `diag`, as all the agents here are, and
reports any other source as unsupported.
-/

@[expose] public section

namespace BoundedLob.Play

open Form Agent

/-- The agents of the tournament. -/
def agents : List (String × Form) :=
  [("CooperateBot", cooperate), ("DefectBot", defect), ("DUPOC", dupoc), ("CUPOD", cupod),
   ("CIMCIC", cimcic), ("DIMCID", dimcid), ("DUPOC(2k)", dupoc (.scale 2 .f)),
   ("DUPOC(k+100)", dupoc (.add .f (.const 100))), ("DUPOC-T", dupocTop)]

/-- The boxes of a filled source, or `none` if it has a search this preflight does not follow
(inside a `coop` or a `diag`). -/
def boxes : Form → Option (List (Bound × Form))
  | .bot | .opp | .me => some []
  | .imp φ ψ | .iff φ ψ => return (← boxes φ) ++ (← boxes ψ)
  | .box t φ => some [(t, φ)]
  | .coop _ _ | .diag _ _ => none

/-- How `search` decides `σ` within `t` without the complete search, if it does. -/
def stage (t : ℕ) (σ : Sen) : Option String :=
  if refuted worldDepth σ.form then some "no proof at any length (false at a world)"
  else
    match (candidates σ).find? (affordable t σ) with
    | some c => some s!"proof of {len c σ} characters ({σ.size} sentence + {c.size} certificate)"
    | none => none

/-- The searches of `a` against `b` at `k = n`, or `none` if one would need the complete search
or `boxes` does not support the source. -/
def stages (n : ℕ) (a b : Form) : Option (List String) := do
  (← boxes (a.fill b a)).mapM fun (t, φ) => stage (t.eval id n) (.inst n φ)

/-- `C` for cooperation, `D` for defection. -/
def moveLabel : Bool → String
  | true => "C"
  | false => "D"

/-- Play one ordered pair. -/
def playOne (n : ℕ) (na : String) (a : Form) (nb : String) (b : Form) : IO Unit := do
  if (boxes (a.fill b a)).isNone || (boxes (b.fill a b)).isNone then
    IO.println s!"{na} vs {nb}: unsupported source, not played"
    return
  match stages n a b, stages n b a with
  | some sa, some sb =>
    let start ← IO.monoNanosNow
    let x := run n a b
    let y := run n b a
    let elapsed := (← IO.monoNanosNow) - start
    IO.println s!"{na} vs {nb}: ({moveLabel x}, {moveLabel y}) in {elapsed / 1000} µs"
    for s in sa do IO.println s!"    {na}: {s}"
    for s in sb do IO.println s!"    {nb}: {s}"
  | _, _ => IO.println s!"{na} vs {nb}: needs the complete search, not played"

/-- Print the proof DUPOC finds against itself at `k = n`. -/
def printProof (n : ℕ) : IO Unit := do
  let σ := Sen.inst n (Form.coop dupoc dupoc)
  match (candidates σ).find? (affordable n σ) with
  | some c =>
    IO.println s!"DUPOC({n}) against itself proves {String.ofList σ.text} with this certificate \
      of {c.size} characters:"
    IO.println (String.ofList c.text)
  | none => IO.println s!"no candidate fits in {n} characters"

end BoundedLob.Play

open BoundedLob.Play in
/-- Play the tournament at `k = K`, or with `--proof` print the proof two copies of DUPOC find. -/
def main (args : List String) : IO UInt32 := do
  let usage := "usage: play_dupoc K [--proof]"
  match args with
  | [raw] | [raw, "--proof"] =>
    match raw.toNat? with
    | none =>
      IO.eprintln s!"K must be a natural number, not {raw}; {usage}"
      return 1
    | some n =>
      if args.length == 2 then printProof n
      else
        IO.println s!"k = {n}"
        for (na, a) in agents do
          for (nb, b) in agents do
            playOne n na a nb b
      return 0
  | _ =>
    IO.eprintln usage
    return 1
