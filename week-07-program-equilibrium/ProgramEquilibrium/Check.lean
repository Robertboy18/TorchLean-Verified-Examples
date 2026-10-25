/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import ProgramEquilibrium.Certificate
public import ProgramEquilibrium.Guards
public import Lean.Data.Json
-- `--run` also needs interpreter implementations of the imported modal evaluator.
meta import ProgramEquilibrium.Certificate
meta import ProgramEquilibrium.Guards

/-!
# JSON-lines entry point

Each input describes one exact finite population, a candidate index and a proposed stopping
certificate. The model supplies data only; no generated Lean source is compiled or executed.
-/

@[expose] public section

namespace ProgramEquilibrium.Checker

open Lean ProgramEquilibrium ProgramEquilibrium.Certificate

/-- Parse a bounded schema. The extra natural is its node count. -/
def parseForm (count : ℕ) : ℕ → Json → Except String
    (ProgramEquilibrium.Form (Target (Fin count)) × ℕ)
  | 0, _ => throw "schema exceeds depth limit 8"
  | fuel + 1, value => do
    let args ← value.getArr?
    let tag ← (← value.getArrVal? 0).getStr?
    let leaf (arity : ℕ) (φ : ProgramEquilibrium.Form (Target (Fin count))) :
        Except String (ProgramEquilibrium.Form (Target (Fin count)) × ℕ) := do
      if args.size != arity then throw s!"wrong arity for {tag}"
      pure (φ, 1)
    let unary (make : ProgramEquilibrium.Form (Target (Fin count)) →
        ProgramEquilibrium.Form (Target (Fin count))) :
        Except String (ProgramEquilibrium.Form (Target (Fin count)) × ℕ) := do
      if args.size != 2 then throw s!"wrong arity for {tag}"
      let (φ, nodes) ← parseForm count fuel (← value.getArrVal? 1)
      if nodes + 1 > 48 then throw "schema exceeds 48 nodes"
      pure (make φ, nodes + 1)
    let binary (make : ProgramEquilibrium.Form (Target (Fin count)) →
        ProgramEquilibrium.Form (Target (Fin count)) →
        ProgramEquilibrium.Form (Target (Fin count))) :
        Except String (ProgramEquilibrium.Form (Target (Fin count)) × ℕ) := do
      if args.size != 3 then throw s!"wrong arity for {tag}"
      let (φ, leftNodes) ← parseForm count fuel (← value.getArrVal? 1)
      let (ψ, rightNodes) ← parseForm count fuel (← value.getArrVal? 2)
      if leftNodes + rightNodes + 1 > 48 then throw "schema exceeds 48 nodes"
      pure (make φ ψ, leftNodes + rightNodes + 1)
    match tag with
    | "me" => leaf 1 (.atom .me)
    | "agent" =>
      if args.size != 2 then throw "wrong arity for agent"
      let index ← (← value.getArrVal? 1).getNat?
      if h : index < count then leaf 2 (.atom (.agent ⟨index, h⟩))
      else throw "agent reference is outside the population"
    | "top" => leaf 1 .top
    | "bot" => leaf 1 .bot
    | "not" => unary .not
    | "and" => binary .and
    | "or" => binary .or
    | "imp" => binary .imp
    | "box" => unary .box
    | _ => throw s!"unknown schema constructor: {tag}"

/-- Require a square Boolean table with one row and column per agent. -/
def parseTable (count : ℕ) (value : Json) : Except String (List (List Bool)) := do
  let rows ← value.getArr?
  if rows.size != count then throw "certificate table has wrong row count"
  let parsed ← rows.mapM fun row => do
    let entries ← row.getArr?
    if entries.size != count then throw "certificate table has wrong column count"
    pure (← entries.mapM Json.getBool?).toList
  pure parsed.toList

/-- Validate data, compute the actual table and adjudicate its certificate and policy. -/
def adjudicate (value : Json) : Except String Json := do
  let _ ← value.getObj?
  let raw ← (← value.getObjVal? "schemas").getArr?
  if hsize : 3 ≤ raw.size ∧ raw.size ≤ 10 then
    let parsed ← raw.mapM fun φ => return (← parseForm raw.size 9 φ).1
    if parsed.size != raw.size then throw "internal parser length mismatch"
    let schema : Fin raw.size → ProgramEquilibrium.Form (Target (Fin raw.size)) :=
      fun index => parsed[index.val]?.getD .bot
    let defector : Fin raw.size := ⟨1, by omega⟩
    let cooperator : Fin raw.size := ⟨2, by omega⟩
    match schema defector with
    | .bot => pure ()
    | _ => throw "agent 1 must be the unconditional defector"
    match schema cooperator with
    | .top => pure ()
    | _ => throw "agent 2 must be the unconditional cooperator"
    if hmodal : ∀ a, (schema a).modalized = true then
      let U : Universe := ⟨Fin raw.size, schema, hmodal⟩
      let agents := List.finRange raw.size
      let candidateIndex ← match value.getObjVal? "candidate" with
        | .ok index => index.getNat?
        | .error _ => pure 0
      if hc : candidateIndex < raw.size then
        let candidate : U.Agent := ⟨candidateIndex, hc⟩
        let world ← (← value.getObjVal? "world").getNat?
        if world > 64 then throw "certificate world exceeds limit 64"
        let claimed ← parseTable raw.size (← value.getObjVal? "table")
        let actual := table U agents world
        let stopped := stoppedCheck U agents world
        let tableMatches := claimed == actual
        let valid := Certificate.check U agents world claimed
        let safe := safeCheck U agents world candidate
        let self := U.play world candidate candidate
        let externalCount := (agents.filter fun b => decide (b ≠ candidate) &&
          U.play world candidate b && U.play world b candidate).length
        pure <| Json.mkObj [
          ("model", toJson "finite-modal-chain"),
          ("candidate", toJson candidateIndex),
          ("world", toJson world),
          ("valid_certificate", toJson valid),
          ("stopped", toJson stopped),
          ("table_matches", toJson tableMatches),
          ("safe", toJson safe),
          ("universal_guard", toJson ((U.schema candidate).isGuardedSchema)),
          ("self_cooperation", toJson self),
          ("external_cooperation_count", toJson externalCount),
          ("accepted", toJson (accepted U agents world claimed candidate)),
          ("table", toJson actual)]
      else throw "candidate index is outside the population"
    else throw "every schema atom must occur under a box"
  else throw "population must contain between 3 and 10 agents"

end ProgramEquilibrium.Checker

/-- Read independent requests until EOF. Errors reject only their own request. -/
def main : IO Unit := do
  let input ← IO.getStdin
  let output ← IO.getStdout
  while true do
    let line ← input.getLine
    if line.isEmpty then break
    let response := if line.utf8ByteSize > 65536 then
        Except.error "request exceeds 65536 bytes"
      else Lean.Json.parse line >>= ProgramEquilibrium.Checker.adjudicate
    let result := match response with
      | .ok result => result
      | .error message => Lean.Json.mkObj [
        ("error", Lean.toJson message), ("valid_certificate", Lean.toJson false),
        ("accepted", Lean.toJson false)]
    output.putStrLn result.compress
    output.flush

