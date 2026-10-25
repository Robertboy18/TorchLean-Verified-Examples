/-
Copyright (c) 2026 Robert Joseph George
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import BoundedLob.Checker.Form

/-!
# The agents

An agent is its source: the formula it tests, written with the placeholders `opp` (the opponent's
source) and `me` (its own). The agent cooperates exactly when the formula, read with the two real
sources filled in, is true. Every placeholder sits under a box, so reading a source only asks for
proofs and never runs the opponent. These are the agents of Critch, Dennis and Russell
(arXiv:2208.07006), each with a budget `t(k)` that is `k` unless given. A box `□_t φ` asks for a
proof of `φ` with at most `t(k)` characters.
-/

@[expose] public section

namespace BoundedLob

open Form

namespace Agent

/-- CooperateBot always cooperates. -/
def cooperate : Form := top

/-- DefectBot always defects. -/
def defect : Form := bot

/-- DUPOC: cooperate exactly when some proof of at most `t(k)` characters shows that the opponent
cooperates. -/
def dupoc (t : Bound := .f) : Form := box t (coop opp me)

/-- CUPOD: defect exactly when some proof of at most `t(k)` characters shows that the opponent
defects. -/
def cupod (t : Bound := .f) : Form := neg (box t (neg (coop opp me)))

/-- CIMCIC: cooperate exactly when some proof of at most `t(k)` characters shows that my
cooperation implies the opponent's. -/
def cimcic (t : Bound := .f) : Form := box t (imp (coop me opp) (coop opp me))

/-- DIMCID: defect exactly when some proof of at most `t(k)` characters shows that my cooperation
implies the opponent's defection. -/
def dimcid (t : Bound := .f) : Form := neg (box t (imp (coop me opp) (neg (coop opp me))))

/-- DUPOC written differently: "if true, then the opponent cooperates". -/
def dupocTop (t : Bound := .f) : Form := box t (imp top (coop opp me))

end Agent

end BoundedLob
