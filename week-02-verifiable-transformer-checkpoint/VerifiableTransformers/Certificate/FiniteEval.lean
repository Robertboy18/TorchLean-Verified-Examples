/-
Finite prompt checks for the quote/bracket task.

Both Neel's PyTorch/Z3 path and the TorchLean CUDA path write candidate-score
traces. The shared Lean contract is simple: every row has to be the canonical
prompt for its index, and the symbolic target token has to beat the alternate
candidate.
-/

import VerifiableTransformers.Generated.UpstreamExportSummary

namespace VerifiableTransformers.Certificate.FiniteEval

/-- The two finite symbolic tasks checked by this example. -/
inductive SyntaxTask where
  | quoteClose
  | bracketType
deriving Repr, DecidableEq

/-- One projected-decision row: one prompt and the two candidate logits we care about. -/
structure CandidateEval where
  task : SyntaxTask
  input : List Nat
  target : Nat
  alternate : Nat
  targetScoreMicros : Int
  alternateScoreMicros : Int
deriving Repr, DecidableEq

/-- A finite-domain evaluation trace plus the checkpoint identity it came from. -/
structure EvalCertificate where
  sourceWeightsSha256 : String
  checkpointPath : String
  scoreScale : Nat
  rows : List CandidateEval
deriving Repr, DecidableEq

/-- Content/filler tokens used by both tasks. -/
def isContentToken (x : Nat) : Bool :=
  x == 5 || x == 6 || x == 7 || x == 8

/-- Domain predicate for `[BOS, quote-task, x1, quote, x2, x3]`. -/
def quoteDomainOk (input : List Nat) (target alternate : Nat) : Bool :=
  match input with
  | 1 :: 2 :: x1 :: quote :: x2 :: x3 :: [] =>
      isContentToken x1 &&
      isContentToken x2 &&
      isContentToken x3 &&
      (quote == 9 || quote == 10) &&
      target == quote &&
      ((target == 9 && alternate == 10) ||
       (target == 10 && alternate == 9))
  | _ => false

/-- Domain predicate for `[BOS, bracket-task, x1, open, x2, x3]`. -/
def bracketDomainOk (input : List Nat) (target alternate : Nat) : Bool :=
  match input with
  | 1 :: 3 :: x1 :: openTok :: x2 :: x3 :: [] =>
      isContentToken x1 &&
      isContentToken x2 &&
      isContentToken x3 &&
      ((openTok == 11 && target == 13 && alternate == 14) ||
       (openTok == 12 && target == 14 && alternate == 13))
  | _ => false

/-- Local row check: syntax/domain shape plus positive target margin. -/
def CandidateEval.domainOk (row : CandidateEval) : Bool :=
  match row.task with
  | .quoteClose => quoteDomainOk row.input row.target row.alternate
  | .bracketType => bracketDomainOk row.input row.target row.alternate

/-- Integer target-minus-alternate margin in micro-logit units. -/
def CandidateEval.marginMicros (row : CandidateEval) : Int :=
  row.targetScoreMicros - row.alternateScoreMicros

/-- The projected decision must prefer the symbolic target. -/
def CandidateEval.marginOk (row : CandidateEval) : Bool :=
  decide (row.alternateScoreMicros < row.targetScoreMicros)

/-- Full per-row check before considering row order. -/
def CandidateEval.check (row : CandidateEval) : Bool :=
  row.domainOk && row.marginOk

/-- Number of rows belonging to one syntax task. -/
def taskCount (task : SyntaxTask) (rows : List CandidateEval) : Nat :=
  rows.foldl (fun acc row => if row.task == task then acc + 1 else acc) 0

/-- Canonical order for content tokens in the exhaustive 256-prompt domain. -/
def expectedContentToken (i : Nat) : Nat :=
  match i % 4 with
  | 0 => 5
  | 1 => 6
  | 2 => 7
  | _ => 8

/-- Expected quote task prompt at a canonical quote-domain index. -/
def expectedQuoteInput (idx : Nat) : List Nat :=
  let core := idx / 2
  let x1 := expectedContentToken (core / 16)
  let x2 := expectedContentToken ((core / 4) % 4)
  let x3 := expectedContentToken (core % 4)
  let quote := if idx % 2 = 0 then 9 else 10
  [1, 2, x1, quote, x2, x3]

/-- Expected bracket task prompt at a canonical bracket-domain index. -/
def expectedBracketInput (idx : Nat) : List Nat :=
  let core := idx / 2
  let x1 := expectedContentToken (core / 16)
  let x2 := expectedContentToken ((core / 4) % 4)
  let x3 := expectedContentToken (core % 4)
  let openTok := if idx % 2 = 0 then 11 else 12
  [1, 3, x1, openTok, x2, x3]

/-- Expected task for the global 256-row trace index. -/
def expectedTask (idx : Nat) : SyntaxTask :=
  if idx < 128 then .quoteClose else .bracketType

/-- Expected prompt for the global 256-row trace index. -/
def expectedInput (idx : Nat) : List Nat :=
  if idx < 128 then
    expectedQuoteInput idx
  else
    expectedBracketInput (idx - 128)

/-- Expected symbolic target token for the global 256-row trace index. -/
def expectedTarget (idx : Nat) : Nat :=
  if idx < 128 then
    if idx % 2 = 0 then 9 else 10
  else
    if (idx - 128) % 2 = 0 then 13 else 14

/-- Expected alternate candidate token for the global 256-row trace index. -/
def expectedAlternate (idx : Nat) : Nat :=
  if idx < 128 then
    if idx % 2 = 0 then 10 else 9
  else
    if (idx - 128) % 2 = 0 then 14 else 13

/-- Row `idx` must be exactly the corresponding prompt in the finite domain. -/
def CandidateEval.matchesExpectedIndex (idx : Nat) (row : CandidateEval) : Bool :=
  row.task == expectedTask idx &&
  row.input == expectedInput idx &&
  row.target == expectedTarget idx &&
  row.alternate == expectedAlternate idx

/-- Candidate chosen by the row's two projected scores. -/
def CandidateEval.predictedCandidate (row : CandidateEval) : Nat :=
  if row.alternateScoreMicros < row.targetScoreMicros then
    row.target
  else
    row.alternate

/-- Index-aware replay check for one generated row. -/
def CandidateEval.replayCheckAt (idx : Nat) (row : CandidateEval) : Bool :=
  row.matchesExpectedIndex idx &&
  row.predictedCandidate == expectedTarget idx &&
  row.marginOk

/-- Tail-recursive replay of a row list from a given canonical index. -/
def rowsReplayFiniteDomainFrom (idx : Nat) : List CandidateEval → Bool
  | [] => true
  | row :: rows =>
      row.replayCheckAt idx && rowsReplayFiniteDomainFrom (idx + 1) rows

/-- Exhaustive trace replay: exactly 256 rows, in canonical quote-then-bracket order. -/
def rowsReplayFiniteDomain (rows : List CandidateEval) : Bool :=
  rows.length == 256 && rowsReplayFiniteDomainFrom 0 rows

/-- Generic trace check parameterized by the expected checkpoint hash. -/
def checkEvalCertificateWithSha (expectedSha256 : String) (cert : EvalCertificate) : Bool :=
  cert.sourceWeightsSha256 == expectedSha256 &&
  cert.scoreScale == 1000000 &&
  rowsReplayFiniteDomain cert.rows &&
  taskCount .quoteClose cert.rows == 128 &&
  taskCount .bracketType cert.rows == 128 &&
  cert.rows.all CandidateEval.check

/-- Check specialized to the committed checkpoint fingerprint from Neel's run. -/
def checkEvalCertificate (cert : EvalCertificate) : Bool :=
  checkEvalCertificateWithSha
    VerifiableTransformers.Generated.UpstreamExportSummary.exportSummary.sha256
    cert

/-- Proposition form of `checkEvalCertificate`. -/
def AcceptedEval (cert : EvalCertificate) : Prop :=
  checkEvalCertificate cert = true

/-- Proposition form of `checkEvalCertificateWithSha`. -/
def AcceptedEvalWithSha (expectedSha256 : String) (cert : EvalCertificate) : Prop :=
  checkEvalCertificateWithSha expectedSha256 cert = true

end VerifiableTransformers.Certificate.FiniteEval
