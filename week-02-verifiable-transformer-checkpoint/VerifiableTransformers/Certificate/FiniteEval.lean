import VerifiableTransformers.Certificate.Fingerprint

/-!
# Exhaustive quote and bracket trace checks

Each row must contain the canonical prompt for its index and a strictly positive score margin
between its target and alternate token. Index checks exclude duplicate and missing prompts, and
the certificate also checks the recorded checkpoint fingerprint and score scale. These checks
compare exported integers, not a recomputed floating-point forward pass.

The last section proves that the index enumeration is the whole task: `expectedInput` is
injective on the 256 indices, and a row passes the domain predicates exactly when it is the
expected row at one of those indices.
-/

namespace VerifiableTransformers.Certificate.FiniteEval

/-- The two finite symbolic tasks checked by this example. -/
inductive SyntaxTask where
  | quoteClose
  | bracketType
deriving Repr, DecidableEq

/-- One trace row: a prompt, its target and alternate tokens, and their scores in micro-logits. -/
structure CandidateEval where
  task : SyntaxTask
  input : List Nat
  target : Nat
  alternate : Nat
  targetScoreMicros : Int
  alternateScoreMicros : Int
deriving Repr, DecidableEq

/-- Finite-domain evaluation trace with its source fingerprint, checkpoint path, and score scale. -/
structure EvalCertificate where
  sourceWeightsSha256 : String
  checkpointPath : String
  scoreScale : Nat
  rows : List CandidateEval
deriving Repr, DecidableEq

/-- Content (filler) tokens, shared by both tasks. -/
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

/-- Domain predicate for one task's prompt layout, target, and alternate token IDs. -/
def taskDomainOk : SyntaxTask → List Nat → Nat → Nat → Bool
  | .quoteClose, input, target, alternate => quoteDomainOk input target alternate
  | .bracketType, input, target, alternate => bracketDomainOk input target alternate

/-- `taskDomainOk` for a row's task, prompt, and tokens. `marginOk` checks the scores. -/
def CandidateEval.domainOk (row : CandidateEval) : Bool :=
  taskDomainOk row.task row.input row.target row.alternate

/-- Integer target-minus-alternate margin in micro-logit units. -/
def CandidateEval.marginMicros (row : CandidateEval) : Int :=
  row.targetScoreMicros - row.alternateScoreMicros

/-- Whether the target scores strictly above the alternate. -/
def CandidateEval.marginOk (row : CandidateEval) : Bool :=
  decide (row.alternateScoreMicros < row.targetScoreMicros)

/-- Per-row check (domain and margin), independent of the row's position. -/
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

/-- Quote-task prompt at index `idx < 128`: the low bit picks the quote token, and `idx / 2` in
base 4 picks `x1 x2 x3`. -/
def expectedQuoteInput (idx : Nat) : List Nat :=
  let core := idx / 2
  let x1 := expectedContentToken (core / 16)
  let x2 := expectedContentToken ((core / 4) % 4)
  let x3 := expectedContentToken (core % 4)
  let quote := if idx % 2 = 0 then 9 else 10
  [1, 2, x1, quote, x2, x3]

/-- Bracket-task prompt at index `idx < 128`, laid out like `expectedQuoteInput`. -/
def expectedBracketInput (idx : Nat) : List Nat :=
  let core := idx / 2
  let x1 := expectedContentToken (core / 16)
  let x2 := expectedContentToken ((core / 4) % 4)
  let x3 := expectedContentToken (core % 4)
  let openTok := if idx % 2 = 0 then 11 else 12
  [1, 3, x1, openTok, x2, x3]

/-- Task of trace row `idx`: quote-close for the first 128 rows, then bracket-type. -/
def expectedTask (idx : Nat) : SyntaxTask :=
  if idx < 128 then .quoteClose else .bracketType

/-- Prompt of trace row `idx`. -/
def expectedInput (idx : Nat) : List Nat :=
  if idx < 128 then
    expectedQuoteInput idx
  else
    expectedBracketInput (idx - 128)

/-- Target token of trace row `idx`. -/
def expectedTarget (idx : Nat) : Nat :=
  if idx < 128 then
    if idx % 2 = 0 then 9 else 10
  else
    if (idx - 128) % 2 = 0 then 13 else 14

/-- Alternate token of trace row `idx`. -/
def expectedAlternate (idx : Nat) : Nat :=
  if idx < 128 then
    if idx % 2 = 0 then 10 else 9
  else
    if (idx - 128) % 2 = 0 then 14 else 13

/-- Whether a row has the task, prompt, and tokens expected at index `idx`. -/
def CandidateEval.matchesExpectedIndex (idx : Nat) (row : CandidateEval) : Bool :=
  row.task == expectedTask idx &&
  row.input == expectedInput idx &&
  row.target == expectedTarget idx &&
  row.alternate == expectedAlternate idx

/-- Candidate with the higher score. A tie goes to the alternate. -/
def CandidateEval.predictedCandidate (row : CandidateEval) : Nat :=
  if row.alternateScoreMicros < row.targetScoreMicros then
    row.target
  else
    row.alternate

/-- Check for the row at index `idx`: expected contents, target predicted, positive margin. -/
def CandidateEval.replayCheckAt (idx : Nat) (row : CandidateEval) : Bool :=
  row.matchesExpectedIndex idx &&
  row.predictedCandidate == expectedTarget idx &&
  row.marginOk

/-- Check consecutive rows against the indices `idx, idx + 1, ...`. -/
def rowsReplayFiniteDomainFrom (idx : Nat) : List CandidateEval → Bool
  | [] => true
  | row :: rows =>
      row.replayCheckAt idx && rowsReplayFiniteDomainFrom (idx + 1) rows

/-- Whether the rows are the 256 domain prompts in canonical quote-then-bracket order. -/
def rowsReplayFiniteDomain (rows : List CandidateEval) : Bool :=
  rows.length == 256 && rowsReplayFiniteDomainFrom 0 rows

/-- Certificate check for a trace whose source file should hash to `expectedSha256`. -/
def checkEvalCertificateWithSha (expectedSha256 : String) (cert : EvalCertificate) : Bool :=
  cert.sourceWeightsSha256 == expectedSha256 &&
  cert.scoreScale == 1000000 &&
  rowsReplayFiniteDomain cert.rows &&
  taskCount .quoteClose cert.rows == 128 &&
  taskCount .bracketType cert.rows == 128 &&
  cert.rows.all CandidateEval.check

/-- Check specialized to the pinned `smt_weights.json` fingerprint of the upstream-code run. -/
def checkEvalCertificate (cert : EvalCertificate) : Bool :=
  checkEvalCertificateWithSha Fingerprint.smtWeightsSha256 cert

/-- Proposition form of `checkEvalCertificate`. -/
def AcceptedEval (cert : EvalCertificate) : Prop :=
  checkEvalCertificate cert = true

/-- Proposition form of `checkEvalCertificateWithSha`. -/
def AcceptedEvalWithSha (expectedSha256 : String) (cert : EvalCertificate) : Prop :=
  checkEvalCertificateWithSha expectedSha256 cert = true


/-! ## The enumeration is the whole task -/

/-- Index of a canonical prompt. It inverts `expectedInput` on the 256 task prompts. -/
def promptIndex : List Nat → Nat
  | [_, taskTok, x1, delim, x2, x3] =>
      let core := 16 * (x1 - 5) + 4 * (x2 - 5) + (x3 - 5)
      if taskTok == 2 then 2 * core + (delim - 9) else 128 + 2 * core + (delim - 11)
  | _ => 0

theorem promptIndex_expectedInput : ∀ i, i < 256 → promptIndex (expectedInput i) = i := by
  decide +kernel

/-- Different indices give different prompts, so a 256-row trace cannot repeat a prompt. -/
theorem expectedInput_injective {i j : Nat} (hi : i < 256) (hj : j < 256)
    (h : expectedInput i = expectedInput j) : i = j := by
  rw [← promptIndex_expectedInput i hi, ← promptIndex_expectedInput j hj, h]

/-- Every expected row is accepted by the domain predicates. -/
theorem expectedRow_domainOk : ∀ i, i < 256 →
    taskDomainOk (expectedTask i) (expectedInput i) (expectedTarget i) (expectedAlternate i) =
      true := by
  decide +kernel

/-- The content tokens, as a list `decide` can run over. -/
def contentTokens : List Nat := [5, 6, 7, 8]

theorem mem_contentTokens {x : Nat} (h : isContentToken x = true) : x ∈ contentTokens := by
  simp only [isContentToken, Bool.or_eq_true, beq_iff_eq] at h
  simp only [contentTokens, List.mem_cons, List.mem_nil_iff, or_false]
  omega

/-- Row data equal to the expected row at index `promptIndex input < 256`. -/
def isExpectedRow (task : SyntaxTask) (input : List Nat) (target alternate : Nat) : Prop :=
  promptIndex input < 256 ∧ expectedTask (promptIndex input) = task ∧
    expectedInput (promptIndex input) = input ∧ expectedTarget (promptIndex input) = target ∧
    expectedAlternate (promptIndex input) = alternate

instance (task : SyntaxTask) (input : List Nat) (target alternate : Nat) :
    Decidable (isExpectedRow task input target alternate) := by
  unfold isExpectedRow; exact inferInstance

theorem quoteRows_enumerated : ∀ x1 ∈ contentTokens, ∀ x2 ∈ contentTokens,
    ∀ x3 ∈ contentTokens,
    isExpectedRow .quoteClose [1, 2, x1, 9, x2, x3] 9 10 ∧
    isExpectedRow .quoteClose [1, 2, x1, 10, x2, x3] 10 9 := by
  decide

theorem bracketRows_enumerated : ∀ x1 ∈ contentTokens, ∀ x2 ∈ contentTokens,
    ∀ x3 ∈ contentTokens,
    isExpectedRow .bracketType [1, 3, x1, 11, x2, x3] 13 14 ∧
    isExpectedRow .bracketType [1, 3, x1, 12, x2, x3] 14 13 := by
  decide

/-- A row accepted by the domain predicates is the expected row at some index below 256. -/
theorem taskDomainOk_expected {task : SyntaxTask} {input : List Nat} {target alternate : Nat}
    (h : taskDomainOk task input target alternate = true) :
    ∃ i, i < 256 ∧ expectedTask i = task ∧ expectedInput i = input ∧
      expectedTarget i = target ∧ expectedAlternate i = alternate := by
  suffices isExpectedRow task input target alternate from ⟨_, this⟩
  cases task with
  | quoteClose =>
    simp only [taskDomainOk, quoteDomainOk] at h
    split at h
    · rename_i x1 quote x2 x3
      simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at h
      obtain ⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, hq⟩, ht⟩, ha⟩ := h
      have r := quoteRows_enumerated x1 (mem_contentTokens h1) x2 (mem_contentTokens h2) x3
        (mem_contentTokens h3)
      rcases hq with rfl | rfl <;> rcases ha with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> simp_all
    · simp at h
  | bracketType =>
    simp only [taskDomainOk, bracketDomainOk] at h
    split at h
    · rename_i x1 openTok x2 x3
      simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at h
      obtain ⟨⟨⟨h1, h2⟩, h3⟩, hb⟩ := h
      have r := bracketRows_enumerated x1 (mem_contentTokens h1) x2 (mem_contentTokens h2) x3
        (mem_contentTokens h3)
      rcases hb with ⟨⟨rfl, rfl⟩, rfl⟩ | ⟨⟨rfl, rfl⟩, rfl⟩ <;> simp_all
    · simp at h

end VerifiableTransformers.Certificate.FiniteEval
