/-
Checks for the circuit summaries exported by `neelsomani/verifiable-transformers`.

Neel's scripts already ran the extraction and Z3 verification steps. Here we
re-check the facts that make the finite claim meaningful: task identity,
retained edge set, candidate tokens, verified counts, zero reported failures,
and robustness radius.
-/

import VerifiableTransformers.Certificate.FiniteEval

namespace VerifiableTransformers.Certificate.Circuit

/-- Circuit/verification task names used by the exported JSON summaries. -/
inductive Task where
  | quoteClose
  | bracketType
deriving Repr, DecidableEq

/-- Directed edge in the pruned circuit graph. -/
structure Edge where
  source : String
  target : String
deriving Repr, DecidableEq

/-- Pruned circuit metadata exported by the extraction pass. -/
structure CircuitSummary where
  task : Task
  nLayers : Nat
  edges : List Edge
  ablation : String
  metric : String
  minAgreementMicros : Nat
  thresholdMicros : Nat
deriving Repr, DecidableEq

/-- One verification summary for either quote-close or bracket-type. -/
structure VerificationSummary where
  task : Task
  numInputs : Nat
  candidateTokens : List Nat
  circuit : CircuitSummary
  pytorchValidationPassed : Bool
  pytorchExamplesChecked : Nat
  pytorchFailures : Nat
  functionalStatusVerified : Bool
  functionalVerifiedCount : Nat
  functionalTotalSequences : Nat
  functionalTimeouts : Nat
  functionalErrors : Nat
  functionalCounterexamples : Nat
  edgeNecessityStatusVerified : Bool
  edgeTotal : Nat
  edgeNecessary : Nat
  edgeUnnecessary : Nat
  edgeUnresolved : Nat
  edgeTimeouts : Nat
  edgeErrors : Nat
  robustnessStatusVerified : Bool
  robustnessVerifiedCount : Nat
  robustnessTimeouts : Nat
  robustnessErrors : Nat
  robustnessEpsilonMicros : Nat
  robustnessViolations : Nat
  robustnessDecisionViolations : Nat
  robustnessBranchUnstable : Nat
deriving Repr, DecidableEq

/-- Convenience constructor for readable edge lists. -/
def edge (source target : String) : Edge :=
  { source, target }

/-- Retained edges for the quote-close circuit reported by the JSON summary. -/
def quoteEdges : List Edge :=
  [ edge "attn_0" "mlp_0"
  , edge "emb" "attn_0"
  , edge "mlp_0" "logits"
  ]

/-- Retained edges for the bracket-type circuit reported by the JSON summary. -/
def bracketEdges : List Edge :=
  [ edge "attn_0" "mlp_0"
  , edge "attn_1" "mlp_1"
  , edge "emb" "attn_0"
  , edge "mlp_0" "attn_1"
  , edge "mlp_0" "logits"
  , edge "mlp_1" "logits"
  ]

/-- Expected retained edge set by task. -/
def expectedEdges : Task → List Edge
  | .quoteClose => quoteEdges
  | .bracketType => bracketEdges

/-- Expected projected candidate tokens by task. -/
def expectedCandidates : Task → List Nat
  | .quoteClose => [9, 10]
  | .bracketType => [13, 14]

/-- Expected circuit extraction threshold, scaled by `1e6`. -/
def expectedThresholdMicros : Task → Nat
  | .quoteClose => 50000
  | .bracketType => 5000

/-- Structural circuit check independent of solver counts. -/
def checkCircuitSummary (c : CircuitSummary) : Bool :=
  c.nLayers == 2 &&
  c.edges == expectedEdges c.task &&
  c.ablation == "zero" &&
  c.metric == "candidate_kl" &&
  c.minAgreementMicros == 1000000 &&
  c.thresholdMicros == expectedThresholdMicros c.task

/-- Full summary check for the two committed circuit-verification runs. -/
def checkVerificationSummary (s : VerificationSummary) : Bool :=
  s.numInputs == 128 &&
  s.candidateTokens == expectedCandidates s.task &&
  s.circuit.task == s.task &&
  checkCircuitSummary s.circuit &&
  s.pytorchValidationPassed &&
  s.pytorchExamplesChecked == 128 &&
  s.pytorchFailures == 0 &&
  s.functionalStatusVerified &&
  s.functionalVerifiedCount == 128 &&
  s.functionalTotalSequences == 128 &&
  s.functionalTimeouts == 0 &&
  s.functionalErrors == 0 &&
  s.functionalCounterexamples == 0 &&
  s.edgeNecessityStatusVerified &&
  s.edgeTotal == s.circuit.edges.length &&
  s.edgeNecessary == s.circuit.edges.length &&
  s.edgeUnnecessary == 0 &&
  s.edgeUnresolved == 0 &&
  s.edgeTimeouts == 0 &&
  s.edgeErrors == 0 &&
  s.robustnessStatusVerified &&
  s.robustnessVerifiedCount == 128 &&
  s.robustnessTimeouts == 0 &&
  s.robustnessErrors == 0 &&
  s.robustnessEpsilonMicros == 10000 &&
  s.robustnessViolations == 0 &&
  s.robustnessDecisionViolations == 0 &&
  s.robustnessBranchUnstable == 0

/-- Proposition form of the full verification-summary check. -/
def AcceptedVerificationSummary (s : VerificationSummary) : Prop :=
  checkVerificationSummary s = true

open VerifiableTransformers.Certificate.FiniteEval

/--
Two rows share the projected structural feature when they belong to the same
task and have the same expected output token.  Content tokens may differ.
-/
def sameStructuralFeature (a b : CandidateEval) : Bool :=
  a.task == b.task && a.target == b.target

/-- Check that one row's projected decision is stable against later comparable rows. -/
def pairwiseContentInvariantFrom (row : CandidateEval) : List CandidateEval → Bool
  | [] => true
  | other :: rest =>
      (if sameStructuralFeature row other then
        row.predictedCandidate == other.predictedCandidate
      else
        true) &&
      pairwiseContentInvariantFrom row rest

/-- Pairwise projected content-invariance check over the row list. -/
def pairwiseContentInvariant : List CandidateEval → Bool
  | [] => true
  | row :: rest => pairwiseContentInvariantFrom row rest && pairwiseContentInvariant rest

/-- Finite trace version of content invariance for the projected candidate decision. -/
def checkFiniteTraceContentInvariance (rows : List CandidateEval) : Bool :=
  rowsReplayFiniteDomain rows &&
  rows.all CandidateEval.check &&
  pairwiseContentInvariant rows

end VerifiableTransformers.Certificate.Circuit

namespace VerifiableTransformers.Certificate.Properties

open VerifiableTransformers.Certificate.FiniteEval
open VerifiableTransformers.Certificate.Circuit

/-- The rows are exactly the exhaustive domain and choose the symbolic target. -/
def checkProjectedFunctionalEquivalence (rows : List CandidateEval) : Bool :=
  rowsReplayFiniteDomain rows && rows.all CandidateEval.check

/-- Changing content/filler tokens does not change the projected decision. -/
def checkProjectedContentInvariance (rows : List CandidateEval) : Bool :=
  checkFiniteTraceContentInvariance rows

/-- Bundle the directly replayable finite-domain properties proved for both traces. -/
def checkTraceProperties (rows : List CandidateEval) : Bool :=
  checkProjectedFunctionalEquivalence rows &&
  checkProjectedContentInvariance rows

/-- Proposition form of the replayable finite-trace property bundle. -/
def TracePropertiesAccepted (rows : List CandidateEval) : Prop :=
  checkTraceProperties rows = true

end VerifiableTransformers.Certificate.Properties
