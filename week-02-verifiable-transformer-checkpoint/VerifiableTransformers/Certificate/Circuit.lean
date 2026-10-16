import VerifiableTransformers.Certificate.FiniteEval

/-!
# Checks for exported circuit summaries

Neel Somani's `scripts/small/extract.py` and `scripts/small/verify.py` produce these
summaries, and `tools/generate_circuit_summaries.py` copies them out of the JSON files they
write. We check that a summary names the pinned checkpoint, uses edges of upstream's
block-level circuit graph, and that its status strings agree with its own counts under the
rule upstream's verifier uses. Acceptance does not check a Z3 proof.

The Lean Float replay in `Replay.UpstreamFloatReplay` recomputes two of the recorded results
independently: the circuit's decision on every prompt, and for each retained edge a prompt on
which removing it changes that decision.

The file ends with two projected properties of a `FiniteEval` score trace: every prompt
prefers its target, and content tokens do not change the decision.
-/

namespace VerifiableTransformers.Certificate.Circuit

/-- The two tasks named in the exported circuit and verification summaries. -/
inductive Task where
  | quoteClose
  | bracketType
deriving Repr, DecidableEq

/-- Directed edge between two named nodes of the circuit graph. -/
structure Edge where
  source : String
  target : String
deriving Repr, DecidableEq

/-- Convenience constructor for readable edge lists. -/
def edge (source target : String) : Edge :=
  { source, target }

/--
All 15 edges of the block-level graph that upstream's `CircuitGraph` builds for two layers:
the embedding feeds every block and the logits, each attention block feeds its own MLP, each
block feeds every block of the next layer, and every block feeds the logits.
-/
def smallCircuitGraphEdges : List Edge :=
  [ edge "emb" "attn_0", edge "emb" "mlp_0", edge "emb" "attn_1", edge "emb" "mlp_1"
  , edge "emb" "logits"
  , edge "attn_0" "mlp_0", edge "attn_1" "mlp_1"
  , edge "attn_0" "attn_1", edge "attn_0" "mlp_1", edge "mlp_0" "attn_1", edge "mlp_0" "mlp_1"
  , edge "attn_0" "logits", edge "mlp_0" "logits", edge "attn_1" "logits", edge "mlp_1" "logits"
  ]

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

/-- Results of `verify.py` for one task, with the circuit and checkpoint they refer to. -/
structure VerificationSummary where
  task : Task
  upstreamCommit : String
  weightsSha256 : String
  checkpointSha256 : String
  numInputs : Nat
  candidateTokens : List Nat
  circuit : CircuitSummary
  pytorchStatus : String
  pytorchExamplesChecked : Nat
  pytorchFailures : Nat
  functionalStatus : String
  functionalVerifiedCount : Nat
  functionalTotalSequences : Nat
  functionalTimeouts : Nat
  functionalErrors : Nat
  functionalCounterexamples : Nat
  edgeStatus : String
  edgeTotal : Nat
  edgeNecessary : Nat
  edgeUnnecessary : Nat
  edgeUnresolved : Nat
  edgeTimeouts : Nat
  edgeErrors : Nat
  robustnessStatus : String
  robustnessVerifiedCount : Nat
  robustnessTimeouts : Nat
  robustnessErrors : Nat
  robustnessEpsilonMicros : Nat
  robustnessViolations : Nat
  robustnessDecisionViolations : Nat
  robustnessBranchUnstable : Nat
deriving Repr, DecidableEq

/-- Expected projected candidate tokens by task. -/
def expectedCandidates : Task → List Nat
  | .quoteClose => [9, 10]
  | .bracketType => [13, 14]

/-- Thresholds of upstream's default `--threshold_sweep`, scaled by `1e6`. -/
def sweepThresholdsMicros : List Nat :=
  [1000, 2000, 5000, 10000, 20000, 50000]

/-- Structural circuit check: real graph edges, no repeats, upstream's extraction settings. -/
def checkCircuitSummary (c : CircuitSummary) : Bool :=
  c.nLayers == 2 &&
  c.edges.all (fun e => smallCircuitGraphEdges.contains e) &&
  c.edges.Nodup &&
  c.ablation == "zero" &&
  c.metric == "candidate_kl" &&
  c.minAgreementMicros == 1000000 &&
  sweepThresholdsMicros.contains c.thresholdMicros

/-- `verify.py` reports `PASSED` for the PyTorch circuit check exactly when nothing failed.
`ERROR` (an exception inside `verify.py`) carries no counts. -/
def pytorchStatusConsistent (s : VerificationSummary) : Bool :=
  s.pytorchStatus == "ERROR" ||
  (s.pytorchExamplesChecked == s.numInputs &&
   (s.pytorchStatus == "PASSED") == (s.pytorchFailures == 0))

/-- Status rule for functional equivalence: `FAILED` means a counterexample, and `VERIFIED`
means every input verified with no timeout or error. `NOT_RUN` marks a property `verify.py` did
not reach. -/
def functionalStatusConsistent (s : VerificationSummary) : Bool :=
  s.functionalStatus == "NOT_RUN" || s.functionalStatus == "ERROR" ||
  (s.functionalTotalSequences == s.numInputs &&
   (s.functionalStatus == "FAILED") == (s.functionalCounterexamples > 0) &&
   (s.functionalStatus == "VERIFIED") ==
     (s.functionalCounterexamples == 0 && s.functionalTimeouts == 0 &&
      s.functionalErrors == 0 && s.functionalVerifiedCount == s.numInputs))

/-- Status rule for edge necessity, plus the edge counts adding up to the circuit's size. -/
def edgeStatusConsistent (s : VerificationSummary) : Bool :=
  s.edgeStatus == "NOT_RUN" || s.edgeStatus == "ERROR" ||
  (s.edgeTotal == s.circuit.edges.length &&
   s.edgeNecessary + s.edgeUnnecessary + s.edgeUnresolved == s.edgeTotal &&
   (s.edgeStatus == "FAILED") == (s.edgeUnnecessary > 0) &&
   (s.edgeStatus == "VERIFIED") ==
     (s.edgeUnnecessary == 0 && s.edgeUnresolved == 0 && s.edgeTimeouts == 0 &&
      s.edgeErrors == 0 && s.edgeNecessary == s.edgeTotal))

/-- Status rule for robustness of the final residual. -/
def robustnessStatusConsistent (s : VerificationSummary) : Bool :=
  s.robustnessStatus == "NOT_RUN" || s.robustnessStatus == "ERROR" ||
  (s.robustnessEpsilonMicros == 10000 &&
   (s.robustnessStatus == "VERIFIED") ==
     (s.robustnessViolations == 0 && s.robustnessBranchUnstable == 0 &&
      s.robustnessTimeouts == 0 && s.robustnessErrors == 0 &&
      s.robustnessVerifiedCount == s.numInputs))

/-- Summary check: pinned checkpoint, task data, circuit structure, and consistent counts. -/
def checkVerificationSummary (s : VerificationSummary) : Bool :=
  s.upstreamCommit == Fingerprint.upstreamCommit &&
  s.weightsSha256 == Fingerprint.smtWeightsSha256 &&
  s.checkpointSha256 == Fingerprint.modelSafetensorsSha256 &&
  s.numInputs == 128 &&
  s.candidateTokens == expectedCandidates s.task &&
  s.circuit.task == s.task &&
  checkCircuitSummary s.circuit &&
  pytorchStatusConsistent s &&
  functionalStatusConsistent s &&
  edgeStatusConsistent s &&
  robustnessStatusConsistent s

/-- Whether the PyTorch check passed and all three SMT properties came back `VERIFIED`. -/
def VerificationSummary.allVerified (s : VerificationSummary) : Bool :=
  s.pytorchStatus == "PASSED" &&
  s.functionalStatus == "VERIFIED" &&
  s.edgeStatus == "VERIFIED" &&
  s.robustnessStatus == "VERIFIED"

/-- Proposition form of `checkVerificationSummary`. -/
def AcceptedVerificationSummary (s : VerificationSummary) : Prop :=
  checkVerificationSummary s = true

open VerifiableTransformers.Certificate.FiniteEval

/--
Two rows share a structural feature when they have the same task and target. Their content
tokens may differ.
-/
def sameStructuralFeature (a b : CandidateEval) : Bool :=
  a.task == b.task && a.target == b.target

/-- Whether `row` predicts the same candidate as every listed row with its structural feature. -/
def pairwiseContentInvariantFrom (row : CandidateEval) : List CandidateEval → Bool
  | [] => true
  | other :: rest =>
      (if sameStructuralFeature row other then
        row.predictedCandidate == other.predictedCandidate
      else
        true) &&
      pairwiseContentInvariantFrom row rest

/-- Whether every two rows with the same structural feature predict the same candidate. -/
def pairwiseContentInvariant : List CandidateEval → Bool
  | [] => true
  | row :: rest => pairwiseContentInvariantFrom row rest && pairwiseContentInvariant rest

/-- `pairwiseContentInvariant` on a trace that covers the domain and whose rows all pass. -/
def checkFiniteTraceContentInvariance (rows : List CandidateEval) : Bool :=
  rowsReplayFiniteDomain rows &&
  rows.all CandidateEval.check &&
  pairwiseContentInvariant rows

end VerifiableTransformers.Certificate.Circuit

namespace VerifiableTransformers.Certificate.Properties

open VerifiableTransformers.Certificate.FiniteEval
open VerifiableTransformers.Certificate.Circuit

/-- Functional equivalence on the finite domain: the rows are the 256 canonical prompts and
each prefers its target. -/
def checkProjectedFunctionalEquivalence (rows : List CandidateEval) : Bool :=
  rowsReplayFiniteDomain rows && rows.all CandidateEval.check

/-- Changing content/filler tokens does not change the projected decision. -/
def checkProjectedContentInvariance (rows : List CandidateEval) : Bool :=
  checkFiniteTraceContentInvariance rows

/-- Both projected properties together. `Generated.UpstreamEvalTrace` proves them for the
upstream and TorchLean traces. -/
def checkTraceProperties (rows : List CandidateEval) : Bool :=
  checkProjectedFunctionalEquivalence rows &&
  checkProjectedContentInvariance rows

/-- Proposition form of `checkTraceProperties`. -/
def TracePropertiesAccepted (rows : List CandidateEval) : Prop :=
  checkTraceProperties rows = true

end VerifiableTransformers.Certificate.Properties
