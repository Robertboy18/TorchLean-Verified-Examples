/-
Generated circuit summaries for the quote/bracket circuits of the upstream-code checkpoint.

Sources: artifacts/upstream-small-gpt/circuits/<task>/circuit.json,
artifacts/upstream-small-gpt/circuits/<task>/verification/verification_results.json, and
artifacts/upstream-small-gpt/circuits/provenance.json, written by Neel Somani's
`scripts/small/extract.py` and `scripts/small/verify.py` at the pinned upstream commit.
Do not edit by hand; regenerate with tools/generate_circuit_summaries.py.
-/

import VerifiableTransformers.Certificate.Circuit

namespace VerifiableTransformers.Generated.UpstreamCircuitSummaries

open VerifiableTransformers.Certificate.Circuit

/-- Retained edges of the quote_close circuit (`circuit.json`). -/
def quoteEdges : List Edge :=
  [ edge "attn_0" "mlp_0"
  , edge "emb" "attn_0"
  , edge "mlp_0" "logits"
  ]

def quoteCircuit : CircuitSummary where
  task := .quoteClose
  nLayers := 2
  edges := quoteEdges
  ablation := "zero"
  metric := "candidate_kl"
  minAgreementMicros := 1000000
  thresholdMicros := 50000

def quoteVerification : VerificationSummary where
  task := .quoteClose
  upstreamCommit := "8e3c54bad2494c17c91109fdb58cf9e2f1c98efc"
  weightsSha256 := "ab61ac5dce70292ab9fa4dcd5265d98e04c34c5d89b809384ab297ef9e7abc34"
  checkpointSha256 := "f5b04594e8a0d9f833bd041e3a239bd865fc502819fc2eb0765e8119c3955b07"
  numInputs := 128
  candidateTokens := [9, 10]
  circuit := quoteCircuit
  pytorchStatus := "PASSED"
  pytorchExamplesChecked := 128
  pytorchFailures := 0
  functionalStatus := "VERIFIED"
  functionalVerifiedCount := 128
  functionalTotalSequences := 128
  functionalTimeouts := 0
  functionalErrors := 0
  functionalCounterexamples := 0
  edgeStatus := "VERIFIED"
  edgeTotal := 3
  edgeNecessary := 3
  edgeUnnecessary := 0
  edgeUnresolved := 0
  edgeTimeouts := 0
  edgeErrors := 0
  robustnessStatus := "VERIFIED"
  robustnessVerifiedCount := 128
  robustnessTimeouts := 0
  robustnessErrors := 0
  robustnessEpsilonMicros := 10000
  robustnessViolations := 0
  robustnessDecisionViolations := 0
  robustnessBranchUnstable := 0

/-- Retained edges of the bracket_type circuit (`circuit.json`). -/
def bracketEdges : List Edge :=
  [ edge "attn_0" "mlp_0"
  , edge "emb" "attn_0"
  , edge "mlp_0" "logits"
  ]

def bracketCircuit : CircuitSummary where
  task := .bracketType
  nLayers := 2
  edges := bracketEdges
  ablation := "zero"
  metric := "candidate_kl"
  minAgreementMicros := 1000000
  thresholdMicros := 20000

def bracketVerification : VerificationSummary where
  task := .bracketType
  upstreamCommit := "8e3c54bad2494c17c91109fdb58cf9e2f1c98efc"
  weightsSha256 := "ab61ac5dce70292ab9fa4dcd5265d98e04c34c5d89b809384ab297ef9e7abc34"
  checkpointSha256 := "f5b04594e8a0d9f833bd041e3a239bd865fc502819fc2eb0765e8119c3955b07"
  numInputs := 128
  candidateTokens := [13, 14]
  circuit := bracketCircuit
  pytorchStatus := "PASSED"
  pytorchExamplesChecked := 128
  pytorchFailures := 0
  functionalStatus := "VERIFIED"
  functionalVerifiedCount := 128
  functionalTotalSequences := 128
  functionalTimeouts := 0
  functionalErrors := 0
  functionalCounterexamples := 0
  edgeStatus := "VERIFIED"
  edgeTotal := 3
  edgeNecessary := 3
  edgeUnnecessary := 0
  edgeUnresolved := 0
  edgeTimeouts := 0
  edgeErrors := 0
  robustnessStatus := "VERIFIED"
  robustnessVerifiedCount := 128
  robustnessTimeouts := 0
  robustnessErrors := 0
  robustnessEpsilonMicros := 10000
  robustnessViolations := 0
  robustnessDecisionViolations := 0
  robustnessBranchUnstable := 0

/- Both summaries name the pinned checkpoint, use real graph edges, and report statuses that
agree with their own counts. -/
theorem circuitSummaries_ok :
    checkVerificationSummary quoteVerification = true ∧
    checkVerificationSummary bracketVerification = true := by
  decide +kernel

/- What the PyTorch check and the three SMT properties reported for each circuit. -/
theorem circuitSummaries_outcome :
    quoteVerification.allVerified = true ∧
    bracketVerification.allVerified = true := by
  decide +kernel

end VerifiableTransformers.Generated.UpstreamCircuitSummaries
