#!/usr/bin/env python3
"""Generate the Lean circuit summaries from the upstream extraction and SMT outputs.

Reads, for each task, the `circuit.json` written by Neel Somani's
`scripts/small/extract.py` and the `verification/verification_results.json` written by
his `scripts/small/verify.py`, plus `circuits/provenance.json` from
`experiments/run_upstream_circuits.py`. Every value in the generated module is copied from
those files. Lean checks the summaries with `Certificate.Circuit.checkVerificationSummary`,
and the Lean Float replay recomputes the circuit decisions and edge-necessity witnesses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import GENERATED_DIR, RUN_DIR, lean_nat_list, lean_string, resolve_path

TASKS = [("quote_close", "quoteClose", "quote"), ("bracket_type", "bracketType", "bracket")]


def micros(x) -> int:
    return int(round(float(x) * 1_000_000))


def property_block(results: dict) -> dict:
    """Flatten verify.py's property list into the Lean summary fields."""
    props = {p["property"]: p for p in results["properties"]}

    def get(name: str) -> dict:
        return props.get(name, {"status": "NOT_RUN"})

    py = get("pytorch_circuit_validation")
    fe = get("projected_functional_equivalence")
    en = get("projected_edge_necessity")
    rb = get("projected_continuous_robustness")
    return {
        "pytorchStatus": py["status"],
        "pytorchExamplesChecked": py.get("examples_checked", 0),
        "pytorchFailures": py.get("num_failures", 0),
        "functionalStatus": fe["status"],
        "functionalVerifiedCount": fe.get("verified_count", 0),
        "functionalTotalSequences": fe.get("total_sequences", 0),
        "functionalTimeouts": fe.get("timeout_count", 0),
        "functionalErrors": fe.get("error_count", 0),
        "functionalCounterexamples": fe.get("num_counterexamples", 0),
        "edgeStatus": en["status"],
        "edgeTotal": en.get("total_edges", 0),
        "edgeNecessary": en.get("necessary_edges", 0),
        "edgeUnnecessary": en.get("unnecessary_edges_found", 0),
        "edgeUnresolved": en.get("unresolved_edges_found", 0),
        "edgeTimeouts": en.get("timeout_count", 0),
        "edgeErrors": en.get("error_count", 0),
        "robustnessStatus": rb["status"],
        "robustnessVerifiedCount": rb.get("verified_count", 0),
        "robustnessTimeouts": rb.get("timeout_count", 0),
        "robustnessErrors": rb.get("error_count", 0),
        "robustnessEpsilonMicros": micros(rb.get("certified_epsilon", 0)),
        "robustnessViolations": rb.get("num_violations", 0),
        "robustnessDecisionViolations": rb.get("num_decision_violations", 0),
        "robustnessBranchUnstable": rb.get("num_branch_unstable", 0),
    }


def lean_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return lean_string(v)


def emit_task(task: str, lean_task: str, short: str, circuit_dir: Path, provenance: dict) -> tuple[str, bool]:
    circuit = json.loads((circuit_dir / "circuit.json").read_text())
    results = json.loads((circuit_dir / "verification" / "verification_results.json").read_text())
    if circuit["task"] != task or results["task"] != task:
        raise SystemExit(f"{circuit_dir} does not hold the {task} circuit")
    edges = "\n  , ".join(f"edge {lean_string(e['source'])} {lean_string(e['target'])}"
                          for e in circuit["edges"])
    fields = property_block(results)
    sha = provenance["sha256"]
    body = f'''/-- Retained edges of the {task} circuit (`circuit.json`). -/
def {short}Edges : List Edge :=
  [ {edges}
  ]

def {short}Circuit : CircuitSummary where
  task := .{lean_task}
  nLayers := {circuit["n_layers"]}
  edges := {short}Edges
  ablation := {lean_string(circuit["ablation"])}
  metric := {lean_string(circuit["metric"])}
  minAgreementMicros := {micros(circuit["min_agreement"])}
  thresholdMicros := {micros(circuit["threshold"])}

def {short}Verification : VerificationSummary where
  task := .{lean_task}
  upstreamCommit := {lean_string(provenance["upstream_commit"])}
  weightsSha256 := {lean_string(sha["smt_weights.json"])}
  checkpointSha256 := {lean_string(sha["checkpoint-final/model.safetensors"])}
  numInputs := {results["num_inputs"]}
  candidateTokens := {lean_nat_list(results["candidate_tokens"])}
  circuit := {short}Circuit
'''
    body += "".join(f"  {k} := {lean_value(v)}\n" for k, v in fields.items())
    all_verified = (fields["pytorchStatus"] == "PASSED" and fields["functionalStatus"] == "VERIFIED"
                    and fields["edgeStatus"] == "VERIFIED" and fields["robustnessStatus"] == "VERIFIED")
    return body, all_verified


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--circuits", type=Path, default=RUN_DIR / "circuits")
    p.add_argument("--output", type=Path, default=GENERATED_DIR / "UpstreamCircuitSummaries.lean")
    args = p.parse_args()
    circuits = resolve_path(args.circuits)
    output = resolve_path(args.output)
    provenance = json.loads((circuits / "provenance.json").read_text())

    blocks, outcomes = [], []
    for task, lean_task, short in TASKS:
        body, ok = emit_task(task, lean_task, short, circuits / task, provenance)
        blocks.append(body)
        outcomes.append((short, ok))

    outcome_names = " ∧\n    ".join(f"{s}Verification.allVerified = {'true' if ok else 'false'}"
                                   for s, ok in outcomes)
    text = f'''/-
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

{chr(10).join(blocks)}
/- Both summaries name the pinned checkpoint, use real graph edges, and report statuses that
agree with their own counts. -/
theorem circuitSummaries_ok :
    checkVerificationSummary quoteVerification = true ∧
    checkVerificationSummary bracketVerification = true := by
  decide +kernel

/- What the PyTorch check and the three SMT properties reported for each circuit. -/
theorem circuitSummaries_outcome :
    {outcome_names} := by
  decide +kernel

end VerifiableTransformers.Generated.UpstreamCircuitSummaries
'''
    output.write_text(text)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
