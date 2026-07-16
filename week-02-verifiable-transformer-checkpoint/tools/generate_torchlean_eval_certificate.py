#!/usr/bin/env python3
"""Generate a Lean eval certificate from a TorchLean-produced logits trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import artifact_display_path, lean_nat_list, lean_nested_nat_list, sha256_file


# Versioned native-model contract. Regeneration must fail if the public GPT constructor changes
# dimensions or parameter order; accepting that change requires reviewing and updating this table.
MODEL_TAG = "torchlean-causal-softmax-gpt-v1"
EXPECTED_DIMENSIONS = {
    "batch": 16,
    "seqLen": 6,
    "vocab": 32,
    "dModel": 16,
    "layers": 2,
    "heads": 1,
    "ffnHidden": 64,
}
EXPECTED_PARAM_SHAPES = [
    [32, 16], [6, 16],
    [16, 16], [16, 16], [16, 16], [16, 16], [16], [16],
    [64, 16], [64], [16, 64], [16], [16], [16],
    [16, 16], [16, 16], [16, 16], [16, 16], [16], [16],
    [64, 16], [64], [16, 64], [16], [16], [16],
    [16], [16], [32, 16], [32],
]


def micros(x: float) -> int:
    return int(round(x * 1_000_000))


def lean_task(task: str) -> str:
    if task == "quoteClose":
        return ".quoteClose"
    if task == "bracketType":
        return ".bracketType"
    raise ValueError(f"unknown task {task!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    trace = json.loads(args.input.read_text())
    checkpoint = json.loads(args.checkpoint.read_text())
    checkpoint_sha = sha256_file(args.checkpoint)
    checkpoint_path = artifact_display_path(args.checkpoint)
    trace_path = artifact_display_path(args.input)
    params = checkpoint["params"]
    param_shapes = [p["shape"] for p in params]
    param_counts = [len(p["values"]) for p in params]
    first16 = params[0]["values"][:16] if params else []
    rows = trace["rows"]

    if checkpoint.get("format") != "torchlean_paramlist_bits_v1":
        raise ValueError("unexpected TorchLean parameter format")
    if trace.get("producer") != "train_torchlean_small_gpt":
        raise ValueError("unexpected eval-trace producer")
    if trace.get("model") != MODEL_TAG:
        raise ValueError(f"unexpected model tag: {trace.get('model')!r}")
    if trace.get("dimensions") != EXPECTED_DIMENSIONS:
        raise ValueError(f"unexpected model dimensions: {trace.get('dimensions')!r}")
    if param_shapes != EXPECTED_PARAM_SHAPES:
        raise ValueError("checkpoint parameter shapes do not match the native TorchLean GPT")
    if len(rows) != 256:
        raise ValueError(f"expected 256 finite-domain rows, found {len(rows)}")

    lean_rows: list[str] = []
    for row in rows:
        lean_rows.append(
            "  { task := "
            + lean_task(row["task"])
            + ", input := "
            + lean_nat_list(row["input"])
            + f", target := {row['target']}"
            + f", alternate := {row['alternate']}"
            + f", targetScoreMicros := {micros(float(row['targetScore']))}"
            + f", alternateScoreMicros := {micros(float(row['alternateScore']))}"
            + " }"
        )

    rows_text = ",\n".join(lean_rows)

    contents = f'''/-
Generated TorchLean CUDA checkpoint summary and eval trace.

Lean checks two facts here: the exported parameter summary has the expected
shape/count metadata, and the finite eval rows satisfy the projected
quote/bracket property.

Sources: {checkpoint_path} and {trace_path}.
-/

import VerifiableTransformers.Certificate.FiniteEval

set_option maxRecDepth 4096

namespace VerifiableTransformers.Generated.TorchLeanEvalTrace

open VerifiableTransformers.Certificate.FiniteEval

def checkpointPath : String :=
  "{checkpoint_path}"

def checkpointSha256 : String :=
  "{checkpoint_sha}"

def formatTag : String :=
  "{checkpoint["format"]}"

def modelTag : String :=
  "{MODEL_TAG}"

/-- TorchLean runtime parameter order for `TorchLean.TrainSmallGPT.mkTrainableModel`. -/
def paramTensorShapes : List (List Nat) :=
  {lean_nested_nat_list(param_shapes)}

def paramValueCounts : List Nat :=
  {lean_nat_list(param_counts)}

def first16WeightBits : List Nat :=
  {lean_nat_list(first16)}

def totalValueCount : Nat :=
  paramValueCounts.foldl (fun acc n => acc + n) 0

def checkpointSummaryOk : Bool :=
  formatTag == "torchlean_paramlist_bits_v1" &&
  modelTag == "torchlean-causal-softmax-gpt-v1" &&
  paramTensorShapes == {lean_nested_nat_list(EXPECTED_PARAM_SHAPES)} &&
  paramValueCounts.length == 30 &&
  totalValueCount == 7616 &&
  first16WeightBits.length == 16

theorem checkpointSummary_ok :
    checkpointSummaryOk = true := by
  rfl

def evalRows : List CandidateEval :=
[
{rows_text}
]

def evalCertificate : EvalCertificate where
  sourceWeightsSha256 := "{checkpoint_sha}"
  checkpointPath := checkpointPath
  scoreScale := 1000000
  rows := evalRows

theorem evalTrace_ok :
    checkEvalCertificateWithSha
      checkpointSha256
      evalCertificate = true := by
  rfl

end VerifiableTransformers.Generated.TorchLeanEvalTrace
'''
    args.output.write_text(contents)


if __name__ == "__main__":
    main()
