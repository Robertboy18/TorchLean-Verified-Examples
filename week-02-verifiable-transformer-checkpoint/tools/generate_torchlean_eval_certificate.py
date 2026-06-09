#!/usr/bin/env python3
"""Generate a Lean eval certificate from a TorchLean-produced logits trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import lean_nat_list, lean_nested_nat_list, sha256_file


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
    params = checkpoint["params"]
    param_shapes = [p["shape"] for p in params]
    param_counts = [len(p["values"]) for p in params]
    first16 = params[0]["values"][:16] if params else []
    rows = trace["rows"]

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

Sources: {args.checkpoint} and {args.input}.
-/

import VerifiableTransformers.Certificate.FiniteEval
import VerifiableTransformers.Generated.UpstreamExportSummary

set_option maxRecDepth 4096

namespace VerifiableTransformers.Generated.TorchLeanEvalTrace

open VerifiableTransformers.Certificate.FiniteEval

def checkpointPath : String :=
  "{args.checkpoint}"

def checkpointSha256 : String :=
  "{checkpoint_sha}"

def formatTag : String :=
  "{checkpoint["format"]}"

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
  paramTensorShapes.length == 37 &&
  totalValueCount == 7712 &&
  totalValueCount ==
    VerifiableTransformers.Certificate.Weights.tensorEntryCount
      VerifiableTransformers.Generated.UpstreamExportSummary.exportSummary.tensorFields &&
  first16WeightBits.length == 16 &&
  VerifiableTransformers.Generated.UpstreamExportSummary.exportSummary.vocabSize == 32 &&
  VerifiableTransformers.Generated.UpstreamExportSummary.exportSummary.maxSeqLen == 6 &&
  VerifiableTransformers.Generated.UpstreamExportSummary.exportSummary.dModel == 16 &&
  VerifiableTransformers.Generated.UpstreamExportSummary.exportSummary.nLayers == 2

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
