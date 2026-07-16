#!/usr/bin/env python3
"""Generate a Lean summary certificate from `smt_weights.json`.

The generated Lean module records metadata and tensor shapes, not the full
floating-point arrays.  Lean then checks the summary with
`Certificate.Weights.checkExportSummary`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import artifact_display_path, sha256_file


META_KEYS = {
    "d_model",
    "n_layers",
    "n_heads",
    "vocab_size",
    "d_ff",
    "head_dim",
    "norm_variant",
    "attn_variant",
    "activation_variant",
    "half_low",
    "half_high",
}


def shape_of(value: Any) -> list[int]:
    if isinstance(value, list):
        if not value:
            return [0]
        return [len(value)] + shape_of(value[0])
    return []


def lean_shape(shape: list[int]) -> str:
    return "[" + ", ".join(str(x) for x in shape) + "]"


def lean_tensor_name(key: str) -> str:
    if key == "wte":
        return ".wte"
    if key == "wpe":
        return ".wpe"
    if key == "final_norm_gamma":
        return ".finalNormGamma"
    if key == "final_norm_beta":
        return ".finalNormBeta"
    if key == "lm_head":
        return ".lmHead"

    parts = key.split("_")
    if len(parts) < 3:
        raise ValueError(f"unsupported tensor key: {key}")

    if parts[0] == "attn":
        layer = int(parts[1])
        suffix = "_".join(parts[2:])
        mapping = {
            "W_q": "attnWq",
            "W_k": "attnWk",
            "W_v": "attnWv",
            "b_q": "attnBq",
            "b_k": "attnBk",
            "b_v": "attnBv",
            "W_o": "attnWo",
            "b_o": "attnBo",
            "norm_gamma": "attnNormGamma",
            "norm_beta": "attnNormBeta",
        }
        if suffix in mapping:
            return f"(.{mapping[suffix]} {layer})"

    if parts[0] == "mlp":
        layer = int(parts[1])
        suffix = "_".join(parts[2:])
        mapping = {
            "W_up": "mlpWUp",
            "b_up": "mlpBUp",
            "W_down": "mlpWDown",
            "b_down": "mlpBDown",
            "norm_gamma": "mlpNormGamma",
            "norm_beta": "mlpNormBeta",
        }
        if suffix in mapping:
            return f"(.{mapping[suffix]} {layer})"

    raise ValueError(f"unsupported tensor key: {key}")


def rat_literal(value: float) -> str:
    # The upstream default exports 4.4 and 8.4.  Use decimal string conversion
    # only for this metadata bridge; full weight import should use exact decimal
    # parsing instead of floats.
    if abs(value - 4.4) < 1e-9:
        return "22 / 5"
    if abs(value - 8.4) < 1e-9:
        return "42 / 5"
    raise ValueError(f"unsupported half-band value for exact Lean literal: {value!r}")


def generate(input_path: Path, module_namespace: str) -> str:
    data = json.loads(input_path.read_text())
    expected_variants = {
        "norm_variant": "signed_l1_band_norm",
        "attn_variant": "sparsemax",
        "activation_variant": "leaky_relu",
    }
    for key, expected in expected_variants.items():
        actual = data.get(key)
        if actual != expected:
            raise ValueError(f"expected {key}={expected!r}, got {actual!r}")

    sha = sha256_file(input_path)
    source_path = artifact_display_path(input_path)
    tensor_lines = []
    for key, value in data.items():
        if key in META_KEYS:
            continue
        tensor_lines.append(f"  , tensor {lean_tensor_name(key)} {lean_shape(shape_of(value))}")

    if tensor_lines:
        first = tensor_lines[0].replace("  ,", "  [", 1)
        rest = tensor_lines[1:]
        tensor_block = "\n".join([first, *rest, "  ]"])
    else:
        tensor_block = "  []"

    return f'''/-
Generated summary certificate for:

  {source_path}

Lean records the metadata and tensor-shape summaries here, while the full
floating-point arrays live in the checkpoint values file. The SHA-256 field ties
this summary to the exact exported checkpoint data.
-/

import VerifiableTransformers.Certificate.Weights

namespace {module_namespace}

open VerifiableTransformers.Certificate.Weights

def tensorFields : List TensorSummary :=
{tensor_block}

def exportSummary : ExportSummary where
  sourcePath := "{source_path}"
  sha256 := "{sha}"
  fieldCount := {len(data)}
  vocabSize := {data["vocab_size"]}
  maxSeqLen := {len(data["wpe"])}
  dModel := {data["d_model"]}
  nLayers := {data["n_layers"]}
  nHeads := {data["n_heads"]}
  dMlp := {data["d_ff"]}
  headDim := {data["head_dim"]}
  normVariant := .signedL1BandNorm
  attnVariant := .sparsemax
  activationVariant := .leakyRelu
  halfLow := {rat_literal(data["half_low"])}
  halfHigh := {rat_literal(data["half_high"])}
  tensorFields := tensorFields

/- The imported export summary passes the checker. -/
theorem exportSummary_ok :
    checkExportSummary exportSummary = true := by
  simp [checkExportSummary, exportSummary, tensorFields,
    tensor, shapeEntries, TensorSummary.wellShaped,
    tensorEntryCount, repoDefaultTensorFields]

end {module_namespace}
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--namespace",
        default="VerifiableTransformers.Generated.UpstreamExportSummary",
    )
    args = parser.parse_args()
    args.output.write_text(generate(args.input, args.namespace))


if __name__ == "__main__":
    main()
