#!/usr/bin/env python3
"""Generate a Lean summary certificate from `smt_weights.json`.

The generated Lean module records the export's metadata, operator tags, and tensor
shapes (not the floating-point arrays), the upstream commit from the training run's
`provenance.json`, and the small-model configuration from the run's `config.json`.
Everything is copied from those files; the generator does not decide whether they are
right. Lean checks the summary with `Certificate.Weights.checkExportSummary` and checks
the saved configuration against `Spec.UpstreamSmallGPT.repoDefaultSmallConfig`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    GENERATED_DIR,
    RUN_DIR,
    artifact_display_path,
    lean_rat,
    lean_string,
    load_json_exact,
    resolve_path,
    sha256_file,
)


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

# `config.json` keys and their `SmallGPTConfig` fields. The two keys left out do not
# change the trained function: the six-token task never uses an EOS token, and the
# initializer range only affects the starting weights.
CONFIG_FIELDS = [
    ("vocab_size", "vocabSize", "nat"),
    ("max_seq_len", "maxSeqLen", "nat"),
    ("pad_token_id", "padTokenId", "nat"),
    ("bos_token_id", "bosTokenId", "nat"),
    ("d_model", "dModel", "nat"),
    ("n_layers", "nLayers", "nat"),
    ("n_heads", "nHeads", "nat"),
    ("d_mlp", "dMlp", "nat"),
    ("norm_variant", "normVariant", "string"),
    ("attn_variant", "attnVariant", "string"),
    ("activation_variant", "activationVariant", "string"),
    ("norm_l1_low_per_dim", "normL1LowPerDim", "rat"),
    ("norm_l1_high_per_dim", "normL1HighPerDim", "rat"),
    ("leaky_relu_negative_slope", "leakyReluNegativeSlope", "rat"),
    ("tie_embeddings", "tieEmbeddings", "bool"),
    ("use_bias", "useBias", "bool"),
    ("dropout", "dropout", "rat"),
    ("attn_pdrop", "attnPdrop", "rat"),
    ("resid_pdrop", "residPdrop", "rat"),
    ("embd_pdrop", "embdPdrop", "rat"),
]
CONFIG_IGNORED = {"eos_token_id", "initializer_range"}


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


def lean_config_value(value: Any, kind: str) -> str:
    if kind == "nat":
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"expected a natural number, got {value!r}")
        return str(value)
    if kind == "string":
        return lean_string(value)
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"expected a Boolean, got {value!r}")
        return "true" if value else "false"
    return lean_rat(value)


def config_block(config: dict) -> str:
    unknown = set(config) - {k for k, _, _ in CONFIG_FIELDS} - CONFIG_IGNORED
    if unknown:
        raise ValueError(f"config.json has keys this generator does not map: {sorted(unknown)}")
    lines = [f"  {field} := {lean_config_value(config[key], kind)}" for key, field, kind in CONFIG_FIELDS]
    return "\n".join(lines)


def generate(input_path: Path, config_path: Path, provenance_path: Path, module_namespace: str) -> str:
    data = load_json_exact(input_path)
    config = load_json_exact(config_path)
    provenance = json.loads(provenance_path.read_text())

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
  {artifact_display_path(config_path)}
  {artifact_display_path(provenance_path)}

Lean records the metadata, operator tags, and tensor-shape summaries here, while the full
floating-point arrays live in the checkpoint payload file. The generator records the
export's SHA-256, and Lean compares it with `Certificate.Fingerprint` without recomputing it
from the checkpoint bytes. Do not edit by hand; regenerate with
tools/generate_export_certificate.py.
-/

import VerifiableTransformers.Certificate.Weights

namespace {module_namespace}

open VerifiableTransformers.Certificate.Weights
open VerifiableTransformers.Spec.UpstreamSmallGPT

def tensorFields : List TensorSummary :=
{tensor_block}

def exportSummary : ExportSummary where
  sourcePath := {lean_string(source_path)}
  sha256 := {lean_string(sha)}
  upstreamCommit := {lean_string(provenance["upstream_commit"])}
  fieldCount := {len(data)}
  vocabSize := {data["vocab_size"]}
  maxSeqLen := {len(data["wpe"])}
  dModel := {data["d_model"]}
  nLayers := {data["n_layers"]}
  nHeads := {data["n_heads"]}
  dMlp := {data["d_ff"]}
  headDim := {data["head_dim"]}
  normVariant := {lean_string(data["norm_variant"])}
  attnVariant := {lean_string(data["attn_variant"])}
  activationVariant := {lean_string(data["activation_variant"])}
  halfLow := {lean_rat(data["half_low"])}
  halfHigh := {lean_rat(data["half_high"])}
  tensorFields := tensorFields

/- The imported export summary passes the checker. -/
theorem exportSummary_ok :
    checkExportSummary exportSummary = true := by
  decide +kernel

/-- The small-model configuration saved by the training run (`config.json`). -/
def trainedConfig : SmallGPTConfig where
{config_block(config)}

/- The run's saved configuration is the one transcribed in `Spec.UpstreamSmallGPT`. -/
theorem trainedConfig_eq_repoDefault :
    trainedConfig = repoDefaultSmallConfig := by
  decide +kernel

end {module_namespace}
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RUN_DIR / "smt_weights.json")
    parser.add_argument("--config", type=Path, default=RUN_DIR / "config.json")
    parser.add_argument("--provenance", type=Path, default=RUN_DIR / "provenance.json")
    parser.add_argument("--output", type=Path, default=GENERATED_DIR / "UpstreamExportSummary.lean")
    parser.add_argument(
        "--namespace",
        default="VerifiableTransformers.Generated.UpstreamExportSummary",
    )
    args = parser.parse_args()
    output = resolve_path(args.output)
    output.write_text(generate(resolve_path(args.input), resolve_path(args.config),
                               resolve_path(args.provenance), args.namespace))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
