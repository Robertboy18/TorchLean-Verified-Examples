#!/usr/bin/env python3
"""Generate Lean constants for replaying the exact small GPT forward pass.

The input is the `smt_weights.json` export from
neelsomani/verifiable-transformers.  The generated module contains
the full numeric parameter values, not just hashes or logits, so a Lean program
can independently replay the trained model on the finite task domain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "artifacts/upstream-small-gpt/smt_weights.json"
DEFAULT_OUT = ROOT / "VerifiableTransformers/Generated/UpstreamCheckpointPayload.lean"


FIELDS = [
    "wte",
    "wpe",
    "attn_0_norm_gamma",
    "attn_0_norm_beta",
    "attn_0_W_q",
    "attn_0_b_q",
    "attn_0_W_k",
    "attn_0_b_k",
    "attn_0_W_v",
    "attn_0_b_v",
    "attn_0_W_o",
    "attn_0_b_o",
    "mlp_0_norm_gamma",
    "mlp_0_norm_beta",
    "mlp_0_W_up",
    "mlp_0_b_up",
    "mlp_0_W_down",
    "mlp_0_b_down",
    "attn_1_norm_gamma",
    "attn_1_norm_beta",
    "attn_1_W_q",
    "attn_1_b_q",
    "attn_1_W_k",
    "attn_1_b_k",
    "attn_1_W_v",
    "attn_1_b_v",
    "attn_1_W_o",
    "attn_1_b_o",
    "mlp_1_norm_gamma",
    "mlp_1_norm_beta",
    "mlp_1_W_up",
    "mlp_1_b_up",
    "mlp_1_W_down",
    "mlp_1_b_down",
    "final_norm_gamma",
    "final_norm_beta",
    "lm_head",
]


def lean_float(x: Any) -> str:
    value = float(x)
    if value == 0.0:
        return "0.0"
    text = format(value, ".17g")
    if "e" not in text and "." not in text:
        text += ".0"
    return text


def lean_array(value: Any, indent: int = 0) -> str:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            inner = ",\n".join(" " * (indent + 2) + lean_array(row, indent + 2) for row in value)
            return f"#[\n{inner}\n{' ' * indent}]"
        return "#[" + ", ".join(lean_float(x) for x in value) + "]"
    raise TypeError(f"expected list, got {type(value)!r}")


def lean_name(key: str) -> str:
    return key.replace("_", "")


def shape_of(value: Any) -> str:
    if isinstance(value, list) and value and isinstance(value[0], list):
        return f"({len(value)}, {len(value[0])})"
    if isinstance(value, list):
        return f"({len(value)})"
    return "()"


def shape_check_expr(field: str, value: Any) -> str:
    lean = lean_name(field)
    if value and isinstance(value[0], list):
        return f"matrixShape {lean} {len(value)} {len(value[0])}"
    return f"vecShape {lean} {len(value)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.input.resolve()
    out = args.output.resolve()
    data = json.loads(weights.read_text())
    lines = [
        "/-",
        "Generated full-parameter values for Lean-side replay of the exact",
        "small GPT-style model trained/exported by neelsomani/verifiable-transformers.",
        "",
        f"  source: {weights.relative_to(ROOT) if weights.is_relative_to(ROOT) else weights}",
        "",
        "Do not edit by hand; regenerate with tools/generate_full_forward_weights.py.",
        "-/",
        "",
        "set_option maxRecDepth 8192",
        "",
        "namespace VerifiableTransformers.Generated.UpstreamCheckpointPayload",
        "",
        "abbrev Vec := Array Float",
        "abbrev Matrix := Array (Array Float)",
        "",
        "/-- Check that a generated vector has the expected exported length. -/",
        "def vecShape (xs : Vec) (n : Nat) : Bool :=",
        "  xs.size == n",
        "",
        "/-- Check that a generated matrix has the expected exported row/column shape. -/",
        "def matrixShape (M : Matrix) (rows cols : Nat) : Bool :=",
        "  M.size == rows && M.all (fun row => row.size == cols)",
        "",
        f"def dModel : Nat := {data['d_model']}",
        f"def dFF : Nat := {data['d_ff']}",
        f"def nLayers : Nat := {data['n_layers']}",
        f"def nHeads : Nat := {data['n_heads']}",
        f"def vocabSize : Nat := {data['vocab_size']}",
        "def seqLen : Nat := 6",
        f"def halfLow : Float := {lean_float(data['half_low'])}",
        f"def halfHigh : Float := {lean_float(data['half_high'])}",
        "def leakySlope : Float := 0.01",
        "",
    ]

    for field in FIELDS:
        value = data[field]
        typ = "Matrix" if value and isinstance(value[0], list) else "Vec"
        lines.append(f"/-- Shape {shape_of(value)} from `{field}`. -/")
        lines.append(f"def {lean_name(field)} : {typ} :=")
        lines.append(lean_array(value))
        lines.append("")

    lines.append("/-- Every generated tensor payload matches the exported shape metadata. -/")
    lines.append("def checkpointPayloadOk : Bool :=")
    checks = [shape_check_expr(field, data[field]) for field in FIELDS]
    for idx, check in enumerate(checks):
        suffix = " &&" if idx + 1 < len(checks) else ""
        lines.append(f"  {check}{suffix}")
    lines.append("")
    lines.append("theorem checkpointPayload_ok :")
    lines.append("    checkpointPayloadOk = true := by")
    lines.append("  native_decide")
    lines.append("")

    lines.append("end VerifiableTransformers.Generated.UpstreamCheckpointPayload")
    lines.append("")
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
