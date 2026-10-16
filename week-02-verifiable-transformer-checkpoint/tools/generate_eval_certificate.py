#!/usr/bin/env python3
"""Generate the finite-domain evaluation certificate for the upstream-code checkpoint.

The checkpoint is loaded with Neel Somani's own `scripts/small/extract.py:load_model`
(pinned upstream commit) and run on the CPU over the 256 prompts of
`scripts/small/dataset.py:get_eval_dataset`. The generator writes

* `artifacts/upstream-small-gpt/eval_trace.json`: every prompt's float32 target and
  alternate logits, plus the full-vocabulary argmax that upstream's `accuracy` metric uses;
* `VerifiableTransformers/Generated/UpstreamEvalTrace.lean`: the same rows as integer
  micro-logits, with the SHA-256 of `smt_weights.json` and of `model.safetensors`.

Before scoring, it checks that `model.safetensors` holds bit for bit the values exported to
`smt_weights.json` (after GPT-2's Conv1D transpose). So this trace, the export summary, and
the Lean replay payload describe one set of weights. Lean then checks the domain, the
targets, the margins, and the fingerprints; this script does not reject a bad row itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import (
    GENERATED_DIR,
    RUN_DIR,
    artifact_display_path,
    check_upstream_checkout,
    lean_nat_list,
    resolve_path,
    sha256_file,
)

SCALE = 1_000_000


def check_export_matches_checkpoint(checkpoint: Path, weights: Path) -> None:
    """Fail unless `smt_weights.json` is exactly the tensors stored in `model.safetensors`."""
    import numpy as np
    from safetensors.numpy import load_file

    st = load_file(str(checkpoint / "model.safetensors"))
    w = json.loads(weights.read_text())
    d = int(w["d_model"])
    pairs = [("wte", st["transformer.wte.weight"]), ("wpe", st["transformer.wpe.weight"]),
             ("lm_head", st["lm_head.weight"]),
             ("final_norm_gamma", st["transformer.ln_f.gamma"]),
             ("final_norm_beta", st["transformer.ln_f.beta"])]
    for i in range(int(w["n_layers"])):
        h = f"transformer.h.{i}"
        qkv_w, qkv_b = st[f"{h}.attn.c_attn.weight"], st[f"{h}.attn.c_attn.bias"]
        pairs += [
            (f"attn_{i}_W_q", qkv_w[:, :d].T), (f"attn_{i}_W_k", qkv_w[:, d:2 * d].T),
            (f"attn_{i}_W_v", qkv_w[:, 2 * d:].T), (f"attn_{i}_b_q", qkv_b[:d]),
            (f"attn_{i}_b_k", qkv_b[d:2 * d]), (f"attn_{i}_b_v", qkv_b[2 * d:]),
            (f"attn_{i}_W_o", st[f"{h}.attn.c_proj.weight"].T),
            (f"attn_{i}_b_o", st[f"{h}.attn.c_proj.bias"]),
            (f"mlp_{i}_W_up", st[f"{h}.mlp.c_fc.weight"].T),
            (f"mlp_{i}_b_up", st[f"{h}.mlp.c_fc.bias"]),
            (f"mlp_{i}_W_down", st[f"{h}.mlp.c_proj.weight"].T),
            (f"mlp_{i}_b_down", st[f"{h}.mlp.c_proj.bias"]),
            (f"attn_{i}_norm_gamma", st[f"{h}.ln_1.gamma"]),
            (f"attn_{i}_norm_beta", st[f"{h}.ln_1.beta"]),
            (f"mlp_{i}_norm_gamma", st[f"{h}.ln_2.gamma"]),
            (f"mlp_{i}_norm_beta", st[f"{h}.ln_2.beta"]),
        ]
    tensor_keys = {k for k, v in w.items() if isinstance(v, list)}
    if tensor_keys != {k for k, _ in pairs}:
        raise SystemExit("smt_weights.json tensor names do not match the checkpoint layout")
    for key, tensor in pairs:
        if not np.array_equal(np.asarray(w[key], dtype=np.float64), tensor.astype(np.float64)):
            raise SystemExit(f"smt_weights.json field {key} differs from model.safetensors")


def emit_header(checkpoint: str, weights: str) -> str:
    return f"""/-
Generated finite-domain evaluation certificate for:

  checkpoint: {checkpoint}
  weights:    {weights}

Scores are final-position candidate logits from the trained checkpoint (PyTorch float32 on
the CPU), scaled by {SCALE} and rounded to integers. Lean checks the task-domain shape, a
strictly positive projected candidate margin for every row, and the two fingerprints. Do not
edit by hand; regenerate with tools/generate_eval_certificate.py.
-/

import VerifiableTransformers.Certificate.FiniteEval
import VerifiableTransformers.Certificate.Circuit
import VerifiableTransformers.Generated.TorchLeanEvalTrace

set_option maxRecDepth 4096

namespace VerifiableTransformers.Generated.UpstreamEvalTrace

open VerifiableTransformers.Certificate
open VerifiableTransformers.Certificate.FiniteEval
open VerifiableTransformers.Certificate.Properties

"""


FOOTER = """
/- The trace passes the finite-domain checker, including the pinned export fingerprint. -/
theorem evalTrace_ok :
    checkEvalCertificate evalCertificate = true := by
  rfl

/- The scores came from the pinned `model.safetensors`. -/
theorem checkpointSha256_pinned :
    checkpointSha256 = Fingerprint.modelSafetensorsSha256 := by
  decide

/- Both finite traces satisfy the projected properties used in the checkpoint claim. -/
theorem generatedTraceProperties_ok :
    TracePropertiesAccepted
      evalCertificate.rows ∧
    TracePropertiesAccepted
      VerifiableTransformers.Generated.TorchLeanEvalTrace.evalCertificate.rows := by
  exact ⟨by rfl, by rfl⟩

end VerifiableTransformers.Generated.UpstreamEvalTrace
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True,
                   help="neelsomani/verifiable-transformers checkout at the pinned commit")
    p.add_argument("--checkpoint", type=Path, default=RUN_DIR / "checkpoint-final")
    p.add_argument("--weights", type=Path, default=RUN_DIR / "smt_weights.json")
    p.add_argument("--trace-json", type=Path, default=RUN_DIR / "eval_trace.json")
    p.add_argument("--output", type=Path, default=GENERATED_DIR / "UpstreamEvalTrace.lean")
    args = p.parse_args()

    repo = resolve_path(args.repo)
    check_upstream_checkout(repo)
    checkpoint = resolve_path(args.checkpoint)
    weights = resolve_path(args.weights)
    trace_json = resolve_path(args.trace_json)
    output = resolve_path(args.output)

    check_export_matches_checkpoint(checkpoint, weights)

    sys.path.insert(0, str(repo))
    import torch

    from scripts.small import vocab
    from scripts.small.dataset import get_eval_dataset
    from scripts.small.extract import load_model
    from scripts.small.extract_weights import load_small_config

    torch.set_num_threads(1)
    config = load_small_config(str(checkpoint))
    device = torch.device("cpu")
    model = load_model(str(checkpoint), config, device)
    model.eval()

    rows: list[str] = []
    trace_rows: list[dict] = []
    with torch.no_grad():
        for task_name, lean_task in [
            ("quote_close", "quoteClose"),
            ("bracket_type", "bracketType"),
        ]:
            task_token = vocab.TASK_NAME_TO_TOKEN[task_name]
            candidates = sorted(vocab.get_candidates(task_token))
            for ex in get_eval_dataset(task_name):
                input_ids = ex["input_ids"]
                target = int(ex["target"])
                alternate = [c for c in candidates if c != target][0]
                tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
                logits = model(tensor).logits[0, -1, :]
                target_logit = float(logits[target].item())
                alternate_logit = float(logits[alternate].item())
                argmax = int(logits.argmax().item())
                trace_rows.append({
                    "task": lean_task,
                    "input": list(map(int, input_ids)),
                    "target": target,
                    "alternate": alternate,
                    "targetLogit": target_logit,
                    "alternateLogit": alternate_logit,
                    "argmax": argmax,
                    "argmaxLogit": float(logits[argmax].item()),
                })
                rows.append(
                    "  { task := ." + lean_task +
                    f", input := {lean_nat_list(input_ids)}, target := {target}, "
                    f"alternate := {alternate}, "
                    f"targetScoreMicros := {int(round(target_logit * SCALE))}, "
                    f"alternateScoreMicros := {int(round(alternate_logit * SCALE))} }}"
                )

    checkpoint_display = artifact_display_path(checkpoint)
    weights_display = artifact_display_path(weights)
    summary = {}
    for task in ["quoteClose", "bracketType"]:
        task_rows = [r for r in trace_rows if r["task"] == task]
        wrong = [r for r in task_rows if r["argmax"] != r["target"]]
        summary[task] = {
            "rows": len(task_rows),
            "candidate_correct": sum(r["targetLogit"] > r["alternateLogit"] for r in task_rows),
            "full_vocabulary_correct": len(task_rows) - len(wrong),
            "full_vocabulary_winners_when_wrong": {
                str(t): sum(r["argmax"] == t for r in wrong) for t in sorted({r["argmax"] for r in wrong})
            },
            "full_vocabulary_winner_lead_when_wrong": (
                [min(r["argmaxLogit"] - r["targetLogit"] for r in wrong),
                 max(r["argmaxLogit"] - r["targetLogit"] for r in wrong)] if wrong else []
            ),
            "min_margin": min(r["targetLogit"] - r["alternateLogit"] for r in task_rows),
        }
    trace_json.write_text(json.dumps({
        "producer": "tools/generate_eval_certificate.py",
        "checkpoint": checkpoint_display,
        "weights": weights_display,
        "sha256": {
            "checkpoint-final/model.safetensors": sha256_file(checkpoint / "model.safetensors"),
            "smt_weights.json": sha256_file(weights),
        },
        "dtype": "float32",
        "device": "cpu",
        "summary": summary,
        "rows": trace_rows,
    }, indent=1) + "\n")

    body = emit_header(checkpoint_display, weights_display)
    body += "def evalRows : List CandidateEval :=\n[\n"
    body += ",\n".join(rows)
    body += "\n]\n\n"
    body += "def evalCertificate : EvalCertificate where\n"
    body += f"  sourceWeightsSha256 := \"{sha256_file(weights)}\"\n"
    body += f"  checkpointPath := \"{checkpoint_display}\"\n"
    body += f"  scoreScale := {SCALE}\n"
    body += "  rows := evalRows\n\n"
    body += "/-- SHA-256 of the `model.safetensors` that produced the scores. -/\n"
    body += f"def checkpointSha256 : String :=\n  \"{sha256_file(checkpoint / 'model.safetensors')}\"\n"
    body += FOOTER

    output.write_text(body)
    print(f"wrote {output} and {trace_json}")


if __name__ == "__main__":
    main()
