# Generated Evidence

This directory contains Lean files generated from real checkpoint and trace
outputs. They are plain on purpose: big constants, hashes, traces, and small
theorems saying the imported data passed the checks.

## Files

| File | Source | Why it exists |
| --- | --- | --- |
| `UpstreamExportSummary.lean` | Neel's `smt_weights.json` export | Checks architecture metadata, tensor names/shapes, parameter count, operator tags, and the export fingerprint. |
| `UpstreamCheckpointPayload.lean` | Neel's `smt_weights.json` export | Embeds the full Float parameter values, checks their exported tensor shapes, and lets Lean replay the trained model forward pass over all 256 prompts. |
| `UpstreamEvalTrace.lean` | Neel's trained checkpoint and Python/Z3 circuit verifier | Embeds the finite-domain logits trace and checks the saved quote/bracket circuit summaries. |
| `TorchLeanEvalTrace.lean` | TorchLean CUDA training run | Checks the TorchLean checkpoint summary and embeds the TorchLean-produced finite-domain logits trace. |

These files are separated by evidence type, not convenience.  The export summary
checks metadata and hashes, the checkpoint values support Lean forward replay,
and the eval traces check projected decisions.  Merging them would make the file
count smaller, but it would make the audit worse.

## Trust Split

Python, PyTorch, Z3, and TorchLean CUDA write JSON, checkpoints, and traces.

Lean checks them.  These files turn those outputs into Lean constants, then prove
that the imported data satisfies the small checkers in
`VerifiableTransformers/Certificate`.

## Regeneration

Use the scripts in `tools/` rather than editing these files by hand. The useful
checked claim is about the exact exported values, so a hand-written miniature would be
the wrong object to trust.

The source paths mirrored by the generators are
`scripts/small/extract_weights.py`, `scripts/small/train.py`,
`scripts/smt/trace.py`, and the quote/bracket circuit outputs from
`scripts/small/extract.py` and `scripts/small/verify.py`.
