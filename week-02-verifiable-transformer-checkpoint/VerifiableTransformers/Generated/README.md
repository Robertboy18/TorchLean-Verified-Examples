# Generated Evidence

The files in this directory are generated from the checkpoint and trace outputs used by the Week 2
experiment. They contain large constants, hashes, finite traces, and the small propositions that
connect those constants to the hand-written checkers. The generators keep evidence out of the
proof definitions while still letting Lean check the exact exported values.

## Files

| File | Source | Why it exists |
| --- | --- | --- |
| `UpstreamExportSummary.lean` | Neel's `smt_weights.json` export | Checks architecture metadata, tensor names/shapes, parameter count, operator tags, and the export fingerprint. |
| `UpstreamCheckpointPayload.lean` | Neel's `smt_weights.json` export | Embeds the full Float parameter values, checks their exported tensor shapes, and lets Lean replay the trained model forward pass over all 256 prompts. |
| `UpstreamEvalTrace.lean` | Neel's trained checkpoint and Python/Z3 circuit verifier | Embeds the finite-domain logits trace and checks the saved quote/bracket circuit summaries. |
| `TorchLeanEvalTrace.lean` | Native TorchLean causal-GPT run | Checks the exact 30-tensor parameter layout, checkpoint identity, and complete TorchLean-produced finite-domain logits trace. |

The export summary checks metadata and hashes, the checkpoint payload supports full forward replay,
and the evaluation traces check projected decisions. Keeping those objects separate lets a reviewer
inspect one claim without opening every generated value in the experiment.

## Trust boundary

Python, PyTorch, Z3, and TorchLean CUDA write JSON, checkpoints, and traces. These generators turn
the outputs into Lean constants. The definitions in `VerifiableTransformers/Certificate` decide
whether those constants satisfy the expected identities, shapes, coverage, and margins.

## Regeneration

Regenerate these files with the scripts in `tools/`; do not edit their constants by hand. The
checked claim concerns the exported values from the measured run, not a hand-written miniature.

The source paths mirrored by the generators are
`scripts/small/extract_weights.py`, `scripts/small/train.py`,
`scripts/smt/trace.py`, and the quote/bracket circuit outputs from
`scripts/small/extract.py` and `scripts/small/verify.py`.
