# Generated Evidence

Everything in this directory is generated from the checkpoint and trace outputs of the Week 2
experiment: large constants, hashes, finite traces, and the small propositions that connect those
constants to the hand-written checkers. Generating them keeps the evidence out of the proof
definitions, and Lean still gets to check the exact exported values.

## Files

| File | Source | Why it exists |
| --- | --- | --- |
| `UpstreamExportSummary.lean` | The `smt_weights.json` export, the run's `config.json` and `provenance.json` | Checks architecture metadata, tensor names/shapes, parameter count, operator tags, the export fingerprint, and the saved configuration. |
| `UpstreamCheckpointPayload.lean` | The `smt_weights.json` export | Embeds the full Float parameter values and the export's fingerprint, checks their exported tensor shapes, and lets Lean replay the trained model forward pass over all 256 prompts. |
| `UpstreamEvalTrace.lean` | The trained checkpoint, scored by Neel's `load_model` on the CPU | Embeds the finite-domain logits trace and checks it against the pinned fingerprints. |
| `UpstreamCircuitSummaries.lean` | `circuit.json` and `verification_results.json` from Neel's `extract.py` and `verify.py` | Copies the quote/bracket circuit summaries and checks their fingerprints, edges, and status counts. |
| `TorchLeanEvalTrace.lean` | Native TorchLean causal-GPT run | Checks the exact 30-tensor parameter layout, the parameter file's pinned hash, and the complete TorchLean-produced finite-domain logits trace. |

Every file records the SHA-256 of whatever it was generated from, and Lean compares that hash with
`Certificate/Fingerprint.lean`, so a file regenerated from a different checkpoint fails until
someone updates the pin. Each object has its own job (the export summary checks metadata and
hashes, the checkpoint payload supports full forward replay, and the evaluation traces check
projected decisions). Keeping them apart means you can inspect one claim without opening every
generated value in the experiment.

## Trust boundary

Python, PyTorch, Z3, and TorchLean CUDA write the JSON, checkpoints, and traces, and these
generators turn those outputs into Lean constants. Whether the constants satisfy the expected
identities, shapes, coverage, and margins is up to the definitions in
`VerifiableTransformers/Certificate`.

## Regeneration

Regenerate these files with the scripts in `tools/`. Please don't edit their constants by hand,
because the checked claim is about the exported values from the measured run (not about some
hand-written miniature).

The generators mirror `scripts/small/extract_weights.py`, `scripts/small/train.py`, and
`scripts/smt/trace.py`, plus the quote/bracket circuit outputs from
`scripts/small/extract.py` and `scripts/small/verify.py`, all at upstream commit `8e3c54b`.
