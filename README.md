# TorchLean Verified Examples

This repository collects small TorchLean case studies. Each week is one focused
example with the Lean code, generated evidence, scripts, and reproduction
commands in one place. The longer writeups live on my personal website; this
repo is where the checked artifacts live.

Main TorchLean codebase: https://github.com/lean-dojo/TorchLean
## Examples

| Week | Example | What it checks |
| --- | --- | --- |
| 01 | [Batch-invariant inference](week-01-batch-invariant-inference/) | Schedule-explicit reductions, Float32 schedule sensitivity, RMSNorm/matmul/attention batch-invariance lemmas, margin-stable greedy decoding, decode/verify/rollback serving, and a tiny CUDA value-reduction certificate. |
| 02 | [Verifiable transformers in Lean](week-02-verifiable-transformer-checkpoint/) | Neel Somani's finite verifiable-transformer run, checked with Lean metadata checks, 256 prompt traces, circuit summaries, Lean Float replay, and a TorchLean reproduction trace. |

## How To Use

This is one Lake project shared by all weekly examples. Build one week at a
time:

```bash
lake build BatchInvariantInference
lake build VerifiableTransformers
lake exe verify_upstream_forward
```

The root `lakefile.lean` depends on TorchLean once for the whole repository.
Start with the folder README, then the main Lean file. If you want the runtime
side, read the TorchLean training/export command and the certificate
regeneration scripts.
