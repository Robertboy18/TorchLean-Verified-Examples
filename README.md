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
| 02 | [Verifiable transformers in Lean](week-02-verifiable-transformer-checkpoint/) | Neel Somani's finite sparsemax-transformer run, checked with Lean metadata, circuit summaries, and full Float replay, alongside a separately trained TorchLean 4.32 causal GPT and its checked 256-row trace. |

## How To Use

This is one Lake project shared by all weekly examples. Build one week at a
time:

```bash
lake build BatchInvariantInference
lake build VerifiableTransformers
lake exe verify_upstream_forward
```

The project uses Lean 4.32 and pins the exact TorchLean revision in
`lake-manifest.json`. On a fresh checkout, run `lake update` once before the
build. The root `lakefile.lean` depends on TorchLean once for the whole
repository.

For a real CUDA build, pass the Lake option when building and running:

```bash
lake -R -K cuda=true build
lake -R -K cuda=true exe train_torchlean_small_gpt \
  --device cuda --show-backend --steps 1 --eval-batches 1
```

See the [TorchLean installation guide](https://lean-dojo.github.io/TorchLean/installation/)
for Elan, platform, CUDA, and LibTorch setup.
