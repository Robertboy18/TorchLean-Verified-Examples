# TorchLean Verified Examples

This repository collects small, runnable TorchLean examples that pair a blog
post with checked Lean code. Each week adds one focused example folder. The
longer essays live on my personal website; this repo keeps the code,
certificates, scripts, and reproduction commands.

Main TorchLean codebase: https://github.com/lean-dojo/TorchLean
## Examples

| Week | Example | What it checks |
| --- | --- | --- |
| 01 | [Batch-invariant inference](week-01-batch-invariant-inference/) | Schedule-explicit reductions, Float32 schedule sensitivity, RMSNorm/matmul/attention batch-invariance lemmas, margin-stable greedy decoding, decode/verify/rollback serving, and a tiny CUDA value-reduction certificate. |

## How To Use

This is one Lake project shared by all weekly examples. To build the first one:

```bash
lake build BatchInvariantInference
```

The root `lakefile.lean` depends on TorchLean once for the whole repository.
Start with the folder README, then the main Lean file, then any scripts or CUDA
certificates if you want the runtime side of the story.
