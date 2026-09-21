# TorchLean Verified Examples

I use this repository for experiments that begin with an ordinary machine-learning question and
end with a precise Lean statement. Can changing the surrounding batch change a greedy answer? Can
a checkpoint exported by Python still be identified and replayed in Lean? Can a
124-million-parameter GPT train from Lean without losing the causal and cache properties we want to
state about it? Each week takes one such question far enough to run the program, inspect the
evidence, and say exactly what has and has not been proved.

The examples use [TorchLean](https://github.com/lean-dojo/TorchLean) for typed tensors, neural-network
models, training, numerical specifications, and verification. Large computations still run through
ordinary CPU or GPU code. Lean checks the mathematical results and certificates described in each
folder; the READMEs name the remaining runtime and hardware assumptions beside those results.

## The examples

| Week | Experiment | Result |
| --- | --- | --- |
| 01 | [Batch-invariant inference](week-01-batch-invariant-inference/) | Makes reduction schedules explicit, exhibits a binary32 counterexample, proves batch-invariance and margin-stability results, and checks a small CUDA reduction certificate. |
| 02 | [Verifiable transformers](week-02-verifiable-transformer-checkpoint/) | Rechecks a finite sparsemax-transformer claim from exported evidence, replays the checkpoint in Lean `Float`, and checks a separate TorchLean causal-GPT run on all 256 prompts. |
| 03 | [GPT-2 Small in Lean](week-03-gpt-training/) | Trains a 124.4M-parameter GPT for 2.319B scheduled tokens on one A100, reruns instruction tuning with dialogue-bounded sampling, accelerates generation with a checked cache model, and proves causal, dialogue-window, numerical, and resume properties. The SFT objective improves, but the resulting checkpoint is not a reliable assistant. |
| 04 | [Kimi K3 specification](week-04-kimi-k3-specification/) | Writes K3's architecture as shape-indexed tensor functions and typed TorchLean graphs, then proves the packed one-token language graph and the vision components have the stated semantics. The public-state bridge, full multimodal graph, released weights, kernels, training runs, and empirical claims remain outside those proofs. |

The longer essays for [Week 1](https://www.robertj1.com/ai4science/batch-invariant-inference/),
[Week 2](https://www.robertj1.com/ai4science/verifiable-transformer-checkpoint/), and
[Week 3](https://www.robertj1.com/ai4science/training-gpt2-in-lean/) give the experiments more room;
Week 4 includes an [annotated report and formalization](week-04-kimi-k3-specification/site/) in the
repository.
The weekly folders remain the source for exact theorem statements, generated evidence, measured
artifacts, and reproduction commands.

## Where Velvet fits

[Velvet](https://github.com/verse-lab/velvet) is used in Week 3 to verify mutable loops in the
actual data path: packing aligned training rows and finding the first invalid token or mask byte.
The [Week 3 explanation](week-03-gpt-training/#proving-the-data-loops-with-velvet) links the
contracts, proofs, and executable checks.

The other developments do not need an imperative rewrite just to use the same tool. Week 1's
reduction and certificate arguments, Week 2's finite checkpoint checks, and Week 4's tensor and
graph equalities already have direct proofs. The cache and resume theorems likewise remain small
recursive arguments. Velvet would become useful there when verifying an actual mutable
implementation against those specifications; wrapping native calls alone would not prove them
correct. The dependency stays in this repository, outside the main TorchLean library.

## Build the Lean developments

All four weeks share one Lake project and one pinned TorchLean dependency. On a fresh checkout:

```bash
git clone https://github.com/Robertboy18/TorchLean-Verified-Examples.git
cd TorchLean-Verified-Examples

lake build BatchInvariantInference
lake build VerifiableTransformers
lake build TorchLeanGPT
lake build KimiK3
```

The Week 2 executable replay is a separate command:

```bash
lake exe verify_upstream_forward
```

CPU builds need no CUDA installation. For examples that run real NVIDIA kernels, pass the CUDA
option through Lake when building and running:

```bash
lake -R -Kcuda=true \
  -KverifiedExamplesBuildDir=.lake/build-cuda \
  -KtorchleanBuildDir=.lake/build-cuda build \
  train_torchlean_gpt \
  generate_torchlean_gpt_cached \
  check_torchlean_gpt_cache \
  benchmark_torchlean_gpt_cache
```

The project targets Lean 4.34. `lake-manifest.json` pins the tested dependencies; a fresh checkout
builds those revisions without running `lake update`. Use the same CUDA build-directory options
when running an executable through `lake exe`. The
[TorchLean installation guide](https://lean-dojo.github.io/TorchLean/installation/) covers Elan,
CPU-only builds, CUDA discovery, and supported platforms.
