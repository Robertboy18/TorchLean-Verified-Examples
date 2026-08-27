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
| 04 | [Kimi K3 specification](week-04-kimi-k3-specification/) | Gives parameterized Lean definitions for the language, vision, training, and speculative-decoding algorithms in the Kimi K3 report, with proofs about chunking, cache compression, routing, and draft acceptance. |

The longer essays for [Week 1](https://www.robertj1.com/ai4science/batch-invariant-inference/),
[Week 2](https://www.robertj1.com/ai4science/verifiable-transformer-checkpoint/), and
[Week 3](https://www.robertj1.com/ai4science/training-gpt2-in-lean/) give the experiments more room.
The weekly folders remain the source for exact theorem statements, generated evidence, measured
artifacts, and reproduction commands.

## Build the Lean developments

All four weeks share one Lake project and one pinned TorchLean dependency. On a fresh checkout:

```bash
git clone https://github.com/Robertboy18/TorchLean-Verified-Examples.git
cd TorchLean-Verified-Examples
lake update

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
lake -R -K cuda=true build \
  train_torchlean_gpt \
  generate_torchlean_gpt_cached \
  check_torchlean_gpt_cache \
  benchmark_torchlean_gpt_cache
```

The project currently uses Lean 4.33. `lake-manifest.json` pins the TorchLean revision used by the
checked build, so later upstream changes cannot silently alter an example. The
[TorchLean installation guide](https://lean-dojo.github.io/TorchLean/installation/) covers Elan,
CPU-only builds, CUDA discovery, and supported platforms.
