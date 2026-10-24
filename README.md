# TorchLean Verified Examples

Here's [my vision for verification in AI](https://www.robertj1.com/ai4science/vision/).
Some examples I've been working on :)

**[Read the essays →](https://robertboy18.github.io/TorchLean-Verified-Examples/)**

## The examples

- [Thinking Machines Lab’s batch-invariance problem, formalized](week-01-batch-invariant-inference/).
  Start with three binary32 numbers, then prove when a reduction schedule preserves the
  scores and when a numerical error bound preserves the winning token.
- [Checking a small transformer in Lean](week-02-verifiable-transformer-checkpoint/).
  Train Neel Somani's small transformer with his own code, check the exported decisions for
  all 256 task inputs, and separately recompute them from the checkpoint.
- [Training a GPT-2-style model in Lean](week-03-gpt-training/).
  Train a 124.4M-parameter model for 2.319B scheduled tokens, compare its updates with PyTorch,
  try instruction tuning, and prove results about targets, resuming, and cached generation.
- [Writing tensor programs in Lean](torchlean-tensor-programming/).
  Write custom CPU/GPU functions, attention contractions with `einsum`, and composed models.
  Then there are performance measurements, higher derivatives, and configured-precision training.
- [Kimi K3: the paper and proofs side by side](week-04-kimi-k3-specification/site/).
  Work through the architecture, prove sequence/streaming and packed-graph equalities,
  and pin down a slip at one endpoint of the expert-balancing argument.
- [Nemotron 3 Ultra: replaying a rounded state](week-05-nemotron-specification/site/).
  Prove replay and rollback results, then follow checkpoint-rounding errors. Fewer
  rounded writes can leave a larger actual error, even with a smaller worst-case budget!
- [Estimating model behavior from its structure](week-06-heuristic-estimators/).
  Study shared inputs and attention, then use the existing backward pass to bound the
  expected error from averaging away input randomness. Check finite examples and numerical
  error bounds, motivated by MAISI and ARC's research.

For K3 and Nemotron, the proofs cover the equations and state transformations described in the
essays (not complete released checkpoints). Each development links its theorem statements,
assumptions, source, and reproduction commands.

## Python examples

`requirements.txt` pins PyTorch 2.13.0 and Transformers 5.10.4, and you'll want them in a virtual
environment so they don't replace an existing TorchLean SDK installation.

Before installing the remaining requirements, pick a PyTorch wheel that matches your platform and
NVIDIA driver. For a CUDA 12.6 installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

For CPU-only Python use, select `https://download.pytorch.org/whl/cpu` instead. The default
PyPI wheel for this PyTorch release uses CUDA 13 but doesn't upgrade your NVIDIA driver when you
install it. (Lean's CPU build doesn't need Python or PyTorch at all.)

If this Python installation also supplies LibTorch, point `TORCHLEAN_LIBTORCH_HOME` at its
`torch` package directory and rebuild the native adapters. Don't reuse adapters linked against
the previous SDK. You'll find the native build in the
[Week 3 instructions](week-03-gpt-training/#build).

## Build the Lean developments

All six weeks share one Lake project and follow TorchLean's `main` branch.

On a fresh checkout:

```bash
git clone https://github.com/Robertboy18/TorchLean-Verified-Examples.git
cd TorchLean-Verified-Examples

lake build BatchInvariantInference
lake build VerifiableTransformers
lake build TorchLeanGPT
lake build KimiK3
lake build Nemotron
lake build HeuristicEstimation
```

Week 2's executable replay is a separate command:

```bash
lake exe verify_upstream_forward
```

CPU builds need no CUDA installation. For the examples that run real NVIDIA kernels, pass the
CUDA option through Lake when building and running:

```bash
lake -R -Kcuda=true \
  -KverifiedExamplesBuildDir=.lake/build-cuda \
  -KtorchleanBuildDir=.lake/build-cuda build \
  train_torchlean_gpt \
  generate_torchlean_gpt_cached \
  check_torchlean_gpt_cache \
  benchmark_torchlean_gpt_cache
```

Use the same CUDA build-directory options when you run an executable through `lake exe`. The
project targets Lean 4.35, and since `lake-manifest.json` pins the dependencies, a fresh checkout
builds those revisions without running `lake update`. For Elan, CPU-only builds, CUDA discovery,
and supported platforms, see the
[TorchLean installation guide](https://lean-dojo.github.io/TorchLean/installation/).

## Build the example site

The essays and reports are static pages. Build them with:

```bash
python -m pip install -r site/requirements.txt
python3 site/build.py
```

The pages go to `build/site`. Building needs no Lean compiler and starts no server.

Types, proof states, and command output on the pages were recorded from Lean in
`site/assets/lean.json`, and `site/build.py` stops if the Lean sources changed after that
recording. So after you change Lean code, build the developments above and TorchLean's `NN`
library, then record again:

```bash
lake build NN
python3 site/lean.py
python3 site/build.py
```
