# Verifiable Transformers in Lean

For Week 2, I wanted to understand the SMT-solver side of neural network
verification more seriously. Not just "can Lean prove a theorem about a toy
network?", but: what does it look like when a transformer circuit claim is first
checked with solver-style finite-domain reasoning, and then brought into
Lean/TorchLean so the exported evidence has to keep matching the claim?

The starting point is Neel Somani's
[`verifiable-transformers`](https://github.com/neelsomani/verifiable-transformers)
work. Credit for the model family, task setup, circuit extraction workflow, and
SMT verification direction belongs to Neel's project. We use that work here as
the thing to check: checkpoint values, finite traces, candidate-token margins,
and circuit summaries leave Python, and Lean checks that they still line up.

The point is not to pretend we proved arbitrary transformer training correct.
We did not. The point is to make a small, real artifact harder to accidentally
misreport. If the checkpoint identity changes, if the prompt coverage is stale,
if a target token is wrong, or if a margin no longer holds, the Lean build should
fail instead of letting the mistake hide in a folder of generated files.

For this finite task, Python, PyTorch, Z3, and TorchLean CUDA produce the
exported files. Lean then checks the model metadata, finite prompt domain,
candidate-token margins, circuit summaries, and a full `Float` replay of Neel's
exported checkpoint values.

## What Is Checked

> The exported small GPT checkpoint, its identity, finite prompt traces, circuit
> summaries, and Lean Float replay all agree on the claimed 256-prompt projected
> decision property.

Lean checks:

- Neel's `smt_weights.json` export summary has the expected architecture,
  operator tags, tensor names/shapes, parameter count, and checkpoint
  fingerprint;
- Neel's finite evaluation trace contains exactly the 256 canonical
  prompts, in the expected quote-then-bracket order;
- every trace row has the expected projected task, target token, alternate
  token, and `targetScore > alternateScore`;
- the saved quote/bracket circuit summaries pass the small Lean circuit-summary
  checkers;
- Neel's exported checkpoint payload is shape-checked in Lean, then replayed
  directly in Lean `Float`, and that replay gives positive target-vs-alternate
  margin on all 256 prompts;
- a separate native TorchLean 4.32 causal GPT is trained on the same finite
  task; Lean checks its exact parameter layout, checkpoint hash, complete
  256-row trace, and the same projected quote/bracket property.

The finite task has two halves:

- `quote_close`: choose token `9` or `10`;
- `bracket_type`: choose token `13` or `14`.

The replay prints:

```text
quote rows passing:   128/128
bracket rows passing: 128/128
minimum target-vs-alternate margin: 0.019695
PASS: all 256 finite-domain prompts satisfy the projected decision property.
```

## What We Took From Neel's Project

The model family comes from Neel Somani's
[`neelsomani/verifiable-transformers`](https://github.com/neelsomani/verifiable-transformers),
described in the project README as an SMT-encodable GPT-2-style Transformer
variant with formal guarantees. Neel's public writeup frames the key move as
replacing hard-to-SMT-encode Transformer components with more solver-friendly
ones: softmax attention becomes sparsemax, LayerNorm becomes a projection-based
normalization, and GELU is replaced by LeakyReLU. The paper abstract for
["Towards Verifiable Transformers: Solver-Checkable Circuit Explanations"](https://arxiv.org/abs/2605.24033)
states the broader goal: turn task-localized Transformer circuits into bounded,
solver-checkable claims.

```text
vocab_size      = 32
max_seq_len     = 6
d_model         = 16
n_layers        = 2
n_heads         = 1
d_mlp           = 64
norm_variant    = signed_l1_band_norm
attn_variant    = sparsemax
activation      = leaky_relu
```

The Lean code mirrors these source files from Neel's project:

- `scripts/small/config.py` for the dimensions and operator choices;
- `scripts/small/train.py` for Signed-L1-BandNorm, sparsemax attention,
  LeakyReLU, and the GPT-2-style wiring;
- `scripts/small/extract_weights.py` for the `smt_weights.json` tensor names,
  shapes, and metadata;
- `scripts/smt/encoders.py` and `scripts/smt/trace.py` for the rational/Z3
  branch encodings;
- `scripts/small/extract.py` and `scripts/small/verify.py` for the saved
  quote/bracket circuit summaries.

Those files create the exported data. They are not the proof. Lean re-checks
the data they wrote.

## Scope And Non-Claims

We use this small transformer run as a compact verification example. It is not
a model release for deployment. The setup is intentionally tiny: a GPT-style
model, a finite symbolic prompt domain, and Lean/TorchLean checks around the
exported evidence.

The decision property is projected to the two relevant candidate tokens for the
task. It is not a full-vocabulary semantic claim.

What we do not claim:

- arbitrary transformer training correctness;
- PyTorch, Python, Z3, CUDA, or hardware correctness;
- full-vocabulary model behavior;
- out-of-domain behavior;
- architectural or bitwise identity between the native TorchLean checkpoint
  and Neel's sparsemax checkpoint.

## Native TorchLean Run

We also train a native TorchLean causal GPT on the same finite task:

```text
vocab = 32
seqLen = 6
dModel = 16
layers = 2
heads = 1
dMlp = 64
```

That path lives in `TorchLean/TrainSmallGPT.lean`. It uses
`nn.models.causalTransformerOneHot`, the public TorchLean 4.32 GPT-2-style
constructor. The model has learned token and positional embeddings, hard-masked
softmax attention, LayerNorm, GELU feed-forward blocks, and an affine language
model head. It can train on CPU or CUDA, save exact parameter bits, and write
the same 256-row finite eval-trace format that the Lean checker understands.

The causal mask is a boolean tensor: allowed entries participate in softmax and
blocked entries contribute exactly zero numerator. On CUDA, the attention node
selects the `native_cuda.flash_attention` capsule, including its fused backward
kernel. Passing `--show-backend` prints that selection and its declared shape,
layout, value, and VJP contracts.

This is not a reimplementation of Neel's sparsemax model. Neel's checkpoint is
checked by the generated Lean constants and `Replay/UpstreamFloatReplay.lean`.
The native run demonstrates the current TorchLean model, trainer, CUDA backend,
and certificate-export path on the same task without identifying the two
models or their parameter layouts.

## How the Pieces Fit

```text
VerifiableTransformers/
  Spec/UpstreamSmallGPT.lean
    Rational contracts for the SMT-facing operators:
    Signed-L1-BandNorm, sparsemax, LeakyReLU, linear maps, and attention.

  Certificate/
    Lean checkers for export metadata, finite evaluation rows, circuit
    summaries, and replayable projected properties.

  Generated/
    Lean constants generated from real outputs: export metadata,
    checkpoint values, Neel eval trace, and TorchLean CUDA eval trace.

  Replay/UpstreamFloatReplay.lean
    Lean Float replay of Neel's exported checkpoint over all 256 prompts.

  TorchLean/
    Formalizations of the upstream custom operators, plus a separate public-API
    TorchLean causal-GPT training and export command.
```

The generated files are separated on purpose. The export summary, checkpoint
values, Neel trace, and TorchLean trace are different pieces of evidence.
Putting them in one huge file would make the repository shorter, but it would
make the audit worse.

## The Main Lean Results

Export metadata:

```lean
theorem exportSummary_ok :
  checkExportSummary exportSummary = true
```

Finite trace from Neel's checkpoint:

```lean
theorem evalTrace_ok :
  checkEvalCertificate evalCertificate = true
```

Circuit summaries:

```lean
theorem neelCircuitSummaries_ok :
  checkVerificationSummary quoteVerification = true ∧
  checkVerificationSummary bracketVerification = true
```

Native TorchLean training trace:

```lean
theorem checkpointSummary_ok :
  checkpointSummaryOk = true

theorem evalTrace_ok :
  checkEvalCertificateWithSha checkpointSha256 evalCertificate = true
```

## Build

This folder is part of the shared Lake project at the repository root. Run
these commands from the root of `TorchLean-Verified-Examples`:

```bash
lake build VerifiableTransformers
lake exe verify_upstream_forward
```

The root `lakefile.lean` depends on TorchLean once for all weekly examples.

Instantiate the native TorchLean model on CPU without running a training step:

```bash
lake exe train_torchlean_small_gpt --device cpu \
  --steps 1 --eval-batches 1 --instantiate-only --inspect-weights
```

For CUDA, configure both the dependency and final executable through Lake:

```bash
lake -R -K cuda=true exe train_torchlean_small_gpt --device cuda \
  --show-backend --steps 1 --eval-batches 1
```

The checked native artifact was produced with 100 optimizer steps over all 16
finite-domain minibatches:

```bash
lake -R -K cuda=true exe train_torchlean_small_gpt --device cuda \
  --steps 100 --eval-batches 16 \
  --save-params week-02-verifiable-transformer-checkpoint/artifacts/torchlean-cuda-small-gpt.parambits.json \
  --save-eval-json week-02-verifiable-transformer-checkpoint/artifacts/torchlean-cuda-small-gpt.eval.json
```

That run completes with loss `0.116249` and projected accuracy `256/256` on the
saved finite trace.

## Regenerate the Certificates

Export summary:

```bash
python week-02-verifiable-transformer-checkpoint/tools/generate_export_certificate.py \
  --input week-02-verifiable-transformer-checkpoint/artifacts/upstream-small-gpt/smt_weights.json \
  --output week-02-verifiable-transformer-checkpoint/VerifiableTransformers/Generated/UpstreamExportSummary.lean
```

Upstream eval certificate:

```bash
python week-02-verifiable-transformer-checkpoint/tools/generate_eval_certificate.py \
  --repo ../verifiable-transformers \
  --checkpoint week-02-verifiable-transformer-checkpoint/artifacts/upstream-small-gpt/checkpoint-final \
  --weights week-02-verifiable-transformer-checkpoint/artifacts/upstream-small-gpt/smt_weights.json \
  --output week-02-verifiable-transformer-checkpoint/VerifiableTransformers/Generated/UpstreamEvalTrace.lean
```

TorchLean CUDA eval certificate:

```bash
python week-02-verifiable-transformer-checkpoint/tools/generate_torchlean_eval_certificate.py \
  --input week-02-verifiable-transformer-checkpoint/artifacts/torchlean-cuda-small-gpt.eval.json \
  --checkpoint week-02-verifiable-transformer-checkpoint/artifacts/torchlean-cuda-small-gpt.parambits.json \
  --output week-02-verifiable-transformer-checkpoint/VerifiableTransformers/Generated/TorchLeanEvalTrace.lean
```

Full forward replay weights:

```bash
python week-02-verifiable-transformer-checkpoint/tools/generate_full_forward_weights.py
```

## What This Does Not Prove

Lean checks the exported metadata, finite-domain certificates, circuit
summaries, Neel checkpoint replay, and the native TorchLean CUDA trace
certificate.
The Python training run, the Z3 search, and the TorchLean CUDA run are still
external runs.

So the claim is not “we verified every line of the training stack.” The actual
claim is narrower:

> we turned the finite claims from a trained transformer run into Lean objects that
> Lean can replay, check, and reject if they stop matching.

That is the step I wanted: not a bigger benchmark, but a cleaner bridge from ML
experiment to machine-checked evidence.
