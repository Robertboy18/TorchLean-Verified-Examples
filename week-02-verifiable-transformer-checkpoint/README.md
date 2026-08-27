# Verifiable Transformers in Lean

Neel Somani's
[`verifiable-transformers`](https://github.com/neelsomani/verifiable-transformers) modifies a
small GPT-style model so that task-local circuits can be encoded for an SMT solver. Sparsemax
replaces softmax, a projection-based normalization replaces LayerNorm, and LeakyReLU replaces
GELU. The resulting model is small enough to exhaust a finite prompt domain and ask a precise
question about two candidate tokens.

I wanted to preserve that claim after its evidence left Python. A JSON checkpoint, a solver trace,
and a circuit summary are easy to separate accidentally: the checkpoint changes, a prompt is
omitted, or a margin is copied from an older run. Here those artifacts become Lean values. Their
shapes, hashes, prompt coverage, target tokens, and margins are checked together, and the exported
weights are replayed directly with Lean `Float` arithmetic on all 256 prompts.

The model family, task, circuit extraction, and SMT workflow come from Neel's project. This
repository adds the Lean import and replay boundary. It also keeps a checked trace from a separate
native TorchLean run on the same finite task.

## What Lean checks

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
- an archived native TorchLean run records its parameter layout, checkpoint hash, and all 256
  projected decisions; Lean checks that certificate independently from Neel's checkpoint.

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

## Background: the solver-friendly model

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

Those Python files produce the evidence. The Lean definitions in this repository give that
evidence its checked meaning.

## Scope of the finite claim

The 256 prompts exhaust the symbolic domain chosen for this experiment. Each row compares the two
tokens relevant to its task; the theorem does not rank the target against every token in the
32-token vocabulary. The checked result also stops at the exported-artifact boundary. It does not
establish:

- arbitrary transformer training correctness;
- PyTorch, Python, Z3, CUDA, or hardware correctness;
- full-vocabulary model behavior;
- out-of-domain behavior;
- architectural or bitwise identity between the archived TorchLean checkpoint and Neel's
  sparsemax checkpoint.

## Archived TorchLean comparison

The repository keeps a certificate from a native TorchLean causal GPT trained on the same finite
task:

```text
vocab = 32
seqLen = 6
dModel = 16
layers = 2
heads = 1
dMlp = 64
```

That model used ordinary causal softmax attention, LayerNorm, GELU feed-forward blocks, and an
affine language-model head. Its saved trace predates the generalized Week 3 trainer, so the old
one-off training command has been removed instead of carrying a second GPT runner. The certificate
stays because Lean still checks its checkpoint identity, parameter layout, and all 256 rows.

Neel's sparsemax checkpoint is checked separately by the generated constants and
`Replay/UpstreamFloatReplay.lean`. No theorem identifies the two architectures or their parameter
layouts.

## Repository map

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

```

The export summary, checkpoint payload, Neel trace, and TorchLean trace remain separate because
they support different checks. A reviewer can inspect checkpoint identity without opening the
large value payload, or inspect the replay values without conflating them with the native run.

## Main Lean results

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

## Regenerate the certificates

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

## Trust boundary

Lean checks the exported metadata, all finite-domain rows, circuit summaries, the full checkpoint
replay, and the native TorchLean trace certificate. Python and PyTorch train the upstream model,
Z3 performs the circuit search, and CUDA runs the native TorchLean experiment. Those producers
remain external. If they emit a different checkpoint, omit a prompt, change a target, or lose a
positive candidate margin, the corresponding Lean check fails.
