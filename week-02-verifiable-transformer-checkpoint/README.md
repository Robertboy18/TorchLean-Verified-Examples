# Verifiable Transformers in Lean

Neel Somani's
[`verifiable-transformers`](https://github.com/neelsomani/verifiable-transformers) modifies a
small GPT-style model so that its task-local circuits can be encoded for an SMT solver. Sparsemax
replaces softmax, a projection-based normalization replaces LayerNorm, and LeakyReLU replaces
GELU. What you get is a model small enough to exhaust a finite prompt domain, so you can ask a
precise question about two candidate tokens.

I wanted to preserve that claim after its evidence left Python. A JSON checkpoint, a solver trace,
and a circuit summary are easy to separate by accident (the checkpoint changes, a prompt goes
missing, or a margin gets copied from an older run). So here those artifacts become Lean values,
and Lean checks their shapes, hashes, prompt coverage, target tokens, and margins together. It also
replays the exported weights directly with Lean `Float` arithmetic on all 256 prompts.

Neel's project supplies the model family, task, training code, circuit extraction, and SMT
workflow. But the checkpoint itself is ours: we trained it with Neel's training code at commit
[`8e3c54b`](https://github.com/neelsomani/verifiable-transformers/tree/8e3c54bad2494c17c91109fdb58cf9e2f1c98efc)
and his default settings, then ran his circuit extraction and Z3 verification on it. What this
repository adds is the Lean import and replay boundary, plus a checked trace from a separate
native TorchLean run on the same finite task.

## What Lean checks

> The exported small GPT checkpoint, its identity, finite prompt traces, circuit
> summaries, and Lean Float replay all agree on the claimed 256-prompt projected
> decision property.

Lean checks:

- The `smt_weights.json` export summary has the expected architecture,
  operator tags, tensor names/shapes, parameter count, and pinned checkpoint
  fingerprint, and the run's saved `config.json` matches our transcription of Neel's
  default configuration.
- The finite evaluation trace contains exactly the 256 canonical prompts, in the
  expected quote-then-bracket order, and those 256 prompts are precisely the ones
  the task's domain predicates accept.
- Every trace row has the expected projected task, target token, alternate
  token, and `targetScore > alternateScore`.
- Both quote/bracket circuit summaries (generated from the JSON that Neel's
  `extract.py` and `verify.py` wrote for this checkpoint) name the same
  fingerprint and pass the small Lean circuit-summary checkers.
- Lean shape-checks the exported checkpoint payload and then replays it
  directly in `Float`. That replay gives a positive target-vs-alternate
  margin on all 256 prompts, matches the PyTorch scores, and reruns both circuits.
- An archived native TorchLean run records its parameter layout and all 256
  projected decisions. Lean checks that certificate, and the hash of its parameter file,
  separately from the sparsemax checkpoint.

The finite task has two halves:

- `quote_close`: choose token `9` or `10`.
- `bracket_type`: choose token `13` or `14`.

Here's what the replay prints (it's also saved in `artifacts/upstream-small-gpt/verify_upstream_forward.txt`):

```text
Lean Float replay of smt_weights.json ab61ac5dce70292ab9fa4dcd5265d98e04c34c5d89b809384ab297ef9e7abc34
quote rows passing:   128/128
bracket rows passing: 128/128
minimum target-vs-alternate margin: 2.324364
largest gap to the PyTorch trace: 1.015033 micro-logits (tolerance 10.000000)
prompt [1, 2, 5, 9, 5, 5]: logit 9 = 5.025436, logit 10 = 2.293491
BandNorm of a constant 1.000000 vector starts [0.550000, -0.550000, ...], distance to the +-0.550000 fallback pattern 0.000000
BandNorm of a constant 0.300000 vector starts [0.275000, -0.275000, ...], distance to the +-0.550000 fallback pattern 0.275000
controlled forward with all 15 edges matches the plain forward pass: true
quote circuit (3 edges): 128/128 prompts decided as the target, smallest margin 1.835221
  recorded: PyTorch PASSED (0 failures), functional VERIFIED, edge necessity VERIFIED (3/3), robustness VERIFIED
  attn_0 -> mlp_0: removing it flips the decision on [1, 2, 5, 10, 5, 5]
  emb -> attn_0: removing it flips the decision on [1, 2, 5, 9, 5, 5]
  mlp_0 -> logits: removing it flips the decision on [1, 2, 5, 10, 5, 5]
bracket circuit (3 edges): 128/128 prompts decided as the target, smallest margin 3.184683
  recorded: PyTorch PASSED (0 failures), functional VERIFIED, edge necessity VERIFIED (3/3), robustness VERIFIED
  attn_0 -> mlp_0: removing it flips the decision on [1, 3, 5, 12, 5, 5]
  emb -> attn_0: removing it flips the decision on [1, 3, 5, 11, 5, 5]
  mlp_0 -> logits: removing it flips the decision on [1, 3, 5, 12, 5, 5]
PASS: all 256 finite-domain prompts satisfy the projected decision property, and the replay agrees with the trace and the circuit summaries.
```

The replay runs in binary64 `Float`, but PyTorch ran the checkpoint in float32, so the trace
check allows 10 micro-logits for the difference. Both circuits came out as the same three edges,
`emb -> attn_0 -> mlp_0 -> logits`. For each one, Z3 verified functional equivalence and
robustness at epsilon 0.01 on all 128 prompts of its task, and it found all three edges necessary.

## Background: the solver-friendly model

This model family comes from Neel Somani's
[`neelsomani/verifiable-transformers`](https://github.com/neelsomani/verifiable-transformers),
which its GitHub page describes as an "SMT-encodable GPT-2 variant with formal
guarantees". In Neel's public writeup the key move is to replace Transformer components
that are hard to encode for SMT with more solver-friendly ones: softmax attention becomes
sparsemax, LayerNorm becomes a projection-based normalization, and GELU is replaced by
LeakyReLU. And the paper abstract for
["Towards Verifiable Transformers: Solver-Checkable Circuit Explanations"](https://arxiv.org/abs/2605.24033)
states the broader goal, which is to turn task-localized Transformer circuits into bounded,
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

In Lean I mirror these source files from Neel's project, all at commit `8e3c54b` (the last
commit before the July 2026 program-circuit rewrite changed the export format):

- `scripts/small/config.py` for the dimensions and operator choices.
- `scripts/small/train.py` for Signed-L1-BandNorm, sparsemax attention,
  LeakyReLU, and the GPT-2-style wiring.
- `scripts/small/extract_weights.py` for the `smt_weights.json` tensor names,
  shapes, and metadata.
- `scripts/smt/encoders.py` and `scripts/smt/trace.py` for the rational/Z3
  branch encodings.
- `scripts/small/extract.py` and `scripts/small/verify.py` for the
  quote/bracket circuit summaries, which we run on our checkpoint.

Those Python files produce the evidence, and the Lean definitions in this repository are what
give that evidence its checked meaning.

## Scope of the finite claim

Together, the 256 prompts exhaust the symbolic domain chosen for this experiment. Each row compares only the
two tokens relevant to its task, so the theorem doesn't rank the target against every token in the
32-token vocabulary. What happens on the full vocabulary, then? This checkpoint gets all 128 quote
prompts right but only 64 of the 128 bracket prompts, because on the 64 prompts whose opening
bracket is `{`, the double quote (token 10) beats the closing `}` by 0.026. And the check stops at
the exported-artifact boundary, so the result doesn't establish:

- arbitrary transformer training correctness
- PyTorch, Python, Z3, CUDA, or hardware correctness
- full-vocabulary model behavior
- out-of-domain behavior
- architectural or bitwise identity between the archived TorchLean checkpoint and the
  sparsemax checkpoint

## Archived TorchLean comparison

There's also a certificate here from a native TorchLean causal GPT trained on the same finite
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
affine language-model head. Its trace names `train_torchlean_small_gpt` as the program that wrote it,
and that program isn't in this repository (the GPT trainer we do keep is the Week 3 one). So why
keep the certificate? Because Lean still checks its parameter layout, the hash of its parameter
file, and all 256 rows.

As for the sparsemax checkpoint, the generated constants and `Replay/UpstreamFloatReplay.lean`
check it on their own, and no theorem identifies the two architectures or their parameter layouts.

## Repository map

```text
VerifiableTransformers/
  Spec/UpstreamSmallGPT.lean
    Rational contracts for the SMT-facing operators:
    Signed-L1-BandNorm, sparsemax, LeakyReLU, linear maps, and attention.

  Certificate/
    Lean checkers for export metadata, finite evaluation rows, circuit
    summaries, and replayable projected properties; Fingerprint.lean pins
    the checkpoint they all have to name.

  Generated/
    Lean constants generated from real outputs: export metadata,
    checkpoint values, the eval trace, the circuit summaries, and the
    TorchLean CUDA eval trace.

  Replay/UpstreamFloatReplay.lean
    Lean Float replay of the exported checkpoint over all 256 prompts and
    both circuits.

experiments/
  train_upstream_model.py, run_upstream_circuits.py
    Run Neel's training, export, extraction, and verification scripts at the
    pinned commit.

artifacts/upstream-small-gpt/
  The checkpoint, smt_weights.json, metrics, provenance, eval trace, and
  circuits/ with the extraction and Z3 outputs.

```

I kept the export summary, checkpoint payload, score trace, circuit summaries, and TorchLean trace
separate because each supports a different check. That way you can inspect checkpoint identity
without opening the large value payload, or look at the replay values without mixing them up with
the native run.

## Main Lean results

Export metadata and the saved configuration:

```lean
theorem exportSummary_ok :
  checkExportSummary exportSummary = true

theorem trainedConfig_eq_repoDefault :
  trainedConfig = repoDefaultSmallConfig
```

Finite trace from the checkpoint:

```lean
theorem evalTrace_ok :
  checkEvalCertificate evalCertificate = true

theorem checkpointSha256_pinned :
  checkpointSha256 = Fingerprint.modelSafetensorsSha256
```

The prompt enumeration is the whole task:

```lean
theorem expectedInput_injective {i j : Nat} (hi : i < 256) (hj : j < 256)
    (h : expectedInput i = expectedInput j) : i = j

theorem taskDomainOk_expected
    (h : taskDomainOk task input target alternate = true) :
    ∃ i, i < 256 ∧ expectedTask i = task ∧ expectedInput i = input ∧
      expectedTarget i = target ∧ expectedAlternate i = alternate
```

Circuit summaries:

```lean
theorem circuitSummaries_ok :
  checkVerificationSummary quoteVerification = true ∧
  checkVerificationSummary bracketVerification = true

theorem circuitSummaries_outcome :
  quoteVerification.allVerified = true ∧
  bracketVerification.allVerified = true
```

Native TorchLean training trace:

```lean
theorem checkpointSummary_ok :
  checkpointSummaryOk = true

theorem evalTrace_ok :
  checkEvalCertificateWithSha
    Fingerprint.torchLeanParamBitsSha256 evalCertificate = true
```

## Build

This folder is part of the shared Lake project at the repository root, so run
these commands from the root of `TorchLean-Verified-Examples`:

```bash
lake build VerifiableTransformers
lake exe verify_upstream_forward
```

(The root `lakefile.lean` depends on TorchLean once for all the weekly examples.)

## Regenerate the certificates

Everything below runs from the repository root too. On the Python side you need Neel's repository
at the pinned commit and the packages in `week-02-verifiable-transformer-checkpoint/requirements.txt`
(Transformers 4.49.0, which is the version upstream pins):

```bash
git clone https://github.com/neelsomani/verifiable-transformers ../verifiable-transformers
git -C ../verifiable-transformers checkout 8e3c54bad2494c17c91109fdb58cf9e2f1c98efc
pip install -r week-02-verifiable-transformer-checkpoint/requirements.txt
```

Checkpoint. This runs upstream `scripts/small/train.py` unmodified, with its defaults (seed 0,
learning rate 0.003, weight decay 0.01, batch size 64, at most 5000 steps, stopping at the first
evaluation where both tasks have perfect candidate accuracy) and PyTorch's deterministic
algorithms on, and then runs `scripts/small/extract_weights.py`:

```bash
CUDA_VISIBLE_DEVICES=1 python week-02-verifiable-transformer-checkpoint/experiments/train_upstream_model.py \
  --repo ../verifiable-transformers
```

Our run stopped at step 200 on an A100, and two runs gave bit-identical weights.
`artifacts/upstream-small-gpt/provenance.json` records the command, the versions, and the
hashes. A new run changes the hashes, so you'll have to update `Certificate/Fingerprint.lean` by
hand before the certificates check again.

Circuits. This runs upstream `extract.py --threshold_sweep` and `verify.py --sanity_check` with
functional equivalence, edge necessity, and continuous robustness on the CPU (about 15 minutes
per task):

```bash
python week-02-verifiable-transformer-checkpoint/experiments/run_upstream_circuits.py \
  --repo ../verifiable-transformers
```

The candidate-KL numbers in `threshold_sweep.json` change in the last digits between runs,
because upstream sums parent activations in Python set order. But the circuits come out the same.

Lean files, each from the outputs above:

```bash
python week-02-verifiable-transformer-checkpoint/tools/generate_export_certificate.py
python week-02-verifiable-transformer-checkpoint/tools/generate_eval_certificate.py --repo ../verifiable-transformers
python week-02-verifiable-transformer-checkpoint/tools/generate_circuit_summaries.py
python week-02-verifiable-transformer-checkpoint/tools/generate_full_forward_weights.py
```

TorchLean CUDA eval certificate:

```bash
python week-02-verifiable-transformer-checkpoint/tools/generate_torchlean_eval_certificate.py \
  --input week-02-verifiable-transformer-checkpoint/artifacts/torchlean-cuda-small-gpt.eval.json \
  --checkpoint week-02-verifiable-transformer-checkpoint/artifacts/torchlean-cuda-small-gpt.parambits.json \
  --output week-02-verifiable-transformer-checkpoint/VerifiableTransformers/Generated/TorchLeanEvalTrace.lean
```

## Trust boundary

Lean checks the exported metadata, all finite-domain rows, the circuit summaries, the full
checkpoint replay, and the native TorchLean trace certificate. Everything that produces those
artifacts stays outside: Python and PyTorch train the upstream model and extract the circuits, Z3
verifies them, and CUDA runs the native TorchLean experiment. If they emit a different checkpoint,
omit a prompt, change a target, or lose a positive candidate margin, the corresponding Lean check
fails. A different checkpoint, for instance, shows up as a recorded hash that no longer matches
`Certificate/Fingerprint.lean`. (Lean compares the recorded hashes rather than hashing the files itself.)
