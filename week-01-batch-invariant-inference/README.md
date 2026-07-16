# Certifying Batch-Invariant LLM Inference with TorchLean

I built this project to work through a concrete serving question raised by the
Thinking Machines Lab post [*Defeating Nondeterminism in LLM Inference*][tml]:
why can temperature-zero LLM inference still vary when the same request is
served in different batch contexts?

If you want the longer writeup, it lives on my personal website. Here I keep
the practical version: what is in the folder, how to build it, and which claims
Lean actually checks.
## What I Formalized

The informal serving promise is:

```text
same selected request + different surrounding batch
  ==> same tokens shown to the user
```

The Lean development breaks that promise into pieces that can actually be
checked:

- A batched forward pass contract: selected output depends on selected request
  state, not unrelated rows.
- Reductions with explicit schedules, using TorchLean's `SumTree` machinery.
- A concrete `IEEE32Exec` counterexample showing why batch-dependent reduction
  schedules can change Float32 results.
- Batch invariance theorems for explicit reductions, RMSNorm, matmul,
  TorchLean tensor matmul, and abstract attention schedules.
- A margin theorem: if logit drift is bounded and the winner has enough margin,
  greedy decoding returns the same token.
- A decode/verify/rollback serving theorem: committed output is a prefix of a
  canonical reference decoder.
- A tiny CUDA value-reduction certificate: PTX/SASS-derived FMA-chain dataflow
  is checked in Lean and connected back to the same batch-invariance story.

The main proof file is [BatchInvariantInference/Core.lean](BatchInvariantInference/Core.lean).

## Layout

```text
BatchInvariantInference/Core.lean
  Main Lean development: schedules, Float32 counterexample,
  matmul/RMSNorm/attention lemmas, greedy margin theorem, and serving theorem.

BatchInvariantInference/CUDA.lean
  Lean checker for the tiny CUDA certificate. This is where the
  FMA-chain spec and checker soundness live.

BatchInvariantInference/Generated/TinyValueReductionCert.lean
  Generated Lean certificate for the current CUDA/PTX/SASS build.

cuda/
  tiny_attn_one_row.cu
    The inspected CUDA microkernel.
  extract_cert.py
    Builds PTX/CUBIN/SASS and emits the generated Lean certificate.
  cert_tiny_attn.json
    JSON copy of the extracted certificate.
  build/
    Generated PTX/CUBIN/SASS files tied to the certificate hashes.
  README.md
    Local explanation of the CUDA certificate folder.

scripts/
  Optional Python probes. These are observations, not trusted proof code.
```

There is no extra wrapper module here; `BatchInvariantInference/Core.lean` is
the main proof file.

## Build

The repository has one shared Lean 4.32 Lake project at the repo root, so run
these commands from the repository root. `lake-manifest.json` pins the exact
TorchLean revision used by the checked build.

Build everything:

```bash
lake build
```

Build the named proof targets:

```bash
lake build BatchInvariantInference
```

Regenerate the CUDA certificate, then check it:

```bash
python3 week-01-batch-invariant-inference/cuda/extract_cert.py
lake build BatchInvariantInference
```

The generated certificate proof reduces by `rfl`. The important part is not the
generated theorem itself; it is the checker and the soundness lemmas in
`week-01-batch-invariant-inference/BatchInvariantInference/CUDA.lean`.

## Observable Probes

The proof is in Lean. The probes are runtime observations that mirror the
systems symptom and make the issue easier to see before reading the formal
definitions.

List hosted Tinker models:

```bash
export TINKER_API_KEY=...
python3 week-01-batch-invariant-inference/scripts/batch_invariance_demo.py --tinker-list-models
```

Run the same style of temperature zero probe as the Thinking Machines post:
fixed prompt, repeated calls, count unique token sequences.

```bash
python3 week-01-batch-invariant-inference/scripts/batch_invariance_demo.py \
  --tinker-repeat \
  --tinker-base-model meta-llama/Llama-3.2-1B \
  --prompt "Tell me about Richard Feynman" \
  --trials 10 \
  --max-tokens 32
```

The core probe logic is intentionally plain:

```python
params = tinker.SamplingParams(max_tokens=max_tokens, temperature=0, seed=0)
outputs = []
for _ in range(trials):
    resp = sampler.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=params,
    ).result()
    outputs.append(tuple(resp.sequences[0].tokens))
print(f"unique_completions={len(Counter(outputs))}")
```

You can also pass a key file during local development:

```bash
python3 week-01-batch-invariant-inference/scripts/batch_invariance_demo.py \
  --tinker-repeat \
  --tinker-key-file ./tinker.txt
```

Optional local Hugging Face probe:

```bash
python3 week-01-batch-invariant-inference/scripts/batch_invariance_demo.py --local-hf-batch --download
```

The local probe compares one prompt alone against the same prompt in a padded
batch. It may or may not find drift on your machine; either way, it is an
observation, not a theorem.

Local margin diagnostic used by the blog:

```bash
python3 week-01-batch-invariant-inference/scripts/batch_invariance_demo.py \
  --local-hf-margin-plot \
  --model sshleifer/tiny-gpt2 \
  --json-out week-01-batch-invariant-inference/results/local_hf_margin_probe_tiny_gpt2.json \
  --svg-out week-01-batch-invariant-inference/results/local_hf_margin_probe_tiny_gpt2.svg
```

This records the top-two logit margin for each prompt and compares it with
`2 * max_abs_logit_delta` between the prompt alone and the same prompt inside a
padded batch. It is a small local diagnostic for the margin theorem, not a
Tinker logit trace.

## Scope

Checked:

- Lean definitions and theorems for batch invariant inference.
- No `sorry`, `admit`, new `axiom`, or `unsafe` declarations in the Lean
  sources.
- Generated CUDA value reduction certificate accepted by a Lean checker.
- The accepted FMA chain certificate denotes the intended value reduction spec.

Not checked here:

- Tinker, hosted model runtime, vLLM, SGLang, or Triton correctness.
- Full PTX/SASS operational semantics.
- NVIDIA hardware correctness.
- Full production FlashAttention or paged-attention verification.

## AI Usage

Some of the proof development, refactoring, debugging, and prose editing were
assisted by GPT-5.5 Pro.

## References

- [Thinking Machines Lab: *Defeating Nondeterminism in LLM Inference*][tml]
- [TorchLean project overview][torchlean]

[tml]: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
[torchlean]: https://lean-dojo.github.io/TorchLean/
