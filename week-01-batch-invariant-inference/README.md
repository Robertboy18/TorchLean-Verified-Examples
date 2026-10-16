# Certifying Batch-Invariant LLM Inference with TorchLean

Temperature zero makes a decoder deterministic, but only once its logits exist. Computing those
logits can still depend on how the server batched your request (a different kernel or reduction
tree rounds binary32 differently), and when two tokens are close, a small logit change is enough
to change the winner. Thinking Machines Lab described this systems problem in
[*Defeating Nondeterminism in LLM Inference*][tml].

This example follows that numerical change all the way to the serving contract. I make each
reduction schedule an explicit input, prove when a selected row doesn't depend on the
unrelated rows around it, and then use a winner-margin theorem to turn a logit bound into equal
greedy tokens. One last theorem covers serving itself (decode, verification, and rollback). There's
also a small CUDA example, which checks one value-reduction certificate in Lean instead of treating
a successful kernel launch as a proof.

If you want the motivation and worked examples, the longer [Week 1 essay](https://robertboy18.github.io/TorchLean-Verified-Examples/week-01-batch-invariant-inference/)
goes through them in more detail.

## The checked argument

Informally, the serving promise is:

```text
same selected request + different surrounding batch
  ==> same tokens shown to the user
```

I prove it in Lean in stages:

- A batched forward pass contract: the selected output depends on the selected
  request's state, not on unrelated rows.
- Reductions whose schedules are spelled out, using FloatLib's `ReductionTree` directly.
- A concrete binary32 counterexample, which shows why a reduction schedule that
  depends on the batch can change Float32 results.
- Batch invariance theorems for explicit reductions, RMSNorm, matmul,
  TorchLean tensor matmul, and abstract attention schedules.
- A margin theorem: if logit drift is bounded and the winner has enough margin,
  greedy decoding returns the same token.
- A decode/verify/rollback serving theorem: whatever the server commits is a
  prefix of what a canonical reference decoder produces.
- A tiny CUDA value-reduction certificate: Lean checks the PTX program for the
  FMA chain, load and store addresses included, and ties it back to the same
  batch-invariance story.

If you want to read the Lean, start with [BatchInvariantInference/Core.lean](BatchInvariantInference/Core.lean).

## Layout

```text
BatchInvariantInference/Core.lean
  Main Lean development: schedules, Float32 counterexample,
  matmul/RMSNorm/attention lemmas, greedy margin theorem, and serving theorem.

BatchInvariantInference/CUDA.lean
  Lean checker for the tiny CUDA certificate. This is where the
  FMA-chain spec and checker soundness live.

BatchInvariantInference/Generated.lean
  Generated Lean certificate for the current CUDA/PTX/SASS build.

cuda/
  tiny_attn_one_row.cu
    The inspected CUDA microkernel.
  extract_cert.py
    Builds PTX/CUBIN/SASS and emits the generated Lean certificate.
  cert_tiny_attn.json
    JSON copy of the extracted certificate.
  run_cubin.cpp, runtime_check.json
    Runs the CUBIN on a GPU against a binary32 FMA reference, and its results.
  build/
    Generated PTX/CUBIN/SASS files tied to the certificate hashes.
  README.md
    Local explanation of the CUDA certificate folder.

scripts/
  Optional Python probes. These are observations, not trusted proof code.
```

There's no wrapper module on top of these files, so `BatchInvariantInference/Core.lean`
is the main proof file.

## Build

All the weeks share one Lean 4.35 Lake project at the repo root, so run these
commands from there. (`lake-manifest.json` pins the exact TorchLean revision
that the checked build used.)

Build everything:

```bash
lake build
```

Build the named proof targets:

```bash
lake build BatchInvariantInference
```

To regenerate the CUDA certificate and then check it, run the two commands below. The
committed PTX, CUBIN, and SASS came from CUDA 13.2 (V13.2.78). `--cuda-bin` names a toolkit
`bin` directory that has `nvcc`, `ptxas`, and `nvdisasm`, and if you'd rather not compile
anything, `--from-build` re-extracts from the committed files:

```bash
python3 week-01-batch-invariant-inference/cuda/extract_cert.py --arch sm_80 --cuda-bin /usr/local/cuda/bin
lake build BatchInvariantInference
```

Lean accepts the generated certificate by `decide`, but the generated theorem isn't the
part worth reading. What matters is the checker and its soundness lemmas in
`week-01-batch-invariant-inference/BatchInvariantInference/CUDA.lean`.

## Runtime probes

All of the proof is in Lean. These probes are runtime observations that mirror the
systems symptom, which makes the issue easier to see before you read any formal
definitions.

List hosted Tinker models:

```bash
export TINKER_API_KEY=...
python3 week-01-batch-invariant-inference/scripts/batch_invariance_demo.py --tinker-list-models
```

This runs the same kind of temperature-zero probe as the Thinking Machines post (fix a
prompt, call it repeatedly, and count the unique token sequences).

```bash
python3 week-01-batch-invariant-inference/scripts/batch_invariance_demo.py \
  --tinker-repeat \
  --tinker-base-model meta-llama/Llama-3.2-1B \
  --prompt "Tell me about Richard Feynman" \
  --trials 10 \
  --max-tokens 32
```

At its core the probe is this loop:

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

The local probe runs one prompt alone and the same prompt in a padded batch, and compares
the two. It may or may not find drift on your machine (either way, it's an
observation, not a theorem).

Local margin diagnostic used by the blog:

```bash
python3 week-01-batch-invariant-inference/scripts/batch_invariance_demo.py \
  --local-hf-margin-plot --download \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --json-out week-01-batch-invariant-inference/results/local_hf_margin_probe_tinyllama.json \
  --svg-out week-01-batch-invariant-inference/results/local_hf_margin_probe_tinyllama.svg
```

For each prompt this records the margin between the top two logits and compares it with
`2 * max_abs_logit_delta`, where the delta is taken between the prompt alone and the same
prompt inside a padded batch. Think of it as a small local diagnostic for the margin theorem.
It isn't a Tinker logit trace.

## Proof boundary

Checked:

- Lean definitions and theorems for batch invariant inference.
- No `sorry`, `admit`, new `axiom`, or `unsafe` declarations in the Lean
  sources.
- The generated CUDA value reduction certificate, accepted by a Lean checker.
- The accepted FMA chain certificate denotes the intended value reduction spec:
  each active thread stores the chain over its own row, its loads and store stay
  in bounds, and distinct threads write distinct outputs.

Not checked here:

- Tinker, hosted model runtime, vLLM, SGLang, or Triton correctness.
- Full PTX/SASS operational semantics. The extractor reads the PTX and traces
  its guards in Python, and the SASS only gets substring checks.
- The launch conditions in `cuda/README.md`: buffer sizes and allocations, grid
  shape, and the batch bound that keeps 32-bit indices from wrapping.
- NVIDIA hardware correctness.
- Full production FlashAttention or paged-attention verification.

## AI assistance

GPT-5.5 Pro assisted with some of the proof development, refactoring, debugging, and
prose editing.

## References

- [Thinking Machines Lab: *Defeating Nondeterminism in LLM Inference*][tml]
- [TorchLean project overview][torchlean]

[tml]: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
[torchlean]: https://lean-dojo.github.io/TorchLean/
