# Experiments with smaller model computations

We ran TorchLean on attention inputs from our trained GPT checkpoint and searched for
coordinates we could discard. Choosing coordinates by their effect on the output worked
better than choosing them by magnitude. The approximation still introduced substantial
error, and packing the smaller tensors cost more than it saved in this run.

We then extended the experiment to deeper layers and longer histories, a small sparsemax
transformer, and a digit classifier. The follow-up results are below. The first run and its
original source hashes remain in `results.json`; `results-models.json` records the expanded run.

## What we ran

The checkpoint has 124,412,160 parameters, a width of 768, and 12 attention heads with
64 coordinates each. We extracted the first block's queries, keys, and values for 320
disjoint 64-token windows from its existing validation shard. The first 64 windows were
used to fit summaries; the other 256 were reserved for evaluation. They were held out from
this search, not necessarily from the original training experiment's model selection.

Each calculation uses the last query and all 64 keys in its causal prefix. TorchLean
replayed all 12 heads, giving 3,072 head queries and 196,608 output coordinates for each
of the full and reduced calculations. This is the first block's attention before its
output projection, not an end-to-end run of the 124-million-parameter model.

Python/PyTorch prepared the embeddings, normalization, and FP32 Q/K/V projections.
TorchLean then computed scores, stable softmax, and value aggregation on CPU with
binary64 `Float` tensors. The reference used the same stored activations in binary64.
The original attention scale, `1 / sqrt(64)`, stayed fixed when coordinates were removed.

## What the search found

For each head, the greedy method repeatedly discarded whichever coordinate caused the
smallest calibration-output error. We compared it with magnitude ranking, a ranking
that removes common key shifts, and a seeded random ordering. All choices used calibration
windows only. Relative L2 error below is the norm of the error across evaluation windows,
heads, and output coordinates, divided by the corresponding full-output norm.

| Retained Q/K coordinates | Random | Magnitude | Centered | Output-aware greedy |
| --- | ---: | ---: | ---: | ---: |
| 48 of 64 | 19.28% | 13.80% | 13.30% | 10.54% |
| 32 of 64 | 38.88% | 29.87% | 28.23% | 24.10% |
| 16 of 64 | 55.60% | 45.99% | 44.97% | 42.09% |
| 8 of 64 | 62.85% | 56.11% | 55.76% | 53.16% |

The greedy method was selected at the 32-coordinate budget using calibration error.
Its evaluation error varied considerably between heads: from 9.05% to 40.17%. The worst
evaluation window had 73.71% relative error. A uniform half-width budget does not preserve
these attention outputs closely. Only the query–key contraction becomes smaller; value
aggregation, normalization, and the original projections are unchanged.

## Did TorchLean handle it?

Yes. All 3,072 cases completed without nonfinite results. Comparing the serialized binary64
words avoids the rounding in Lean's ordinary JSON float display. Maximum disagreements
with the reference were:

- Full output: `2.22e-16`; reduced output: `1.11e-16`.
- Full scores: `3.55e-15`; reduced scores: `8.88e-16`.
- Full probabilities: `3.33e-16`; reduced probabilities: `5.55e-17`.

The native CPU runner took 7.62 seconds wall time, including input decoding and JSON output,
with a peak RSS of 77,800 KiB (about 76 MiB). Summed internal timings were 1.97 seconds for
full attention, 1.55 seconds for reduced attention, and 0.99 seconds for packing retained
coordinates. Including packing, the reduced route took 2.54 seconds. These are single-pass
diagnostics on an AMD EPYC 7742, not a repeated, interleaved performance benchmark.

The GPU calibration search took 0.37 seconds and extraction took 0.49 seconds in this run;
those timings exclude Python startup, file hashing, and Lean replay. There is no demonstrated
end-to-end speedup here. Reusing already-packed keys might change the cost, but we have not
measured that workload.

## What we checked mathematically

The new `Softmax.output_sub_le_centered` theorem carries score error through TorchLean's
real-valued stable softmax. If a common score shift is removed and every remaining score
error is at most epsilon, each value-weighted output coordinate changes by at most
`2 * V * epsilon`, where `V` bounds the value magnitudes. The bound has no sequence-length
factor. This is a formalization of a standard derivative bound, not a new inequality.

For all 12 heads of the first evaluation window, FloatLib decoded the stored binary64
inputs exactly into rational tensors. We constructed the restricted Gram matrix and
checked its absolute-row-sum certificate. All 12 passed, and all 12 deliberately incorrect
bounds were rejected. The general soundness theorem gives a quadratic score-error bound
for every real query against each fixed key bank. The observed query's exact score MSE
also stayed below its certified budget in every case.

We separately computed exact centered score-error intervals for those 12 queries. The
softmax theorem gives per-coordinate output bounds between 0.01305 and 0.31816. For the
first head, the bound was 0.01867 and the measured maximum output error was about 0.00115.
The bounds are conservative. Gram construction took 16.78 seconds; checking the proposed
row bounds took 109 ms total. Computing useful certificate premises is not free.

The general theorems were checked by Lean's kernel. The data-specific rational checker
ran as a program; it did not emit a kernel proof term for the input file. The real-arithmetic
theorem also does not prove the rounding behavior of the concrete floating-point replay.

## Different layers and longer histories

The follow-up used the same trained GPT checkpoint, with 64 calibration windows and 128
evaluation windows in each setting. We tested layers 1, 6, and 12 at 64, 128, and 512 tokens.
All settings share the same final source tokens and final position embedding, 511. Shorter
windows truncate the history while retaining their absolute positions. They are not ordinary
short prompts whose positions begin at zero. At deeper layers, removing history also changes
the upstream hidden states.

The table shows relative attention-head output error when retaining 32 of 64 Q/K coordinates.
Each layer/context pair fits its own mask using calibration outputs only. The full grid,
including all four coordinate budgets and all ranking baselines, is in `results-models.json`.

| History length | Layer 1 | Layer 6 | Layer 12 |
| --- | ---: | ---: | ---: |
| 64 tokens | 30.33% | 1.12% | 1.25% |
| 128 tokens | 30.16% | 1.26% | 1.54% |
| 512 tokens | 34.55% | 20.82% | 41.12% |

The short-history middle and late layers looked much easier to summarize. Part of that
advantage came from large, nearly constant output components. Relative to variation across
evaluation windows, their errors were 6.87–10.23%, rather than 1–1.5%. This diagnostic uses
the evaluation mean only to assess error; it does not change the fitted masks. At 512 tokens,
both measures show substantial error. A summary that works at one history length need not
work at another.

TorchLean replayed 13,824 head queries across the nine settings. Maximum output disagreement
with the supplied-activation reference was below `3e-15`. Native runs streamed one head case
at a time and peaked below 83 MiB RSS. This still checks attention on inputs prepared by the
matching PyTorch reconstruction, not every preceding block in TorchLean. The reduced route
including packing was slower than the full calculation in every setting measured.

For the first head of the first evaluation window in each setting, we also ran the exact
rational Gram checker. All nine certificates passed and rejected deliberately incorrect
bounds. These are nine fixed key banks, not certificates for every evaluation window.
The layer-12, 512-token example had centered score error about 2. Its `2 * V * epsilon`
softmax bound is therefore weaker than the trivial `2 * V` bound on two weighted outputs.
A correct certificate is not necessarily a useful one.

## Removing score coordinates from the small transformer

For a different trained architecture, we reused the small sparsemax transformer from Week 2.
Its two tasks compare two candidate tokens for closing quotes and matching brackets. We
changed only the first attention block's Q/K coordinates, retaining its original sparsemax,
normalization, second block, and output head.

We fitted a greedy score-error ordering on 128 prompts, balanced by task and candidate target,
and evaluated on the other 128. Unlike the GPT search, this ordering uses a quadratic
score-error objective rather than the final attention-output error. These are model-specific
case studies, not a matched ranking of architectures.

| Q/K coordinates retained | Held-out two-candidate decisions correct | Full-logit relative L2 error |
| --- | ---: | ---: |
| 16 of 16, unchanged control | 128 / 128 | 0% |
| 12 of 16 | 128 / 128 | 0.000528% |
| 8 of 16 | 128 / 128 | 0.001131% |
| 4 of 16 | 128 / 128 | 0.002017% |

Every budget preserved all 256 two-candidate task decisions, including the calibration
prompts. The minimum target margin remained 2.324364, even with only four coordinates.
This checkpoint's finite task tolerated the coordinate removal; we did not observe a
decision change at any of the tested budgets.

The unchanged-width Tensor implementation reproduced the original replay's complete logit
vector exactly on every prompt. Full-vocabulary top-1 accuracy stayed at 192/256 for every
budget: all 128 quote prompts and 64 of the 128 bracket prompts. The two-candidate task
therefore remains distinct from unrestricted token prediction. These are floating-point
observations on this checkpoint and prompt domain, not proofs that coordinate removal
preserves an arbitrary model's answers.

## Removing pixels from a digit classifier

We used the bundled trained 64-input, ten-class linear digit classifier. The first 120
examples fitted coordinate masks without labels; the remaining 240 were used for evaluation.
Removed pixels were replaced by their calibration means. The full model correctly classified
232 of those 240 examples.

All calculations below interpret the exported decimal numbers as exact rationals. The
certificate compares the approximate winner's margin with a bound on the discarded
winner–competitor logit difference. Strict inequalities reject ties.

| Selection | Pixels retained | Correct labels / 240 | Matches original / 240 | Certified matches |
| --- | ---: | ---: | ---: | ---: |
| Weight magnitude | 48 | 228 | 228 | 180 |
| Variance-aware | 48 | 232 | 240 | 237 |
| Greedy output error | 48 | 231 | 239 | 235 |
| Weight magnitude | 32 | 204 | 204 | 2 |
| Variance-aware | 32 | 220 | 222 | 131 |
| Greedy output error | 32 | 228 | 230 | 106 |
| Weight magnitude | 16 | 97 | 97 | 0 |
| Variance-aware | 16 | 182 | 183 | 0 |
| Greedy output error | 16 | 196 | 199 | 0 |

Variance-aware selection removed a quarter of the pixels without changing any held-out
prediction; exact margin checks certified 237 of the 240 matches. At half the pixels,
greedy selection did better. No one ranking won at every budget. A certified match means
agreement with the original classifier, including its mistakes, not correctness of the label.

The input-specific certificate reads the actual removed pixels. A more conservative bound
lets those pixels take any value in `[0,1]`, keeping retained pixels fixed. That bound
certified 19 cases for magnitude-48 and none for the other configurations. We have not
obtained a cheap region-wide certificate for the stronger summaries. Independent Python
rational arithmetic reproduced the selected variance-48 and greedy-32 experiments, including
every prediction, residual identity, exact MSE, and certificate flag.

## Quantizing the classifier's parameters

We also rounded all weights and biases nearest-even through FloatLib, using three formats
chosen before inspecting their evaluation results. Forward arithmetic remained exact rational
arithmetic, isolating the effect of parameter quantization. This is not a measurement of
binary16 or bfloat16 inference with rounded accumulations.

| Parameter format | Correct labels / 240 | Matches original / 240 | Certified matches |
| --- | ---: | ---: | ---: |
| Binary16, exponent 5 / fraction 10 | 232 | 240 | 240 |
| Bfloat16, exponent 8 / fraction 7 | 232 | 240 | 239 |
| IEEE binary, exponent 4 / fraction 3 | 230 | 238 | 231 |

The last format uses IEEE-style encoding; it is not a claim about a particular OCP FP8
format. An independent nearest-even rational implementation matched all 1,950 rounded
parameters and all 720 evaluation predictions, margin bounds, certificate flags, and exact
MSE values. The complete digits run, including fitting and all checks, took 59.17 seconds
through Lean's interpreter. We did not measure an inference speedup.

## Averaging independent inputs

The earlier experiments compare full and reduced outputs. For this experiment we ask a
different question: can we remove input randomness while bounding the change in the
expected output? We use constructed three-layer ReLU networks, rather than trained models,
so that independent calculations can check the answer exactly.

Four independent signs are retained. The other signs are replaced by their mean, zero.
The estimator enumerates the 16 retained states and propagates input-error radii through
the layers. For each ReLU it charges the local remainder at most
`max(0, radius - abs(reference preactivation))`, multiplied by the absolute cotangent
from the reduced execution's backward pass. The summary is fixed before the three final
linear readouts are supplied.

| Network | Independent input signs | Reduced states | First readout's absolute error | Error bound |
| --- | ---: | ---: | ---: | ---: |
| Small nonlinear control | 8 | 16 | 0.00456746 | 0.19498698 |
| Stable activation branches | 64 | 16 | 0 | 0 |
| Nonlinear branches | 64 | 16 | 0.00040516 | 0.19498698 |

The stable network has 180 nonzero weights from discarded inputs into its first layer.
An independent affine collapse certifies that its activation branches do not change over
the full sign cube; it recovers exactly the same expectation for all three readouts.
The nonlinear network's repeated discarded columns permit a separate binomial calculation:
976 weighted states represent all `2^64` inputs. This is a convenient reference, not a claim
that our estimator is faster than every method using that structure. Its bound is much
larger than the observed error, although smaller than the corresponding budget that ignores
distance to activation boundaries.

For the small control, exhaustive enumeration checks every one of the 256 inputs and 768
readout discrepancy identities. We also deliberately violate conditional centering:
for `ReLU(2*A + R)` with `A = R` a fair sign, replacing `R` by zero changes the mean from
`3/2` to `1`, although the reduced activation never approaches zero. The discarded input's
marginal mean is zero, but its conditional mean given `A` is not. The theorem's hypothesis
rejects this example.

All these calculations use Python's exact `Fraction` arithmetic. The general ReLU,
graph-composition and conditional-averaging theorems are checked separately in Lean;
the script does not produce a kernel proof term for its generated networks. The current
Lean graph bound also requires supplied, proved local remainder radii. An automatic
certificate generator and a trained-model comparison are not part of this experiment.

The complete record, including the signed intervals and all three readouts, is in
[`results-boundary.json`](results-boundary.json). Reproduce it with:

```bash
python3 week-06-heuristic-estimators/experiments/boundary.py
```

## Reproduce the model experiments

From the repository root, with the pinned dependencies built and the archived Week 3
checkpoint and validation shard present:

```bash
python3 week-06-heuristic-estimators/experiments/attention.py prepare /tmp/attention-run
attention_cpu=$(lake query TorchLean/torchlean_tensor_cpu_shared)
lake env lean --load-dynlib="$attention_cpu" --run \
  week-06-heuristic-estimators/experiments/Attention.lean \
  /tmp/attention-run/attention.bin /tmp/attention-run/replay.jsonl
python3 week-06-heuristic-estimators/experiments/attention.py compare \
  /tmp/attention-run --replay /tmp/attention-run/replay.jsonl
lake env lean --run week-06-heuristic-estimators/experiments/Certificate.lean \
  /tmp/attention-run/attention.bin 12
```

The measured replay used native compilation; the commands above use Lean's interpreter
and will not reproduce its timing. The shared library supplies TorchLean's packed CPU
tensor primitives; a plain `lake env lean --run` does not load it automatically. A
one-case interpreter replay with that library matched the native result bit for bit.
The small [results record](results.json) preserves the
checkpoint, input and source hashes, window offsets, every search baseline, numerical
comparisons, and exact rational certificates. The large activation and replay files remain
outside the repository.

For the expanded GPT grid, use `--calibration 64 --evaluation 128 --layer L --context T
--window-span 512 --position-anchor 512` with `L` in `1, 6, 12` and `T` in `64, 128, 512`.
Use a different output directory for each setting. The same replay and comparison commands
apply. To reproduce the certificate scope above, pass `1` rather than `12` to `Certificate.lean`.

The sparsemax experiment reuses a source module that already owns `main`, so its terminal
`#eval` runs with a focused Lean check, without `--run`:

```bash
tiny_cpu=$(lake query TorchLean/torchlean_tensor_cpu_shared)
lake env lean --load-dynlib="$tiny_cpu" week-06-heuristic-estimators/experiments/Tiny.lean
python3 week-06-heuristic-estimators/experiments/tiny.py \
  --output /tmp/tiny-results.json
lake env lean --run week-06-heuristic-estimators/experiments/Digits.lean \
  ../TorchLean/NN/Examples/Verification/Robustness/digits_linear_weights.json \
  ../TorchLean/NN/Examples/Verification/Robustness/digits_test.json
```

The digits paths assume the main TorchLean checkout is beside this repository. Both source
artifacts are read-only. [The expanded record](results-models.json) includes all nine GPT
settings, the digits results and independent validation summaries, and source/data hashes.
[The small-transformer record](results-tiny.json) also separates calibration, evaluation,
complete-domain, per-task, and full-vocabulary observations.
Its weight-export hash identifies `smt_weights.json`; the expanded record separately names
the archived safetensors file. The generalized GPT harness also regenerated the original
experiment's binary input bit for bit.

Output-aware summaries are worth investigating further. This run also shows the two costs
the next experiment has to address: errors on sensitive heads and the work needed to
construct or check the summary. It does not establish a new estimation algorithm or a
sampling advantage.
