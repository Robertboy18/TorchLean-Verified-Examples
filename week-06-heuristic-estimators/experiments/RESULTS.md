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
diagnostics on an x86_64 CPU, not a repeated, interleaved performance benchmark.

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
first head, the bound was 0.01867 and the measured maximum output error was about 0.00115
(that measured error isn't stored in `results.json`). The bounds are conservative. Gram
construction took 16.78 seconds; checking the proposed row bounds took 109 ms total.
Computing useful certificate premises is not free.

The general theorems were checked by Lean's kernel. The data-specific rational checker
ran as a program; it did not emit a kernel proof term for the input file. The real-arithmetic
theorem also does not prove the rounding behavior of the concrete floating-point replay.

## Different layers and longer histories

The follow-up used the same trained GPT checkpoint, with 64 calibration windows and 128
evaluation windows in each setting. We tested layers 1, 6, and 12 at 64, 128, and 512 tokens.
All settings share the same final source tokens and final position embedding, 511. Shorter
windows truncate the history while retaining their absolute positions. They are not ordinary
short prompts whose positions begin at zero. At deeper layers, removing history also changes
the upstream hidden states. So the layer-1, 64-token cell below (30.33%) isn't a rerun of the
first table's 24.10%: that run used positions 0 to 63, different token windows, and 256
evaluation windows.

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
every prediction, residual identity, exact MSE, and certificate flag. That validator isn't in
the repository; `results-models.json` keeps its summary and source hash.

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
format. An independent nearest-even rational implementation (also kept outside the repository,
with its summary in `results-models.json`) matched all 1,950 rounded parameters and all 720
evaluation predictions, margin bounds, certificate flags, and exact MSE values. The complete
digits run, including fitting and all checks, took about a minute through Lean's interpreter
(the record doesn't store the time). We did not measure an inference speedup.

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
python3 week-06-heuristic-estimators/experiments/boundary.py \
  --output week-06-heuristic-estimators/experiments/results-boundary.json
```

Without `--output` the script prints the record instead of writing it. A rerun matches the
committed file except for the `wall_seconds_including_reference` timings.

## Reproduce the model experiments

These runs need two files the repository doesn't distribute. The first is the archived binary32
Week 3 checkpoint that `attention.py` reads, at
`week-03-gpt-training/artifacts/gpt2-small-2.5b-continuation-20260806/final-parameters.tlf32`
(SHA-256 `cced9d9bf412bee57fdda623a3f88f6dc9186896a2efe5a8359f5f41cd2cbed0`, as recorded in
`results.json`). Checkpoints written by the current Week 3 code use a newer format
(`torchlean_state_bits_v1` JSON) that `attention.py` doesn't read. The second is the validation
shard, `week-03-gpt-training/data/fineweb-edu-continuation-1.5b/val.bin`, whose hash is recorded
too. The Python preparation ran with torch 2.11.0+cu128 on CUDA (also recorded in
`results.json`), not the `torch==2.13.0` that `requirements.txt` pins now. From the repository
root, with the Lean dependencies built:

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
  .lake/packages/TorchLean/NN/Examples/Verification/Robustness/digits_linear_weights.json \
  .lake/packages/TorchLean/NN/Examples/Verification/Robustness/digits_test.json
```

The two digits files ship with the TorchLean package that Lake fetches into `.lake/packages`,
and their hashes match the recorded ones. Both source artifacts are read-only.
[The expanded record](results-models.json) includes all nine GPT settings, the digits results
and independent validation summaries, and source/data hashes.
[The small-transformer record](results-tiny.json) also separates calibration, evaluation,
complete-domain, per-task, and full-vocabulary observations.
Its weight-export hash identifies `smt_weights.json`; the expanded record separately names
the archived safetensors file. The generalized GPT harness also regenerated the original
experiment's binary input bit for bit.

Output-aware summaries are worth investigating further. This run also shows the two costs
the next experiment has to address: errors on sensitive heads and the work needed to
construct or check the summary. It does not establish a new estimation algorithm or a
sampling advantage.

## Matching sampling on random ReLU networks

The article describes the benchmark that the matching sampling principle asks for, and this
section runs it. MAISI's statement asks for a structural estimator, whose advice can't depend on
the context, with mean squared error over contexts comparable to random sampling at comparable
cost. It names ARC's covariance propagation as an example that works well on random wide networks.
So we tested covariance propagation on random ReLU networks, together with a cheaper diagonal
version and the essay's own conditional-averaging certificate, against plain Monte Carlo given the
same number of FLOPs.

### What we ran

Each network maps 64 inputs through two hidden ReLU layers of width w (three in a stress setting)
to a linear readout, `y = a · h_L` with `h_l = ReLU(W_l h_{l-1} + b_l)`. A context is one draw of
the weights: `W_l ~ N(0, σ_w²/fan_in)`, `b_l ~ N(0, σ_b²)` and `a ~ N(0, 1/w)`. The inputs are
either 64 independent signs, as in the essay, or a standard Gaussian vector, which makes the first
layer exactly Gaussian. The grid crosses depth 2 and 3, both input laws, σ_w² ∈ {1, 2, 4},
σ_b ∈ {0, 0.1} and w ∈ {16, 64, 256, 1024}. That's 96 settings with 200 contexts each (100 at
width 1024), 16,800 contexts in all. Contexts share their standard-normal draws across σ_w², σ_b
and the input law. With σ_b = 0 a ReLU network is positively homogeneous, so the three σ_w²
values give the same relative errors and only rescale the absolute ones. The σ_b = 0.1 rows can
differ, but only slightly.

Every estimator sees the weights and nothing else that depends on the context.

- **S** is plain Monte Carlo. Its sample count n gives it the FLOPs of another estimator, and we
  also run 10 and 100 times that budget. The S columns below use the exact RMSE of unbiased
  sampling, `sqrt(mean_c Var_x(y) / n)`. Independent sampled runs are in the record as well and
  agree with it (median deviation 4%).
- **D** keeps one mean and one variance per neuron and drops every cross-covariance.
- **C** propagates the mean vector and the full covariance matrix. The affine maps are exact. Each
  ReLU uses the exact moments of a bivariate Gaussian with nonzero means: Rosenbaum's
  truncated-moment formulas, with the orthant probability from Genz's bivariate normal algorithm
  (20-point Gauss–Legendre). The one approximation is to treat each layer's preactivations as
  jointly Gaussian. For the mean, C stops at the diagonal of the last layer's covariance.
- **A** is the conditional averaging from the averaging experiment above, for sign inputs. It
  keeps the first k ∈ {4, 8} signs, sets the other 60 or 56 to zero, evaluates the 2^k reference
  states, and adds the `Averaging.mean_error_le` budget with the ReLU boundary radius, computed
  with `boundary.py`'s algorithm in float64.

The main target (T1) is each context's mean output `E_x[y]`. The second (T2) is the tail
probability `P_x[y > t]` with `t = μ_C + k σ_C` for k = 3 and 4, where μ_C and σ_C are C's own
mean and standard deviation of y. That threshold is a fixed rule applied to the weights, it uses
no sampling, and S, D and C are all scored at the same t. C's tail estimate is therefore the
Gaussian value Q(k) in every context.

Costs are analytic FLOP counts, given by `flops_forward`, `flops_D`, `flops_C` and `flops_A` in
`msp.py`. A multiply-add counts as two FLOPs, other arithmetic as one, and each exp, sin, asin,
square root or normal CDF as 20. Random-number generation isn't charged to S. If transcendental
functions cost one FLOP instead, C gets cheaper and its comparisons only improve (the record has
both versions).

The truth for each context is antithetic Monte Carlo in float64 on one A100, with 2^24 samples
at widths 16 and 64 and 2^23 at 256 and 1024. We subtract the truth's mean squared standard error
from every estimator's mean squared error. In the main grid the truth's standard error stays
below a tenth of C's RMSE (at most 0.099 of it), and the S columns don't depend on it. Every seed
is fixed. A two-context pilot estimated the grid at 1.25 hours, under our 3-hour cap, so the full
grid ran. It took 1.9 hours after we doubled the width-1024 truth.

### Checks before the sweep

The sweep reruns every check at its start and stores the results in the record.

- The univariate ReLU moments match quadrature to 4.4e-16 (relative to the scale). The bivariate
  moment matches an independent one-dimensional quadrature, which avoids the orthant formula, to
  7.6e-16 over 240 cases, including correlations of ±(1 − 10⁻⁶). Genz's orthant probability
  matches Owen's T function (`scipy.special.owens_t`) to 3.3e-16 for |ρ| < 0.925 and 4.2e-13
  above, over 80,000 cases. Only 12 pairs in the whole sweep needed the |ρ| ≥ 0.925 branch.
- On one-hidden-layer networks with Gaussian inputs, where the first layer is exactly Gaussian,
  D's and C's means agree with per-neuron quadrature to 2.2e-16, and C's output standard
  deviation agrees with per-pair quadrature to 4.4e-16. D's standard deviation is off by up to
  0.093, as it should be, because D drops the cross-covariances.
- A's bound holds on all 576 exhaustively enumerated sign networks (d ∈ {8, 12, 16}, widths 16
  and 64, depths 2 and 3, every σ, k ∈ {4, 8}, four contexts each). So do its signed interval and
  the pointwise radius and remainder certificates at every input and neuron. The actual error was
  at most 1.7% of the bound (median 0.14%).
- `boundary.py` itself, in exact rationals, gave the same estimates and bounds as our float64
  code, to 3e-14, on seven instances, three of them with 64 signs. Its exhaustive check of the
  exact pointwise identity passed on the four 10-sign instances.
- The GPU truth estimator matched exact answers on a Gaussian depth-one network and on a 16-sign
  depth-two network (z = 0.79 and −1.32).

### The mean output

Costs in FLOPs and the matched sample counts n = FLOPs / FLOPs per sample. Neither the input law nor the sigmas change them.

| Depth | Width | One sample | D | C (T1) | C (T2) | A, k=4 | A, k=8 | n for D | n for C | n for C (T2) | n for A4 | n for A8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 16 | 2.66e3 | 8.86e3 | 1.94e5 | 3.49e5 | 3.93e4 | 6.29e5 | 3 | 73 | 131 | 15 | 237 |
| 3 | 16 | 3.20e3 | 1.14e4 | 3.59e5 | 5.15e5 | 7.05e4 | 1.12e6 | 4 | 112 | 161 | 22 | 351 |
| 2 | 64 | 1.68e4 | 5.08e4 | 3.56e6 | 6.56e6 | 4.55e5 | 7.24e6 | 3 | 212 | 391 | 27 | 432 |
| 3 | 64 | 2.51e4 | 7.62e4 | 7.10e6 | 1.01e7 | 8.78e5 | 1.39e7 | 3 | 283 | 402 | 35 | 556 |
| 2 | 256 | 1.65e5 | 4.49e5 | 8.23e7 | 1.56e8 | 6.59e6 | 1.04e8 | 3 | 498 | 943 | 40 | 632 |
| 3 | 256 | 2.97e5 | 7.96e5 | 1.90e8 | 2.63e8 | 1.30e7 | 2.07e8 | 3 | 639 | 887 | 44 | 697 |
| 2 | 1024 | 2.23e6 | 5.73e6 | 2.93e9 | 5.72e9 | 1.03e8 | 1.63e9 | 3 | 1311 | 2560 | 46 | 728 |
| 3 | 1024 | 4.33e6 | 1.10e7 | 7.87e9 | 1.07e10 | 2.05e8 | 3.24e9 | 3 | 1816 | 2460 | 47 | 749 |

C beat sampling at its own FLOP budget in all 96 settings. Its RMSE was 0.08 to 0.51 times that
of FLOP-matched sampling, and at depth 2 and width 256 sampling needed about 150 times C's budget
to match it. C also beat 10 times its budget in 84 settings (all but depth 3 at width 16) and 100
times its budget in 13 (the twelve depth-2, width-256 settings and one at width 64). D does even
better per FLOP because it costs about three samples. Its RMSE was 0.06 to 0.18 times that of
three-sample sampling, though its relative error is about ten times C's (3.0 to 8.2% against 0.27
to 4.4%). Neither closure showed a measurable bias across contexts: the largest bias was 1.9
standard errors for C and 2.1 for D.

Width helps, but less and less. From width 16 to 256, C's relative RMSE falls by a factor of 6.5
to 8.7. From 256 to 1024 it falls by only 1.07 for sign inputs at depth 2 (0.36% to 0.34%) and by
1.35 to 1.56 elsewhere. C costs order w³ FLOPs and a sample costs order w², so the matched sample
count grows with width. C's advantage therefore peaks at width 256 and shrinks at 1024, from
0.08–0.09 to 0.10–0.13 at depth 2 and from 0.14–0.19 to 0.18–0.22 at depth 3. D's error levels off
between 3.0 and 3.6%.

The settings added later (the last table in this section) point to the input dimension as what
sets this floor. With 16 inputs, C's relative RMSE stays near 1.35% from width 256 to 1024, and its
advantage over matched sampling shrinks to 0.4. With 256 inputs it keeps falling, to 0.077% for
signs and 0.10% for Gaussians at width 1024. Sampling then needs 1,400 times C's FLOP budget
(signs) or 790 times (Gaussians) to match it. This fits the picture in which every hidden
preactivation is a function of only d independent inputs: a wider layer averages more neurons, but
it can't make a function of 16 inputs Gaussian. So whether covariance propagation improves with
width here depends on the input dimension growing with it.

In the tables below, the relative columns divide by the standard deviation of the truth across
contexts. C / S@C is C's RMSE divided by that of FLOP-matched sampling.

T1 (mean output), depth 2, sign inputs.

| sigma_w^2 | sigma_b | Width | Truth std | D RMSE | D rel. | S@D rel. | C RMSE | C rel. | C bias | S@C rel. | S@10C rel. | S@100C rel. | C / S@C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 16 | 0.34 | 0.0154 | 4.52% | 61.8% | 0.00855 | 2.51% | -0.00047 | 12.5% | 3.97% | 1.25% | 0.2 |
| 1 | 0.1 | 16 | 0.344 | 0.015 | 4.37% | 61.4% | 0.0084 | 2.44% | -0.00031 | 12.4% | 3.94% | 1.25% | 0.2 |
| 2 | 0 | 16 | 0.681 | 0.0308 | 4.52% | 61.8% | 0.0171 | 2.51% | -0.00095 | 12.5% | 3.97% | 1.25% | 0.2 |
| 2 | 0.1 | 16 | 0.681 | 0.0305 | 4.48% | 61.9% | 0.017 | 2.5% | -0.00076 | 12.5% | 3.97% | 1.26% | 0.2 |
| 4 | 0 | 16 | 1.36 | 0.0616 | 4.52% | 61.8% | 0.0342 | 2.51% | -0.0019 | 12.5% | 3.97% | 1.25% | 0.2 |
| 4 | 0.1 | 16 | 1.36 | 0.0614 | 4.51% | 61.9% | 0.0342 | 2.51% | -0.0017 | 12.6% | 3.97% | 1.26% | 0.2 |
| 1 | 0 | 64 | 0.342 | 0.0132 | 3.85% | 60.8% | 0.0025 | 0.731% | -0.00016 | 7.23% | 2.29% | 0.723% | 0.1 |
| 1 | 0.1 | 64 | 0.349 | 0.013 | 3.72% | 59.5% | 0.00244 | 0.7% | -0.00011 | 7.08% | 2.24% | 0.708% | 0.099 |
| 2 | 0 | 64 | 0.684 | 0.0263 | 3.85% | 60.8% | 0.005 | 0.731% | -0.00029 | 7.23% | 2.29% | 0.723% | 0.1 |
| 2 | 0.1 | 64 | 0.686 | 0.0262 | 3.82% | 60.6% | 0.00494 | 0.721% | -0.00028 | 7.21% | 2.28% | 0.72% | 0.1 |
| 4 | 0 | 64 | 1.37 | 0.0526 | 3.85% | 60.8% | 0.00998 | 0.729% | -0.0006 | 7.23% | 2.29% | 0.723% | 0.1 |
| 4 | 0.1 | 64 | 1.37 | 0.0525 | 3.84% | 60.8% | 0.00996 | 0.729% | -0.00061 | 7.24% | 2.29% | 0.724% | 0.1 |
| 1 | 0 | 256 | 0.356 | 0.0128 | 3.59% | 58% | 0.00131 | 0.369% | -0.00017 | 4.5% | 1.42% | 0.45% | 0.082 |
| 1 | 0.1 | 256 | 0.368 | 0.0127 | 3.46% | 56.3% | 0.00131 | 0.355% | -0.00016 | 4.37% | 1.38% | 0.437% | 0.081 |
| 2 | 0 | 256 | 0.711 | 0.0255 | 3.58% | 58% | 0.00262 | 0.369% | -0.00036 | 4.5% | 1.42% | 0.45% | 0.082 |
| 2 | 0.1 | 256 | 0.722 | 0.0255 | 3.53% | 57.2% | 0.00259 | 0.359% | -0.00033 | 4.44% | 1.41% | 0.444% | 0.081 |
| 4 | 0 | 256 | 1.42 | 0.051 | 3.58% | 58% | 0.00521 | 0.366% | -0.00063 | 4.5% | 1.42% | 0.45% | 0.081 |
| 4 | 0.1 | 256 | 1.43 | 0.051 | 3.56% | 57.6% | 0.00524 | 0.365% | -0.00064 | 4.47% | 1.41% | 0.447% | 0.082 |
| 1 | 0 | 1024 | 0.375 | 0.0117 | 3.11% | 54.3% | 0.00129 | 0.343% | -7.6e-05 | 2.6% | 0.821% | 0.26% | 0.13 |
| 1 | 0.1 | 1024 | 0.381 | 0.0116 | 3.06% | 53.6% | 0.00127 | 0.333% | -9.4e-05 | 2.56% | 0.81% | 0.256% | 0.13 |
| 2 | 0 | 1024 | 0.75 | 0.0234 | 3.11% | 54.3% | 0.0026 | 0.346% | -0.00014 | 2.6% | 0.821% | 0.26% | 0.13 |
| 2 | 0.1 | 1024 | 0.751 | 0.0234 | 3.12% | 54.2% | 0.00252 | 0.336% | -0.00021 | 2.59% | 0.82% | 0.259% | 0.13 |
| 4 | 0 | 1024 | 1.5 | 0.0467 | 3.11% | 54.3% | 0.00512 | 0.341% | -0.00032 | 2.6% | 0.821% | 0.26% | 0.13 |
| 4 | 0.1 | 1024 | 1.5 | 0.0469 | 3.13% | 54.4% | 0.00505 | 0.337% | -0.00033 | 2.6% | 0.822% | 0.26% | 0.13 |

T1 (mean output), depth 2, gauss inputs.

| sigma_w^2 | sigma_b | Width | Truth std | D RMSE | D rel. | S@D rel. | C RMSE | C rel. | C bias | S@C rel. | S@10C rel. | S@100C rel. | C / S@C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 16 | 0.339 | 0.0161 | 4.74% | 62.3% | 0.00937 | 2.76% | -0.00042 | 12.6% | 4% | 1.26% | 0.22 |
| 1 | 0.1 | 16 | 0.343 | 0.0157 | 4.57% | 61.8% | 0.00919 | 2.68% | -0.00027 | 12.5% | 3.97% | 1.25% | 0.21 |
| 2 | 0 | 16 | 0.678 | 0.0321 | 4.74% | 62.3% | 0.0188 | 2.77% | -0.00082 | 12.6% | 4% | 1.26% | 0.22 |
| 2 | 0.1 | 16 | 0.679 | 0.0319 | 4.7% | 62.3% | 0.0187 | 2.75% | -0.00068 | 12.6% | 4% | 1.26% | 0.22 |
| 4 | 0 | 16 | 1.36 | 0.0643 | 4.74% | 62.3% | 0.0375 | 2.76% | -0.0016 | 12.6% | 4% | 1.26% | 0.22 |
| 4 | 0.1 | 16 | 1.35 | 0.0641 | 4.73% | 62.4% | 0.0375 | 2.77% | -0.0014 | 12.6% | 4% | 1.27% | 0.22 |
| 1 | 0 | 64 | 0.341 | 0.0135 | 3.95% | 61.2% | 0.00319 | 0.937% | -0.00024 | 7.28% | 2.3% | 0.728% | 0.13 |
| 1 | 0.1 | 64 | 0.348 | 0.0133 | 3.81% | 59.9% | 0.00309 | 0.889% | -0.0002 | 7.13% | 2.25% | 0.713% | 0.12 |
| 2 | 0 | 64 | 0.681 | 0.0269 | 3.95% | 61.2% | 0.00639 | 0.938% | -0.00049 | 7.28% | 2.3% | 0.728% | 0.13 |
| 2 | 0.1 | 64 | 0.683 | 0.0268 | 3.92% | 61% | 0.00631 | 0.923% | -0.00045 | 7.26% | 2.3% | 0.726% | 0.13 |
| 4 | 0 | 64 | 1.36 | 0.0539 | 3.95% | 61.2% | 0.0127 | 0.935% | -0.00098 | 7.28% | 2.3% | 0.728% | 0.13 |
| 4 | 0.1 | 64 | 1.36 | 0.0537 | 3.95% | 61.3% | 0.0127 | 0.934% | -0.00096 | 7.29% | 2.31% | 0.729% | 0.13 |
| 1 | 0 | 256 | 0.354 | 0.0128 | 3.61% | 58.5% | 0.00149 | 0.42% | -9e-05 | 4.54% | 1.44% | 0.454% | 0.093 |
| 1 | 0.1 | 256 | 0.366 | 0.0128 | 3.49% | 56.7% | 0.00146 | 0.399% | -8.5e-05 | 4.4% | 1.39% | 0.44% | 0.091 |
| 2 | 0 | 256 | 0.708 | 0.0256 | 3.62% | 58.5% | 0.00299 | 0.423% | -0.00017 | 4.54% | 1.44% | 0.454% | 0.093 |
| 2 | 0.1 | 256 | 0.719 | 0.0257 | 3.57% | 57.7% | 0.00296 | 0.412% | -0.00016 | 4.48% | 1.42% | 0.448% | 0.092 |
| 4 | 0 | 256 | 1.42 | 0.0512 | 3.61% | 58.5% | 0.00596 | 0.421% | -0.0004 | 4.54% | 1.44% | 0.454% | 0.093 |
| 4 | 0.1 | 256 | 1.43 | 0.0512 | 3.59% | 58.1% | 0.00595 | 0.417% | -0.00034 | 4.51% | 1.43% | 0.451% | 0.093 |
| 1 | 0 | 1024 | 0.374 | 0.0115 | 3.09% | 54.7% | 0.001 | 0.269% | -0.0001 | 2.62% | 0.828% | 0.262% | 0.1 |
| 1 | 0.1 | 1024 | 0.38 | 0.0115 | 3.02% | 54% | 0.00101 | 0.266% | -0.0001 | 2.58% | 0.817% | 0.258% | 0.1 |
| 2 | 0 | 1024 | 0.747 | 0.0231 | 3.09% | 54.7% | 0.00202 | 0.27% | -0.00013 | 2.62% | 0.828% | 0.262% | 0.1 |
| 2 | 0.1 | 1024 | 0.748 | 0.0231 | 3.09% | 54.7% | 0.00203 | 0.271% | -0.00021 | 2.62% | 0.827% | 0.262% | 0.1 |
| 4 | 0 | 1024 | 1.49 | 0.0462 | 3.09% | 54.7% | 0.00411 | 0.275% | -0.00032 | 2.62% | 0.828% | 0.262% | 0.11 |
| 4 | 0.1 | 1024 | 1.49 | 0.0463 | 3.1% | 54.8% | 0.00406 | 0.272% | -0.00032 | 2.62% | 0.829% | 0.262% | 0.1 |

T1 (mean output), depth 3, sign inputs.

| sigma_w^2 | sigma_b | Width | Truth std | D RMSE | D rel. | S@D rel. | C RMSE | C rel. | C bias | S@C rel. | S@10C rel. | S@100C rel. | C / S@C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 16 | 0.252 | 0.0201 | 7.99% | 45.1% | 0.0103 | 4.1% | 0.00031 | 8.52% | 2.69% | 0.851% | 0.48 |
| 1 | 0.1 | 16 | 0.265 | 0.0194 | 7.33% | 43.2% | 0.0101 | 3.8% | 0.00018 | 8.17% | 2.58% | 0.816% | 0.46 |
| 2 | 0 | 16 | 0.713 | 0.0569 | 7.98% | 45.1% | 0.0292 | 4.1% | 0.00085 | 8.52% | 2.69% | 0.851% | 0.48 |
| 2 | 0.1 | 16 | 0.715 | 0.0567 | 7.93% | 45.1% | 0.0292 | 4.09% | 0.001 | 8.51% | 2.69% | 0.85% | 0.48 |
| 4 | 0 | 16 | 2.02 | 0.161 | 7.98% | 45.1% | 0.0826 | 4.09% | 0.0024 | 8.52% | 2.69% | 0.851% | 0.48 |
| 4 | 0.1 | 16 | 2.01 | 0.161 | 8.02% | 45.3% | 0.0829 | 4.12% | 0.0029 | 8.56% | 2.7% | 0.855% | 0.48 |
| 1 | 0 | 64 | 0.276 | 0.0126 | 4.57% | 47.3% | 0.00348 | 1.26% | 0.00011 | 4.87% | 1.54% | 0.487% | 0.26 |
| 1 | 0.1 | 64 | 0.283 | 0.0127 | 4.49% | 46.5% | 0.00325 | 1.15% | 0.00026 | 4.78% | 1.51% | 0.479% | 0.24 |
| 2 | 0 | 64 | 0.78 | 0.0357 | 4.57% | 47.3% | 0.00985 | 1.26% | 0.00031 | 4.87% | 1.54% | 0.487% | 0.26 |
| 2 | 0.1 | 64 | 0.778 | 0.0359 | 4.62% | 47.5% | 0.00969 | 1.25% | 0.00049 | 4.89% | 1.55% | 0.489% | 0.25 |
| 4 | 0 | 64 | 2.21 | 0.101 | 4.57% | 47.3% | 0.0279 | 1.26% | 0.00094 | 4.87% | 1.54% | 0.487% | 0.26 |
| 4 | 0.1 | 64 | 2.2 | 0.101 | 4.6% | 47.5% | 0.0278 | 1.26% | 0.0011 | 4.89% | 1.55% | 0.489% | 0.26 |
| 1 | 0 | 256 | 0.272 | 0.011 | 4.05% | 47.3% | 0.0013 | 0.476% | 0.00015 | 3.24% | 1.02% | 0.324% | 0.15 |
| 1 | 0.1 | 256 | 0.28 | 0.0112 | 4% | 46.3% | 0.00127 | 0.454% | 0.00015 | 3.17% | 1% | 0.317% | 0.14 |
| 2 | 0 | 256 | 0.769 | 0.0312 | 4.06% | 47.3% | 0.00367 | 0.476% | 0.00041 | 3.24% | 1.02% | 0.324% | 0.15 |
| 2 | 0.1 | 256 | 0.771 | 0.0315 | 4.09% | 47.3% | 0.00362 | 0.469% | 0.00041 | 3.24% | 1.03% | 0.324% | 0.14 |
| 4 | 0 | 256 | 2.18 | 0.0883 | 4.06% | 47.3% | 0.0104 | 0.476% | 0.0012 | 3.24% | 1.02% | 0.324% | 0.15 |
| 4 | 0.1 | 256 | 2.18 | 0.0887 | 4.08% | 47.4% | 0.0103 | 0.471% | 0.0011 | 3.25% | 1.03% | 0.325% | 0.15 |
| 1 | 0 | 1024 | 0.263 | 0.00957 | 3.63% | 48.1% | 0.000902 | 0.343% | 7.4e-05 | 1.96% | 0.619% | 0.196% | 0.18 |
| 1 | 0.1 | 1024 | 0.283 | 0.00919 | 3.25% | 45.3% | 0.000952 | 0.336% | 6.1e-05 | 1.84% | 0.582% | 0.184% | 0.18 |
| 2 | 0 | 1024 | 0.744 | 0.0271 | 3.64% | 48.1% | 0.00257 | 0.345% | 0.0002 | 1.96% | 0.619% | 0.196% | 0.18 |
| 2 | 0.1 | 1024 | 0.757 | 0.0264 | 3.48% | 47.5% | 0.00264 | 0.348% | 0.00018 | 1.93% | 0.611% | 0.193% | 0.18 |
| 4 | 0 | 1024 | 2.11 | 0.0765 | 3.63% | 48.1% | 0.00723 | 0.343% | 0.00054 | 1.96% | 0.619% | 0.196% | 0.18 |
| 4 | 0.1 | 1024 | 2.11 | 0.0753 | 3.56% | 48% | 0.00726 | 0.343% | 0.00055 | 1.95% | 0.617% | 0.195% | 0.18 |

T1 (mean output), depth 3, gauss inputs.

| sigma_w^2 | sigma_b | Width | Truth std | D RMSE | D rel. | S@D rel. | C RMSE | C rel. | C bias | S@C rel. | S@10C rel. | S@100C rel. | C / S@C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 16 | 0.251 | 0.0205 | 8.18% | 45.5% | 0.011 | 4.39% | 0.00033 | 8.6% | 2.72% | 0.859% | 0.51 |
| 1 | 0.1 | 16 | 0.264 | 0.0198 | 7.51% | 43.5% | 0.0107 | 4.04% | 0.00019 | 8.23% | 2.6% | 0.822% | 0.49 |
| 2 | 0 | 16 | 0.71 | 0.0581 | 8.18% | 45.5% | 0.0312 | 4.39% | 0.00092 | 8.6% | 2.72% | 0.859% | 0.51 |
| 2 | 0.1 | 16 | 0.713 | 0.0579 | 8.13% | 45.4% | 0.0312 | 4.37% | 0.0011 | 8.59% | 2.71% | 0.858% | 0.51 |
| 4 | 0 | 16 | 2.01 | 0.164 | 8.18% | 45.5% | 0.0883 | 4.4% | 0.0025 | 8.6% | 2.72% | 0.859% | 0.51 |
| 4 | 0.1 | 16 | 2 | 0.164 | 8.21% | 45.7% | 0.0884 | 4.42% | 0.0031 | 8.63% | 2.73% | 0.862% | 0.51 |
| 1 | 0 | 64 | 0.275 | 0.0128 | 4.65% | 47.8% | 0.00418 | 1.52% | 3.1e-05 | 4.92% | 1.56% | 0.492% | 0.31 |
| 1 | 0.1 | 64 | 0.282 | 0.0128 | 4.54% | 46.9% | 0.00385 | 1.37% | 0.0002 | 4.83% | 1.53% | 0.483% | 0.28 |
| 2 | 0 | 64 | 0.777 | 0.0362 | 4.65% | 47.8% | 0.0118 | 1.52% | 0.00012 | 4.92% | 1.56% | 0.492% | 0.31 |
| 2 | 0.1 | 64 | 0.775 | 0.0363 | 4.69% | 48% | 0.0116 | 1.5% | 0.00031 | 4.94% | 1.56% | 0.494% | 0.3 |
| 4 | 0 | 64 | 2.2 | 0.102 | 4.65% | 47.8% | 0.0334 | 1.52% | 0.00029 | 4.92% | 1.56% | 0.492% | 0.31 |
| 4 | 0.1 | 64 | 2.19 | 0.102 | 4.68% | 47.9% | 0.0333 | 1.52% | 0.00047 | 4.94% | 1.56% | 0.494% | 0.31 |
| 1 | 0 | 256 | 0.271 | 0.0111 | 4.08% | 47.7% | 0.00171 | 0.631% | 0.00012 | 3.27% | 1.03% | 0.327% | 0.19 |
| 1 | 0.1 | 256 | 0.279 | 0.0112 | 4.02% | 46.7% | 0.00164 | 0.588% | 0.00014 | 3.2% | 1.01% | 0.32% | 0.18 |
| 2 | 0 | 256 | 0.766 | 0.0313 | 4.08% | 47.7% | 0.0048 | 0.626% | 0.00035 | 3.27% | 1.03% | 0.327% | 0.19 |
| 2 | 0.1 | 256 | 0.768 | 0.0316 | 4.11% | 47.8% | 0.00475 | 0.619% | 0.00037 | 3.27% | 1.04% | 0.327% | 0.19 |
| 4 | 0 | 256 | 2.17 | 0.0884 | 4.08% | 47.7% | 0.0136 | 0.627% | 0.001 | 3.27% | 1.03% | 0.327% | 0.19 |
| 4 | 0.1 | 256 | 2.17 | 0.0888 | 4.1% | 47.8% | 0.0136 | 0.627% | 0.0011 | 3.28% | 1.04% | 0.328% | 0.19 |
| 1 | 0 | 1024 | 0.262 | 0.00955 | 3.64% | 48.6% | 0.00113 | 0.432% | 3e-05 | 1.97% | 0.624% | 0.197% | 0.22 |
| 1 | 0.1 | 1024 | 0.282 | 0.00919 | 3.26% | 45.7% | 0.00113 | 0.401% | 7e-06 | 1.86% | 0.587% | 0.186% | 0.22 |
| 2 | 0 | 1024 | 0.742 | 0.027 | 3.64% | 48.6% | 0.00324 | 0.437% | 0.00011 | 1.97% | 0.624% | 0.197% | 0.22 |
| 2 | 0.1 | 1024 | 0.754 | 0.0264 | 3.5% | 48% | 0.00321 | 0.426% | 5.6e-05 | 1.95% | 0.616% | 0.195% | 0.22 |
| 4 | 0 | 1024 | 2.1 | 0.0764 | 3.64% | 48.6% | 0.00907 | 0.433% | 0.00016 | 1.97% | 0.624% | 0.197% | 0.22 |
| 4 | 0.1 | 1024 | 2.11 | 0.0754 | 3.58% | 48.5% | 0.00906 | 0.43% | 0.00015 | 1.97% | 0.623% | 0.197% | 0.22 |

### Measured wall time

This GPU favors the closures less than the FLOP count does. Our C launches a few dozen small
kernels from Python, while sampling at these sizes is a handful of large kernels that take about
half a millisecond whatever n is. Matched to measured median wall time on the A100, sampling beat
C in all 72 settings up to width 256 and lost in all 24 at width 1024. There C's RMSE was 0.32 to
0.70 times sampling's. Sampling beat D at equal wall time everywhere. A fused or batched closure
would change the small-width rows, but we haven't written one.

Measured GPU wall time (median per context, sign inputs, sigma_w^2 = 2, sigma_b = 0.1). n_wall is the largest S budget that ran no slower than C.

| Depth | Width | D | C (T1) | A, k=8 | S@C | S@100C | Truth | n for C | n_wall | C / S at n_wall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 16 | 0.92 ms | 2.84 ms | 1.11 ms | 0.52 ms | 0.52 ms | 49 ms | 73 | 23673 | 3.6 |
| 3 | 16 | 1.23 ms | 5.35 ms | 1.64 ms | 0.56 ms | 0.56 ms | 57 ms | 112 | 35141 | 8.5 |
| 2 | 64 | 0.93 ms | 2.91 ms | 1.14 ms | 0.54 ms | 0.53 ms | 82 ms | 212 | 43156 | 1.4 |
| 3 | 64 | 1.25 ms | 5.49 ms | 1.67 ms | 0.59 ms | 0.58 ms | 106 ms | 283 | 55570 | 3.6 |
| 2 | 256 | 0.91 ms | 2.96 ms | 1.12 ms | 0.53 ms | 1.26 ms | 156 ms | 498 | 94325 | 1.1 |
| 3 | 256 | 1.25 ms | 5.54 ms | 1.62 ms | 0.57 ms | 2.21 ms | 240 ms | 639 | 88687 | 1.7 |
| 2 | 1024 | 1.04 ms | 3.78 ms | 1.15 ms | 0.62 ms | 20.99 ms | 1317 ms | 1311 | 13108 | 0.41 |
| 3 | 1024 | 1.32 ms | 6.92 ms | 1.59 ms | 0.93 ms | 52.10 ms | 2400 ms | 1816 | 18160 | 0.57 |

### Conditional averaging

A was the weakest estimator here, and its certificate was always correct but never informative.
Keeping 4 or 8 of 64 signs recovers little of the variation across contexts. A's relative RMSE
was 56 to 75%, against 3 to 28% for sampling at the same cost, so A was 2.6 to 21 times worse than
sampling. The bound held in all 16,800 combinations of context and k, without any allowance for
the truth's sampling error, and no signed interval was missed by more than four standard errors.
But the median bound was 194 to 380,000 times the standard deviation of the truth across
contexts, growing with width and depth, and it was at least 122 times the actual error in every
context.

The reason is the dense first layer. Every first-layer neuron receives 56 or more discarded signs
through Gaussian weights, so the worst-case radius of its discarded part is about six times the
preactivation's standard deviation. Every neuron is then charged as though it could cross zero,
and the radius compounds through the later layers. The method suits networks whose discarded
inputs rarely move a neuron across its boundary, like the constructed networks above, and random
dense networks are the opposite case. In the last column of these tables, each context counts
once for each k.

A (conditional averaging), depth 2, sign inputs. The bound columns are medians over contexts.

| sigma_w^2 | sigma_b | Width | A4 RMSE | A4 rel. | S@A4 rel. | A4 bound | A4 bound / truth std | A8 RMSE | A8 rel. | S@A8 rel. | A8 bound | A8 bound / truth std | Bound holds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 16 | 0.254 | 74.5% | 27.7% | 73 | 214 | 0.218 | 64.2% | 6.96% | 67.2 | 197 | 400/400 |
| 1 | 0.1 | 16 | 0.244 | 70.8% | 27.5% | 72.3 | 210 | 0.212 | 61.5% | 6.91% | 66.8 | 194 | 400/400 |
| 2 | 0 | 16 | 0.507 | 74.5% | 27.7% | 146 | 214 | 0.437 | 64.2% | 6.96% | 134 | 197 | 400/400 |
| 2 | 0.1 | 16 | 0.5 | 73.4% | 27.7% | 145 | 213 | 0.432 | 63.4% | 6.96% | 134 | 197 | 400/400 |
| 4 | 0 | 16 | 1.01 | 74.5% | 27.7% | 292 | 214 | 0.874 | 64.2% | 6.96% | 269 | 197 | 400/400 |
| 4 | 0.1 | 16 | 1.01 | 74.2% | 27.7% | 292 | 214 | 0.87 | 64% | 6.97% | 268 | 197 | 400/400 |
| 1 | 0 | 64 | 0.258 | 75.4% | 20.3% | 263 | 769 | 0.221 | 64.5% | 5.06% | 243 | 712 | 400/400 |
| 1 | 0.1 | 64 | 0.245 | 70.2% | 19.8% | 262 | 750 | 0.212 | 60.7% | 4.96% | 242 | 694 | 400/400 |
| 2 | 0 | 64 | 0.516 | 75.4% | 20.3% | 526 | 769 | 0.441 | 64.5% | 5.06% | 487 | 712 | 400/400 |
| 2 | 0.1 | 64 | 0.506 | 73.7% | 20.2% | 525 | 765 | 0.435 | 63.4% | 5.05% | 486 | 708 | 400/400 |
| 4 | 0 | 64 | 1.03 | 75.4% | 20.3% | 1.05e+03 | 769 | 0.883 | 64.5% | 5.06% | 974 | 712 | 400/400 |
| 4 | 0.1 | 64 | 1.02 | 74.9% | 20.3% | 1.05e+03 | 768 | 0.877 | 64.2% | 5.07% | 973 | 712 | 400/400 |
| 1 | 0 | 256 | 0.266 | 74.9% | 15.9% | 1.01e+03 | 2.84e+03 | 0.23 | 64.8% | 4% | 933 | 2.62e+03 | 400/400 |
| 1 | 0.1 | 256 | 0.259 | 70.3% | 15.4% | 1.01e+03 | 2.74e+03 | 0.225 | 61.2% | 3.88% | 932 | 2.53e+03 | 400/400 |
| 2 | 0 | 256 | 0.533 | 74.9% | 15.9% | 2.02e+03 | 2.84e+03 | 0.461 | 64.8% | 4% | 1.87e+03 | 2.62e+03 | 400/400 |
| 2 | 0.1 | 256 | 0.526 | 72.8% | 15.7% | 2.02e+03 | 2.8e+03 | 0.457 | 63.3% | 3.94% | 1.86e+03 | 2.58e+03 | 400/400 |
| 4 | 0 | 256 | 1.07 | 74.9% | 15.9% | 4.04e+03 | 2.84e+03 | 0.922 | 64.8% | 4% | 3.73e+03 | 2.62e+03 | 400/400 |
| 4 | 0.1 | 256 | 1.06 | 73.9% | 15.8% | 4.04e+03 | 2.82e+03 | 0.918 | 64% | 3.97% | 3.73e+03 | 2.6e+03 | 400/400 |
| 1 | 0 | 1024 | 0.281 | 74.9% | 13.9% | 3.95e+03 | 1.05e+04 | 0.24 | 64% | 3.49% | 3.65e+03 | 9.72e+03 | 200/200 |
| 1 | 0.1 | 1024 | 0.27 | 70.8% | 13.7% | 3.94e+03 | 1.03e+04 | 0.233 | 61.2% | 3.44% | 3.64e+03 | 9.56e+03 | 200/200 |
| 2 | 0 | 1024 | 0.562 | 74.9% | 13.9% | 7.89e+03 | 1.05e+04 | 0.48 | 64% | 3.49% | 7.3e+03 | 9.72e+03 | 200/200 |
| 2 | 0.1 | 1024 | 0.553 | 73.6% | 13.8% | 7.89e+03 | 1.05e+04 | 0.474 | 63.1% | 3.48% | 7.29e+03 | 9.7e+03 | 200/200 |
| 4 | 0 | 1024 | 1.12 | 74.9% | 13.9% | 1.58e+04 | 1.05e+04 | 0.96 | 64% | 3.49% | 1.46e+04 | 9.72e+03 | 200/200 |
| 4 | 0.1 | 1024 | 1.12 | 74.5% | 13.9% | 1.58e+04 | 1.05e+04 | 0.955 | 63.7% | 3.49% | 1.46e+04 | 9.74e+03 | 200/200 |

A (conditional averaging), depth 3, sign inputs. The bound columns are medians over contexts.

| sigma_w^2 | sigma_b | Width | A4 RMSE | A4 rel. | S@A4 rel. | A4 bound | A4 bound / truth std | A8 RMSE | A8 rel. | S@A8 rel. | A8 bound | A8 bound / truth std | Bound holds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 16 | 0.186 | 73.9% | 19.2% | 237 | 941 | 0.16 | 63.6% | 4.82% | 218 | 867 | 400/400 |
| 1 | 0.1 | 16 | 0.173 | 65.1% | 18.4% | 238 | 898 | 0.151 | 57% | 4.61% | 219 | 825 | 400/400 |
| 2 | 0 | 16 | 0.527 | 73.9% | 19.2% | 671 | 941 | 0.454 | 63.6% | 4.81% | 618 | 867 | 400/400 |
| 2 | 0.1 | 16 | 0.513 | 71.7% | 19.2% | 673 | 941 | 0.446 | 62.3% | 4.81% | 619 | 866 | 400/400 |
| 4 | 0 | 16 | 1.49 | 73.9% | 19.2% | 1.9e+03 | 941 | 1.28 | 63.6% | 4.82% | 1.75e+03 | 867 | 400/400 |
| 4 | 0.1 | 16 | 1.48 | 73.4% | 19.3% | 1.9e+03 | 945 | 1.28 | 63.5% | 4.83% | 1.75e+03 | 871 | 400/400 |
| 1 | 0 | 64 | 0.208 | 75.5% | 13.8% | 1.72e+03 | 6.23e+03 | 0.18 | 65.2% | 3.47% | 1.59e+03 | 5.75e+03 | 400/400 |
| 1 | 0.1 | 64 | 0.194 | 68.7% | 13.6% | 1.72e+03 | 6.07e+03 | 0.171 | 60.3% | 3.41% | 1.59e+03 | 5.61e+03 | 400/400 |
| 2 | 0 | 64 | 0.589 | 75.5% | 13.8% | 4.86e+03 | 6.23e+03 | 0.509 | 65.2% | 3.47% | 4.49e+03 | 5.75e+03 | 400/400 |
| 2 | 0.1 | 64 | 0.578 | 74.3% | 13.9% | 4.84e+03 | 6.23e+03 | 0.501 | 64.4% | 3.49% | 4.48e+03 | 5.76e+03 | 400/400 |
| 4 | 0 | 64 | 1.67 | 75.5% | 13.8% | 1.38e+04 | 6.23e+03 | 1.44 | 65.2% | 3.47% | 1.27e+04 | 5.75e+03 | 400/400 |
| 4 | 0.1 | 64 | 1.66 | 75.4% | 13.9% | 1.37e+04 | 6.23e+03 | 1.43 | 65.1% | 3.49% | 1.27e+04 | 5.77e+03 | 400/400 |
| 1 | 0 | 256 | 0.205 | 75.3% | 12.3% | 1.3e+04 | 4.79e+04 | 0.176 | 64.7% | 3.1% | 1.21e+04 | 4.43e+04 | 400/400 |
| 1 | 0.1 | 256 | 0.191 | 68.1% | 12.1% | 1.3e+04 | 4.64e+04 | 0.167 | 59.7% | 3.04% | 1.2e+04 | 4.3e+04 | 400/400 |
| 2 | 0 | 256 | 0.579 | 75.3% | 12.3% | 3.69e+04 | 4.79e+04 | 0.498 | 64.7% | 3.1% | 3.41e+04 | 4.43e+04 | 400/400 |
| 2 | 0.1 | 256 | 0.569 | 73.8% | 12.4% | 3.68e+04 | 4.78e+04 | 0.492 | 63.8% | 3.1% | 3.41e+04 | 4.42e+04 | 400/400 |
| 4 | 0 | 256 | 1.64 | 75.3% | 12.3% | 1.04e+05 | 4.79e+04 | 1.41 | 64.7% | 3.1% | 9.64e+04 | 4.43e+04 | 400/400 |
| 4 | 0.1 | 256 | 1.63 | 75% | 12.4% | 1.04e+05 | 4.79e+04 | 1.4 | 64.5% | 3.11% | 9.64e+04 | 4.43e+04 | 400/400 |
| 1 | 0 | 1024 | 0.197 | 74.7% | 12.2% | 1e+05 | 3.82e+05 | 0.169 | 64.1% | 3.05% | 9.29e+04 | 3.53e+05 | 200/200 |
| 1 | 0.1 | 1024 | 0.18 | 63.6% | 11.4% | 1e+05 | 3.54e+05 | 0.158 | 55.8% | 2.87% | 9.28e+04 | 3.28e+05 | 200/200 |
| 2 | 0 | 1024 | 0.556 | 74.7% | 12.2% | 2.84e+05 | 3.82e+05 | 0.477 | 64.1% | 3.05% | 2.63e+05 | 3.53e+05 | 200/200 |
| 2 | 0.1 | 1024 | 0.541 | 71.5% | 12% | 2.84e+05 | 3.75e+05 | 0.468 | 61.8% | 3.01% | 2.63e+05 | 3.47e+05 | 200/200 |
| 4 | 0 | 1024 | 1.57 | 74.7% | 12.2% | 8.04e+05 | 3.82e+05 | 1.35 | 64.1% | 3.05% | 7.43e+05 | 3.53e+05 | 200/200 |
| 4 | 0.1 | 1024 | 1.56 | 73.8% | 12.1% | 8.04e+05 | 3.8e+05 | 1.34 | 63.4% | 3.04% | 7.43e+05 | 3.52e+05 | 200/200 |

### The tail probability

The Gaussian closure is a poor tail estimator at small width and a fair one at large width. At
width 16 the true probability above C's 3σ threshold was 2.9 to 4.2 times Q(3) on average, and
above 4σ it was 17 to 38 times Q(4). At width 1024 the 3σ probability averaged 0.0013 to 0.0017
against Q(3) = 0.00135, and C's relative RMSE was 17 to 35%. Sampling at matched FLOPs sees too
few samples of these events. C beat it in 84 of 96 settings at k = 3 and in all 96 at k = 4. It
beat 10 times the budget in 12 and 42 settings and never beat 100 times. D's tail, evaluated at
C's threshold, was worse than C's in all 192 comparisons. The record has every setting, as well as
the true probability at D's own threshold.

T2 (tail probability at t = C mean + k C sd), sigma_w^2 = 2, sigma_b = 0.1. Relative RMSE divides by the mean true p over contexts.

| Depth | Input | Width | k | Mean true p | Q(k) | C rel. | D rel. | S@C2 n | S@C2 rel. | S@10C2 rel. | S@100C2 rel. |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | sign | 16 | 3 | 0.00391 | 0.00135 | 125% | 150% | 131 | 139% | 43.9% | 13.9% |
| 2 | sign | 16 | 4 | 0.000545 | 3.17e-05 | 187% | 196% | 131 | 374% | 118% | 37.3% |
| 2 | gauss | 16 | 3 | 0.00427 | 0.00135 | 124% | 143% | 131 | 133% | 42% | 13.3% |
| 2 | gauss | 16 | 4 | 0.000677 | 3.17e-05 | 178% | 182% | 131 | 335% | 106% | 33.5% |
| 3 | sign | 16 | 3 | 0.0053 | 0.00135 | 132% | 183% | 161 | 108% | 34.1% | 10.8% |
| 3 | sign | 16 | 4 | 0.00103 | 3.17e-05 | 183% | 225% | 161 | 245% | 77.6% | 24.5% |
| 3 | gauss | 16 | 3 | 0.00564 | 0.00135 | 131% | 174% | 161 | 104% | 33% | 10.4% |
| 3 | gauss | 16 | 4 | 0.00119 | 3.17e-05 | 177% | 208% | 161 | 228% | 72% | 22.8% |
| 2 | sign | 64 | 3 | 0.00192 | 0.00135 | 74.6% | 119% | 391 | 115% | 36.5% | 11.5% |
| 2 | sign | 64 | 4 | 0.000105 | 3.17e-05 | 197% | 242% | 391 | 493% | 156% | 49.3% |
| 2 | gauss | 64 | 3 | 0.00229 | 0.00135 | 80.5% | 108% | 391 | 106% | 33.4% | 10.6% |
| 2 | gauss | 64 | 4 | 0.000171 | 3.17e-05 | 174% | 188% | 391 | 386% | 122% | 38.6% |
| 3 | sign | 64 | 3 | 0.00213 | 0.00135 | 88.6% | 133% | 402 | 108% | 34.1% | 10.8% |
| 3 | sign | 64 | 4 | 0.000158 | 3.17e-05 | 172% | 218% | 402 | 397% | 125% | 39.7% |
| 3 | gauss | 64 | 3 | 0.00252 | 0.00135 | 94.6% | 120% | 402 | 99.1% | 31.3% | 9.9% |
| 3 | gauss | 64 | 4 | 0.000241 | 3.17e-05 | 168% | 179% | 402 | 321% | 101% | 32.1% |
| 2 | sign | 256 | 3 | 0.00137 | 0.00135 | 27% | 91.8% | 943 | 87.9% | 27.8% | 8.79% |
| 2 | sign | 256 | 4 | 3.49e-05 | 3.17e-05 | 62.2% | 230% | 943 | 551% | 174% | 55.1% |
| 2 | gauss | 256 | 3 | 0.00175 | 0.00135 | 40% | 73.9% | 943 | 77.7% | 24.6% | 7.77% |
| 2 | gauss | 256 | 4 | 7.58e-05 | 3.17e-05 | 87.1% | 119% | 943 | 374% | 118% | 37.4% |
| 3 | sign | 256 | 3 | 0.00147 | 0.00135 | 37.2% | 77.9% | 887 | 87.4% | 27.6% | 8.74% |
| 3 | sign | 256 | 4 | 4.6e-05 | 3.17e-05 | 84.7% | 148% | 887 | 495% | 157% | 49.5% |
| 3 | gauss | 256 | 3 | 0.00186 | 0.00135 | 51.3% | 69.3% | 887 | 77.7% | 24.6% | 7.77% |
| 3 | gauss | 256 | 4 | 9.32e-05 | 3.17e-05 | 106% | 112% | 887 | 348% | 110% | 34.8% |
| 2 | sign | 1024 | 3 | 0.00127 | 0.00135 | 19.3% | 76.8% | 2560 | 55.4% | 17.5% | 5.54% |
| 2 | sign | 1024 | 4 | 2.66e-05 | 3.17e-05 | 47.9% | 206% | 2560 | 383% | 121% | 38.3% |
| 2 | gauss | 1024 | 3 | 0.00164 | 0.00135 | 29.9% | 59% | 2560 | 48.8% | 15.4% | 4.88% |
| 2 | gauss | 1024 | 4 | 6.08e-05 | 3.17e-05 | 69.9% | 93.9% | 2560 | 253% | 80.2% | 25.3% |
| 3 | sign | 1024 | 3 | 0.00132 | 0.00135 | 17% | 62.5% | 2460 | 55.4% | 17.5% | 5.54% |
| 3 | sign | 1024 | 4 | 2.98e-05 | 3.17e-05 | 40.4% | 141% | 2460 | 369% | 117% | 36.9% |
| 3 | gauss | 1024 | 3 | 0.00169 | 0.00135 | 34.3% | 51.9% | 2460 | 49% | 15.5% | 4.9% |
| 3 | gauss | 1024 | 4 | 6.71e-05 | 3.17e-05 | 81.2% | 85.4% | 2460 | 246% | 77.8% | 24.6% |

### Settings added later

We added these eight settings after seeing the width-1024 results, to test whether the floor
depends on the input dimension. They use depth 2, σ_w² = 2, σ_b = 0.1 and the same context
counts, and the d = 64 rows repeat the main grid. At d = 256 and width 1024 the truth's standard
error is 0.37 of C's RMSE for signs and 0.28 for Gaussians, so the subtraction matters there.

Added later: input dimension, depth 2, sigma_w^2 = 2, sigma_b = 0.1 (the d = 64 rows come from the main grid).

| Input | d | Width | Truth std | D rel. | C rel. | S@C n | S@C rel. | C / S@C |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gauss | 16 | 256 | 0.669 | 7.71% | 1.33% | 540 | 4.6% | 0.29 |
| gauss | 16 | 1024 | 0.576 | 8.57% | 1.32% | 1324 | 3.43% | 0.39 |
| gauss | 64 | 256 | 0.719 | 3.57% | 0.412% | 498 | 4.48% | 0.092 |
| gauss | 64 | 1024 | 0.748 | 3.09% | 0.271% | 1311 | 2.62% | 0.1 |
| gauss | 256 | 256 | 0.647 | 2.03% | 0.239% | 408 | 5.45% | 0.044 |
| gauss | 256 | 1024 | 0.696 | 1.64% | 0.103% | 1268 | 2.88% | 0.036 |
| sign | 16 | 256 | 0.679 | 7.73% | 1.35% | 540 | 4.47% | 0.3 |
| sign | 16 | 1024 | 0.585 | 8.56% | 1.41% | 1324 | 3.34% | 0.42 |
| sign | 64 | 256 | 0.722 | 3.53% | 0.359% | 498 | 4.44% | 0.081 |
| sign | 64 | 1024 | 0.751 | 3.12% | 0.336% | 1311 | 2.59% | 0.13 |
| sign | 256 | 256 | 0.648 | 2.02% | 0.191% | 408 | 5.44% | 0.035 |
| sign | 256 | 1024 | 0.697 | 1.64% | 0.0768% | 1268 | 2.88% | 0.027 |

### What this does and doesn't show

C is a structural estimator with fixed advice, and it beat FLOP-matched sampling on the mean
output of random ReLU networks in every setting we ran. In RMSE it won by up to a factor of 12 in
the main grid and 37 with 256 inputs. That's the kind of result the matching sampling principle
asks for, on the family where it was expected to hold, but it isn't more than that. These are
random networks at initialization, where a Gaussian closure is known to describe the layers
well. Trained networks, attention, and the shared-branch cases from the essay are untested. C
carries no error bound of its own, and the one certified method here, A, was sound but
uninformative on this family. The verdict depends on the cost model: per FLOP the closures win,
while per GPU millisecond in our implementation they lose below width 1024. And at a fixed input
dimension, C's advantage over sampling stops growing with width.

### Reproduce the matching-sampling benchmark

```bash
python3 week-06-heuristic-estimators/experiments/msp.py validate --output /tmp/msp-validation.json
python3 week-06-heuristic-estimators/experiments/msp.py pilot --contexts 2 \
  --output /tmp/msp-run/pilot.json
python3 week-06-heuristic-estimators/experiments/msp.py sweep --added-later \
  --pilot /tmp/msp-run/pilot.json --work-dir /tmp/msp-run/work \
  --output week-06-heuristic-estimators/experiments/results-msp-full.json
python3 week-06-heuristic-estimators/experiments/msp.py tables \
  week-06-heuristic-estimators/experiments/results-msp-full.json
cd week-06-heuristic-estimators/experiments && python3 split_msp.py
```

We ran these on one A100 (`CUDA_VISIBLE_DEVICES=5`) with torch 2.13.0+cu130 and scipy 1.18.0.
Create `/tmp/msp-run` first. The sweep checks first and stops if any check fails. It reuses every
setting already in `--work-dir`, so an interrupted run resumes, and `--only` takes
comma-separated keys such as `L2-sign-sw2-sb0.1-w64` to rerun single settings. With fixed seeds
and deterministic GPU algorithms, a rerun on the same GPU model and software should reproduce
everything except the wall times.

Two details of our own run differ from these commands. The pilot ran before we doubled the
width-1024 truth, so its record shows 2^21 pairs there. The 96 main settings came from an earlier
revision of `msp.py` (SHA-256 `bcd69da3e4d9508ad30b05e9c714ae508bc18171bd776423e8b5685dffecf551`),
which lacked the added-later settings, the wall-time comparison, `--only` and the current table
layout. We then ran the `sweep --added-later` command above with the committed script on the
same work directory. That reran the checks, reused the 96 settings and added the new eight.
Rerunning two of the main settings with the committed script reproduced every per-context value
bit for bit. The full record stores the committed script's hash. It is about 12 MB, mostly
per-context values, so `split_msp.py` keeps the summary, configuration and checks in
`results-msp.json` and moves the per-context values to `results-msp-contexts.json.gz`;
`python3 split_msp.py --join` rebuilds the full record exactly.
