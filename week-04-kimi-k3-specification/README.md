# Kimi K3 in Lean

Kimi K3's technical report combines several recent ideas in one model: recurrent Kimi Delta
Attention, periodic latent attention, block-level residual connections, a routed mixture of
experts, a vision encoder, Muon updates, reinforcement learning, quantization-aware training, and
speculative decoding. If you read those components only as implementation prose, it's hard to see
which equations compose, or which properties survive when you evaluate the model in chunks.

Here I've written K3's architecture equations as shape-indexed tensor functions. I also built the
core language operations from primitive TorchLean graph nodes, and Lean proves that the
packed one-token graph returns the same updated state and vocabulary logits as those packed
functions. K3's published dimensions are only one valid instance, and small instances let us
evaluate the same equations without constructing the 2.78-trillion-parameter checkpoint. MoonViT
embedding, encoder blocks, and merge/project have primitive-graph denotation theorems too. And a
separate proof shows that the public sequence calculation agrees with stateful token-by-token
evaluation, automatic expert routing included.

The [annotated formalization](site/index.html) rebuilds the relevant parts of the report as a
web-native article, with paper equations, diagrams, and interactive explanations next to the
matching Lean definitions, proof assumptions, and boundaries. Its styles and Lean blocks come from
the site build, so if you want to read it locally, build the site from the repository root:

```bash
python3 -m pip install -r site/requirements.txt
python3 site/build.py
```

and open `build/site/week-04-kimi-k3-specification/site/index.html` in a browser. Building fills
in the Lean excerpts, while the "Full source" panels fetch their files, so they work on the published site.

Build the development from the repository root:

```bash
lake build KimiK3
```

MoonViT's definitions use named-axis `einsum` for patch projection and `Tensor.mapLeading` to
apply normalization across tokens. Their graph implementations use TorchLean's batch-aware
primitives, so you don't have to flatten the frame, row, and head axes into one long matrix. Over
exact reals, the correspondence proofs hold for arbitrary dimensions (native floating-point and
GPU execution are separate checks).

## From the report to Lean

K3's language backbone alternates Kimi Delta Attention (KDA) with periodic NoPE Multi-head Latent
Attention (MLA). I wrote KDA as a causal recurrence over its fixed-size state. MLA stores a
compressed key/value latent and reconstructs head-specific keys and values when it attends, and
then the model composes those sequence mixers with Block Attention Residuals and Stable LatentMoE
([report, Sections 2.1--2.3, pp. 4--9](https://arxiv.org/pdf/2607.24653#page=4)).

On the vision side, the Lean specifies MoonViT-V2's attention over all patches of an item with 2-D
rotary positions, temporal pooling, 2-by-2 spatial merge, and projection to the language width
([Section 2.4, pp. 9--10](https://arxiv.org/pdf/2607.24653#page=9)). Where the report describes
factorized spatial and temporal passes, the released
[`modeling_kimi_k3.py`](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab6/modeling_kimi_k3.py)
has one attention over all `t × h × w` tokens, and the Lean follows the code. For training, the Lean records
Per-Head Muon contracts, the clipped MOPD reward, the reported deployment precision policy, and the
seven-step self-feeding EAGLE-3 draft procedure
([Sections 2.5, 3.3--3.4, and 4.1, pp. 10--14](https://arxiv.org/pdf/2607.24653#page=10)).

| Report component | Report location | Lean coverage |
| --- | --- | --- |
| Published dimensions and 3:1 KDA/MLA schedule | [Figure 2 and Section 2.1, pp. 3--4; Table 1, p. 11](https://arxiv.org/pdf/2607.24653#page=3) | [`KimiK3/Config.lean`](KimiK3/Config.lean): definitions plus proofs of the 69/24 layer counts, cache-width arithmetic, and configuration well-formedness |
| Kimi Delta Attention | [Section 2.1.1, pp. 4--5, Eqs. 1--6](https://arxiv.org/pdf/2607.24653#page=4) | [`KimiK3/Sequence/KDA.lean`](KimiK3/Sequence/KDA.lean) and [`KimiK3/GraphSpec/KDA.lean`](KimiK3/GraphSpec/KDA.lean): recurrence, projections, chunked execution of the token recurrence, and graph-denotation proofs; the chunkwise-parallel form of Eqs. 3--4 is not formalized |
| Gated MLA | [Section 2.1.2, pp. 5--6, Eq. 7](https://arxiv.org/pdf/2607.24653#page=5) | [`KimiK3/Sequence/MLA.lean`](KimiK3/Sequence/MLA.lean) and [`KimiK3/GraphSpec/MLA.lean`](KimiK3/GraphSpec/MLA.lean): compressed cache semantics and fixed-cache graph equivalence |
| Block Attention Residuals | [Section 2.2, p. 6](https://arxiv.org/pdf/2607.24653#page=6) | [`KimiK3/Sequence/AttnRes.lean`](KimiK3/Sequence/AttnRes.lean) and [`KimiK3/GraphSpec/AttnRes.lean`](KimiK3/GraphSpec/AttnRes.lean): block state, retrieval, boundary invariant, and graph semantics |
| SiTU-GLU, Stable LatentMoE, Quantile Balancing | [Section 2.3, pp. 6--9, Eqs. 11--14; Appendices B--D, pp. 43--45](https://arxiv.org/pdf/2607.24653#page=6) | [`KimiK3/FeedForward.lean`](KimiK3/FeedForward.lean), [`KimiK3/FeedForward/Quantile.lean`](KimiK3/FeedForward/Quantile.lean), [`KimiK3/FeedForward/Histogram.lean`](KimiK3/FeedForward/Histogram.lean), and [`KimiK3/GraphSpec/MoE.lean`](KimiK3/GraphSpec/MoE.lean): SiTU bounds, deterministic top-k, exact coordinate minimization, pooled histograms and interpolation error, and route-specialized graph equivalence |
| MoonViT-V2 | [Section 2.4, pp. 9--10](https://arxiv.org/pdf/2607.24653#page=9) | [`KimiK3/Vision.lean`](KimiK3/Vision.lean) and [`KimiK3/GraphSpec/Vision.lean`](KimiK3/GraphSpec/Vision.lean): shape-indexed reference semantics of the released code and primitive-graph denotation theorems for embedding, one encoder block, and merge/project (cut at the exact GELU); no proof of the empirical stability claims |
| Language and multimodal composition | [Figure 2, p. 3, and Sections 2.1--2.4, pp. 4--10](https://arxiv.org/pdf/2607.24653#page=3) | [`KimiK3/Model.lean`](KimiK3/Model.lean) defines the high-level text/visual embedding interface; [`KimiK3/GraphSpec/LanguageModel.lean`](KimiK3/GraphSpec/LanguageModel.lean) proves a packed, route-staged, one-text-token language graph. The packed/list-state bridge and complete multimodal graph composition remain open. |
| Per-Head Muon and pre-training objective | [Section 2.5, p. 10, and Sections 3.3--3.4, pp. 11--12](https://arxiv.org/pdf/2607.24653#page=10) | [`KimiK3/Training/Muon.lean`](KimiK3/Training/Muon.lean) and [`KimiK3/Training/Pretraining.lean`](KimiK3/Training/Pretraining.lean): per-head split/merge, orthogonalization contracts, and the masked next-token objective |
| MOPD and length control | [Sections 4.1.2--4.1.3, pp. 13--14, Eq. 15](https://arxiv.org/pdf/2607.24653#page=13) | [`KimiK3/Training/Distillation.lean`](KimiK3/Training/Distillation.lean) and [`KimiK3/Training/Controls.lean`](KimiK3/Training/Controls.lean): clipped rewards, all nine typed teacher indices, token-budget override, and verbosity control; generated trajectories and learned policies remain empirical inputs |
| QAT and EAGLE-3 draft training | [Section 4.1.4, p. 14, Eq. 16 and the surrounding procedure](https://arxiv.org/pdf/2607.24653#page=14) | [`KimiK3/Microscaling.lean`](KimiK3/Microscaling.lean) and [`KimiK3/Training/Draft.lean`](KimiK3/Training/Draft.lean): MX format semantics and error bounds, feature fusion, self-feeding draft definitions, acceptance identities, and seven-step loss |
| KDA Context Parallelism | [Section 5.1.2](https://arxiv.org/html/2607.24653v2#S5.SS1.SSS2) | [`KimiK3/Parallel/Context.lean`](KimiK3/Parallel/Context.lean): affine segment summaries, associative composition, prefix execution, and append laws; no distributed collective or kernel implementation |
| MoonEP counting bounds | [Section 5.2 and Appendix E](https://arxiv.org/html/2607.24653v2#S5.SS2) | [`KimiK3/Parallel/Experts.lean`](KimiK3/Parallel/Experts.lean): the key lemma by the report's filling construction, the redundant-expert upper bound for every router output, and the capacity lower bound; no online planner, migration runtime, or communication proof |
| XTML chat template | [Appendix F](https://arxiv.org/html/2607.24653v2#A6) | [`KimiK3/ChatTemplate.lean`](KimiK3/ChatTemplate.lean): typed context zones, generation modes, history channels, and indexed parallel tool call/result matching; tokenizer IDs and natural-language option rendering are outside the schema proof |

## Main proofs

I picked public theorems that check behavior rather than repeat record fields.

* `KDA.run_append` and `GatedMLA.run_append` show that chunked causal execution returns the same
  state and outputs as one uninterrupted pass. Both are instances of `CausalScan.run_append`,
  which holds for any step function, so what they check is that each layer is written as a causal
  scan. They formalize a semantic condition behind the
  recurrent attention and cache-management discussion
  ([Sections 2.1 and 5.4.1, pp. 4--6 and 22--23](https://arxiv.org/pdf/2607.24653#page=4)).
* `KDA.paper_retention_bounds` proves that every K3 retention factor lies strictly between
  `exp (-5)` and `1`, for every real gate input
  ([Section 2.1.1, p. 5, Eq. 5](https://arxiv.org/pdf/2607.24653#page=5)).
* `BlockState.finishLayer_wf` proves that an AttnRes state cannot advance past its block boundary
  ([Section 2.2, p. 6](https://arxiv.org/pdf/2607.24653#page=6)).
* `paperConfig_mla_cache_compression` checks the cache-width reduction given by the released
  configuration: 576 stored scalars per token instead of 30,720, a ratio of `160 / 3`
  ([Section 2.1.2, pp. 5--6](https://arxiv.org/pdf/2607.24653#page=5), the released configuration,
  and the checked Lean arithmetic).
* `SiTU.paper_caps_bound` proves the SiTU-GLU coordinate bound of 100 for all real preactivations
  ([Appendix B, p. 43](https://arxiv.org/pdf/2607.24653#page=43)).
* `routedTokenCount_neg_eq` is the bookkeeping step of the Quantile Balancing update: when the
  selected threshold already has the exact strict-exceedance count, using its negation as the bias
  gives the expert exactly that load (a single `linarith` step). Ties at the threshold are left out
  of that count by definition
  ([Section 2.3.3, pp. 8--9, and Appendices C--D, pp. 43--45](https://arxiv.org/pdf/2607.24653#page=8)).
* `coordinateObjective_minimum_iff` characterizes every coordinate minimizer: the real target
  lies between the counts strictly above and at or above the threshold. This includes fractional
  targets, ties, and the upper endpoint missed by the strict-count clause in Appendix C (the
  interval stated just after that clause is correct). The exact-count theorem follows as a special
  case, though neither theorem constructs a threshold. `cumulativeThrough_sum` proves additive
  histogram pooling, and `abs_estimate_sub_le_width` is the elementary part of Appendix D's error
  bound: the interpolated estimate stays in the selected bin, so a true quantile known to lie in
  that bin is within one bin width of it.
* `ContextParallel.composeAll_apply` proves that composing supplied affine summaries agrees with
  applying those same summaries sequentially. I haven't yet proved a bridge from actual KDA updates
  to these summaries.
* `MoonEP.moonEP_upper_bound` proves the report's redundant-expert upper bound for every router
  output when each rank hosts `E / R` experts. `exists_fill` builds the key lemma's plan, so the
  one-source property is proved, not assumed
  ([Appendix E, p. 45](https://arxiv.org/pdf/2607.24653#page=45)).
* `FeatureFusion.initial_fusion_eq_high` proves that the report's `[0 0 I]` initialization sends the
  high-level target feature to the pretrained MTP layer unchanged
  ([Section 4.1.4, p. 14](https://arxiv.org/pdf/2607.24653#page=14)).
* `acceptanceRate_eq_one_sub_totalVariation` identifies lossless speculative acceptance with one
  minus the total-variation distance between the target and draft distributions, refining the
  acceptance expression used by Eq. 16
  ([Section 4.1.4, p. 14](https://arxiv.org/pdf/2607.24653#page=14)).
* `Draft.Model.trainingTimeTestStep` makes the EAGLE-3 recurrence self-feeding by construction.
  `trainingTimeTest_length` and `trainingTimeTest_append` prove its output length and chunking law.
  `sevenStepDistributions` fixes the reported unroll depth in the type
  ([Section 4.1.4, p. 14](https://arxiv.org/pdf/2607.24653#page=14)).

## Running the same sequence in two ways

You can evaluate the sequence layer by layer, as `LanguageModel.logits` does, or keep each
layer's causal state and process one token at a time. Do the two agree?
`LanguageModel.run_eq_logits` proves that they give the same vocabulary logits for every
sequence length and every model value. It uses the existing `BackboneLayer.forwardToken`, with
both AttnRes retrievals and the model's own sparse router, so there's no second attention
implementation hiding in the proof.

The streaming interface takes an embedding tensor and returns a logit tensor, plus the state for
the next chunk:

```lean
import KimiK3

#check KimiK3.LanguageModel.initialState
#check KimiK3.LanguageModel.step
#check KimiK3.LanguageModel.run
#check KimiK3.LanguageModel.run_eq_logits
#check KimiK3.LanguageModel.run_append
```

`run_append` also proves that splitting an embedding tensor into two chunks preserves every
logit and the final decoder state, provided the second chunk uses the state returned by the first.

All of this lives in [`Model/Streaming.lean`](KimiK3/Model/Streaming.lean): real-valued
specifications and proofs, not a runnable checkpoint loader. By contrast, the graph
modules work on TorchLean's typed DAG, which records tensor operations and their shapes (`GraphSpec` is
the library's name for that interface, not another autodiff system), and they keep packed caches
where this file keeps the public causal state.

## Code layout

I kept the equations close to the proofs that use them. `Sequence/` separates KDA, MLA, and
AttnRes, and `FeedForward/` separates the SiTU activation, experts, routing, and MoE composition
(the quantile and histogram proofs live there too, next to the router they describe). `Parallel/`
contains context scans and expert-placement bounds, while `Training/` holds the objectives,
updates, draft procedure, and post-training controls.
On the graph side, `GraphSpec/Decoder/` builds the decoder components and `GraphSpec/Backbone/`
separates shared depth state from the four KDA/MLA and dense/sparse layer variants. Then
`GraphSpec/LanguageModel.lean` adds embeddings and vocabulary logits, and umbrella imports bring
all these modules back together.

KDA's types are `KDA.Input`, `KDA.Head`, and `KDA.Layer`. MLA stores its per-head projections
directly as tensors with a leading head dimension, without a separate projection record or
packing functions. Model weights, persistent state, and encoding evidence still get records,
because their fields have different shapes or carry assumptions that the proofs use.

For number formats I reuse FloatLib's OCP decoders for E2M1, E4M3, E5M2, and the E8M0 scale
exponent. K3's adapter restricts these to finite codes and converts their exact rational values
to reals for the model proofs. It keeps the real-valued quantizer and block error bounds, which
reason about arbitrary real inputs and the supplied scale instead of assuming a particular
production encoder.

## Proof boundary

This isn't a formalization of all of Kimi K3. The development covers reference equations, typed
composition, selected graph denotations, and the named invariants above. It doesn't contain the
released parameter tensors, and it doesn't prove that the 2.78-trillion-parameter checkpoint
implements these definitions, replay pre-training or post-training, or verify the native kernels.
And the report's data quality, scaling-law, benchmark, distributed-training, RL-environment,
kernel-latency, and fleet-scheduling results are empirical or systems claims rather than
consequences of the equations, so I haven't restated them as Lean theorems.

The composed GraphSpec result is for one autoregressive token step with fixed-shape tensor caches,
and it proves equality to the packed decoder semantics used to assemble that graph. What I haven't
proved yet is that packing the public list-based KDA/MLA histories and running this graph is equal
to `BackboneLayer.forwardToken` or the full-sequence `LanguageModel.logits` path. You can lower the
graph to a TorchLean `Program`, but equivalence to a fused finite-precision kernel would be another
refinement on top of that.

For Per-Head Muon, the proofs cover matrix split/merge and orthogonalization contracts, not the
full distributed optimizer. Post-training is similar. Those modules formalize deterministic reward,
selection, budget, and chat-schema rules, not teacher quality, environment behavior, or
optimization success. MoonViT-V2 is specified and its component graphs are checked, but its
reported gradient stability isn't proved. MXFP4/MXFP8 blocks and error bounds are formalized too,
although the report doesn't fix every MXFP8 encoding and scaling choice, so those choices are
stated as assumptions. And the released Hugging Face checkpoint leaves out the separately trained
MTP draft layer, while `paperConfig` keeps the one-layer MTP design described in the technical
report.

So far the masked next-token gradient is recorded as a formula, with no derivative theorem
connecting it to the loss. Similarly, the histogram error result assumes the true quantile belongs to the chosen
bin, and I haven't proved that the bin-selection procedure establishes that assumption. Both are
open proof obligations in this development, not errors I found in the report.

Target loads can be real numbers in both the QB coordinate objective and the histogram estimator.
For fractional `q = mk/n`, histogram selection uses `ceil q` while interpolation uses `q` itself,
as in Appendix D, and `fraction_mem_of_isSelectedBin` proves that this gives a positive
interpolation fraction at most one.

## Report edge cases

Right after Eq. 24, Appendix C says the coordinate objective is "minimized exactly when k margins
lie above α, i.e., for any α* between the k-th and (k+1)-th largest entries". That first clause
misses a boundary case, even with distinct margins. Take margins `1, 0`
and load `1`, where the objective is `a + max(0, 1-a) + max(0, -a)`. Every `a` in `[0, 1]` minimizes
it, but at `a = 1` the strict exceedance count is zero. Meanwhile the interval in the second clause,
with both ends included, is exactly `[0, 1]`. I checked both facts in Lean against the definitions
above (`endpointMargins_minimizer_iff` and `not_forall_minimizer_iff_isExactTargetQuantile`). So
the report's chosen lower endpoint is still a minimizer, and the slip is in one clause of the
characterization of all minimizers, not in the update rule.

The exact condition is `exceedanceCount ≤ targetLoad ≤ atLeastCount`, where the second count includes
values equal to the threshold. `coordinateObjective_minimum_iff` proves both directions for any
finite batch and real target, and it covers tied margins without pretending that a strict
threshold can split a tie.

Appendix C sets score ties aside when deriving the strict-threshold assignment rule, so the example
here doesn't refute its argument. But what happens when two scores tie? With two equal scores and
a requested load of one, a strict threshold selects either both or neither. Yet the tied cutoff can
still minimize the coordinate objective. So coordinate minimization and an exact strict-exceedance
count aren't equivalent for every batch, and tie-breaking needs a separate rule. I checked this
example in Lean too (`exceedanceCount_tiedMargins_ne_one` and `tiedMargins_zero_minimizer`).

Equation 16 also needs a boundary convention when the target and draft distributions have zero
overlap, because the negative-log acceptance loss is then infinite. Our `likelihoodLoss` uses `EReal`
to keep that case instead of silently falling back on Lean's real-valued `log 0` convention. None of
these observations shows a bug in the released model.

## Sources

* Kimi Team, [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653), 2026.
* Moonshot AI, [released Kimi K3 configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json).
* Yuhui Li et al., [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via
  Training-Time Test](https://arxiv.org/abs/2503.01840), 2025.
