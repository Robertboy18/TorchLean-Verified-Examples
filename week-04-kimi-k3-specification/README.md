# Kimi K3 in Lean

The Kimi K3 technical report combines several recent ideas in one model: recurrent Kimi Delta
Attention, periodic latent attention, block-level residual connections, a routed mixture of
experts, a vision encoder, Muon updates, reinforcement learning, quantization-aware training, and
speculative decoding. Reading those components only as implementation prose makes it difficult to
see which equations compose and which properties survive when the model is evaluated in chunks.

This development writes K3's architecture equations as shape-indexed tensor functions. The core
language operations are also built from primitive TorchLean graph nodes, and Lean proves that the
packed one-token graph returns the same updated state and vocabulary logits as those packed
functions. The published dimensions are one valid instance; small instances let us evaluate the
same equations without constructing the 2.78-trillion-parameter checkpoint. MoonViT embedding,
divided-attention blocks, and merge/project also have primitive-graph denotation theorems. The
public list-state bridge, trained weights, kernels, training runs, distributed systems, and
benchmark measurements remain outside the proof.

The [annotated formalization](site/index.html) rebuilds the relevant parts of the report as a
web-native article. Paper equations, diagrams, and interactive explanations sit beside the
corresponding Lean definitions, proof assumptions, and explicit boundaries. To preview it locally, run
`python3 -m http.server 8000` at the repository root and open
`http://localhost:8000/week-04-kimi-k3-specification/site/`.

Build the development from the repository root:

```bash
lake build KimiK3
```

The MoonViT definitions use named-axis `einsum` for patch projection and the residual MLP,
`rearrange` to exchange frame and patch axes, and `Tensor.mapLeading` to apply normalization
and attention across batches. Their graph implementations use TorchLean's batch-aware primitives,
so these axes need not be flattened into one long matrix. The correspondence proofs use arbitrary
dimensions over exact reals; native floating-point and GPU execution are separate checks.

## From the report to Lean

The language backbone alternates Kimi Delta Attention (KDA) with periodic NoPE Multi-head Latent
Attention (MLA). KDA is written as a causal recurrence over its fixed-size state. MLA stores a
compressed key/value latent and reconstructs head-specific keys and values when it attends. The
model then composes those sequence mixers with Block Attention Residuals and Stable LatentMoE
([report, Sections 2.1--2.3, pp. 4--8](https://arxiv.org/pdf/2607.24653#page=4)).

The vision path specifies MoonViT-V2's spatial and temporal attention, temporal pooling, 2-by-2
spatial merge, and projection to the language width
([Section 2.4, pp. 9--10](https://arxiv.org/pdf/2607.24653#page=9)). The training file records
Per-Head Muon contracts, the clipped MOPD reward, the reported deployment precision policy, and the
seven-step self-feeding EAGLE-3 draft procedure
([Sections 2.5, 3.3--3.4, and 4.1, pp. 10--14](https://arxiv.org/pdf/2607.24653#page=10)).

| Report component | Report location | Lean coverage |
| --- | --- | --- |
| Published dimensions and 3:1 KDA/MLA schedule | [Figure 2 and Section 2.1, pp. 3--4; Table 1, p. 11](https://arxiv.org/pdf/2607.24653#page=3) | [`KimiK3/Config.lean`](KimiK3/Config.lean): definitions plus proofs of the 69/24 layer counts, cache-width arithmetic, and configuration well-formedness |
| Kimi Delta Attention | [Section 2.1.1, pp. 4--5, Eqs. 1--6](https://arxiv.org/pdf/2607.24653#page=4) | [`KimiK3/Sequence.lean`](KimiK3/Sequence.lean) and [`KimiK3/GraphSpec/KDA.lean`](KimiK3/GraphSpec/KDA.lean): recurrence, projections, chunking, and graph-denotation proofs |
| Gated MLA | [Section 2.1.2, pp. 5--6, Eq. 7](https://arxiv.org/pdf/2607.24653#page=5) | [`KimiK3/Sequence.lean`](KimiK3/Sequence.lean) and [`KimiK3/GraphSpec/MLA.lean`](KimiK3/GraphSpec/MLA.lean): compressed cache semantics and fixed-cache graph equivalence |
| Block Attention Residuals | [Section 2.2, p. 6](https://arxiv.org/pdf/2607.24653#page=6) | [`KimiK3/Sequence.lean`](KimiK3/Sequence.lean) and [`KimiK3/GraphSpec/AttnRes.lean`](KimiK3/GraphSpec/AttnRes.lean): block state, retrieval, boundary invariant, and graph semantics |
| SiTU-GLU, Stable LatentMoE, Quantile Balancing | [Section 2.3, pp. 6--8, Eqs. 11--14; Appendices B--D, pp. 43--44](https://arxiv.org/pdf/2607.24653#page=6) | [`KimiK3/FeedForward.lean`](KimiK3/FeedForward.lean), [`KimiK3/QuantileBalancing.lean`](KimiK3/QuantileBalancing.lean), [`KimiK3/QuantileHistogram.lean`](KimiK3/QuantileHistogram.lean), and [`KimiK3/GraphSpec/MoE.lean`](KimiK3/GraphSpec/MoE.lean): SiTU bounds, deterministic top-k, exact coordinate minimization, pooled histograms and interpolation error, and route-specialized graph equivalence |
| MoonViT-V2 | [Section 2.4, pp. 9--10](https://arxiv.org/pdf/2607.24653#page=9) | [`KimiK3/Vision.lean`](KimiK3/Vision.lean) and [`KimiK3/GraphSpec/Vision.lean`](KimiK3/GraphSpec/Vision.lean): shape-indexed reference semantics and primitive-graph denotation theorems for embedding, one divided-attention block, and merge/project; no proof of the empirical stability claims |
| Language and multimodal composition | [Figure 2, p. 3, and Sections 2.1--2.4, pp. 4--10](https://arxiv.org/pdf/2607.24653#page=3) | [`KimiK3/Model.lean`](KimiK3/Model.lean) defines the high-level text/visual embedding interface; [`KimiK3/GraphSpec/LanguageModel.lean`](KimiK3/GraphSpec/LanguageModel.lean) proves a packed, route-staged, one-text-token language graph. The packed/list-state bridge and complete multimodal graph composition remain open. |
| Per-Head Muon and pre-training objective | [Section 2.5, p. 10, and Sections 3.3--3.4, pp. 11--12](https://arxiv.org/pdf/2607.24653#page=10) | [`KimiK3/Training.lean`](KimiK3/Training.lean): per-head split/merge and orthogonalization contracts plus the masked next-token objective |
| MOPD and length control | [Section 4.1.3, pp. 13--14, Eq. 15](https://arxiv.org/pdf/2607.24653#page=13) | [`KimiK3/Training.lean`](KimiK3/Training.lean) and [`KimiK3/TrainingProtocols.lean`](KimiK3/TrainingProtocols.lean): clipped rewards, all nine typed teacher indices, token-budget override, and verbosity control; generated trajectories and learned policies remain empirical inputs |
| QAT and EAGLE-3 draft training | [Section 4.1.4, p. 14, Eqs. 16 and the surrounding procedure](https://arxiv.org/pdf/2607.24653#page=14) | [`KimiK3/Microscaling.lean`](KimiK3/Microscaling.lean) and [`KimiK3/Training.lean`](KimiK3/Training.lean): MX format semantics and error bounds, feature fusion, self-feeding draft definitions, acceptance identities, and seven-step loss |
| KDA Context Parallelism | [Section 5.1.2](https://arxiv.org/html/2607.24653v2#S5.SS1.SSS2) | [`KimiK3/ContextParallel.lean`](KimiK3/ContextParallel.lean): affine segment summaries, associative composition, prefix execution, and append laws; no distributed collective or kernel implementation |
| MoonEP counting bounds | [Section 5.2 and Appendix E](https://arxiv.org/html/2607.24653v2#S5.SS2) | [`KimiK3/MoonEP.lean`](KimiK3/MoonEP.lean): explicit hypotheses for the redundant-expert upper bound and the capacity lower bound; no online planner, migration runtime, or communication proof |
| XTML chat template | [Appendix F](https://arxiv.org/html/2607.24653v2#A6) | [`KimiK3/ChatTemplate.lean`](KimiK3/ChatTemplate.lean): typed context zones, generation modes, history channels, and indexed parallel tool call/result matching; tokenizer IDs and natural-language option rendering are outside the schema proof |

## Main proofs

The public theorems are chosen to check behavior rather than repeat record fields.

* `KDA.run_append` and `GatedMLA.run_append` show that chunked causal execution returns the same
  state and outputs as one uninterrupted pass. They formalize a semantic condition behind the
  recurrent attention and cache-management discussion
  ([Sections 2.1 and 5.4.1, pp. 4--6 and 22--23](https://arxiv.org/pdf/2607.24653#page=4)).
* `KDA.paper_retention_bounds` proves that every K3 retention factor lies strictly between
  `exp (-5)` and `1`, for every real gate input
  ([Section 2.1.1, p. 5, Eq. 5](https://arxiv.org/pdf/2607.24653#page=5)).
* `BlockState.finishLayer_wf` proves that an AttnRes state cannot advance past its block boundary
  ([Section 2.2, p. 6](https://arxiv.org/pdf/2607.24653#page=6)).
* `paperConfig_mla_cache_compression` checks the exact published cache-width reduction: 576 stored
  scalars per token instead of 30,720, a ratio of `160 / 3`
  ([Section 2.1.2, pp. 5--6](https://arxiv.org/pdf/2607.24653#page=5), the released configuration,
  and the checked Lean arithmetic).
* `SiTU.paper_caps_bound` proves the SiTU-GLU coordinate bound of 100 for all real preactivations
  ([Appendix B, p. 43](https://arxiv.org/pdf/2607.24653#page=43)).
* `neg_quantile_bias_hits_target` proves that the Quantile Balancing update gives an expert its
  requested load when the selected threshold already has the exact strict-exceedance count; ties
  at the threshold are excluded from that count by definition
  ([Section 2.3.3, p. 8, and Appendices C--D, pp. 43--44](https://arxiv.org/pdf/2607.24653#page=8)).
* `coordinateObjective_minimized_at_exactQuantile` proves coordinate minimization when the
  threshold has exactly the requested number of strict exceedances. Ties are allowed only when
  that assumption holds; the theorem does not construct a suitable threshold for every tied
  batch. `cumulativeThrough_sum` proves additive histogram pooling. `abs_estimate_sub_le_width`
  bounds interpolation error when the true quantile is already known to lie in the selected bin.
* `ContextParallel.composeAll_apply` proves that composing supplied affine summaries agrees with
  applying those same summaries sequentially. A bridge from actual KDA updates to these summaries
  has not yet been proved.
* `MoonEP.peakRedundant_le_of_one_source` proves the report's redundant-expert upper bound after
  making its per-rank capacity and one-source hypotheses explicit.
* `FeatureFusion.initial_fusion_eq_high` proves that the report's `[0 0 I]` initialization sends the
  high-level target feature to the pretrained MTP layer unchanged
  ([Section 4.1.4, p. 14](https://arxiv.org/pdf/2607.24653#page=14)).
* `acceptanceRate_eq_one_sub_totalVariation` identifies lossless speculative acceptance with one
  minus the total-variation distance between the target and draft distributions, refining the
  acceptance expression used by Eq. 16
  ([Section 4.1.4, p. 14](https://arxiv.org/pdf/2607.24653#page=14)).
* `Draft.Model.trainingTimeTestStep` makes the EAGLE-3 recurrence self-feeding by construction;
  `trainingTimeTest_length` and `trainingTimeTest_append` prove its output length and chunking law.
  `sevenStepDistributions` fixes the reported unroll depth in the type
  ([Section 4.1.4, p. 14](https://arxiv.org/pdf/2607.24653#page=14)).

## Proof boundary

This is not a formalization of all of Kimi K3. The development covers reference equations, typed
composition, selected graph denotations, and the named invariants above. It does not contain the
released parameter tensors, prove that the 2.78-trillion-parameter checkpoint implements these
definitions, replay pre-training or post-training, or verify the native kernels. The report's data
quality, scaling-law, benchmark, distributed-training, RL-environment, kernel-latency, and fleet-
scheduling results are empirical or systems claims rather than consequences of the equations, so
they are not restated as Lean theorems.

The composed GraphSpec result is for one autoregressive token step with fixed-shape tensor caches.
It proves equality to the packed decoder semantics used to assemble that graph. The development does
not yet prove that packing the public list-based KDA/MLA histories and running this graph is equal to
`BackboneLayer.forwardToken` or the full-sequence `LanguageModel.logits` path. The graph can lower to
a TorchLean `Program`; equivalence to a fused finite-precision kernel remains another refinement.

The Per-Head Muon section proves matrix split/merge and orthogonalization contracts, not the full
distributed optimizer. The post-training modules formalize deterministic reward, selection,
budget, and chat-schema rules, not teacher quality, environment behavior, or optimization success.
MoonViT-V2 is specified and its component graphs are checked; the reported gradient stability is
not proved. MXFP4/MXFP8 blocks and error bounds are formalized, but the report does not fix every
MXFP8 encoding and scaling choice, so those choices remain explicit assumptions. The released
Hugging Face checkpoint also omits the separately trained MTP draft layer, while `paperConfig`
retains the one-layer MTP design described in the technical report.

The masked next-token gradient is recorded as a formula, without a derivative theorem connecting
it to the loss. The histogram error result assumes the true quantile belongs to the chosen bin;
the bin-selection procedure has not been proved to establish that assumption. These are remaining
proof obligations in this development, not errors demonstrated in the report.

The QB coordinate objective and histogram estimator currently take natural-number target loads.
Appendix D also describes fractional `q = mk/n`: it selects a bin using `ceil q` and interpolates
with `q` itself. Our estimator covers the integral case, not that fractional interpolation rule.

## Report edge cases

Appendix C sets score ties aside when deriving the strict-threshold assignment rule. With two
equal scores and a requested load of one, a strict threshold selects either both or neither.
The tied cutoff can still minimize the coordinate objective. Thus coordinate minimization and
an exact strict-exceedance count are not equivalent for every batch; tie-breaking needs a separate
rule. We checked this example in Lean. It does not refute the report's stated no-tie argument.

Equation 16 also needs a boundary convention when target and draft distributions have zero
overlap: the negative-log acceptance loss is infinite. Our `likelihoodLoss` uses `EReal` to retain
that case instead of silently using Lean's real-valued `log 0` convention. Neither observation
establishes a bug in the released model.

## Sources

* Kimi Team, [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653), 2026.
* Moonshot AI, [released Kimi K3 configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json).
* Yuhui Li et al., [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via
  Training-Time Test](https://arxiv.org/abs/2503.01840), 2025.
