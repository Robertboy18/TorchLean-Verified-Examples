# Kimi K3 in Lean

The Kimi K3 technical report combines several recent ideas in one model: recurrent Kimi Delta
Attention, periodic latent attention, block-level residual connections, a routed mixture of
experts, a vision encoder, Muon updates, reinforcement learning, quantization-aware training, and
speculative decoding. Reading those components only as implementation prose makes it difficult to
see which equations compose and which properties survive when the model is evaluated in chunks.

This development writes the architecture as parameterized Lean definitions. The published
dimensions instantiate those definitions at 2.78 trillion total parameters, while small dimensions
let us evaluate the same equations and prove general results without constructing the released
checkpoint. The code specifies the algorithms in the report; it does not load or run Kimi's trained
weights.

Build the development from the repository root:

```bash
lake build KimiK3
```

## From the report to Lean

The language backbone alternates Kimi Delta Attention (KDA) with periodic NoPE Multi-head Latent
Attention (MLA). KDA is written as a causal recurrence over its fixed-size state. MLA stores a
compressed key/value latent and reconstructs head-specific keys and values when it attends. The
model then composes those sequence mixers with Block Attention Residuals and Stable LatentMoE.

The vision path specifies MoonViT-V2's spatial and temporal attention, temporal pooling, 2-by-2
spatial merge, and projection to the language width. The training file covers the progressive
context schedule, Per-Head Muon contracts, clipped MOPD rewards, the reported deployment precision
policy, and the seven-step self-feeding EAGLE-3 draft procedure.

| Report component | Lean module |
| --- | --- |
| Published dimensions and 3:1 KDA/MLA schedule | [`KimiK3/Config.lean`](KimiK3/Config.lean) |
| SiTU-GLU, Stable LatentMoE, Quantile Balancing | [`KimiK3/FeedForward.lean`](KimiK3/FeedForward.lean) |
| KDA, Gated MLA, Block AttnRes | [`KimiK3/Sequence.lean`](KimiK3/Sequence.lean) |
| MoonViT-V2 | [`KimiK3/Vision.lean`](KimiK3/Vision.lean) |
| Language-backbone composition | [`KimiK3/Model.lean`](KimiK3/Model.lean) |
| Muon, MOPD, QAT policy, EAGLE-3 | [`KimiK3/Training.lean`](KimiK3/Training.lean) |

## Main proofs

The public theorems are chosen to check behavior rather than repeat record fields.

* `KDA.run_append` and `GatedMLA.run_append` show that chunked causal execution returns the same
  state and outputs as one uninterrupted pass. This is the property needed for prefix reuse and
  replay after speculative decoding.
* `KDA.paper_retention_bounds` proves that every K3 retention factor lies strictly between
  `exp (-5)` and `1`, for every real gate input.
* `BlockState.finishLayer_wf` proves that an AttnRes state cannot advance past its block boundary.
* `paperConfig_mla_cache_compression` checks the exact published cache-width reduction: 576 stored
  scalars per token instead of 30,720, a ratio of `160 / 3`.
* `SiTU.paper_caps_bound` proves the SiTU-GLU coordinate bound of 100 for all real preactivations.
* `neg_quantile_bias_hits_target` proves that the Quantile Balancing update gives an expert its
  requested load when the selected threshold is an exact no-ties quantile.
* `FeatureFusion.initial_fusion_eq_high` proves that the report's `[0 0 I]` initialization sends the
  high-level target feature to the pretrained MTP layer unchanged.
* `acceptanceRate_eq_one_sub_totalVariation` identifies lossless speculative acceptance with one
  minus the total-variation distance between the target and draft distributions.
* `unroll_isSelfFeeding` proves that every EAGLE-3 draft step after the first consumes a token
  selected from the preceding draft distribution, as required by the report's seven-step training
  procedure.

## Proof boundary

The report also contains measurements about data quality, benchmark scores, distributed training,
kernel latency, and fleet scheduling. Those observations are inputs to the report, not consequences
of the architectural equations, so this development does not restate them as Lean theorems.

The MXFP4/MXFP8 declaration records the reported assignment of formats to routed-expert weights and
activations; bit-level MX block scaling remains outside the present model. The released Hugging Face
checkpoint also omits the separately trained MTP draft layer, while `paperConfig` retains the
one-layer MTP design described in the technical report.

## Sources

* Kimi Team, [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653), 2026.
* Moonshot AI, [released Kimi K3 configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json).
* Yuhui Li et al., [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via
  Training-Time Test](https://arxiv.org/abs/2503.01840), 2025.
