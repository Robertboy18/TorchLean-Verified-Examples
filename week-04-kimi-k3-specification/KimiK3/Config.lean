/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Json
public import NN.Spec.Core.Shape

/-!
# Kimi K3 architecture configuration

This file records the dimensions and layer schedule of Kimi K3 independently of any runtime
backend.  The configuration follows Section 2 and Table 1 of the Kimi K3 technical report, with
the more precise vocabulary and projection dimensions taken from the released model config.

There are two closely related public configurations:

* `paperConfig` contains the one multi-token-prediction layer described in the report;
* `releasedCheckpointConfig` records that the released Hugging Face checkpoint omits that draft
  layer (`num_nextn_predict_layers = 0`).

References:

* Kimi Team, "Kimi K3: Open Frontier Intelligence", 2026, Section 2 and Table 1:
  https://arxiv.org/abs/2607.24653
* Released configuration:
  https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json
-/

@[expose] public section

namespace KimiK3

open Lean

/-- The two sequence-mixing mechanisms used by the language backbone. -/
inductive AttentionKind where
  /-- Kimi Delta Attention, the recurrent linear-attention layer. -/
  | kda
  /-- Gated Multi-head Latent Attention, the periodic global-attention layer. -/
  | mla
  deriving Repr, DecidableEq

/-- Dimensions of the Kimi K3 language backbone.

The record is deliberately parameterized.  Small models can preserve the K3 dataflow while using
fewer layers or experts; the paper-sized model is just one value of this type.
-/
structure TextConfig where
  /-- Width of every language-model token representation. -/
  hiddenDim : Nat
  /-- Number of backbone layers, excluding the token embedding and MTP draft layers. -/
  numLayers : Nat
  /-- Number of query/output heads in KDA and MLA. -/
  numHeads : Nat
  /-- Width of one KDA query/key head. -/
  kdaHeadDim : Nat
  /-- Width of one KDA value head. -/
  kdaValueDim : Nat
  /-- Kernel width of the causal short convolutions in KDA. -/
  shortConvWidth : Nat
  /-- Lower endpoint of the KDA log-retention interval.  Kimi K3 uses `-5`. -/
  logDecayFloor : Int
  /-- Low-rank query projection width in MLA. -/
  queryLatentDim : Nat
  /-- Shared compressed key/value width in MLA. -/
  kvLatentDim : Nat
  /-- Non-positional query/key width per MLA head. -/
  qkNopeHeadDim : Nat
  /-- Shared query/key content width in MLA.

  The released implementation calls this the RoPE head dimension because it inherits MLA's
  projection layout.  Kimi K3 applies no positional rotation: these coordinates remain ordinary
  content features, with one key shared by all heads.
  -/
  qkReservedHeadDim : Nat
  /-- Value width per MLA head. -/
  valueHeadDim : Nat
  /-- Width of the compact routed-expert representation. -/
  routedLatentDim : Nat
  /-- Hidden width inside each routed expert. -/
  routedExpertHiddenDim : Nat
  /-- Hidden width of the initial dense feed-forward layer. -/
  denseHiddenDim : Nat
  /-- Number of routed experts in each sparse layer. -/
  numRoutedExperts : Nat
  /-- Number of routed experts selected for each token. -/
  activeExperts : Nat
  /-- Number of full-width shared experts in each sparse layer. -/
  numSharedExperts : Nat
  /-- Number of initial layers using a dense feed-forward network instead of LatentMoE. -/
  firstDenseLayers : Nat
  /-- Number of ordinary layers accumulated into one Attention Residual block. -/
  attnResBlockSize : Nat
  /-- Vocabulary size of the released tokenizer/model head. -/
  vocabSize : Nat
  /-- Maximum supported context length. -/
  maxContext : Nat
  /-- Number of multi-token-prediction draft layers described by the architecture. -/
  numMtpLayers : Nat
  /-- Gate-branch soft cap in SiTU-GLU. -/
  situGateCap : Nat
  /-- Up-branch soft cap in SiTU-GLU. -/
  situUpCap : Nat
  deriving Repr, DecidableEq

/-- Dimensions of MoonViT-V2 and its projection into the language backbone. -/
structure VisionConfig where
  /-- Square spatial patch size. -/
  patchSize : Nat
  /-- Width of MoonViT-V2 tokens. -/
  hiddenDim : Nat
  /-- Joint query/key/value projection width in each MoonViT-V2 attention block. -/
  qkvHiddenDim : Nat
  /-- Hidden width of the MoonViT feed-forward network. -/
  intermediateDim : Nat
  /-- Number of MoonViT-V2 blocks. -/
  numLayers : Nat
  /-- Number of spatial/temporal attention heads. -/
  numHeads : Nat
  /-- Height factor used by the final pixel-shuffle token merger. -/
  mergeHeight : Nat
  /-- Width factor used by the final pixel-shuffle token merger. -/
  mergeWidth : Nat
  /-- Language-model width produced by the multimodal projector. -/
  textHiddenDim : Nat
  deriving Repr, DecidableEq

/-- Complete architecture configuration. -/
structure Config where
  text : TextConfig
  vision : VisionConfig
  deriving Repr, DecidableEq

/-- Arithmetic conditions needed by the shape-indexed specifications. -/
structure Config.WF (cfg : Config) : Prop where
  hiddenDim_pos : 0 < cfg.text.hiddenDim
  numLayers_pos : 0 < cfg.text.numLayers
  numHeads_pos : 0 < cfg.text.numHeads
  kdaHeadDim_pos : 0 < cfg.text.kdaHeadDim
  kdaValueDim_pos : 0 < cfg.text.kdaValueDim
  shortConvWidth_pos : 0 < cfg.text.shortConvWidth
  queryLatentDim_pos : 0 < cfg.text.queryLatentDim
  kvLatentDim_pos : 0 < cfg.text.kvLatentDim
  qkNopeHeadDim_pos : 0 < cfg.text.qkNopeHeadDim
  valueHeadDim_pos : 0 < cfg.text.valueHeadDim
  routedLatentDim_pos : 0 < cfg.text.routedLatentDim
  routedExpertHiddenDim_pos : 0 < cfg.text.routedExpertHiddenDim
  routedLatentDim_mx_aligned : 32 ∣ cfg.text.routedLatentDim
  routedExpertHiddenDim_mx_aligned : 32 ∣ cfg.text.routedExpertHiddenDim
  denseHiddenDim_pos : 0 < cfg.text.denseHiddenDim
  numRoutedExperts_pos : 0 < cfg.text.numRoutedExperts
  activeExperts_pos : 0 < cfg.text.activeExperts
  activeExperts_le : cfg.text.activeExperts ≤ cfg.text.numRoutedExperts
  numSharedExperts_pos : 0 < cfg.text.numSharedExperts
  firstDenseLayers_le : cfg.text.firstDenseLayers ≤ cfg.text.numLayers
  attnResBlockSize_pos : 0 < cfg.text.attnResBlockSize
  vocabSize_pos : 0 < cfg.text.vocabSize
  maxContext_pos : 0 < cfg.text.maxContext
  situGateCap_pos : 0 < cfg.text.situGateCap
  situUpCap_pos : 0 < cfg.text.situUpCap
  visionPatchSize_pos : 0 < cfg.vision.patchSize
  visionHiddenDim_pos : 0 < cfg.vision.hiddenDim
  visionQkvHiddenDim_pos : 0 < cfg.vision.qkvHiddenDim
  visionIntermediateDim_pos : 0 < cfg.vision.intermediateDim
  visionLayers_pos : 0 < cfg.vision.numLayers
  visionHeads_pos : 0 < cfg.vision.numHeads
  visionHeads_dvd_qkvHidden : cfg.vision.numHeads ∣ cfg.vision.qkvHiddenDim
  mergeHeight_pos : 0 < cfg.vision.mergeHeight
  mergeWidth_pos : 0 < cfg.vision.mergeWidth
  projector_matches_text : cfg.vision.textHiddenDim = cfg.text.hiddenDim

namespace TextConfig

/-- Kimi K3's `3 KDA : 1 MLA` schedule, with a final global-attention layer.

Layers are represented by zero-based `Fin` indices.  The report numbers layers from one, hence the
`layer.val + 1` conversion below.
-/
def attentionKindAt (cfg : TextConfig) (layer : Fin cfg.numLayers) : AttentionKind :=
  let paperIndex := layer.val + 1
  if paperIndex = cfg.numLayers || paperIndex % 4 = 0 then .mla else .kda

/-- Count layers of one attention kind in a finite configuration. -/
def countAttentionKind (cfg : TextConfig) (kind : AttentionKind) : Nat :=
  (List.finRange cfg.numLayers).countP (fun layer => cfg.attentionKindAt layer = kind)

/-- Scalars cached per token by one MLA layer: the compressed key/value latent and the shared
content key. -/
def mlaCompressedScalarsPerToken (cfg : TextConfig) : Nat :=
  cfg.kvLatentDim + cfg.qkReservedHeadDim

/-- Scalars that an unfactorized multi-head cache would retain for one token. -/
def mlaUncompressedScalarsPerToken (cfg : TextConfig) : Nat :=
  cfg.numHeads * (cfg.qkNopeHeadDim + cfg.qkReservedHeadDim + cfg.valueHeadDim)

/-- Number of depth sources retained after grouping decoder layers into AttnRes blocks, including
the token embedding. -/
def attnResSourceCapacity (cfg : TextConfig) : Nat :=
  1 + (cfg.numLayers + cfg.attnResBlockSize - 1) / cfg.attnResBlockSize

/-- Size of the fixed recurrent matrix retained by one KDA layer. Unlike an attention KV cache,
this quantity does not depend on sequence length. -/
def kdaStateScalars (cfg : TextConfig) : Nat :=
  cfg.numHeads * cfg.kdaHeadDim * cfg.kdaValueDim

/-- Number of 32-value OCP blocks on the routed-expert latent axis. -/
def routedLatentBlocks (cfg : TextConfig) : Nat :=
  cfg.routedLatentDim / 32

/-- Number of 32-value OCP blocks on the routed-expert hidden axis. -/
def routedExpertHiddenBlocks (cfg : TextConfig) : Nat :=
  cfg.routedExpertHiddenDim / 32

/-- A well-formed configuration's latent axis is recovered exactly from its MX block count. -/
theorem routedLatentBlocks_mul_blockSize (cfg : Config) (hcfg : cfg.WF) :
    cfg.text.routedLatentBlocks * 32 = cfg.text.routedLatentDim :=
  Nat.div_mul_cancel hcfg.routedLatentDim_mx_aligned

/-- A well-formed configuration's routed hidden axis is recovered exactly from its MX block
count. -/
theorem routedExpertHiddenBlocks_mul_blockSize (cfg : Config) (hcfg : cfg.WF) :
    cfg.text.routedExpertHiddenBlocks * 32 = cfg.text.routedExpertHiddenDim :=
  Nat.div_mul_cancel hcfg.routedExpertHiddenDim_mx_aligned

end TextConfig

/-- The architecture reported for Kimi K3, including its pre-training MTP layer. -/
def paperConfig : Config :=
  { text :=
      { hiddenDim := 7168
        numLayers := 93
        numHeads := 96
        kdaHeadDim := 128
        kdaValueDim := 128
        shortConvWidth := 4
        logDecayFloor := -5
        queryLatentDim := 1536
        kvLatentDim := 512
        qkNopeHeadDim := 128
        qkReservedHeadDim := 64
        valueHeadDim := 128
        routedLatentDim := 3584
        routedExpertHiddenDim := 3072
        denseHiddenDim := 33792
        numRoutedExperts := 896
        activeExperts := 16
        numSharedExperts := 2
        firstDenseLayers := 1
        attnResBlockSize := 12
        vocabSize := 163840
        maxContext := 1048576
        numMtpLayers := 1
        situGateCap := 4
        situUpCap := 25 }
    vision :=
      { patchSize := 14
        hiddenDim := 1024
        qkvHiddenDim := 1536
        intermediateDim := 4096
        numLayers := 27
        numHeads := 12
        mergeHeight := 2
        mergeWidth := 2
        textHiddenDim := 7168 } }

/-- Configuration of the public checkpoint.  It excludes the separately fine-tuned MTP draft. -/
def releasedCheckpointConfig : Config :=
  { paperConfig with text := { paperConfig.text with numMtpLayers := 0 } }

/-! ## Released configuration validation -/

namespace ReleasedCheckpoint

/-- One-based layer numbers assigned to a sequence mixer by the formal K3 schedule. -/
def layerNumbers (cfg : TextConfig) (kind : AttentionKind) : Array Nat :=
  ((List.finRange cfg.numLayers).filter fun layer => cfg.attentionKindAt layer = kind).map
    (fun layer => layer.val + 1) |>.toArray

/-- Require a natural-number field to equal the architecture value used by the specification. -/
def expectNatFieldEq (json : Json) (context field : String) (expected : Nat) :
    Except String Unit := do
  let value ← TorchLean.Json.expectNatE s!"{context}.{field}"
    (← TorchLean.Json.expectFieldE context field json)
  unless value = expected do
    throw s!"{context}.{field}: expected {expected}, found {value}"

/-- Require a string field to equal the expected checkpoint metadata. -/
def expectStringFieldEq (json : Json) (context field expected : String) :
    Except String Unit := do
  let value ← TorchLean.Json.expectStringE s!"{context}.{field}"
    (← TorchLean.Json.expectFieldE context field json)
  unless value = expected do
    throw s!"{context}.{field}: expected {repr expected}, found {repr value}"

/-- Require a Boolean field to equal the expected checkpoint metadata. -/
def expectBoolFieldEq (json : Json) (context field : String) (expected : Bool) :
    Except String Unit := do
  let valueJson ← TorchLean.Json.expectFieldE context field json
  let value ← match valueJson with
    | .bool value => pure value
    | _ => throw s!"{context}.{field}: expected boolean"
  unless value = expected do
    throw s!"{context}.{field}: expected {expected}, found {value}"

/-- Require a finite JSON number to equal an exactly representable metadata value. This is used
only for small integral constants that the released file writes with decimal syntax. -/
def expectFloatFieldEq (json : Json) (context field : String) (expected : Float) :
    Except String Unit := do
  let valueJson ← TorchLean.Json.expectFieldE context field json
  let value ← match valueJson with
    | .num number => pure number.toFloat
    | _ => throw s!"{context}.{field}: expected number"
  unless value.isFinite && value == expected do
    throw s!"{context}.{field}: expected {expected}, found {value}"

/-- Require a natural-number array to equal a schedule derived by the specification. -/
def expectNatArrayFieldEq (json : Json) (context field : String)
    (expected : Array Nat) : Except String Unit := do
  let fieldContext := s!"{context}.{field}"
  let entries ← TorchLean.Json.expectArrayE fieldContext
    (← TorchLean.Json.expectFieldE context field json)
  let value ← entries.mapM (TorchLean.Json.expectNatE fieldContext)
  unless value = expected do
    throw s!"{context}.{field}: expected {repr expected}, found {repr value}"

/-- Validate the architecture-bearing fields of a released Kimi K3 `config.json`.

Generation defaults and Transformers bookkeeping are intentionally ignored. The checked fields
are exactly those used by the Lean architecture, the multimodal splice, and the packed MXFP4
checkpoint representation.
-/
def validateJson (json : Json) : Except String Unit := do
  let _ ← TorchLean.Json.expectObjE "Kimi K3 config" json
  expectStringFieldEq json "Kimi K3 config" "model_type" "kimi_k3"
  expectNatFieldEq json "Kimi K3 config" "media_placeholder_token_id" 163605

  let text ← TorchLean.Json.expectFieldE "Kimi K3 config" "text_config" json
  let cfg := releasedCheckpointConfig.text
  expectNatFieldEq text "text_config" "hidden_size" cfg.hiddenDim
  expectNatFieldEq text "text_config" "num_hidden_layers" cfg.numLayers
  expectNatFieldEq text "text_config" "num_attention_heads" cfg.numHeads
  expectNatFieldEq text "text_config" "v_head_dim" cfg.valueHeadDim
  expectNatFieldEq text "text_config" "kv_lora_rank" cfg.kvLatentDim
  expectNatFieldEq text "text_config" "q_lora_rank" cfg.queryLatentDim
  expectNatFieldEq text "text_config" "qk_nope_head_dim" cfg.qkNopeHeadDim
  expectNatFieldEq text "text_config" "qk_rope_head_dim" cfg.qkReservedHeadDim
  expectNatFieldEq text "text_config" "intermediate_size" cfg.denseHiddenDim
  expectNatFieldEq text "text_config" "moe_intermediate_size" cfg.routedExpertHiddenDim
  expectNatFieldEq text "text_config" "routed_expert_hidden_size" cfg.routedLatentDim
  expectNatFieldEq text "text_config" "num_experts" cfg.numRoutedExperts
  expectNatFieldEq text "text_config" "num_experts_per_token" cfg.activeExperts
  expectNatFieldEq text "text_config" "num_shared_experts" cfg.numSharedExperts
  expectNatFieldEq text "text_config" "first_k_dense_replace" cfg.firstDenseLayers
  expectNatFieldEq text "text_config" "attn_res_block_size" cfg.attnResBlockSize
  expectNatFieldEq text "text_config" "vocab_size" cfg.vocabSize
  expectNatFieldEq text "text_config" "max_position_embeddings" cfg.maxContext
  expectNatFieldEq text "text_config" "num_nextn_predict_layers" cfg.numMtpLayers
  expectFloatFieldEq text "text_config" "activation_situ_beta" cfg.situGateCap.toFloat
  expectFloatFieldEq text "text_config" "activation_situ_linear_beta" cfg.situUpCap.toFloat

  let linearAttention ← TorchLean.Json.expectFieldE "text_config" "linear_attn_config" text
  expectNatFieldEq linearAttention "linear_attn_config" "head_dim" cfg.kdaHeadDim
  expectNatFieldEq linearAttention "linear_attn_config" "num_heads" cfg.numHeads
  expectNatFieldEq linearAttention "linear_attn_config" "short_conv_kernel_size"
    cfg.shortConvWidth
  expectFloatFieldEq linearAttention "linear_attn_config" "gate_lower_bound" (-5.0)
  expectNatArrayFieldEq linearAttention "linear_attn_config" "full_attn_layers"
    (layerNumbers cfg .mla)
  expectNatArrayFieldEq linearAttention "linear_attn_config" "kda_layers"
    (layerNumbers cfg .kda)

  let vision ← TorchLean.Json.expectFieldE "Kimi K3 config" "vision_config" json
  let visionCfg := releasedCheckpointConfig.vision
  expectNatFieldEq vision "vision_config" "patch_size" visionCfg.patchSize
  expectNatFieldEq vision "vision_config" "vt_hidden_size" visionCfg.hiddenDim
  expectNatFieldEq vision "vision_config" "qkv_hidden_size" visionCfg.qkvHiddenDim
  expectNatFieldEq vision "vision_config" "vt_intermediate_size" visionCfg.intermediateDim
  expectNatFieldEq vision "vision_config" "vt_num_hidden_layers" visionCfg.numLayers
  expectNatFieldEq vision "vision_config" "vt_num_attention_heads" visionCfg.numHeads
  expectNatArrayFieldEq vision "vision_config" "merge_kernel_size"
    #[visionCfg.mergeHeight, visionCfg.mergeWidth]
  expectNatFieldEq vision "vision_config" "text_hidden_size" visionCfg.textHiddenDim

  let quantization ← TorchLean.Json.expectFieldE "text_config" "quantization_config" text
  expectStringFieldEq quantization "quantization_config" "format" "mxfp4-pack-quantized"
  let groups ← TorchLean.Json.expectFieldE "quantization_config" "config_groups" quantization
  let group ← TorchLean.Json.expectFieldE "config_groups" "group_0" groups
  let weights ← TorchLean.Json.expectFieldE "group_0" "weights" group
  expectNatFieldEq weights "quantization weights" "group_size" 32
  expectNatFieldEq weights "quantization weights" "num_bits" 4
  expectStringFieldEq weights "quantization weights" "scale_dtype" "torch.uint8"
  expectStringFieldEq weights "quantization weights" "strategy" "group"
  expectBoolFieldEq weights "quantization weights" "symmetric" true

/-- Read and validate a released Kimi K3 Hugging Face configuration file. -/
def validateFile (path : System.FilePath) : IO Unit := do
  let json ← TorchLean.Json.parseFile path
  match validateJson json with
  | .ok () => pure ()
  | .error message => throw <| IO.userError s!"{path}: {message}"

end ReleasedCheckpoint

/-- The dimensions transcribed from the paper and released config are internally consistent. -/
theorem paperConfig_wf : paperConfig.WF := by
  constructor <;> native_decide

/-- The paper schedule contains 69 recurrent KDA layers. -/
theorem paperConfig_kda_layer_count : paperConfig.text.countAttentionKind .kda = 69 := by
  native_decide

/-- The paper schedule contains 24 global Gated MLA layers, including the final layer. -/
theorem paperConfig_mla_layer_count : paperConfig.text.countAttentionKind .mla = 24 := by
  native_decide

/-- At the published dimensions, MLA stores 576 scalars per token instead of 30,720. The integer
identity records the exact compression ratio `160 / 3 = 53 1/3` without introducing division. -/
theorem paperConfig_mla_cache_compression :
    paperConfig.text.mlaUncompressedScalarsPerToken * 3 =
      paperConfig.text.mlaCompressedScalarsPerToken * 160 := by
  native_decide

/-- Block AttnRes retains the embedding and eight block summaries, rather than all 93 layer
outputs. -/
theorem paperConfig_attnRes_source_capacity :
    paperConfig.text.attnResSourceCapacity = 9 := by
  native_decide

end KimiK3
