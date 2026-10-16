/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.FeedForward.MoE

/-!
# Kimi K3 feed-forward modules

SiTU activation bounds, shared and quantized experts, top-k routing, and Stable LatentMoE
composition live in their own modules. Importing this file brings in all of them through
`KimiK3.FeedForward.MoE`.
-/

@[expose] public section
