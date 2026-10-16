/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Sequence.Scan
public import KimiK3.Sequence.KDA
public import KimiK3.Sequence.MLA
public import KimiK3.Sequence.AttnRes

/-!
# Kimi K3 sequence and depth mixing

The component modules contain KDA, Gated MLA, Block AttnRes, and the causal scan laws shared by
the sequence and streaming APIs. Import this module to use them together.
-/
