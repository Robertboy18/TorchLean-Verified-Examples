/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.AttnRes
public import KimiK3.GraphSpec.Backbone
public import KimiK3.GraphSpec.Expert
public import KimiK3.GraphSpec.KDA
public import KimiK3.GraphSpec.LanguageModel
public import KimiK3.GraphSpec.MLA
public import KimiK3.GraphSpec.MoE
public import KimiK3.GraphSpec.Vision

/-!
# Executable Kimi K3 graphs

This namespace contains typed GraphSpec representations of K3 components together with theorems
relating their pure graph semantics to the mathematical architecture definitions.

KDA, MLA, AttnRes, SiTU experts, routed MoE, and MoonViT blocks have component-level denotation
theorems. The assembled decoder theorem targets a fixed-shape, packed one-token semantics. A
further theorem is still needed to relate that packed state to the list-based streaming interface
in `KimiK3.Model`.

The composed language graph follows Figure 2 and Sections 2.1--2.3 of the Kimi K3 report
(pp. 3--8). The vision embedding, divided-attention block, and merge/project terms follow
Section 2.4; a concrete checkpoint-sized graph still requires the released block parameters:
https://arxiv.org/abs/2607.24653.
-/
