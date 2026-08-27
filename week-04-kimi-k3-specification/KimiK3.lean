/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Config
public import KimiK3.Common
public import KimiK3.Microscaling
public import KimiK3.FeedForward
public import KimiK3.GraphSpec
public import KimiK3.Sequence
public import KimiK3.Vision
public import KimiK3.Model
public import KimiK3.Training

/-!
# Kimi K3

Lean specifications for the Kimi K3 architecture and the training and deployment procedures
described in its technical report. The development is parameterized, so the definitions can be
inspected at paper scale or instantiated at smaller dimensions without changing their equations.

Reference: Kimi Team, "Kimi K3: Open Frontier Intelligence", 2026,
https://arxiv.org/abs/2607.24653.
-/
