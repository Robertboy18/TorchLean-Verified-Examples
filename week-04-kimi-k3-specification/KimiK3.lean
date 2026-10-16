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
public import KimiK3.FeedForward.Quantile
public import KimiK3.FeedForward.Histogram
public import KimiK3.GraphSpec
public import KimiK3.Sequence
public import KimiK3.Vision
public import KimiK3.Model
public import KimiK3.Model.Streaming
public import KimiK3.Training
public import KimiK3.Parallel.Context
public import KimiK3.Parallel.Experts
public import KimiK3.ChatTemplate

/-!
# Kimi K3

Lean specifications for the mathematical architecture and deterministic training and deployment
protocols described in the Kimi K3 technical report. The development is parameterized, so the
definitions can be inspected at paper scale or instantiated at smaller dimensions without changing
their equations. Empirical results and distributed implementations remain external evidence.

Reference: Kimi Team, "Kimi K3: Open Frontier Intelligence", 2026. Architecture and training
equations are in Sections 2--4, pp. 3--14: https://arxiv.org/abs/2607.24653.
-/
