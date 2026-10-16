/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Nemotron.Cache.Replay
public import Nemotron.Cache.Error
public import Nemotron.Cache.Rounding
public import Nemotron.Cache.Speculation
public import Nemotron.Cache.Stochastic
public import Nemotron.Training.Accumulation
public import Nemotron.SSM
public import Nemotron.Architecture.Schedule
public import Nemotron.Architecture.Dimensions
public import Nemotron.Architecture.Mamba2

/-!
# Nemotron SSM checkpoint contracts

The released block schedule, shape-derived cache accounting, exact replay and speculative
rollback, storage-error propagation, stochastic grid writes, gradient accumulation, and a
shape-indexed prepared SSM core with grouped Mamba-2 coefficient preparation. Rounding comes
from FloatLib. See the README for the report correspondence and what is left to refine.
-/
