/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import KimiK3.Microscaling
public import KimiK3.Training.Muon
public import KimiK3.Training.Pretraining
public import KimiK3.Training.Distillation
public import KimiK3.Training.Draft
public import KimiK3.Training.Controls

/-!
# Kimi K3 training and deployment

The component modules contain per-head Muon, the masked pretraining objective, MOPD rewards,
the EAGLE-3 draft model and sampling proofs, and the post-training budget and verbosity controls.
Import this module to use them together.
-/
