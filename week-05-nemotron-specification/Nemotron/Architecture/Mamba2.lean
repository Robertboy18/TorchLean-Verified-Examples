/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Nemotron.SSM
public import NN.Spec.Layers.Activation
import NN.Proofs.Gradients.Activation
import Mathlib.Tactic.NormNum

/-!
# Grouped Mamba-2 coefficient preparation

Builds the prepared inputs of `Nemotron.SSM` from the grouped recurrence in the model source.
TorchLean's own `SelectiveMamba` is a different architecture and is not used for Ultra. Source,
pinned at the inspected revision:
https://github.com/huggingface/transformers/blob/c8b25ab3af135bcfd9496eff3ef507303da17f15/src/transformers/models/nemotron_h/modeling_nemotron_h.py

`mamba2_selective_state_update` (lines 191–249) expands each B/C group across contiguous heads,
uses `exp(dt*A)`, writes `state*dA + (dt*B)*x`, reads that new state against C, and adds D*x.
`NemotronHMamba2Mixer` sets `A = -exp(A_log)` (line 461) and supplies softplus time steps.
Chunked prefill also clamps each time step below at `time_step_min` (lines 275, 394,
574), and the decode update has no clamp. `Parameters.timeStepMin` selects either path.

Inputs arrive *after* input projection and causal convolution, as values, B, C, and raw
time-step logits. This module covers discretization, contiguous grouping, the state update, and
the D skip. It omits convolution state, grouped gated RMS normalization, output
projection, padding, finite casts, and native kernels. All theorems are exact-real statements
for fixed prepared inputs. Like the source, `step` reads the new state before any storage cast,
so quantizing the cache does not change the readout.
-/

@[expose] public section

namespace Nemotron.Architecture.Mamba2

open Spec TorchLean

/-- Contiguous group assignment, as expand-then-reshape gives in the source. Writing the head
count as `groups * headsPerGroup` builds in divisibility. -/
def groupIndex {groups headsPerGroup : Nat} (hHeadsPerGroup : 0 < headsPerGroup)
    (head : Fin (groups * headsPerGroup)) : Fin groups :=
  ⟨head.val / headsPerGroup, (Nat.div_lt_iff_lt_mul hHeadsPerGroup).mpr head.isLt⟩

/-- Per-head mixer parameters, fixed across tokens. -/
structure Parameters (heads : Nat) where
  logA : Tensor ℝ [heads]
  dtBias : Tensor ℝ [heads]
  skip : Tensor ℝ [heads]
  /-- Lower time-step limit: `time_step_min` (0.001 in the released configuration) for the
  chunked path, or `0` for the decode update, which has no clamp. -/
  timeStepMin : ℝ

/-- One token after projection and convolution. B and C stay per group, not copied per head. -/
structure Token (groups headsPerGroup headDim stateDim : Nat) where
  value : Tensor ℝ [groups * headsPerGroup, headDim]
  B : Tensor ℝ [groups, stateDim]
  C : Tensor ℝ [groups, stateDim]
  rawDt : Tensor ℝ [groups * headsPerGroup]

/-- Unit-scale softplus of `raw + bias` (TorchLean's branch-stable real definition), then the
source's lower clamp. PyTorch's large-input threshold branch and finite precision are not
modeled. -/
noncomputable def timeStep (minimum raw bias : ℝ) : ℝ :=
  max (Activation.Math.softplusSpec (raw + bias)) minimum

private theorem softplus_pos (x : ℝ) : 0 < Activation.Math.softplusSpec x := by
  rw [Proofs.softplus_spec_eq_log_one_add_exp]
  exact Real.log_pos (lt_add_of_pos_right 1 (Real.exp_pos _))

theorem timeStep_pos (minimum raw bias : ℝ) : 0 < timeStep minimum raw bias :=
  lt_max_of_lt_left (softplus_pos _)

/-- Without a positive floor the clamp does nothing: this is the decode path's plain softplus. -/
theorem timeStep_of_nonpos (minimum raw bias : ℝ) (hminimum : minimum ≤ 0) :
    timeStep minimum raw bias = Activation.Math.softplusSpec (raw + bias) :=
  max_eq_left (hminimum.trans (softplus_pos _).le)

/-- Per-head decay `exp(delta * A)` with the source's `A = -exp(logA)`. -/
noncomputable def retention (delta logA : ℝ) : ℝ := Real.exp (-(delta * Real.exp logA))

/-- For a nonnegative time step the decay lies in `(0, 1]`, whatever the learned log-rate, so
discretization never expands the prior state. -/
theorem retention_bounds (delta logA : ℝ) (hdelta : 0 ≤ delta) :
    0 < retention delta logA ∧ retention delta logA ≤ 1 := by
  constructor
  · exact Real.exp_pos _
  · apply Real.exp_le_one_iff.mpr
    exact neg_nonpos.mpr (mul_nonneg hdelta (Real.exp_nonneg _))

/-- Prepared SSM input for one head, where the time step scales B but not C. -/
noncomputable def prepareHead {groups headsPerGroup headDim stateDim : Nat}
    (hHeadsPerGroup : 0 < headsPerGroup)
    (parameters : Parameters (groups * headsPerGroup))
    (input : Token groups headsPerGroup headDim stateDim)
    (head : Fin (groups * headsPerGroup)) : SSM.Input ℝ headDim stateDim :=
  let delta := timeStep parameters.timeStepMin (input.rawDt.getScalar head)
    (parameters.dtBias.getScalar head)
  let group := groupIndex hHeadsPerGroup head
  { retention := retention delta (parameters.logA.getScalar head)
    value := Tensor.unstack input.value head
    write := Tensor.scaleSpec (Tensor.unstack input.B group) delta
    read := Tensor.unstack input.C group }

/-- Every prepared head has retention at most `1` in absolute value. -/
theorem prepareHead_bounded {groups headsPerGroup headDim stateDim : Nat}
    (hHeadsPerGroup : 0 < headsPerGroup)
    (parameters : Parameters (groups * headsPerGroup))
    (input : Token groups headsPerGroup headDim stateDim)
    (head : Fin (groups * headsPerGroup)) :
    |(prepareHead hHeadsPerGroup parameters input head).retention| ≤ 1 := by
  have hb := retention_bounds
    (timeStep parameters.timeStepMin (input.rawDt.getScalar head)
      (parameters.dtBias.getScalar head))
    (parameters.logA.getScalar head) (timeStep_pos _ _ _).le
  simpa only [prepareHead, abs_of_pos hb.1] using hb.2

/-- Coordinatewise, the update is the source's scalar Mamba-2 equation
`state * dA + x * (B * dt)`. -/
theorem updateHead_apply {groups headsPerGroup headDim stateDim : Nat}
    (hHeadsPerGroup : 0 < headsPerGroup)
    (parameters : Parameters (groups * headsPerGroup))
    (input : Token groups headsPerGroup headDim stateDim)
    (head : Fin (groups * headsPerGroup)) (state : SSM.State ℝ headDim stateDim)
    (valueIndex : Fin headDim) (stateIndex : Fin stateDim) :
    SSM.update state (prepareHead hHeadsPerGroup parameters input head)
        (valueIndex, stateIndex, PUnit.unit) =
      let delta := timeStep parameters.timeStepMin (input.rawDt.getScalar head)
        (parameters.dtBias.getScalar head)
      state (valueIndex, stateIndex, PUnit.unit) * retention delta (parameters.logA.getScalar head) +
        (Tensor.unstack input.value head).getScalar valueIndex *
          ((Tensor.unstack input.B (groupIndex hHeadsPerGroup head)).getScalar stateIndex * delta) := by
  simp [SSM.update, prepareHead, Tensor.addSpec, Tensor.scaleSpec, Tensor.outerProductSpec,
    Tensor.map2Spec, Tensor.map, Tensor.getScalar_eq_apply]

/-- Update every head's state matrix with its own prepared input, reusing `SSM.update`. -/
noncomputable def updateHeads {groups headsPerGroup headDim stateDim : Nat}
    (hHeadsPerGroup : 0 < headsPerGroup)
    (parameters : Parameters (groups * headsPerGroup))
    (input : Token groups headsPerGroup headDim stateDim)
    (state : Tensor ℝ [groups * headsPerGroup, headDim, stateDim]) :
    Tensor ℝ [groups * headsPerGroup, headDim, stateDim] :=
  Tensor.stackLeading fun head => SSM.update (Tensor.unstack state head)
    (prepareHead hHeadsPerGroup parameters input head)

/-- One mixer step: update every head, read the new state against C, and add the D skip. -/
noncomputable def step {groups headsPerGroup headDim stateDim : Nat}
    (hHeadsPerGroup : 0 < headsPerGroup)
    (parameters : Parameters (groups * headsPerGroup))
    (state : Tensor ℝ [groups * headsPerGroup, headDim, stateDim])
    (input : Token groups headsPerGroup headDim stateDim) :
    Tensor ℝ [groups * headsPerGroup, headDim, stateDim] ×
      Tensor ℝ [groups * headsPerGroup, headDim] :=
  let next := updateHeads hHeadsPerGroup parameters input state
  (next, Tensor.stackLeading fun head =>
    Tensor.addSpec
      (Tensor.matVecMulSpec (Tensor.unstack next head)
        (Tensor.unstack input.C (groupIndex hHeadsPerGroup head)))
      (Tensor.scaleSpec (Tensor.unstack input.value head) (parameters.skip.getScalar head)))

/-- One update over all heads does not increase a coordinate error in the prior state, since
every prepared retention is at most `1`. Both sides use the same token and parameters. -/
theorem updateHeads_within {groups headsPerGroup headDim stateDim : Nat}
    (hHeadsPerGroup : 0 < headsPerGroup)
    (parameters : Parameters (groups * headsPerGroup))
    (input : Token groups headsPerGroup headDim stateDim)
    (x y : Tensor ℝ [groups * headsPerGroup, headDim, stateDim])
    (e : ℝ) (hxy : Cache.Within e x y) :
    Cache.Within e (updateHeads hHeadsPerGroup parameters input x)
      (updateHeads hHeadsPerGroup parameters input y) := by
  intro coordinate
  have hhead : Cache.Within e (Tensor.unstack x coordinate.1)
      (Tensor.unstack y coordinate.1) := by
    intro inner
    simpa [Tensor.unstack] using hxy (coordinate.1, inner)
  have hnext := SSM.update_within (Tensor.unstack x coordinate.1)
    (Tensor.unstack y coordinate.1)
    (prepareHead hHeadsPerGroup parameters input coordinate.1) 1 e (by norm_num)
    (prepareHead_bounded hHeadsPerGroup parameters input coordinate.1) hhead
  simpa [updateHeads, Tensor.stackLeading, Tensor.dim] using hnext coordinate.2

end Nemotron.Architecture.Mamba2
