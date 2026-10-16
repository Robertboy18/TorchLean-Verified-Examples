/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.List.Count
public import Mathlib.Data.List.Zip

import Lean.Elab.Tactic.Omega

/-!
# The released Ultra block schedule

Figure 2 and Table 1 of the Ultra report (pp. 3–4) describe a 108-block hybrid stack. Each
entry is a residual block containing one mixer, not a pair of mixer and feed-forward blocks.
`releasedSchedule` transcribes `layers_block_type` from the released BF16 configuration:
https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/blob/77df655d5e9f8362164ed14dd8b48f8bce657498/config.json#L19-L128

MTP is separate. Its stored shared head is attention followed by MoE, and the report's two MTP
heads share those parameters (§2.1, p. 3; §2.7, p. 12 counts 2 MTP blocks).
-/

@[expose] public section

namespace Nemotron.Architecture

/-- The three mixer kinds in the main Ultra stack. -/
inductive BlockKind where
  | mamba
  | attention
  | moe
  deriving DecidableEq, BEq, ReflBEq, LawfulBEq, Repr

/-- Main-model residual blocks in execution order, without MTP. -/
def releasedSchedule : List BlockKind := [
  .mamba, .moe, .mamba, .moe, .mamba, .moe, .mamba, .attention, .moe,
  .mamba, .moe, .mamba, .moe, .mamba, .attention, .moe, .mamba, .moe,
  .mamba, .moe, .mamba, .moe, .mamba, .attention, .moe, .mamba, .moe,
  .mamba, .moe, .mamba, .moe, .mamba, .attention, .moe, .mamba, .moe,
  .mamba, .moe, .mamba, .attention, .moe, .mamba, .moe, .mamba, .moe,
  .mamba, .moe, .mamba, .attention, .moe, .mamba, .moe, .mamba, .moe,
  .mamba, .moe, .mamba, .attention, .moe, .mamba, .moe, .mamba, .moe,
  .mamba, .attention, .moe, .mamba, .moe, .mamba, .moe, .mamba, .moe,
  .mamba, .attention, .moe, .mamba, .moe, .mamba, .moe, .mamba, .moe,
  .mamba, .attention, .moe, .mamba, .moe, .mamba, .moe, .mamba, .attention,
  .moe, .mamba, .moe, .mamba, .moe, .mamba, .moe, .mamba, .attention,
  .moe, .mamba, .moe, .mamba, .moe, .mamba, .moe, .mamba, .moe]

/-- Zero-based indices of the global-attention anchors. -/
def attentionPositions : List Nat :=
  (releasedSchedule.zipIdx.filter (fun entry => entry.1 == .attention)).map Prod.snd

/-- 108 residual blocks in the main stack. -/
theorem releasedSchedule_length : releasedSchedule.length = 108 := by decide +kernel

/-- Block counts: 48 Mamba, 12 attention, and 48 MoE. -/
theorem releasedSchedule_counts :
    releasedSchedule.count .mamba = 48 ∧
    releasedSchedule.count .attention = 12 ∧
    releasedSchedule.count .moe = 48 := by decide +kernel

/-- Zero-based positions of the twelve attention blocks in the configuration list. -/
theorem attentionPositions_eq :
    attentionPositions = [7, 14, 23, 32, 39, 48, 57, 64, 73, 82, 89, 98] := by
  decide +kernel

/-- In any schedule the three counts add up to the length, with no hidden feed-forward layer. -/
theorem counts_partition (schedule : List BlockKind) :
    schedule.count .mamba + schedule.count .attention + schedule.count .moe =
      schedule.length := by
  induction schedule with
  | nil => simp
  | cons kind rest ih =>
      cases kind <;> simp_all <;> omega

/-- The stored MTP head, reused rather than copied for each training application. -/
def mtpSchedule : List BlockKind := [.attention, .moe]

end Nemotron.Architecture
