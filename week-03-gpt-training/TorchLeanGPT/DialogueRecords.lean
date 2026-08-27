/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import Mathlib.Data.Nat.Basic

/-!
# Dialogue-window contracts

Instruction tuning samples one bounded record at a time. A record contains an input window and a
contiguous interval of assistant targets inside that window. These definitions state the indexing
conditions enforced by the binary-record loader, independently of tokenization and device code.
-/

@[expose] public section

namespace TorchLeanGPT
namespace DialogueRecords

/-- The absolute corpus position predicted by row `row` of a window beginning at `offset`. -/
def targetIndex (offset row : Nat) : Nat :=
  offset + row + 1

/-- Whether a next-token row belongs to the assistant-target interval of one record. -/
def isTargetRow
    (offset targetOffset targetLength row : Nat) : Prop :=
  targetOffset ≤ targetIndex offset row ∧
    targetIndex offset row < targetOffset + targetLength

/-- Executable form of `isTargetRow`. -/
def targetRowEnabled (offset targetOffset targetLength row : Nat) : Bool :=
  targetOffset ≤ targetIndex offset row &&
    targetIndex offset row < targetOffset + targetLength

/-- The executable target selector recognizes exactly the rows described by `isTargetRow`. -/
theorem targetRowEnabled_eq_true_iff
    {offset targetOffset targetLength row : Nat} :
    targetRowEnabled offset targetOffset targetLength row = true ↔
      isTargetRow offset targetOffset targetLength row := by
  simp [targetRowEnabled, isTargetRow]

/-- Every selected target row predicts a token that lies inside the record window. -/
theorem target_index_lt_record_end
    {offset length targetOffset targetLength row : Nat}
    (hTargetsInside : targetOffset + targetLength ≤ offset + length)
    (hRow : isTargetRow offset targetOffset targetLength row) :
    targetIndex offset row < offset + length := by
  exact lt_of_lt_of_le hRow.2 hTargetsInside

/-- A row at or after the padded end of a record cannot be selected as an assistant target. -/
theorem not_target_row_of_record_end_le
    {offset length targetOffset targetLength row : Nat}
    (hTargetsInside : targetOffset + targetLength ≤ offset + length)
    (hEnd : length ≤ row + 1) :
    ¬ isTargetRow offset targetOffset targetLength row := by
  intro hRow
  have hInside := target_index_lt_record_end hTargetsInside hRow
  simp only [targetIndex] at hInside
  omega

end DialogueRecords
end TorchLeanGPT
