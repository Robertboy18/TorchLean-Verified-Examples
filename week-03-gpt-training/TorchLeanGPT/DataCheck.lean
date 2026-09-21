/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

import TorchLeanGPT.Run

/-!
# Training-data regression checks

Check the tensor and file-I/O boundaries not covered by the packing and scanning proofs.
The caller supplies an empty scratch directory for the file fixtures.
-/

open TorchLean TorchLeanGPT TorchLeanGPT.Run

private def check (label : String) (ok : Bool) : IO Unit :=
  unless ok do throw <| IO.userError label

private def checkBatches : IO Unit := do
  let bytes := ByteArray.mk <| (Array.range 32).flatMap (fun i => #[i.toUInt8, 0])
  let shard : TokenShard := { bytes, contentHash := hash bytes }
  let maskBytes := ByteArray.mk <| (Array.range 32).map
    (fun i => if i % 3 == 0 then 0 else 1)
  let mask : TargetMask := { bytes := maskBytes, contentHash := hash maskBytes }
  -- Empty tensors, all-padding rows, partial records, and a full prediction window.
  for (length, batch, seqLen) in [(0, 0, 0), (1, 1, 3), (3, 2, 4), (12, 4, 8)] do
    let records : DialogueRecords :=
      { entries := #[{ offset := 0, length, targetOffset := 2, targetLength := 3 },
          { offset := 13, length, targetOffset := 14, targetLength := 20 }]
        contentHash := 0 }
    let actual := causalLmMaskedTokenBatchFromRecords (α := Float)
      32 batch seqLen shard mask records 7 11 31
    let key := Spec.Random.keyOf 7 11
    let rows := (List.range batch).flatMap fun bi =>
      let record := records.entries[Spec.Random.sampleNat key bi records.entries.size]!
      (List.range seqLen).map fun i =>
        if i + 1 < record.length then
          let target := record.offset + i + 1
          (record.offset + i, target,
            decide (record.targetOffset ≤ target ∧
              target < record.targetOffset + record.targetLength) && mask.getD target)
        else (31, 31, false)
    let active := (rows.filter fun row => row.2.2).length
    let weight := if active == 0 then 0.0 else 1.0 / Float.ofNat active
    check s!"batch {length}/{batch}/{seqLen}" <|
      (Tensor.to actual.1 (List (Fin 32))).map Fin.val == rows.map Prod.fst &&
      (Tensor.to actual.2.1 (List (Fin 32))).map Fin.val == rows.map (fun r => r.2.1) &&
      Tensor.to actual.2.2 (List Float) == rows.map (fun r => if r.2.2 then weight else 0)

private def expectError {α : Type} (label expected : String) (action : IO α) : IO Unit := do
  let result ← try
      let _ ← action
      pure none
    catch error => pure (some error.toString)
  check label (result == some expected)

private def checkFiles (directory : System.FilePath) : IO Unit := do
  let tokens := directory / "tokens.bin"
  let mask := directory / "mask.bin"
  for bytes in [ByteArray.empty, ByteArray.mk #[0, 0, 255, 0, 0, 1, 255, 255]] do
    IO.FS.writeBinFile tokens bytes
    let loaded ← readTokenShard tokens 65536
    check "token bytes/hash" (loaded.bytes == bytes && loaded.contentHash == hash bytes)
  IO.FS.writeBinFile tokens (ByteArray.mk #[0])
  expectError "truncated token" s!"{exeName}: {tokens} has an odd byte count"
    (readTokenShard tokens 32)
  IO.FS.writeBinFile tokens (ByteArray.mk #[0, 0, 32, 0, 255, 255])
  expectError "first invalid token"
    s!"{exeName}: token id 32 in {tokens} is outside vocabulary [0, 32)"
    (readTokenShard tokens 32)
  for bytes in [ByteArray.empty, ByteArray.mk #[0, 1, 1, 0]] do
    IO.FS.writeBinFile mask bytes
    let loaded ← readTargetMask mask
    check "mask bytes/hash" (loaded.bytes == bytes && loaded.contentHash == hash bytes)
  for value in [2:256] do
    IO.FS.writeBinFile mask (ByteArray.mk #[0, 1, value.toUInt8, 255])
    expectError s!"invalid mask byte {value}"
      s!"{exeName}: target mask {mask} contains byte {value} at offset 2; expected 0 or 1"
      (readTargetMask mask)

public def main (args : List String) : IO Unit := do
  let [directory] := args
    | throw <| IO.userError "usage: check_torchlean_gpt_data EMPTY_SCRATCH_DIRECTORY"
  unless (← System.FilePath.readDir directory).isEmpty do
    throw <| IO.userError "the fixture directory must be empty"
  IO.println s!"File fixtures: {directory}"
  checkBatches
  checkFiles directory
  IO.println "PASS: masked tensor boundaries and shard/mask file validation"
