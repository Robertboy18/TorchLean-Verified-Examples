/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
Authors: Robert Joseph George
-/

import HeuristicEstimation.Sketch
import Lean.Data.Json.Printer
import NN.Runtime.Autograd.Torch.Core.CheckpointIO

/-!
# Exact certificates for an extracted attention head

The requested `HEATTN01` cases are decoded through FloatLib's binary64 semantics into rational
tensors. Each Gram certificate bounds every residual query, not only the extracted query.
The actual discarded scores also give a centered perturbation bound for the nonlinear
softmax theorem. This is exact arithmetic on stored checkpoint activations, not a proof
that the external checkpoint extractor or its floating-point execution is correct.
-/

open TorchLean Lean HeuristicEstimation
open scoped BigOperators
open Runtime.Autograd.Torch.Internal.CheckpointIO (readExact readNat64)

namespace AttentionCertificate

private abbrev Binary64 := FloatLib.Floats.ExecFloat.Binary
  (exponentBits := 11) (fractionBits := 52)

/-- Framing bytes are little endian; FloatLib owns numerical decoding and nonfinite rejection. -/
private def decode (bytes : ByteArray) (offset : Nat) : Except String Rat := do
  unless offset + 8 ≤ bytes.size do throw "truncated scalar"
  let mut word : UInt64 := 0
  for i in [0:8] do
    word := word ||| (bytes[offset + i]!.toUInt64 <<< UInt64.ofNat (8 * i))
  let value : Binary64 := FloatLib.Floats.ExecFloat.Binary.ofNatBits word.toNat
  match FloatLib.Floats.ExecFloat.Binary.toRat? value with
  | some exact => return exact
  | none => throw s!"nonfinite binary64 input at byte {offset}"

private def tensor (shape : Shape) (bytes : ByteArray) (offset : Nat) :
    Except String (Tensor Rat shape) :=
  Tensor.generateFlatM shape fun i => decode bytes (offset + 8 * i.val)

private def mask (width : Nat) (bytes : ByteArray) : Except String (Tensor Bool [width]) :=
  Tensor.generateFlatM [width] fun i => do
    unless i.val < bytes.size do throw "missing coordinate mask"
    let bit := bytes[i.val]!
    unless bit == 0 || bit == 1 do throw s!"invalid mask coordinate {i.val}"
    return bit == 1

private def rational (value : Rat) : Json :=
  Json.mkObj [("exact", .str (toString value)),
    ("approximate", toJson (Float.ofInt value.num / Float.ofNat value.den))]

/-- Gram construction and checking are timed separately from the observed-query calculation. -/
private def certify (tokens index : Nat) (positive : 0 < tokens) (bytes : ByteArray) :
    IO Json := do
  let keep ← IO.ofExcept (mask 64 bytes)
  let query ← IO.ofExcept (tensor [64] bytes 64)
  let keys ← IO.ofExcept (tensor [tokens, 64] bytes (64 + 8 * 64))
  let values ← IO.ofExcept (tensor [tokens, 64] bytes (64 + 8 * (64 + tokens * 64)))
  let tails := Tensor.map (fun x : Rat => x / 8) keys
  let weights : Tensor Rat [tokens] := Tensor.full [tokens] (1 / tokens)
  let started ← IO.monoNanosNow
  let matrix ← IO.lazyPure fun _ => Sketch.restrict keep (Sketch.gram weights tails)
  let rowSums ← IO.lazyPure fun _ => Tensor.ofFn fun i : Fin 64 =>
    ∑ j : Fin 64, |matrix[(i, j)]|
  let bound ← IO.lazyPure fun _ => Tensor.foldl max 0 rowSums
  let prepared ← IO.monoNanosNow
  let accepted ← IO.lazyPure fun _ => Sketch.check matrix bound
  let checked ← IO.monoNanosNow
  unless accepted do throw <| IO.userError "computed Gram row certificate was rejected"
  let rejectsIncorrect ← IO.lazyPure fun _ => !Sketch.check matrix (bound - 1)
  unless rejectsIncorrect do throw <| IO.userError "an incorrect Gram certificate was accepted"
  let residual := Sketch.residual keep query
  let energy := Sketch.energy residual
  let mse := Sketch.error weights tails residual
  let budget := bound * energy
  unless mse ≤ budget do throw <| IO.userError "exact score error exceeds its Gram certificate"
  let differences := Tensor.matvec tails residual
  let initial := differences.getScalar ⟨0, positive⟩
  let low := Tensor.foldl min initial differences
  let high := Tensor.foldl max initial differences
  -- Scores here are full-minus-sketch; the softmax theorem uses the opposite common shift.
  let shift := (low + high) / 2
  let epsilon := (high - low) / 2
  unless Tensor.foldl (fun valid x => valid && decide (|x - shift| ≤ epsilon)) true
      differences do
    throw <| IO.userError "centered logit interval failed"
  let magnitude := Tensor.foldl (fun best x => max best |x|) 0 values
  return Json.mkObj [
    ("case", toJson index),
    ("tokens", toJson tokens),
    ("retained_coordinates", toJson (Tensor.foldl (fun n bit => if bit then n + 1 else n) 0 keep)),
    ("gram_certificate", toJson accepted),
    ("rejects_incorrect_bound", toJson rejectsIncorrect),
    ("gram_construction_ns", toJson (prepared - started)),
    ("gram_check_ns", toJson (checked - prepared)),
    ("gram_row_bound", rational bound),
    ("query_residual_energy", rational energy),
    ("exact_logit_mse", rational mse),
    ("certified_logit_mse", rational budget),
    ("full_minus_sketch_shift", rational shift),
    ("centered_logit_epsilon", rational epsilon),
    ("value_max_abs", rational magnitude),
    ("softmax_output_bound", rational (2 * magnitude * epsilon))]

/-- Check framing for the complete artifact and certify the requested initial head/query cases. -/
def run (input : System.FilePath) (count : Nat := 1) : IO (Array Json) := do
  let fileSize := (← input.metadata).byteSize.toNat
  IO.FS.withFile input .read fun source => do
    unless (← readExact "attention certificate" source 8) == "HEATTN01".toUTF8 do
      throw <| IO.userError "attention certificate: invalid input magic"
    let cases ← readNat64 "attention certificate" source
    let tokens ← readNat64 "attention certificate" source
    let width ← readNat64 "attention certificate" source
    unless 0 < cases && width == 64 do
      throw <| IO.userError "attention certificate: expected nonempty 64-wide cases"
    unless 0 < count && count ≤ cases do
      throw <| IO.userError "attention certificate: requested count is outside the artifact"
    let caseBytes := width + 8 * (width + 2 * tokens * width)
    unless fileSize == 32 + cases * caseBytes do
      throw <| IO.userError "attention certificate: payload length does not match its header"
    if positive : 0 < tokens then
      let mut results := #[]
      for index in [0:count] do
        let bytes ← readExact "attention certificate" source caseBytes
        results := results.push (← certify tokens index positive bytes)
      return results
    else
      throw <| IO.userError "attention certificate: expected a positive prefix length"

end AttentionCertificate

def main (arguments : List String) : IO UInt32 := do
  let request := match arguments with
    | [input] => some (input, 1)
    | [input, count] => count.toNat?.map (input, ·)
    | _ => none
  match request with
  | some (input, count) =>
      try
        for result in ← AttentionCertificate.run input count do IO.println result.compress
        return 0
      catch error =>
        (← IO.getStderr).putStrLn error.toString
        return 1
  | none =>
      (← IO.getStderr).putStrLn "usage: attention-certificate INPUT.bin [COUNT]"
      return 1
