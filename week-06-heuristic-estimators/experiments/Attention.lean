/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
Authors: Robert Joseph George
-/

import Lean.Data.Json.Printer
import NN.Runtime.Autograd.Torch.Core.CheckpointIO
import NN.Spec.Layers.Activation
import NN.Tensor

/-!
# A trained attention head with fewer query–key coordinates

The input contains queries, keys, and values extracted from a trained checkpoint. This
runner evaluates the final query against its complete causal prefix using TorchLean tensors.
The comparison physically packs the retained query/key coordinates before multiplying;
values retain all their coordinates. The scale remains the original head width's square root.

The binary framing is `HEATTN01`, followed by little-endian unsigned 64-bit counts for cases,
prefix length, and head width. Each case contains one byte per coordinate (zero or one), then
binary64 query, key, and value entries in row-major order. Finite inputs and exact payload
lengths are required. Each output line is a JSON object containing both tensor calculations.

Packing, baseline evaluation, and reduced evaluation have separate timings. They exclude
file decoding and JSON encoding; the experiment must charge packing when comparing total cost.
The decimal arrays are convenient for inspection. The corresponding `_bits` arrays preserve
every binary64 bit and are used for numerical comparison, since Lean's ordinary JSON float
printer rounds its decimal display. This runner measures concrete floating-point execution.
The real-arithmetic error theorem lives in the accompanying formalization and is not a proof
about this executable's rounding.
-/

open TorchLean Lean
open Runtime.Autograd.Torch.Internal.CheckpointIO (readExact readNat64)

namespace AttentionExperiment

private def formatName : String := "attention experiment"

/-- Decode an external binary64 word without a native-endian reinterpretation. -/
private def word (bytes : ByteArray) (offset : Nat) : UInt64 := Id.run do
  let mut bits : UInt64 := 0
  for i in [0:8] do
    bits := bits ||| (bytes[offset + i]!.toUInt64 <<< UInt64.ofNat (8 * i))
  return bits

/-- Materialize one validated row-major tensor at the binary input boundary. -/
private def tensor (shape : Shape) (bytes : ByteArray) (offset : Nat) :
    Except String (Tensor Float shape) := do
  let count := shape.size
  unless offset + 8 * count ≤ bytes.size do
    throw "tensor payload extends past its case"
  let mut values : Array Float := Array.emptyWithCapacity count
  for i in [0:count] do
    let value := Float.ofBits (word bytes (offset + 8 * i))
    if value.isNaN || value.isInf then
      throw s!"nonfinite input at tensor coordinate {i}"
    values := values.push value
  if h : values.size = shape.size then
    return (Tensor.from values).reshape shape (by simpa [Shape.size] using h)
  else
    throw "decoded tensor has the wrong number of entries"

/-- A coordinate mask is data, not a floating-point multiplication by zero. -/
private def mask (width : Nat) (bytes : ByteArray) : Except String (Tensor Bool [width]) := do
  unless width ≤ bytes.size do
    throw "missing coordinate mask"
  for i in [0:width] do
    unless bytes[i]! == 0 || bytes[i]! == 1 do
      throw s!"mask coordinate {i} is neither zero nor one"
  return Tensor.ofFn fun (i : Fin width) => bytes[i.val]! == 1

/-- Scores, probabilities, and the value-weighted result for one final query. -/
private def attention {tokens width valueWidth : Nat}
    (query : Tensor Float [width]) (keys : Tensor Float [tokens, width])
    (values : Tensor Float [tokens, valueWidth]) (scale : Float) :
    Tensor Float [tokens] × Tensor Float [tokens] × Tensor Float [valueWidth] :=
  let logits := Tensor.map (fun value => value / scale) (Tensor.matvec keys query)
  let weights := Activation.softmaxVecSpec logits
  (logits, weights, Tensor.vecmat weights values)

private def finite {shape : Shape} (values : Tensor Float shape) : Bool :=
  Tensor.foldl (fun valid value => valid && !value.isNaN && !value.isInf) true values

private def floatJson (value : Float) : Lean.Json :=
  match Lean.JsonNumber.fromFloat? value with
  | .inr number => .num number
  | .inl spelling => .str spelling

private def tensorJson {shape : Shape} (values : Tensor Float shape) : Lean.Json :=
  .arr ((Tensor.to values (Array Float)).map floatJson)

private def bitsJson {shape : Shape} (values : Tensor Float shape) : Lean.Json :=
  toJson ((Tensor.to values (Array Float)).map Float.toBits)

private def resultJson {tokens width : Nat}
    (result : Tensor Float [tokens] × Tensor Float [tokens] × Tensor Float [width]) :
    Lean.Json :=
  Lean.Json.mkObj [
    ("logits", tensorJson result.1),
    ("weights", tensorJson result.2.1),
    ("output", tensorJson result.2.2),
    ("logits_bits", bitsJson result.1),
    ("weights_bits", bitsJson result.2.1),
    ("output_bits", bitsJson result.2.2)]

/-- Evaluate a case without rebuilding the baseline for each observed error. -/
private def runCase (tokens width index : Nat) (bytes : ByteArray) : IO Lean.Json := do
  let selected ← IO.ofExcept (mask width bytes)
  let query ← IO.ofExcept (tensor [width] bytes width)
  let keys ← IO.ofExcept (tensor [tokens, width] bytes (width + 8 * width))
  let values ← IO.ofExcept
    (tensor [tokens, width] bytes (width + 8 * width + 8 * tokens * width))
  let scale := Float.sqrt (Float.ofNat width)

  let fullStart ← IO.monoNanosNow
  let full ← IO.lazyPure fun _ => attention query keys values scale
  let fullFinish ← IO.monoNanosNow

  let packStart ← IO.monoNanosNow
  let kept ← IO.lazyPure fun _ =>
    (Array.ofFn fun i : Fin width => i).filter fun i => selected.getScalar i
  let reducedQuery ← IO.lazyPure fun _ =>
    Tensor.ofFn fun i : Fin kept.size => query.getScalar kept[i]
  let reducedKeys ← IO.lazyPure fun _ =>
    Tensor.matrix fun (i : Fin tokens) (j : Fin kept.size) => keys[(i, kept[j])]
  let packFinish ← IO.monoNanosNow

  let reducedStart ← IO.monoNanosNow
  let reduced ← IO.lazyPure fun _ => attention reducedQuery reducedKeys values scale
  let reducedFinish ← IO.monoNanosNow

  unless finite full.1 && finite full.2.1 && finite full.2.2 &&
      finite reduced.1 && finite reduced.2.1 && finite reduced.2.2 do
    throw <| IO.userError s!"case {index}: nonfinite attention result"
  let outputDifference := full.2.2 - reduced.2.2
  return Lean.Json.mkObj [
    ("case", toJson index),
    ("rank", toJson kept.size),
    ("full", resultJson full),
    ("sketch", resultJson reduced),
    ("logit_max_abs", floatJson (Tensor.maxAbsDiff full.1 reduced.1)),
    ("weight_l1", floatJson (Tensor.sum (Tensor.map Float.abs (full.2.1 - reduced.2.1)))),
    ("output_max_abs", floatJson (Tensor.maxAbs outputDifference)),
    ("output_squared_error", floatJson (Tensor.sum (outputDifference * outputDifference))),
    ("full_ns", toJson (fullFinish - fullStart)),
    ("packing_ns", toJson (packFinish - packStart)),
    ("sketch_ns", toJson (reducedFinish - reducedStart))]

/-- Stream one case at a time; an incomplete or overlong artifact is rejected. -/
def run (input output : System.FilePath) : IO Unit := do
  if input == output then
    throw <| IO.userError "attention experiment: input and output must be different files"
  let fileSize := (← input.metadata).byteSize.toNat
  IO.FS.withFile input .read fun source => do
    let magic ← readExact formatName source 8
    unless magic == "HEATTN01".toUTF8 do
      throw <| IO.userError "attention experiment: invalid input magic"
    let cases ← readNat64 formatName source
    let tokens ← readNat64 formatName source
    let width ← readNat64 formatName source
    unless 0 < tokens && 0 < width do
      throw <| IO.userError "attention experiment: prefix and head width must be positive"
    let caseBytes := width + 8 * (width + 2 * tokens * width)
    unless fileSize == 32 + cases * caseBytes do
      throw <| IO.userError s!"attention experiment: payload length does not match {cases} cases"
    IO.FS.withFile output .write fun destination => do
      for index in [0:cases] do
        let bytes ← readExact formatName source caseBytes
        let result ← runCase tokens width index bytes
        destination.putStrLn result.compress
      destination.flush
    unless (← source.read 1).isEmpty do
      throw <| IO.userError "attention experiment: trailing input bytes"

end AttentionExperiment

def main (arguments : List String) : IO UInt32 := do
  match arguments with
  | [input, output] =>
      try
        AttentionExperiment.run input output
        return 0
      catch error =>
        (← IO.getStderr).putStrLn error.toString
        return 1
  | _ =>
      (← IO.getStderr).putStrLn "usage: attention-experiment INPUT.bin OUTPUT.jsonl"
      return 1
