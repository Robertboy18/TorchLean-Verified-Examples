/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
Authors: Robert Joseph George
-/

import HeuristicEstimation.Sketch
import Lean.Data.Json.Parser
import Lean.Data.Json.Printer
import NN.Runtime.Autograd.Model.Metrics
import NN.Spec.Layers.Linear

/-!
# Removing pixels from a trained digits classifier

We fit coordinate masks on the first 120 exported examples, without using their labels, and
evaluate the remaining 240. Removed coordinates are replaced by their calibration means.
Forward calculations use TorchLean's linear specification and physically packed tensor
products. All numbers are interpreted as the exact decimal rationals in the JSON artifacts;
these are not claims about the original PyTorch training arithmetic.

Certificates bound each discarded winner–competitor logit gap by a rational triangle bound.
The stricter domain certificate uses only the fact that every input pixel lies in `[0,1]`.
Checking a finite file natively is separate from emitting a kernel proof term for that file.

The parameter-quantization sweep rounds both weights and biases through FloatLib's configured
binary formats. Its forward arithmetic remains rational; this does not measure FP16 inference.
-/

open TorchLean Lean HeuristicEstimation
open scoped BigOperators

namespace DigitsExperiment

private def scalar (json : Json) : Except String Rat := do
  let number ← json.getNum?
  return (number.mantissa : Rat) / (10 ^ number.exponent : Nat)

private def vector (n : Nat) (json : Json) : Except String (Tensor Rat [n]) := do
  let entries ← json.getArr?
  unless entries.size == n do throw s!"expected {n} vector entries"
  Tensor.generateFlatM [n] fun i => scalar entries[i.val]!

private def load (weightsPath datasetPath : String) :
    IO (Spec.LinearSpec Rat 64 10 × Array (Tensor Rat [64] × Nat)) := do
  let weights ← IO.ofExcept (Json.parse (← IO.FS.readFile weightsPath))
  let rows ← IO.ofExcept ((← IO.ofExcept (weights.getObjVal? "layers.0.weight")).getArr?)
  unless rows.size == 10 do throw <| IO.userError "expected ten weight rows"
  let matrix ← IO.ofExcept <| Tensor.stackLeadingM fun i : Fin 10 => vector 64 rows[i.val]!
  let bias ← IO.ofExcept (vector 10 (← IO.ofExcept (weights.getObjVal? "layers.0.bias")))
  let dataset ← IO.ofExcept (Json.parse (← IO.FS.readFile datasetPath))
  let examples ← IO.ofExcept ((← IO.ofExcept (dataset.getObjVal? "examples")).getArr?)
  unless examples.size == 360 do throw <| IO.userError "expected 360 exported examples"
  let entries ← examples.mapM fun entry => do
    let x ← IO.ofExcept (vector 64 (← IO.ofExcept (entry.getObjVal? "x")))
    let label ← IO.ofExcept ((← IO.ofExcept (entry.getObjVal? "y")).getNat?)
    unless label < 10 && Tensor.foldl (fun ok x => ok && decide (0 ≤ x ∧ x ≤ 1)) true x do
      throw <| IO.userError "invalid label or pixel outside [0,1]"
    return (x, label)
  return (⟨matrix, bias⟩, entries)

private def rational (x : Rat) : Json :=
  Json.mkObj [("exact", .str (toString x)),
    ("approximate", toJson (Float.ofInt x.num / Float.ofNat x.den))]

private def sha256 (path : String) : IO String := do
  let result ← IO.Process.output { cmd := "sha256sum", args := #[path] }
  unless result.exitCode == 0 do throw <| IO.userError result.stderr
  return (result.stdout.splitOn " ").head!

/-- Strict estimated gaps dominate error gaps, so the unique winner is preserved. -/
private theorem winner_preserved {n : Nat} (full estimate error : Tensor Rat [n])
    (identity : ∀ i, full.getScalar i = estimate.getScalar i + error.getScalar i)
    (winner : Fin n)
    (margin : ∀ j, j ≠ winner →
      |error.getScalar winner - error.getScalar j| <
        estimate.getScalar winner - estimate.getScalar j) :
    ∀ j, j ≠ winner → full.getScalar j < full.getScalar winner := by
  intro j hj
  have h := margin j hj
  have herror := neg_abs_le (error.getScalar winner - error.getScalar j)
  rw [identity j, identity winner]
  linarith

/-- The covariance Gram is the calibration mean squared output error for a removed mask. -/
private def fit (model : Spec.LinearSpec Rat 64 10)
    (samples : Array (Tensor Rat [64] × Nat)) :
    Tensor Rat [64] × Tensor Rat [64, 64] × Tensor Rat [64] :=
  let mean := Tensor.ofFn fun i : Fin 64 =>
    (∑ s : Fin samples.size, samples[s.val]!.1.getScalar i) / samples.size
  let covariance := Tensor.matrix fun i j : Fin 64 =>
    (∑ s : Fin samples.size,
      (samples[s.val]!.1.getScalar i - mean.getScalar i) *
        (samples[s.val]!.1.getScalar j - mean.getScalar j)) / samples.size
  let gram := Tensor.matrix fun i j : Fin 64 => covariance[(i, j)] *
    ∑ c : Fin 10, model.weights[(c, i)] * model.weights[(c, j)]
  let magnitude := Tensor.ofFn fun i : Fin 64 =>
    ∑ c : Fin 10, model.weights[(c, i)] ^ 2
  (mean, gram, magnitude)

/-- Remove the coordinate causing the smallest increase in total calibration output MSE. -/
private def greedy (gram : Tensor Rat [64, 64]) : Array (Fin 64) := Id.run do
  let mut removed : Array (Fin 64) := #[]
  for _ in [0:64] do
    let available := (Array.finRange 64).filter fun i => !removed.contains i
    let cost := fun i : Fin 64 => gram[(i, i)] +
      2 * removed.foldl (fun sum j => sum + gram[(i, j)]) 0
    let best := available.foldl (fun best i => if cost i < cost best then i else best)
      available[0]!
    removed := removed.push best
  return removed

private def select (order : Array (Fin 64)) (retained : Nat) : Tensor Bool [64] :=
  Tensor.ofFn fun i => !(order.extract 0 (64 - retained)).contains i

private def evaluate (model : Spec.LinearSpec Rat 64 10) (mean : Tensor Rat [64])
    (samples : Array (Tensor Rat [64] × Nat)) (keep : Tensor Bool [64]) : IO Json := do
  let coordinates := (Array.finRange 64).filter fun i => keep.getScalar i
  let packed := Tensor.matrix fun (c : Fin 10) (i : Fin coordinates.size) =>
    model.weights[(c, coordinates[i])]
  let anchor := Spec.linearSpec model mean
  let domain := Tensor.matrix fun c j : Fin 10 => ∑ i : Fin 64,
    if keep.getScalar i then 0 else
      |model.weights[(c, i)] - model.weights[(j, i)]| *
        max (mean.getScalar i) (1 - mean.getScalar i)
  let started ← IO.monoNanosNow
  let mut records : Array Json := #[]
  let mut totalError : Rat := 0
  let mut correct := 0
  let mut agree := 0
  let mut certified := 0
  let mut domainCertified := 0
  for index in [0:samples.size] do
    let (input, label) := samples[index]!
    let full := Spec.linearSpec model input
    let centered := Tensor.ofFn fun i : Fin coordinates.size =>
      input.getScalar coordinates[i] - mean.getScalar coordinates[i]
    let approximation : Tensor Rat [10] := Tensor.matvec packed centered + anchor
    let residual := Tensor.ofFn fun i : Fin 64 =>
      if keep.getScalar i then 0 else input.getScalar i - mean.getScalar i
    let error := Tensor.matvec model.weights residual
    unless Tensor.foldl (fun ok x => ok && x == 0) true (full - approximation - error) do
      throw <| IO.userError "linear residual identity failed"
    let winner : Fin 10 := Fin.cast (by decide)
      ((Metrics.argmax? approximation).getD ⟨0, by decide⟩)
    let original : Fin 10 := Fin.cast (by decide)
      ((Metrics.argmax? full).getD ⟨0, by decide⟩)
    let gapBound := fun j : Fin 10 => ∑ i : Fin 64,
      |model.weights[(winner, i)] - model.weights[(j, i)]| * |residual.getScalar i|
    let checked := (Array.finRange 10).all fun j => j == winner || decide
      (gapBound j < approximation.getScalar winner - approximation.getScalar j)
    let checkedDomain := (Array.finRange 10).all fun j => j == winner || decide
      (domain[(winner, j)] < approximation.getScalar winner - approximation.getScalar j)
    unless (Array.finRange 10).all (fun j => decide
        (|error.getScalar winner - error.getScalar j| ≤ gapBound j ∧
          gapBound j ≤ domain[(winner, j)])) do
      throw <| IO.userError "discarded logit-gap bound failed"
    unless (!checked && !checkedDomain) || winner == original do
      throw <| IO.userError "a certified prediction disagrees with the full model"
    if winner.val == label then correct := correct + 1
    if winner == original then agree := agree + 1
    if checked then certified := certified + 1
    if checkedDomain then domainCertified := domainCertified + 1
    totalError := totalError + Tensor.sum (Tensor.square (full - approximation))
    records := records.push <| Json.mkObj [
      ("index", toJson (index + 120)), ("label", toJson label),
      ("prediction", toJson winner.val), ("full_prediction", toJson original.val),
      ("input_certificate", toJson checked), ("pixel_domain_certificate", toJson checkedDomain)]
  let finished ← IO.monoNanosNow
  return Json.mkObj [
    ("coordinates", toJson (coordinates.map Fin.val)), ("correct", toJson correct),
    ("agreement", toJson agree), ("input_certified", toJson certified),
    ("pixel_domain_certified", toJson domainCertified),
    ("logit_mse", rational (totalError / (samples.size * 10 : Nat))),
    ("evaluation_with_checks_ns", toJson (finished - started)), ("examples", .arr records)]

/-- Quantize parameters once, then measure their effect with exact forward arithmetic. -/
private def quantization (format : FloatLib.Floats.Formats.BinaryInterchange.FloatFormat)
    (model : Spec.LinearSpec Rat 64 10) (samples : Array (Tensor Rat [64] × Nat)) : IO Json := do
  let convert : Rat → Except String Rat := fun x => do
    let rounded : FloatLib.Floats.ExecFloat.Binary format.expWidth format.fracWidth
        format.encoding format.exponentBias format.expWidth_ge_two format.fracWidth_pos
        format.exponentBias_pos format.exponentBias_le_maxFinite := Rat.cast x
    match FloatLib.Floats.ExecFloat.Binary.toRat? rounded with
    | some value => pure value
    | none => throw "parameter quantization produced NaN or infinity"
  let weights ← IO.ofExcept (Tensor.mapM convert model.weights)
  let bias ← IO.ofExcept (Tensor.mapM convert model.bias)
  let rounded : Spec.LinearSpec Rat 64 10 := ⟨weights, bias⟩
  let deltaW : Tensor Rat [10, 64] := model.weights - weights
  let deltaB : Tensor Rat [10] := model.bias - bias
  let mut correct := 0
  let mut agreement := 0
  let mut certified := 0
  let mut totalError : Rat := 0
  let mut records : Array Json := #[]
  for index in [0:samples.size] do
    let (input, label) := samples[index]!
    let full := Spec.linearSpec model input
    let approximate := Spec.linearSpec rounded input
    let error : Tensor Rat [10] := Tensor.matvec deltaW input + deltaB
    unless Tensor.foldl (fun ok x => ok && x == 0) true (full - approximate - error) do
      throw <| IO.userError "parameter perturbation identity failed"
    let winner : Fin 10 := Fin.cast (by decide)
      ((Metrics.argmax? approximate).getD ⟨0, by decide⟩)
    let original : Fin 10 := Fin.cast (by decide)
      ((Metrics.argmax? full).getD ⟨0, by decide⟩)
    let bound := fun j : Fin 10 => |deltaB.getScalar winner - deltaB.getScalar j| +
      ∑ i : Fin 64, |deltaW[(winner, i)] - deltaW[(j, i)]| * |input.getScalar i|
    let checked := (Array.finRange 10).all fun j => j == winner || decide
      (bound j < approximate.getScalar winner - approximate.getScalar j)
    unless (Array.finRange 10).all (fun j => decide
        (|error.getScalar winner - error.getScalar j| ≤ bound j)) do
      throw <| IO.userError "quantized logit-gap bound failed"
    unless !checked || winner == original do
      throw <| IO.userError "a certified quantized prediction changed"
    if winner.val == label then correct := correct + 1
    if winner == original then agreement := agreement + 1
    if checked then certified := certified + 1
    totalError := totalError + Tensor.sum (Tensor.square (full - approximate))
    records := records.push <| Json.mkObj [
      ("index", toJson (index + 120)), ("prediction", toJson winner.val),
      ("full_prediction", toJson original.val), ("label", toJson label),
      ("certificate", toJson checked)]
  return Json.mkObj [
    ("exponent_bits", toJson format.expWidth), ("fraction_bits", toJson format.fracWidth),
    ("encoding", toJson "IEEE"), ("rounding", toJson "nearest-even"),
    ("scope", toJson "weights and biases rounded once; exact rational forward arithmetic"),
    ("correct", toJson correct), ("agreement", toJson agreement),
    ("certified", toJson certified),
    ("logit_mse", rational (totalError / (samples.size * 10 : Nat))),
    ("weights", toJson ((Tensor.to weights (Array Rat)).map toString)),
    ("bias", toJson ((Tensor.to bias (Array Rat)).map toString)), ("examples", .arr records)]

def run (weights dataset : String) : IO Json := do
  let (model, samples) ← load weights dataset
  let calibration := samples.extract 0 120
  let evaluation := samples.extract 120 samples.size
  let started ← IO.monoNanosNow
  let (mean, gram, magnitude) ← IO.lazyPure fun _ => fit model calibration
  let removal ← IO.lazyPure fun _ => greedy gram
  let sorted := fun score : Tensor Rat [64] =>
    (Array.finRange 64).qsort fun i j => decide (score.getScalar i < score.getScalar j)
  let orders := #[
    ("magnitude", sorted magnitude),
    ("variance", sorted (Tensor.ofFn fun i : Fin 64 => gram[(i, i)])),
    ("greedy", removal)]
  let fitted ← IO.monoNanosNow
  let mut results := #[]
  for (method, order) in orders do
    for retained in #[48, 32, 16] do
      results := results.push <| Json.mkObj [
        ("method", toJson method), ("retained", toJson retained),
        ("evaluation", ← evaluate model mean evaluation (select order retained))]
  let baseline := evaluation.foldl (fun correct (input, label) =>
    if ((Metrics.argmax? (Spec.linearSpec model input)).getD ⟨0, by decide⟩).val == label
      then correct + 1 else correct) 0
  let formats := #[FloatLib.Floats.ExecFloat.Binary.format 5 10,
    FloatLib.Floats.ExecFloat.Binary.format 8 7, FloatLib.Floats.ExecFloat.Binary.format 4 3]
  let rounded ← formats.mapM fun format => quantization format model evaluation
  return Json.mkObj [
    ("weights_sha256", toJson (← sha256 weights)),
    ("dataset_sha256", toJson (← sha256 dataset)),
    ("calibration_examples", toJson calibration.size),
    ("evaluation_examples", toJson evaluation.size), ("baseline_correct", toJson baseline),
    ("fit_ns", toJson (fitted - started)), ("results", .arr results),
    ("parameter_quantization", .arr rounded)]

end DigitsExperiment

def main (arguments : List String) : IO UInt32 := do
  match arguments with
  | [weights, dataset] =>
      try
        IO.println (← DigitsExperiment.run weights dataset).compress
        return 0
      catch error =>
        (← IO.getStderr).putStrLn error.toString
        return 1
  | _ =>
      (← IO.getStderr).putStrLn "usage: digits-experiment WEIGHTS.json DATASET.json"
      return 1
