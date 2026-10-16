import BatchInvariantInference.Core
import KimiK3.FeedForward.Activation
import KimiK3.GraphSpec.Expert
import KimiK3.Sequence.KDA
import Nemotron.SSM
import NN.API
import NN.Tensor
import NN.Kernel
import NN.Tactic.Autograd
import NN.Proofs.Tensor.Algebra
import TorchLeanGPT.TrainingCorrectness
import VerifiableTransformers.Certificate.FiniteEval

/-! Small runnable calculations embedded in the articles. The website records Lean's types
and command output from this file; editing a calculation requires refreshing that recording. -/

-- ANCHOR: rounding
open BatchInvariantInference.IEEE32
open FloatLib.Floats

-- Use the same binary32 bit patterns as the non-associativity theorem.
#eval ExecFloat.add (ExecFloat.add big negBig) one
#eval ExecFloat.add big (ExecFloat.add negBig one)
#check BatchInvariantInference.IEEE32.add_nonassoc_witness
-- ANCHOR_END: rounding

-- ANCHOR: prompts
open VerifiableTransformers.Certificate.FiniteEval

-- The first quote prompt and the first bracket prompt in the finite task.
#eval expectedInput 0
#eval expectedTarget 0
#eval expectedInput 128
#eval expectedTarget 128
#check CandidateEval.matchesExpectedIndex
-- ANCHOR_END: prompts

-- ANCHOR: resume
open TorchLeanGPT.Training

-- A small indexed update makes a reset of the global step visible.
def addStep (step value : Nat) : Nat := value + step + 1

#eval runSteps addStep 0 4 0
#eval runSteps addStep 2 2 (runSteps addStep 0 2 0)
#check runSteps_add
-- ANCHOR_END: resume

-- ANCHOR: cap
open KimiK3.SiTU

-- Float gives us a runnable value; the real-valued theorem quantifies over every input.
#eval softCap (4 : Float) 2
#eval scalar (4 : Float) 25 2 6
#check softCap
#check paper_caps_bound
-- ANCHOR_END: cap

-- ANCHOR: kda
open TorchLean

-- One head with two key coordinates and one value coordinate.
def memory : KimiK3.KDA.State Float32 2 1 := [[0], [0]]
def token : KimiK3.KDA.Input Float32 2 1 :=
  { query := [1, 0]
    key := [1, 0]
    value := [3]
    retention := [1, 1]
    writeStrength := 1 }

-- Read the first output coordinate.
#eval (KimiK3.KDA.step memory token).2.getScalar ⟨0, by decide⟩
#check KimiK3.KDA.step
-- ANCHOR_END: kda

-- ANCHOR: ssm
open TorchLean

-- A rank-one write updates a two-coordinate state, then reads its first coordinate.
def ssmMemory : Nemotron.SSM.State Float32 1 2 := [[10, 0]]
def ssmToken : Nemotron.SSM.Input Float32 1 2 :=
  { retention := 1
    value := [2]
    write := [1, 0]
    read := [1, 0] }

#eval (Nemotron.SSM.step ssmMemory ssmToken).2.getScalar ⟨0, by decide⟩
#check Nemotron.SSM.step
-- ANCHOR_END: ssm

-- ANCHOR: expert
open TorchLean

namespace ExpertExample

-- Two input coordinates, two hidden coordinates, and one output coordinate.
noncomputable def expert : KimiK3.Expert ℝ 2 2 1 :=
  { gateWeight := [[1, 0], [0, 1]]
    upWeight := [[2, 0], [0, 3]]
    downWeight := [[1], [-1]] }

noncomputable def input : Tensor ℝ [2] := [1, 2]

-- Instantiate the existing theorem with these weights and the paper's caps.
example :
    (KimiK3.GraphSpec.Expert.model 2 2 1).specFwd
        (KimiK3.GraphSpec.Expert.parameters expert)
        (KimiK3.GraphSpec.Expert.inputs input 4 25) =
      expert.forward 4 25 input := by
  exact KimiK3.GraphSpec.Expert.specFwd_eq_forward expert input 4 25

end ExpertExample
-- ANCHOR_END: expert

-- ANCHOR: expertBound
open TorchLean
open scoped BigOperators

example {d h o : Nat} (expert : KimiK3.Expert ℝ d h o)
    (input : Tensor ℝ [d]) (k : Fin o) :
    |(expert.forward 4 25 input).getScalar k| ≤
      100 * ∑ j : Fin h, |Spec.get2 expert.downWeight j k| := by
  let z := KimiK3.SiTU.vector 4 25
    (Spec.vecMatMulSpec input expert.gateWeight)
    (Spec.vecMatMulSpec input expert.upWeight)
  have hz (j : Fin h) : |z.getScalar j| ≤ 100 := by
    simpa [z, KimiK3.SiTU.vector, Tensor.stackLeading] using
      KimiK3.SiTU.paper_caps_bound
        ((Spec.vecMatMulSpec input expert.gateWeight).getScalar j)
        ((Spec.vecMatMulSpec input expert.upWeight).getScalar j)
  change |(Spec.vecMatMulSpec z expert.downWeight).getScalar k| ≤ _
  rw [Proofs.TensorAlgebra.getScalar_vec_mat_mul_spec]
  calc
    |∑ j : Fin h, z.getScalar j * Spec.get2 expert.downWeight j k| ≤
        ∑ j : Fin h, |z.getScalar j * Spec.get2 expert.downWeight j k| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j : Fin h, 100 * |Spec.get2 expert.downWeight j k| := by
      apply Finset.sum_le_sum
      intro j _
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_right (hz j) (abs_nonneg _)
    _ = 100 * ∑ j : Fin h, |Spec.get2 expert.downWeight j k| := by
      rw [Finset.mul_sum]
-- ANCHOR_END: expertBound

-- ANCHOR: tensorFunction
open TorchLean

def square {α : Type} [Mul α] (x : α) : α := x * x
def input : Tensor Float32 [3] := [1, 2, 3]

-- The device belongs to the call, so the function stays unchanged.
def onGpu : IO (Tensor Float32 [3]) := square.run input (device := gpu)
#eval square.run input (device := cpu)
-- ANCHOR_END: tensorFunction

-- ANCHOR: tensorGradient
open TorchLean

def energy (x : Tensor Float32 [3]) : Float32 := Tensor.sum (x * x)
def trainingInput : Tensor Float32 [3] := [1, 2, 3]

-- The default output seed is one: differentiate the scalar energy.
#eval do
  let result ← energy.run trainingInput (grad := true)
  IO.println result.value
  IO.println (← result.backward)

def gpuEnergy := energy.run trainingInput (device := gpu) (grad := true)
-- ANCHOR_END: tensorGradient

-- ANCHOR: tensorRecurrence
open TorchLean

-- Keep the element type general; the step needs multiplication and addition.
def evolve {α : Type} [One α] [Mul α] [Add α] (x : α) : Nat → α
  | 0 => x
  | n + 1 =>
      let previous := evolve x n
      previous * previous + 1

def starts : Tensor Float32 [2] := [1, 2]
def evolution (steps : UInt64) (device : NN.Backend.Device := cpu) :=
  (fun x => evolve x steps.toNat).run starts (device := device)

#eval evolution 2

-- Recording visits both updates, so backward includes both local derivatives.
#eval do
  let result ← (fun (x : Float32) => evolve x 2).run starts (grad := true)
  IO.println (← result.backward)
-- ANCHOR_END: tensorRecurrence

-- ANCHOR: tensorRows
open TorchLean NN.Kernel
open scoped NN.Kernel

def rowSum (width : UInt64) : Program Float32 :=
  Program.of (fun (read : Reader Float32) (row : UInt64) =>
    iterate (fun column acc => do
      let x ← read 0 (row * width + column)
      pure (acc + x)) width.toNat 0 0)

def matrix : Tensor Float32 [2, 3] := [[1, 2, 3], [4, 5, 6]]
#eval (rowSum 3).run (Arguments.empty.push matrix) [2]

def gpuRows : IO (Tensor Float32 [2]) :=
  (rowSum 3).run (Arguments.empty.push matrix) [2] (device := gpu)

#check Program.eval_eq_reference
#check Program.lower_correct
-- ANCHOR_END: tensorRows

-- ANCHOR: tensorPrecision
open TorchLean
open FloatLib.Floats

namespace PrecisionExample

abbrev Wide := ExecFloat.Binary 15 112

-- The same generic square as the first example.
def square {α : Type} [Mul α] (x : α) : α := x * x

-- Construct the extra digit from a rational, without passing through binary64.
def a : Wide := Rat.cast (1 + 1 / (2 ^ 100 : Nat) : Rat)
def wideInput : Tensor Wide [1] := [a]
def gpuWide : IO (Tensor Wide [1]) := square.run wideInput (device := gpu)

#eval do
  let output ← square.run wideInput
  pure (ExecFloat.Binary.toRat? output[0] == some (1 + 1 / (2 ^ 99 : Nat) : Rat))

end PrecisionExample
-- ANCHOR_END: tensorPrecision

-- ANCHOR: tensorProof
open TorchLean

-- A real derivative statement uses the existing calculus tactic.
example (x : ℝ) : HasDerivAt (fun x : ℝ => x * x) (2 * x) x := by
  autograd
-- ANCHOR_END: tensorProof

-- ANCHOR: tensorBlend
open TorchLean

def blend (weight left right : Float32) : Float32 :=
  (1 - weight) * left + weight * right

def left : Tensor Float32 [3] := [0, 4, 8]
def right : Tensor Float32 [3] := [8, 4, 0]

#eval (blend 0.25).zip left right
def gpuBlend : IO (Tensor Float32 [3]) :=
  (blend 0.25).zip left right (device := gpu)
-- ANCHOR_END: tensorBlend

-- ANCHOR: tensorRelaxation
open TorchLean

-- Move a fraction of the remaining distance to the target at each step.
def relax {α : Type} [Add α] [Sub α] [Mul α] (rate target x : α) : Nat → α
  | 0 => x
  | n + 1 =>
      let previous := relax rate target x n
      previous + rate * (target - previous)

def initial : Tensor Float32 [3] := [0, 2, 4]
def approach (steps : UInt64) (device : NN.Backend.Device := cpu) :
    IO (Tensor Float32 [3]) :=
  (fun x => relax (0.5 : Float32) 2 x steps.toNat).run initial (device := device)

#eval approach 3

-- The real error is multiplied by the same factor at every step.
example (rate target x : ℝ) (steps : Nat) :
    relax rate target x steps = target + (1 - rate) ^ steps * (x - target) := by
  induction steps with
  | zero => simp [relax]
  | succ steps ih =>
      rw [relax, ih, pow_succ]
      ring
-- ANCHOR_END: tensorRelaxation

-- ANCHOR: tensorLoss
open TorchLean

def targets : Tensor Float32 [3] := [0, 1, 1]
def predictions : Tensor Float32 [3] := [1, 2, 3]
def loss (prediction : Tensor Float32 [3]) : Float32 :=
  let residual := prediction - targets
  Tensor.sum (residual * residual)

-- Optimize the predictions directly to make this one update easy to inspect.
#eval do
  let result ← loss.run predictions (grad := true)
  let gradient ← result.backward
  let next := predictions - Tensor.full [3] 0.25 * gradient
  IO.println result.value
  IO.println gradient
  IO.println next
  IO.println (loss next)

def gpuLoss := loss.run predictions (device := gpu) (grad := true)

-- This is the real derivative of one loss coordinate, with its target fixed.
example (target x : ℝ) :
    HasDerivAt (fun x : ℝ => (x - target) * (x - target)) (2 * (x - target)) x := by
  autograd
-- ANCHOR_END: tensorLoss

-- ANCHOR: tensorContraction
open TorchLean
open TorchLean.Tensor

namespace ContractionExample

def inputs : Tensor Float32 [2, 3] := [[1, 2, 3], [4, 5, 6]]
def weights : Tensor Float32 [3, 2] := [[1, 2], [3, 4], [5, 6]]

-- Keep row and output; multiply and sum along feature.
def projected := einsum inputs, weights "row feature, feature output -> row output"

-- Both operands read the same row; only feature is summed out.
def squaredNorms := einsum inputs, inputs "row feature, row feature -> row"

#check projected
#eval projected
#eval squaredNorms

end ContractionExample
-- ANCHOR_END: tensorContraction

-- ANCHOR: tensorPatterns
open TorchLean
open TorchLean.Tensor

namespace ShapeExample

def matrix : Tensor Float32 [2, 3] := [[1, 2, 3], [4, 5, 6]]
def bias : Tensor Float32 [3] := [10, 20, 30]

-- Move coordinates, rather than regrouping the unchanged flat payload.
def transposed := rearrange matrix "row column -> column row"
def flattened := rearrange matrix "row column -> (row column)"

-- Explicitly give each row the same three bias values.
def shifted := matrix + (expand bias "column -> row column" with row := 2)
def rowSums := reduce matrix "row column -> row" by sum

#eval transposed
#eval flattened
#eval shifted
#eval rowSums

end ShapeExample
-- ANCHOR_END: tensorPatterns

-- ANCHOR: tensorAttention
open TorchLean
open TorchLean.Tensor

namespace AttentionExample

-- One batch, two heads, two token positions, two features per head.
def queries : Tensor Float32 [1, 2, 2, 2] :=
  [[[[1, 0], [0, 1]], [[1, 1], [2, 1]]]]
def keys : Tensor Float32 [1, 2, 2, 2] :=
  [[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]]
def scores := einsum queries, keys
  "batch head query feature, batch head key feature -> batch head query key"

-- Supply normalized weights to isolate the value contraction.
-- A complete attention layer computes these using scaling, masking, and softmax.
def weights : Tensor Float32 [1, 2, 2, 2] :=
  [[[[0.75, 0.25], [0.25, 0.75]], [[0.5, 0.5], [1, 0]]]]
def values : Tensor Float32 [1, 2, 2, 1] := [[[[10], [20]], [[30], [50]]]]
def attended := einsum weights, values
  "batch head query key, batch head key channel -> batch head query channel"
def joined := rearrange attended "batch head query channel -> batch query (head channel)"

#check scores
#eval scores
#eval attended
#eval joined

end AttentionExample
-- ANCHOR_END: tensorAttention

-- ANCHOR: tensorModel
open TorchLean

def model :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

def trainer := Trainer.new model
  { objective := Trainer.Objective.mse
    optimizer := optim.adam { learningRate := 0.03 }
    device := cpu }
-- ANCHOR_END: tensorModel

-- ANCHOR: tensorChainRule
example (x : ℝ) : HasDerivAt (fun y : ℝ => Real.exp (y * y))
    (2 * x * Real.exp (x * x)) x := by
  autograd
-- ANCHOR_END: tensorChainRule
