/-
TorchLean training/export command for a small native causal GPT.

The upstream checkpoint verified elsewhere in this example uses sparsemax,
Signed-L1-BandNorm, and LeakyReLU. This command has a different purpose: it runs
the same finite quote/bracket task through TorchLean's public GPT-2-style API.
That path uses standard causal softmax attention and therefore reaches the
fused CUDA attention forward and VJP introduced in TorchLean 4.32.

The dimensions and finite task still mirror `scripts/small/config.py` and
`scripts/small/dataset.py`. The command can export its own checkpoint and the
same 256-row projected eval-trace schema consumed by the Lean checker.

What we prove about this path is the exported trace contract: the checkpoint
summary has the expected shape/hash metadata, and the finite eval rows satisfy
the same projected quote/bracket property. Neel's exact sparsemax checkpoint is
checked independently by the generated Lean constants and
`Replay/UpstreamFloatReplay`; the two parameter layouts are intentionally not
identified.
-/

import NN.API
import NN.API.Models.Gpt2

open Spec Tensor
open NN.API

namespace VerifiableTransformers.TorchLean.TrainSmallGPT

/-- Lake executable name for the TorchLean-side training and trace command. -/
def exeName : String := "train_torchlean_small_gpt"

/-- Minibatch size used when cycling over the 256 finite prompts. -/
def batch : Nat := 16
/-- Fixed prompt length for both quote and bracket tasks. -/
def seqLen : Nat := 6
/-- Custom vocabulary size from Neel's `scripts/small/vocab.py`. -/
def vocab : Nat := 32
/-- Neel's small configuration uses one attention head. -/
def numHeads : Nat := 1
/-- Head dimension; with one head this equals `dModel`. -/
def headDim : Nat := 16
/-- Hidden width from `SmallVerifiableConfig`. -/
def dModel : Nat := numHeads * headDim
/-- MLP hidden width from `SmallVerifiableConfig`. -/
def ffnHidden : Nat := 64
/-- Number of transformer blocks in the small GPT. -/
def layers : Nat := 2

abbrev OneHotShape : Shape := .dim batch (.dim seqLen (.dim vocab .scalar))

local instance : NeZero seqLen := ⟨by decide⟩
local instance : NeZero dModel := ⟨by decide⟩

/-- Content token at a canonical content index. -/
def contentToken (i : Nat) : Nat :=
  match i % 4 with
  | 0 => 5
  | 1 => 6
  | 2 => 7
  | _ => 8

/-- Canonical quote-task prompt for a global finite-domain index. -/
def quoteWindow (idx : Nat) : List Nat :=
  let core := idx / 2
  let x1 := contentToken (core / 16)
  let x2 := contentToken ((core / 4) % 4)
  let x3 := contentToken (core % 4)
  let quote := if idx % 2 = 0 then 9 else 10
  [1, 2, x1, quote, x2, x3, quote]

/-- Canonical bracket-task prompt for a global finite-domain index. -/
def bracketWindow (idx : Nat) : List Nat :=
  let core := idx / 2
  let x1 := contentToken (core / 16)
  let x2 := contentToken ((core / 4) % 4)
  let x3 := contentToken (core % 4)
  let openClose := if idx % 2 = 0 then (11, 13) else (12, 14)
  [1, 3, x1, openClose.1, x2, x3, openClose.2]

/-- Canonical task prompt for a global finite-domain index. -/
def taskWindow (idx : Nat) : List Nat :=
  if idx < 128 then quoteWindow idx else bracketWindow (idx - 128)

/-- Symbolic target token for a global finite-domain index. -/
def taskTarget (idx : Nat) : Nat :=
  (taskWindow idx).getD seqLen 0

/-- Build one one-hot minibatch starting at `offset` in the 256-row finite domain. -/
def mkSample {α : Type} [_root_.TorchLean.Runtime.SemanticScalar α]
    [_root_.TorchLean.Runtime.Scalar α] (offset : Nat) :
    _root_.TorchLean.SupervisedSample α OneHotShape OneHotShape :=
  let xF : Tensor Float OneHotShape :=
    Tensor.dim (fun bi =>
      Tensor.dim (fun t =>
        text.oneHotTokenFloat vocab ((taskWindow ((offset + bi.val) % 256)).getD t.val 0)))
  let yF : Tensor Float OneHotShape :=
    Tensor.dim (fun bi =>
      Tensor.dim (fun t =>
        if t.val = seqLen - 1 then
          text.oneHotTokenFloat vocab (taskTarget ((offset + bi.val) % 256))
        else
          Spec.fill (α := Float) 0.0 (.dim vocab .scalar)))
  _root_.TorchLean.Sample.mk
    (Common.castTensor Runtime.ofFloat xF) (Common.castTensor Runtime.ofFloat yF)

/-- First sample, used for before/after loss reporting. -/
def firstSample {α : Type} [_root_.TorchLean.Runtime.SemanticScalar α]
    [_root_.TorchLean.Runtime.Scalar α] :
    _root_.TorchLean.SupervisedSample α OneHotShape OneHotShape :=
  mkSample (α := α) 0

/-- Extract the vocabulary logits for one `(batch,row)` and sequence position. -/
def logitsArrayAt (logits : Tensor Float OneHotShape) (row : Fin batch) (pos : Fin seqLen) :
    Array Float :=
  match logits with
  | Tensor.dim batches =>
      match batches row with
      | Tensor.dim rows =>
          match rows pos with
          | Tensor.dim cols =>
              Array.ofFn (fun j : Fin vocab =>
                match cols j with
                | Tensor.scalar x => x)

/-- Choose the better of two projected candidates. -/
def bestOfCandidates (scores : Array Float) (a b : Nat) : Nat :=
  if scores.getD a 0.0 >= scores.getD b 0.0 then a else b

/-- Count scalars across a TorchLean parameter list. -/
def paramElementCount : {ss : List Shape} → TorchLean.TList Float ss → Nat
  | [], .nil => 0
  | s :: ss, .cons _t ts => Shape.size s + paramElementCount (ss := ss) ts

/-- Flatten the first `limit` scalars for inspection without dumping the checkpoint. -/
def paramFlatPrefix (limit : Nat) :
    {ss : List Shape} → TorchLean.TList Float ss → List Float
  | [], .nil => []
  | _s :: ss, .cons t ts =>
      let here := (Spec.toList t).take limit
      if _h : here.length < limit then
        here ++ paramFlatPrefix (limit - here.length) (ss := ss) ts
      else
        here

/-- Print a compact parameter summary for reproducibility checks. -/
def printParamInspection {ss : List Shape} (ps : TorchLean.TList Float ss) : IO Unit := do
  IO.println s!"trained_parameter_tensors={ss.length}"
  IO.println s!"trained_parameter_scalars={paramElementCount ps}"
  IO.println s!"first_16_trained_weights={paramFlatPrefix 16 ps}"

/-- Evaluate projected quote/bracket accuracy over exhaustive minibatches. -/
def evalCandidateAccuracy
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential OneHotShape OneHotShape)
    (params : TorchLean.ParamList Float (nn.paramShapes model))
    (evalBatches : Nat) : IO Float := do
  let mut correct : Nat := 0
  let batches := Nat.min 16 evalBatches
  for k in [0:batches] do
    let offset := (k * batch) % 256
    let sample := mkSample (α := Float) offset
    let logits ← _root_.TorchLean.nn.predict (α := Float) model opts params
      (_root_.TorchLean.Sample.x sample)
    for bi in [0:batch] do
      if hbi : bi < batch then
        let global := (offset + bi) % 256
        let row : Fin batch := ⟨bi, hbi⟩
        let pos : Fin seqLen := ⟨seqLen - 1, by decide⟩
        let scores := logitsArrayAt logits row pos
        let pred :=
          if global < 128 then
            bestOfCandidates scores 9 10
          else
            bestOfCandidates scores 13 14
        if pred = taskTarget global then
          correct := correct + 1
  let total := batches * batch
  if total = 0 then
    pure 0.0
  else
    pure (Float.ofNat correct / Float.ofNat total)

/-- Competing token for the two-way projected finite task. -/
def alternateTarget (idx target : Nat) : Nat :=
  if idx < 128 then
    if target = 9 then 10 else 9
  else
    if target = 13 then 14 else 13

/-- Human-readable task family name for eval trace rows. -/
def taskName (idx : Nat) : String :=
  if idx < 128 then "quoteClose" else "bracketType"

/-- JSON array for the prompt prefix used by one finite-domain row. -/
def inputPrefixString (idx : Nat) : String :=
  let xs := (taskWindow idx).take seqLen
  "[" ++ String.intercalate ", " (xs.map toString) ++ "]"

/-- Serialize one projected target/alternate score row. -/
def evalRowJson (idx target alternate : Nat) (targetScore alternateScore : Float) : String :=
  "  {\"task\":\"" ++ taskName idx ++
    "\",\"input\":" ++ inputPrefixString idx ++
    ",\"target\":" ++ toString target ++
    ",\"alternate\":" ++ toString alternate ++
    ",\"targetScore\":" ++ toString targetScore ++
    ",\"alternateScore\":" ++ toString alternateScore ++ "}"

/-- Collect the finite-domain projected score trace from the current TorchLean parameters. -/
def collectEvalRowsJson
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential OneHotShape OneHotShape)
    (params : TorchLean.ParamList Float (nn.paramShapes model)) : IO (List String) := do
  let mut rows : List String := []
  for k in [0:16] do
    let offset := (k * batch) % 256
    let sample := mkSample (α := Float) offset
    let logits ← _root_.TorchLean.nn.predict (α := Float) model opts params
      (_root_.TorchLean.Sample.x sample)
    for bi in [0:batch] do
      if hbi : bi < batch then
        let global := (offset + bi) % 256
        let row : Fin batch := ⟨bi, hbi⟩
        let pos : Fin seqLen := ⟨seqLen - 1, by decide⟩
        let scores := logitsArrayAt logits row pos
        let target := taskTarget global
        let alternate := alternateTarget global target
        let targetScore := scores.getD target 0.0
        let alternateScore := scores.getD alternate 0.0
        rows := rows ++ [evalRowJson global target alternate targetScore alternateScore]
  pure rows

/-- Write a JSON trace that later generators convert into Lean certificate data. -/
def writeEvalJson
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential OneHotShape OneHotShape)
    (params : TorchLean.ParamList Float (nn.paramShapes model))
    (path : System.FilePath)
    (checkpointPath : System.FilePath) : IO Unit := do
  let rows ← collectEvalRowsJson opts model params
  let body :=
    "{\n" ++
    " \"producer\":\"train_torchlean_small_gpt\",\n" ++
    " \"model\":\"torchlean-causal-softmax-gpt-v1\",\n" ++
    " \"dimensions\":{\"batch\":" ++ toString batch ++
      ",\"seqLen\":" ++ toString seqLen ++
      ",\"vocab\":" ++ toString vocab ++
      ",\"dModel\":" ++ toString dModel ++
      ",\"layers\":" ++ toString layers ++
      ",\"heads\":" ++ toString numHeads ++
      ",\"ffnHidden\":" ++ toString ffnHidden ++ "},\n" ++
    " \"checkpointPath\":\"" ++ checkpointPath.toString ++ "\",\n" ++
    " \"scoreScale\":1000000,\n" ++
    " \"rows\":[\n" ++ String.intercalate ",\n" rows ++ "\n ]\n}\n"
  IO.FS.writeFile path body

/-- Public TorchLean configuration used by the native training path. -/
def modelConfig : nn.models.CausalOneHotConfig where
  batch := batch
  seqLen := seqLen
  vocab := vocab
  numHeads := numHeads
  headDim := headDim
  ffnHidden := ffnHidden
  layers := layers
  seedStride := 1000

/--
Native TorchLean causal GPT.

The public constructor supplies learned token and positional embeddings, a structurally causal
boolean mask, Transformer blocks, final normalization, and the language-model head. On CUDA its
attention nodes select TorchLean's fused attention capsule and fused VJP.
-/
def mkTrainableModel : nn.M (nn.Sequential OneHotShape OneHotShape) :=
  nn.models.causalTransformerOneHot modelConfig

/-- Print the run configuration before training or trace generation. -/
def printRunHeader (mode : String) (steps evalBatches : Nat) : IO Unit := do
  IO.println s!"TorchLean GPT dimensions: vocab={vocab}, seqLen={seqLen}, dModel={dModel}, layers={layers}, heads={numHeads}, dMlp={ffnHidden}"
  IO.println "operators: hard-masked softmax attention + LayerNorm + GELU + affine lm_head"
  IO.println s!"mode={mode}"
  IO.println s!"steps={steps}"
  IO.println s!"eval_batches={Nat.min 16 evalBatches} (out of 16 exhaustive minibatches)"

/-- Instantiate, optionally train, and optionally export parameters/eval traces. -/
def trainModel
    (opts : Runtime.Autograd.Torch.Options) (steps evalBatches : Nat)
    (instantiateOnly : Bool) (saveParams? : Option System.FilePath)
    (saveEvalJson? : Option System.FilePath) (inspectWeights : Bool) :
    IO Unit := do
  nn.withModel mkTrainableModel fun model => do
    let modDef := nn.crossEntropyOneHotScalarModuleDef model (reduction := .mean)
    let m ← _root_.TorchLean.Module.instantiate (α := Float) opts modDef id
    if instantiateOnly then
      printRunHeader "instantiate-only" steps evalBatches
      IO.println "instantiated trainable TorchLean GPT module; skipped forward/backward by request"
      let ps ← _root_.Runtime.Autograd.Torch.ParamList.valuesSynced (α := Float)
        (ss := nn.paramShapes model) m.trainer.params
      if inspectWeights then
        printParamInspection ps
      if let some path := saveParams? then
        _root_.NN.API.TorchLean.ParamIO.saveParamBits path ps
        IO.println s!"saved initial parameters to {path}"
      if let some path := saveEvalJson? then
        writeEvalJson opts model m.trainer.params path (saveParams?.getD "initial-params-not-saved")
        IO.println s!"saved initial eval trace to {path}"
      return ()
    let sample0 := firstSample (α := Float)
    let loss0 ← m.forward sample0
    let acc0 ← evalCandidateAccuracy opts model m.trainer.params evalBatches
    let opt := TorchLean.Optim.adam (α := Float)
      (paramShapes := nn.paramShapes model)
      (lr := 5e-4)
      (beta1 := 0.9)
      (beta2 := 0.999)
      (epsilon := 1e-8)
    let optH ← TorchLean.Optim.handle (α := Float) m opt
    for step in [0:steps] do
      optH.step (mkSample (α := Float) ((step * batch) % 256))
    let trainedParams ← _root_.Runtime.Autograd.Torch.ParamList.valuesSynced (α := Float)
      (ss := nn.paramShapes model) m.trainer.params
    let loss1 ← m.forward sample0
    let acc1 ← evalCandidateAccuracy opts model m.trainer.params evalBatches
    if let some path := saveParams? then
      _root_.NN.API.TorchLean.ParamIO.saveParamBits path trainedParams
    if let some path := saveEvalJson? then
      writeEvalJson opts model m.trainer.params path (saveParams?.getD "trained-params-not-saved")
    printRunHeader "train" steps evalBatches
    IO.println s!"loss0={Tensor.toScalar loss0} loss1={Tensor.toScalar loss1}"
    IO.println s!"candidate_accuracy0={acc0} candidate_accuracy1={acc1}"
    if inspectWeights then
      printParamInspection trainedParams
    if let some path := saveParams? then
      IO.println s!"saved trained parameters to {path}"
    if let some path := saveEvalJson? then
      IO.println s!"saved TorchLean eval trace to {path}"

/-- Parse flags for training, instantiation, parameter export, and eval trace export. -/
def parseRunFlags (args : List String) :
    Except String (Nat × Nat × Bool × Option System.FilePath × Option System.FilePath × Bool × List String) := do
  let (steps?, rest) ← _root_.TorchLean.CLI.takeNatFlagOnce args "steps"
  let (evalBatches?, rest) ← _root_.TorchLean.CLI.takeNatFlagOnce rest "eval-batches"
  let (instantiateOnly, rest) ← _root_.TorchLean.CLI.takeBoolFlagOnce rest "instantiate-only"
  let (saveParams?, rest) ← _root_.TorchLean.CLI.takePathFlagOnce rest "save-params"
  let (saveEvalJson?, rest) ← _root_.TorchLean.CLI.takePathFlagOnce rest "save-eval-json"
  let (inspectWeights, rest) ← _root_.TorchLean.CLI.takeBoolFlagOnce rest "inspect-weights"
  pure (steps?.getD 10, evalBatches?.getD 1, instantiateOnly, saveParams?, saveEvalJson?, inspectWeights, rest)

/-- Help text for the command-specific flags layered on top of TorchLean's runtime flags. -/
def usage : String :=
  _root_.NN.API.TorchLean.Module.runUsage exeName ++ "\n\n" ++
  String.intercalate "\n"
    [ "Small GPT flags:"
    , "  --steps N                 optimizer steps (default: 10)"
    , "  --eval-batches N          finite-domain minibatches to evaluate (default: 1, max: 16)"
    , "  --instantiate-only        initialize the model without forward/backward"
    , "  --inspect-weights         print parameter counts and the first 16 weights"
    , "  --save-params PATH        write exact Float parameter bits as JSON"
    , "  --save-eval-json PATH     write the finite projected-score trace"
    ]

/-- CLI entrypoint for TorchLean training, parameter export, and eval-trace export. -/
def main (args : List String) : IO UInt32 := do
  if _root_.TorchLean.CLI.hasHelp args then
    IO.println usage
    return 0
  _root_.NN.API.TorchLean.Module.run exeName args
    (.float (fun opts rest => do
      let (steps, evalBatches, instantiateOnly, saveParams?, saveEvalJson?, inspectWeights, rest) ←
        Common.orThrow exeName <| parseRunFlags rest
      _root_.TorchLean.CLI.requireNoArgs exeName rest
      trainModel opts steps evalBatches instantiateOnly saveParams? saveEvalJson? inspectWeights))
    { banner? := some (fun opts =>
        s!"{exeName}: TorchLean GPT task training (device={opts.device.cliName})")
      printOk := true }

end VerifiableTransformers.TorchLean.TrainSmallGPT

/-- Lake executable entrypoint. -/
def main (args : List String) : IO UInt32 :=
  VerifiableTransformers.TorchLean.TrainSmallGPT.main args
