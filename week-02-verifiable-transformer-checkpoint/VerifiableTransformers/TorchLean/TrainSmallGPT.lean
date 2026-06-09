/-
TorchLean training/export command for the small GPT architecture.

Here TorchLean trains a reproduction run with the same architecture: vocabulary
32, sequence length 6, width 16, two blocks, one sparsemax attention head, and a
64-wide MLP.
The command can export a CUDA checkpoint plus the same 256-row finite eval trace
that the Lean certificate checker understands.

The dimensions and finite task mirror `scripts/small/config.py` and
`scripts/small/dataset.py`.  The sparsemax, BandNorm, and LeakyReLU wiring
follows `scripts/small/train.py`.

What we prove about this path is the exported trace contract: the checkpoint
summary has the expected shape/hash metadata, and the finite eval rows satisfy
the same projected quote/bracket property. Neel's exact checkpoint is checked by
the generated Lean constants plus `Replay/UpstreamFloatReplay`; this native
TorchLean run is a reproduction path, not a claim of bitwise identity with
Neel's checkpoint.
-/

import NN.API.Models.Gpt2
import NN.Runtime.Autograd.TorchLean.ParamIO
import VerifiableTransformers.TorchLean.Ops

open Spec Tensor
open NN.API
open VerifiableTransformers.TorchLean.Ops

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

abbrev ModelShape : Shape := .dim batch (.dim seqLen (.dim dModel .scalar))
abbrev OneHotShape : Shape := .dim batch (.dim seqLen (.dim vocab .scalar))

local instance : NeZero seqLen := ⟨by decide⟩
local instance : NeZero dModel := ⟨by decide⟩

namespace TL

/-- Apply one row-major affine layer to each row of a matrix. -/
def linearRowsRef {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    {rows inDim outDim : Nat}
    (w : TorchLean.RefTy (m := m) (α := α) (.dim outDim (.dim inDim .scalar)))
    (b : TorchLean.RefTy (m := m) (α := α) (.dim outDim .scalar))
    (x : TorchLean.RefTy (m := m) (α := α) (.dim rows (.dim inDim .scalar))) :
    m (TorchLean.RefTy (m := m) (α := α) (.dim rows (.dim outDim .scalar))) := do
  let wt ← TorchLean.transpose2d (m := m) (α := α)
    (mDim := outDim) (nDim := inDim) w
  let y ← TorchLean.matmul (m := m) (α := α)
    (mDim := rows) (nDim := inDim) (pDim := outDim) x wt
  let bRows ← TorchLean.broadcastTo (m := m) (α := α)
    (s₁ := .dim outDim .scalar) (s₂ := .dim rows (.dim outDim .scalar))
    ((inferInstance : Shape.BroadcastTo (.dim outDim .scalar)
      (.dim rows (.dim outDim .scalar))).proof) b
  TorchLean.add (m := m) (α := α) (s := .dim rows (.dim outDim .scalar)) y bRows

/-- Row-wise linear map without a bias term, used for the final unembedding. -/
def linearRowsNoBiasRef {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [TorchLean.Ops (m := m) (α := α)]
    {rows inDim outDim : Nat}
    (w : TorchLean.RefTy (m := m) (α := α) (.dim outDim (.dim inDim .scalar)))
    (x : TorchLean.RefTy (m := m) (α := α) (.dim rows (.dim inDim .scalar))) :
    m (TorchLean.RefTy (m := m) (α := α) (.dim rows (.dim outDim .scalar))) := do
  let wt ← TorchLean.transpose2d (m := m) (α := α)
    (mDim := outDim) (nDim := inDim) w
  TorchLean.matmul (m := m) (α := α)
    (mDim := rows) (nDim := inDim) (pDim := outDim) x wt

/-- Additive causal mask: zero on visible positions and a large negative value otherwise. -/
def causalMaskTensor (α : Type) [Context α] (batch seqLen : Nat) :
    Tensor α (.dim batch (.dim seqLen (.dim seqLen .scalar))) :=
  Tensor.dim (fun _ =>
    Tensor.dim (fun q =>
      Tensor.dim (fun k =>
        Tensor.scalar <|
          if k.val <= q.val then (0 : α) else -(((10000 : Nat) : α)))))

end TL

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
def mkSample {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] (offset : Nat) :
    sample.Supervised α OneHotShape OneHotShape :=
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
  sample.mk (Common.castTensor Runtime.ofFloat xF) (Common.castTensor Runtime.ofFloat yF)

/-- First sample, used for before/after loss reporting. -/
def firstSample {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] :
    sample.Supervised α OneHotShape OneHotShape :=
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
    let logits ← nn.eval1NoGrad (α := Float) opts model params (NN.API.sample.x sample)
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
    let logits ← nn.eval1NoGrad (α := Float) opts model params (NN.API.sample.x sample)
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
    " \"checkpointPath\":\"" ++ checkpointPath.toString ++ "\",\n" ++
    " \"scoreScale\":1000000,\n" ++
    " \"rows\":[\n" ++ String.intercalate ",\n" rows ++ "\n ]\n}\n"
  IO.FS.writeFile path body

/-- No-bias linear layer over a prefix shape, matching GPT-2's untied `lm_head`. -/
def linearNoBiasLayer (inDim outDim : Nat) (seedW : Nat := 0)
    (pfx : Shape := .scalar) :
    TorchLean.NN.LayerDef (pfx.appendDim inDim) (pfx.appendDim outDim) :=
  let WShape : Shape := .dim outDim (.dim inDim .scalar)
  let w0 : Tensor Float WShape := _root_.Runtime.Autograd.Torch.Init.xavierW
    (outDim := outDim) (inDim := inDim) (seed := seedW)
  let rows := Shape.size pfx
  { paramShapes := [WShape]
    initParams := TorchLean.tlist1 w0
    paramRequiresGrad := [true]
    forward := fun _mode {α} _ _ =>
      fun {m} _ _ =>
        fun w x =>
        ((do
          let sIn : Shape := pfx.appendDim inDim
          let sOut : Shape := pfx.appendDim outDim
          let xRows ← TorchLean.reshape (m := m) (α := α)
            (s₁ := sIn)
            (s₂ := .dim rows (.dim inDim .scalar))
            x (by simp [sIn, rows, Shape.size_appendDim, Shape.size])
          let yRows ← TL.linearRowsNoBiasRef (m := m) (α := α)
            (rows := rows) (inDim := inDim) (outDim := outDim) w xRows
          TorchLean.reshape (m := m) (α := α)
            (s₁ := .dim rows (.dim outDim .scalar))
            (s₂ := sOut)
            yRows (by simp [sOut, rows, Shape.size_appendDim, Shape.size])
        ) : m (TorchLean.RefTy (m := m) (α := α) (pfx.appendDim outDim)))
  }

/-- Sequential wrapper for the no-bias unembedding layer. -/
def linearNoBias (inDim outDim : Nat) (seedW : Nat := 0)
    (pfx : Shape := .scalar) :
    nn.Sequential (pfx.appendDim inDim) (pfx.appendDim outDim) :=
  nn.of (linearNoBiasLayer inDim outDim seedW (pfx := pfx))

/-- Sparsemax causal self-attention for one head (`dModel = headDim = 16`). -/
def sparsemaxCausalSelfAttentionLayer (batch seqLen dModel : Nat) (seedBase : Nat := 0) :
    TorchLean.NN.LayerDef
      (.dim batch (.dim seqLen (.dim dModel .scalar)))
      (.dim batch (.dim seqLen (.dim dModel .scalar))) :=
  let WShape : Shape := .dim dModel (.dim dModel .scalar)
  let bShape : Shape := .dim dModel .scalar
  let wq0 : Tensor Float WShape := _root_.Runtime.Autograd.Torch.Init.xavierW
    (outDim := dModel) (inDim := dModel) (seed := seedBase + 0)
  let bq0 : Tensor Float bShape := _root_.Runtime.Autograd.Torch.Init.tensor
    (s := bShape) (sch := .zeros) (seed := seedBase + 1)
  let wk0 : Tensor Float WShape := _root_.Runtime.Autograd.Torch.Init.xavierW
    (outDim := dModel) (inDim := dModel) (seed := seedBase + 2)
  let bk0 : Tensor Float bShape := _root_.Runtime.Autograd.Torch.Init.tensor
    (s := bShape) (sch := .zeros) (seed := seedBase + 3)
  let wv0 : Tensor Float WShape := _root_.Runtime.Autograd.Torch.Init.xavierW
    (outDim := dModel) (inDim := dModel) (seed := seedBase + 4)
  let bv0 : Tensor Float bShape := _root_.Runtime.Autograd.Torch.Init.tensor
    (s := bShape) (sch := .zeros) (seed := seedBase + 5)
  let wo0 : Tensor Float WShape := _root_.Runtime.Autograd.Torch.Init.xavierW
    (outDim := dModel) (inDim := dModel) (seed := seedBase + 6)
  let bo0 : Tensor Float bShape := _root_.Runtime.Autograd.Torch.Init.tensor
    (s := bShape) (sch := .zeros) (seed := seedBase + 7)
  let rows := batch * seqLen
  let s3 : Shape := .dim batch (.dim seqLen (.dim dModel .scalar))
  let sRows : Shape := .dim rows (.dim dModel .scalar)
  let sScores : Shape := .dim batch (.dim seqLen (.dim seqLen .scalar))
  { paramShapes := [WShape, bShape, WShape, bShape, WShape, bShape, WShape, bShape]
    initParams :=
      .cons wq0 (.cons bq0 (.cons wk0 (.cons bk0 (.cons wv0 (.cons bv0 (.cons wo0 (.cons bo0 .nil)))))))
    paramRequiresGrad := [true, true, true, true, true, true, true, true]
    forward := fun _mode {α} _ _ =>
      fun {m} _ _ =>
        fun wq bq wk bk wv bv wo bo x =>
        ((do
          let xRows ← TorchLean.reshape (m := m) (α := α)
            (s₁ := s3) (s₂ := sRows) x (by
              simp [s3, sRows, rows, Shape.size, Nat.mul_left_comm, Nat.mul_comm])
          let qRows ← TL.linearRowsRef (m := m) (α := α)
            (rows := rows) (inDim := dModel) (outDim := dModel) wq bq xRows
          let kRows ← TL.linearRowsRef (m := m) (α := α)
            (rows := rows) (inDim := dModel) (outDim := dModel) wk bk xRows
          let vRows ← TL.linearRowsRef (m := m) (α := α)
            (rows := rows) (inDim := dModel) (outDim := dModel) wv bv xRows
          let q ← TorchLean.reshape (m := m) (α := α)
            (s₁ := sRows) (s₂ := s3) qRows (by
              simp [s3, sRows, rows, Shape.size, Nat.mul_left_comm, Nat.mul_comm])
          let k ← TorchLean.reshape (m := m) (α := α)
            (s₁ := sRows) (s₂ := s3) kRows (by
              simp [s3, sRows, rows, Shape.size, Nat.mul_left_comm, Nat.mul_comm])
          let v ← TorchLean.reshape (m := m) (α := α)
            (s₁ := sRows) (s₂ := s3) vRows (by
              simp [s3, sRows, rows, Shape.size, Nat.mul_left_comm, Nat.mul_comm])
          let kt ← _root_.Runtime.Autograd.Torch.transpose3dLastTwo (m := m) (α := α)
            (a := batch) (b := seqLen) (c := dModel) k
          let rawScores ← _root_.Runtime.Autograd.Torch.bmm (m := m) (α := α)
            (batch := batch) (mDim := seqLen) (nDim := dModel) (pDim := seqLen) q kt
          let scaledScores ← TorchLean.scale (m := m) (α := α) (s := sScores)
            rawScores ((1 : α) / ((4 : Nat) : α))
          let causal ← TorchLean.const (m := m) (α := α) (s := sScores)
            (TL.causalMaskTensor α batch seqLen)
          let masked ← TorchLean.add (m := m) (α := α) (s := sScores) scaledScores causal
          let scoreRows ← TorchLean.reshape (m := m) (α := α)
            (s₁ := sScores) (s₂ := .dim rows (.dim seqLen .scalar))
            masked (by simp [sScores, rows, Shape.size, Nat.mul_left_comm, Nat.mul_comm])
          let attnRows ← TL.sparsemaxRowsRef (m := m) (α := α)
            (rows := rows) (keyLen := seqLen) scoreRows
          let attn ← TorchLean.reshape (m := m) (α := α)
            (s₁ := .dim rows (.dim seqLen .scalar)) (s₂ := sScores)
            attnRows (by simp [sScores, rows, Shape.size, Nat.mul_left_comm, Nat.mul_comm])
          let ctx ← _root_.Runtime.Autograd.Torch.bmm (m := m) (α := α)
            (batch := batch) (mDim := seqLen) (nDim := seqLen) (pDim := dModel) attn v
          let ctxRows ← TorchLean.reshape (m := m) (α := α)
            (s₁ := s3) (s₂ := sRows) ctx (by
              simp [s3, sRows, rows, Shape.size, Nat.mul_left_comm, Nat.mul_comm])
          let outRows ← TL.linearRowsRef (m := m) (α := α)
            (rows := rows) (inDim := dModel) (outDim := dModel) wo bo ctxRows
          TorchLean.reshape (m := m) (α := α)
            (s₁ := sRows) (s₂ := s3) outRows (by
              simp [s3, sRows, rows, Shape.size, Nat.mul_left_comm, Nat.mul_comm])
        ) : m (TorchLean.RefTy (m := m) (α := α) s3))
  }

/-- Sequential wrapper for sparsemax causal self-attention. -/
def sparsemaxCausalSelfAttention (batch seqLen dModel : Nat) (seedBase : Nat := 0) :
    nn.Sequential
      (.dim batch (.dim seqLen (.dim dModel .scalar)))
      (.dim batch (.dim seqLen (.dim dModel .scalar))) :=
  nn.of (sparsemaxCausalSelfAttentionLayer batch seqLen dModel seedBase)

/-- Sequence-shaped Signed-L1-BandNorm layer over all batch/position rows. -/
def signedL1BandNormSequenceLayer (batch seqLen dModel : Nat) :
    TorchLean.NN.LayerDef
      (.dim batch (.dim seqLen (.dim dModel .scalar)))
      (.dim batch (.dim seqLen (.dim dModel .scalar))) :=
  let rows := batch * seqLen
  let gammaShape : Shape := .dim dModel .scalar
  let betaShape : Shape := .dim dModel .scalar
  let gamma0 : Tensor Float gammaShape := Spec.fill (α := Float) 1.0 gammaShape
  let beta0 : Tensor Float betaShape := Spec.fill (α := Float) 0.0 betaShape
  let s3 : Shape := .dim batch (.dim seqLen (.dim dModel .scalar))
  let sRows : Shape := .dim rows (.dim dModel .scalar)
  { paramShapes := [gammaShape, betaShape]
    initParams := TorchLean.tlist2 gamma0 beta0
    paramRequiresGrad := [true, true]
    forward := fun _mode {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
        ((do
          let xRows ← TorchLean.reshape (m := m) (α := α)
            (s₁ := s3) (s₂ := sRows) x (by
              simp [s3, sRows, rows, Shape.size, Nat.mul_left_comm, Nat.mul_comm])
          let yRows ← TL.signedL1BandNormRowsRef (m := m) (α := α)
            (rows := rows) (n := dModel) gamma beta xRows
          TorchLean.reshape (m := m) (α := α)
            (s₁ := sRows) (s₂ := s3) yRows (by
              simp [s3, sRows, rows, Shape.size, Nat.mul_left_comm, Nat.mul_comm])
        ) : m (TorchLean.RefTy (m := m) (α := α) s3))
  }

/-- Sequential wrapper for sequence-shaped Signed-L1-BandNorm. -/
def signedL1BandNormSequence (batch seqLen dModel : Nat) :
    nn.Sequential
      (.dim batch (.dim seqLen (.dim dModel .scalar)))
      (.dim batch (.dim seqLen (.dim dModel .scalar))) :=
  nn.of (signedL1BandNormSequenceLayer batch seqLen dModel)

/-- One pre-norm residual attention block followed by a residual MLP block. -/
def transformerBlock (seedBase : Nat) :
    nn.M (nn.Sequential ModelShape ModelShape) := do
  let norm1 := signedL1BandNormSequence batch seqLen dModel
  let attn := sparsemaxCausalSelfAttention batch seqLen dModel (seedBase + 10)
  let norm2 := signedL1BandNormSequence batch seqLen dModel
  let ffn ←
    (nn.sequential![
      pure (nn.pure.linear dModel ffnHidden (seedBase + 20) (seedBase + 21)
        (pfx := .dim batch (.dim seqLen .scalar))),
      pure (leakyRelu (s := .dim batch (.dim seqLen (.dim ffnHidden .scalar))) 100),
      pure (nn.pure.linear ffnHidden dModel (seedBase + 22) (seedBase + 23)
        (pfx := .dim batch (.dim seqLen .scalar)))
    ] : nn.M (nn.Sequential ModelShape ModelShape))
  let normAttn ← (nn.sequential![pure norm1, pure attn] : nn.M (nn.Sequential ModelShape ModelShape))
  let normFfn ← (nn.sequential![pure norm2, pure ffn] : nn.M (nn.Sequential ModelShape ModelShape))
  nn.sequential![
    pure (nn.pure.blocks.residual normAttn),
    pure (nn.pure.blocks.residual normFfn)
  ]

/-- Trainable same-size TorchLean model used to produce the CUDA eval trace. -/
def mkTrainableModel : nn.M (nn.Sequential OneHotShape OneHotShape) :=
  nn.sequential![
    nn.embedding vocab dModel (pfx := .dim batch (.dim seqLen .scalar)),
    nn.learnedPositionalEmbedding (batch := batch) (seqLen := seqLen) (embedDim := dModel),
    transformerBlock 1000,
    transformerBlock 2000,
    pure (signedL1BandNormSequence batch seqLen dModel),
    pure (linearNoBias dModel vocab 3000 (pfx := .dim batch (.dim seqLen .scalar)))
  ]

/-- Print the run configuration before training or trace generation. -/
def printRunHeader (mode : String) (steps evalBatches : Nat) : IO Unit := do
  IO.println s!"TorchLean GPT dimensions: vocab={vocab}, seqLen={seqLen}, dModel={dModel}, layers={layers}, heads={numHeads}, dMlp={ffnHidden}"
  IO.println s!"parameter count: 7712"
  IO.println s!"operators: Signed-L1-BandNorm + sparsemax causal attention + LeakyReLU + no-bias lm_head"
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
    let m ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts
    if instantiateOnly then
      printRunHeader "instantiate-only" steps evalBatches
      IO.println "instantiated trainable TorchLean GPT module; skipped forward/backward by request"
      let ps ← _root_.Runtime.Autograd.Torch.ParamList.valuesSynced (α := Float)
        (ss := nn.paramShapes model) m.trainer.params
      if inspectWeights then
        printParamInspection ps
      if let some path := saveParams? then
        Runtime.Autograd.TorchLean.ParamIO.writeTListBits path ps
        IO.println s!"saved initial parameters to {path}"
      if let some path := saveEvalJson? then
        writeEvalJson opts model m.trainer.params path (saveParams?.getD "initial-params-not-saved")
        IO.println s!"saved initial eval trace to {path}"
      return ()
    let sample0 := firstSample (α := Float)
    let loss0 ← TorchLean.Module.forward (α := Float) m sample0
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
    let loss1 ← TorchLean.Module.forward (α := Float) m sample0
    let acc1 ← evalCandidateAccuracy opts model m.trainer.params evalBatches
    if let some path := saveParams? then
      Runtime.Autograd.TorchLean.ParamIO.writeTListBits path trainedParams
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
  let (steps?, rest) ← CLI.takeNatFlagOnce args "steps"
  let (evalBatches?, rest) ← CLI.takeNatFlagOnce rest "eval-batches"
  let (instantiateOnly, rest) ← CLI.takeBoolFlagOnce rest "instantiate-only"
  let (saveParams?, rest) ← CLI.takePathFlagOnce rest "save-params"
  let (saveEvalJson?, rest) ← CLI.takePathFlagOnce rest "save-eval-json"
  let (inspectWeights, rest) ← CLI.takeBoolFlagOnce rest "inspect-weights"
  pure (steps?.getD 10, evalBatches?.getD 1, instantiateOnly, saveParams?, saveEvalJson?, inspectWeights, rest)

/-- CLI entrypoint for TorchLean training, parameter export, and eval-trace export. -/
def main (args : List String) : IO UInt32 := do
  TorchLean.Module.run exeName args
    (.float (fun opts rest => do
      let (steps, evalBatches, instantiateOnly, saveParams?, saveEvalJson?, inspectWeights, rest) ←
        Common.orThrow exeName <| parseRunFlags rest
      Common.orThrow exeName <| CLI.requireNoArgs rest
      trainModel opts steps evalBatches instantiateOnly saveParams? saveEvalJson? inspectWeights))
    { banner? := some (fun opts =>
        s!"{exeName}: TorchLean GPT task training (device={if opts.useGpu then "cuda" else "cpu"})")
      printOk := true }

end VerifiableTransformers.TorchLean.TrainSmallGPT

/-- Lake executable entrypoint. -/
def main (args : List String) : IO UInt32 :=
  VerifiableTransformers.TorchLean.TrainSmallGPT.main args
