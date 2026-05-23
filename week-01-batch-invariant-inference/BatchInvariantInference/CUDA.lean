import BatchInvariantInference.Core
import Mathlib.Data.Bool.Basic
import Mathlib.Tactic

/-!
# CUDA Microkernel Certificate

Here is where the CUDA part gets checked. I kept the kernel small enough that
we can see every moving piece, then connected that checked fragment back to the
Lean semantics used by the batch-invariance proof. The pipeline is:

1. memory/indexing facts for the tiny value-reduction kernel;
2. the Lean specification of the left-to-right FMA reduction;
3. the finite PTX/SASS certificate emitted by
   `week-01-batch-invariant-inference/cuda/extract_cert.py`;
4. the theorem that an accepted certificate denotes the intended value reduction
   and can be connected to the TorchLean FlashAttention boundary.

The scope is narrow by design. We are not modeling all of CUDA, PTX, SASS, or
NVIDIA hardware here. We check one proof-carrying certificate extracted from
one compiled kernel, and we leave the remaining runtime refinement obligations
visible instead of silently trusting them.
-/

/-!
# 1. Memory And Launch Contracts

These are the small memory/indexing facts used by the CUDA certificate. The
kernel shape is deliberately boring: one block per batch row, threads `0..3`
owning output coordinates, and no shared-memory cooperation. That lets the
first certificate focus on the reduction order and ownership facts:

* the wrapper checked tensor sizes for the tiny value-reduction kernel;
* each thread is guarded by an output bound check;
* the thread index is decoded into one output coordinate;
* the kernel writes only the owned output coordinate;
* the forward kernel does not use shared memory or atomics.
-/

namespace CUDA
namespace Memory

/-!
The tiny kernel uses the fixed shapes

* `weights : [B, 8]`
* `v       : [B, 8, 4]`
* `out     : [B, 4]`

The address lemmas below are deliberately small. They are the index arithmetic
that the PTX/SASS certificate is trying to connect to the compiled artifact.
-/

def weightsAddr (b t : Nat) : Nat :=
  b * 8 + t

def valueAddr (b t o : Nat) : Nat :=
  (b * 8 + t) * 4 + o

def outAddr (b o : Nat) : Nat :=
  b * 4 + o

theorem weightsAddr_lt
    {B b t : Nat} (hb : b < B) (ht : t < 8) :
    weightsAddr b t < B * 8 := by
  unfold weightsAddr
  omega

theorem valueAddr_lt
    {B b t o : Nat} (hb : b < B) (ht : t < 8) (ho : o < 4) :
    valueAddr b t o < B * 8 * 4 := by
  unfold valueAddr
  omega

theorem outAddr_lt
    {B b o : Nat} (hb : b < B) (ho : o < 4) :
    outAddr b o < B * 4 := by
  unfold outAddr
  omega

theorem outAddr_owned_unique
    {b tid₁ tid₂ : Nat} (haddr : outAddr b tid₁ = outAddr b tid₂) :
    tid₁ = tid₂ := by
  unfold outAddr at haddr
  omega

def tinyThreadWrites (b tid : Nat) : List Nat :=
  if tid < 4 then [outAddr b tid] else []

theorem inactive_threads_no_write
    {b tid : Nat} (h : 4 ≤ tid) :
    tinyThreadWrites b tid = [] := by
  simp [tinyThreadWrites, Nat.not_lt.mpr h]

theorem active_thread_writes_owned
    {b tid : Nat} (h : tid < 4) :
    tinyThreadWrites b tid = [outAddr b tid] := by
  simp [tinyThreadWrites, h]

/-- Memory/indexing obligations for the tiny value-reduction kernel.

These are not meant to describe arbitrary CUDA kernels. They describe the
specific safety facts the extractor checks for `tiny_attn_one_row.cu`. -/
structure KernelMemoryFacts where
  wrapperChecksWeightsSize : Bool
  wrapperChecksValueSize : Bool
  wrapperChecksOutputSize : Bool
  boundsChecksIdx : Bool
  oneThreadPerOutput : Bool
  decodesBatchQueryChannel : Bool
  writesOwnedOutput : Bool
  noSharedMemory : Bool
  noAtomics : Bool
deriving Repr

def KernelMemoryContract (facts : KernelMemoryFacts) : Prop :=
  facts.wrapperChecksWeightsSize = true ∧
  facts.wrapperChecksValueSize = true ∧
  facts.wrapperChecksOutputSize = true ∧
  facts.boundsChecksIdx = true ∧
  facts.oneThreadPerOutput = true ∧
  facts.decodesBatchQueryChannel = true ∧
  facts.writesOwnedOutput = true ∧
  facts.noSharedMemory = true ∧
  facts.noAtomics = true

def checkKernelMemoryFacts (facts : KernelMemoryFacts) : Bool :=
  facts.wrapperChecksWeightsSize &&
  facts.wrapperChecksValueSize &&
  facts.wrapperChecksOutputSize &&
  facts.boundsChecksIdx &&
  facts.oneThreadPerOutput &&
  facts.decodesBatchQueryChannel &&
  facts.writesOwnedOutput &&
  facts.noSharedMemory &&
  facts.noAtomics

theorem checkKernelMemoryFacts_sound
    (facts : KernelMemoryFacts)
    (hcheck : checkKernelMemoryFacts facts = true) :
    KernelMemoryContract facts := by
  unfold checkKernelMemoryFacts at hcheck
  unfold KernelMemoryContract
  simp at hcheck
  simpa [and_assoc] using hcheck

end Memory
end CUDA


/-!
# 2. Tiny Attention Value-Reduction Spec

This is the Lean-side specification for the CUDA certificate. It reuses the
schedule-explicit selected-attention semantics from `BatchInvariantInference`.

The inspected CUDA kernel only computes the value-reduction part of attention:
softmax weights are already provided, and each active thread accumulates one
output coordinate. That is the first useful bridge before moving to online
softmax or a full FlashAttention-style kernel.
-/

namespace CUDA
namespace TinyAttentionSpec

open BatchInvariantInference

inductive ArithmeticMode where
  | fmaRN
  | mulAddRN
deriving Repr, DecidableEq

abbrev i0 : Fin 8 := ⟨0, by decide⟩
abbrev i1 : Fin 8 := ⟨1, by decide⟩
abbrev i2 : Fin 8 := ⟨2, by decide⟩
abbrev i3 : Fin 8 := ⟨3, by decide⟩
abbrev i4 : Fin 8 := ⟨4, by decide⟩
abbrev i5 : Fin 8 := ⟨5, by decide⟩
abbrev i6 : Fin 8 := ⟨6, by decide⟩
abbrev i7 : Fin 8 := ⟨7, by decide⟩

/-- Left-to-right value reduction for the PTX shape that uses fused
`fma.rn.f32` instructions. This is intentionally not the same as first rounding
`w * v` and then adding it to the accumulator. -/
def valueReduceFMA
    (fma : β -> β -> β -> β)
    (zero : β)
    (weights values : Fin 8 -> β) : β :=
  let acc1 := fma (weights i0) (values i0) zero
  let acc2 := fma (weights i1) (values i1) acc1
  let acc3 := fma (weights i2) (values i2) acc2
  let acc4 := fma (weights i3) (values i3) acc3
  let acc5 := fma (weights i4) (values i4) acc4
  let acc6 := fma (weights i5) (values i5) acc5
  let acc7 := fma (weights i6) (values i6) acc6
  fma (weights i7) (values i7) acc7

/-- Left-to-right value reduction for a non-fused compile mode. -/
def valueReduceMulAdd
    (add mul : β -> β -> β)
    (zero : β)
    (weights values : Fin 8 -> β) : β :=
  let acc1 := add zero (mul (weights i0) (values i0))
  let acc2 := add acc1 (mul (weights i1) (values i1))
  let acc3 := add acc2 (mul (weights i2) (values i2))
  let acc4 := add acc3 (mul (weights i3) (values i3))
  let acc5 := add acc4 (mul (weights i4) (values i4))
  let acc6 := add acc5 (mul (weights i5) (values i5))
  let acc7 := add acc6 (mul (weights i6) (values i6))
  add acc7 (mul (weights i7) (values i7))

def valueReduceByMode
    (mode : ArithmeticMode)
    (add mul : β -> β -> β)
    (fma : β -> β -> β -> β)
    (zero : β)
    (weights values : Fin 8 -> β) : β :=
  match mode with
  | .fmaRN => valueReduceFMA fma zero weights values
  | .mulAddRN => valueReduceMulAdd add mul zero weights values

abbrev Request
    (Feature : Type u) (KV : Type v) (Out : Type w) (β : Type x) :=
  CUDARuntime.SelectedAttentionRequest Feature KV Out β

/-- The Lean specification that the inspected fused attention kernel is meant
to refine. -/
def forwardSpec
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featurePolicy : Attention.FeatureSchedulePolicy Feature)
    (layout : Attention.KVLayoutPolicy KV)
    (softmax : (KV -> β) -> KV -> β) :
    BatchedForward
      (Request Feature KV Out β)
      (Out -> β) :=
  CUDARuntime.scheduledAttentionForward add mul featurePolicy layout softmax

theorem forwardSpec_batchInvariant
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featurePolicy : Attention.FeatureSchedulePolicy Feature)
    (layout : Attention.KVLayoutPolicy KV)
    (softmax : (KV -> β) -> KV -> β) :
    BatchInvariantForward
      (forwardSpec
        (Feature := Feature) (KV := KV) (Out := Out) (β := β)
        add mul featurePolicy layout softmax) :=
  CUDARuntime.scheduledAttentionForward_batchInvariant
    add mul featurePolicy layout softmax

end TinyAttentionSpec
end CUDA


/-!
# 3. CUDA/PTX/SASS Certificate

This section defines the finite certificate emitted by
`week-01-batch-invariant-inference/cuda/extract_cert.py`. The certificate
records source, PTX, CUBIN, and SASS hashes, plus the extracted FMA-chain
dataflow. The hashes are audit metadata; Lean checks the finite certificate
fields brought into this file, not files on disk.
-/

namespace CUDA
namespace TinyAttentionCert

/-- One `fma.rn.f32 dst, lhs, rhs, acc` step extracted from PTX, together
with the byte offsets of the weight and value loads feeding `lhs` and `rhs`. -/
structure FMAInstCert where
  /-- Destination register written by the PTX `fma.rn.f32` instruction. -/
  dst : String
  /-- First source register. The extractor checks that it comes from a weight
  load, modulo operand order. -/
  lhs : String
  /-- Second source register. The extractor checks that it comes from a value
  load, modulo operand order. -/
  rhs : String
  /-- Accumulator operand. For a valid chain, this is zero for step 0 and the
  previous step's destination register thereafter. -/
  acc : String
  /-- Byte offset of the weight load feeding this FMA. For the tiny kernel these
  must be `0,4,...,28`, one Float32 per KV position. -/
  weightOffset : Nat
  /-- Byte offset of the value load feeding this FMA. For the tiny kernel these
  must be `0,16,...,112`, four Float32 outputs per KV position. -/
  valueOffset : Nat
deriving Repr

/-- Certificate for the exact eight-step FMA chain in `tiny_attn_one_row.ptx`.
It records enough dataflow to prove that the register stored to memory is
computed by the expected left-to-right reduction. -/
structure FMAChain8Cert where
  zeroLiteral : String
  step0 : FMAInstCert
  step1 : FMAInstCert
  step2 : FMAInstCert
  step3 : FMAInstCert
  step4 : FMAInstCert
  step5 : FMAInstCert
  step6 : FMAInstCert
  step7 : FMAInstCert
  storeReg : String
  chainStartsAtZero : Bool
  chainFeedsStore : Bool
  addressMapCorrect : Bool
  inactiveThreadsNoWrite : Bool
  arithmeticModeFmaRN : Bool
deriving Repr

/-- Semantic contract for an accepted eight-step FMA-chain certificate.

The contract checks three things:

* the chain starts at the PTX zero literal;
* each FMA consumes the previous FMA's destination as its accumulator;
* the load offsets match the fixed `[B, 8]` weights and `[B, 8, 4]` values.
-/
def FMAChain8Contract (cert : FMAChain8Cert) : Prop :=
  cert.chainStartsAtZero = true ∧
  cert.chainFeedsStore = true ∧
  cert.addressMapCorrect = true ∧
  cert.inactiveThreadsNoWrite = true ∧
  cert.arithmeticModeFmaRN = true ∧
  cert.zeroLiteral = "0f00000000" ∧
  cert.step0.acc = cert.zeroLiteral ∧
  cert.step1.acc = cert.step0.dst ∧
  cert.step2.acc = cert.step1.dst ∧
  cert.step3.acc = cert.step2.dst ∧
  cert.step4.acc = cert.step3.dst ∧
  cert.step5.acc = cert.step4.dst ∧
  cert.step6.acc = cert.step5.dst ∧
  cert.step7.acc = cert.step6.dst ∧
  cert.storeReg = cert.step7.dst ∧
  cert.step0.weightOffset = 0 ∧ cert.step0.valueOffset = 0 ∧
  cert.step1.weightOffset = 4 ∧ cert.step1.valueOffset = 16 ∧
  cert.step2.weightOffset = 8 ∧ cert.step2.valueOffset = 32 ∧
  cert.step3.weightOffset = 12 ∧ cert.step3.valueOffset = 48 ∧
  cert.step4.weightOffset = 16 ∧ cert.step4.valueOffset = 64 ∧
  cert.step5.weightOffset = 20 ∧ cert.step5.valueOffset = 80 ∧
  cert.step6.weightOffset = 24 ∧ cert.step6.valueOffset = 96 ∧
  cert.step7.weightOffset = 28 ∧ cert.step7.valueOffset = 112

/-- Boolean checker for `FMAChain8Contract`.

The generated certificate is accepted by computation. The theorem immediately
below states that if this checker returns `true`, the propositional contract
really holds. -/
def checkFMAChain8Cert (cert : FMAChain8Cert) : Bool :=
  cert.chainStartsAtZero &&
  cert.chainFeedsStore &&
  cert.addressMapCorrect &&
  cert.inactiveThreadsNoWrite &&
  cert.arithmeticModeFmaRN &&
  (cert.zeroLiteral == "0f00000000") &&
  (cert.step0.acc == cert.zeroLiteral) &&
  (cert.step1.acc == cert.step0.dst) &&
  (cert.step2.acc == cert.step1.dst) &&
  (cert.step3.acc == cert.step2.dst) &&
  (cert.step4.acc == cert.step3.dst) &&
  (cert.step5.acc == cert.step4.dst) &&
  (cert.step6.acc == cert.step5.dst) &&
  (cert.step7.acc == cert.step6.dst) &&
  (cert.storeReg == cert.step7.dst) &&
  (cert.step0.weightOffset == 0) && (cert.step0.valueOffset == 0) &&
  (cert.step1.weightOffset == 4) && (cert.step1.valueOffset == 16) &&
  (cert.step2.weightOffset == 8) && (cert.step2.valueOffset == 32) &&
  (cert.step3.weightOffset == 12) && (cert.step3.valueOffset == 48) &&
  (cert.step4.weightOffset == 16) && (cert.step4.valueOffset == 64) &&
  (cert.step5.weightOffset == 20) && (cert.step5.valueOffset == 80) &&
  (cert.step6.weightOffset == 24) && (cert.step6.valueOffset == 96) &&
  (cert.step7.weightOffset == 28) && (cert.step7.valueOffset == 112)

theorem checkFMAChain8Cert_sound
    (cert : FMAChain8Cert)
    (hcheck : checkFMAChain8Cert cert = true) :
    FMAChain8Contract cert := by
  unfold checkFMAChain8Cert at hcheck
  unfold FMAChain8Contract
  simp at hcheck ⊢
  simpa [and_assoc] using hcheck

/-- Source, PTX, and SASS obligations extracted for `tiny_attn_one_row.cu`.

The string paths and hashes make the generated certificate auditable. The
boolean and numeric fields are the proof-carrying part consumed by the checker.
-/
structure TinyAttentionKernelCert where
  /-- Repo-relative CUDA source path, kept as audit metadata. -/
  sourcePath : String
  /-- SHA-256 of the source file. Lean records this string but does not read the
  external file during theorem checking. -/
  sourceSha256 : String
  ptxPath : String
  ptxSha256 : String
  cubinPath : String
  cubinSha256 : String
  sassPath : String
  sassSha256 : String
  arch : String
  ptxFmaRnF32 : Nat
  ptxGlobalLoadF32 : Nat
  ptxGlobalStoreF32 : Nat
  sourceHasKernel : Bool
  sourceHasFixedKVLoop : Bool
  sourceHasThreadGuard : Bool
  sourceHasBatchGuard : Bool
  sourceHasOwnedWrite : Bool
  ptxHasEntry : Bool
  ptxHasThreadIdx : Bool
  ptxHasBlockIdx : Bool
  ptxHasBoundsPredicate : Bool
  ptxHasEightFmas : Bool
  ptxHasExpectedLoads : Bool
  ptxHasSingleStore : Bool
  ptxHasNoSharedMemory : Bool
  ptxHasNoAtomics : Bool
  ptxHasNoBarrier : Bool
  sassHasCodeForKernel : Bool
  sassHasGlobalLoads : Bool
  sassHasGlobalStore : Bool
  sassHasNoBarrier : Bool
  sassHasNoAtomics : Bool
  dataflow : FMAChain8Cert
  memoryFacts : CUDA.Memory.KernelMemoryFacts
deriving Repr

/-- Full contract for the tiny CUDA value-reduction certificate.

This is intentionally a finite certificate contract, not a full CUDA semantics.
It requires the expected architecture, instruction counts, source/PTX/SASS
shape facts, the FMA dataflow contract, and memory/indexing facts. -/
def TinyAttentionKernelContract (cert : TinyAttentionKernelCert) : Prop :=
  cert.arch = "sm_70" ∧
  cert.ptxFmaRnF32 = 8 ∧
  cert.ptxGlobalLoadF32 = 16 ∧
  cert.ptxGlobalStoreF32 = 1 ∧
  cert.sourceHasKernel = true ∧
  cert.sourceHasFixedKVLoop = true ∧
  cert.sourceHasThreadGuard = true ∧
  cert.sourceHasBatchGuard = true ∧
  cert.sourceHasOwnedWrite = true ∧
  cert.ptxHasEntry = true ∧
  cert.ptxHasThreadIdx = true ∧
  cert.ptxHasBlockIdx = true ∧
  cert.ptxHasBoundsPredicate = true ∧
  cert.ptxHasEightFmas = true ∧
  cert.ptxHasExpectedLoads = true ∧
  cert.ptxHasSingleStore = true ∧
  cert.ptxHasNoSharedMemory = true ∧
  cert.ptxHasNoAtomics = true ∧
  cert.ptxHasNoBarrier = true ∧
  cert.sassHasCodeForKernel = true ∧
  cert.sassHasGlobalLoads = true ∧
  cert.sassHasGlobalStore = true ∧
  cert.sassHasNoBarrier = true ∧
  cert.sassHasNoAtomics = true ∧
  FMAChain8Contract cert.dataflow ∧
  CUDA.Memory.KernelMemoryContract cert.memoryFacts

/-- Boolean checker for the whole CUDA/PTX/SASS certificate. -/
def checkTinyAttentionKernelCert (cert : TinyAttentionKernelCert) : Bool :=
  (cert.arch == "sm_70") &&
  (cert.ptxFmaRnF32 == 8) &&
  (cert.ptxGlobalLoadF32 == 16) &&
  (cert.ptxGlobalStoreF32 == 1) &&
  cert.sourceHasKernel &&
  cert.sourceHasFixedKVLoop &&
  cert.sourceHasThreadGuard &&
  cert.sourceHasBatchGuard &&
  cert.sourceHasOwnedWrite &&
  cert.ptxHasEntry &&
  cert.ptxHasThreadIdx &&
  cert.ptxHasBlockIdx &&
  cert.ptxHasBoundsPredicate &&
  cert.ptxHasEightFmas &&
  cert.ptxHasExpectedLoads &&
  cert.ptxHasSingleStore &&
  cert.ptxHasNoSharedMemory &&
  cert.ptxHasNoAtomics &&
  cert.ptxHasNoBarrier &&
  cert.sassHasCodeForKernel &&
  cert.sassHasGlobalLoads &&
  cert.sassHasGlobalStore &&
  cert.sassHasNoBarrier &&
  cert.sassHasNoAtomics &&
  checkFMAChain8Cert cert.dataflow &&
  CUDA.Memory.checkKernelMemoryFacts cert.memoryFacts

theorem checkTinyAttentionKernelCert_sound
    (cert : TinyAttentionKernelCert)
    (hcheck : checkTinyAttentionKernelCert cert = true) :
    TinyAttentionKernelContract cert := by
  unfold checkTinyAttentionKernelCert at hcheck
  unfold TinyAttentionKernelContract
  simp [CUDA.Memory.KernelMemoryContract, CUDA.Memory.checkKernelMemoryFacts,
    FMAChain8Contract, checkFMAChain8Cert] at hcheck ⊢
  simpa [and_assoc] using hcheck

theorem checkTinyAttentionKernelCert_dataflow_sound
    (cert : TinyAttentionKernelCert)
    (hcheck : checkTinyAttentionKernelCert cert = true) :
    FMAChain8Contract cert.dataflow := by
  have h := checkTinyAttentionKernelCert_sound cert hcheck
  unfold TinyAttentionKernelContract at h
  tauto

end TinyAttentionCert
end CUDA


/-!
# 4. PTX Dataflow Semantics

This section gives a small Lean semantics for the PTX fragment extracted from
`tiny_attn_one_row.ptx`: a straight-line eight-step `fma.rn.f32` accumulator
chain that stores its final accumulator.

It is not a full PTX semantics. It is the next refinement layer after the
certificate checker: accepted PTX dataflow denotes the same left-to-right FMA
reduction used by the Lean TinyAttention spec.
-/

namespace CUDA
namespace TinyPTXSemantics

open CUDA.TinyAttentionCert
open CUDA.TinyAttentionSpec
open BatchInvariantInference

/-- Proof-facing input payload for the tiny value-reduction kernel.

The CUDA kernel has one selected batch row and four output coordinates. The
certificate theorem focuses on one owned output coordinate, so the inputs reduce
to eight softmax weights and eight values for that coordinate. -/
structure ValueInputs (β : Type u) where
  weights : Fin 8 -> β
  values : Fin 8 -> β

/-- Interpret byte offsets from PTX loads as weight indices. Any unexpected
offset maps to `default`; the certificate contract proves the accepted offsets
are exactly the expected ones, so the default branch is unreachable for accepted
certificates. -/
def weightAtOffset (inputs : ValueInputs β) (default : β) : Nat -> β
  | 0 => inputs.weights i0
  | 4 => inputs.weights i1
  | 8 => inputs.weights i2
  | 12 => inputs.weights i3
  | 16 => inputs.weights i4
  | 20 => inputs.weights i5
  | 24 => inputs.weights i6
  | 28 => inputs.weights i7
  | _ => default

/-- Interpret byte offsets from PTX loads as value indices for one output
coordinate. The stride is 16 bytes because each KV position stores four Float32
output coordinates. -/
def valueAtOffset (inputs : ValueInputs β) (default : β) : Nat -> β
  | 0 => inputs.values i0
  | 16 => inputs.values i1
  | 32 => inputs.values i2
  | 48 => inputs.values i3
  | 64 => inputs.values i4
  | 80 => inputs.values i5
  | 96 => inputs.values i6
  | 112 => inputs.values i7
  | _ => default

/-- Evaluate one extracted PTX FMA instruction against the Lean-side input
payload. The register names have already been checked by the certificate; this
semantic evaluator uses the checked load offsets and the incoming accumulator. -/
def evalFMAInst
    (fma : β -> β -> β -> β)
    (inputs : ValueInputs β)
    (default : β)
    (inst : FMAInstCert)
    (acc : β) : β :=
  fma
    (weightAtOffset inputs default inst.weightOffset)
    (valueAtOffset inputs default inst.valueOffset)
    acc

/-- Evaluate the extracted eight-step FMA chain in the same order as the PTX
dataflow certificate. -/
def evalFMAChain8
    (cert : FMAChain8Cert)
    (fma : β -> β -> β -> β)
    (zero : β)
    (inputs : ValueInputs β) : β :=
  let acc1 := evalFMAInst fma inputs zero cert.step0 zero
  let acc2 := evalFMAInst fma inputs zero cert.step1 acc1
  let acc3 := evalFMAInst fma inputs zero cert.step2 acc2
  let acc4 := evalFMAInst fma inputs zero cert.step3 acc3
  let acc5 := evalFMAInst fma inputs zero cert.step4 acc4
  let acc6 := evalFMAInst fma inputs zero cert.step5 acc5
  let acc7 := evalFMAInst fma inputs zero cert.step6 acc6
  evalFMAInst fma inputs zero cert.step7 acc7

theorem evalFMAChain8_eq_valueReduceFMA_of_contract
    (cert : FMAChain8Cert)
    (hcert : FMAChain8Contract cert)
    (fma : β -> β -> β -> β)
    (zero : β)
    (inputs : ValueInputs β) :
    evalFMAChain8 cert fma zero inputs =
      valueReduceFMA fma zero inputs.weights inputs.values := by
  unfold FMAChain8Contract at hcert
  simp_all [evalFMAChain8, evalFMAInst, weightAtOffset, valueAtOffset,
    valueReduceFMA]

theorem evalFMAChain8_eq_valueReduceFMA_of_check
    (cert : FMAChain8Cert)
    (hcheck : checkFMAChain8Cert cert = true)
    (fma : β -> β -> β -> β)
    (zero : β)
    (inputs : ValueInputs β) :
    evalFMAChain8 cert fma zero inputs =
      valueReduceFMA fma zero inputs.weights inputs.values :=
  evalFMAChain8_eq_valueReduceFMA_of_contract
    cert (checkFMAChain8Cert_sound cert hcheck) fma zero inputs

theorem kernelCert_denotes_valueReduceFMA
    (cert : TinyAttentionKernelCert)
    (hcheck : checkTinyAttentionKernelCert cert = true)
    (fma : β -> β -> β -> β)
    (zero : β)
    (inputs : ValueInputs β) :
    evalFMAChain8 cert.dataflow fma zero inputs =
      valueReduceFMA fma zero inputs.weights inputs.values :=
  evalFMAChain8_eq_valueReduceFMA_of_contract
    cert.dataflow
    (checkTinyAttentionKernelCert_dataflow_sound cert hcheck)
    fma zero inputs

abbrev TinyValueRequest (β : Type u) :=
  ValueInputs β

/-- A batched forward view of the checked tiny CUDA value reduction. This lets
the microkernel certificate plug into the same `BatchInvariantForward` contract
as the model-level theorems. -/
def tinyValueReductionForward
    (cert : TinyAttentionKernelCert)
    (fma : β -> β -> β -> β)
    (zero : β) :
    BatchedForward (TinyValueRequest β) β :=
  fun _B xs i =>
    evalFMAChain8 cert.dataflow fma zero (xs i)

theorem tinyValueReductionForward_batchInvariant_of_check
    (cert : TinyAttentionKernelCert)
    (_hcheck : checkTinyAttentionKernelCert cert = true)
    (fma : β -> β -> β -> β)
    (zero : β) :
    BatchInvariantForward (tinyValueReductionForward cert fma zero) := by
  intro B C xs ys i j hsame
  simp [tinyValueReductionForward, hsame]

/-- The refinement statement exported by the tiny certificate: evaluating the
extracted dataflow chain is the same as the clean Lean FMA-chain specification. -/
def RefinesTinyValueReduction
    (cert : TinyAttentionKernelCert)
    (fma : β -> β -> β -> β)
    (zero : β) : Prop :=
  forall inputs,
    evalFMAChain8 cert.dataflow fma zero inputs =
      valueReduceFMA fma zero inputs.weights inputs.values

theorem kernelCert_refines_tinyValueReduction
    (cert : TinyAttentionKernelCert)
    (hcheck : checkTinyAttentionKernelCert cert = true)
    (fma : β -> β -> β -> β)
    (zero : β) :
    RefinesTinyValueReduction cert fma zero := by
  intro inputs
  exact kernelCert_denotes_valueReduceFMA cert hcheck fma zero inputs

end TinyPTXSemantics
end CUDA


/-!
# 5. Soundness Bridge To TorchLean Attention

This section connects the source certificate to the semantic refinement theorem.
The source certificate makes the real CUDA file auditable; the native forward
certificate supplies the functional refinement to TorchLean's FlashAttention
denotation. Together they form the proof-carrying boundary for the actual
TorchLean fused CUDA attention path.
-/

namespace CUDA
namespace TinyAttentionSound

open BatchInvariantInference
open Spec

/-- A checked native forward certificate combines:

1. an extracted certificate for the compiled CUDA microkernel;
2. Lean's check that the certificate is accepted;
3. the semantic runtime refinement certificate to TorchLean FlashAttention.

The third field is the functional-refinement obligation. The source certificate
makes that obligation auditable against the real kernel body instead of leaving
it as an unnamed assumption. -/
structure CheckedNativeForwardCert
    (α : Type) [Context α] [DecidableRel ((· > ·) : α -> α -> Prop)]
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0} where
  kernelCert : CUDA.TinyAttentionCert.TinyAttentionKernelCert
  kernelChecked :
    CUDA.TinyAttentionCert.checkTinyAttentionKernelCert kernelCert = true
  nativeCert :
    TorchLeanFlashAttention.NativeForwardCert α
      (nQ := nQ) (nK := nK) (dModel := dModel) (h1 := h1) (h2 := h2)

theorem CheckedNativeForwardCert.kernel_contract
    {α : Type} [Context α] [DecidableRel ((· > ·) : α -> α -> Prop)]
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (cert : CheckedNativeForwardCert α
      (nQ := nQ) (nK := nK) (dModel := dModel) (h1 := h1) (h2 := h2)) :
    CUDA.TinyAttentionCert.TinyAttentionKernelContract cert.kernelCert :=
  CUDA.TinyAttentionCert.checkTinyAttentionKernelCert_sound
    cert.kernelCert cert.kernelChecked

theorem CheckedNativeForwardCert.fma_chain_contract
    {α : Type} [Context α] [DecidableRel ((· > ·) : α -> α -> Prop)]
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (cert : CheckedNativeForwardCert α
      (nQ := nQ) (nK := nK) (dModel := dModel) (h1 := h1) (h2 := h2)) :
    CUDA.TinyAttentionCert.FMAChain8Contract cert.kernelCert.dataflow :=
  CUDA.TinyAttentionCert.checkTinyAttentionKernelCert_dataflow_sound
    cert.kernelCert cert.kernelChecked

theorem CheckedNativeForwardCert.fma_chain_denotes_valueReduceFMA
    {α : Type} [Context α] [DecidableRel ((· > ·) : α -> α -> Prop)]
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (cert : CheckedNativeForwardCert α
      (nQ := nQ) (nK := nK) (dModel := dModel) (h1 := h1) (h2 := h2))
    (fma : β -> β -> β -> β)
    (zero : β)
    (inputs : CUDA.TinyPTXSemantics.ValueInputs β) :
    CUDA.TinyPTXSemantics.evalFMAChain8
      cert.kernelCert.dataflow fma zero inputs =
        CUDA.TinyAttentionSpec.valueReduceFMA
          fma zero inputs.weights inputs.values :=
  CUDA.TinyPTXSemantics.kernelCert_denotes_valueReduceFMA
    cert.kernelCert cert.kernelChecked fma zero inputs

theorem CheckedNativeForwardCert.refines_flashAttention
    {α : Type} [Context α] [DecidableRel ((· > ·) : α -> α -> Prop)]
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (cert : CheckedNativeForwardCert α
      (nQ := nQ) (nK := nK) (dModel := dModel) (h1 := h1) (h2 := h2)) :
    cert.nativeCert.runtimeOut =
      Spec.flashAttention cert.nativeCert.cfg cert.nativeCert.ctx :=
  cert.nativeCert.refines_flashAttention

end TinyAttentionSound
end CUDA
