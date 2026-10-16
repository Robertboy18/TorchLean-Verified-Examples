module

public import BatchInvariantInference.Core
public import Mathlib.Data.Bool.Basic
public import Mathlib.Tactic

@[expose] public section

/-!
# CUDA Microkernel Certificate

Here is where the CUDA part gets checked. I kept the kernel small enough to see
every moving piece, then connected the checked fragment back to the Lean
semantics of the batch-invariance proof. The sections are:

1. memory and indexing facts for the tiny value-reduction kernel
2. the Lean specification of the left-to-right FMA reduction
3. the active path of the compiled PTX, as recorded by
   `week-01-batch-invariant-inference/cuda/extract_cert.py` (its global loads with
   their byte addresses, its eight `fma.rn.f32` instructions with their registers,
   and its store), with a small semantics to run it
4. the certificate and its checker
5. the theorems for an accepted certificate: run by any active thread on any
   batch laid out in the kernel's buffers, it stores the intended value reduction
   for that thread's row and output coordinate
6. the connection to the TorchLean FlashAttention boundary

The scope is narrow on purpose. Nothing here models CUDA, PTX, SASS, or NVIDIA
hardware in general. Lean checks one certificate extracted from one compiled
kernel, and the remaining runtime refinement obligations stay visible as separate
fields and assumptions.
-/

/-!
# 1. Memory And Launch Contracts

Index arithmetic for the CUDA certificate. The kernel shape is deliberately
boring: one block per batch row, threads `0..3` owning the output coordinates,
and no shared-memory cooperation. So the certificate can focus on reduction
order and ownership:

* each load of an active thread reads a coordinate of its own row
* those loads and the store stay inside the `[B, 8]`, `[B, 8, 4]`, and `[B, 4]`
  buffers
* distinct active threads write distinct output coordinates

Lean proves these from the addresses recorded in the certificate (section 5).
The launch itself is an assumption. `cuda/README.md` lists what it needs:
allocation sizes, separate buffers, the grid shape, and the batch bound that
keeps the kernel's 32-bit indices from wrapping.
-/

namespace CUDA
namespace Memory

/-!
The tiny kernel uses the fixed shapes

* `weights : [B, 8]`
* `v       : [B, 8, 4]`
* `out     : [B, 4]`

The element indices below are the row-major positions the kernel source uses.
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

/-- Different active threads own different output coordinates. -/
theorem outAddr_injective
    {b₁ b₂ tid₁ tid₂ : Nat} (ht₁ : tid₁ < 4) (ht₂ : tid₂ < 4)
    (haddr : outAddr b₁ tid₁ = outAddr b₂ tid₂) :
    b₁ = b₂ ∧ tid₁ = tid₂ := by
  unfold outAddr at haddr
  omega

end Memory
end CUDA


/-!
# 2. Tiny Attention Value-Reduction Spec

The Lean specification behind the CUDA certificate: the FMA value reduction the
kernel computes, and the selected-attention spec from `BatchInvariantInference`
that a full attention kernel would refine.

The inspected kernel computes only the value-reduction part of attention. Its
softmax weights arrive precomputed, and each active thread accumulates one
output coordinate.
-/

namespace CUDA
namespace TinyAttentionSpec

open BatchInvariantInference
open FloatLib.Numerics (ReductionTree)

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

/-- Left-to-right value reduction with a fused multiply-add at each step, the
shape of the kernel's `fma.rn.f32` chain. It differs from `valueReduceMulAdd`,
which rounds `w * v` before adding it to the accumulator. -/
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

/-- Left-to-right value reduction with separate multiply and add, as a non-fused
build computes it. -/
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

/-- Selected-attention spec (`CUDARuntime.scheduledAttentionForward`) for a full
fused attention kernel. The tiny kernel's certificate targets the smaller
`TinyPTX.tinyValueReductionSpec` instead. -/
def forwardSpec
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featurePolicy : Attention.FeatureSchedulePolicy Feature)
    (layout : Attention.KVLayoutPolicy KV)
    (softmax : ReductionTree KV -> (KV -> β) -> KV -> β) :
    BatchedForward
      (Request Feature KV Out β)
      (Out -> β) :=
  CUDARuntime.scheduledAttentionForward add mul featurePolicy layout softmax

theorem forwardSpec_batchInvariant
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featurePolicy : Attention.FeatureSchedulePolicy Feature)
    (layout : Attention.KVLayoutPolicy KV)
    (softmax : ReductionTree KV -> (KV -> β) -> KV -> β) :
    BatchInvariantForward
      (forwardSpec
        (Feature := Feature) (KV := KV) (Out := Out) (β := β)
        add mul featurePolicy layout softmax) :=
  CUDARuntime.scheduledAttentionForward_batchInvariant
    add mul featurePolicy layout softmax

end TinyAttentionSpec
end CUDA


/-!
# 3. The Extracted PTX Program

The extractor follows the kernel's integer address arithmetic and its guards,
then records the active path as a straight-line program over floating-point
registers: global loads (with the byte address each one reads), `fma.rn.f32`
instructions (with their register operands), and the final store. Here the
program gets a small register-file semantics and a check by symbolic execution,
so Lean checks the register names and load addresses too.

It is not a full PTX semantics. The extractor folds the integer instructions
into the recorded addresses, and a recorded fact (section 4) stands in for the
guard branches.
-/

namespace CUDA
namespace TinyPTX

open CUDA.Memory
open CUDA.TinyAttentionSpec

/-- The PTX spelling of the binary32 constant `+0.0`. -/
def zeroLiteral : String := "0f00000000"

/-- Names for the kernel's three global buffers, in parameter order. -/
inductive Buffer where
  | weights
  | values
  | out
deriving Repr, DecidableEq

/-- A global address as the extractor reads it from PTX: a buffer base plus
`rowStrideBytes * blockIdx.x + laneStrideBytes * threadIdx.x + offsetBytes`.
The extractor derives these coefficients from 32-bit index arithmetic, so reading
them as natural numbers needs the batch bound in `cuda/README.md` (no wraparound). -/
structure AffineAddress where
  buffer : Buffer
  rowStrideBytes : Nat
  laneStrideBytes : Nat
  offsetBytes : Nat
deriving Repr, DecidableEq

namespace AffineAddress

/-- Byte offset from the buffer base for block `b` and thread `tid`. -/
def byteOffset (a : AffineAddress) (b tid : Nat) : Nat :=
  a.rowStrideBytes * b + a.laneStrideBytes * tid + a.offsetBytes

/-- Whether every coefficient is a multiple of four, which makes each access one
aligned binary32 word. -/
def aligned (a : AffineAddress) : Bool :=
  a.rowStrideBytes % 4 == 0 && a.laneStrideBytes % 4 == 0 && a.offsetBytes % 4 == 0

/-- Element index of the accessed binary32 word. -/
def index (a : AffineAddress) (b tid : Nat) : Nat :=
  a.byteOffset b tid / 4

end AffineAddress

/-- One floating-point instruction on the kernel's active path. -/
inductive FloatInst where
  /-- `ld.global.f32 dst, [addr]`, or its `.nc` form. -/
  | load (dst : String) (addr : AffineAddress)
  /-- `fma.rn.f32 dst, lhs, rhs, acc`. -/
  | fma (dst lhs rhs acc : String)
deriving Repr, DecidableEq

/-- The final `st.global.f32 [addr], src`. -/
structure StoreInst where
  src : String
  addr : AffineAddress
deriving Repr, DecidableEq

/-- An active path in PTX order, as recorded for `tiny_attn_one_row`: loads and
FMAs, then the single store. -/
structure Program where
  insts : List FloatInst
  store : StoreInst
deriving Repr

/-- Overwrite one register. -/
def setReg {γ : Type u} (env : String -> Option γ) (r : String) (v : Option γ) :
    String -> Option γ :=
  fun r' => if r' = r then v else env r'

/-- Read an operand: the zero literal, or a register (`none` if it was never written). -/
def readReg {γ : Type u} (env : String -> Option γ) (zero : γ) (r : String) : Option γ :=
  if r = zeroLiteral then some zero else env r

/-- One `fma.rn.f32` over register values. An undefined operand makes the result
undefined. -/
def fmaStep {γ : Type u} (fma : γ -> γ -> γ -> γ) (zero : γ)
    (env : String -> Option γ) (x y z : String) : Option γ := do
  let vx ← readReg env zero x
  let vy ← readReg env zero y
  let vz ← readReg env zero z
  pure (fma vx vy vz)

/-- Contents of the kernel's input buffers, indexed by binary32 element. -/
structure GlobalInputs (β : Type u) where
  weights : Nat -> β
  values : Nat -> β

/-- A global load by block `b`, thread `tid`. A misaligned address or a read of
`out` (which the kernel never reads) gives `none`. -/
def readInput {β : Type u} (mem : GlobalInputs β) (b tid : Nat) (a : AffineAddress) :
    Option β :=
  if a.aligned then
    match a.buffer with
    | .weights => some (mem.weights (a.index b tid))
    | .values => some (mem.values (a.index b tid))
    | .out => none
  else none

/-- Execute one instruction for block `b`, thread `tid`. -/
def execInst {β : Type u} (fma : β -> β -> β -> β) (zero : β) (mem : GlobalInputs β)
    (b tid : Nat) (env : String -> Option β) : FloatInst -> String -> Option β
  | .load dst a => setReg env dst (readInput mem b tid a)
  | .fma dst x y z => setReg env dst (fmaStep fma zero env x y z)

/-- Run the active path for block `b`, thread `tid`, and return the stored value. -/
def runThread {β : Type u} (prog : Program) (fma : β -> β -> β -> β) (zero : β)
    (mem : GlobalInputs β) (b tid : Nat) : Option β :=
  readReg (prog.insts.foldl (execInst fma zero mem b tid) (fun _ => none)) zero
    prog.store.src

/-- Symbolic register contents: the loads and FMAs that produced a value. -/
inductive SymVal where
  | zero
  | load (addr : AffineAddress)
  | fma (x y z : SymVal)
deriving Repr, DecidableEq

def symLoad (a : AffineAddress) : Option SymVal :=
  if a.aligned then
    match a.buffer with
    | .weights => some (.load a)
    | .values => some (.load a)
    | .out => none
  else none

def symInst (env : String -> Option SymVal) : FloatInst -> String -> Option SymVal
  | .load dst a => setReg env dst (symLoad a)
  | .fma dst x y z => setReg env dst (fmaStep SymVal.fma SymVal.zero env x y z)

/-- Symbolic execution of the active path: the expression that gets stored. -/
def symRun (prog : Program) : Option SymVal :=
  readReg (prog.insts.foldl symInst (fun _ => none)) SymVal.zero prog.store.src

/-- Evaluate a symbolic expression for block `b`, thread `tid`. -/
def SymVal.denote {β : Type u} (fmaOp : β -> β -> β -> β) (zero : β)
    (mem : GlobalInputs β) (b tid : Nat) : SymVal -> β
  | .zero => zero
  | .load a =>
      match a.buffer with
      | .weights => mem.weights (a.index b tid)
      | .values => mem.values (a.index b tid)
      | .out => zero
  | .fma x y z =>
      fmaOp (x.denote fmaOp zero mem b tid) (y.denote fmaOp zero mem b tid)
        (z.denote fmaOp zero mem b tid)

section Simulation

variable {β : Type u} (fma : β -> β -> β -> β) (zero : β) (mem : GlobalInputs β)
  (b tid : Nat)

theorem readReg_map {env : String -> Option β} {senv : String -> Option SymVal}
    (h : forall r, env r = (senv r).map (SymVal.denote fma zero mem b tid)) (r : String) :
    readReg env zero r =
      (readReg senv SymVal.zero r).map (SymVal.denote fma zero mem b tid) := by
  unfold readReg
  split
  · simp [SymVal.denote]
  · exact h r

theorem fmaStep_map {env : String -> Option β} {senv : String -> Option SymVal}
    (h : forall r, env r = (senv r).map (SymVal.denote fma zero mem b tid))
    (x y z : String) :
    fmaStep fma zero env x y z =
      (fmaStep SymVal.fma SymVal.zero senv x y z).map
        (SymVal.denote fma zero mem b tid) := by
  unfold fmaStep
  rw [readReg_map fma zero mem b tid h x, readReg_map fma zero mem b tid h y,
    readReg_map fma zero mem b tid h z]
  generalize readReg senv SymVal.zero x = ox
  generalize readReg senv SymVal.zero y = oy
  generalize readReg senv SymVal.zero z = oz
  cases ox <;> cases oy <;> cases oz <;> simp [SymVal.denote]

theorem readInput_map (a : AffineAddress) :
    readInput mem b tid a = (symLoad a).map (SymVal.denote fma zero mem b tid) := by
  unfold readInput symLoad
  split
  · cases hbuf : a.buffer <;> simp [SymVal.denote, hbuf]
  · rfl

theorem execInst_map {env : String -> Option β} {senv : String -> Option SymVal}
    (h : forall r, env r = (senv r).map (SymVal.denote fma zero mem b tid))
    (inst : FloatInst) (r : String) :
    execInst fma zero mem b tid env inst r =
      (symInst senv inst r).map (SymVal.denote fma zero mem b tid) := by
  cases inst with
  | load dst a =>
      simp only [execInst, symInst, setReg]
      split
      · exact readInput_map fma zero mem b tid a
      · exact h r
  | fma dst x y z =>
      simp only [execInst, symInst, setReg]
      split
      · exact fmaStep_map fma zero mem b tid h x y z
      · exact h r

theorem foldl_execInst_map (insts : List FloatInst) :
    forall {env : String -> Option β} {senv : String -> Option SymVal},
      (forall r, env r = (senv r).map (SymVal.denote fma zero mem b tid)) ->
      forall r, insts.foldl (execInst fma zero mem b tid) env r =
        (insts.foldl symInst senv r).map (SymVal.denote fma zero mem b tid) := by
  induction insts with
  | nil =>
      intro env senv h r
      exact h r
  | cons inst rest ih =>
      intro env senv h r
      exact ih (fun r' => execInst_map fma zero mem b tid h inst r') r

/-- Running the program agrees with evaluating its symbolic execution. -/
theorem runThread_eq_symRun (prog : Program) :
    runThread prog fma zero mem b tid =
      (symRun prog).map (SymVal.denote fma zero mem b tid) :=
  readReg_map fma zero mem b tid
    (foldl_execInst_map fma zero mem b tid prog.insts
      (env := fun _ => none) (senv := fun _ => none) (fun _ => rfl)) prog.store.src

end Simulation

/-- `weights[b, t]`: byte `32 b + 4 t` of the weight buffer. -/
def weightAddress (t : Nat) : AffineAddress :=
  { buffer := .weights, rowStrideBytes := 32, laneStrideBytes := 0, offsetBytes := 4 * t }

/-- `v[b, t, tid]`: byte `128 b + 4 tid + 16 t` of the value buffer. -/
def valueAddress (t : Nat) : AffineAddress :=
  { buffer := .values, rowStrideBytes := 128, laneStrideBytes := 4, offsetBytes := 16 * t }

/-- `out[b, tid]`: byte `16 b + 4 tid` of the output buffer. -/
def outAddress : AffineAddress :=
  { buffer := .out, rowStrideBytes := 16, laneStrideBytes := 4, offsetBytes := 0 }

@[simp] theorem weightAddress_buffer (t : Nat) : (weightAddress t).buffer = .weights := rfl

@[simp] theorem valueAddress_buffer (t : Nat) : (valueAddress t).buffer = .values := rfl

@[simp] theorem weightAddress_index (b tid t : Nat) :
    (weightAddress t).index b tid = weightsAddr b t := by
  simp only [weightAddress, AffineAddress.index, AffineAddress.byteOffset, weightsAddr]
  omega

@[simp] theorem valueAddress_index (b tid t : Nat) :
    (valueAddress t).index b tid = valueAddr b t tid := by
  simp only [valueAddress, AffineAddress.index, AffineAddress.byteOffset, valueAddr]
  omega

@[simp] theorem outAddress_index (b tid : Nat) :
    outAddress.index b tid = outAddr b tid := by
  simp only [outAddress, AffineAddress.index, AffineAddress.byteOffset, outAddr]
  omega

/-- FMA chain over keys `0, ..., n - 1`, starting from `+0.0`. Step `t`
multiplies the row's weight `t` by its value `t`, and `expectedChain 8` is what
the store must receive. -/
def expectedChain : Nat -> SymVal
  | 0 => .zero
  | n + 1 => .fma (.load (weightAddress n)) (.load (valueAddress n)) (expectedChain n)

theorem expectedChain_denote {β : Type u} (fma : β -> β -> β -> β) (zero : β)
    (mem : GlobalInputs β) (b tid : Nat) :
    (expectedChain 8).denote fma zero mem b tid =
      valueReduceFMA fma zero
        (fun t => mem.weights (weightsAddr b t))
        (fun t => mem.values (valueAddr b t tid)) := by
  simp [expectedChain, SymVal.denote, valueReduceFMA]

/-- Whether a load reads one coordinate of its own row, `weights[b, t]` or
`v[b, t, tid]` with `t < 8`. FMA instructions always pass. -/
def loadShapeOK : FloatInst -> Bool
  | .load _ a =>
      (a.buffer == .weights && a.rowStrideBytes == 32 && a.laneStrideBytes == 0 &&
        a.offsetBytes % 4 == 0 && decide (a.offsetBytes < 32)) ||
      (a.buffer == .values && a.rowStrideBytes == 128 && a.laneStrideBytes == 4 &&
        a.offsetBytes % 16 == 0 && decide (a.offsetBytes < 128))
  | .fma .. => true

/-- Check a program: symbolic execution stores `expectedChain 8`, the store
targets `out[b, tid]`, and every load has the row shape above. -/
def checkProgram (prog : Program) : Bool :=
  (symRun prog == some (expectedChain 8)) &&
  (prog.store.addr == outAddress) &&
  prog.insts.all loadShapeOK

/-- Propositional form of `checkProgram`. -/
def ProgramContract (prog : Program) : Prop :=
  symRun prog = some (expectedChain 8) ∧
  prog.store.addr = outAddress ∧
  ∀ inst ∈ prog.insts, loadShapeOK inst = true

theorem checkProgram_sound (prog : Program) (h : checkProgram prog = true) :
    ProgramContract prog := by
  simpa [checkProgram, ProgramContract, and_assoc] using h

section Accepted

variable {prog : Program} (hprog : checkProgram prog = true)
include hprog

/-- An accepted program stores the left-to-right FMA reduction over the thread's
own row: weights `weights[b, 0..7]` and values `v[b, 0..7, tid]`. -/
theorem runThread_of_check {β : Type u} (fma : β -> β -> β -> β) (zero : β)
    (mem : GlobalInputs β) (b tid : Nat) :
    runThread prog fma zero mem b tid =
      some (valueReduceFMA fma zero
        (fun t => mem.weights (weightsAddr b t))
        (fun t => mem.values (valueAddr b t tid))) := by
  rw [runThread_eq_symRun, (checkProgram_sound prog hprog).1]
  simp only [Option.map_some, expectedChain_denote]

/-- A load by block `b`, thread `tid` reads inside `weights : [B, 8]` or
`v : [B, 8, 4]`. FMA instructions satisfy it trivially. -/
def LoadInBounds (B b tid : Nat) : FloatInst -> Prop
  | .load _ a =>
      (a.buffer = .weights ∧ a.index b tid < B * 8) ∨
      (a.buffer = .values ∧ a.index b tid < B * 8 * 4)
  | .fma .. => True

omit hprog in
theorem loadInBounds_of_shape {inst : FloatInst} (h : loadShapeOK inst = true)
    {B b tid : Nat} (hb : b < B) (htid : tid < 4) :
    LoadInBounds B b tid inst := by
  cases inst with
  | fma => trivial
  | load dst a =>
      simp only [loadShapeOK, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq,
        decide_eq_true_eq] at h
      rcases h with ⟨⟨⟨⟨hbuf, hrow⟩, hlane⟩, hmod⟩, hlt⟩ | ⟨⟨⟨⟨hbuf, hrow⟩, hlane⟩, hmod⟩, hlt⟩
      · left
        refine ⟨hbuf, ?_⟩
        have hidx : a.index b tid = weightsAddr b (a.offsetBytes / 4) := by
          simp only [AffineAddress.index, AffineAddress.byteOffset, weightsAddr, hrow, hlane]
          omega
        rw [hidx]
        exact weightsAddr_lt hb (by omega)
      · right
        refine ⟨hbuf, ?_⟩
        have hidx : a.index b tid = valueAddr b (a.offsetBytes / 16) tid := by
          simp only [AffineAddress.index, AffineAddress.byteOffset, valueAddr, hrow, hlane]
          omega
        rw [hidx]
        exact valueAddr_lt hb (by omega) htid

/-- Every access of an active thread (`b < B`, `tid < 4`) stays in its buffer. -/
theorem accesses_inBounds_of_check {B b tid : Nat} (hb : b < B) (htid : tid < 4) :
    (∀ inst ∈ prog.insts, LoadInBounds B b tid inst) ∧
      prog.store.addr.index b tid < B * 4 := by
  have hc := checkProgram_sound prog hprog
  refine ⟨fun inst hinst => loadInBounds_of_shape (hc.2.2 inst hinst) hb htid, ?_⟩
  rw [hc.2.1, outAddress_index]
  exact outAddr_lt hb htid

theorem store_index_of_check (b tid : Nat) :
    prog.store.addr.index b tid = outAddr b tid := by
  rw [(checkProgram_sound prog hprog).2.1, outAddress_index]

/-- Two active threads that store to the same word are the same thread. -/
theorem store_owned_of_check {b₁ b₂ tid₁ tid₂ : Nat} (ht₁ : tid₁ < 4) (ht₂ : tid₂ < 4)
    (heq : prog.store.addr.index b₁ tid₁ = prog.store.addr.index b₂ tid₂) :
    b₁ = b₂ ∧ tid₁ = tid₂ := by
  rw [store_index_of_check hprog, store_index_of_check hprog] at heq
  exact outAddr_injective ht₁ ht₂ heq

end Accepted

/-- Active threads of a launch over `B` rows: `blockIdx.x < B` and `threadIdx.x < 4`. -/
def activeThreads (B : Nat) : List (Nat × Nat) :=
  (List.range B).flatMap fun b => (List.range 4).map fun tid => (b, tid)

theorem mem_activeThreads {B b tid : Nat} :
    (b, tid) ∈ activeThreads B ↔ b < B ∧ tid < 4 := by
  simp [activeThreads]

/-- Output word `n` after a launch over `B` rows: the value stored by the active
thread whose store address is `n`. Inactive threads are taken not to access
memory, which is the extractor's recorded guard-path fact
(`guardPathsNoMemoryAccess`, section 4) and not something Lean derives from the
PTX. -/
def kernelOutput {β : Type u} (prog : Program) (fma : β -> β -> β -> β) (zero : β)
    (mem : GlobalInputs β) (B n : Nat) : Option β :=
  match (activeThreads B).find? (fun p => prog.store.addr.index p.1 p.2 == n) with
  | some p => runThread prog fma zero mem p.1 p.2
  | none => none

theorem kernelOutput_of_check {prog : Program} (hprog : checkProgram prog = true)
    {β : Type u} (fma : β -> β -> β -> β) (zero : β) (mem : GlobalInputs β)
    {B b tid : Nat} (hb : b < B) (htid : tid < 4) :
    kernelOutput prog fma zero mem B (outAddr b tid) = runThread prog fma zero mem b tid := by
  unfold kernelOutput
  cases hfind : (activeThreads B).find?
      (fun p => prog.store.addr.index p.1 p.2 == outAddr b tid) with
  | none =>
      exfalso
      rw [List.find?_eq_none] at hfind
      have hnot := hfind (b, tid) (mem_activeThreads.mpr ⟨hb, htid⟩)
      simp [store_index_of_check hprog] at hnot
  | some p =>
      obtain ⟨b', tid'⟩ := p
      have hp := List.find?_some hfind
      have hmem := mem_activeThreads.mp (List.mem_of_find?_eq_some hfind)
      simp only [beq_iff_eq, store_index_of_check hprog] at hp
      obtain ⟨rfl, rfl⟩ := outAddr_injective hmem.2 htid hp
      rfl

/-- One request for the tiny kernel: its eight softmax weights and its `[8, 4]`
value block. -/
structure TinyRequest (β : Type u) where
  weights : Fin 8 -> β
  values : Fin 8 -> Fin 4 -> β

/-- Lay a batch out in the kernel's row-major buffers `weights : [B, 8]` and
`v : [B, 8, 4]`. Words past the batch hold `pad`. -/
def layoutBatch {β : Type u} (pad : β) {B : Nat} (reqs : Fin B -> TinyRequest β) :
    GlobalInputs β where
  weights n :=
    if h : n / 8 < B then
      (reqs ⟨n / 8, h⟩).weights ⟨n % 8, Nat.mod_lt _ (by decide)⟩
    else pad
  values n :=
    if h : n / 32 < B then
      (reqs ⟨n / 32, h⟩).values ⟨n % 32 / 4, by omega⟩ ⟨n % 4, Nat.mod_lt _ (by decide)⟩
    else pad

theorem layoutBatch_weights {β : Type u} (pad : β) {B : Nat}
    (reqs : Fin B -> TinyRequest β) (b : Fin B) (t : Fin 8) :
    (layoutBatch pad reqs).weights (weightsAddr b t) = (reqs b).weights t := by
  have hb := b.isLt
  have ht := t.isLt
  have hrow : (b.val * 8 + t.val) / 8 = b.val := by omega
  have hcol : (b.val * 8 + t.val) % 8 = t.val := by omega
  simp [layoutBatch, weightsAddr, hrow, hcol]

theorem layoutBatch_values {β : Type u} (pad : β) {B : Nat}
    (reqs : Fin B -> TinyRequest β) (b : Fin B) (t : Fin 8) (o : Fin 4) :
    (layoutBatch pad reqs).values (valueAddr b t o) = (reqs b).values t o := by
  have hb := b.isLt
  have ht := t.isLt
  have ho := o.isLt
  have hrow : ((b.val * 8 + t.val) * 4 + o.val) / 32 = b.val := by omega
  have hkey : ((b.val * 8 + t.val) * 4 + o.val) % 32 / 4 = t.val := by omega
  have hcol : ((b.val * 8 + t.val) * 4 + o.val) % 4 = o.val := by omega
  simp [layoutBatch, valueAddr, hrow, hkey, hcol]

/-- The kernel as a batched forward pass: request `i` of a batch of `B` is row `i`
of the buffers, and its output coordinate `o` is the word `out[i, o]` after the
launch. -/
def tinyKernelForward {β : Type u} (prog : Program) (fma : β -> β -> β -> β) (zero : β) :
    BatchInvariantInference.BatchedForward (TinyRequest β) (Fin 4 -> Option β) :=
  fun B reqs i o => kernelOutput prog fma zero (layoutBatch zero reqs) B (outAddr i o)

/-- The value-reduction spec as a batched forward pass. It reads only the selected
request. -/
def tinyValueReductionSpec {β : Type u} (fma : β -> β -> β -> β) (zero : β) :
    BatchInvariantInference.BatchedForward (TinyRequest β) (Fin 4 -> Option β) :=
  BatchInvariantInference.pointwiseForward fun req o =>
    some (valueReduceFMA fma zero req.weights (fun t => req.values t o))

/-- An accepted program refines the spec on every batch. -/
theorem tinyKernelForward_refines_of_check {prog : Program}
    (hprog : checkProgram prog = true) {β : Type u} (fma : β -> β -> β -> β) (zero : β) :
    BatchInvariantInference.RefinesForward
      (tinyKernelForward prog fma zero) (tinyValueReductionSpec fma zero) := by
  intro B reqs i
  funext o
  simp only [tinyKernelForward, tinyValueReductionSpec,
    BatchInvariantInference.pointwiseForward]
  rw [kernelOutput_of_check hprog fma zero _ i.isLt o.isLt,
    runThread_of_check hprog]
  simp only [layoutBatch_weights, layoutBatch_values]

end TinyPTX
end CUDA


/-!
# 4. CUDA/PTX/SASS Certificate

The finite certificate emitted by
`week-01-batch-invariant-inference/cuda/extract_cert.py`. It records the source,
PTX, CUBIN, and SASS hashes and the toolkit version (audit metadata, since Lean
does not read files on disk), the extractor's instruction counts, text checks,
and guard checks, and the extracted active-path program from section 3.

Lean checks the program itself (`checkProgram`). The counts and booleans are
facts the extractor computed: regular-expression counts of PTX instructions,
substring tests on the source, PTX, and SASS text, and the four-case guard
trace. For those, Lean only requires the recorded counts to be 8, 16, and 1 and
every boolean to be true.
-/

namespace CUDA
namespace TinyAttentionCert

open CUDA.TinyPTX

/-- Source, PTX, and SASS facts extracted for `tiny_attn_one_row.cu`, together with
the active-path PTX program. -/
structure TinyAttentionKernelCert where
  /-- Repo-relative CUDA source path, kept as audit metadata. -/
  sourcePath : String
  /-- SHA-256 of the source file, as a string. -/
  sourceSha256 : String
  ptxPath : String
  ptxSha256 : String
  cubinPath : String
  cubinSha256 : String
  sassPath : String
  sassSha256 : String
  /-- Toolkit line from the PTX header, for example
  `Cuda compilation tools, release 13.2, V13.2.78`. -/
  toolkit : String
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
  /-- The extractor followed all four combinations of the batch guard `b < B` and
  the thread guard `tid < 4`. On the three inactive ones the PTX returns without
  a global memory access. -/
  guardPathsNoMemoryAccess : Bool
  /-- Active-path program, checked by Lean. -/
  program : Program
deriving Repr

/-- Contract for a tiny value-reduction certificate at target `expectedArch`: the
architecture matches, the PTX counts are 8 FMAs, 16 loads, and 1 store, every
recorded text and guard fact is true, and the program satisfies
`ProgramContract`. -/
def TinyAttentionKernelContractFor
    (expectedArch : String) (cert : TinyAttentionKernelCert) : Prop :=
  cert.arch = expectedArch ∧
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
  cert.guardPathsNoMemoryAccess = true ∧
  ProgramContract cert.program

/-- Boolean check of the whole certificate against target `expectedArch`. -/
def checkTinyAttentionKernelCertFor
    (expectedArch : String) (cert : TinyAttentionKernelCert) : Bool :=
  (cert.arch == expectedArch) &&
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
  cert.guardPathsNoMemoryAccess &&
  checkProgram cert.program

theorem checkTinyAttentionKernelCertFor_program
    (expectedArch : String)
    (cert : TinyAttentionKernelCert)
    (hcheck : checkTinyAttentionKernelCertFor expectedArch cert = true) :
    checkProgram cert.program = true := by
  unfold checkTinyAttentionKernelCertFor at hcheck
  simp only [Bool.and_eq_true] at hcheck
  exact hcheck.2

theorem checkTinyAttentionKernelCertFor_sound
    (expectedArch : String)
    (cert : TinyAttentionKernelCert)
    (hcheck : checkTinyAttentionKernelCertFor expectedArch cert = true) :
    TinyAttentionKernelContractFor expectedArch cert := by
  have hprog := checkProgram_sound cert.program
    (checkTinyAttentionKernelCertFor_program expectedArch cert hcheck)
  unfold checkTinyAttentionKernelCertFor at hcheck
  unfold TinyAttentionKernelContractFor
  simp only [Bool.and_eq_true, beq_iff_eq] at hcheck
  simp only [hcheck, hprog, and_self]

end TinyAttentionCert
end CUDA


/-!
# 5. What An Accepted Certificate Gives

Consequences of `checkTinyAttentionKernelCertFor` for an accepted certificate.
They hold for every batch size and every input. Applying them to a live launch
needs the assumptions in `cuda/README.md`, which stay outside Lean.
-/

namespace CUDA
namespace TinyPTXSemantics

open CUDA.Memory
open CUDA.TinyAttentionCert
open CUDA.TinyAttentionSpec
open CUDA.TinyPTX
open BatchInvariantInference

/-- An accepted certificate's program stores, for block `b` and thread `tid`, the
left-to-right FMA reduction over `weights[b, 0..7]` and `v[b, 0..7, tid]`. -/
theorem kernelCert_denotes_valueReduceFMA
    (expectedArch : String)
    (cert : TinyAttentionKernelCert)
    (hcheck : checkTinyAttentionKernelCertFor expectedArch cert = true)
    (fma : β -> β -> β -> β)
    (zero : β)
    (mem : GlobalInputs β)
    (b tid : Nat) :
    runThread cert.program fma zero mem b tid =
      some (valueReduceFMA fma zero
        (fun t => mem.weights (weightsAddr b t))
        (fun t => mem.values (valueAddr b t tid))) :=
  runThread_of_check (checkTinyAttentionKernelCertFor_program expectedArch cert hcheck)
    fma zero mem b tid

/-- Every access of an active thread stays inside the `[B, 8]`, `[B, 8, 4]`, and
`[B, 4]` buffers. -/
theorem kernelCert_accesses_inBounds
    (expectedArch : String)
    (cert : TinyAttentionKernelCert)
    (hcheck : checkTinyAttentionKernelCertFor expectedArch cert = true)
    {B b tid : Nat} (hb : b < B) (htid : tid < 4) :
    (∀ inst ∈ cert.program.insts, LoadInBounds B b tid inst) ∧
      cert.program.store.addr.index b tid < B * 4 :=
  accesses_inBounds_of_check
    (checkTinyAttentionKernelCertFor_program expectedArch cert hcheck) hb htid

/-- Distinct active threads store to distinct output words. -/
theorem kernelCert_store_owned
    (expectedArch : String)
    (cert : TinyAttentionKernelCert)
    (hcheck : checkTinyAttentionKernelCertFor expectedArch cert = true)
    {b₁ b₂ tid₁ tid₂ : Nat} (ht₁ : tid₁ < 4) (ht₂ : tid₂ < 4)
    (heq : cert.program.store.addr.index b₁ tid₁ = cert.program.store.addr.index b₂ tid₂) :
    b₁ = b₂ ∧ tid₁ = tid₂ :=
  store_owned_of_check
    (checkTinyAttentionKernelCertFor_program expectedArch cert hcheck) ht₁ ht₂ heq

/-- Batched forward pass of the certificate's kernel, so it plugs into the same
`BatchInvariantForward` contract as the model-level theorems. -/
def tinyValueReductionForward
    (cert : TinyAttentionKernelCert)
    (fma : β -> β -> β -> β)
    (zero : β) :
    BatchedForward (TinyRequest β) (Fin 4 -> Option β) :=
  tinyKernelForward cert.program fma zero

/-- The refinement statement exported by the tiny certificate: on every batch, the
output word `out[i, o]` holds the FMA chain over request `i`'s own data. -/
def RefinesTinyValueReduction
    (cert : TinyAttentionKernelCert)
    (fma : β -> β -> β -> β)
    (zero : β) : Prop :=
  RefinesForward (tinyValueReductionForward cert fma zero) (tinyValueReductionSpec fma zero)

theorem kernelCert_refines_tinyValueReduction
    (expectedArch : String)
    (cert : TinyAttentionKernelCert)
    (hcheck : checkTinyAttentionKernelCertFor expectedArch cert = true)
    (fma : β -> β -> β -> β)
    (zero : β) :
    RefinesTinyValueReduction cert fma zero :=
  tinyKernelForward_refines_of_check
    (checkTinyAttentionKernelCertFor_program expectedArch cert hcheck) fma zero

/-- The certified kernel is batch-invariant: the check gives refinement of the
pointwise spec, and refinement transfers batch invariance. -/
theorem tinyValueReductionForward_batchInvariant_of_check
    (expectedArch : String)
    (cert : TinyAttentionKernelCert)
    (hcheck : checkTinyAttentionKernelCertFor expectedArch cert = true)
    (fma : β -> β -> β -> β)
    (zero : β) :
    BatchInvariantForward (tinyValueReductionForward cert fma zero) :=
  batchInvariant_of_refinesForward _ _
    (kernelCert_refines_tinyValueReduction expectedArch cert hcheck fma zero)
    (pointwise_batchInvariant _)

end TinyPTXSemantics
end CUDA


/-!
# 6. Soundness Bridge To TorchLean Attention

Two pieces of evidence sit side by side here. The extracted certificate
describes the tiny value-reduction kernel, and its checker gives the program
contract above. The native forward certificate carries its own proof of equality
with TorchLean's FlashAttention denotation, since checking the FMA chain says
nothing about full attention.
-/

namespace CUDA
namespace TinyAttentionSound

open BatchInvariantInference
open Spec
open TorchLean (Storage)

/-- A checked native forward certificate combines:

1. an extracted certificate for the compiled CUDA microkernel
2. Lean's check that the certificate is accepted for `expectedArch`
3. a runtime refinement certificate to TorchLean FlashAttention

Each field stands on its own, and `refines_flashAttention` reads only
`nativeCert`. -/
structure CheckedNativeForwardCert
    (expectedArch : String)
    (α : Type) [Storage α] [Context α] [DecidableRel ((· > ·) : α -> α -> Prop)]
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0} where
  kernelCert : CUDA.TinyAttentionCert.TinyAttentionKernelCert
  kernelChecked :
    CUDA.TinyAttentionCert.checkTinyAttentionKernelCertFor expectedArch kernelCert = true
  nativeCert :
    TorchLeanFlashAttention.NativeForwardCert α
      (nQ := nQ) (nK := nK) (dModel := dModel) (h1 := h1) (h2 := h2)

theorem CheckedNativeForwardCert.kernel_contract
    {expectedArch : String}
    {α : Type} [Storage α] [Context α] [DecidableRel ((· > ·) : α -> α -> Prop)]
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (cert : CheckedNativeForwardCert expectedArch α
      (nQ := nQ) (nK := nK) (dModel := dModel) (h1 := h1) (h2 := h2)) :
    CUDA.TinyAttentionCert.TinyAttentionKernelContractFor expectedArch cert.kernelCert :=
  CUDA.TinyAttentionCert.checkTinyAttentionKernelCertFor_sound
    expectedArch cert.kernelCert cert.kernelChecked

theorem CheckedNativeForwardCert.kernel_refines_valueReduction
    {expectedArch : String}
    {α : Type} [Storage α] [Context α] [DecidableRel ((· > ·) : α -> α -> Prop)]
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (cert : CheckedNativeForwardCert expectedArch α
      (nQ := nQ) (nK := nK) (dModel := dModel) (h1 := h1) (h2 := h2))
    (fma : β -> β -> β -> β)
    (zero : β) :
    CUDA.TinyPTXSemantics.RefinesTinyValueReduction cert.kernelCert fma zero :=
  CUDA.TinyPTXSemantics.kernelCert_refines_tinyValueReduction
    expectedArch cert.kernelCert cert.kernelChecked fma zero

theorem CheckedNativeForwardCert.refines_flashAttention
    {expectedArch : String}
    {α : Type} [Storage α] [Context α] [DecidableRel ((· > ·) : α -> α -> Prop)]
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (cert : CheckedNativeForwardCert expectedArch α
      (nQ := nQ) (nK := nK) (dModel := dModel) (h1 := h1) (h2 := h2)) :
    cert.nativeCert.runtimeOut =
      Spec.flashAttention cert.nativeCert.cfg cert.nativeCert.ctx :=
  cert.nativeCert.refines_flashAttention

end TinyAttentionSound
end CUDA
