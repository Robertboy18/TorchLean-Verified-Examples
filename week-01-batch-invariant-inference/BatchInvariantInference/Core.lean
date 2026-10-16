module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured.NativeDispatch
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Instances
public import FloatLib.Numerics.Reduction.Tree
public import NN.IR.Check
public import NN.IR.Semantics
public import NN.Spec.Layers.FlashAttention

/-!
Batch-invariant inference as a TorchLean proof development.

The question behind the file is simple. When the same user request is served in
different batches, what has to stay fixed for the returned token stream to stay
fixed?

We answer it in layers. First, reduction schedules become trees that the
computation takes as input. Next, a binary32 witness shows that a schedule change
is observable. Then we prove the same request-local contract for matmul,
RMSNorm, attention schedules, greedy token choice under margins, and a
decode/verify/rollback serving loop. Concrete CUDA/Triton kernels sit one layer
below this file and must refine the semantics here (`CUDA.lean` checks one small
kernel).
-/

@[expose] public section

universe u v w x

open TorchLean (Storage Tensor)

namespace BatchInvariantInference

open FloatLib.Floats (ExecFloat)
open FloatLib.Numerics (ReductionTree)

/-! FloatLib's `ReductionTree` records the order in which partial results are
combined. A schedule works with any scalar type and combiner, and only the
non-associativity witness below fixes IEEE binary32 arithmetic. -/

/-- A batched forward pass, seen from one selected request.

A batch `Fin B -> X` holds `B` request-local states, and the selected request
is an index `Fin B`, so it is always in bounds. -/
abbrev BatchedForward (X : Type u) (Y : Type v) :=
  (B : Nat) -> (Fin B -> X) -> Fin B -> Y

/-- Batch invariance of a forward pass.

Two selected requests with the same local state get the same output, whatever
batches surround them. -/
def BatchInvariantForward {X : Type u} {Y : Type v}
    (forward : BatchedForward X Y) : Prop :=
  forall {B C : Nat}
    (xs : Fin B -> X) (ys : Fin C -> X)
    (i : Fin B) (j : Fin C),
      xs i = ys j -> forward B xs i = forward C ys j

/-- `runtime` refines `spec` when the two agree on every batch and every
selected request. A CUDA or Triton kernel connects to the semantics through this
relation. -/
def RefinesForward {X : Type u} {Y : Type v}
    (runtime spec : BatchedForward X Y) : Prop :=
  forall {B : Nat} (xs : Fin B -> X) (i : Fin B),
    runtime B xs i = spec B xs i

/-- Batch invariance transfers along a refinement, which turns a kernel
certificate into a batch-invariance guarantee for the kernel. -/
theorem batchInvariant_of_refinesForward {X : Type u} {Y : Type v}
    (runtime spec : BatchedForward X Y)
    (hrefine : RefinesForward runtime spec)
    (hspec : BatchInvariantForward spec) :
    BatchInvariantForward runtime := by
  intro B C xs ys i j hselected
  calc
    runtime B xs i = spec B xs i := hrefine xs i
    _ = spec C ys j := hspec xs ys i j hselected
    _ = runtime C ys j := (hrefine ys j).symm

/-- Reduction of the selected row over the tree `sched`. Other rows are in the
batch, but only the selected one is read. -/
def batchedReduce {ι : Type u} {β : Type v} {B : Nat}
    (combine : β -> β -> β)
    (sched : ReductionTree ι)
    (rows : Fin B -> ι -> β)
    (b : Fin B) : β :=
  sched.eval combine (rows b)

/-- Selected rows with the same leaves, reduced over the same tree, give the
same result in any surrounding batch. -/
theorem batchedReduce_batchInvariant {ι : Type u} {β : Type v}
    {B C : Nat}
    (combine : β -> β -> β)
    (sched : ReductionTree ι)
    (rows1 : Fin B -> ι -> β)
    (rows2 : Fin C -> ι -> β)
    (b : Fin B) (c : Fin C)
    (hrow : forall k, rows1 b k = rows2 c k) :
    batchedReduce combine sched rows1 b =
      batchedReduce combine sched rows2 c := by
  exact congrArg (fun row => sched.eval combine row) (funext hrow)

/-- A runtime policy that picks a reduction schedule from the batch size and the
selected index. -/
abbrev ScheduleSelector (ι : Type u) :=
  (B : Nat) -> Fin B -> ReductionTree ι

/-- Every selected row gets the same schedule, whatever its batch size or
position. -/
def BatchIndependentSchedule {ι : Type u} (choose : ScheduleSelector ι) : Prop :=
  forall {B C : Nat} (i : Fin B) (j : Fin C), choose B i = choose C j

/-- Request-local metadata a kernel may use when it picks a schedule. Different
requests may get different schedules from these local facts, but an unrelated
co-batched request must not change the choice. -/
structure LocalKernelCtx where
  hiddenDim : Nat
  seqLen : Nat
  kvLen : Nat
  dtypeTag : Nat
  blockSize : Nat
deriving Repr, DecidableEq

/-- A schedule selector that also receives the selected request's local
metadata. -/
abbrev ScheduleSelectorWithCtx (ι : Type u) :=
  (B : Nat) -> Fin B -> LocalKernelCtx -> ReductionTree ι

/-- Request-local schedule invariance: selected requests with equal local
contexts get the same schedule, whatever batch they sit in. -/
def RequestLocalScheduleInvariant {ι : Type u}
    (choose : ScheduleSelectorWithCtx ι) : Prop :=
  forall {B C : Nat} (i : Fin B) (j : Fin C)
    (ctx1 ctx2 : LocalKernelCtx),
      ctx1 = ctx2 -> choose B i ctx1 = choose C j ctx2

/-- Reduction where the schedule is selected from the batch context. -/
def reduceWithScheduleSelector {ι : Type u} {β : Type v} {B : Nat}
    (combine : β -> β -> β)
    (choose : ScheduleSelector ι)
    (rows : Fin B -> ι -> β)
    (b : Fin B) : β :=
  (choose B b).eval combine (rows b)

/-- A selected-row reduction as a batched forward pass. Each request's input is
its row of reduction leaves. -/
def reduceForwardWithScheduleSelector {ι : Type u} {β : Type v}
    (combine : β -> β -> β)
    (choose : ScheduleSelector ι) : BatchedForward (ι -> β) β :=
  fun _ rows b => reduceWithScheduleSelector combine choose rows b

/-- Reduction where schedule selection may depend on request-local metadata. -/
def reduceWithCtxSelector {ι : Type u} {β : Type v} {B : Nat}
    (combine : β -> β -> β)
    (choose : ScheduleSelectorWithCtx ι)
    (rows : Fin B -> ι -> β)
    (ctxs : Fin B -> LocalKernelCtx)
    (b : Fin B) : β :=
  (choose B b (ctxs b)).eval combine (rows b)

/-- With a batch-independent selector, equal selected rows reduce to equal
results. -/
theorem reduce_batchInvariant_of_batchIndependentSchedule {ι : Type u} {β : Type v}
    {B C : Nat}
    (combine : β -> β -> β)
    (choose : ScheduleSelector ι)
    (hchoose : BatchIndependentSchedule choose)
    (rows1 : Fin B -> ι -> β)
    (rows2 : Fin C -> ι -> β)
    (b : Fin B) (c : Fin C)
    (hrow : forall k, rows1 b k = rows2 c k) :
    reduceWithScheduleSelector combine choose rows1 b =
      reduceWithScheduleSelector combine choose rows2 c := by
  unfold reduceWithScheduleSelector
  rw [hchoose b c]
  exact batchedReduce_batchInvariant combine (choose C c) rows1 rows2 b c hrow

theorem reduceForward_batchInvariant_of_batchIndependentSchedule
    {ι : Type u} {β : Type v}
    (combine : β -> β -> β)
    (choose : ScheduleSelector ι)
    (hchoose : BatchIndependentSchedule choose) :
    BatchInvariantForward (reduceForwardWithScheduleSelector combine choose) := by
  intro B C rows1 rows2 b c hrow
  exact reduce_batchInvariant_of_batchIndependentSchedule
    combine choose hchoose rows1 rows2 b c (by
      intro k
      exact congrFun hrow k)

/-- Context-aware version of `reduce_batchInvariant_of_batchIndependentSchedule`,
closer to real kernels. Here the schedule may vary with the request's metadata,
and equal rows with equal metadata still reduce to equal results. -/
theorem reduce_batchInvariant_of_requestLocalScheduleInvariant
    {ι : Type u} {β : Type v} {B C : Nat}
    (combine : β -> β -> β)
    (choose : ScheduleSelectorWithCtx ι)
    (hchoose : RequestLocalScheduleInvariant choose)
    (rows1 : Fin B -> ι -> β)
    (rows2 : Fin C -> ι -> β)
    (ctxs1 : Fin B -> LocalKernelCtx)
    (ctxs2 : Fin C -> LocalKernelCtx)
    (b : Fin B) (c : Fin C)
    (hctx : ctxs1 b = ctxs2 c)
    (hrow : forall k, rows1 b k = rows2 c k) :
    reduceWithCtxSelector combine choose rows1 ctxs1 b =
      reduceWithCtxSelector combine choose rows2 ctxs2 c := by
  unfold reduceWithCtxSelector
  rw [hchoose b c (ctxs1 b) (ctxs2 c) hctx]
  exact batchedReduce_batchInvariant combine (choose C c (ctxs2 c)) rows1 rows2 b c hrow

/-- Left-associated three-term reduction, `combine (combine a b) c`. -/
def leftAssoc3 {β : Type v} (combine : β -> β -> β) (a b c : β) : β :=
  ReductionTree.eval combine id
    (ReductionTree.node (ReductionTree.node (ReductionTree.leaf a) (ReductionTree.leaf b)) (ReductionTree.leaf c))

/-- Right-associated three-term reduction, `combine a (combine b c)`. -/
def rightAssoc3 {β : Type v} (combine : β -> β -> β) (a b c : β) : β :=
  ReductionTree.eval combine id
    (ReductionTree.node (ReductionTree.leaf a) (ReductionTree.node (ReductionTree.leaf b) (ReductionTree.leaf c)))

/-- A combiner that is non-associative at `a, b, c` gives different left- and
right-associated results. -/
theorem different_schedules_can_differ {β : Type v}
    (combine : β -> β -> β) (a b c : β)
    (h : combine (combine a b) c ≠ combine a (combine b c)) :
    leftAssoc3 combine a b c ≠ rightAssoc3 combine a b c := by
  change combine (combine a b) c ≠ combine a (combine b c)
  exact h

/-- A row with the three leaves `a, b, c`.

Three terms are enough to witness non-associativity, and
`IEEE32_batchDependentSchedule_counterexample` fills them with binary32 values
from FloatLib's executable model. -/
def row3 {β : Type v} (a b c : β) : Fin 3 -> β
  | ⟨0, _⟩ => a
  | ⟨1, _⟩ => b
  | _ => c

def leftTree3 : ReductionTree (Fin 3) :=
  ReductionTree.node (ReductionTree.node (ReductionTree.leaf (0 : Fin 3)) (ReductionTree.leaf (1 : Fin 3)))
    (ReductionTree.leaf (2 : Fin 3))

def rightTree3 : ReductionTree (Fin 3) :=
  ReductionTree.node (ReductionTree.leaf (0 : Fin 3))
    (ReductionTree.node (ReductionTree.leaf (1 : Fin 3)) (ReductionTree.leaf (2 : Fin 3)))

/-- A batch-size-dependent selector: a batch of one gets the left tree, and any
larger batch gets the right tree. It models the runtime bug and is not a
recommended policy. -/
def chooseByBatchSize3 : ScheduleSelector (Fin 3) :=
  fun B _ => if B = 1 then leftTree3 else rightTree3

namespace IEEE32

abbrev Exec := ExecFloat.Binary (exponentBits := 8) (fractionBits := 23)

/-- Nearest binary32 value to `1e20`, given by its bit pattern. -/
def big : Exec := ExecFloat.Binary.ofBits32 (0x60AD78EC : UInt32)

/-- Its negation, the binary32 value nearest `-1e20`. -/
def negBig : Exec := ExecFloat.Binary.ofBits32 (0xE0AD78EC : UInt32)

/-- `1.0` in binary32. -/
def one : Exec := ExecFloat.Binary.ofBits32 (0x3F800000 : UInt32)

/-- Non-associativity of addition in FloatLib's executable binary32 model:
`(1e20 + -1e20) + 1 = 1`, while `1e20 + (-1e20 + 1) = 0`. -/
theorem add_nonassoc_witness :
    ExecFloat.add
      (ExecFloat.add big negBig) one ≠
      ExecFloat.add big
        (ExecFloat.add negBig one) := by
  decide

end IEEE32

/-- Any non-associativity witness for the combiner makes `chooseByBatchSize3`
fail batch invariance. This is the bug in formal terms: the selected row stays
the same, but moving it from a batch of one to a batch of two changes its
reduction schedule. -/
theorem batchDependentSchedule_counterexample {β : Type v}
    (combine : β -> β -> β) (a b c : β)
    (h : combine (combine a b) c ≠ combine a (combine b c)) :
    ¬ BatchInvariantForward
      (reduceForwardWithScheduleSelector combine chooseByBatchSize3) := by
  intro hInv
  let row := row3 a b c
  let xs : Fin 1 -> Fin 3 -> β := fun _ => row
  let ys : Fin 2 -> Fin 3 -> β := fun _ => row
  have hEq := hInv xs ys (0 : Fin 1) (0 : Fin 2) rfl
  have hright :
      ReductionTree.eval combine (row3 a b c) rightTree3 =
        combine a (combine b c) := by
    rfl
  simp only [reduceForwardWithScheduleSelector, reduceWithScheduleSelector,
    chooseByBatchSize3, xs, ys, row] at hEq
  norm_num at hEq
  rw [hright] at hEq
  exact h hEq

/-- Under FloatLib's executable binary32 addition, `chooseByBatchSize3` is not
batch-invariant. -/
theorem IEEE32_batchDependentSchedule_counterexample :
    ¬ BatchInvariantForward
      (reduceForwardWithScheduleSelector
        (ExecFloat.add : IEEE32.Exec → IEEE32.Exec → IEEE32.Exec)
        chooseByBatchSize3) := by
  exact batchDependentSchedule_counterexample
    ExecFloat.add
    IEEE32.big IEEE32.negBig IEEE32.one
    IEEE32.add_nonassoc_witness

/-- Dot product of `x` and `w`, summed over the tree `sched`. A matrix product is
many of these. -/
def scheduledDot {K : Type u} {β : Type v}
    (add mul : β -> β -> β)
    (sched : ReductionTree K)
    (x : K -> β)
    (w : K -> β) : β :=
  sched.eval add (fun k => mul (x k) (w k))

/-- Output `o` of the selected row of a batched matrix product. -/
def batchedMatmul {K : Type u} {Out : Type v} {β : Type w} {B : Nat}
    (add mul : β -> β -> β)
    (sched : ReductionTree K)
    (x : Fin B -> K -> β)
    (w : Out -> K -> β)
    (b : Fin B)
    (o : Out) : β :=
  scheduledDot add mul sched (x b) (w o)

/-- Selected matmul outputs agree when the input rows and the weights for output
`o` agree and the reduction tree is shared. -/
theorem matmul_batchInvariant {K : Type u} {Out : Type v} {β : Type w}
    {B C : Nat}
    (add mul : β -> β -> β)
    (sched : ReductionTree K)
    (x1 : Fin B -> K -> β)
    (x2 : Fin C -> K -> β)
    (w1 w2 : Out -> K -> β)
    (i : Fin B) (j : Fin C) (o : Out)
    (hrow : forall k, x1 i k = x2 j k)
    (hweight : forall k, w1 o k = w2 o k) :
    batchedMatmul add mul sched x1 w1 i o =
      batchedMatmul add mul sched x2 w2 j o := by
  unfold batchedMatmul scheduledDot
  rw [funext hrow, funext hweight]

/-- Selected matmul output when the runtime picks the reduction schedule from the
batch context and request-local metadata. -/
def batchedMatmulWithCtxSelector {K : Type u} {Out : Type v} {β : Type w} {B : Nat}
    (add mul : β -> β -> β)
    (choose : ScheduleSelectorWithCtx K)
    (x : Fin B -> K -> β)
    (ctxs : Fin B -> LocalKernelCtx)
    (w : Out -> K -> β)
    (b : Fin B)
    (o : Out) : β :=
  scheduledDot add mul (choose B b (ctxs b)) (x b) (w o)

/-- Matmul batch invariance under a request-local schedule policy: the selected
row, its metadata, and the weights fix the output, whatever else is in the
batch. -/
theorem matmul_batchInvariant_of_requestLocalSchedule
    {K : Type u} {Out : Type v} {β : Type w} {B C : Nat}
    (add mul : β -> β -> β)
    (choose : ScheduleSelectorWithCtx K)
    (hchoose : RequestLocalScheduleInvariant choose)
    (x1 : Fin B -> K -> β)
    (x2 : Fin C -> K -> β)
    (ctxs1 : Fin B -> LocalKernelCtx)
    (ctxs2 : Fin C -> LocalKernelCtx)
    (w : Out -> K -> β)
    (i : Fin B) (j : Fin C) (o : Out)
    (hctx : ctxs1 i = ctxs2 j)
    (hrow : forall k, x1 i k = x2 j k) :
    batchedMatmulWithCtxSelector add mul choose x1 ctxs1 w i o =
      batchedMatmulWithCtxSelector add mul choose x2 ctxs2 w j o := by
  unfold batchedMatmulWithCtxSelector
  rw [hchoose i j (ctxs1 i) (ctxs2 j) hctx]
  exact matmul_batchInvariant add mul (choose C j (ctxs2 j)) x1 x2 w w i j o hrow
    (fun _ => rfl)

/-- Row `b` of a TorchLean `[B, K]` tensor as a function, so the function-level
lemmas above apply to shape-indexed tensors. -/
def tensorRow {α : Type} [TorchLean.Storage α] {B K : Nat}
    (x : TorchLean.Tensor α (.dim B (.dim K .scalar))) (b : Fin B) : Fin K -> α :=
  Spec.get2 x b

/-- Row `o` of a TorchLean `[Out, K]` weight matrix as a function. -/
def tensorWeightRow {α : Type} [TorchLean.Storage α] {Out K : Nat}
    (w : TorchLean.Tensor α (.dim Out (.dim K .scalar))) (o : Fin Out) : Fin K -> α :=
  fun k => (Spec.get (Spec.get w o) k).item

/-- Output `o` of row `b` of a tensor matmul, reduced over `sched`. -/
def tensorBatchedMatmul {β : Type} [TorchLean.Storage β] {B K Out : Nat}
    (add mul : β -> β -> β)
    (sched : ReductionTree (Fin K))
    (x : TorchLean.Tensor β (.dim B (.dim K .scalar)))
    (w : TorchLean.Tensor β (.dim Out (.dim K .scalar)))
    (b : Fin B)
    (o : Fin Out) : β :=
  scheduledDot add mul sched (tensorRow x b) (tensorWeightRow w o)

/-- Tensor version of `matmul_batchInvariant`, for selected input rows that are
equal as shape-indexed tensors and one shared weight tensor. -/
theorem tensorMatmul_batchInvariant {β : Type} [TorchLean.Storage β] {B C K Out : Nat}
    (add mul : β -> β -> β)
    (sched : ReductionTree (Fin K))
    (x1 : TorchLean.Tensor β (.dim B (.dim K .scalar)))
    (x2 : TorchLean.Tensor β (.dim C (.dim K .scalar)))
    (w : TorchLean.Tensor β (.dim Out (.dim K .scalar)))
    (i : Fin B) (j : Fin C) (o : Fin Out)
    (hrow : Spec.get x1 i = Spec.get x2 j) :
    tensorBatchedMatmul add mul sched x1 w i o =
      tensorBatchedMatmul add mul sched x2 w j o := by
  unfold tensorBatchedMatmul
  apply matmul_batchInvariant
  · intro k
    unfold tensorRow Spec.get2
    rw [hrow]
  · intro k
    rfl

/-- RMSNorm of one row, with the sum of squares reduced over `sched`. `scaleMean`
and `rsqrt` stay abstract, so the definition covers real arithmetic,
rounded-real FP32 models, and executable IEEE-style scalars. -/
def rmsNormRow {ι : Type u} {β : Type v}
    (add mul : β -> β -> β)
    (scaleMean rsqrt : β -> β)
    (sched : ReductionTree ι)
    (x weight : ι -> β) : ι -> β :=
  let sqsum := sched.eval add (fun k => mul (x k) (x k))
  let invRms := rsqrt (scaleMean sqsum)
  fun j => mul (mul (x j) invRms) (weight j)

/-- Equal rows and weights give equal RMSNorm outputs under the same
hidden-dimension tree. -/
theorem rmsNorm_batchInvariant {ι : Type u} {β : Type v}
    (add mul : β -> β -> β)
    (scaleMean rsqrt : β -> β)
    (sched : ReductionTree ι)
    (x1 x2 weight1 weight2 : ι -> β)
    (hrow : forall k, x1 k = x2 k)
    (hweight : forall k, weight1 k = weight2 k) :
    rmsNormRow add mul scaleMean rsqrt sched x1 weight1 =
      rmsNormRow add mul scaleMean rsqrt sched x2 weight2 := by
  funext j
  induction sched with
  | leaf k =>
      simp [rmsNormRow, ReductionTree.eval, hrow, hweight]
  | node l r ihL ihR =>
      simp [rmsNormRow, ReductionTree.eval, hrow, hweight]

/-- RMSNorm of a selected batch row, with the hidden-dimension schedule picked
from the batch context and request-local metadata. -/
def batchedRMSNormWithCtxSelector {ι : Type u} {β : Type v} {B : Nat}
    (add mul : β -> β -> β)
    (scaleMean rsqrt : β -> β)
    (choose : ScheduleSelectorWithCtx ι)
    (x : Fin B -> ι -> β)
    (ctxs : Fin B -> LocalKernelCtx)
    (weight : ι -> β)
    (b : Fin B) : ι -> β :=
  rmsNormRow add mul scaleMean rsqrt (choose B b (ctxs b)) (x b) weight

/-- RMSNorm batch invariance under a request-local schedule policy. -/
theorem rmsNorm_batchInvariant_of_requestLocalSchedule
    {ι : Type u} {β : Type v} {B C : Nat}
    (add mul : β -> β -> β)
    (scaleMean rsqrt : β -> β)
    (choose : ScheduleSelectorWithCtx ι)
    (hchoose : RequestLocalScheduleInvariant choose)
    (x1 : Fin B -> ι -> β)
    (x2 : Fin C -> ι -> β)
    (ctxs1 : Fin B -> LocalKernelCtx)
    (ctxs2 : Fin C -> LocalKernelCtx)
    (weight : ι -> β)
    (i : Fin B) (j : Fin C)
    (hctx : ctxs1 i = ctxs2 j)
    (hrow : forall k, x1 i k = x2 j k) :
    batchedRMSNormWithCtxSelector add mul scaleMean rsqrt choose x1 ctxs1 weight i =
      batchedRMSNormWithCtxSelector add mul scaleMean rsqrt choose x2 ctxs2 weight j := by
  unfold batchedRMSNormWithCtxSelector
  rw [hchoose i j (ctxs1 i) (ctxs2 j) hctx]
  exact rmsNorm_batchInvariant add mul scaleMean rsqrt (choose C j (ctxs2 j))
    (x1 i) (x2 j) weight weight hrow (fun _ => rfl)

/-! ## Attention scheduling

Attention needs one more layer than RMSNorm or matmul. Its reductions still
follow trees supplied as input, but the KV tree must come from a request-local
layout policy rather than from the surrounding server batch.

These definitions model the contract for a fixed split-size attention
implementation:

* the selected request has a logical KV sequence,
* `ChunkPlan` represents chunking and prefill choices,
* a layout policy maps the local context and chunk plan to an ordered KV
  reduction tree,
* canonicality says that changing the chunk plan does not change that tree for
  the same request-local context.

These are not proofs about FlashAttention, paged attention, or a CUDA kernel.
They are the Lean-side objects those implementations would have to refine.
-/

namespace Attention

/-- How the server slices a request into chunks. No proof inspects the chunks.
A canonical layout is one whose KV reduction tree does not depend on this plan. -/
structure ChunkPlan where
  chunks : List Nat
deriving Repr, DecidableEq

/-- Request-local attention metadata, the attention counterpart of
`LocalKernelCtx`. -/
structure AttentionLocalCtx where
  qLen : Nat
  kvLen : Nat
  headDim : Nat
  dtypeTag : Nat
  splitSize : Nat
deriving Repr, DecidableEq

/-- A KV layout policy: the ordered KV reduction tree for a request, from its
local context and chunk plan. Its `canonical` field is the fixed-split-size
contract, under which equal contexts give equal trees for any two chunk plans. -/
structure KVLayoutPolicy (KV : Type u) where
  kvTree : AttentionLocalCtx -> ChunkPlan -> ReductionTree KV
  canonical :
    forall {ctx1 ctx2 : AttentionLocalCtx} (chunk1 chunk2 : ChunkPlan),
      ctx1 = ctx2 -> kvTree ctx1 chunk1 = kvTree ctx2 chunk2

/-- A KV block order computed from the request-local context alone: a first
block and the rest. With no chunk plan among its inputs, server chunking cannot
perturb the resulting tree. An implementation would compute the blocks from the
split size, the KV length, and its layout.
-/
structure FixedKVBlockPlan (KV : Type u) where
  first : AttentionLocalCtx -> KV
  rest : AttentionLocalCtx -> List KV

/-- Convert a nonempty ordered block list to a right-associated reduction tree. -/
def treeOfNonemptyList {KV : Type u} : KV -> List KV -> ReductionTree KV
  | head, [] => ReductionTree.leaf head
  | head, next :: rest => ReductionTree.node (ReductionTree.leaf head) (treeOfNonemptyList next rest)

/-- Right-nested KV tree over the blocks of a `FixedKVBlockPlan`. It accepts the
server's `ChunkPlan` and ignores it. -/
def fixedSplitKVTree {KV : Type u}
    (plan : FixedKVBlockPlan KV)
    (ctx : AttentionLocalCtx)
    (_chunk : ChunkPlan) : ReductionTree KV :=
  treeOfNonemptyList (plan.first ctx) (plan.rest ctx)

/-- Fixed-split KV trees are canonical by construction: equal local contexts
give equal trees, whatever the chunk plans. -/
theorem fixedSplitKVTree_canonical {KV : Type u}
    (plan : FixedKVBlockPlan KV)
    {ctx1 ctx2 : AttentionLocalCtx}
    (chunk1 chunk2 : ChunkPlan)
    (hctx : ctx1 = ctx2) :
    fixedSplitKVTree plan ctx1 chunk1 = fixedSplitKVTree plan ctx2 chunk2 := by
  cases hctx
  rfl

/-- Package a concrete fixed-layout KV block plan as a canonical layout policy. -/
def fixedSplitKVLayout {KV : Type u}
    (plan : FixedKVBlockPlan KV) : KVLayoutPolicy KV where
  kvTree := fixedSplitKVTree plan
  canonical := by
    intro ctx1 ctx2 chunk1 chunk2 hctx
    exact fixedSplitKVTree_canonical plan chunk1 chunk2 hctx

/-! ### Fixed split size

The construction above takes any request-local block order. Here is the concrete
one from the Thinking Machines post: cut the KV positions `0, ..., kvLen - 1` into
consecutive blocks of `splitSize` (the last one shorter when `splitSize` does not
divide `kvLen`), reduce each block left to right, then combine the block results
left to right. The tree reads only `ctx.splitSize` and `ctx.kvLen`. -/

/-- Block lengths for a fixed split size: full blocks, then the remainder. A
split size of zero gives one block. -/
def fixedSplitSizes (splitSize kvLen : Nat) : List Nat :=
  if splitSize = 0 then [kvLen] else
    List.replicate (kvLen / splitSize) splitSize ++
      (if kvLen % splitSize = 0 then [] else [kvLen % splitSize])

/-- Block lengths when the reduction is divided evenly among `splits` workers
(any remainder is dropped). -/
def evenSplitSizes (splits kvLen : Nat) : List Nat :=
  List.replicate splits (kvLen / splits)

/-- 1,000 keys with split size 256: three blocks of 256 and one of 232. -/
theorem fixedSplitSizes_256_1000 : fixedSplitSizes 256 1000 = [256, 256, 256, 232] := by
  decide

/-- Dividing 1,000 keys among four or eight workers gives different blocks. -/
theorem evenSplitSizes_1000 :
    evenSplitSizes 4 1000 = [250, 250, 250, 250] ∧
      evenSplitSizes 8 1000 = List.replicate 8 125 := by
  decide

/-- Consecutive position blocks with the given lengths, starting at `start`. -/
def positionBlocks : Nat -> List Nat -> List (List Nat)
  | _, [] => []
  | start, len :: rest => List.range' start len :: positionBlocks (start + len) rest

/-- Left-to-right tree over a nonempty list. -/
def leftChainTree {KV : Type u} (first : KV) (rest : List KV) : ReductionTree KV :=
  rest.foldl (fun acc kv => ReductionTree.node acc (ReductionTree.leaf kv)) (ReductionTree.leaf first)

/-- The fixed split-size tree: each block left to right, then the block results
left to right. Attention always has at least one key, so for `kvLen = 0` the
tree is just a placeholder leaf. -/
def fixedSplitSizeTree (splitSize kvLen : Nat) : ReductionTree Nat :=
  match (positionBlocks 0 (fixedSplitSizes splitSize kvLen)).filterMap
      (fun block => match block with
        | [] => none
        | kv :: rest => some (leftChainTree kv rest)) with
  | [] => ReductionTree.leaf 0
  | first :: rest => rest.foldl ReductionTree.node first

/-- With split size 4, ten keys reduce as `((0+1+2+3) + (4+5+6+7)) + (8+9)`, each
block left to right, and every position appears once, in order. -/
theorem fixedSplitSizeTree_4_10 :
    fixedSplitSizeTree 4 10 =
      ReductionTree.node
        (ReductionTree.node
          (leftChainTree 0 [1, 2, 3])
          (leftChainTree 4 [5, 6, 7]))
        (leftChainTree 8 [9]) ∧
    (fixedSplitSizeTree 4 10).leaves = List.range 10 := by
  decide

/-- Layout policy for a fixed split size: the KV tree depends on the request's
split size and KV length, never on the chunk plan. -/
def fixedSplitSizeLayout : KVLayoutPolicy Nat where
  kvTree ctx _chunk := fixedSplitSizeTree ctx.splitSize ctx.kvLen
  canonical := by
    intro ctx1 ctx2 chunk1 chunk2 hctx
    cases hctx
    rfl

/-- For a fixed request-local context, the chunk plan cannot change a layout's KV
tree. -/
theorem canonical_blocks_independent_of_chunking {KV : Type u}
    (layout : KVLayoutPolicy KV)
    (ctx : AttentionLocalCtx)
    (chunk1 chunk2 : ChunkPlan) :
    layout.kvTree ctx chunk1 = layout.kvTree ctx chunk2 :=
  layout.canonical chunk1 chunk2 rfl

/-- A feature-dimension reduction tree chosen from request-local attention
metadata. Its type already keeps the batch out, so `canonical` holds for every
`featureTree`. -/
structure FeatureSchedulePolicy (Feature : Type u) where
  featureTree : AttentionLocalCtx -> ReductionTree Feature
  canonical :
    forall {ctx1 ctx2 : AttentionLocalCtx},
      ctx1 = ctx2 -> featureTree ctx1 = featureTree ctx2

/-- Request-local tensors for one selected attention token and head: its query
vector, and the keys and values of the logical KV cache. -/
structure SelectedAttentionInput
    (Feature : Type u) (KV : Type v) (Out : Type w) (β : Type x) where
  query : Feature -> β
  key : KV -> Feature -> β
  value : KV -> Out -> β

/-- Attention scores for the selected token: one query-key dot product per key,
reduced over `featureTree`. -/
def scores {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featureTree : ReductionTree Feature)
    (inp : SelectedAttentionInput Feature KV Out β) : KV -> β :=
  fun kv => scheduledDot add mul featureTree inp.query (inp.key kv)

/-- Attention output at coordinate `out` for the selected token. `softmax` is any
function from the complete score vector to KV weights, and it receives the same
KV tree as the value sum, so its normalizer can follow that schedule too
(`treeSoftmax` below). -/
def scheduledAttentionOut
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featureTree : ReductionTree Feature)
    (kvTree : ReductionTree KV)
    (softmax : ReductionTree KV -> (KV -> β) -> KV -> β)
    (inp : SelectedAttentionInput Feature KV Out β)
    (out : Out) : β :=
  let sc := scores add mul featureTree inp
  let weight := softmax kvTree sc
  kvTree.eval add (fun kv => mul (weight kv) (inp.value kv out))

/-- A softmax whose normalizer is reduced over the supplied KV tree:
`exp (s k) / sum over the tree of exp (s k')`. The scalar `exp` and `div` stay
abstract, as `add` and `mul` do. -/
def treeSoftmax {KV : Type v} {β : Type x}
    (add : β -> β -> β) (exp : β -> β) (div : β -> β -> β) :
    ReductionTree KV -> (KV -> β) -> KV -> β :=
  fun kvTree sc kv => div (exp (sc kv)) (kvTree.eval add (fun k => exp (sc k)))

/-- Same query and keys imply the same selected-token score vector. -/
theorem scores_eq_of_same_inputs
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featureTree : ReductionTree Feature)
    (inp1 inp2 : SelectedAttentionInput Feature KV Out β)
    (hquery : forall f, inp1.query f = inp2.query f)
    (hkey : forall kv f, inp1.key kv f = inp2.key kv f) :
    scores add mul featureTree inp1 = scores add mul featureTree inp2 := by
  funext kv
  unfold scores scheduledDot
  rw [funext hquery, funext (hkey kv)]

/-- Equal queries, keys, and values give equal attention outputs under the same
feature and KV trees. -/
theorem scheduledAttentionOut_eq_of_same_inputs
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featureTree : ReductionTree Feature)
    (kvTree : ReductionTree KV)
    (softmax : ReductionTree KV -> (KV -> β) -> KV -> β)
    (inp1 inp2 : SelectedAttentionInput Feature KV Out β)
    (out : Out)
    (hquery : forall f, inp1.query f = inp2.query f)
    (hkey : forall kv f, inp1.key kv f = inp2.key kv f)
    (hvalue : forall kv out, inp1.value kv out = inp2.value kv out) :
    scheduledAttentionOut add mul featureTree kvTree softmax inp1 out =
      scheduledAttentionOut add mul featureTree kvTree softmax inp2 out := by
  unfold scheduledAttentionOut
  have hscores :
      scores add mul featureTree inp1 = scores add mul featureTree inp2 :=
    scores_eq_of_same_inputs add mul featureTree inp1 inp2 hquery hkey
  rw [hscores, funext fun kv => funext (hvalue kv)]

/-- Selected attention with a fixed feature tree and the KV tree from `layout`. -/
def selectedAttentionWithLayout
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x} {B : Nat}
    (add mul : β -> β -> β)
    (featureTree : ReductionTree Feature)
    (layout : KVLayoutPolicy KV)
    (softmax : ReductionTree KV -> (KV -> β) -> KV -> β)
    (inputs : Fin B -> SelectedAttentionInput Feature KV Out β)
    (ctxs : Fin B -> AttentionLocalCtx)
    (chunks : Fin B -> ChunkPlan)
    (b : Fin B)
    (out : Out) : β :=
  scheduledAttentionOut add mul featureTree
    (layout.kvTree (ctxs b) (chunks b)) softmax (inputs b) out

/-- Selected attention with both the feature tree and the KV tree chosen by
request-local policies. -/
def selectedAttentionWithPolicies
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x} {B : Nat}
    (add mul : β -> β -> β)
    (featurePolicy : FeatureSchedulePolicy Feature)
    (layout : KVLayoutPolicy KV)
    (softmax : ReductionTree KV -> (KV -> β) -> KV -> β)
    (inputs : Fin B -> SelectedAttentionInput Feature KV Out β)
    (ctxs : Fin B -> AttentionLocalCtx)
    (chunks : Fin B -> ChunkPlan)
    (b : Fin B)
    (out : Out) : β :=
  scheduledAttentionOut add mul
    (featurePolicy.featureTree (ctxs b))
    (layout.kvTree (ctxs b) (chunks b))
    softmax (inputs b) out

/-- Selected attention under a KV layout policy. With equal request-local
contexts and equal Q/K/V, different batches and chunk plans give the same
output.
-/
theorem selectedAttention_batchInvariant_of_canonicalLayout
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x} {B C : Nat}
    (add mul : β -> β -> β)
    (featureTree : ReductionTree Feature)
    (layout : KVLayoutPolicy KV)
    (softmax : ReductionTree KV -> (KV -> β) -> KV -> β)
    (inputs1 : Fin B -> SelectedAttentionInput Feature KV Out β)
    (inputs2 : Fin C -> SelectedAttentionInput Feature KV Out β)
    (ctxs1 : Fin B -> AttentionLocalCtx)
    (ctxs2 : Fin C -> AttentionLocalCtx)
    (chunks1 : Fin B -> ChunkPlan)
    (chunks2 : Fin C -> ChunkPlan)
    (i : Fin B) (j : Fin C) (out : Out)
    (hctx : ctxs1 i = ctxs2 j)
    (hquery : forall f, (inputs1 i).query f = (inputs2 j).query f)
    (hkey : forall kv f, (inputs1 i).key kv f = (inputs2 j).key kv f)
    (hvalue : forall kv out, (inputs1 i).value kv out = (inputs2 j).value kv out) :
    selectedAttentionWithLayout add mul featureTree layout softmax inputs1 ctxs1 chunks1 i out =
      selectedAttentionWithLayout add mul featureTree layout softmax inputs2 ctxs2 chunks2 j out := by
  unfold selectedAttentionWithLayout
  rw [layout.canonical (chunks1 i) (chunks2 j) hctx]
  exact scheduledAttentionOut_eq_of_same_inputs
    add mul featureTree (layout.kvTree (ctxs2 j) (chunks2 j)) softmax
    (inputs1 i) (inputs2 j) out hquery hkey hvalue

/-- Policy version of `selectedAttention_batchInvariant_of_canonicalLayout`, with
the feature tree also chosen from the request-local context. -/
theorem selectedAttention_batchInvariant_of_schedulePolicies
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x} {B C : Nat}
    (add mul : β -> β -> β)
    (featurePolicy : FeatureSchedulePolicy Feature)
    (layout : KVLayoutPolicy KV)
    (softmax : ReductionTree KV -> (KV -> β) -> KV -> β)
    (inputs1 : Fin B -> SelectedAttentionInput Feature KV Out β)
    (inputs2 : Fin C -> SelectedAttentionInput Feature KV Out β)
    (ctxs1 : Fin B -> AttentionLocalCtx)
    (ctxs2 : Fin C -> AttentionLocalCtx)
    (chunks1 : Fin B -> ChunkPlan)
    (chunks2 : Fin C -> ChunkPlan)
    (i : Fin B) (j : Fin C) (out : Out)
    (hctx : ctxs1 i = ctxs2 j)
    (hquery : forall f, (inputs1 i).query f = (inputs2 j).query f)
    (hkey : forall kv f, (inputs1 i).key kv f = (inputs2 j).key kv f)
    (hvalue : forall kv out, (inputs1 i).value kv out = (inputs2 j).value kv out) :
    selectedAttentionWithPolicies add mul featurePolicy layout softmax
      inputs1 ctxs1 chunks1 i out =
    selectedAttentionWithPolicies add mul featurePolicy layout softmax
      inputs2 ctxs2 chunks2 j out := by
  unfold selectedAttentionWithPolicies
  rw [featurePolicy.canonical hctx]
  rw [layout.canonical (chunks1 i) (chunks2 j) hctx]
  exact scheduledAttentionOut_eq_of_same_inputs
    add mul (featurePolicy.featureTree (ctxs2 j))
    (layout.kvTree (ctxs2 j) (chunks2 j)) softmax
    (inputs1 i) (inputs2 j) out hquery hkey hvalue

end Attention

/-! ## CUDA/FlashAttention refinement boundary

TorchLean already has a CUDA fused attention path and a FlashAttention
denotation in `NN.Spec.Layers.FlashAttention`. Lean does not inspect the native
code, so the boundary is a certificate. It must show that the runtime path
refines the scheduled attention semantics, and batch invariance then follows
from the theorem above.
-/

namespace CUDARuntime

/-- One attention request as a fused CUDA/Triton kernel sees it: the logical
Q/K/V payload, the request-local metadata, and the server's chunk plan. -/
structure SelectedAttentionRequest
    (Feature : Type u) (KV : Type v) (Out : Type w) (β : Type x) where
  input : Attention.SelectedAttentionInput Feature KV Out β
  ctx : Attention.AttentionLocalCtx
  chunk : Attention.ChunkPlan

/-- `Attention.selectedAttentionWithPolicies` as a batched forward pass. -/
def scheduledAttentionForward
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featurePolicy : Attention.FeatureSchedulePolicy Feature)
    (layout : Attention.KVLayoutPolicy KV)
    (softmax : ReductionTree KV -> (KV -> β) -> KV -> β) :
    BatchedForward
      (SelectedAttentionRequest Feature KV Out β)
      (Out -> β) :=
  fun B reqs b out =>
    Attention.selectedAttentionWithPolicies add mul featurePolicy layout softmax
      (fun i : Fin B => (reqs i).input)
      (fun i : Fin B => (reqs i).ctx)
      (fun i : Fin B => (reqs i).chunk)
      b out

/-- `scheduledAttentionForward` is batch-invariant for every feature policy and
KV layout, since both choose trees from request-local metadata. -/
theorem scheduledAttentionForward_batchInvariant
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featurePolicy : Attention.FeatureSchedulePolicy Feature)
    (layout : Attention.KVLayoutPolicy KV)
    (softmax : ReductionTree KV -> (KV -> β) -> KV -> β) :
    BatchInvariantForward
      (scheduledAttentionForward
        (Feature := Feature) (KV := KV) (Out := Out) (β := β)
        add mul featurePolicy layout softmax) := by
  intro B C reqs1 reqs2 i j hreq
  funext out
  have hinput : (reqs1 i).input = (reqs2 j).input :=
    congrArg SelectedAttentionRequest.input hreq
  have hctx : (reqs1 i).ctx = (reqs2 j).ctx :=
    congrArg SelectedAttentionRequest.ctx hreq
  exact Attention.selectedAttention_batchInvariant_of_schedulePolicies
    add mul featurePolicy layout softmax
    (fun i : Fin B => (reqs1 i).input)
    (fun j : Fin C => (reqs2 j).input)
    (fun i : Fin B => (reqs1 i).ctx)
    (fun j : Fin C => (reqs2 j).ctx)
    (fun i : Fin B => (reqs1 i).chunk)
    (fun j : Fin C => (reqs2 j).chunk)
    i j out hctx
    (by
      intro f
      exact congrFun (congrArg Attention.SelectedAttentionInput.query hinput) f)
    (by
      intro kv f
      exact congrFun (congrFun (congrArg Attention.SelectedAttentionInput.key hinput) kv) f)
    (by
      intro kv o
      exact congrFun (congrFun (congrArg Attention.SelectedAttentionInput.value hinput) kv) o)

/-- Certificate for a native fused attention forward pass. `runtimeForward` can
stand for TorchLean's CUDA FFI path, a Triton kernel, or a hosted verifier
replay, and the obligation is refinement of `scheduledAttentionForward`. -/
structure FusedAttentionRuntimeCert
    (Feature : Type u) (KV : Type v) (Out : Type w) (β : Type x) where
  add : β -> β -> β
  mul : β -> β -> β
  featurePolicy : Attention.FeatureSchedulePolicy Feature
  layout : Attention.KVLayoutPolicy KV
  softmax : ReductionTree KV -> (KV -> β) -> KV -> β
  runtimeForward :
    BatchedForward
      (SelectedAttentionRequest Feature KV Out β)
      (Out -> β)
  refinesScheduleSpec :
    RefinesForward runtimeForward
      (scheduledAttentionForward add mul featurePolicy layout softmax)

/-- A certified fused attention runtime is batch-invariant, the theorem a real
kernel certificate should target. -/
theorem fusedAttentionRuntime_batchInvariant
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (cert : FusedAttentionRuntimeCert Feature KV Out β) :
    BatchInvariantForward cert.runtimeForward :=
  batchInvariant_of_refinesForward
    cert.runtimeForward
    (scheduledAttentionForward cert.add cert.mul cert.featurePolicy cert.layout cert.softmax)
    cert.refinesScheduleSpec
    (scheduledAttentionForward_batchInvariant
      cert.add cert.mul cert.featurePolicy cert.layout cert.softmax)

end CUDARuntime

namespace TorchLeanFlashAttention

open Spec

/-- Runtime refinement certificate for TorchLean's FlashAttention denotation.

Its `runtime_refines_spec` field is a Lean proof that `runtimeOut` equals
`Spec.flashAttention cfg ctx`. Numerical tests of the native path are evidence
about an implementation, but they cannot construct this equality, which needs a
verified analyzer or a kernel semantics. -/
structure NativeForwardCert
    (α : Type) [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α -> α -> Prop)]
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0} where
  cfg : Spec.FlashAttentionConfig
  ctx : Spec.AttentionContext α nQ nK dModel h1 h2
  runtimeOut : TorchLean.Tensor α (Spec.Shape.dim nQ (Spec.Shape.dim dModel Spec.Shape.scalar))
  runtime_refines_spec :
    runtimeOut = Spec.flashAttention cfg ctx

/-- A certified native output also equals the standard scaled-dot-product
attention spec. -/
theorem NativeForwardCert.refines_scaledDotProduct
    {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α -> α -> Prop)]
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (cert : NativeForwardCert α (nQ := nQ) (nK := nK) (dModel := dModel)
      (h1 := h1) (h2 := h2)) :
    cert.runtimeOut = Spec.scaledDotProductAttention cert.ctx := by
  calc
    cert.runtimeOut = Spec.flashAttention cert.cfg cert.ctx :=
      cert.runtime_refines_spec
    _ = Spec.scaledDotProductAttention cert.ctx :=
      Spec.flashAttention_eq_scaledDotProductAttention cert.cfg cert.ctx

/-- `runtime_refines_spec` as a theorem: the certified output equals
`Spec.flashAttention cfg ctx`. -/
theorem NativeForwardCert.refines_flashAttention
    {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α -> α -> Prop)]
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (cert : NativeForwardCert α (nQ := nQ) (nK := nK) (dModel := dModel)
      (h1 := h1) (h2 := h2)) :
    cert.runtimeOut = Spec.flashAttention cert.cfg cert.ctx :=
  cert.runtime_refines_spec

end TorchLeanFlashAttention

/-- Apply `f` to the selected row and ignore the rest of the batch. A pointwise
layer like this is batch-invariant (`pointwise_batchInvariant`). -/
def pointwiseForward {X : Type u} {Y : Type v} (f : X -> Y) :
    BatchedForward X Y :=
  fun _ xs i => f (xs i)

theorem pointwise_batchInvariant {X : Type u} {Y : Type v} (f : X -> Y) :
    BatchInvariantForward (pointwiseForward f) := by
  intro B C xs ys i j h
  simp [pointwiseForward, h]

/-- Run `f` on every row of the batch, then `g` on the results. -/
def composeForward {X : Type u} {Y : Type v} {Z : Type w}
    (f : BatchedForward X Y) (g : BatchedForward Y Z) :
    BatchedForward X Z :=
  fun B xs i => g B (fun k => f B xs k) i

/-- Batch invariance is preserved by `composeForward`, so it extends layer by
layer through a pipeline. -/
theorem compose_batchInvariant {X : Type u} {Y : Type v} {Z : Type w}
    (f : BatchedForward X Y) (g : BatchedForward Y Z)
    (hf : BatchInvariantForward f)
    (hg : BatchInvariantForward g) :
    BatchInvariantForward (composeForward f g) := by
  intro B C xs ys i j h
  apply hg
  exact hf xs ys i j h

/-! ## Logit margins and argmax stability

The equality theorems above give bitwise reproducibility. In deployed inference
a margin argument is often more useful: logits may move slightly when schedules
change, but the greedy token stays put if the top logit leads by enough.
-/

namespace Margin

/-- Absolute value on `Rat`. Rationals keep the margin theorems executable and
elementary, and they read the same as the usual real-valued statements. -/
def absQ (x : Rat) : Rat :=
  if x < 0 then -x else x

theorem absQ_nonneg (x : Rat) : 0 <= absQ x := by
  unfold absQ
  split
  · linarith
  · linarith

theorem absQ_le_iff {x e : Rat} (he : 0 <= e) :
    absQ x <= e <-> -e <= x ∧ x <= e := by
  unfold absQ
  constructor
  · intro h
    by_cases hx : x < 0
    · simp [hx] at h
      constructor
      · linarith
      · linarith
    · simp [hx] at h
      constructor
      · linarith
      · exact h
  · intro h
    by_cases hx : x < 0
    · simp [hx]
      linarith
    · simp [hx]
      exact h.2

/-- Two values within `eps` of a shared reference are within `2 * eps` of each
other. -/
theorem absQ_sub_le_two_mul_of_shared_reference
    {ref x y eps : Rat}
    (heps : 0 <= eps)
    (hx : absQ (x - ref) <= eps)
    (hy : absQ (y - ref) <= eps) :
    absQ (y - x) <= 2 * eps := by
  have hxBounds := (absQ_le_iff heps).mp hx
  have hyBounds := (absQ_le_iff heps).mp hy
  have htwo : 0 <= 2 * eps := by linarith
  exact (absQ_le_iff htwo).mpr (by
    constructor <;> linarith)

/-- A token `best` has margin `m` if every other logit is at least `m` below it. -/
def HasMargin {Vocab : Type u} [DecidableEq Vocab]
    (z : Vocab -> Rat) (best : Vocab) (m : Rat) : Prop :=
  forall tok, tok ≠ best -> z tok + m <= z best

/-- `z'` is within `eps` of `z` in every coordinate. -/
def LinfClose {Vocab : Type u} (z z' : Vocab -> Rat) (eps : Rat) : Prop :=
  forall tok, absQ (z' tok - z tok) <= eps

/-- Observed logits within `eps` of the reference logits (`LinfClose`, named for
drift bounds). Reduction enclosures such as TorchLean's
`dotTreeResult_enclosure` are where such bounds would come from. -/
abbrev LogitDriftBound {Vocab : Type u}
    (reference observed : Vocab -> Rat) (eps : Rat) : Prop :=
  LinfClose reference observed eps

/-- Coordinatewise `absQ_sub_le_two_mul_of_shared_reference`: logit vectors
within `eps` of one reference are within `2 * eps` of each other. -/
theorem linfClose_of_shared_reference {Vocab : Type u}
    (reference z1 z2 : Vocab -> Rat) (eps : Rat)
    (heps : 0 <= eps)
    (h1 : LogitDriftBound reference z1 eps)
    (h2 : LogitDriftBound reference z2 eps) :
    LinfClose z1 z2 (2 * eps) := by
  intro tok
  exact absQ_sub_le_two_mul_of_shared_reference
    (ref := reference tok) (x := z1 tok) (y := z2 tok)
    heps (h1 tok) (h2 tok)

/-- An implementation-level contract for greedy choice: whenever a token
strictly beats every other token, `choose` returns that token. -/
def StrictArgmaxSound {Vocab : Type u} [DecidableEq Vocab]
    (choose : (Vocab -> Rat) -> Vocab) : Prop :=
  forall (z : Vocab -> Rat) (best : Vocab),
    (forall tok, tok ≠ best -> z tok < z best) ->
    choose z = best

/-- If `best` leads by a margin above `2 * eps` and every logit moves by at most
`eps`, then `best` still strictly beats `tok`. -/
theorem winner_stable_of_margin {Vocab : Type u} [DecidableEq Vocab]
    (z z' : Vocab -> Rat) (best tok : Vocab) (eps margin : Rat)
    (heps : 0 <= eps)
    (hclose : LinfClose z z' eps)
    (hmargin : HasMargin z best margin)
    (hgap : 2 * eps < margin)
    (htok : tok ≠ best) :
    z' tok < z' best := by
  have hTokAbs := hclose tok
  have hBestAbs := hclose best
  have hTokBounds := (absQ_le_iff heps).mp hTokAbs
  have hBestBounds := (absQ_le_iff heps).mp hBestAbs
  have hTokUpper : z' tok <= z tok + eps := by linarith
  have hBestLower : z best - eps <= z' best := by linarith
  have hMargin := hmargin tok htok
  linarith

/-- A certified greedy choice: `choose z = best`, and `best` leads every other
token by `margin > 0`. Since the choice function is a parameter, any
deterministic tie-breaking rule fits. -/
structure GreedyMarginCert {Vocab : Type u} [DecidableEq Vocab]
    (choose : (Vocab -> Rat) -> Vocab) (z : Vocab -> Rat) where
  best : Vocab
  margin : Rat
  chosen : choose z = best
  positive : 0 < margin
  cert : HasMargin z best margin

/-- If a new logit vector is close enough, the old greedy winner still strictly
beats every other token under the new logits. -/
theorem certified_winner_survives_drift {Vocab : Type u} [DecidableEq Vocab]
    (choose : (Vocab -> Rat) -> Vocab)
    (z z' : Vocab -> Rat) (eps : Rat)
    (cert : GreedyMarginCert choose z)
    (heps : 0 <= eps)
    (hclose : LinfClose z z' eps)
    (hgap : 2 * eps < cert.margin) :
    forall tok, tok ≠ cert.best -> z' tok < z' cert.best := by
  intro tok htok
  exact winner_stable_of_margin z z' cert.best tok eps cert.margin
    heps hclose cert.cert hgap htok

/-- With a strict-argmax `choose`, a certified margin above `2 * eps` keeps the
greedy token when every logit moves by at most `eps`. -/
theorem choose_stable_of_margin {Vocab : Type u} [DecidableEq Vocab]
    (choose : (Vocab -> Rat) -> Vocab)
    (z z' : Vocab -> Rat) (eps : Rat)
    (cert : GreedyMarginCert choose z)
    (hsound : StrictArgmaxSound choose)
    (heps : 0 <= eps)
    (hclose : LinfClose z z' eps)
    (hgap : 2 * eps < cert.margin) :
    choose z' = choose z := by
  have hbestNew : choose z' = cert.best := by
    apply hsound
    exact certified_winner_survives_drift choose z z' eps cert heps hclose hgap
  calc
    choose z' = cert.best := hbestNew
    _ = choose z := cert.chosen.symm

/-- Two logit vectors, each within `eps` of a reference whose certified margin
exceeds `2 * eps`, choose the same token. Use it when bitwise equality fails
but the numerical drift is bounded. -/
theorem choose_stable_of_shared_reference_error_bounds
    {Vocab : Type u} [DecidableEq Vocab]
    (choose : (Vocab -> Rat) -> Vocab)
    (reference z1 z2 : Vocab -> Rat) (eps : Rat)
    (cert : GreedyMarginCert choose reference)
    (hsound : StrictArgmaxSound choose)
    (heps : 0 <= eps)
    (h1 : LogitDriftBound reference z1 eps)
    (h2 : LogitDriftBound reference z2 eps)
    (hgap : 2 * eps < cert.margin) :
    choose z1 = choose z2 := by
  have hchoose1 : choose z1 = choose reference :=
    choose_stable_of_margin choose reference z1 eps cert hsound heps h1 hgap
  have hchoose2 : choose z2 = choose reference :=
    choose_stable_of_margin choose reference z2 eps cert hsound heps h2 hgap
  calc
    choose z1 = choose reference := hchoose1
    _ = choose z2 := hchoose2.symm

/-- Token stability when the margin certificate belongs to `z1` itself rather
than to a reference, and `z2` is within `2 * eps` of `z1`. Then the margin must
exceed `4 * eps`. -/
theorem choose_stable_under_schedule_drift
    {Vocab : Type u} [DecidableEq Vocab]
    (choose : (Vocab -> Rat) -> Vocab)
    (z1 z2 : Vocab -> Rat) (eps : Rat)
    (cert : GreedyMarginCert choose z1)
    (hsound : StrictArgmaxSound choose)
    (heps : 0 <= eps)
    (hdrift : LinfClose z1 z2 (2 * eps))
    (hgap : 2 * (2 * eps) < cert.margin) :
    choose z2 = choose z1 := by
  have htwo : 0 <= 2 * eps := by linarith
  exact choose_stable_of_margin choose z1 z2 (2 * eps)
    cert hsound htwo hdrift hgap

end Margin

/-! ## Reference-equivalent decode/verify/rollback serving

The previous sections prove local kernel and token-stability facts. This
serving theorem is more systems-shaped. A fast path may speculate with arbitrary
dynamic batching and schedule-dependent numerics, but user-visible tokens are
appended only through a verifier-sound reference window, so the committed output
is always a prefix of the reference decoder's output.
-/

namespace DVR

/-- A reference decoder: one deterministic step from a state to a token and the
next state. For a Transformer, `State` would hold the prompt, KV cache, position,
and parameters, and `step` would evaluate the TorchLean IR under fixed reference
schedules (`IRDecoderConfig`). -/
structure ReferenceDecoder (State Token : Type u) where
  step : State -> Token × State

/-- Run the reference decoder for `n` tokens. -/
def refRun {State Token : Type u}
    (ref : ReferenceDecoder State Token) :
    Nat -> State -> List Token × State
  | 0, st => ([], st)
  | n + 1, st =>
      let (tok, st') := ref.step st
      let (rest, final) := refRun ref n st'
      (tok :: rest, final)

@[simp]
theorem refRun_zero {State Token : Type u}
    (ref : ReferenceDecoder State Token) (st : State) :
    refRun ref 0 st = ([], st) := rfl

theorem refRun_succ {State Token : Type u}
    (ref : ReferenceDecoder State Token) (n : Nat) (st : State) :
    refRun ref (n + 1) st =
      let (tok, st') := ref.step st
      let (rest, final) := refRun ref n st'
      (tok :: rest, final) := rfl

theorem refRun_append {State Token : Type u}
    (ref : ReferenceDecoder State Token) (m n : Nat) (st : State) :
    refRun ref (m + n) st =
      let first := refRun ref m st
      let second := refRun ref n first.2
      (first.1 ++ second.1, second.2) := by
  induction m generalizing st with
  | zero =>
      simp [refRun]
  | succ m ih =>
      cases hstep : ref.step st with
      | mk tok st' =>
          simp [refRun, hstep, Nat.succ_add, ih]

theorem refRun_tokens_length {State Token : Type u}
    (ref : ReferenceDecoder State Token) :
    forall (n : Nat) (st : State), (refRun ref n st).1.length = n
  | 0, _ => rfl
  | n + 1, st => by
      cases hstep : ref.step st with
      | mk tok st' =>
          simp [refRun, hstep, refRun_tokens_length ref n st']

/-- Server state: the committed user-visible tokens, the reference state after
them, and the fast path's private state and candidate tokens. -/
structure ServerState (State Token Fast : Type u) where
  committed : List Token
  refState : State
  fastState : Fast
  candidates : List Token

def observe {State Token Fast : Type u} (s : ServerState State Token Fast) : List Token :=
  s.committed

/-- An accepted window is verifier-sound when, for some `n`, it is the next `n`
reference tokens from `st` and `st'` is the reference state after them. -/
def VerifyWindowSound {State Token : Type u}
    (ref : ReferenceDecoder State Token)
    (st : State) (accepted : List Token) (st' : State) : Prop :=
  exists n : Nat, accepted = (refRun ref n st).1 ∧ st' = (refRun ref n st).2

/-- Serving invariant: the committed output and stored reference state agree
with running the reference decoder from `initialRef` for some number of tokens. -/
def DVRInvariant {State Token Fast : Type u}
    (ref : ReferenceDecoder State Token)
    (initialRef : State)
    (server : ServerState State Token Fast) : Prop :=
  exists n : Nat,
    server.committed = (refRun ref n initialRef).1 ∧
    server.refState = (refRun ref n initialRef).2

theorem invariant_of_initial {State Token Fast : Type u}
    (ref : ReferenceDecoder State Token)
    (initialRef : State) (fast0 : Fast) :
    DVRInvariant ref initialRef
      ({ committed := ([] : List Token),
         refState := initialRef,
         fastState := fast0,
         candidates := ([] : List Token) } : ServerState State Token Fast) := by
  exact ⟨0, rfl, rfl⟩

/-- Legal serving steps. Speculation and rollback change only the fast-path
state and candidates. `verifyAccept`, the one rule that commits tokens, needs a
verifier-sound reference window. -/
inductive DVRStep {State Token Fast : Type u}
    (ref : ReferenceDecoder State Token) :
    ServerState State Token Fast -> ServerState State Token Fast -> Prop where
  | speculate (s : ServerState State Token Fast)
      (fast' : Fast) (candidates' : List Token) :
      DVRStep ref s
        { s with fastState := fast', candidates := candidates' }
  | rollback (s : ServerState State Token Fast)
      (fast' : Fast) :
      DVRStep ref s
        { s with fastState := fast', candidates := [] }
  | verifyAccept (s : ServerState State Token Fast)
      (accepted rest : List Token) (ref' : State) (fast' : Fast)
      (hverify : VerifyWindowSound ref s.refState accepted ref') :
      DVRStep ref s
        { committed := s.committed ++ accepted
          refState := ref',
          fastState := fast',
          candidates := rest }

/-- Every legal DVR step preserves `DVRInvariant`. -/
theorem dvr_step_preserves_invariant {State Token Fast : Type u}
    (ref : ReferenceDecoder State Token)
    (initialRef : State)
    {s s' : ServerState State Token Fast}
    (hinv : DVRInvariant ref initialRef s)
    (hstep : DVRStep ref s s') :
    DVRInvariant ref initialRef s' := by
  rcases hinv with ⟨n, hcommitted, hrefState⟩
  cases hstep with
  | speculate fast' candidates' =>
      exact ⟨n, hcommitted, hrefState⟩
  | rollback fast' =>
      exact ⟨n, hcommitted, hrefState⟩
  | verifyAccept accepted rest ref' fast' hverify =>
      rcases hverify with ⟨k, haccepted, href'⟩
      refine ⟨n + k, ?_, ?_⟩
      · calc
          s.committed ++ accepted =
              (refRun ref n initialRef).1 ++ (refRun ref k s.refState).1 := by
                rw [hcommitted, haccepted]
          _ = (refRun ref n initialRef).1 ++
              (refRun ref k (refRun ref n initialRef).2).1 := by
                rw [hrefState]
          _ = (refRun ref (n + k) initialRef).1 := by
                exact (congrArg Prod.fst (refRun_append ref n k initialRef)).symm
      · calc
          ref' = (refRun ref k s.refState).2 := href'
          _ = (refRun ref k (refRun ref n initialRef).2).2 := by
                rw [hrefState]
          _ = (refRun ref (n + k) initialRef).2 := by
                exact (congrArg Prod.snd (refRun_append ref n k initialRef)).symm

/-- A legal trace is a sequence of legal DVR serving steps. -/
inductive DVRTrace {State Token Fast : Type u}
    (ref : ReferenceDecoder State Token) :
    ServerState State Token Fast -> ServerState State Token Fast -> Prop where
  | refl (s : ServerState State Token Fast) : DVRTrace ref s s
  | cons {s t u : ServerState State Token Fast}
      (hstep : DVRStep ref s t)
      (hrest : DVRTrace ref t u) :
      DVRTrace ref s u

/-- Legal traces preserve the reference-prefix invariant. -/
theorem dvr_trace_preserves_invariant {State Token Fast : Type u}
    (ref : ReferenceDecoder State Token)
    (initialRef : State)
    {s t : ServerState State Token Fast}
    (hinv : DVRInvariant ref initialRef s)
    (htrace : DVRTrace ref s t) :
    DVRInvariant ref initialRef t := by
  induction htrace with
  | refl s =>
      exact hinv
  | cons hstep hrest ih =>
      exact ih (dvr_step_preserves_invariant ref initialRef hinv hstep)

/-- Reference-equivalent dynamic serving: whatever the speculation, dynamic
batching, fast-path schedules, and rollbacks, a legal DVR trace from an
invariant state shows the user a prefix of the reference decoder's output. -/
theorem ServeDVR_refines_RefDecode {State Token Fast : Type u}
    (ref : ReferenceDecoder State Token)
    (initialRef : State)
    {s0 sN : ServerState State Token Fast}
    (hinv0 : DVRInvariant ref initialRef s0)
    (htrace : DVRTrace ref s0 sN) :
    exists n : Nat, observe sN = (refRun ref n initialRef).1 := by
  rcases dvr_trace_preserves_invariant ref initialRef hinv0 htrace with
    ⟨n, hcommitted, _hrefState⟩
  exact ⟨n, hcommitted⟩

/-- Observable determinism: two legal traces from invariant states (same decoder,
same `initialRef`) show the user the same tokens whenever they commit the same
number of them. Schedule choices cannot change the committed prefix. -/
theorem user_observable_determinism_of_same_commit_length
    {State Token Fast : Type u}
    (ref : ReferenceDecoder State Token)
    (initialRef : State)
    {s0a sNa s0b sNb : ServerState State Token Fast}
    (hinvA : DVRInvariant ref initialRef s0a)
    (hinvB : DVRInvariant ref initialRef s0b)
    (htraceA : DVRTrace ref s0a sNa)
    (htraceB : DVRTrace ref s0b sNb)
    (hlen : (observe sNa).length = (observe sNb).length) :
    observe sNa = observe sNb := by
  rcases dvr_trace_preserves_invariant ref initialRef hinvA htraceA with
    ⟨nA, hcommittedA, _hrefA⟩
  rcases dvr_trace_preserves_invariant ref initialRef hinvB htraceB with
    ⟨nB, hcommittedB, _hrefB⟩
  have hnA : nA = (observe sNa).length := by
    rw [observe, hcommittedA, refRun_tokens_length]
  have hnB : nB = (observe sNb).length := by
    rw [observe, hcommittedB, refRun_tokens_length]
  have hn : nA = nB := by
    rw [hnA, hnB, hlen]
  calc
    observe sNa = (refRun ref nA initialRef).1 := hcommittedA
    _ = (refRun ref nB initialRef).1 := by rw [hn]
    _ = observe sNb := hcommittedB.symm

/-- A reference decoder backed by a TorchLean IR graph. `referenceStep_eq_denote`
fixes each step: turn the request state into a shape-erased tensor, evaluate the
shared `NN.IR.Graph` with `payload`, choose a token from the result, and update
the request-local state. This connects the abstract DVR theorem to TorchLean's
op-tagged graph semantics. -/
structure IRDecoderConfig
    (α : Type) [TorchLean.Storage α] [Context α]
    (State Token : Type) where
  graph : NN.IR.Graph
  payload : NN.IR.Payload α
  outputId : Nat
  inputOf : State -> Spec.SomeTensor α
  choose : Except String (Spec.SomeTensor α) -> Token
  update : State -> Token -> State
  referenceStep : State -> Token × State
  referenceStep_eq_denote :
    forall st : State,
      referenceStep st =
        let out : Except String (Spec.SomeTensor α) := NN.IR.Graph.denote
          (α := α) (g := graph) (payload := payload)
          (input := inputOf st) (outputId := outputId)
        let tok := choose out
        (tok, update st tok)

def toReferenceDecoder
    {α : Type} [TorchLean.Storage α] [Context α] [DecidableEq _root_.Spec.Shape]
    {State Token : Type}
    (cfg : IRDecoderConfig α State Token) :
    ReferenceDecoder State Token where
  step := cfg.referenceStep

/-- `VerifyWindowSound` for the reference decoder built on `NN.IR.Graph.denote`.
Executable window checkers should target this predicate. -/
def IRVerifyWindowSound
    {α : Type} [TorchLean.Storage α] [Context α] [DecidableEq _root_.Spec.Shape]
    {State Token : Type}
    (cfg : IRDecoderConfig α State Token)
    (st : State) (accepted : List Token) (st' : State) : Prop :=
  VerifyWindowSound
    (toReferenceDecoder (cfg := cfg))
    st accepted st'

/-- `ServeDVR_refines_RefDecode` for an IR-backed decoder. For any legal DVR trace
whose commits are sound with respect to `NN.IR.Graph.denote`, the user-visible
tokens are a prefix of the IR decoder's output. -/
theorem ServeDVR_refines_TorchLeanIRDecode
    {α : Type} [TorchLean.Storage α] [Context α] [DecidableEq _root_.Spec.Shape]
    {State Token Fast : Type}
    (cfg : IRDecoderConfig α State Token)
    (initialRef : State)
    {s0 sN : ServerState State Token Fast}
    (hinv0 : DVRInvariant
      (toReferenceDecoder (cfg := cfg))
      initialRef s0)
    (htrace : DVRTrace
      (toReferenceDecoder (cfg := cfg))
      s0 sN) :
    exists n : Nat,
      observe sN =
        (refRun
          (toReferenceDecoder (cfg := cfg))
          n initialRef).1 :=
  ServeDVR_refines_RefDecode
    (toReferenceDecoder (cfg := cfg))
    initialRef hinv0 htrace

/-- `user_observable_determinism_of_same_commit_length` for an IR-backed
decoder. -/
theorem TorchLeanIR_user_observable_determinism_of_same_commit_length
    {α : Type} [TorchLean.Storage α] [Context α] [DecidableEq _root_.Spec.Shape]
    {State Token Fast : Type}
    (cfg : IRDecoderConfig α State Token)
    (initialRef : State)
    {s0a sNa s0b sNb : ServerState State Token Fast}
    (hinvA : DVRInvariant
      (toReferenceDecoder (cfg := cfg))
      initialRef s0a)
    (hinvB : DVRInvariant
      (toReferenceDecoder (cfg := cfg))
      initialRef s0b)
    (htraceA : DVRTrace
      (toReferenceDecoder (cfg := cfg))
      s0a sNa)
    (htraceB : DVRTrace
      (toReferenceDecoder (cfg := cfg))
      s0b sNb)
    (hlen : (observe sNa).length = (observe sNb).length) :
    observe sNa = observe sNb :=
  user_observable_determinism_of_same_commit_length
    (toReferenceDecoder (cfg := cfg))
    initialRef hinvA hinvB htraceA htraceB hlen

/-- Dimensions of the causal Transformer a package serves. No theorem reads
them, and they document the model family. -/
structure TransformerDims where
  seqLen : Nat
  vocabSize : Nat
  dModel : Nat
  numHeads : Nat
  headDim : Nat
  ffnHidden : Nat
deriving Repr, DecidableEq

/-- Request-local state for decoder-only serving: the prompt, the generated
tokens, the KV cache, and the next position. `KV` stays abstract because
deployments store their caches differently. -/
structure CausalRequestState (Token KV : Type u) where
  prompt : List Token
  generated : List Token
  kvCache : KV
  position : Nat
deriving Repr

/-- Everything needed to serve a TorchLean causal Transformer decoder: its
dimensions, the IR decoder (graph, payload, request encoder, token chooser, and
request-local update), the initial reference and fast-path states, and the graph
checker's well-formedness and well-shapedness evidence.

Because the request state and token type are parameters, an instance can choose
byte or BPE tokens, its KV-cache state, logits postprocessing, or a
margin-certified argmax policy. What stays fixed is the reference step, which
must be `NN.IR.Graph.denote` followed by token choice and the request-local
update. -/
structure CausalTransformerIRPackage
    (α : Type) [TorchLean.Storage α] [Context α] [DecidableEq _root_.Spec.Shape]
    (State Token Fast : Type) where
  dims : TransformerDims
  cfg : IRDecoderConfig α State Token
  initialRef : State
  initialFast : Fast
  graph_wellFormed : cfg.graph.WellFormed
  graph_wellShaped : cfg.graph.WellShaped

/-- Common specialization: a package whose request state is
`CausalRequestState`. -/
abbrev ConcreteCausalTransformerPackage
    (α : Type) [TorchLean.Storage α] [Context α] [DecidableEq _root_.Spec.Shape]
    (Token KV Fast : Type) :=
  CausalTransformerIRPackage α (CausalRequestState Token KV) Token Fast

namespace CausalTransformerIRPackage

def refDecoder
    {α : Type} [TorchLean.Storage α] [Context α] [DecidableEq _root_.Spec.Shape]
    {State Token Fast : Type}
    (pkg : CausalTransformerIRPackage α State Token Fast) :
    ReferenceDecoder State Token :=
  toReferenceDecoder (cfg := pkg.cfg)

def initialServer
    {α : Type} [TorchLean.Storage α] [Context α] [DecidableEq _root_.Spec.Shape]
    {State Token Fast : Type}
    (pkg : CausalTransformerIRPackage α State Token Fast) :
    ServerState State Token Fast :=
  { committed := ([] : List Token),
    refState := pkg.initialRef,
    fastState := pkg.initialFast,
    candidates := ([] : List Token) }

theorem initial_invariant
    {α : Type} [TorchLean.Storage α] [Context α] [DecidableEq _root_.Spec.Shape]
    {State Token Fast : Type}
    (pkg : CausalTransformerIRPackage α State Token Fast) :
    DVRInvariant (pkg.refDecoder) pkg.initialRef pkg.initialServer :=
  invariant_of_initial (pkg.refDecoder) pkg.initialRef pkg.initialFast

/-- Package-level reference equivalence: every legal decode/verify/rollback trace
from `pkg.initialServer` releases a prefix of the IR decoder's output. A concrete
Transformer only has to fill the package fields and check each committed window
against `pkg.refDecoder`. -/
theorem serve_refines_reference
    {α : Type} [TorchLean.Storage α] [Context α] [DecidableEq _root_.Spec.Shape]
    {State Token Fast : Type}
    (pkg : CausalTransformerIRPackage α State Token Fast)
    {sN : ServerState State Token Fast}
    (htrace : DVRTrace pkg.refDecoder pkg.initialServer sN) :
    exists n : Nat,
      observe sN = (refRun pkg.refDecoder n pkg.initialRef).1 :=
  ServeDVR_refines_TorchLeanIRDecode
    pkg.cfg pkg.initialRef (pkg.initial_invariant) htrace

/-- Two executions of the same package that commit the same number of verified
tokens show the user the same tokens, even when their fast-path state,
speculative candidates, and batching schedule differ. -/
theorem observable_determinism_of_same_commit_length
    {α : Type} [TorchLean.Storage α] [Context α] [DecidableEq _root_.Spec.Shape]
    {State Token Fast : Type}
    (pkg : CausalTransformerIRPackage α State Token Fast)
    {sNa sNb : ServerState State Token Fast}
    (htraceA : DVRTrace pkg.refDecoder pkg.initialServer sNa)
    (htraceB : DVRTrace pkg.refDecoder pkg.initialServer sNb)
    (hlen : (observe sNa).length = (observe sNb).length) :
    observe sNa = observe sNb :=
  TorchLeanIR_user_observable_determinism_of_same_commit_length
    pkg.cfg pkg.initialRef
    (pkg.initial_invariant)
    (pkg.initial_invariant)
    htraceA htraceB hlen

end CausalTransformerIRPackage

end DVR

/-! ## Multi-step greedy decoding

Here the selected request may sit in a different batch at every generation step.
The theorem below lifts one-step request-local invariance to aligned traces.
-/

/-- One greedy decoding step for an abstract inference server. The selected
request reads its logits, chooses a token deterministically, and updates only its
own request-local state. -/
structure GreedyStep (State Logits Token : Type u) where
  forward : BatchedForward State Logits
  choose : Logits -> Token
  update : State -> Token -> State

def stepRequest {State Logits Token : Type u}
    (sem : GreedyStep State Logits Token)
    {B : Nat} (states : Fin B -> State) (i : Fin B) :
    Token × State :=
  let logits := sem.forward B states i
  let tok := sem.choose logits
  (tok, sem.update (states i) tok)

/-- A batch-invariant forward pass gives the selected request the same token and
next state in any batch. Induction over steps gives
`batched_greedy_trace_invariant`. -/
theorem greedy_step_observationally_invariant {State Logits Token : Type u}
    (sem : GreedyStep State Logits Token)
    (hforward : BatchInvariantForward sem.forward)
    {B C : Nat}
    (states1 : Fin B -> State) (states2 : Fin C -> State)
    (i : Fin B) (j : Fin C)
    (hstate : states1 i = states2 j) :
    stepRequest sem states1 i = stepRequest sem states2 j := by
  simp [stepRequest, hstate, hforward states1 states2 i j hstate]

/-- One server step: a batch and the position of the selected request in it.
Traces may change the batch size and the surrounding rows at every step. -/
structure StepContext (State : Type u) where
  B : Nat
  states : Fin B -> State
  selected : Fin B

namespace StepContext

def selectedState {State : Type u} (ctx : StepContext State) : State :=
  ctx.states ctx.selected

end StepContext

/-- Greedy tokens the selected request receives along a trace, one per step. -/
def tokensOfTrace {State Logits Token : Type u}
    (sem : GreedyStep State Logits Token) :
    List (StepContext State) -> List Token
  | [] => []
  | ctx :: rest =>
      (stepRequest sem ctx.states ctx.selected).1 :: tokensOfTrace sem rest

/-- Two traces are aligned when their selected request-local states agree at
each step. The co-batched rows and batch sizes may differ. -/
def SameSelectedTrace {State : Type u} :
    List (StepContext State) -> List (StepContext State) -> Prop
  | [], [] => True
  | ctx1 :: rest1, ctx2 :: rest2 =>
      ctx1.selectedState = ctx2.selectedState ∧ SameSelectedTrace rest1 rest2
  | _, _ => False

/-- If the forward pass is batch-invariant and two traces agree on the selected
request's state at every step, they produce the same tokens, whatever batches
surround the request. -/
theorem batched_greedy_trace_invariant {State Logits Token : Type u}
    (sem : GreedyStep State Logits Token)
    (hforward : BatchInvariantForward sem.forward) :
    forall (trace1 trace2 : List (StepContext State)),
      SameSelectedTrace trace1 trace2 ->
      tokensOfTrace sem trace1 = tokensOfTrace sem trace2
  | [], [], _ => rfl
  | [], _ :: _, h => False.elim h
  | _ :: _, [], h => False.elim h
  | ctx1 :: rest1, ctx2 :: rest2, h => by
      have hstate : ctx1.selectedState = ctx2.selectedState := h.1
      have htail : SameSelectedTrace rest1 rest2 := h.2
      have hstep :
          stepRequest sem ctx1.states ctx1.selected =
            stepRequest sem ctx2.states ctx2.selected :=
        greedy_step_observationally_invariant
          sem hforward ctx1.states ctx2.states ctx1.selected ctx2.selected hstate
      simp [tokensOfTrace, hstep,
        batched_greedy_trace_invariant sem hforward rest1 rest2 htail]

end BatchInvariantInference
