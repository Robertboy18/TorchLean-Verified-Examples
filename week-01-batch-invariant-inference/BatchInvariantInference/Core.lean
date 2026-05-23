/-
Batch-invariant inference as a TorchLean proof development.

The question behind the file is simple: when the same user request is served in
different batches, what exactly has to stay fixed for the returned token stream
to stay fixed?

The answer is developed in layers. First we make reduction schedules explicit.
Then we show why Float32 makes schedule changes observable. Then we prove the
same request-local contract for matmul, RMSNorm, attention-style schedules,
greedy token choice under margins, and a decode/verify/rollback serving loop.
Concrete CUDA/Triton kernels sit one layer below this file: they must refine the
semantics checked here.
-/

import NN
import NN.Floats.IEEEExec.Reductions
import Mathlib.Data.Rat.Lemmas
import Mathlib.Tactic.Linarith

universe u v w x

namespace BatchInvariantInference

/-- Use TorchLean's deployment-aware reduction tree as the schedule object.

Here, a `SumTree` is the mathematical record of "which pair of partial
sums was combined first." That order is invisible over exact reals, but it is
visible for Float32. -/
abbrev SumTree (ι : Type u) := TorchLean.Floats.IEEE754.SumTree ι

namespace SumTree

def leaf {ι : Type u} (i : ι) : SumTree ι :=
  TorchLean.Floats.IEEE754.SumTree.leaf i

def node {ι : Type u} (l r : SumTree ι) : SumTree ι :=
  TorchLean.Floats.IEEE754.SumTree.node l r

/-- Evaluate a schedule with an explicit combiner and leaf interpretation. -/
def evalWith {ι : Type u} {β : Type v}
    (combine : β -> β -> β) (leaf : ι -> β) : SumTree ι -> β
  | TorchLean.Floats.IEEE754.SumTree.leaf i => leaf i
  | TorchLean.Floats.IEEE754.SumTree.node l r =>
      combine (evalWith combine leaf l) (evalWith combine leaf r)

end SumTree

/-- A batched forward pass, viewed from the selected row/request.

`Fin B -> X` means "a batch with exactly `B` request-local states." A selected
request is then an index `Fin B`, so Lean knows the index is in bounds. -/
abbrev BatchedForward (X : Type u) (Y : Type v) :=
  (B : Nat) -> (Fin B -> X) -> Fin B -> Y

/-- The request-local property we want from a model forward pass.

If two selected requests have the same local state, their selected outputs
should agree even when they appear in different surrounding batches. -/
def BatchInvariantForward {X : Type u} {Y : Type v}
    (forward : BatchedForward X Y) : Prop :=
  forall {B C : Nat}
    (xs : Fin B -> X) (ys : Fin C -> X)
    (i : Fin B) (j : Fin C),
      xs i = ys j -> forward B xs i = forward C ys j

/-- A runtime forward path refines a proof-facing spec when every selected
request output agrees with the spec output on the same batch state. This is the
generic bridge used for CUDA/Triton kernels: once an external kernel is shown
to refine the schedule-explicit semantics, all semantic theorems transfer. -/
def RefinesForward {X : Type u} {Y : Type v}
    (runtime spec : BatchedForward X Y) : Prop :=
  forall {B : Nat} (xs : Fin B -> X) (i : Fin B),
    runtime B xs i = spec B xs i

/-- Batch-invariance transfers across a refinement proof. This is the theorem
that turns a verified kernel certificate into a serving-level invariant. -/
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

/-- Batched reduction for a selected row. The other rows are present in the
batch, but this semantic object only reads the selected row. -/
def batchedReduce {ι : Type u} {β : Type v} {B : Nat}
    (combine : β -> β -> β)
    (sched : SumTree ι)
    (rows : Fin B -> ι -> β)
    (b : Fin B) : β :=
  sched.evalWith combine (rows b)

/-- If the selected row has the same leaves and the same reduction schedule,
then the selected reduction result is independent of the surrounding batch. -/
theorem batchedReduce_batchInvariant {ι : Type u} {β : Type v}
    {B C : Nat}
    (combine : β -> β -> β)
    (sched : SumTree ι)
    (rows1 : Fin B -> ι -> β)
    (rows2 : Fin C -> ι -> β)
    (b : Fin B) (c : Fin C)
    (hrow : forall k, rows1 b k = rows2 c k) :
    batchedReduce combine sched rows1 b =
      batchedReduce combine sched rows2 c := by
  unfold batchedReduce
  induction sched with
  | leaf k =>
      exact hrow k
  | node l r ihL ihR =>
      simp [SumTree.evalWith, ihL, ihR]

/-- A runtime policy that chooses a reduction schedule for a selected row. -/
abbrev ScheduleSelector (ι : Type u) :=
  (B : Nat) -> Fin B -> SumTree ι

/-- The scheduling condition needed for request-level equality: selected rows
from different batch contexts receive the same schedule. -/
def BatchIndependentSchedule {ι : Type u} (choose : ScheduleSelector ι) : Prop :=
  forall {B C : Nat} (i : Fin B) (j : Fin C), choose B i = choose C j

/-- Request-local metadata that a real kernel may legitimately use when picking
a schedule. The important rule is not "every request uses one global schedule";
it is "the schedule may depend on local facts, but not on unrelated co-batched
requests." -/
structure LocalKernelCtx where
  hiddenDim : Nat
  seqLen : Nat
  kvLen : Nat
  dtypeTag : Nat
  blockSize : Nat
deriving Repr, DecidableEq

/-- A more realistic schedule selector: the runtime sees the batch context and
the selected row, but also receives request-local metadata. -/
abbrev ScheduleSelectorWithCtx (ι : Type u) :=
  (B : Nat) -> Fin B -> LocalKernelCtx -> SumTree ι

/-- Request-local schedule invariance: if two selected requests have the same
local kernel context, then changing the surrounding batch does not change the
selected request's reduction schedule. -/
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
  (choose B b).evalWith combine (rows b)

/-- A selected-row reduction as a batched forward pass. The input for each
request is its row of reduction leaves. -/
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
  (choose B b (ctxs b)).evalWith combine (rows b)

/-- If the schedule selector is batch-independent, the earlier selected-row
invariance theorem still applies even though scheduling is an explicit input. -/
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

/-- The context-aware version of selected-row reduction invariance. This is the
one closer to real kernels: schedules may vary with hidden size, sequence
length, dtype, or block size, but not with irrelevant co-batched requests. -/
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

/-- Two concrete parenthesizations of a three-term reduction. -/
def leftAssoc3 {β : Type v} (combine : β -> β -> β) (a b c : β) : β :=
  SumTree.evalWith combine id
    (SumTree.node (SumTree.node (SumTree.leaf a) (SumTree.leaf b)) (SumTree.leaf c))

/-- The other parenthesization of a three-term reduction. -/
def rightAssoc3 {β : Type v} (combine : β -> β -> β) (a b c : β) : β :=
  SumTree.evalWith combine id
    (SumTree.node (SumTree.leaf a) (SumTree.node (SumTree.leaf b) (SumTree.leaf c)))

/-- If the combiner is non-associative at `a,b,c`, then changing the schedule
can change the result. This is the formal shape of the floating-point bug. -/
theorem different_schedules_can_differ {β : Type v}
    (combine : β -> β -> β) (a b c : β)
    (h : combine (combine a b) c ≠ combine a (combine b c)) :
    leftAssoc3 combine a b c ≠ rightAssoc3 combine a b c := by
  simpa [leftAssoc3, rightAssoc3, SumTree.evalWith] using h

/-- A three-leaf row used to instantiate schedule sensitivity.

This is small on purpose: three terms are enough to witness non-associativity.
The later `IEEE32_batchDependentSchedule_counterexample` plugs in concrete
TorchLean Float32 values. -/
def row3 {β : Type v} (a b c : β) : Fin 3 -> β
  | ⟨0, _⟩ => a
  | ⟨1, _⟩ => b
  | _ => c

def leftTree3 : SumTree (Fin 3) :=
  SumTree.node (SumTree.node (SumTree.leaf (0 : Fin 3)) (SumTree.leaf (1 : Fin 3)))
    (SumTree.leaf (2 : Fin 3))

def rightTree3 : SumTree (Fin 3) :=
  SumTree.node (SumTree.leaf (0 : Fin 3))
    (SumTree.node (SumTree.leaf (1 : Fin 3)) (SumTree.leaf (2 : Fin 3)))

/-- A batch-size-dependent schedule selector: batch size one gets the left tree;
larger batch contexts get the right tree. This is the formal shape of the
runtime bug, not a recommended scheduling policy. -/
def chooseByBatchSize3 : ScheduleSelector (Fin 3) :=
  fun B _ => if B = 1 then leftTree3 else rightTree3

namespace IEEE32

abbrev Exec := TorchLean.Floats.IEEE754.IEEE32Exec

/-- `1e20` as an executable IEEE-style binary32 value. -/
def big : Exec := TorchLean.Floats.IEEE754.IEEE32Exec.ofBits (0x60AD78EC : UInt32)

/-- `-1e20` as an executable IEEE-style binary32 value. -/
def negBig : Exec := TorchLean.Floats.IEEE754.IEEE32Exec.ofBits (0xE0AD78EC : UInt32)

/-- `1.0` as an executable IEEE-style binary32 value. -/
def one : Exec := TorchLean.Floats.IEEE754.IEEE32Exec.ofBits (0x3F800000 : UInt32)

/-- Concrete non-associativity witness for TorchLean's executable IEEE32 model:
`(1e20 + -1e20) + 1 = 1`, while `1e20 + (-1e20 + 1) = 0`. -/
theorem add_nonassoc_witness :
    TorchLean.Floats.IEEE754.IEEE32Exec.add
      (TorchLean.Floats.IEEE754.IEEE32Exec.add big negBig) one ≠
      TorchLean.Floats.IEEE754.IEEE32Exec.add big
        (TorchLean.Floats.IEEE754.IEEE32Exec.add negBig one) := by
  decide

end IEEE32

/-- If the scalar combiner has a non-associative witness, then the bad
batch-size-dependent schedule is not batch-invariant. This is the formal bug:
the selected request row is the same, but changing the surrounding batch context
changes the selected request's reduction schedule. -/
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
  exact different_schedules_can_differ combine a b c h (by
    simpa [reduceForwardWithScheduleSelector, reduceWithScheduleSelector,
      chooseByBatchSize3, leftTree3, rightTree3, leftAssoc3, rightAssoc3,
      row, row3, SumTree.evalWith] using hEq)

/-- Concrete finite-precision punchline: under TorchLean's executable IEEE32
addition, a batch-size-dependent schedule selector can make a selected request
fail batch invariance. -/
theorem IEEE32_batchDependentSchedule_counterexample :
    ¬ BatchInvariantForward
      (reduceForwardWithScheduleSelector
        TorchLean.Floats.IEEE754.IEEE32Exec.add
        chooseByBatchSize3) := by
  exact batchDependentSchedule_counterexample
    TorchLean.Floats.IEEE754.IEEE32Exec.add
    IEEE32.big IEEE32.negBig IEEE32.one
    IEEE32.add_nonassoc_witness

/-- A schedule-explicit dot product. Matrix multiplication is many copies of
this object. -/
def scheduledDot {K : Type u} {β : Type v}
    (add mul : β -> β -> β)
    (sched : SumTree K)
    (x : K -> β)
    (w : K -> β) : β :=
  sched.evalWith add (fun k => mul (x k) (w k))

/-- A selected output element of a batched matrix multiply. -/
def batchedMatmul {K : Type u} {Out : Type v} {β : Type w} {B : Nat}
    (add mul : β -> β -> β)
    (sched : SumTree K)
    (x : Fin B -> K -> β)
    (w : Out -> K -> β)
    (b : Fin B)
    (o : Out) : β :=
  scheduledDot add mul sched (x b) (w o)

/-- If the selected input row, the selected weights, and the reduction schedule
are fixed, then the selected matmul output is batch-invariant. -/
theorem matmul_batchInvariant {K : Type u} {Out : Type v} {β : Type w}
    {B C : Nat}
    (add mul : β -> β -> β)
    (sched : SumTree K)
    (x1 : Fin B -> K -> β)
    (x2 : Fin C -> K -> β)
    (w1 w2 : Out -> K -> β)
    (i : Fin B) (j : Fin C) (o : Out)
    (hrow : forall k, x1 i k = x2 j k)
    (hweight : forall k, w1 o k = w2 o k) :
    batchedMatmul add mul sched x1 w1 i o =
      batchedMatmul add mul sched x2 w2 j o := by
  unfold batchedMatmul scheduledDot
  induction sched with
  | leaf k =>
      simp [SumTree.evalWith, hrow k, hweight k]
  | node l r ihL ihR =>
      simp [SumTree.evalWith, ihL, ihR]

/-- Read one row of a TorchLean vector batch as a function. This bridges the
earlier function-level lemmas to TorchLean's shape-indexed tensor API. -/
def tensorRow {α : Type} {B K : Nat}
    (x : Spec.Tensor α (.dim B (.dim K .scalar))) (b : Fin B) : Fin K -> α :=
  fun k => Spec.Tensor.toScalar (Spec.get (Spec.get x b) k)

/-- Read one output row of a TorchLean weight matrix as a function. -/
def tensorWeightRow {α : Type} {Out K : Nat}
    (w : Spec.Tensor α (.dim Out (.dim K .scalar))) (o : Fin Out) : Fin K -> α :=
  fun k => Spec.Tensor.toScalar (Spec.get (Spec.get w o) k)

/-- A selected scalar output of schedule-explicit tensor matmul. -/
def tensorBatchedMatmul {β : Type} {B K Out : Nat}
    (add mul : β -> β -> β)
    (sched : SumTree (Fin K))
    (x : Spec.Tensor β (.dim B (.dim K .scalar)))
    (w : Spec.Tensor β (.dim Out (.dim K .scalar)))
    (b : Fin B)
    (o : Fin Out) : β :=
  scheduledDot add mul sched (tensorRow x b) (tensorWeightRow w o)

/-- TorchLean tensor version of selected-row matmul invariance. The selected
input rows are equal as shape-indexed tensors, and the weight payload is shared. -/
theorem tensorMatmul_batchInvariant {β : Type} {B C K Out : Nat}
    (add mul : β -> β -> β)
    (sched : SumTree (Fin K))
    (x1 : Spec.Tensor β (.dim B (.dim K .scalar)))
    (x2 : Spec.Tensor β (.dim C (.dim K .scalar)))
    (w : Spec.Tensor β (.dim Out (.dim K .scalar)))
    (i : Fin B) (j : Fin C) (o : Fin Out)
    (hrow : Spec.get x1 i = Spec.get x2 j) :
    tensorBatchedMatmul add mul sched x1 w i o =
      tensorBatchedMatmul add mul sched x2 w j o := by
  unfold tensorBatchedMatmul
  apply matmul_batchInvariant
  · intro k
    unfold tensorRow
    rw [hrow]
  · intro k
    rfl

/-- Schedule-explicit RMSNorm for one row. We keep `scaleMean` and `rsqrt`
abstract so this theorem applies to real arithmetic, rounded-real FP32 models,
or executable IEEE-style scalar models. -/
def rmsNormRow {ι : Type u} {β : Type v}
    (add mul : β -> β -> β)
    (scaleMean rsqrt : β -> β)
    (sched : SumTree ι)
    (x weight : ι -> β) : ι -> β :=
  let sqsum := sched.evalWith add (fun k => mul (x k) (x k))
  let invRms := rsqrt (scaleMean sqsum)
  fun j => mul (mul (x j) invRms) (weight j)

/-- RMSNorm is batch-invariant for a selected row when the row, weights, and
hidden-dimension reduction schedule are fixed. -/
theorem rmsNorm_batchInvariant {ι : Type u} {β : Type v}
    (add mul : β -> β -> β)
    (scaleMean rsqrt : β -> β)
    (sched : SumTree ι)
    (x1 x2 weight1 weight2 : ι -> β)
    (hrow : forall k, x1 k = x2 k)
    (hweight : forall k, weight1 k = weight2 k) :
    rmsNormRow add mul scaleMean rsqrt sched x1 weight1 =
      rmsNormRow add mul scaleMean rsqrt sched x2 weight2 := by
  funext j
  induction sched with
  | leaf k =>
      simp [rmsNormRow, SumTree.evalWith, hrow, hweight]
  | node l r ihL ihR =>
      simp [rmsNormRow, SumTree.evalWith, hrow, hweight]

/-! ## Attention scheduling

Attention needs one more layer than RMSNorm or matmul. The numerical reductions
are still schedule-explicit, but the KV schedule must be derived from a
request-local layout policy rather than from the surrounding server batch.

The definitions below model the semantic contract we want from a fixed
split-size attention implementation:

* the selected request has a logical KV sequence,
* chunking/prefill choices are represented by `ChunkPlan`,
* a canonical layout policy maps the local context and chunk plan to an ordered
  KV reduction tree,
* canonicality says that changing the chunk plan does not change that tree for
  the same request-local context.

This is not a proof of FlashAttention, paged attention, or a CUDA kernel. It is
the Lean-side object those implementations would have to refine.
-/

namespace Attention

/-- A server-side way of slicing a request. The proof below does not inspect
the chunks; a canonical layout is precisely one whose selected-token KV
reduction tree is independent of this plan. -/
structure ChunkPlan where
  chunks : List Nat
deriving Repr, DecidableEq

/-- Request-local attention metadata. A real kernel may depend on these fields,
but not on unrelated co-batched requests. -/
structure AttentionLocalCtx where
  qLen : Nat
  kvLen : Nat
  headDim : Nat
  dtypeTag : Nat
  splitSize : Nat
deriving Repr, DecidableEq

/-- A canonical KV layout policy chooses the ordered KV reduction tree for a
selected request. The `canonical` field is the fixed-split-size style contract:
for the same request-local context, different chunk plans produce the same
logical KV reduction tree. -/
structure KVLayoutPolicy (KV : Type u) where
  kvTree : AttentionLocalCtx -> ChunkPlan -> SumTree KV
  canonical :
    forall {ctx1 ctx2 : AttentionLocalCtx} (chunk1 chunk2 : ChunkPlan),
      ctx1 = ctx2 -> kvTree ctx1 chunk1 = kvTree ctx2 chunk2

/-- A concrete canonical KV block plan. The ordered KV block tree is derived
from request-local context only, so server chunking cannot perturb the selected
token's KV reduction schedule. A concrete implementation can instantiate these
fields with blocks computed from fixed split size, KV length, and layout policy.
-/
structure FixedKVBlockPlan (KV : Type u) where
  first : AttentionLocalCtx -> KV
  rest : AttentionLocalCtx -> List KV

/-- Convert a nonempty ordered block list to a right-associated reduction tree. -/
def treeOfNonemptyList {KV : Type u} : KV -> List KV -> SumTree KV
  | head, [] => SumTree.leaf head
  | head, next :: rest => SumTree.node (SumTree.leaf head) (treeOfNonemptyList next rest)

/-- The concrete fixed-layout KV tree. The `ChunkPlan` argument is accepted
because the serving system may provide one, but this canonical schedule ignores
it and uses only request-local layout metadata. -/
def fixedSplitKVTree {KV : Type u}
    (plan : FixedKVBlockPlan KV)
    (ctx : AttentionLocalCtx)
    (_chunk : ChunkPlan) : SumTree KV :=
  treeOfNonemptyList (plan.first ctx) (plan.rest ctx)

/-- Fixed-split KV trees are canonical by construction: equal local contexts
produce equal KV schedules regardless of chunk plan. -/
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

/-- Direct form of the canonical-layout theorem: chunking alone cannot change
the ordered KV blocks. This is the formal starting point for fixed split-size
attention. -/
theorem canonical_blocks_independent_of_chunking {KV : Type u}
    (layout : KVLayoutPolicy KV)
    (ctx : AttentionLocalCtx)
    (chunk1 chunk2 : ChunkPlan) :
    layout.kvTree ctx chunk1 = layout.kvTree ctx chunk2 :=
  layout.canonical chunk1 chunk2 rfl

/-- The feature-dimension schedule may also depend on request-local attention
metadata, but it must not depend on unrelated batch context. -/
structure FeatureSchedulePolicy (Feature : Type u) where
  featureTree : AttentionLocalCtx -> SumTree Feature
  canonical :
    forall {ctx1 ctx2 : AttentionLocalCtx},
      ctx1 = ctx2 -> featureTree ctx1 = featureTree ctx2

/-- Request-local tensors needed for one selected attention token/head. `query`
is the selected query vector, `key` and `value` are the logical KV cache. -/
structure SelectedAttentionInput
    (Feature : Type u) (KV : Type v) (Out : Type w) (β : Type x) where
  query : Feature -> β
  key : KV -> Feature -> β
  value : KV -> Out -> β

/-- Schedule-explicit attention scores for one selected token/head. -/
def scores {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featureTree : SumTree Feature)
    (inp : SelectedAttentionInput Feature KV Out β) : KV -> β :=
  fun kv => scheduledDot add mul featureTree inp.query (inp.key kv)

/-- Schedule-explicit attention output for one selected token/head/output
coordinate. `softmax` is abstract but deterministic: it maps the complete
score vector to a KV weight function. -/
def scheduledAttentionOut
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featureTree : SumTree Feature)
    (kvTree : SumTree KV)
    (softmax : (KV -> β) -> KV -> β)
    (inp : SelectedAttentionInput Feature KV Out β)
    (out : Out) : β :=
  let sc := scores add mul featureTree inp
  let weight := softmax sc
  kvTree.evalWith add (fun kv => mul (weight kv) (inp.value kv out))

/-- Same query and keys imply the same selected-token score vector. -/
theorem scores_eq_of_same_inputs
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featureTree : SumTree Feature)
    (inp1 inp2 : SelectedAttentionInput Feature KV Out β)
    (hquery : forall f, inp1.query f = inp2.query f)
    (hkey : forall kv f, inp1.key kv f = inp2.key kv f) :
    scores add mul featureTree inp1 = scores add mul featureTree inp2 := by
  funext kv
  unfold scores scheduledDot
  induction featureTree with
  | leaf f =>
      simp [SumTree.evalWith, hquery f, hkey kv f]
  | node l r ihL ihR =>
      simp [SumTree.evalWith, ihL, ihR]

/-- If the selected query, logical KV cache, feature schedule, and KV schedule
are the same, then the selected attention output is the same. -/
theorem scheduledAttentionOut_eq_of_same_inputs
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featureTree : SumTree Feature)
    (kvTree : SumTree KV)
    (softmax : (KV -> β) -> KV -> β)
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
  induction kvTree with
  | leaf kv =>
      have hweight :
          softmax (scores add mul featureTree inp1) kv =
            softmax (scores add mul featureTree inp2) kv := by
        rw [hscores]
      simp [SumTree.evalWith, hweight, hvalue kv out]
  | node l r ihL ihR =>
      simp [SumTree.evalWith, ihL, ihR]

/-- Selected attention using a canonical KV layout and an explicit feature
reduction schedule. -/
def selectedAttentionWithLayout
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x} {B : Nat}
    (add mul : β -> β -> β)
    (featureTree : SumTree Feature)
    (layout : KVLayoutPolicy KV)
    (softmax : (KV -> β) -> KV -> β)
    (inputs : Fin B -> SelectedAttentionInput Feature KV Out β)
    (ctxs : Fin B -> AttentionLocalCtx)
    (chunks : Fin B -> ChunkPlan)
    (b : Fin B)
    (out : Out) : β :=
  scheduledAttentionOut add mul featureTree
    (layout.kvTree (ctxs b) (chunks b)) softmax (inputs b) out

/-- Selected attention where both feature and KV schedules are request-local
policies. This is the proof-facing form for Transformer attention: feature
reductions and KV reductions are both explicit schedule objects. -/
def selectedAttentionWithPolicies
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x} {B : Nat}
    (add mul : β -> β -> β)
    (featurePolicy : FeatureSchedulePolicy Feature)
    (layout : KVLayoutPolicy KV)
    (softmax : (KV -> β) -> KV -> β)
    (inputs : Fin B -> SelectedAttentionInput Feature KV Out β)
    (ctxs : Fin B -> AttentionLocalCtx)
    (chunks : Fin B -> ChunkPlan)
    (b : Fin B)
    (out : Out) : β :=
  scheduledAttentionOut add mul
    (featurePolicy.featureTree (ctxs b))
    (layout.kvTree (ctxs b) (chunks b))
    softmax (inputs b) out

/-- Fixed-split/canonical-layout selected attention theorem. Different
surrounding batches and different chunk plans cannot change the selected output
when the selected request-local context and logical Q/K/V payload are the same.
-/
theorem selectedAttention_batchInvariant_of_canonicalLayout
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x} {B C : Nat}
    (add mul : β -> β -> β)
    (featureTree : SumTree Feature)
    (layout : KVLayoutPolicy KV)
    (softmax : (KV -> β) -> KV -> β)
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

/-- Schedule-policy selected attention theorem. If feature schedules and KV
schedules are canonical functions of the selected request-local context, then
attention is invariant to surrounding batch context and server chunking. -/
theorem selectedAttention_batchInvariant_of_schedulePolicies
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x} {B C : Nat}
    (add mul : β -> β -> β)
    (featurePolicy : FeatureSchedulePolicy Feature)
    (layout : KVLayoutPolicy KV)
    (softmax : (KV -> β) -> KV -> β)
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

TorchLean already has a CUDA fused attention path and a proof-facing
FlashAttention denotation in `NN.Spec.Layers.FlashAttention`. The native code is
not inspected by Lean, but the theorem boundary should be precise: a native
kernel certificate must show that the runtime path refines the schedule-explicit
attention semantics. Once that certificate is supplied, batch-invariance follows
from the semantic theorem above.
-/

namespace CUDARuntime

/-- One selected attention request as seen by a fused CUDA/Triton attention
kernel: logical Q/K/V payload, request-local attention metadata, and the server
chunk plan used by the runtime. -/
structure SelectedAttentionRequest
    (Feature : Type u) (KV : Type v) (Out : Type w) (β : Type x) where
  input : Attention.SelectedAttentionInput Feature KV Out β
  ctx : Attention.AttentionLocalCtx
  chunk : Attention.ChunkPlan

/-- The schedule-explicit selected-attention spec as a batched forward pass. -/
def scheduledAttentionForward
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featurePolicy : Attention.FeatureSchedulePolicy Feature)
    (layout : Attention.KVLayoutPolicy KV)
    (softmax : (KV -> β) -> KV -> β) :
    BatchedForward
      (SelectedAttentionRequest Feature KV Out β)
      (Out -> β) :=
  fun B reqs b out =>
    Attention.selectedAttentionWithPolicies add mul featurePolicy layout softmax
      (fun i : Fin B => (reqs i).input)
      (fun i : Fin B => (reqs i).ctx)
      (fun i : Fin B => (reqs i).chunk)
      b out

/-- The schedule-explicit attention spec is batch-invariant when its feature
and KV schedules are canonical functions of request-local metadata. -/
theorem scheduledAttentionForward_batchInvariant
    {Feature : Type u} {KV : Type v} {Out : Type w} {β : Type x}
    (add mul : β -> β -> β)
    (featurePolicy : Attention.FeatureSchedulePolicy Feature)
    (layout : Attention.KVLayoutPolicy KV)
    (softmax : (KV -> β) -> KV -> β) :
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

/-- Proof-carrying contract for a native fused attention forward path. The
`runtimeForward` field can represent TorchLean's CUDA FFI path, a Triton kernel,
or a hosted verifier replay. The proof obligation is exactly refinement to the
schedule-explicit selected-attention spec. -/
structure FusedAttentionRuntimeCert
    (Feature : Type u) (KV : Type v) (Out : Type w) (β : Type x) where
  add : β -> β -> β
  mul : β -> β -> β
  featurePolicy : Attention.FeatureSchedulePolicy Feature
  layout : Attention.KVLayoutPolicy KV
  softmax : (KV -> β) -> KV -> β
  runtimeForward :
    BatchedForward
      (SelectedAttentionRequest Feature KV Out β)
      (Out -> β)
  refinesScheduleSpec :
    RefinesForward runtimeForward
      (scheduledAttentionForward add mul featurePolicy layout softmax)

/-- If a CUDA/Triton fused attention runtime refines the checked
schedule-explicit attention spec, then the runtime itself is batch-invariant.
This is the exact theorem a real kernel certificate should target. -/
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

/-- Runtime refinement certificate for TorchLean's proof-facing FlashAttention
operator. The native CUDA path in `csrc/cuda/kernels/torchlean_cuda_kernels.cu`
should be validated against `Spec.cudaLoopFlashAttention`; this certificate is
the Lean boundary once that validation has been supplied by tests, an analyzer,
or a future proof-carrying kernel extractor. -/
structure NativeForwardCert
    (α : Type) [Context α] [DecidableRel ((· > ·) : α -> α -> Prop)]
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0} where
  cfg : Spec.FlashAttentionConfig
  ctx : Spec.AttentionContext α nQ nK dModel h1 h2
  runtimeOut : Spec.Tensor α (Spec.Shape.dim nQ (Spec.Shape.dim dModel Spec.Shape.scalar))
  runtime_eq_cudaLoop :
    runtimeOut = Spec.cudaLoopFlashAttention cfg ctx

/-- A native output that refines TorchLean's CUDA-loop FlashAttention
denotation also refines the standard scaled-dot-product attention spec. -/
theorem NativeForwardCert.refines_scaledDotProduct
    {α : Type} [Context α] [DecidableRel ((· > ·) : α -> α -> Prop)]
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (cert : NativeForwardCert α (nQ := nQ) (nK := nK) (dModel := dModel)
      (h1 := h1) (h2 := h2)) :
    cert.runtimeOut = Spec.scaledDotProductAttention cert.ctx := by
  calc
    cert.runtimeOut = Spec.cudaLoopFlashAttention cert.cfg cert.ctx :=
      cert.runtime_eq_cudaLoop
    _ = Spec.scaledDotProductAttention cert.ctx :=
      Spec.cudaLoopFlashAttention_eq_scaledDotProductAttention cert.cfg cert.ctx

/-- The same native output also refines TorchLean's named FlashAttention
semantic operator. This is the bridge from the existing TorchLean FlashAttention
spec to the runtime-refinement certificate above. -/
theorem NativeForwardCert.refines_flashAttention
    {α : Type} [Context α] [DecidableRel ((· > ·) : α -> α -> Prop)]
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (cert : NativeForwardCert α (nQ := nQ) (nK := nK) (dModel := dModel)
      (h1 := h1) (h2 := h2)) :
    cert.runtimeOut = Spec.flashAttention cert.cfg cert.ctx := by
  calc
    cert.runtimeOut = Spec.scaledDotProductAttention cert.ctx :=
      cert.refines_scaledDotProduct
    _ = Spec.flashAttention cert.cfg cert.ctx :=
      (Spec.flashAttention_eq_scaledDotProductAttention cert.cfg cert.ctx).symm

end TorchLeanFlashAttention

/-- A pointwise layer is batch-invariant because it only reads the selected row. -/
def pointwiseForward {X : Type u} {Y : Type v} (f : X -> Y) :
    BatchedForward X Y :=
  fun _ xs i => f (xs i)

theorem pointwise_batchInvariant {X : Type u} {Y : Type v} (f : X -> Y) :
    BatchInvariantForward (pointwiseForward f) := by
  intro B C xs ys i j h
  simp [pointwiseForward, h]

/-- Compose two batch-aware forward passes. -/
def composeForward {X : Type u} {Y : Type v} {Z : Type w}
    (f : BatchedForward X Y) (g : BatchedForward Y Z) :
    BatchedForward X Z :=
  fun B xs i => g B (fun k => f B xs k) i

/-- Batch invariance composes over a graph/layer pipeline. This is the
composition principle used by graph-level invariants over a TorchLean IR DAG. -/
theorem compose_batchInvariant {X : Type u} {Y : Type v} {Z : Type w}
    (f : BatchedForward X Y) (g : BatchedForward Y Z)
    (hf : BatchInvariantForward f)
    (hg : BatchInvariantForward g) :
    BatchInvariantForward (composeForward f g) := by
  intro B C xs ys i j h
  apply hg
  exact hf xs ys i j h

/-! ## Logit margins and argmax stability

The equality theorems above are the bitwise-reproducibility path. In deployed
inference, a more useful claim is often margin-based: logits may move slightly
when schedules change, but the greedy token is stable if the top logit has
enough margin.
-/

namespace Margin

/-- Absolute value for rationals. We use `Rat` here to keep the theorem
executable and elementary while still matching the usual real-valued statement. -/
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

/-- If two quantities are each close to the same reference, they are close to
each other. This is the small numerical bridge behind schedule-drift
certificates. -/
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

/-- Two logit vectors are uniformly close by `eps`. -/
def LinfClose {Vocab : Type u} (z z' : Vocab -> Rat) (eps : Rat) : Prop :=
  forall tok, absQ (z' tok - z tok) <= eps

/-- A named certificate shape for "this scheduled/runtime logit vector is within
`eps` of a reference vector." In a fuller TorchLean development, the hypotheses
can be discharged by reduction enclosures such as `dotTreeResult_enclosure`. -/
abbrev LogitDriftBound {Vocab : Type u}
    (reference observed : Vocab -> Rat) (eps : Rat) : Prop :=
  LinfClose reference observed eps

/-- If two scheduled logit vectors are both within `eps` of the same reference
logits, then they are within `2*eps` of each other. -/
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

/-- The old winner remains strictly above every non-best token under bounded
logit drift when the original margin is larger than `2*eps`. -/
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

/-- A certified greedy choice: `choose z = best`, and `best` has margin `margin`.
We keep the choice function explicit so the theorem applies to any deterministic
tie-breaking implementation. -/
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

/-- If `choose` implements strict argmax, a certified margin converts bounded
logit drift into actual greedy-token stability. -/
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

/-- Reduction-error-to-token theorem: if two scheduled/runtime logit vectors are
each certified within the same error budget of a reference vector with a strict
greedy margin, then both schedules choose the same token. This is the useful
statement when bitwise equality fails but numerical drift is bounded. -/
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

/-- Same result, stated from one schedule to another using the derived
`2*eps` closeness. This is useful when the margin certificate is attached to the
first scheduled logit vector rather than to the exact reference. -/
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

The previous sections prove local kernel and token-stability facts. The serving
theorem below is the more systems-shaped result: a fast path may speculate using
arbitrary dynamic batching and schedule-dependent numerics, but user-visible
tokens are appended only through a verifier-sound reference window. Therefore
the committed output is always a prefix of the canonical reference decoder.
-/

namespace DVR

/-- Canonical finite-precision reference decoder. In a full Transformer theorem,
`State` would contain the prompt, KV cache, position, and parameter payload, and
`step` would evaluate the TorchLean Transformer IR under fixed reference
schedules. -/
structure ReferenceDecoder (State Token : Type u) where
  step : State -> Token × State

/-- Run the canonical decoder for `n` tokens. -/
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

/-- A server state contains committed user-visible tokens, the reference state
after those committed tokens, and arbitrary fast-path/speculative state. -/
structure ServerState (State Token Fast : Type u) where
  committed : List Token
  refState : State
  fastState : Fast
  candidates : List Token

def observe {State Token Fast : Type u} (s : ServerState State Token Fast) : List Token :=
  s.committed

/-- A verifier-sound accepted window is exactly some number of canonical
reference tokens from the current reference state, together with the resulting
reference state. -/
def VerifyWindowSound {State Token : Type u}
    (ref : ReferenceDecoder State Token)
    (st : State) (accepted : List Token) (st' : State) : Prop :=
  exists n : Nat, accepted = (refRun ref n st).1 ∧ st' = (refRun ref n st).2

/-- The committed output and stored reference state agree with running the
canonical decoder from the initial reference state for some number of tokens. -/
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

/-- Legal serving steps. The fast path may speculate or roll back arbitrarily.
The only rule that changes user-visible committed tokens is `verifyAccept`, and
it requires a verifier-sound reference window. -/
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

/-- One legal DVR step preserves reference equivalence. -/
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

/-- Reference-equivalent dynamic serving: regardless of arbitrary speculation,
dynamic batching, fast-path schedules, and rollbacks, every observable committed
token stream produced by a legal DVR trace is exactly a canonical reference
decoder prefix. -/
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

/-- Two arbitrary dynamic-serving traces from the same reference decoder produce
the same user-visible output whenever they commit the same number of tokens.
This is the observable determinism statement: schedule choices do not affect the
committed token prefix. -/
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

/-- A concrete TorchLean-IR reference decoder. One reference step turns the
current request state into an `NN.IR.DVal`, evaluates a shared `NN.IR.Graph`
with an explicit payload, chooses a token from the denotational result, and
updates the request-local state. This is the bridge from the abstract DVR
serving theorem to TorchLean's actual op-tagged graph semantics. -/
structure IRDecoderConfig
    (α : Type) [Context α]
    (State Token : Type) where
  graph : NN.IR.Graph
  payload : NN.IR.Payload α
  outputId : Nat
  inputOf : State -> NN.IR.DVal α
  choose : Except String (NN.IR.DVal α) -> Token
  update : State -> Token -> State
  referenceStep : State -> Token × State
  referenceStep_eq_denote :
    forall st : State,
      referenceStep st =
        let out : Except String (NN.IR.DVal α) := NN.IR.Graph.denote
          (α := α) (g := graph) (payload := payload)
          (input := inputOf st) (outputId := outputId)
        let tok := choose out
        (tok, update st tok)

def toReferenceDecoder
    {α : Type} [Context α] [Inhabited α] [DecidableEq _root_.Spec.Shape]
    {State Token : Type}
    (cfg : IRDecoderConfig α State Token) :
    ReferenceDecoder State Token where
  step := cfg.referenceStep

/-- Verifier-sound windows for a TorchLean-IR decoder are exactly windows that
match the canonical `NN.IR.Graph.denote`-based reference decoder. Executable
certificate checkers should target this predicate. -/
def IRVerifyWindowSound
    {α : Type} [Context α] [Inhabited α] [DecidableEq _root_.Spec.Shape]
    {State Token : Type}
    (cfg : IRDecoderConfig α State Token)
    (st : State) (accepted : List Token) (st' : State) : Prop :=
  VerifyWindowSound
    (toReferenceDecoder (cfg := cfg))
    st accepted st'

/-- Reference-equivalent serving specialized to TorchLean's actual IR
semantics. For any legal DVR trace whose commits are sound with respect to
`NN.IR.Graph.denote`, the user-visible tokens are a prefix of the canonical
TorchLean-IR decoder. -/
theorem ServeDVR_refines_TorchLeanIRDecode
    {α : Type} [Context α] [Inhabited α] [DecidableEq _root_.Spec.Shape]
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

/-- Two arbitrary dynamic-serving traces from the same TorchLean-IR reference
decoder produce the same committed tokens whenever they commit the same number
of tokens. -/
theorem TorchLeanIR_user_observable_determinism_of_same_commit_length
    {α : Type} [Context α] [Inhabited α] [DecidableEq _root_.Spec.Shape]
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

/-- Concrete metadata for a causal Transformer decoder package. The theorem
below does not depend on these numbers computationally; they keep the package
honest about the model family being served. -/
structure TransformerDims where
  seqLen : Nat
  vocabSize : Nat
  dModel : Nat
  numHeads : Nat
  headDim : Nat
  ffnHidden : Nat
deriving Repr, DecidableEq

/-- A concrete request-local state shape for decoder-only serving. `KV` is kept
abstract because different deployments represent KV caches differently, but the
state explicitly records the prompt, committed/generated suffix, cache payload,
and next position. -/
structure CausalRequestState (Token KV : Type u) where
  prompt : List Token
  generated : List Token
  kvCache : KV
  position : Nat
deriving Repr

/-- A proof-facing package for serving a concrete TorchLean causal Transformer
decoder. It contains the actual TorchLean IR graph and payload, the request
encoder, token chooser, request-local update, initial state, and graph checker
evidence.

The package is intentionally generic in the request state and token type: a
real instantiation can choose byte tokens, BPE tokens, KV-cache state, logits
postprocessing, or a margin-certified argmax policy. What is fixed here is the
important semantic target: the reference step must be exactly
`NN.IR.Graph.denote` followed by token choice and request-local update. -/
structure CausalTransformerIRPackage
    (α : Type) [Context α] [Inhabited α] [DecidableEq _root_.Spec.Shape]
    (State Token Fast : Type) where
  dims : TransformerDims
  cfg : IRDecoderConfig α State Token
  initialRef : State
  initialFast : Fast
  graph_wellFormed : cfg.graph.WellFormed
  graph_wellShaped : cfg.graph.WellShaped

/-- The common concrete specialization: a decoder package whose request state
has prompt tokens, generated tokens, a request-local KV cache, and a position. -/
abbrev ConcreteCausalTransformerPackage
    (α : Type) [Context α] [Inhabited α] [DecidableEq _root_.Spec.Shape]
    (Token KV Fast : Type) :=
  CausalTransformerIRPackage α (CausalRequestState Token KV) Token Fast

namespace CausalTransformerIRPackage

def refDecoder
    {α : Type} [Context α] [Inhabited α] [DecidableEq _root_.Spec.Shape]
    {State Token Fast : Type}
    (pkg : CausalTransformerIRPackage α State Token Fast) :
    ReferenceDecoder State Token :=
  toReferenceDecoder (cfg := pkg.cfg)

def initialServer
    {α : Type} [Context α] [Inhabited α] [DecidableEq _root_.Spec.Shape]
    {State Token Fast : Type}
    (pkg : CausalTransformerIRPackage α State Token Fast) :
    ServerState State Token Fast :=
  { committed := ([] : List Token),
    refState := pkg.initialRef,
    fastState := pkg.initialFast,
    candidates := ([] : List Token) }

theorem initial_invariant
    {α : Type} [Context α] [Inhabited α] [DecidableEq _root_.Spec.Shape]
    {State Token Fast : Type}
    (pkg : CausalTransformerIRPackage α State Token Fast) :
    DVRInvariant (pkg.refDecoder) pkg.initialRef pkg.initialServer :=
  invariant_of_initial (pkg.refDecoder) pkg.initialRef pkg.initialFast

/-- Package-level reference equivalence theorem. A concrete Transformer
instance only has to fill the package fields and ensure each committed window is
checked against `pkg.refDecoder`; every legal decode/verify/rollback trace then
releases exactly a prefix of the canonical TorchLean-IR decoder. -/
theorem serve_refines_reference
    {α : Type} [Context α] [Inhabited α] [DecidableEq _root_.Spec.Shape]
    {State Token Fast : Type}
    (pkg : CausalTransformerIRPackage α State Token Fast)
    {sN : ServerState State Token Fast}
    (htrace : DVRTrace pkg.refDecoder pkg.initialServer sN) :
    exists n : Nat,
      observe sN = (refRun pkg.refDecoder n pkg.initialRef).1 :=
  ServeDVR_refines_TorchLeanIRDecode
    pkg.cfg pkg.initialRef (pkg.initial_invariant) htrace

/-- Two dynamic-serving executions of the same packaged Transformer decoder are
observationally equal whenever they commit the same number of verified tokens.
The fast-path state, speculative candidates, and batching schedule may differ. -/
theorem observable_determinism_of_same_commit_length
    {α : Type} [Context α] [Inhabited α] [DecidableEq _root_.Spec.Shape]
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

The serving statement is about generated token streams under changing batch
contexts. The theorem below lifts one-step request-local invariance to aligned
batch-context traces.
-/

/-- One greedy decoding step for an abstract inference server. The selected
request observes logits, chooses a token deterministically, and updates only
its own request-local state. -/
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

/-- If the forward pass is batch-invariant, then one greedy decoding step is
observationally invariant for the selected request. Repeating this theorem by
induction gives the usual multi-token decoding statement. -/
theorem greedy_step_observationally_invariant {State Logits Token : Type u}
    (sem : GreedyStep State Logits Token)
    (hforward : BatchInvariantForward sem.forward)
    {B C : Nat}
    (states1 : Fin B -> State) (states2 : Fin C -> State)
    (i : Fin B) (j : Fin C)
    (hstate : states1 i = states2 j) :
    stepRequest sem states1 i = stepRequest sem states2 j := by
  simp [stepRequest, hstate, hforward states1 states2 i j hstate]

/-- One abstract server step context: the selected request appears in some
batch with some co-batched requests. Different traces may use different batch
sizes and different surrounding rows at every step. -/
structure StepContext (State : Type u) where
  B : Nat
  states : Fin B -> State
  selected : Fin B

namespace StepContext

def selectedState {State : Type u} (ctx : StepContext State) : State :=
  ctx.states ctx.selected

end StepContext

/-- Observed greedy tokens along a batch-context trace. -/
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

/-- Batch-aware trace theorem: at every generation step, the selected request
may be placed in a different batch context. If the selected local state agrees
step-by-step and the forward pass is batch-invariant, the observed token stream
is identical. -/
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
