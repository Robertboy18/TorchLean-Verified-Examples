# Writing tensor programs in Lean

I've wanted to use TorchLean for calculations that don't need a neural-network model around
them. If I'm experimenting with an activation, or repeating an update in a simulation, I'd
like to write that function in Lean, pick CPU or GPU when I run it, and then be able to prove
something about the function I wrote.

TorchLean's tensor library already gives us operations on shaped tensors, automatic differentiation
(autograd), and FloatLib's numerical formats. Autograd records the operations and intermediate
values it needs to propagate derivatives backward (I'll call that recording a tape below).
What I've been working on is connecting custom functions to those
pieces, and we can now compile supported Lean functions for GPU execution and use them through
the tensor API.

A GPU is useful when a calculation has lots of independent work. Apply the same update to a
million starting values and you get a million computations that can run alongside one another,
but inside each computation the next update may depend on the previous one. So we need a way to
express both (the independent work and that dependency) without maintaining another version of
the function in CUDA.

Lean is an interesting place to do this. It's a functional programming language as well as a
theorem prover, so we can write the calculation, evaluate it and prove things about its
definition, and I want all of that to feel like parts of the same project.
And the GPU compiler adds one more question we can state precisely: does the lowered program
still compute the function we wrote?

Let's try it with a repeated update, a custom loss, and a row sum, and then write matrix
products and attention with `einsum`, compose a model, and look at what takes time when
we run it. I also want to show the precision and higher-derivative work, because those
features change which calculations we can write at all (not only how we spell them).

<div id="where-this-came-from"></div>

## Functional programs on a GPU

Functional programming has a long history of treating a collection of calculations as a
program the compiler can transform. In
[Accelerate](https://www.acceleratehs.org/documentation/users-guide/language.html), you write
array computations inside Haskell (with operations such as maps and reductions), and its backends
compile those computations for a target device. That split between the host language
and the embedded array program matters: an ordinary Haskell function doesn't become a GPU
program merely because you're using Accelerate.

[Futhark](https://futhark-lang.org/publications/pldi17.pdf) goes the other way and designs
a whole functional array language for GPU compilation. Its work on nested parallelism
asks how a compiler can expose useful parallel work while keeping loops and updates
efficient, because a short functional expression can describe many calculations and the
compiler still has to decide how to organize them on the machine.

[Futhark's introduction](https://futhark-lang.org/) gives a motivation I share:

> Because it's nicer than writing CUDA or OpenCL by hand!

Here we also get to write a theorem about the function before and after lowering it, and
that connection is what I want to keep as we make the compiler more useful.

I find [Dex](https://arxiv.org/abs/2104.05372) particularly relevant to tensor programming.
It makes indices part of how we write array computations, and it studies differentiation that
preserves their parallel structure. Indices tell us which entries a calculation reads and
which outputs it produces, which is information a compiler and a proof both want.

[Bend](https://github.com/bendlang/bend) exposes parallel work through functions too.
Its [parallel-call model](https://github.com/bendlang/bend/blob/main/guide/GUIDE.md#parallelism)
lets a programmer split independent calls and run them through a CPU/GPU runtime. It's
after something broader than our current tensor compiler (recursive data, divide-and-conquer
programs), while our custom path starts with tensor entries and ordered numerical loops.

Bend also uses affine ownership, meaning an ordinary value can be used at most once unless it's
explicitly reusable, and that restriction helps it control sharing and updates. It's a different
design choice from adding a tensor compiler to Lean. We aren't introducing an affine language
here, or replacing Lean's runtime.

We start from the tensor system and the certified `einsum` compiler we already have
(which connects a generated computation to reference semantics), and we want to extend that
approach to functions users write themselves. Mathlib supplies the mathematics we can reuse
for proofs. FloatLib
lets us describe the arithmetic instead of assuming that a floating-point calculation is
an exact real-number one.

That numerical support is also a real difference from Bend's current implementation, whose
[documented limitations](https://github.com/bendlang/bend#limitations) describe native
F32 arithmetic as axiomatic. FloatLib supplies executable arithmetic with correctness and
rounding theorems. Connecting those theorems to a foreign GPU implementation is further work,
but we already have the numerical definitions to make that question precise. I wouldn't read
a speed advantage into any of this, though. Speed needs a matched experiment.

<div id="one-function-many-entries"></div>

## Square, blend, and choose a device

Let's square the values 1, 2, and 3:

<pre data-lean-example="tensorFunction"></pre>

The answer is `[1, 4, 9]`. Each output needs only its own input, so the three
calculations are independent, and a tensor with a million entries would expose the same kind
of work. Shape stays where it belongs, on the tensor. Square each entry of a matrix and
every result lands in the same row and column, so you can use this function on different tensor
shapes without naming a new operation for every rank.

This definition also leaves the number type open. So who picks it? In the code, `α` is the
type of an entry and `[Mul α]` asks for multiplication on that type, so it's the input that picks `Float32`
for this example. Later we'll use the same generic `square` with a wider FloatLib format.

We choose `cpu` or `gpu` when we run the function. CPU evaluates the Lean calculation.
On GPU, the custom path recognizes supported arithmetic, produces a checked representation,
and generates CUDA source that NVIDIA's [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)
compiles at runtime. None of that leaks into the function itself, which never mentions a device.

A square is a small place to start, but the whole reason for this interface is to let the
body grow (local bindings, arithmetic comparisons and supported conditionals can all go
in it). You should be able to try a calculation before deciding it deserves a
permanent name in the library.

<div id="combine-two-tensors-with-a-parameter"></div>

An update often needs two inputs: a previous estimate and a new observation, or two fields
we want to interpolate. Let's blend corresponding entries with a weight of one quarter:

<pre data-lean-example="tensorBlend"></pre>

For the first pair, three quarters of zero plus one quarter of eight gives two, the middle
pair stays at four, and the last gives six. If you want a different weight you change the
number, and there's no need for a new function named after a particular coefficient.

Both inputs have shape `[3]`, and the type of `.zip` requires the shapes to agree, so it can't
truncate one input to the other's length or pick a broadcasting we didn't ask for.
That shared shape carries over to the result, and a tensor of another rank can use the same scalar blend.

All the entries share that weight. What happens when you change it? When we compile for GPU,
a captured scalar like this one becomes a constant in the generated source, so changing it can mean recompiling (the runtime
does reuse identical source through its cache). Per-entry values that change often can come
in through tensor inputs instead, and you want to know which of the two you're doing before you
drop a small example into a longer loop.

<div id="a-loop-inside-each-tensor-entry"></div>

## Repeated updates and their closed forms

Now repeat the update $x\mapsto x^2+1$. Starting at one, the first update gives two, the
second five, and the third twenty-six. Starting at two gives a different history, and
neither history ever needs to read the other.

<figure class="interactive" id="tensor-recurrence">
  <div class="margin-controls">
    <label>Number of updates
      <input type="range" min="0" max="3" value="2" step="1" aria-label="Number of recurrence updates">
      <span class="recurrence-count">2</span>
    </label>
  </div>
  <output aria-live="polite"></output>
  <figcaption>Follow two entries through the same update. The two rows can run side by side, but
    within a row each arrow has to wait for the value before it. Your browser rounds each
    multiplication and addition to binary32 here, so don't read it as a GPU measurement.</figcaption>
</figure>

Move the slider back to zero. The output is the starting value, and at two updates the
outputs are five and twenty-six. That's also how we define the recurrence in Lean:

<pre data-lean-example="tensorRecurrence"></pre>

The zero case returns the input. In the successor case we compute the previous result, square
it and add one, so the recursion spells out the dependency (the new value comes from the
old one). And I haven't written a separate implementation for GPU.

For a forward-only GPU call, the frontend uses the recurrence's zero and successor equations
to derive an ordered loop, and it constructs a Lean proof that the loop agrees with those
equations. It doesn't paste the body into the generated program once per update.
Ask for a bigger count and you give the same loop more work to do.

This also makes the available parallelism easy to see. A thousand starting values can evolve
independently, but ten updates of one value form a dependency chain, and no compiler can
run the second update before it knows the first result. Some recurrences do admit more elaborate
parallel algorithms. Using one would mean showing that the reorganized computation still has
the meaning we intend.

Right now the recursion we recognize keeps its parameters fixed and recurses on the predecessor
of its final natural-number argument, and in the call a `UInt64` count gets converted to a natural
number, which gives the GPU loop a bounded count without truncating an arbitrary `Nat`.
Tree recursion and dynamically allocated recursive structures are outside this path
(ordinary CPU Lean can express a much larger range of programs).

Those boundaries are why I find the connection to functional GPU languages useful.
Parallelizing a collection of ordered updates is one kind of problem, and scheduling a recursive
tree whose branches have different amounts of work is a different one. For the first, I think the
tensor interface gives us a good start. It doesn't solve the second for free.

<div id="an-update-we-can-solve-exactly"></div>

Squaring blows up quickly, so let's try an update with a useful numerical purpose:
move part of the remaining distance to a target. If the current value is zero, the target
is two, and we move halfway each time, the first three values are one, one and a half,
and one and three quarters.

<pre data-lean-example="tensorRelaxation"></pre>

Our tensor starts at `[0, 2, 4]`, and after three updates its entries are `[1.75, 2, 2.25]`:
the middle entry already sits at the target and stays put, while the other two close in
on it from opposite sides. To take the GPU route for the same recurrence you call `approach 3 (device := gpu)`.

But can we understand every step without running all of them? If we write the rate as $r$,
the target as $t$ and the initial value as $x_0$, subtracting the target from one update gives

<div class="math-display" data-tex="x_{n+1}-t=(1-r)(x_n-t),\qquad x_n=t+(1-r)^n(x_0-t).">Each update multiplies the remaining error by one minus the rate.</div>

I proved the second identity for every real rate, target, initial value and
natural-number step count, and my proof uses the same generic `relax` definition that the
executable example runs, with induction handling the repeated updates and mathlib's `ring` tactic
checking the polynomial identity in the successor case.

That formula tells us how the rate changes the behavior. A rate of one half halves
the remaining distance. At one and a half the error gets multiplied by −1/2, so the value
jumps across the target every time even though the distance still shrinks. At rate two the
factor is −1 and the error keeps oscillating, and any rate greater than two amplifies it.

One identity covers all of that, every natural-number step count and real initial value,
including the rates that make the update unstable. The rounded recurrence is another matter.
There we'd need to add a finite-precision error bound using the numerical operations we selected,
and I haven't done that yet.

<div id="differentiate-the-calculation-we-wrote"></div>

## Gradients and a custom loss

That recurrence block also records its two updates and runs backward through them. In real
arithmetic, two updates give

<div class="math-display" data-tex="f(x)=(x^2+1)^2+1,\qquad f'(x)=4x(x^2+1).">Two updates and their derivative.</div>

At one the derivative is eight, and at two it's forty. Backward walks both updates in reverse
and multiplies their local derivatives, because if we differentiated only the final square
we'd miss how the first update depended on the input.

All of this goes through the existing tape. You keep the function definition and ask for recording
with `grad := true`. Recording holds on to intermediate values for backward, so it costs
something different from the forward-only compiled loop (on GPU, a recorded recurrence
currently launches tape operations per step).

We can write a whole-tensor function through the same interface, so let's add up the squares:

<pre data-lean-example="tensorGradient"></pre>

For `[1, 2, 3]` the energy is fourteen and the gradient is `[2, 4, 6]`: increasing a
coordinate $x_i$ changes its contribution $x_i^2$ at rate $2x_i$. Since the function takes a
tensor and returns one number, its forward result has scalar shape `[]`, while the gradient has
the input's shape `[3]`.

Each multiplication uses the input twice. Backward accumulates a contribution from both
uses, and then the sum passes its output sensitivity to every coordinate. Larger models reuse these
same local derivative rules whenever they call the operations.

By default the backward seed differentiates the sum of the output entries (for a scalar
energy, that means a seed of one). For a tensor-valued function, `result.backward seed` lets us say
how much each output contributes, which in calculus terms is a vector–Jacobian product (VJP):
we push those output sensitivities back to the input without ever materializing a full Jacobian.

On GPU, LibTorch supplies supported native tensor values and local gradient operations
(with its own autograd engine disabled), and configured binary arithmetic keeps its complete
format on the encoded tape, so custom functions record through the same differentiation
system we've already been using for the model examples.

Conditionals need care here. Recording follows the branch that was taken, so an unused branch
can't add an invalid division to that execution, and the derivative we get describes that path.
Right at a branch boundary there may be no mathematical derivative at all. Scalar branch
decisions on GPU also synchronize back to Lean, while the supported arithmetic and backward
operations stay on the device.

Backward consumes its recording. If you only need the forward value, `result.close` releases
the saved intermediates, which starts to matter once a calculation keeps a large
tensor at every step.

<div id="write-a-loss-and-take-a-gradient-step"></div>

Now let's use a function of our own as a training objective, comparing three predictions
with three target values and summing their squared residuals:

<pre data-lean-example="tensorLoss"></pre>

The residuals are `[1, 1, 2]`, so the loss is six and the gradient is `[2, 2, 4]`.
Subtract a quarter of that gradient and we get `[0.5, 1.5, 2]`, whose residuals are
`[0.5, 0.5, 1]`, so the loss falls to one and a half. (Every number in this example is
exactly representable in binary32, so you can check them all by hand.)

I'm optimizing the predictions directly here so the update is easy to follow. In a
network those predictions depend on parameters through earlier layers, and their gradient
would have to keep going back through those layers before the optimizer updated the parameters.
In this function the targets are captured constants, so the call differentiates the
predictions we supplied and leaves the targets alone.

Below the run there's a real derivative proof for one loss coordinate: for a fixed target $t$,
the derivative of $(x-t)^2$ is $2(x-t)$, checked with the existing `autograd` tactic.
On the tensor side, recording also uses the existing operation rules for subtraction,
multiplication, and summation. I didn't need to add a loss-specific tape or a custom native operator.

For this objective, a learning rate $\eta$ turns each real residual $e$ into
$(1-2\eta)e$, so the new loss is $(1-2\eta)^2$ times the old one. At
$\eta=1/4$ that's a quarter of the original, exactly as the small run shows, and in exact
arithmetic any rate between zero and one shrinks a nonzero loss while a rate greater than one
makes it grow. (That's about independently adjustable predictions and this quadratic
objective, and it makes no convergence claim about arbitrary neural-network training.)

A new objective is a new function to inspect and a new derivative question, but the runtime
doesn't change, so we can keep using it to record the gradient, run an update, and then prove
the mathematical rule behind that calculation.

## Writing contractions with `einsum`

Let's move from functions on individual entries to calculations that combine them.
A linear layer takes several input features, multiplies them by weights and adds up the
products, and attention does that twice (first to compare queries with keys, then to combine
values using the attention weights). We could write an indexing loop for each calculation,
but I'd rather write down which axes meet and which axes survive :)

That's what `einsum` lets us do. Its name comes from Einstein summation notation, although
our pattern spells out the output axes instead of making us infer them from a convention.
For a matrix product, we give the first tensor a row axis and a feature axis, and the second
a feature axis and an output axis. Row and output survive, and we multiply and sum over feature:

<div class="math-display" data-tex="Y_{ro}=\sum_f X_{rf}W_{fo}.">Each output combines one input row with one weight column.</div>

Here's that calculation on two small matrices, plus one that uses the input twice to get
the squared length of each row:

<pre data-lean-example="tensorContraction"></pre>

Read the pattern from left to right. A comma separates the operands, reusing `feature`
connects their feature coordinates, and the arrow lists the axes of the result. Since
`feature` doesn't appear after the arrow, the calculation sums it out, while `row` and `output`
stay (in that order). From the input shapes `[2, 3]` and `[3, 2]`, Lean infers a `[2, 2]` result.

The first output is $1\times1+2\times3+3\times5=22$. Next comes the same input row
with the second weight column, giving $1\times2+2\times4+3\times6=28$, and the second input
row gives 49 and 64. You can follow all four entries without looking inside the generated
loop (and these small integer results are exactly representable in binary32).

In the second pattern, `row feature, row feature -> row`, we keep `row` in both operands and
in the output. It combines matching entries within each row and then sums their products
over `feature`, giving $1^2+2^2+3^2=14$ and $4^2+5^2+6^2=77$.
So does repeating an axis name across operands mean we sum it? Not by itself. What decides that
is whether the axis appears in the output, which is also how we keep a batch or attention-head
axis while contracting another one.

I like having those axis names in the expression. We can read `feature output` as a weight
matrix mapping incoming features to outputs, and see the same feature name in the input.
Changing the number of rows or outputs changes the tensor dimensions while the pattern means
the same thing, and the elaborator checks that the dimensions are compatible when it constructs
the calculation. A shared non-singleton axis with incompatible lengths gets rejected before
evaluation. Singleton broadcasting is supported, though, so matching axes don't always need
identical physical lengths.

Those names belong to the pattern and aren't permanent semantic labels stored in the
tensor type. If two dimensions happen to be equal and you swap their intended roles, you can still
get a well-shaped but wrong calculation. Shape checking catches incompatible sizes.
Catching the wrong equation takes a theorem relating the result to the one we intended.

### Moving axes, broadcasting, and reducing

Once a contraction is written we often need to arrange its input or output for the next
operation, so let's take the same two-by-three matrix and transpose it, flatten it, add a bias
to every row, and sum each row:

<pre data-lean-example="tensorPatterns"></pre>

`rearrange matrix "row column -> column row"` moves coordinates around. Its first column
`[1, 4]` becomes the first row, so the result is `[[1, 4], [2, 5], [3, 6]]`.
If you merely regrouped the unchanged flat payload into three rows of two, you'd get
`[[1, 2], [3, 4], [5, 6]]` instead, which has the same shape `[3, 2]` but isn't a transpose at all!
So the pattern pins down which coordinate goes where, as well as the output dimensions.

Parentheses combine axes. `row column -> (row column)` flattens the matrix into
`[1, 2, 3, 4, 5, 6]`, keeping that row-then-column order, whereas flattening the transposed
tensor would give `[1, 4, 2, 5, 3, 6]`. That matters as soon as a projection expects features
in a particular order. Two layouts with equal element counts aren't interchangeable.

For the bias (shape `[3]`), `expand` adds a row axis of length two so the same
three values are available at both rows before the addition, and that gives
`[[11, 22, 33], [14, 25, 36]]`. I picked the new axis and its size myself, so you don't have
to guess which broadcasting convention applies. Finally `reduce`
sums away the column axis and returns `[6, 15]`.

All of these take tensors and return tensors, because I don't want you
to learn another numerical container just to rearrange a matrix. A pattern says how coordinates
are used and the resulting type records the shape, but reductions add one more thing to care
about, the arithmetic order, since a real-valued sum and a rounded sum need different reasoning.

### The two contractions inside attention

Now let's keep more axes. Suppose queries and keys each have shape
`[batch, head, token, feature]`, and every head compares each query position with each key
position, summing over features while keeping the batch and head separate:

<div class="math-display" data-tex="S_{bhqk}=\sum_f Q_{bhqf}K_{bhkf},\qquad O_{bhqc}=\sum_k A_{bhqk}V_{bhkc}.">Queries and keys produce scores; attention weights and values produce outputs.</div>

From the first equation we get raw dot-product scores, which a complete attention layer would then
scale, mask and normalize before the second equation combines values with those normalized
weights. To keep both contractions easy to inspect, I'll compute the raw scores and then hand
the value calculation its own small set of weights. (I picked those weights by hand. They
aren't the softmax of the displayed scores.)

<pre data-lean-example="tensorAttention"></pre>

There's one batch and there are two heads, each with two query positions, two key positions
and two features. In the first head the query and key vectors are the coordinate
vectors, so its score matrix is `[[1, 0], [0, 1]]`. In the second head the keys are the same
coordinate vectors, but the queries are `[1, 1]` and `[2, 1]`, giving `[[1, 1], [2, 1]]`.
Every head's calculation stays separate. Nothing in the pattern ever forms a dot product between
a query from the first head and a key from the second.

For the value calculation, the first head reads 10 and 20. Its first query puts three
quarters of the weight on 10 and one quarter on 20, which gives 12.5, and its second query
reverses those weights to get 17.5. Over in the other head the values are 30 and 50: equal weights
give 40, and selecting the first value gives 30. That's one output per query and head.

A final `rearrange` moves the head axis after query and combines head with channel, so the
first query's output row is `[12.5, 40]` and the second's is `[17.5, 30]`. Had we kept the old
flat order and only changed the shape, the values would have been grouped differently, so it's
the pattern that says which head outputs sit beside one another before the next projection uses them.

A matrix product, a row norm and attention are different patterns on the same
contraction machinery, and I like being able to change the axes in the expression without
writing another model-specific helper. These examples run as CPU tensor calculations, though.
An arbitrary `einsum` expression doesn't automatically get a GPU implementation or
a backward rule merely because its pattern was accepted (recorded model contractions use
their supported runtime operations).

`einsum` is a term elaborator. When Lean elaborates the definition it reads the literal pattern,
checks it against the tensor shapes, and constructs the computation along with its certificates,
so we aren't parsing those axis names again for every output entry at runtime. Proof tactics
such as `einops` have a separate job (helping us prove statements about tensor transformations).

There's a compiler connection worth looking at too. On the executable side, the `einsum` path can read
its operands directly and accumulate each output without ever constructing a full tensor of
intermediate products. Generated code carries certificates identifying its indexing
and ordered arithmetic with the reference evaluator, and the
[lowering theorem](https://github.com/lean-dojo/TorchLean/blob/main/NN/Tensor/Internal/Lowering/Einsum.lean)
then connects that evaluator to the independent contraction definition. Its general
ordered-fold theorem only needs addition and multiplication operations, with no claim that
they obey exact-real algebraic laws, and since reordering a contraction does require more laws, the
planner only goes that way when the scalar instances supply them.

That's why we can reuse the contraction interface with native floats or a configured
FloatLib scalar while its numerical interpretation stays visible. A proof about the
ordered evaluator preserves that particular calculation. It doesn't make every grouping
of the floating-point sum equal, and it doesn't prove that some external GPU kernel followed that order.
If you want more patterns, and the implementation choices behind them, they're in the
[tensor chapter](https://lean-dojo.github.io/TorchLean/blueprint/Building-Models/Tensors-That-Remember-Their-Shapes/).

### What fusion saves

For the matrix product above, a literal implementation could first build every product
$X_{rf}W_{fo}$ and then reduce the feature axis. How big is that temporary? It has three
axes (row, feature and output), so with 256 rows, 1024 features and 256 outputs it would hold 67,108,864
binary32 values, or 256 MiB of scalar payload, while the final `[256, 256]` result needs only
256 KiB. Those are element counts rather than allocator measurements. Still, you can see why I'd
rather not build one.

Our checked lowering reads the operands and accumulates each output directly. It compiles
the named axes into a stride plan, so a read becomes some coordinate arithmetic followed by
an access to the operand's storage, and singleton broadcasting and diagonal selection are
part of that same plan. Nothing materializes a broadcast tensor or a tensor of all the
products. For the proof, the reference product tensor is still there in the mathematical
semantics.

`einsumTensorKernel_correct` is the theorem that connects the generated result to that reference.
It needs storage, addition, multiplication, zero and one, and it doesn't require floating
addition to be associative, because the operand products and contracted additions keep their
specified order. That's especially useful for FloatLib scalars, where changing the order
can change the answer. A faster regrouping needs the right algebraic law or a
numerical comparison (an assumed real-number identity isn't enough).

Avoiding the temporary doesn't turn this into a tuned GPU matrix library. These pattern
examples use the compiled Lean tensor path, while GPU model matrix products go through the LibTorch
backend. I want both routes to have clear semantics, but their schedules and performance
have to be measured separately.

## From tensor expressions to a model

Contractions make the axes visible, and when I'm building a network I also want
the layer composition to be easy to read. Here's the public API used by the
[MLP quickstart](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean):

<pre data-lean-example="tensorModel"></pre>

This model accepts two features, expands to eight, applies ReLU, and returns one value, and
Lean checks the shapes wherever the layers meet. Objective, optimizer and execution settings
come from the trainer (which doesn't change the model definition). After that the quickstart
gives it tensor data and runs the training loop, and if you want the supported native GPU path,
you select `gpu` in the settings.

Keeping these apart also matters when we choose precision. Builders come with seeded
initializers, and the typed state records the scalar type of the parameters and buffers.
So if extra digits matter, you can construct that state directly from exact rationals instead
of widening values an initializer already rounded. That's exactly what the binary128
affine-training example linked below does.

Larger examples use the same ideas. A transformer composes projections, attention,
normalization, residuals and feed-forward layers, a Fourier neural operator adds spectral
transforms and frequency mixing, and a diffusion model repeats a learned denoising update.
Their state and training objectives differ, but they all use the shared tensor,
layer, and differentiation APIs. You'll find those calculations in the
[model examples](https://lean-dojo.github.io/TorchLean/examples/), alongside convolutional,
recurrent, reinforcement-learning, and self-supervised models.

## Where the time goes

I don't want “runs on GPU” to stand in for a speed result. What are we actually paying for?
There are several costs we can inspect: generating and compiling a custom body, launching operations, moving values,
saving intermediates, and doing the arithmetic itself.

For a forward-only custom recurrence, the frontend can generate one ordered loop. With
gradient recording, the existing tape records the steps and retains what backward needs,
and on GPU that recorded path launches operations per step. Both execute the recurrence,
but their launch count and memory use can be very different. That means the faster
route for one forward call isn't automatically the route we can use for training.

The runtime also caches identical generated CUDA source. A first call can include NVRTC
compilation while a warm call can reuse the compiled program, and since capturing a different
scalar constant can produce different source, a benchmark should say whether it's timing
compilation, warm execution, or both. Asking for a scalar branch decision on GPU can also
synchronize with Lean, which is a cost to think about if you put the branch inside a long loop.

For standard model operations we moved to LibTorch, because maintaining our own numerical
kernels duplicated infrastructure that PyTorch already provides. TorchLean still owns the
model and the differentiation traversal, while ATen (PyTorch's C++ tensor library) computes the
supported tensor values and local gradients with LibTorch autograd disabled. Matrix backward
can pass transpose flags to the native matrix operation instead of first creating transposed
copies. Those are implementation choices. They aren't a promise that every model is now faster than PyTorch.

I do have a retained end-to-end comparison from my
[GPT training experiment](../week-03-gpt-training/#matching-50-updates-against-pytorch):

| Archived FP32 run on one A100 | TorchLean | PyTorch |
| --- | ---: | ---: |
| Median scheduled tokens per second | 5,596 | 19,207 |
| Final validation loss after 50 matched updates | 2.431321 | 2.431322932 |

Exact numbers are in the [comparison record](../week-03-gpt-training/results/matched-runtime-50-step/comparison.json).
PyTorch was about 3.43 times faster on that setup, and the
largest training-loss difference was about 0.00001193. Those measurements used the earlier
FP32 implementation, so treat them as a baseline for a new comparison (they aren't timings for
the current LibTorch adapter or the custom compiler).

In the same article, the
[cached-generation benchmark](../week-03-gpt-training/#cached-generation-and-its-correctness-proof)
asks a different performance question. In that archived benchmark (one
checkpoint, one prompt, an eleven-token continuation), three warm runs took about 168–172 ms
with the cache versus 9.43–9.74 s when recomputing the full prefix, and the roughly 56-fold
difference comes from reusing earlier work on that workload. On the current
LibTorch adapter in binary64, with a one-update GPT-2-small checkpoint, 16 tokens took
383–416 ms with the cache and 25.5–26.6 s without it, about 64–67 times faster! Its cache
theorems tell us what the stored keys and values mean, and the timings show what that reuse
saved in those runs. I'd like to keep making that kind of comparison as we improve the
runtime, with the configurations and results beside the code.

<div id="when-an-output-reads-more-than-one-input"></div>

## Indexed reads and row sums

Our square read one entry per output, but a row sum reads several entries and combines
them. Let's make that access pattern explicit:

<pre data-lean-example="tensorRows"></pre>

This gives `[6, 15]` again, as the earlier `reduce` did, with the first row reading 1, 2, 3
and the second reading 4, 5, 6.
Rows can run independently, but the additions within a row follow the order we wrote.

Inside the body I pick a row-major address, `row * width + column`, and add that value to
the accumulator. `read 0` names the first input tensor, `Arguments.empty.push matrix` supplies
it, and output shape `[2]` asks for two row results. If you read out of range you get
an error, since the output shape alone can't establish that every address in the body is valid.

For a plain row sum I'd use the `reduce` pattern above. This indexed interface
is for when we need a particular sequence of reads or an unusual accumulation, and the
result is still a tensor (no new collection type for readers to learn).

This is also where numerical semantics start to affect the compiler. Over real numbers several
addition trees compute the same sum, but in floating point they can differ, and in my
[first essay](../week-01-batch-invariant-inference/#why-moving-the-parentheses-changes-the-sum) there's a
three-number example in which moving the parentheses changes one into zero!
So the row-sum lowering keeps its additions in order and never assumes associativity.

Before we can differentiate or certify an indexed program, though, we need more information.
Compiling its reads and arithmetic proves a source correspondence, but it doesn't automatically
give me the derivative of an arbitrary body or an enclosure over an input region.
Recording worked in the examples above because their operations already have supported local rules.

<div id="keep-the-digits-we-chose"></div>

## Beyond hardware precision

GPU hardware comes with a fixed collection of number formats. What if we want more digits than
that? A mathematical definition doesn't have to stay inside that collection, but something has to
implement the arithmetic for those extra digits.

FloatLib lets us configure the representation a tensor uses. Here I'll pick binary128,
which has fifteen exponent bits and 112 fraction bits, and choose
$a=1+2^{-100}$, whose extra term is too small to survive being stored in binary64.

<pre data-lean-example="tensorPrecision"></pre>

Building $a$ from a rational keeps that digit before any execution begins. If you first
converted the value to binary64 and then widened it you'd get one, because a wider destination can't
recover information the earlier conversion lost. The exact square is $1+2^{-99}+2^{-200}$,
and binary128 nearest-even rounding keeps
$1+2^{-99}$ while losing the much smaller final term. In the example I compare the CPU result's
exact rational value with that rounded answer, so `true` means the retained digit is there.
(Printing it through a binary64 conversion would be a poor way to check.)

On the custom GPU path, wide values travel as complete integer words, and software arithmetic
on the device works on those words without narrowing them to a native float.
We keep the precision we chose and pay for it with more instructions than a hardware multiply.
(This example is about the arithmetic result. I didn't measure its throughput.)

Standard configured binary formats can use compatible hardware operations, depending on the
complete format, the operation and the device capability, and when those don't match, the
configured backend keeps its software path. Separate products and sums also stay separate,
so the generated arithmetic has to preserve that choice and can't silently replace the
pair with one fused multiply-add.

<details markdown="1">
<summary>Formats and execution limits</summary>

On the configured GPU backend we support up to thirty exponent bits and 4096 fraction bits,
with nearest-even rounding and gradual underflow. On compatible GPUs, binary16 and bfloat16 use
native addition, subtraction and multiplication but keep their division in software, while
configured IEEE binary32 and binary64 can use native division too.

Ask for GPU with a decimal, a posit or another CPU-only carrier and you get a reported CPU
fallback that preserves the type and precision. An unsupported body for a GPU-capable type,
an unavailable GPU, or a failed compilation still reports an error.

Configured GPU recording supports the arithmetic and structural operations these examples use
(not every native model operator). Meanwhile the separate canonical `Graph.runBuffers` runner
still requires binary32 inputs and payloads, and choosing `Wide` in a custom function doesn't
change that runner's contract.

A real-valued definition is useful for a theorem, but arbitrary mathematical reals aren't
something a device can compute with, so runnable calculations use executable carriers instead (native
floats, FloatLib formats, or exact rationals where the required operations support them).

</details>

<div id="the-proofs-belong-beside-the-program"></div>

## Proving the function and its lowering

Once we've written a calculation we can ask a mathematical question about it, and for the square
we can prove the real derivative using the existing calculus tactic:

<pre data-lean-example="tensorProof"></pre>

Here the `autograd` tactic uses derivative rules to construct a proof, Lean's kernel checks it,
and the conclusion holds for every real input. A rounded floating-point function isn't a smooth
real function, though. What connects its values to the real calculation is a numerical error bound.

For compilation we need to show that the generated loop returns the result of the original
recurrence, including what happens when an input read fails, and once we name intermediate
results, later instructions have to read the intended value (rather than another value with
the same type).

<figure class="evidence-figure">
  <div><strong>The function we wrote</strong><span>Tensor reads, ordered arithmetic, branches, recurrence equations</span></div>
  <div><strong>The lowered program</strong><span>Expression evaluation, loops, and named intermediate values</span></div>
  <div><strong>The running device</strong><span>CUDA compilation, input buffers, and numerical instructions</span></div>
  <figcaption>We prove source correspondence and lowering in Lean. Carrying those results over to
    physical execution also needs the external arithmetic, memory, and compiler contracts.</figcaption>
</figure>

`Program.eval_eq_reference` identifies the checked evaluator with the source calculation,
and `Program.lower_correct` carries that result into the structured loop representation
(their types appear with the row-sum example above). As for the native FP32/FP64 emitted-source
theorem, it has explicit arithmetic and input-buffer contracts and doesn't verify NVRTC or the GPU.
Configured-format CUDA arithmetic is checked against FloatLib in runtime tests, and it
sits outside the kernel-verified part.

Keeping these apart helps me decide what to prove next. A function can have correct lowering
and still lack a numerical error theorem, a derivative rule can be correct over the reals
while the computed gradient rounds, and a tensor can have the right shape while its values are
wrong. Each of those properties has its own statement we can inspect and improve.

For the examples on this page, Lean checks the displayed code and records the CPU outputs.
We typecheck the GPU calls, and the scalar examples have accepted frontend compilation
results, but this article doesn't present a fresh GPU run. If you want to run them on a GPU
yourself, the
[custom computation guide](https://lean-dojo.github.io/TorchLean/examples/custom-computations/)
has the device setup and the full API.

<div id="what-we-gain-by-keeping-it-in-lean"></div>

The relaxation example is my favorite on this page, because it connects three things that are
often maintained separately. Its
definition takes the operations it needs from an element type, so we instantiate it with
`Float32` to run a tensor computation and with real numbers to prove the closed form, and on GPU
the same `Float32` call gets a lowered body with a source-correspondence proof. All of those
connections sit in one development, where we can inspect them.

The square, blend, and loss are familiar calculations. What's new is connecting their
execution to statements about their meaning. Our relaxation theorem covers every real input
and step count (rather than only the starting values displayed above), and the compiler theorem preserves the specified
read and arithmetic order. When a real-valued equality isn't the right description of a rounded
result, the numerical definitions let us describe the rounding.

[PyTorch's autograd](https://docs.pytorch.org/docs/2.14/notes/autograd.html) records tensor
operations for differentiation, and
[`torch.compile`](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html) provides
a compiler interface for Python/PyTorch computations. We reuse LibTorch's numerical
infrastructure because those execution capabilities are valuable, and what we add here is Lean
statements about the source functions and transformations, reusing mathlib's calculus and
algebra so there's no second informal argument to maintain beside the code.

Formal numerical reasoning exists elsewhere too.
[Flocq](https://flocq.gitlabpages.inria.fr/) provides multi-radix, multi-precision
floating-point theory and computations in Rocq, and other languages and proof assistants can
support related workflows. In this library we've brought the tensor API, differentiation
runtime, compiler proofs, mathlib, and FloatLib together, and the examples show how those pieces
fit, which isn't the same as showing that other languages couldn't express these programs.

<div id="more-of-torchlean-id-like-to-show"></div>

## Other features we've been working on

I started with a square because we can understand its value and derivative immediately.
We've also been working on the calculations around it (arranging axes, composing layers,
taking derivatives, choosing the arithmetic), and the contraction examples above are one
part of that work. I'd like the model and proof APIs to be as convenient to use as those
patterns.

Our differentiation work goes beyond one backward pass, too. Nested dual numbers represent
higher-order directional derivatives, and the operation theorems connect them to mathlib's
`iteratedFDeriv` at every finite order. We can combine supported arithmetic and smooth
functions (including local rules for division, logarithms and square roots when their
domain conditions hold), and a new named function can expose its definition or register a
proved rule for the `autograd` tactic to reuse. I like this because the proof can explain
why a derivative calculation works, assumptions and all, instead of leaving me with
only a printed Hessian. You can see the scalar and tensor statements in the
[tactic examples](https://github.com/lean-dojo/TorchLean/blob/main/NN/Tactic/README.md),
and the
[scientific-model example](https://lean-dojo.github.io/TorchLean/examples/scientific-ml/)
uses differentiation in models for numerical problems.

### Differentiating through a differential equation

For a concrete use of the higher-derivative API, take the physics-informed neural network (PINN)
example. We ask a smooth network to
represent a function $u_\theta(x)$ satisfying $u''(x)=-2$ on $[-1,1]$ with zero values at
both endpoints, and its loss penalizes the equation residual at interior points along with the
boundary values. It never gets training samples of the known solution $1-x^2$.

An input direction has the same shape as an input tensor. Stack that direction twice and
`autograd.model.derivative` gives a second directional derivative, while two different
directions give a mixed derivative. So the order is the length of that leading axis,
and there's no separate function named after every derivative order.

Training on the residual also needs a parameter derivative. Since the second input derivative
still depends on the network's weights, `autograd.model.derivativeVjp` pulls an output
cotangent (the supplied output sensitivities) through that calculation to the state.
Under the hood it nests dual scalars
around the existing model evaluator and reverse pass, which means we aren't approximating $u''$ by
subtracting nearby predictions.

We use tanh in the example because the equation needs a smooth field, and ReLU's branch derivative
doesn't give a classical second derivative at its kink. More generally, the real
higher-derivative theorems keep their smoothness and domain hypotheses. A low sampled
residual is useful training evidence. A guarantee over the whole interval is a different
thing, and it needs a uniform residual bound and a theorem connecting it to solution error.

For a small scalar proof, the existing tactic can check a proposed derivative:

<pre data-lean-example="tensorChainRule"></pre>

That's the chain rule for an exponential composed with a square, and we can inspect the
mathlib derivative statement independently of any floating-point gradient run. For a new
named function, unfolding its definition or registering a proved rule is enough for the same
tactic to work with it (no new differentiation language needed).

A different objective shows up in the
[Burgers FNO example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Operators/Fno1dBurgers.lean),
which learns a map from an initial field to a later solution, with Fourier transforms and
frequency mixing as layers in that model. Both examples use
the tensor and training APIs. But what would we prove about a trained PDE
surrogate? That depends on the equation, the boundary conditions and the error notion we choose.

### Keeping precision through training and saving

Earlier, the binary128 square only followed one value, while typed training keeps the selected
scalar in the samples, model state, predictions, gradients and losses.
`trainer.openTyped` opens that CPU session, and its checkpoint encoding preserves the
scalar words through saving and loading. There's more on these operations and the conversion
boundaries in the
[precision API](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Precision.lean).

There are still choices to make at the beginning of the calculation, because widening seeded
`Float` parameters can't create random bits that weren't there, and the trainer's optimizer
settings also start as `Float` values before conversion. If the learning rate itself
needs extra precision, the typed `nn.sgdStep` call accepts it in the selected scalar type.
In the [affine example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/TypedTraining.lean)
we call `nn.sgdStep` directly and build the parameters from $1+2^{-100}$ (a rational that
binary64 can't hold). Saving all the resulting
words only helps if we kept those digits on the way into the computation.

Complex values use the same idea with two components. `Complex α` stores the real and
imaginary parts in `α`, and a tensor of complex values keeps its shape as usual. In the
[complex regression example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Operators/ComplexRegression.lean)
we minimize a real squared-residual loss and differentiate both component
coordinates with the forward-mode reference method, which is useful for complex parameters
and never identifies a real loss gradient with a holomorphic complex derivative.

### Starting from JAX or PyTorch

Sometimes the model you want to inspect already exists in Python. In the
[Jaxpr example](https://lean-dojo.github.io/TorchLean/examples/jaxpr/),
supported operations get captured into TorchLean's tensor intermediate representation (IR).
(Jaxpr is JAX's traced representation of the calculation, which isn't the Python source itself.)
The importer checks shapes and precision
metadata and then gives that graph a Lean semantics, and PyTorch has a supported graph-import
path as well. We're still extending the coverage (these adapters aren't translators for
arbitrary Python programs).

A small JAX example captures `sum(x * x)`. Its theorem first identifies the graph's
real-valued denotation with the tensor sum of squares and then proves that permuting a vector
preserves the graph's answer, for every admitted vector length and permutation (rather than only
the input we used to compare the JAX and Lean runs). Floating-point permutation invariance
is another question, since reordering a reduction can change its rounded answer, as we saw
in the first essay.

This is the direction I'd like to develop further: keep the frontend someone already uses,
bring its supported computation into a common representation, and work on a theorem about
that graph. The importer, the numerical execution and the graph property each need their own
evidence. A successful sample comparison is a good start, but it doesn't prove the
translation correct for every input.

### Bounds and model verification

On the proof side, the updated FloatLib `interval` tactic handles more real expressions
(including positive-base powers, logarithms to a chosen base, and inverse hyperbolic functions),
and we can register another function's enclosure together with its containment proof and reuse
that scalar theory when bounding tensor calculations. That matters whenever an equality
over real numbers isn't enough, because a rounding bound or an enclosure can describe the
behavior we need to protect. (Those proofs stay distinct from assumptions about external GPU
execution.) The
[floating-point chapter](https://lean-dojo.github.io/TorchLean/blueprint/Floating-Point-and-Native-Boundaries/Floating-Point-Semantics/)
has examples of the arithmetic and interval proofs.

For classifiers, the
[verification API](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Verification.lean)
lets us check a trained model over an input region by asking whether a label survives the
allowed perturbations. That guarantee comes from a supported certificate and its soundness
hypotheses. Switching on gradients alone won't give you one. For a ReLU classifier we might want a
fixed label on an input box, and for a smooth neural field we might want a residual bound,
which are different statements even when both start from tensors and the same model APIs.

Convergence proofs need a statement about the update too. For that, the `converges` tactic can
apply registered contraction and gradient-descent results (keeping their step-size,
smoothness, and other hypotheses), and it won't infer convergence from a loss curve that
happens to go down. I want to keep these tools close to the runnable examples so we can work
on whatever property a calculation needs, with mathlib's definitions right underneath.

## Conclusion

Try changing the update in the relaxation example, or the objective in the custom loss.
Nothing changes in the execution API, but the closed-form and derivative proofs need to match your
new function. Start somewhere away from the target and try a rate of one half, then two,
and the real recurrence switches from approaching the target to oscillating around it. With the
theorem and the run side by side, we can ask which behavior we meant to implement :)

## References

- TorchLean. [Custom tensor computations](https://lean-dojo.github.io/TorchLean/examples/custom-computations/) and [compiler source](https://github.com/lean-dojo/TorchLean/tree/main/NN/Kernel). The execution API and lowering theorems used here.
- FloatLib. [Arithmetic and proofs](https://lean-dojo.github.io/FloatLib/). Configured formats and numerical theory.
- David Thrane Christiansen. [Functional Programming in Lean](https://lean-lang.org/functional_programming_in_lean/introduction.html). Lean's programming and proof facilities.
- Accelerate. [The Accelerate language](https://www.acceleratehs.org/documentation/users-guide/language.html). Embedded array computations.
- Troels Henriksen and colleagues. [Futhark: Purely Functional GPU-Programming with Nested Parallelism and In-Place Array Updates](https://futhark-lang.org/publications/pldi17.pdf), 2017.
- Adam Paszke and colleagues. [Getting to the Point: Index Sets and Parallelism-Preserving Autodiff for Pointful Array Programming](https://arxiv.org/abs/2104.05372), 2021. Dex.
- Higher Order Company. [Bend](https://github.com/bendlang/bend) and its [language guide](https://github.com/bendlang/bend/blob/main/guide/GUIDE.md). Dependent types, ownership, and parallel calls.
- NVIDIA. [NVRTC documentation](https://docs.nvidia.com/cuda/nvrtc/index.html).
- PyTorch. [Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html) and [`torch.compile`](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html).
- Sylvie Boldo and Guillaume Melquiond. [Flocq](https://flocq.gitlabpages.inria.fr/). Formal floating-point arithmetic in Rocq.
