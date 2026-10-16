# Checking a small transformer in Lean

[Neel Somani's verifiable transformer](https://github.com/neelsomani/verifiable-transformers)
has a task I really like: match a quote or close a bracket. There are only 256 possible
prompts. So we can examine every one and check which delimiter the model prefers. Neel
designs the calculations as circuits for a satisfiability modulo theories (SMT) solver,
a tool that reasons about logical and arithmetic constraints. I trained his model with his own
training code and wanted to bring the checkpoint and its evidence into Lean.

I imported the checkpoint and its exported scores, then added a forward pass that uses those
weights, so now we can compare the saved decisions with the ones we recompute in Lean. I also
check that the export contains the right prompts. Repeating an easy case in place of a difficult
one shouldn't count as checking the whole task.

There are two checks to follow here. Lean's kernel verifies that the exported integer-score
records cover the task and give every intended candidate a positive margin, and separately, a
compiled Lean program recomputes the floating-point forward pass from the imported weights.
Circuit queries are the upstream solver's job (I ran it on this checkpoint). My trace proofs
don't certify that solver, and I haven't proved the rounded forward pass correct either, so
we'll keep those results apart as we go.

It's a small model. It has two transformer layers, one attention head, hidden width 16, a
32-token vocabulary, and an input six tokens long. For each prompt the claim is about a choice
between two candidates, and it says the intended quote or bracket must score higher than the
other one. That pairwise comparison (not a ranking of the intended token against the other 30
vocabulary entries) is the property I check throughout the experiment.

I think that small input domain is what makes the example useful. With a large language model,
even short prompts give you far too many possibilities to enumerate, but here we fix the length,
the content vocabulary, and the delimiter positions, and then we check every combination. So our
result is about that defined task. Longer prompts or nested brackets would give us a new task to
check.

We'll follow one prompt through the task definition, the exported record, and the forward
calculation. Along the way I'll explain why the model uses different attention and
normalization operations from GPT-2 (and why checking a saved score is different from
recomputing it from the weights).

<div id="enumerating-all-256-prompts"></div>

## A task with 256 inputs

A quote prompt has the form `[BOS, quote-task, x₁, quote, x₂, x₃]`, where `BOS` is the
beginning-of-sequence token and `quote-task` selects the task. Each content token
<span data-tex="x_i">xᵢ</span> is one of IDs 5, 6, 7, or 8. A quote is ID 9 or 10, and at
the prediction position the model should prefer that same quote. Bracket prompts have the same
layout with a bracket-task token and an opening bracket, and opening IDs 11 and 12 should lead
to closing IDs 13 and 14 respectively.

Three content positions with four choices and one delimiter with two choices give 128 prompts
per task:

<div class="math-display" data-tex="4^3 \times 2 = 128 \qquad 128 + 128 = 256">128 inputs for each task, 256 in total.</div>

I'll use the quote prompt `[1, 2, 5, 9, 5, 5]` as the running example (these IDs belong to this
synthetic vocabulary, not to a natural language tokenizer). Token 1 starts the sequence, token 2
selects the quote task, the three 5s sit in the content positions, and the delimiter is 9, so
the intended prediction is 9 and the alternative is 10 (change only the delimiter to 10 and
those roles swap). For the bracket task, `[1, 3, 5, 11, 5, 5]` instead asks for closing token
13, compared with token 14.

Why not test a few representative prompts? Content positions vary independently. So we can't
replace their 64 combinations with four examples in which all three content tokens happen to
agree, and we can't check one delimiter and assume the other behaves symmetrically either,
because the learned embeddings and output weights are allowed to treat their IDs differently. I
enumerate every combination, take two coordinates of each prompt's 32 raw scores (the logits),
and ask for a strict inequality:

<div class="math-display" data-tex="z_{\mathrm{target}}(x) > z_{\mathrm{alternate}}(x)">The intended candidate must have the larger score.</div>

We don't need softmax on those final scores to compare the candidates, because softmax
preserves their order over real numbers, and we don't need the two probabilities to sum to one
when viewed apart from the other 30 tokens either. If a third token has a higher score than
either candidate, the pairwise property can still pass, which is why an unrestricted
greedy-generation claim would need a stronger check than the one I use here.

You can see the difference in the
[evaluation saved by training](artifacts/upstream-small-gpt/metrics.json), where the bracket
task has candidate accuracy 1.0 but full-vocabulary accuracy 0.5. In other words, 100% on the
two-candidate task and 50% when choosing from the full vocabulary. On the 64 prompts whose
opening bracket is `{`, the closing `}` wins against the other closing bracket, while an
unrestricted choice among all 32 tokens picks the double quote, token 10, by 0.026 (the
[upstream evaluation code](https://github.com/neelsomani/verifiable-transformers/blob/8e3c54bad2494c17c91109fdb58cf9e2f1c98efc/scripts/small/train.py#L490)
computes both decisions separately). Given the whole vocabulary, the model would close a curly
brace with a double quote! If you restrict a decoder to the two task candidates, it relies on
the property I'm checking, but a general text generator needs the stronger vocabulary-wide
property.

<figure class="interactive" id="prompt-example">
  <div class="prompt-controls">
    <label>Task<select><option value="2">Match a quote</option><option value="3">Close a bracket</option></select></label>
    <label>Delimiter<select><option value="0">First choice</option><option value="1">Second choice</option></select></label>
    <label>Content 1<select><option>5</option><option>6</option><option>7</option><option>8</option></select></label>
    <label>Content 2<select><option>5</option><option>6</option><option>7</option><option>8</option></select></label>
    <label>Content 3<select><option>5</option><option>6</option><option>7</option><option>8</option></select></label>
  </div>
  <output aria-live="polite"></output>
  <figcaption>Pick any of the 256 inputs to see its token IDs, its target, and the alternative.
    The figure only enumerates the prompt definition, and the forward pass itself runs in the Lean replay below.</figcaption>
</figure>

<div id="why-this-transformer-uses-different-operators"></div>
<div id="what-a-circuit-check-asks"></div>

## Circuits, sparse attention, and normalization

[Neel describes the design](https://github.com/neelsomani/verifiable-transformers) as one
whose:

> forward pass is built from SMT-representable primitives

That choice reaches into the model itself. We have to follow the sparse attention,
normalization, and activation he used instead of substituting the operators from a standard
GPT implementation, so let's look at those choices before we interpret any solver result.

What counts as a circuit here? It's a selected computation inside the model (the retained
dependencies and operations used for a task decision). It isn't a physical logic chip. Neel's
[formal definitions](https://github.com/neelsomani/verifiable-transformers#formal-definitions)
keep agreement with the reference program apart from two other questions: whether the retained
connections are necessary and whether irrelevant content can change the decision.

For our pairwise task, a counterexample query would ask for an allowed prompt whose intended
candidate fails to beat its alternative:

<div class="math-display" data-tex="\exists x\in D:\quad z_{\mathrm{target}(x)}(x)\le z_{\mathrm{alternate}(x)}(x).">Find a prompt in the declared domain that violates the candidate comparison.</div>

If a solver proves that query unsatisfiable, it excludes such a prompt under the encoded
semantics, which still leaves us to establish the connection from the encoding to the actual
checkpoint execution. Our Lean trace check verifies the exported records instead, and the
replay recomputes the small checkpoint's forward pass. I stick to that small-model
artifact here (I don't import all the circuit properties, or the later GPT-2-scale results now
described upstream).

<div id="choosing-operators-the-solver-can-represent"></div>

To suit an SMT encoding, Neel's model changes three familiar transformer operations: sparsemax
replaces softmax, SignedL1BandNorm replaces LayerNorm, and LeakyReLU replaces GELU. (The
upstream project supplies the model family, task, training code, circuit extraction, and solver
workflow.)

An SMT solver checks formulas that combine logical conditions with theories such as
arithmetic. For this model the upstream encoder describes each operator with explicit
comparisons and arithmetic branches, and a solver can then reason about the encoded task.
Since the operator choices determine those formulas, we'll examine them before checking the
exported evidence.

<div id="sparsemax-can-give-a-token-zero-attention-weight"></div>

### Sparsemax and zero attention weights

Ordinary softmax exponentiates every allowed score and normalizes the results, so for finite
real scores every weight comes out positive. How can a token get zero weight at all?
[Sparsemax](https://arxiv.org/abs/1602.02068) (introduced by Martins and Astudillo) subtracts a
threshold instead and clips negative results to zero, with the threshold chosen so that the
remaining weights sum to one:

<div class="math-display" data-tex="p_i=\max(z_i-\tau,0),\qquad \sum_i p_i=1">Sparsemax clips shifted scores to obtain a probability vector.</div>

Say the scores are `[1, 0.8, 0]`. A threshold of 0.4 gives `[0.6, 0.4, 0]`, and the last
position contributes nothing at all to the weighted value sum. With `[2, 1, 0]`, a threshold of
1 leaves `[1, 0, 0]`. These examples use exact arithmetic, so you can check them by hand. In
floating point, the replay also has to specify how it computes the threshold.

To find the threshold, the implementation sorts the scores from largest to smallest and scans
prefix sums. For a proposed group of <span data-tex="k">k</span> active entries, subtracting
the same threshold from all of them gives total mass one when
<span data-tex="\tau=(\sum_{i=1}^{k}z_{(i)}-1)/k">τ = (sum of the top k scores − 1)/k</span>,
and the scan checks which entries remain above that threshold. Once the active entries are
fixed, their weights are affine expressions in the scores (weighted sums plus constants).
Comparisons decide the branches, which is handy for the upstream solver encoding.

<div id="how-signedl1bandnorm-rescales-a-vector"></div>

### SignedL1BandNorm, including the zero vector

SignedL1BandNorm first subtracts the vector's mean, then splits the centered coordinates into
positive magnitudes and negative magnitudes and treats those two nonnegative vectors
separately. Each one has an L1 mass (the sum of its coordinates). If a mass is too high, a
projection reduces it, and if it's too low, the implementation distributes the deficit over the
positive coordinates, called its active support. An empty support needs an explicit fallback.
After that, the replay recombines the signed parts, centers again, and applies the learned
scale and shift.

You can see what the fallback does by giving the layer a constant vector, say all ones.
Subtracting its mean leaves 16 zeros, so both signed halves have empty support. The exported
lower mass target is approximately 4.4 for each half, and since the positive fallback activates
the eight even-numbered coordinates and the negative fallback the eight odd-numbered ones, each
active coordinate receives approximately 4.4/8 = 0.55. So before the learned scale and shift,
recombination produces alternating +0.55 and −0.55 rather than a zero vector. Is that a bug?
No, it's a deliberate part of the
[upstream normalization](https://github.com/neelsomani/verifiable-transformers/blob/8e3c54bad2494c17c91109fdb58cf9e2f1c98efc/scripts/small/train.py#L83),
and the Lean replay follows it (on the all-ones vector it gives exactly ±0.55). But the constant
has to be one whose mean comes out exact in floating point. Try a constant 0.3 and the
rounded mean leaves tiny nonzero coordinates, the supports aren't empty, and the replay gives
±0.275 instead.

Those branches matter even for a hidden vector of length 16. Swap in LayerNorm and you get a
different denominator and different outputs, and dropping the empty-support fallback would
also change the computation on inputs where an entire half is zero. So the replay spells those
cases out instead of leaving normalization as a name.

LeakyReLU is easier to describe. Positive inputs pass through unchanged and negative inputs
are multiplied by a fixed slope, which gives another explicit branch without GELU's smooth
nonlinear calculation. “Solver-friendly” here means these operations have useful branchwise
encodings. It doesn't mean every possible transformer built from them becomes easy to verify.

Those details belong in the claim. Substituting softmax or LayerNorm would change the model
being checked even if the tensor dimensions stayed the same, which is why my Lean export summary
checks the operator tags as well as the dimensions, tensor inventory, and parameter count (the
export lists 7,712 scalar parameters in 37 tensors).

<div id="checking-prompts-scores-and-checkpoint-identifiers"></div>

## What the exported evidence proves

The export produces several useful objects. There's a summary describing the architecture and
tensor layout, a finite trace recording each prompt's target and alternative scores, and
circuit summaries with the results of the upstream task-local extraction and solver run on this
checkpoint. And then there's the full parameter payload, which lets me recompute the forward pass.

<figure class="evidence-figure">
  <div><strong>Checkpoint</strong><span>Weights, dimensions, recorded fingerprint</span></div>
  <div><strong>Evaluation evidence</strong><span>Every prompt and its two scores</span></div>
  <div><strong>Lean checks</strong><span>Fingerprint agreement, complete domain, positive margins</span></div>
  <figcaption>My checker only accepts a trace that names the pinned export fingerprint and has the canonical
    prompt at every row. I import the full parameter arrays separately, for the forward replay.</figcaption>
</figure>

Why not just count the rows? Because that would miss duplicates. A file could repeat one prompt
and omit another while keeping the right length. So in
[FiniteEval.lean](VerifiableTransformers/Certificate/FiniteEval.lean) my checker derives the
expected prompt from the row index, where the first 128 rows enumerate quotes and the next 128
enumerate brackets. At each index the task, all six input IDs, the target, and the alternative
must match.

Scores are integers in micro-logit units. The certificate must declare a scale of one million,
and every target score must strictly exceed its alternative score. For the generated summary
and trace we have Lean theorems showing that their checkers accept them.

<div id="rejecting-duplicate-and-missing-prompts"></div>

Our example from above is the first quote row, and its record says 5025436 for token 9 and
2293491 for token 10. Row two changes the quote from 9 to 10. Row three returns to quote 9 and
advances the final content position from 5 to 6, and the enumeration continues like that
through every content combination before it begins the bracket task at global index 128.
Dividing the index by two removes the delimiter choice, and the remaining base-four digits
select the three content tokens.

That ordering gives the checker an independently computed expectation at every row. Here's
the row-identity check in `FiniteEval.lean`:

<pre data-lean-block="finite-row"></pre>

The double equals signs return Boolean comparisons, and `&&` requires all four to pass.
`expectedInput` computes the canonical prompt itself and never reads the claimed prompt
back from the evidence. So length 256 plus successful identity checks at all indices exclude
a trace that quietly repeats an easy case in place of a difficult one. Lean also proves that
`expectedInput` gives 256 different prompts and that they're exactly the prompts the domain
predicates accept, which means no allowed prompt is left out either.

You can inspect the two prompts that begin the quote and bracket tasks, and the commands below
also show the required target for each. Hover over `matchesExpectedIndex` to see the row type
it checks.

<pre data-lean-example="prompts"></pre>

The score check comes afterward. Integer micro-logits let Lean compare saved margins without
executing a floating-point subtraction (a score of one million denotes one unit under the
declared scale), and acceptance proves a positive difference between the recorded integer
scores. But acceptance doesn't prove that exporting and rounding the scores preserved a
particular native margin, which is why the independent floating-point replay is useful.

The fingerprint deserves the same care. My generators compute a SHA-256 hash of each file
they read, and Lean compares the hash recorded in the summary, the trace, the payload, and the
circuit summaries with one pinned value (it doesn't hash the checkpoint bytes in this
development). Those comparisons help keep artifacts from different exports apart, provided the
generators hashed the right files. To authenticate arbitrary newly supplied files, we'd also
need to connect their actual bytes to those identifiers.

A different prompt order, wrong target, missing case, changed recorded fingerprint,
or nonpositive candidate margin fails the corresponding
check. For this checkpoint both extracted circuits are the same three edges (from the
embeddings through the first attention block and the first MLP to the logits), and Z3 verified
functional equivalence and robustness on all 128 prompts of each task and found every edge
necessary. Our circuit-summary checks are small checks of those saved summaries. Producing the
solver evidence is still the upstream SMT solver's job.

<div id="what-survives-changing-the-producer"></div>

So these checks let us inspect the inputs behind an exported accuracy number: we know the
canonical prompt and token IDs occur at each index, and we know the recorded margins are
positive. What we still need is the separate connection from those scores to the checkpoint.

If you rewrite the Python exporter, the new evidence still has to pass the same Lean
checks for the domain, targets, and candidate margins. Adding another prompt requires
changing the expected enumeration too, because changing a label in the export can't enlarge
the theorem. These checks establish what the saved records say about the task.
But do the weights actually produce those decisions? Let's run the model itself.

<div id="running-the-imported-weights-in-lean"></div>

## Recomputing the scores from the weights

The [forward replay](VerifiableTransformers/Replay/UpstreamFloatReplay.lean) starts from the imported
parameter arrays. It adds token and position embeddings, runs both residual attention and
feed-forward blocks, applies the final normalization, and computes the two candidate logits
from the last hidden state.

The replay specifies its arithmetic directly. Dot products use a left fold in coordinate
order, and the custom normalization and sparsemax threshold scans are plain Lean `Float`
functions, so you can inspect each numerical operation without a Python or GPU call in the
replay itself. (The payload's vector and matrix shapes are checked separately in Lean.) Lean
`Float` is binary64 while PyTorch ran the checkpoint in float32, and the replay's sparsemax
scans only the visible positions where PyTorch masks the rest with −10⁴. Even so, on all 256
prompts the replayed logits stay within about one micro-logit of the PyTorch trace!

For one prompt, the token IDs select six embedding vectors of width 16, and position
embeddings are added before the first block. At each position the block normalizes the hidden
vector and computes its query, key, and value projections. Position five may attend to
positions zero through five (position zero only has itself available). Sparsemax is
applied to each position's allowed score vector, and its weights combine the corresponding
values.

Then the attention output is projected and added back to the hidden sequence. The
feed-forward part applies another normalization, expands each width-16 vector to width 64,
applies LeakyReLU, and projects back to width 16 before another residual addition, and the
second block repeats this calculation with its own weights. Finally, a normalization and
language-model head turn the last position's hidden vector into candidate scores. We only
inspect the final position's logits, but earlier positions affect that decision through
attention.

Stored matrices are row-major, so one affine output coordinate takes a weight row, forms its
coordinate-order dot product with the input vector, and adds the bias. The exported inventory
distinguishes the attention projections, feed-forward projections, embeddings, normalization
parameters, and language-model head. Counting 7,712 numbers is a useful consistency check, but
we need the shapes and names too, because the right number of values could still be arranged
in the wrong matrix.

The recorded replay passes all 128 quote prompts and all 128 bracket prompts. For
`[1, 2, 5, 9, 5, 5]` it ends with 5.025436 for token 9 and 2.293491 for token 10, and its
smallest target-minus-alternative margin is 2.324364. A nonpositive margin makes the executable
fail. It also reruns both circuits with the removed edges set to zero (the way upstream's
`controlled_forward` does). Each circuit picks the target on all 128 of its prompts, and
removing any one of its three edges flips at least one decision, which agrees with Z3. These are
results of running the compiled Lean `Float` program, alongside the kernel-checked shape and
integer-trace results, and I haven't proved any theorem that identifies that floating-point run
with the rational SMT computation or with every Python and CUDA execution.

The integer-trace theorem establishes a fact about the saved integer trace, while the replay
executable runs floating-point operations on the imported values and reports its decisions.
That second route reduces our dependence on the supplied score columns, but it still relies on
the compiled program and Lean's native `Float` operations. It doesn't turn a test of 256
compiled forward passes into a kernel proof of those floating-point computations. Btw,
“kernel” here means Lean's proof checker, not a CUDA kernel.

There's also a rational specification for the solver-facing operators in
[UpstreamSmallGPT.lean](VerifiableTransformers/Spec/UpstreamSmallGPT.lean). Exact rational
arithmetic avoids rounding while the replay rounds its operations, so to transfer a rational
decision theorem to the replay we'd need either an error argument strong enough to preserve each
candidate margin or a direct proof about the rounded computation. I have neither yet. The
minimum recorded margin doesn't supply that bound. It tells us what such an argument would have
to protect.

I also kept an archived CUDA trace from our conventional causal transformer (softmax,
LayerNorm, and GELU). It uses the same finite task and dimensions, and its certificate passes
the same domain checker, but it has its own weights and fingerprint, so we check it as a
separate model.

The two models make a useful comparison for this task. But their agreement on the delimiter
choice wouldn't establish that their attention weights or hidden vectors are equal, because
sparsemax and softmax can produce different mixtures while both lead to the right
candidate. Keeping their evidence separate lets us ask the same task question without
pretending they're the same computation.

If you extend the task, the input definition has to change before the evidence does. A seventh
token, another content ID, or nested brackets would create cases outside these 256 prompts, so
you'd need a new enumeration and new scores, and the old accepted trace would no longer cover
the claim. For this experiment, though, the complete task definition is small enough to keep
beside the checkpoint checks and the replay source.

## Conclusion

The next proof I want is the connection between the imported weights, their rounded forward
calculation, and these decisions. Right now the trace theorem checks the saved integer scores
and the replay runs the floating-point model. FloatLib gives me tools to bound the rounding
error between calculations and compare that bound with the winning margins, and since I have
all 256 prompts and the checkpoint right here, I can attempt that connection on the entire
defined task. If you're working on verified transformers too, I'd be happy to compare approaches.

## References

- Neel Somani. [Verifiable Transformers](https://github.com/neelsomani/verifiable-transformers). The model, training code (commit 8e3c54b), synthetic task, custom normalization, circuit extraction, and SMT encoding used here.
- André F. T. Martins and Ramón Fernandez Astudillo. [From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068), 2016. Sparsemax and its threshold construction.
- Ashish Vaswani and colleagues. [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017. The attention calculation underlying the transformer blocks; Somani's model changes several of its operators.
