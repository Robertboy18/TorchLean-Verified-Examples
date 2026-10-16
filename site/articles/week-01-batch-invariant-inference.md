# Thinking Machines Lab’s batch-invariance problem, formalized

We give a model the same prompt twice, but putting one request in a different batch can
change its answer. Even with sampling turned off! As the
[Thinking Machines Lab post on batch-invariant inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
explains, the other requests can affect how the backend groups arithmetic operations, and
floating-point addition depends on that grouping. I wanted to work through that calculation
in TorchLean and FloatLib.

Let's start with three numbers whose sum changes when we move the parentheses, evaluate both
orders in Lean, and then prove conditions under which changing the batch preserves the scores
(or at least the chosen token).

Why doesn't temperature zero save us? It removes randomness from the token-selection rule, but
it doesn't fix the arithmetic that produces the scores. Greedy decoding picks the token with the largest logit
(the model's score for that possible next token), so if the calculation changes the scores, a
close contest can have a different winner. Imagine serving the same prompt twice. The first
time it's the only request waiting, and the second time the server processes it beside several
longer prompts, so padding, tile sizes, and the amount of parallel work may all differ. We
still want the selected request to get the same answer. But fixing the prompt and the weights
doesn't fix all those implementation choices.

[Horace He and the Thinking Machines team](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
state the requirement for their root-mean-square normalization (RMSNorm) reduction precisely:

> the reduction order for each element must be fixed regardless of the batch-size of the kernel.

I like that sentence, because it's specific enough to formalize. We can let the server group requests however it likes
as long as each selected row keeps its own addition tree, and if the tree does change, we need
a different argument (one that uses an error bound and the winning margin).

A tile is a group of coordinates the kernel processes together, and if you change the tile
size, partial sums can get combined in different places even when the selected row contains
the same numbers.
That's the implementation choice we'll represent explicitly below.

So we have two questions to answer. We can keep the arithmetic fixed and ask for exactly the
same scores, or we can allow small numerical differences and ask whether the winning token
stays the same. I'll work through both, and then look at what happens when a server drafts
several tokens and has to reject some of them.

## Why moving the parentheses changes the sum

I picked the smallest useful example, which has three numbers: a large positive number, its
negative, and one. Let <span data-tex="a">a</span> be the binary32 approximation of 10²⁰.
Then there are two ways to parenthesize their sum:

<div class="math-display" data-tex="(a + (-a)) + 1 \qquad\text{and}\qquad a + ((-a) + 1)">Two parenthesizations of a + (−a) + 1.</div>

On the left, the large numbers cancel exactly and leave one. On the right, the first addition
tries to add one to a number whose magnitude is about 10²⁰, and binary32 can't represent that
small a change at that scale, so rounding loses the one and the final addition gives zero.

Binary32 stores each number in 32 bits, and (as
[Goldberg's account of floating-point arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
explains) a normal binary32 number has 24 bits of precision, counting the implicit leading
bit. Its exponent lets it reach large magnitudes without giving it any extra significant
digits, so near 10²⁰ neighboring representable values are much farther apart than one, and
adding one can leave the stored value unchanged.

All three inputs are identical in both calculations. But each addition rounds before the next
one starts, which makes the parentheses part of the program, and if we replaced the
floating-point sum with an equal sum over the real numbers we'd erase the very behavior I want
to study.

<figure class="interactive" id="reduction-example">
  <label>Large term a
    <select aria-label="Large term in the binary32 sum"><option value="1e20">binary32 approximation of 10²⁰</option><option value="16777216">2²⁴ = 16,777,216</option><option value="1024">1,024</option></select>
  </label>
  <output aria-live="polite"></output>
  <figcaption>Each addition rounds to binary32 with <code>Math.fround</code>. Pick 10²⁰ and the two
    orders give one and zero. Both 1,024 and 2²⁴ keep the one (at −2²⁴,
    the step toward zero is still representable).</figcaption>
</figure>

In [Core.lean](BatchInvariantInference/Core.lean), `IEEE32.add_nonassoc_witness` checks this
inequality with FloatLib's executable binary32 model. I give its large operand as the bit
pattern `0x60AD78EC` (and the negative and the one as bits too), so Lean can decide the
concrete inequality without assuming that addition is associative.

Here are those two calculations in Lean. (You can hover over a name to see its type.) The output below
comes from running the commands on the same operands the theorem uses.

<pre data-lean-example="rounding"></pre>

Why bother with bit patterns? Because a decimal expression such as `1e20` is already an
approximation once it's stored in binary32. Giving the bits fixes the operand before we ask
Lean to evaluate either tree, and the model fixes the rounded addition, so the inequality is
about those particular floating-point values and operations rather than some unexplained
difference between two decimal printouts.

Try the smaller magnitudes in the figure. At 1,024, you'll see the one survive, because the
spacing of representable values is small enough there. The 2²⁴ option is more subtle, because binary32 spacing
changes at powers of two and the step from −2²⁴ toward zero is still one, so `−a + 1` stays
representable here. Changing the order of additions can matter without forcing a different
answer for every choice of inputs. But the Lean witness gives one definite failure.

## Choosing the order of a sum

Where does a real model pick an order for its sums? A dot product multiplies corresponding
entries and then adds the products, and a matrix multiplication does many such reductions. A backend may split a long reduction across
workers, combine partial sums, or switch to a different kernel when the batch gets larger, and
Thinking Machines Lab traces batch-dependent numerics like this through matrix multiplication,
RMSNorm, and attention.

I'll use “reduction” for combining several values into one (eg adding the products in a dot
product), and I represent its addition schedule as a tree (a leaf names an input, and an
internal node adds the results of its two children). So the expressions above become two trees
with the same three leaves, and choosing the tree is now an explicit input to the computation.

<div id="representing-the-addition-order-with-a-tree"></div>

For four inputs, a left fold computes $((x_0+x_1)+x_2)+x_3$ while a balanced tree computes
$(x_0+x_1)+(x_2+x_3)$. Both read each input once, in the same left-to-right leaf order, but
the balanced version can do its first two additions independently. That's parallel work a GPU
can use. But it also means rounding at different intermediate values.

I reuse `ReductionTree` from FloatLib, and its evaluator takes both a tree and a function for
combining two values (rounded addition, exact rational addition, or some other operation).
The basic batch-invariance proof doesn't need associativity at all. If the selected leaves and
the whole tree agree, every operation receives the same arguments, so the proof can keep the
numerical schedule instead of replacing it with an unordered mathematical sum, which is why I
want the schedule to be an input too.

My counterexample gives a singleton batch the left tree and a larger batch the right tree.
It keeps the selected row fixed and still changes that row's result from one to zero! So this small
constructed schedule (which says nothing about the schedule any particular production kernel
uses) is enough to prove that batch-dependent arithmetic can violate the serving property.

For the positive result I use the same definitions: equal selected rows evaluated with equal trees
give equal results. A scheduling policy is allowed to depend on the request's hidden dimension,
sequence length, dtype, or block size, as long as an unrelated request can't change the
selected row's schedule. Both the matmul and RMSNorm results spell out this condition for
their own reductions.

Two requests with different context lengths may use different trees, since the condition only
compares requests with the same local metadata, and what it excludes is a tree change caused
solely by someone else's request entering the batch. To say that a particular server obeys it, we'd
have to connect that server's policy to the definition, and I haven't done that for any real
server yet.

Attention adds a second complication. A server can divide a sequence into chunks or put some
keys and values in a cache while computing the rest, and if those choices change where partial
sums meet, they can change rounding too. So the attention specification includes the ordered
key/value reduction tree, and its canonical-layout condition requires that the same
request-local context give the same tree across chunk plans, which our fixed-block
construction satisfies by deriving the block order from the request itself.

Say you have a history of 1,000 keys. Dividing it among four workers gives four groups of 250.
Divide it among eight and the groups shrink to 125 keys, which changes the intermediate sums.
A fixed split size of 256 instead gives groups of 256, 256, 256, and 232, no matter how many
query tokens the kernel is processing at once, and workers can still process those groups in
parallel while keeping the specified order for combining their results. That's the
[fixed-size Split-KV strategy](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#batch-invariant-attention)
that Thinking Machines Lab describes.

<div id="matrix-multiplication-rmsnorm-and-attention"></div>

### The reductions inside a transformer

Take one output coordinate of a matrix product. Its leaves are the products of the selected
input row with one weight column, and no other batch row has any place in that formula. So if
the input row, weights, multiplications, and addition tree stay fixed, that output coordinate
stays fixed. Repeat the argument for every column and you get the matrix result. In the
tensor theorem, the rows and weights have shapes checked in their types.

Zhang and Sennrich's [RMSNorm](https://arxiv.org/abs/1910.07467) uses a reduction too.
For a row of width <span data-tex="d">d</span>, it computes
the mean of the squared coordinates, adds a small positive constant, and rescales the row:

<div class="math-display" data-tex="y_j = w_j x_j\Big/\sqrt{\frac{1}{d}\sum_{k=0}^{d-1}x_k^2 + \delta}">RMSNorm rescales every coordinate using the row's mean square.</div>

Here <span data-tex="w_j">wⱼ</span> is a learned scale and <span data-tex="\delta">δ</span>
is the normalization constant. Every output coordinate shares the sum in the denominator, so if
its rounding changes, the whole row can change with it, but our theorem keeps the supplied
reduction and scalar operations fixed, which means equal rows produce equal denominators and
equal normalized outputs. (A proof about a runtime would also have to say which square root
and other scalar operations it implements.)

Attention combines several reductions: query–key dot products, the normalization of
attention weights, and weighted sums of value vectors. If we preserved only the final
value-sum tree, the earlier scores would still be free to change, which is why my attention
definitions state the feature schedules and the ordered key/value layout the calculation uses.
Then the proof follows those inputs through scores, weights, and outputs.

## When score changes preserve the next token

A changed score can still leave the next token unchanged. How far can the scores move before
the winner changes? Suppose the best two scores are 8 and 3. Moving each one by at most 0.01 still leaves a large gap, but with scores of 8 and 7.999
the same error budget lets the order flip.

Let the winning margin be <span data-tex="m">m</span> (measured against every competing
token), and bound the absolute change of each logit by <span data-tex="\varepsilon">ε</span>.
The winner can move down by ε while a competitor moves up by ε, so the gap shrinks by at most 2ε:

<div class="math-display" data-tex="m > 2\varepsilon \quad\Longrightarrow\quad \text{the winning token stays the same}">A margin greater than twice the error keeps the winner unchanged.</div>

`winner_stable_of_margin` proves this argument with the logits as rational numbers, and
`choose_stable_of_margin` connects it to a decoder that implements strict argmax. Together
they let me state a token-level guarantee without requiring the two logit vectors to be
bit-for-bit equal, although to apply the theorem to a runtime we still need a bound on that
runtime's logit error.

You can follow the proof without knowing any tactics. Write the reference winning score as
<span data-tex="z_b">zᵦ</span> and one competitor as <span data-tex="z_t">zₜ</span>.
In the perturbed calculation the winner is at least
<span data-tex="z_b-\varepsilon">zᵦ − ε</span> and the competitor is at most
<span data-tex="z_t+\varepsilon">zₜ + ε</span>, so subtracting gives:

<div class="math-display" data-tex="z'_b-z'_t \ge (z_b-z_t)-2\varepsilon \ge m-2\varepsilon > 0">The perturbed gap stays positive.</div>

This inequality has to hold for every competitor (not just a pair we picked after looking at
the result), and a uniform bound on all vocabulary coordinates gives us that premise. At
equality, <span data-tex="m=2\varepsilon">m = 2ε</span>, the allowed intervals can touch, so
the theorem can't guarantee a strict winner and the decoder's tie-breaking rule might matter.
So I made the hypothesis a strict inequality.

There's another error convention you have to watch. Suppose two implementations are each within ε of
one shared reference, so the triangle inequality puts them within 2ε of each other. If we
apply the direct-perturbation theorem to that bound, we need a margin greater than 4ε in the
first implementation, but if the reference itself has margin greater than 2ε, we can apply the
theorem to each implementation separately and show that both choose the reference winner.
I wrote a separate result for each of these premises, since calling them all “error ε” would hide
which comparison we'd proved.

Here we're comparing two executions of the same request, with the same input and weights.
Classifier robustness asks a related question (does the winner survive changes to the input
itself?), and the margin argument can help with either once we have the appropriate logit
bound. Where that bound comes from is different, though. A numerical bound has to follow
rounding and execution choices, while a robustness bound has to follow the allowed input
perturbation through the model.

<figure class="interactive margin-demo" id="margin-example">
  <div class="margin-controls">
    <label>Original scores<select aria-label="Choose a score margin"><option value="5">8 and 3: a large gap</option><option value="0.001">8 and 7.999: a close contest</option></select></label>
    <label>Error allowance ε<input type="range" min="0" max="3" step="0.001" value="0.01" aria-label="Absolute error allowance per score"><span class="margin-value"></span></label>
  </div>
  <output aria-live="polite"></output>
  <figcaption>Each bar shows every score allowed by ±ε. Try the close contest, then increase ε.
    As long as a strict gap separates the two intervals, the winner can't change. Once they overlap, a tie or
    reversal becomes possible (though neither has to happen). I picked these scores by hand, so they aren't model measurements.</figcaption>
</figure>

<figure class="data-chart" id="measured-margins">
  <picture>
    <source media="(max-width: 600px)" srcset="results/local_hf_margin_probe_tinyllama-mobile.svg">
    <img src="results/local_hf_margin_probe_tinyllama.svg" alt="TinyLlama-1.1B: top-two score gaps above, twice the observed batching drift below">
  </picture>
  <figcaption>We compared 12 prompts alone and in padded batches using TinyLlama-1.1B in FP16 on an A100.
    Every gap was more than twice the measured drift, and every next-token winner stayed the same.
    (The two panels use different scales, otherwise you couldn't see
    the small changes.)
    <a href="results/local_hf_margin_probe_tinyllama.json">Recorded top-two logits, margins, and drift</a>.</figcaption>
</figure>

I also wanted to see how big these differences get in a model. So I ran the FP16 probe above.
The logits moved! But they moved by much less than the gap between the two highest scores, and
the winning tokens agreed. To make the same argument for later tokens we need the margin and error
bound at each new prefix, since every selected token becomes part of the next input.

Each prompt number refers to one measured pair of executions. So an observed maximum
difference bounds that pair's recorded logits, and it doesn't become a bound for all future
requests or for every kernel the server might select. I keep it beside the theorem anyway,
because it makes the size of the margin concrete while leaving the theorem's premise visible.

To carry the result through generation, start with equal prefixes, which give us the same
reference input for the next step, and if that step's margin and error satisfy the inequality,
both implementations append the same token and we're back to equal prefixes, ready to repeat
the argument. One failed margin condition stops this proof, even when the two programs happen to
keep agreeing in a test.

<div id="following-an-error-bound-through-the-model"></div>

## From layer errors to logit errors

How do we get an error bound for the logits, one small enough to leave every competing score
below the winner? Let's follow an input error through a linear layer.

In a linear layer with fixed real weights, an error in one input coordinate contributes
through every weight that reads it. If $y_j=\sum_i x_i W_{ij}$ and the second calculation uses
the same weights, the triangle inequality gives

<div class="math-display" data-tex="|y_j-\widehat y_j|\le\sum_i |W_{ij}|\,|x_i-\widehat x_i|.">The weights determine how an input error reaches each output coordinate.</div>

A numerical implementation adds local rounding error too. Let each input-coordinate error be
at most $E$ and let $L$ bound every column sum $\sum_i |W_{ij}|$, so that with a bound $\rho$
for the local arithmetic the next allowance is $LE+\rho$ (amplify the incoming error by at most
$L$, then add the local allowance $\rho$). So a large weight can amplify an earlier difference, and
a small local rounding error on its own doesn't control the final logits. Errors in stored
weights would contribute more terms, which means the fixed-weight premise is doing real work
here.

Stack several layers and the later maps can amplify errors introduced earlier. So their
numerical contracts have to account for the actual operations and ranges (normalization
denominators included). Once the resulting bound covers every logit at the current prefix, the
margin theorem tells us whether it's enough for the next token. These equations explain the
analysis that's required, though I haven't built a complete numerical certificate for a
production transformer yet.

If you write a new backend, you can reuse the argmax proof once you establish its numerical
bound, and a new scheduler can reuse the reduction-tree proof if it preserves the selected request's tree.
Either way, the work is checking the implementation against the hypotheses of a theorem we
already have.

<div id="keeping-speculative-generation-consistent"></div>

## Committing a draft without corrupting the cache

A serving system can speculate ahead, verify a window of proposed tokens, and throw away the
rejected candidates. [Leviathan, Kalman, and Matias](https://arxiv.org/abs/2211.17192)
describe speculative decoding and prove a distribution-preserving sampling procedure. What I
wanted to check here is the committed state and tokens, because those committed tokens (the
ones that survive verification) are what a user receives.

The invariant relates two pieces of state: the committed token list, and the reference decoder
state after those tokens. Speculation changes only the fast-path state and candidates, rollback
clears candidates, and a commit must carry evidence that the accepted window and resulting
reference state match an execution of the reference decoder. An induction over these
transitions proves `ServeDVR_refines_RefDecode` (every committed output is a prefix of a
reference continuation), so two legal traces that commit the same number of tokens show the
user the same tokens.

Say a draft proposes three tokens and the verifier accepts only the first two. What should the
state look like after that? The committed prefix advances by those two tokens, but the rejected third token must not advance the
reference state, and a later draft starts from the reference state after the accepted pair. An
implementation that discards the token but keeps its cache update would violate the invariant,
because its state would describe a longer history than the user has received.

My favorite thing about this proof is that it doesn't care how clever the draft is. Speculation may change its private state and
produce poor candidates all it wants. What matters is the commit rule, where the verifier
supplies agreement for the accepted window and its resulting reference state, and rollback
never touches that committed state. Induction over a serving trace then applies
the same local argument after every speculation, rollback, or commit.

This also explains the “same number of tokens” premise. Two servers may stop at different
times or commit different-length windows, so their outputs can be different-length prefixes of
the same reference continuation. Once those lengths agree, the theorem gives equality (it says
nothing about equal latency, or about an equal number of completed tokens at a given instant).

The verifier's agreement with the reference is a hypothesis of this theorem, and it means exact
agreement with running the reference one token at a time. A verifier that scores a whole
window in one pass processes several query tokens at once (the chunked-attention case
from earlier), so its reduction schedules have to match the one-token reference too. We also
connect the serving statement to our IR decoder (whose reference computation is the
denotation of a model graph), and there's a smaller experiment closer to compiled code as
well. The
[CUDA reduction checker](BatchInvariantInference/CUDA.lean) accepts a certificate extracted
from the PTX of an eight-step fused multiply-add chain and proves that the accepted program
stores the intended value reduction for each row and output coordinate. To apply it to a live
kernel, you need the stated indexing, allocation, and launch conditions. And the
certificate only covers that reduction fragment, so a full attention kernel still needs the
rest of its computation connected to the attention specification.

A fused multiply-add rounds the combined product and addition once, where separate
multiplication and addition round twice. So the CUDA fragment has its own FMA-chain semantics,
and we shouldn't silently identify it with every reduction written using separate operators.
The accepted certificate connects the extracted chain to that specified reduction. Lean checks
the program the extractor recorded (every load and store address included), since arithmetic
agreement alone can't show that a kernel loaded the intended coordinates. Reading the PTX and
its guard branches is the extractor's job, and the launch conditions are assumptions.

If you handed me a larger serving implementation, I'd first identify the request-local reduction policies,
then establish numerical or exact agreement for the layer computations, and only then apply
the token and commit theorems. The small certificate makes one of those steps inspectable,
which leaves the remaining layer and runtime connections to supply.

## Conclusion

I'd like to take this further with an inference implementation, which would mean getting its
actual reduction schedules and cache policy and then a bound on the score differences at each
reached prefix. The margin theorem tells us how to use that bound, and getting it for the
running model is the next job. If you're working on those kernels or serving policies, I'd love
to compare examples :)

## References

- Horace He and collaborators at Thinking Machines Lab. [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), 2025. The batch-invariance problem and fixed-size reduction strategies that motivated this example.
- David Goldberg. [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html), 1991. Representation, rounding, and non-associative arithmetic.
- Biao Zhang and Rico Sennrich. [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), 2019.
- Yaniv Leviathan, Matan Kalman, and Yossi Matias. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), 2023. The original sampling result is distinct from our state-and-commit theorem.
