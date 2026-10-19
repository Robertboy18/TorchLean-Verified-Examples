# Estimating model behavior from its structure

A new institute, the [Mathematical AI Safety Institute, or MAISI](https://maisi.org/), is
bringing mathematicians to questions about the safety of powerful AI systems. One of its
research directions,
[developing good heuristics](https://maisi.org/research/directions/heuristic-estimators),
caught my attention because it asks how much we can learn from a model's structure.

Usually we learn what a model does by running it. We pick some inputs, collect the outputs,
and estimate a quantity we care about: an average loss, an attention score, or the frequency
of a defined failure. But we also have the program and its weights. I'd like to know how
much of that evaluation we can replace with a calculation about the model itself. The
Alignment Research Center (ARC) has been developing this question through its work on
mechanistic estimation and the matching sampling principle.

MAISI's page describes ARC's two ingredients as an argument plus a checker. In the
computational no-coincidence conjecture, a polynomial-time verifier $V$ checks a short
advice string $\pi$. In the matching sampling principle, advice $\pi$ describes a model's
structure and an estimator uses it to compete with sampling. I like to use Lean the same
way: a search, a heuristic or a person proposes the sketch or the bound, and Lean checks
what the proposal implies. The two checks do different jobs, though. Their $V$ has to be
fast and accept heuristic arguments, while a Lean check proves a mathematical statement,
which is stricter and usually slower. What TorchLean adds is one definition of the network,
shared by the model, the estimator and the theorem, with exact or FloatLib arithmetic, so we
can say which step of an estimate is exact and which one is an assumption.

I wanted to work through this with the tools we've been building. In earlier posts we
[trained a GPT](../week-03-gpt-training/),
[worked through K3's equations](../week-04-kimi-k3-specification/site/), and
[followed rounding through Nemotron's recurrent state](../week-05-nemotron-specification/site/).
Here I'll use TorchLean's tensor calculations to study an estimator and Lean to check what
its answer means, and FloatLib comes in once we store and execute the numerical summary.

We'll start with a calculation small enough to understand completely: throw away a
dependency on purpose, then recover it with covariance. The harder question is which
dependencies we can afford to forget in a large model, and what we can prove about the error
when we do. I think that's where there's room for an interesting research project. For this
post I wrote Lean proofs of the query–key expectation theorem, a finite sketch with an
executable error certificate, a coefficient-rounding bound using FloatLib, a sampling
theorem, a graph-wide error identity, a conditional-averaging theorem for removing some
input randomness, and the moments of a ReLU applied to a Gaussian. We'll run the small
instances, follow the general arguments, and then race three structural estimators against
sampling on 16,800 random ReLU networks. All of these pieces are meant to support a larger
project: a useful automatic estimator for trained models.

## Two branches, one input

Suppose two independent random signs, $X$ and $Z$, are each equally likely to be $-1$
or $1$, and we form a second branch that partly reuses $X$:

$$Y_c = cX + (1-c)Z.$$

Here $c$ controls how much input the branches share: at $c=0$ they're independent, and at
$c=1$ they're the same random variable. Now multiply them:

$$F_c = XY_c = X\bigl(cX+(1-c)Z\bigr).$$

Both branches have mean zero, so an estimator that remembers only their means and multiplies
them at the last operation returns zero for every $c$. But $X^2=1$, so

$$F_c = c+(1-c)XZ, \qquad \mathbb E[F_c]=c.$$

At $c=1$ the estimator predicts zero for a program that returns one on every input. Nothing
unusual happened during execution. The mistake was treating two uses of the same input as
independent information.

Here's the calculation with tensors. Its four rows enumerate every pair of signs with equal
weight, using rational arithmetic instead of a random sample.

<pre data-lean-example="sharedExpectation"></pre>

For $c=1/2$ the outputs are $[1,0,0,1]$, whose mean is $1/2$, while the last command
multiplies the two branch means and returns zero. Both calculations use the same tensor
operations. What differs is when we discarded information.

<figure class="interactive" id="dependency-estimate">
  <div class="margin-controls">
    <label>Shared-input coefficient <span data-shared-label>c = 0.50</span>
      <input aria-label="Shared-input coefficient" type="range" min="0" max="100" value="50">
    </label>
    <label>Independent samples
      <select aria-label="Number of independent samples"><option value="1">1</option><option value="4">4</option><option value="16" selected>16</option><option value="64">64</option></select>
    </label>
  </div>
  <div class="estimate-pairs" aria-label="All four equally likely inputs">
    <div><span>(−1, −1)</span><strong data-sign-output>1.00</strong></div>
    <div><span>(−1, +1)</span><strong data-sign-output>0.00</strong></div>
    <div><span>(+1, −1)</span><strong data-sign-output>0.00</strong></div>
    <div><span>(+1, +1)</span><strong data-sign-output>1.00</strong></div>
  </div>
  <div role="status" aria-live="polite">
    <p data-estimate-summary>The true mean is 0.50; the mean-only estimate is 0.</p>
    <table><thead><tr><th>Method</th><th>Mean squared error</th></tr></thead><tbody>
      <tr><td>Mean-only rule</td><td data-naive-error>0.250000</td></tr>
      <tr><td>Average of independent samples</td><td data-sampling-error>0.015625</td></tr>
      <tr><td>Exact covariance correction</td><td>0</td></tr>
    </tbody></table>
  </div>
  <figcaption>These are analytic errors for this four-point distribution, not measured runtimes.
    The sample average is unbiased and has mean squared error
    <span data-tex="(1-c)^2/n">(1 − c)² / n</span>. The mean-only rule has squared error
    <span data-tex="c^2">c²</span>. Moving the slider changes the program, not a sampled batch.</figcaption>
</figure>

The variance is $(1-c)^2$, since $XZ$ is another mean-zero sign, and that gives a useful
comparison with sampling. Averaging $n$ independent evaluations gives expected squared error
$(1-c)^2/n$, but the mean-only estimate has squared error $c^2$ however many times we repeat
its calculation. A cheap estimate can be confidently wrong.

Below are the tensor definition and the two checked identities. `response` works for any
shape and any scalar type that provides the operations it uses, while the mean and variance
theorems specialize to real arithmetic and the four equally weighted sign pairs. They
quantify over every real `shared`, including values the slider can't reach.

<pre data-lean-block="heuristic-response"></pre>

We can execute the same definition with rationals, native floats, or supported FloatLib
formats. Changing the scalar type changes the arithmetic, though, and the real-valued
theorem alone doesn't say that all those executions give identical answers.

## What covariance buys us

The correction is familiar:

$$\mathbb E[XY] = \mathbb E[X]\mathbb E[Y] + \operatorname{Cov}(X,Y).$$

For our two branches $\operatorname{Cov}(X,Y_c)=c$, and keeping that one number recovers the
answer. There's no Gaussian approximation or independence assumption here. For
square-integrable real random variables on a probability space, the identity is exact.

Mathlib already has this result in its covariance API, and I restated it in the form we
want for a multiplication rule:

<pre data-lean-block="heuristic-covariance"></pre>

The hypotheses `MemLp x 2 μ` and `MemLp y 2 μ` say that the variables have finite second
moments, with the required measurability conditions, and that's what justifies the integrals
in the statement. I'd rather reuse that probability theory than introduce a separate meaning
of expectation inside a tensor estimator.

ARC's [*Formalizing the presumption of independence*](https://arxiv.org/html/2211.06738v1)
studies mean, covariance, and cumulant propagation in Appendix D. Covariance repairs the
product-mean rule, but propagating products creates demands for higher moments, and
approximating the intermediate variables as jointly Gaussian supplies a closure rule that
fills in information we didn't retain. Our shared-input example is a small case of that
known difficulty, and I didn't find a new error in their work.

We can see the next difficulty without a network. Let $A$ be a uniform sign, and let $B$
equal $-\sqrt2$ or $\sqrt2$, each with probability $1/4$, and zero with probability $1/2$.
Both have mean zero and second moment one. But

$$\mathbb E[A^4]=1, \qquad \mathbb E[B^4]=2.$$

A summary containing only the first two moments describes $A$ and $B$ identically, yet if
the remaining program computes the fourth power, the correct answers differ. In my Lean
counterexample, $A$ is represented by $(-1,-1,1,1)$ and $B$ by $(-\sqrt2,0,0,\sqrt2)$, four
equally weighted rows each, and Lean checks their means, second moments, and fourth moments
together. Covariance alone isn't enough for a nonlinear continuation.

<details>
<summary>Check the matching-moments example in Lean</summary>
<pre data-lean-block="heuristic-higher-moments"></pre>
</details>

I want an implementation to tell us when it used an exact identity and when it used a
closure assumption. For an affine layer the usual mean and covariance transforms are exact,
as long as the input moments exist:

$$\mu_{Wx+b}=W\mu_x+b, \qquad
  \Sigma_{Wx+b}=W\Sigma_xW^{\mathsf T}.$$

I prove these with the same matrix–vector and addition operations as the tensor program. The
covariance theorem allows two different affine projections of one input, giving
$W_q\Sigma_x W_k^{\mathsf T}$ for their cross-covariance. Taking both maps equal recovers
the formula above. We'll need the shared-input case for attention, and nothing here assumes
that the input coordinates or the two projections are independent.

<details>
<summary>Read the affine moment theorems</summary>
<pre data-lean-block="heuristic-affine"></pre>
</details>

A nonlinear layer needs more information or an approximation. A residual addition needs the
cross-covariance between the two branches, and an attention score multiplies queries and
keys that may depend on the same input. If we treat those operations as independent just
because they sit in separate nodes, we recreate our four-row mistake at a much larger scale.

## Keep what the next computation needs

Storing every pairwise covariance quickly gets expensive, and higher-order interactions cost
more. I'd like to avoid choosing between a mean-only summary that loses the answer and a
complete distribution that we can't afford to carry.

A more useful question is what the *rest of the program* will do with an intermediate.
Suppose a partial computation produces $g(x)$, we replace it by a cheaper sketch
$\widetilde g(x)$, and then we multiply by a later function $h(x)$. The error in the final
expectation is

$$\mathbb E[h g]-\mathbb E[h\widetilde g]
  =\mathbb E\bigl[h(g-\widetilde g)\bigr].$$

The discarded residual $g-\widetilde g$ matters through its interaction with $h$, and
controlling only its mean doesn't control that interaction. Take $g(X)=X$, replace it by its
mean zero, and let $h(X)=X$. Before multiplication the sketch has the right mean. After
multiplication the true mean is one and the sketched mean is zero.

<pre data-lean-example="discardedSignal"></pre>

I checked this general error identity in Lean too. Its two integrability assumptions make
subtracting the integrals legitimate:

<pre data-lean-block="heuristic-projection"></pre>

ARC's
[work on expectations of random products](https://www.alignment.org/blog/mechanistic-estimation-for-expectations-of-random-products/)
develops this downstream-sensitive view. Their deduction–projection estimators alternate
computation with simplification, and the sketch should retain whatever information helps
pair the computed head of a product with its remaining tail. They study a second-moment
operator for the tail family and project onto leading eigendirections, and symmetry makes
some instances tractable (their examples include random halfspace intersections,
random #3-SAT, and permanents). I think this is a richer starting point than applying the
same moment cutoff everywhere.

We can work through
[Hilton's sketching argument (§3)](https://www.alignment.org/content/files/2026/05/deductive_estimation_for_random_halfspace_intersections.pdf#page=3)
with finite tensors and give each proposed projection an explicit account of the future
computations it has to preserve. For a finite family of tails with probabilities $p_c$,
define the residual $r=g-\widetilde g$. Then

$$\sum_c p_c\bigl(\mathbb E[h_c r]\bigr)^2$$

measures how much that projection changes the expected answers across contexts. On a finite
input space the distribution-weighted pairing is a vector dot product (we absorb the input
weights into the vectors), so the expression becomes $r^{\mathsf T}Kr$ for a
positive-semidefinite matrix $K=\sum_c p_c h_c h_c^{\mathsf T}$, and keeping leading
eigendirections becomes a concrete linear-algebra problem. I proved the Gram identity for
arbitrary finite dimensions, not only for the numbers below, by expanding the square and
exchanging finite sums:

<pre data-lean-block="heuristic-gram"></pre>

Here's a case where the choice of sketch matters. Say our intermediate vector is $g=(100,1)$
and we can keep only one coordinate, while two equally likely continuations read its second
coordinate with weights one and two. Keeping the first coordinate minimizes reconstruction
error, since we lose only one unit. But both continuations ignore that large first
coordinate. Keeping the second gives the exact downstream answers, although its
reconstruction error is much larger.

<pre data-lean-example="sensitiveSketch"></pre>

Each of the first two outputs pairs squared reconstruction error with squared downstream
error. Keeping the largest coefficient gives $(1,5/2)$ and keeping the useful coefficient
gives $(10000,0)$, so for this task a reconstruction metric would choose the wrong
coordinate. Here the Gram matrix explains the difference:

$$K=\begin{pmatrix}0&0\\0&5/2\end{pmatrix}.$$

I also wrote a checker for a uniform error bound. For a symmetric matrix it verifies

$$\sum_j |K_{ij}|\leq\lambda\quad\text{for every row }i.$$

Lean proves that this implies $r^{\mathsf T}Kr\leq\lambda\|r\|_2^2$ for every residual, and
the proof bounds each cross-term using $2|r_i r_j|\leq r_i^2+r_j^2$, sums over the matrix,
and uses symmetry to combine the row and column sums. While the executable check uses exact
rationals, its soundness theorem also covers the interpreted matrix acting on arbitrary real
residuals, so we don't have to enumerate those vectors.

<pre data-lean-block="heuristic-certificate"></pre>

In the example the checker rejects the claimed bound two and accepts $5/2$. Once we retain
the useful coordinate, `restrict` zeros its row and column in the remaining-error matrix.
That leaves a zero matrix, and a zero bound passes. With that check, `useful_sketch_exact`
proves zero downstream error for *every* rational intermediate vector under those
continuations, not merely for $(100,1)$.

<pre data-lean-block="heuristic-refinement"></pre>

This certificate covers coordinate sketches. We can inspect a candidate without trusting how
it was chosen, but the certificate doesn't find a good candidate, and an absolute row-sum
bound can overestimate the largest eigenvalue. More economical certificates, arbitrary
subspaces, and matrices available only implicitly are still open here. I think they're
interesting problems.

Adaptive refinement splits into the same two jobs: an algorithm proposes which coordinate to
retain, and the checker verifies its claimed remaining bound. Retaining more coordinates
never increases the residual's squared norm (that's `residual_energy_mono`), but actual
downstream error need not decrease at every step, because discarded terms can cancel. What
we can justify without assuming away those cross-terms is a certified budget that never
increases.

## One model definition, several analyses

In the earlier articles we used TorchLean to connect model equations to typed graphs and to
reason about arithmetic and recurrent state. I want to reuse those definitions here instead
of writing the estimator against its own copy of the network, because a separate, manually
transcribed network would be one more place for the model and its analysis to disagree.

<figure class="evidence-figure">
  <div><strong>Tensor program</strong><span>Inputs, parameters, shapes, shared nodes, and an explicit output observable.</span></div>
  <div><strong>Proposed estimator</strong><span>Moment summaries or sketches, with explicit projection choices and residual assumptions.</span></div>
  <div><strong>Comparison</strong><span>Exact finite cases, sampled estimates, and checked conditional bounds for the same target.</span></div>
  <figcaption>The proposed workflow starts from the program we already use. This article
    checks moments, finite sketches, sampling error, coefficient rounding, and an error
    identity on TorchLean's existing autograd graph. Each operation still needs a justified
    local bound before that identity becomes a useful certificate.</figcaption>
</figure>

The output observable matters. We might ask for an average loss, the expected score of one
token, or the probability that a defined check fails, and those are different functions of
the model output. After the remaining nonlinear operations, a good approximation to the mean
hidden activation need not answer any of them accurately.

My composition theorem works on TorchLean's existing proof-carrying autograd graph, shared
branches included. It compares a full execution with a reference execution and uses the
reference backward pass to account for each node's contribution to the output error. There's
no second graph representation and no second differentiation engine. That identity is exact.
Any useful approximation comes from bounding its remainder terms.

The hard part is obtaining useful residual bounds cheaply. A theorem whose hypothesis says
that every approximation is accurate doesn't solve that problem. It does give us a place to
connect a later analytic result, a finite certificate, or a deliberately labeled heuristic
assumption, and we should be able to inspect which of those supplied each bound.

Sharing belongs in that representation too: if two branches use the same node, we must
retain that fact. Rewriting a real-valued polynomial graph into an equal polynomial keeps
its true expectation, but an approximate estimator may change its answer because the rewrite
changes what it discards. That makes running both forms a useful diagnostic. We can prove
the targets equal and then look at any disagreement between the estimates, which won't prove
either estimate correct but does show sensitivity to representation.

The backward pass supplies the sensitivity weights, but a derivative at one activation
doesn't bound a large random change, so we also have to bound what the local linearization
misses. For ReLU networks we can identify that missing contribution clearly.

## When can we average away an input?

Suppose we keep part of a random input and replace the rest by its mean. I'd like to know
when that preserves the model's expected output, even if it changes individual predictions.
A linear layer is easy. Averaging commutes with its calculation, but at a ReLU we have
to ask whether the removed contribution can switch the neuron on or off.

Write a scalar preactivation as $z+u$, where $z$ is the retained contribution, then subtract
the branch's linear response and call the remainder $r$:

$$r(z,u)=\operatorname{ReLU}(z+u)-\operatorname{ReLU}(z)
  -\mathbf 1_{z>0}\,u.$$

If $|u|\leq\delta$, then

$$0\leq r(z,u)\leq\max(0,\delta-|z|).$$

Far enough from zero the chosen branch is affine and the remainder vanishes. Near zero we
charge only the distance by which an allowed perturbation could cross the boundary, which is
tighter than charging the whole perturbation everywhere. The Lean theorems use TorchLean's
actual scalar ReLU, including the zero-subgradient convention its JVP uses, and a separate
tensor theorem identifies the same remainder after subtracting that existing JVP.

<pre data-lean-block="heuristic-boundary"></pre>

For a whole graph, let $x$ be the full input and $\widetilde x$ the reference input. With a
fixed linear output readout, the comparison has the form

$$F(x)-F(\widetilde x)
  =\langle g_{\mathrm{input}}(\widetilde x),x-\widetilde x\rangle
    +\sum_v\langle g_v(\widetilde x),r_v(x,\widetilde x)\rangle.$$

Here the $g_v$ are the cotangents from the reference backward pass, and the local remainders
$r_v$ account for everything its linearization misses. `Composition.eval_difference` proves
this identity by induction over the existing graph, and its algebraic statement also permits
seeds on intermediate nodes without assuming that an arbitrary supplied JVP is an analytical
derivative.

Weighting local errors by their effect on a chosen output has a long history in
[goal-oriented error estimation](https://www.cambridge.org/core/journals/acta-numerica/article/abs/an-optimal-control-approach-to-a-posteriori-error-estimation-in-finite-element-methods/5C67A03F528C6FA69F37A97DF5C3BE19).
Here we use the network's existing VJPs for those weights and keep the nonlinear remainders
explicit. I'm not claiming that sensitivity-weighted error analysis is new. What I'm working
toward is a checked, computable estimator for model behavior.

<details>
<summary>Read the graph discrepancy theorem</summary>
<pre data-lean-block="heuristic-graph-remainder"></pre>
</details>

Now write the input as $(Z,U)$ and use $(Z,0)$ as the reference. If
$\mathbb E[U\mid Z]=0$, the input-linear term disappears in expectation, because the reference
backward pass depends only on $Z$ while the discarded perturbation has conditional mean
zero. Independent centered input coordinates satisfy this condition. Mean zero or zero
covariance alone does not.

For a finite conditional input law, `Averaging.mean_error_le` proves the resulting bound
from coordinatewise certificates for every node's remainder, and its right-hand side
evaluates only reference states and their VJPs. For a ReLU, the radius above supplies the
local rule once we've bounded its preactivation change. Later hidden-layer perturbations
need not be centered: we bound their remainders pointwise and cancel only the original input
term.

<details>
<summary>Read the conditional-averaging theorem</summary>
<pre data-lean-block="heuristic-averaging"></pre>
</details>

I tried this on constructed three-layer networks with 64 independent sign inputs, where
keeping four signs leaves 16 reference states to evaluate. One network had stable activation
branches throughout the full input cube, so the budget was zero. An independent affine
calculation recovered the same expectation, even though all 60 discarded signs had nonzero
weights into active first-layer neurons and could still change individual outputs. In a
nonlinear example the first readout's actual mean error was about $0.000405$, within a very
conservative bound of $0.195$, and its repeated input weights made a separate exact binomial
calculation possible. I also checked a smaller network exhaustively, plus a correlated-input
counterexample that violates conditional centering. All the exact fractions are in the
[experiment record](experiments/results-boundary.json).

These instances use exact rational Python calculations. Lean checks the general graph and
averaging theorems, but the experiment doesn't emit proof terms for those networks. To make
this automatic I still need operation-specific radius propagation and an executable
certificate generator. Smooth activations need curvature bounds and multiplication needs its
cross-perturbation term, and while the same graph identity gives us somewhere to connect
those rules, the ReLU result doesn't establish them.

## An attention score we can calculate exactly

An attention head makes a useful next target. Let $q(x),k(x)\in\mathbb R^d$ be a query
and key computed from the same random input, and consider their unscaled score
$q(x)^{\mathsf T}k(x)$. When the coordinates have finite second moments,

$$\mathbb E[q^{\mathsf T}k]
  =\mu_q^{\mathsf T}\mu_k+\operatorname{tr}\bigl(\operatorname{Cov}(q,k)\bigr).$$

This is our two-branch example with vectors, and it gives us a direct reason to preserve
cross-covariances between the query and key projections. Sampling the two branches
independently would change the distribution and remove the very dependency we want to study.
We can write their dot product with the same tensor contractions used in the
[attention example](../torchlean-tensor-programming/#the-two-contractions-inside-attention).

I picked four query–key pairs where each key scales the first query coordinate by $1/2$ and
the second by $1/4$. Both marginal means vanish. Every dot product is $3/4$, and the
cross-covariance diagonal contains the missing $1/2$ and $1/4$:

<pre data-lean-example="attentionMoments"></pre>

The first output averages the actual scores, the second multiplies the marginal means and
loses the answer, and the third evaluates the covariance-aware rule and recovers $3/4$.
`score_mean` proves the same expectation identity for any finite dimension and any query
and key distribution whose coordinates have finite second moments:

<pre data-lean-block="heuristic-attention"></pre>

The identity is only useful once we have the true moments, and getting them from a trained
model isn't free. Affine projections are one useful case, since their input means and
cross-covariances transform by exact matrix formulas, but beyond that we need an
approximation rule and an account of the information it loses.

We can also compare a full score with one computed after projecting both vectors into a
smaller subspace. For an orthogonal projector $P$ the pointwise score difference is

$$q^{\mathsf T}k-(Pq)^{\mathsf T}(Pk)=q^{\mathsf T}(I-P)k.$$

In the tensor theorem the right side is the score against $k-Pk$. It requires
$P^{\mathsf T}=P$ and $P^2=P$, and it uses TorchLean's existing matrix–vector operations:

<pre data-lean-block="heuristic-attention-projection"></pre>

That gives the projection rule an explicit target. Which subspace keeps the expected score
accurate, and how much does it cost to identify it? I don't know yet. Moving to the expected
*squared* score would require fourth-order information, assuming the relevant fourth moments
exist, and a sketch that works for the score might fail for its square.

I then tried this on attention states from our trained 124-million-parameter GPT, where
TorchLean replayed 3,072 first-block head queries. Keeping half the query–key coordinates
with an output-aware ordering gave 24.1% relative attention-output error, compared with
38.9% for the seeded random ordering. Packing the retained coordinates cost more than the
smaller contraction saved. A later sweep covered layers 1, 6, and 12 at histories of 64,
128, and 512 tokens, and the quality of the summary depended strongly on both.

Those runs compare attention outputs on fixed supplied activations, before the output
projection, so they don't yet construct an expectation estimator from the model and input
law. They do give us larger computations on which to test the proposed summaries and their
costs. Our [experiment report](experiments/RESULTS.md) includes the exact scope, the softmax
error certificates, and follow-up checks on a small transformer and a trained digit
classifier. Rerunning the GPT part needs the archived binary32 Week 3 checkpoint and its
validation shard, which the repository doesn't distribute. Checkpoints written by the
current Week 3 code use a newer format that `attention.py` doesn't read.

## Sampling the part we cannot explain

We don't have to give up sampling while we develop this. Suppose we have a cheap surrogate
$S(x)$ whose expectation $m_S$ is known, and a target $T(x)$. For independent samples
$X_1,\ldots,X_n$ from the target distribution, consider

$$\widehat m = m_S + \frac1n\sum_{i=1}^n\bigl(T(X_i)-S(X_i)\bigr).$$

If $T$ and $S$ are integrable and $m_S=\mathbb E[S]$, this estimate is unbiased: the sample
average of $T-S$ has mean $\mathbb E[T-S]$ (`Sampling.mean_unbiased`), and
$\mathbb E[S]+\mathbb E[T-S]=\mathbb E[T]$. That population identity is short enough to
show in full:

<pre data-lean-block="heuristic-residual"></pre>

Under finite second moments its variance is $\operatorname{Var}(T-S)/n$. With a sufficiently
correlated surrogate the residual can be much easier to estimate than the original output. A
poor surrogate can make the variance worse. If the surrogate mean is only approximate, its
error becomes a bias that residual sampling doesn't remove.

We can make that bias quantitative. Write $\delta=\widetilde m_S-\mathbb E[S]$ for the error
in the surrogate mean we supply, and the checked sampling theorem gives

$$\mathbb E\!\left[(\widehat m-\mathbb E[T])^2\right]
  =\frac{\operatorname{Var}(T-S)}{n}+\delta^2.$$

It uses `Tensor.mean` on the residual samples and mathlib's probability definitions. In
`residual_mse` the real `surrogateMean` stands for $\mathbb E[S]$ and `residualMean` for
$\mathbb E[T-S]$, the common mean of the residual samples, so by `residual_correction` their
sum is $\mathbb E[T]$. Pairwise zero covariance between sample residuals is enough for the
calculation, and independence implies that condition. These are relationships between
separate samples, though. They don't give us permission to erase dependencies inside the
network.

<pre data-lean-block="heuristic-sampling"></pre>

The averaging construction gives us one way to obtain the surrogate mean, by summing over
its retained states, and its boundary certificate bounds the bias if we stop there. If we
also sample the residual, the theorem above describes the correction. A small mean error
doesn't imply a small residual variance, though. Even in a stable affine region a discarded
mean-zero input can cause large fluctuations that cancel only after averaging.

That suggests a practical intermediate experiment. A structural estimator would construct a
surrogate with a tractable expectation, and we'd measure whether it reduces the residual
variance, counting the cost of constructing and evaluating it alongside the target. A
surrogate fitted on a pilot sample has to be fixed before we draw the independent correction
sample, since otherwise the elementary fixed-surrogate argument above needs a different
analysis.

This hybrid isn't automatically a mechanistic estimator in ARC's sense, because the final
answer still uses random observations. I'd use it as a baseline and as a way to find which
dependencies are worth modeling, and keep the purely structural question separate. It also
gives a useful failure report: whether the approximation went wrong because it missed
structure, because its residual estimate was noisy, or because its known-mean premise was
inaccurate.

## What competing with sampling would require

ARC's [*Competing with sampling*](https://www.alignment.org/blog/competing-with-sampling/)
motivates structural estimation partly through events too rare to encounter in ordinary
evaluation. Its matching sampling formulation pairs the structural advice $\pi$ with a
context $c$, and requiring the same advice across contexts stops the advice from simply
storing each answer. The comparison is mean squared error averaged over contexts, with an
execution budget comparable to sampling, and matching this general benchmark is a different
goal from substantially beating it on useful structured models.

Our four-sign family has a visible context parameter, but proving its mean isn't evidence
for the general principle, since four evaluations enumerate its entire input space. I chose
it to expose a dependency and make the target unambiguous. A serious benchmark needs larger
instances, a specified context distribution, a fixed advice construction, and an accounting
of the work needed to obtain that advice.

I'd report error against total cost, including preprocessing, memory, and any samples used
to choose a sketch. For a reusable explanation I'd also show the number of contexts over
which preprocessing is amortized, and I'd separate time spent finding a candidate from time
spent checking a claimed bound. A fast checker with an impossibly expensive candidate search
may be mathematically interesting. It still hasn't given us an efficient estimator.

A fair comparison would include plain sampling, mean propagation, a covariance-based method,
and a structure-aware sketch at comparable cost. Tiny exact instances would supply ground
truth, and larger experiments could use independent high-budget estimates with their own
uncertainty reported. I'd make rewrites, shared branches, and distributions with matching
low-order moments deliberate stress cases, rather than examples chosen only because a method
succeeds on them.

For random wide networks there's already relevant analysis.
[Dunbar and Aaronson's wide-network baseline](https://arxiv.org/html/2510.06527v2) studies
decorrelation under conditions on inputs, scaling, and activations, using a perturbative
expansion. I'd like to isolate a concrete covariance recurrence and formalize the conditions
under which it contracts before trying to connect it to a finite-width model, because a
limit calculation needs a finite-width error argument before it becomes a quantitative
statement about a particular trained network.

## Competing with sampling on random ReLU networks

I ran a benchmark on the family MAISI points to, random wide ReLU networks. Each context is
one draw of the weights (entries with variance $2/\text{fan-in}$, biases with standard
deviation $0.1$, a random readout), and the target is the mean output over 64 independent
sign inputs, the same inputs as before. Every estimator gets the weights and nothing else,
so its advice is just its rule. I compared three structural estimators with plain sampling:

- covariance propagation, which carries the mean vector and covariance matrix through each
  layer and treats the preactivations as jointly Gaussian (the cumulant-propagation idea from
  Appendix D, adapted to ReLU);
- a diagonal version that keeps only each neuron's mean and variance;
- the conditional averaging from earlier, keeping 8 of the 64 signs, with its certified bound.

For sampling I used the exact error of a Monte Carlo average with the same number of
floating-point operations, and I measured every estimator against an estimate of each
network's true mean from 8 to 17 million samples. The grid has 96 settings (two or three
hidden layers, sign or Gaussian inputs, three weight scales, two bias scales, widths 16 to
1024), with 200 networks per setting and 100 at width 1024.

<figure>
  <picture>
    <source media="(max-width: 600px)" srcset="experiments/msp-error-vs-cost-mobile.svg">
    <img src="experiments/msp-error-vs-cost.svg" alt="Four panels, one per width: covariance propagation sits below the sampling line at every width, the diagonal closure sits below it at a tiny cost, and certified averaging sits above it">
  </picture>
  <figcaption>Two hidden layers, 64 sign inputs, weight variance 2/fan-in, bias scale 0.1. Error
    is the RMSE over 200 networks (100 at width 1024), relative to how much the true mean varies
    across networks. The dashed line is the exact error of sampling at each compute budget.
    <a href="experiments/results-msp.json">The full record</a> has every setting.</figcaption>
</figure>

| Width | Covariance | Sampling at the same compute | Diagonal | Averaging (8 kept) |
| --- | --- | --- | --- | --- |
| 16 | 2.50% | 12.5% | 4.48% | 63.4% |
| 64 | 0.721% | 7.21% | 3.82% | 63.4% |
| 256 | 0.359% | 4.44% | 3.53% | 63.3% |
| 1024 | 0.336% | 2.59% | 3.12% | 63.1% |

Covariance propagation beat sampling at equal compute in all 96 settings. Its error was
between 0.08 and 0.51 of sampling's, it beat ten times the sampling budget in 84 settings
and a hundred times in 13, and at width 256 sampling needs about 150 times the compute to
match it. The diagonal version costs about three samples and also beat sampling at its own
cost everywhere, but its error stalls around 3%, ten times the covariance error.

Two things surprised me. The covariance error stops improving with width once the input
dimension is fixed: from width 256 to 1024 it barely moves, from 0.359% to 0.336%. When I
added settings with 256 inputs, marked in the record as added after seeing these results, it
reached 0.077% at width 1024, and sampling needs about 1,400 times the compute to match
that. In this family the wide-network picture held only when the input dimension grew too.
And my own certified method lost badly! Its bound held on all 8,400 sign-input networks,
for both values of $k$, and on 576 small instances I checked exhaustively, but it was at least 122 times the actual
error every time, and as an estimator it was 2.6 to 21 times worse than sampling at its
cost. With 56 or more discarded signs, each first-layer neuron's worst-case perturbation is
about six standard deviations of its preactivation, so the boundary rule charges every
neuron as if it could switch, and that charge grows through the next layer. A worst-case
radius is the wrong tool for a sum of many small independent terms.

Per FLOP the structural estimate wins, but per GPU millisecond it doesn't yet. My covariance
propagation launches a few dozen small kernels from Python, and at equal wall time sampling
beat it at every width up to 256 and lost to it at 1024. I haven't fused the closure, so I'd
read the wall-time result as a statement about my implementation. Tail probabilities are
harder too. With a threshold three standard deviations out, the Gaussian closure's answer is
just $Q(3)$, and at width 16 the true tail was 2.9 to 4.2 times larger.

Which step of covariance propagation is the approximation? Lean answers part of that. For a
Gaussian preactivation $Z\sim\mathcal N(\mu,\sigma^2)$ with $\sigma>0$, I proved

$$\mathbb E[\operatorname{ReLU}(Z)]=\mu\,\Phi(\mu/\sigma)+\sigma\,\varphi(\mu/\sigma),\qquad
  \mathbb E[\operatorname{ReLU}(Z)^2]=(\mu^2+\sigma^2)\,\Phi(\mu/\sigma)+\mu\sigma\,\varphi(\mu/\sigma),$$

using TorchLean's own ReLU and Mathlib's Gaussian. That means the per-neuron moment rule is
exact whenever the input to that neuron really is Gaussian, and the estimator's error comes
from the closure, from treating a sum of ReLU outputs as Gaussian too. For the covariance
between two different ReLU outputs we'd need the bivariate formula, which I haven't proved
yet.

<details>
<summary>Read the Gaussian moment theorems</summary>
<pre data-lean-block="heuristic-gaussian"></pre>
</details>

This is still the easy case. These are random networks at initialization, where Gaussian
closure is expected to work, and nothing here is trained, adversarial or rare. But it gives
the matching sampling question a concrete benchmark with every cost counted, and it shows
where a checked argument helps today and where it's still too loose to matter.

## Numerical error is another part of the answer

An estimator can implement its chosen rules perfectly and still make a bad statistical
approximation. Or it can choose sound rules and accumulate enough numerical error to spoil
the result. TorchLean and FloatLib let us treat these as separate questions.

Let $m$ be the true expectation, $a$ the ideal real-arithmetic estimator, and
$\widehat a$ its computed value. If we can establish

$$|a-m|\leq\delta_{\mathrm{model}},\qquad
  |\widehat a-a|\leq\delta_{\mathrm{num}},$$

then the triangle inequality gives

$$|\widehat a-m|\leq\delta_{\mathrm{model}}+\delta_{\mathrm{num}}.$$

The first term needs probabilistic or approximation analysis. For the second we can use
arithmetic contracts, including the accumulation order and the range assumptions required by
the rounding bounds. Increasing precision can improve the numerical term, but it can't
recover the covariance we threw away on purpose. An exact moment identity can also lose
accuracy when we implement it with a long, poorly conditioned floating-point reduction.

In the first finite examples, rational tensors let us avoid rounding while we check the
mathematics, and then we can round a sketch's coefficients on purpose and track what
changes. Let $\Delta>0$ be a rational grid spacing, and let my encoder apply FloatLib's
executable nearest-even integer rounding to $x/\Delta$ before multiplying by $\Delta$:

$$Q_\Delta(x)=\Delta\,\operatorname{roundEven}(x/\Delta),\qquad
  |Q_\Delta(x)-x|\leq\Delta/2.$$

For example, storing $(1/3,1/7)$ on a grid of eighths gives $(3/8,1/8)$:

<pre data-lean-example="encodedSketch"></pre>

The next output is the exact squared encoding error, and the final one is the general bound
$d(\Delta/2)^2$, here $1/128$. FloatLib supplies the half-unit bound for its integer
rounding (`roundRatEven_error_le_half`), which I scale by $\Delta$ to get the half-spacing
theorem `round_error`, and the tensor proof sums it over coordinates. This is a rational
grid encoder, with no finite exponent range or packed byte representation, and a fixed grid
spacing alone doesn't specify how many bits its unbounded integer coefficients take to
store.

We can carry that error through the same downstream certificate. If $r$ is the ideal
sketch residual and $e$ the coefficient-rounding residual, then

$$ (r+e)^{\mathsf T}K(r+e)
   \leq 2\lambda\left(\|r\|_2^2+d(\Delta/2)^2\right).$$

`encoded_error_le` proves this bound without assuming the two residuals are independent or
orthogonal, and the factor two is what covers arbitrary $r$ and $e$. A coordinate sketch
doesn't need it, because $r$ lives on the discarded coordinates and $e$ on the retained
ones, so $\|r+e\|_2^2=\|r\|_2^2+\|e\|_2^2$, although the theorem states only the general
version. In an orthonormal coordinate basis coefficient error has this same Euclidean
interpretation, while a different basis would need its own norm factor.

<pre data-lean-block="heuristic-encoding"></pre>

That leaves a numerical trade-off to investigate: retain more coefficients, increase their
precision, or spend more work on residual sampling. Proving the reference arithmetic doesn't
verify a foreign GPU kernel, though, so a GPU result would need a connection to the kernel's
behavior or would have to keep that backend assumption explicit.

I don't want Lean's kernel to evaluate millions of tensor entries every time we want an
estimate. A general theorem can establish how a local error rule composes, and an instance
then supplies the premises and any certificates that must be checked. How much work that
saves depends on the certificate format. I haven't measured it yet.

## An average is not yet a safety guarantee

If our observable is the indicator of a precisely defined failure event, its expectation is
the event's probability under the specified input distribution. But an estimate of the mean
score, loss, or activation doesn't determine a rare-event probability, because distributions
with the same mean and variance can have different tails. A tail bound needs its own theorem
and premises. A point estimate isn't an upper bound.

The distribution is a substantive assumption too. A result averaged over one input law
doesn't establish worst-case robustness, and it doesn't automatically cover a deployment
process that changes that law. I'd like these distinctions to be visible in the statement
itself instead of implicit in an experiment script.

ARC's no-coincidence conjecture asks for a verifier with completeness and a soundness
condition for random circuits. It doesn't ask for universal soundness for every accepted
argument. My Lean proofs are ordinary conditional mathematical theorems, so checking them
doesn't establish that complexity-theoretic conjecture. What I want to pursue is a more
specific connection: use executable formalizations to investigate candidate estimators and
make their assumptions and failures easier to examine.

## What I want to build from this

The attention calculation and finite sketch give us a common problem to work on: one tensor
program, one input distribution, competing summaries, and exact small-instance answers. The
benchmark against sampling already picked the next theorem for me: a certified bound for
averaging that uses concentration instead of the worst case, so that it can say something
about networks like these. I'd like the experiments and the formalization to keep developing
together like that, so that a failed approximation helps us choose the next theorem to prove
and a theorem tells us which assumptions to check in the next run.

What I'd find most exciting is an estimator that keeps improving its explanation. It might
begin with a cheap summary, identify a discarded interaction that could affect the answer,
refine that part of the graph, and return a tighter interval when the new premises justify
one. An algorithm could propose those refinements while Lean checks the mathematical
implications of the accepted steps. Choosing useful refinements cheaply and obtaining the
premises without solving the original problem are the central research questions, and I
don't have answers to them yet.

That's the contribution I'd like to explore with people working on this agenda. I'd be glad
to work with MAISI or ARC researchers on a concrete family where a formally specified
structural estimator would teach us something new. We'll be adding more examples as we
develop this direction :)