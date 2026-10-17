# Estimating model behavior from its structure

There's a new institute I'd like to talk about: the
[Mathematical AI Safety Institute, or MAISI](https://maisi.org/). It is bringing mathematicians
to questions about the safety of powerful AI systems. One of its research directions,
[developing good heuristics](https://maisi.org/research/directions/heuristic-estimators),
caught my attention because it asks how much we can learn from a model's structure.

We usually learn what a model does by running it. Pick some inputs, collect outputs, and
estimate a quantity we care about: an average loss, an attention score, or the frequency
of a defined failure. But we also have the program and its weights. It seems worth asking
how much of that evaluation we can replace with a calculation about the model itself.
The Alignment Research Center (ARC) has been developing this question through its work
on mechanistic estimation and the matching sampling principle.

I wanted to work through it using the tools we've been building. In the earlier posts,
we [trained a GPT](../week-03-gpt-training/),
[worked through K3's equations](../week-04-kimi-k3-specification/site/), and
[followed rounding through Nemotron's recurrent state](../week-05-nemotron-specification/site/).
Here we'll use TorchLean's tensor calculations to study an estimator, and Lean to check
what its answer means. FloatLib becomes relevant when we store and execute the numerical
summary. I want to understand which dependencies we can afford to forget, and what we
can prove about the error when we do.

Let's begin with a calculation small enough to understand completely. We'll deliberately
throw away a dependency, recover it with covariance, and then look at a harder question:
which dependencies are worth keeping in a large model? I think that last question is where
there's room for an interesting research project. The Lean development accompanying this
post now includes the query–key expectation theorem, a finite sketch with an executable
error certificate, a coefficient-rounding bound using FloatLib, and a sampling theorem.
It also includes a graph-wide error identity and a conditional-averaging theorem, which we'll
use to study when we can remove some input randomness. We'll run the small instances and
follow the general arguments. A useful automatic estimator for trained models is the larger
project these pieces are meant to support.

## Two branches, one input

Suppose that two independent random signs, $X$ and $Z$, are each equally likely to be $-1$
or $1$. We form a second branch that partly reuses $X$:

$$Y_c = cX + (1-c)Z.$$

The parameter $c$ controls how much input the branches share. At $c=0$, the branches are
independent. At $c=1$, they are the same random variable. Now multiply them:

$$F_c = XY_c = X\bigl(cX+(1-c)Z\bigr).$$

Both branches have mean zero. An estimator that remembers only their means, and multiplies
those means at the last operation, returns zero for every $c$. But $X^2=1$, so

$$F_c = c+(1-c)XZ, \qquad \mathbb E[F_c]=c.$$

At $c=1$, the estimator predicts zero for a program that returns one on every input. Nothing
unusual happened during execution. The mistake was treating two uses of the same input as
independent information.

Here's the calculation with tensors. The four rows enumerate every pair of signs, with
equal weight; we're using rational arithmetic rather than a random sample.

<pre data-lean-example="sharedExpectation"></pre>

For $c=1/2$, the outputs are $[1,0,0,1]$. Their mean is $1/2$. The last command instead
multiplies the two branch means and returns zero. We used the same tensor operations in
both calculations; the difference is when we discarded information.

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

The variance is $(1-c)^2$, since $XZ$ is another mean-zero sign. This gives a useful comparison
with sampling. Averaging $n$ independent evaluations has expected squared error
$(1-c)^2/n$. The mean-only estimate has squared error $c^2$, however many times we repeat
its calculation. A cheap estimate can be confidently wrong.

Below are the tensor definition and the two checked identities. `response` works for any
shape and any scalar type providing the operations it uses. The mean and variance theorems
specialize to real arithmetic and the four equally weighted sign pairs. They quantify over
every real `shared`, not just the slider positions.

<pre data-lean-block="heuristic-response"></pre>

We can execute the same definition with rationals, native floats, or supported FloatLib
formats. Changing the scalar type changes the arithmetic. The real-valued theorem alone
doesn't assert that all those executions have identical answers.

## What covariance buys us

The correction is familiar:

$$\mathbb E[XY] = \mathbb E[X]\mathbb E[Y] + \operatorname{Cov}(X,Y).$$

For our two branches, $\operatorname{Cov}(X,Y_c)=c$. Keeping that one number recovers the
answer. We haven't needed a Gaussian approximation or an independence assumption. With
square-integrable real random variables on a probability space, the identity is exact.

Mathlib already has this result in its covariance API. The accompanying Lean statement
puts it in the form we want for a multiplication rule:

<pre data-lean-block="heuristic-covariance"></pre>

The hypotheses `MemLp x 2 μ` and `MemLp y 2 μ` say that the variables have finite second
moments, including the required measurability conditions. They justify the integrals in the
statement. We should reuse that probability theory rather than introduce an unrelated
meaning of expectation inside a tensor estimator.

ARC's [*Formalizing the presumption of independence*](https://arxiv.org/html/2211.06738v1)
studies mean, covariance, and cumulant propagation in Appendix D. In particular, covariance
repairs the product-mean rule, but propagating products creates demands for higher moments.
Approximating the intermediate variables as jointly Gaussian supplies a closure rule: we
fill in information that we did not retain. The shared-input problem above is an illustration
of this known difficulty, not a new error we found in their work.

We can see the next difficulty without a network. Let $A$ be a uniform sign. Let $B$ equal
$-\sqrt2$ or $\sqrt2$, each with probability $1/4$, and equal zero with probability $1/2$.
Both have mean zero and second moment one. But

$$\mathbb E[A^4]=1, \qquad \mathbb E[B^4]=2.$$

Any summary containing only the first two moments gives the same description of $A$ and
$B$. If the remaining program computes the fourth power, the correct answers differ.
The Lean counterexample represents $A$ by $(-1,-1,1,1)$ and $B$ by
$(-\sqrt2,0,0,\sqrt2)$, with four equally weighted rows each. It checks their means,
second moments, and fourth moments together. Adding covariance therefore cannot be the
whole solution to a nonlinear continuation.

<details>
<summary>Check the matching-moments example in Lean</summary>
<pre data-lean-block="heuristic-higher-moments"></pre>
</details>

I want an implementation to tell us when it used an exact identity and when it used a
closure assumption. For an affine layer, the usual mean and covariance transforms are
exact when the input moments exist:

$$\mu_{Wx+b}=W\mu_x+b, \qquad
  \Sigma_{Wx+b}=W\Sigma_xW^{\mathsf T}.$$

We prove these using the same matrix–vector and addition operations as the tensor
program. The covariance theorem allows two different affine projections of one input,
giving $W_q\Sigma_x W_k^{\mathsf T}$ for their cross-covariance. Taking both maps equal
recovers the formula above. This shared-input case is the one we'll need for attention;
we haven't assumed that the input coordinates or the two projections are independent.

<details>
<summary>Read the affine moment theorems</summary>
<pre data-lean-block="heuristic-affine"></pre>
</details>

A nonlinear layer needs more information or an approximation. A residual addition needs
the cross-covariance between the two branches. An attention score multiplies queries and
keys that may depend on the same input. Treating those operations as independent because
they occupy separate nodes would recreate our four-row mistake at a much larger scale.

## Keep what the next computation needs

Storing every pairwise covariance quickly becomes expensive. Higher-order interactions cost
more. I'd like to avoid choosing between a mean-only summary that loses the answer and a
complete distribution that we cannot afford to carry.

A more useful question is what the *rest of the program* will do with an intermediate.
Suppose a partial computation produces $g(x)$. We replace it by a cheaper sketch
$\widetilde g(x)$, then multiply by a later function $h(x)$. The error in the final expectation is

$$\mathbb E[h g]-\mathbb E[h\widetilde g]
  =\mathbb E\bigl[h(g-\widetilde g)\bigr].$$

The discarded residual $g-\widetilde g$ matters through its interaction with $h$.
Controlling only its mean doesn't control that interaction. For example, take
$g(X)=X$, replace it by its mean zero, and let $h(X)=X$. Before multiplication the sketch
has exactly the right mean. After multiplication, the true mean is one and the sketched
mean is zero.

<pre data-lean-example="discardedSignal"></pre>

This general error identity is also checked in Lean. Its two integrability assumptions
ensure that subtracting the integrals is legitimate:

<pre data-lean-block="heuristic-projection"></pre>

ARC's [work on expectations of random products](https://www.alignment.org/blog/mechanistic-estimation-for-expectations-of-random-products/)
develops this downstream-sensitive view. Their deduction–projection estimators alternate
computation with simplification. The sketch should retain information useful for pairing
the computed head of a product with its remaining tail. They study a second-moment operator
for the tail family and project onto leading eigendirections. Symmetry makes some instances
tractable; their examples include random halfspace intersections, counting problems, and
permanents. This is a richer starting point than applying the same moment cutoff everywhere.

We can work through [Hilton's sketching argument (§3)](https://www.alignment.org/content/files/2026/05/deductive_estimation_for_random_halfspace_intersections.pdf#page=3)
with finite tensors. Give each proposed projection an explicit account of the future
computations it needs to preserve. For a finite family of tails with probabilities $p_c$,
define the residual $r=g-\widetilde g$. Then

$$\sum_c p_c\bigl(\mathbb E[h_c r]\bigr)^2$$

measures how much that projection changes the expected answers across contexts. On a
finite input space, write the distribution-weighted pairing as a vector dot product. The
expression becomes $r^{\mathsf T}Kr$ for a positive-semidefinite matrix
$K=\sum_c p_c h_c h_c^{\mathsf T}$, with the input weights absorbed into the vectors.
Keeping leading eigendirections is then a concrete linear-algebra problem. The development
proves the Gram identity for arbitrary finite dimensions, not only for the numbers below.
Expanding the square and exchanging finite sums gives the proof:

<pre data-lean-block="heuristic-gram"></pre>

Let's make the choice of sketch consequential. Our intermediate vector is $g=(100,1)$,
and we can keep only one coordinate. Two equally likely continuations read its second
coordinate with weights one and two. Keeping the first coordinate minimizes reconstruction
error: we lose only one unit. But both continuations ignore that large first coordinate.
Keeping the second gives the exact downstream answers, although its reconstruction error
is much larger.

<pre data-lean-example="sensitiveSketch"></pre>

Each of the first two outputs pairs squared reconstruction error with squared downstream
error. Keeping the largest coefficient gives $(1,5/2)$; keeping the useful coefficient
gives $(10000,0)$. For this task, a reconstruction metric would choose the wrong coordinate.
The Gram matrix explains the difference:

$$K=\begin{pmatrix}0&0\\0&5/2\end{pmatrix}.$$

We also have a checker for a uniform error bound. For a symmetric matrix, it verifies

$$\sum_j |K_{ij}|\leq\lambda\quad\text{for every row }i.$$

Lean proves that this implies $r^{\mathsf T}Kr\leq\lambda\|r\|_2^2$ for every residual.
The proof bounds each cross-term using $2|r_i r_j|\leq r_i^2+r_j^2$, sums over the
matrix, and uses symmetry to combine the row and column sums. The executable check uses
exact rationals. Its soundness theorem also covers the interpreted matrix acting on
arbitrary real residuals; we don't have to enumerate those vectors.

<pre data-lean-block="heuristic-certificate"></pre>

In the example, the claimed bound two is rejected and $5/2$ is accepted. After retaining
the useful coordinate, `restrict` zeros its row and column in the remaining-error matrix.
That matrix is zero, and a zero bound passes. This proves zero downstream error for
*every* intermediate vector under those continuations, not merely for $(100,1)$.

<pre data-lean-block="heuristic-refinement"></pre>

The certificate covers coordinate sketches. We can inspect a candidate without trusting
how it was chosen, but the check doesn't discover a good candidate for us. An absolute
row-sum bound can also overestimate the largest eigenvalue. More economical certificates,
arbitrary subspaces, and matrices available only implicitly remain interesting problems.

Even here we can separate two jobs in adaptive refinement. An algorithm proposes which
coordinate to retain; the checker verifies its claimed remaining bound. Retaining more
coordinates decreases the residual's squared norm, as proved by `residual_energy_mono`.
Actual downstream error need not decrease at every step because discarded terms can
cancel. A decreasing certified budget is the property we can justify without assuming
away those cross-terms.

## One model definition, several analyses

In the earlier articles, we've used TorchLean to connect model equations to typed graphs,
and to reason about arithmetic and recurrent state. I want to reuse those definitions here.
An estimator written against a separate, manually transcribed network would create another
place for the model and its analysis to disagree.

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
token, or the probability that a defined check fails. Those are different functions of the
model output. A good approximation to the mean hidden activation need not answer any of
them accurately after the remaining nonlinear operations.

The composition theorem now works on TorchLean's existing proof-carrying autograd graph,
including shared branches. It compares a full execution with a reference execution and
uses the reference backward pass to account for each node's contribution to the output
error. We haven't introduced another graph representation or another differentiation engine.
The identity is exact; the useful approximation comes from bounding its remainder terms.

The hard part is obtaining useful residual bounds cheaply. A theorem with a hypothesis
saying that every approximation is accurate doesn't solve that problem. It gives us a
place to connect a later analytic result, a finite certificate, or a deliberately labeled
heuristic assumption. We should be able to inspect which of those supplied each bound.

Sharing also belongs in that representation. If two branches use the same node, we must
retain that fact. If we rewrite a real-valued polynomial graph into an equal polynomial,
its true expectation stays the same. An approximate estimator may change its answer
because the rewrite changes what it discards. Running both forms is a useful diagnostic:
we can prove equality of the targets, then investigate disagreement between estimates.
It doesn't prove one estimate correct, but it identifies sensitivity to representation.

The backward pass supplies the sensitivity weights, but a derivative at one activation
doesn't bound a large random change. We must also bound what the local linearization misses.
For ReLU networks, we can identify that missing contribution particularly clearly.

## When can we average away an input?

Suppose we keep part of a random input and replace the rest by its mean. I'd like to know
when that preserves the model's expected output, even if it changes individual predictions.
A linear layer is easy: averaging commutes with its calculation. At a ReLU, we need to
consider whether the removed contribution can switch the neuron on or off.

Write a scalar preactivation as $z+u$, where $z$ is the retained contribution. Subtract the
branch's linear response and call the remainder $r$:

$$r(z,u)=\operatorname{ReLU}(z+u)-\operatorname{ReLU}(z)
  -\mathbf 1_{z>0}\,u.$$

If $|u|\leq\delta$, then

$$0\leq r(z,u)\leq\max(0,\delta-|z|).$$

Far enough from zero, the chosen branch is affine and the remainder vanishes. Near zero,
we charge only the distance that an allowed perturbation could cross the boundary. This
is tighter than charging the whole perturbation everywhere. The Lean theorem uses
TorchLean's actual scalar ReLU, including the zero-subgradient convention used by its JVP.
A separate tensor theorem identifies the same remainder after subtracting that existing JVP.

<pre data-lean-block="heuristic-boundary"></pre>

For a whole graph, let $x$ be the full input and $\widetilde x$ the reference input. With a
fixed linear output readout, the comparison has the form

$$F(x)-F(\widetilde x)
  =\langle g_{\mathrm{input}}(\widetilde x),x-\widetilde x\rangle
    +\sum_v\langle g_v(\widetilde x),r_v(x,\widetilde x)\rangle.$$

The $g_v$ are the cotangents from the reference backward pass. The local remainders $r_v$
account for everything its linearization misses. `Composition.eval_difference` proves
this identity by induction over the existing graph. Its algebraic statement also permits
seeds on intermediate nodes; it does not assume that an arbitrary supplied JVP is an
analytical derivative.

Weighting local errors by their effect on a chosen output has a long history in
[goal-oriented error estimation](https://www.cambridge.org/core/journals/acta-numerica/article/abs/an-optimal-control-approach-to-a-posteriori-error-estimation-in-finite-element-methods/5C67A03F528C6FA69F37A97DF5C3BE19).
Here we use the network's existing VJPs for those weights and keep the nonlinear remainders
explicit. The contribution we're working toward is a checked, computable estimator for
model behavior, rather than a claim that sensitivity-weighted error analysis is new.

<details>
<summary>Read the graph discrepancy theorem</summary>
<pre data-lean-block="heuristic-graph-remainder"></pre>
</details>

Now write the input as $(Z,U)$ and use $(Z,0)$ as the reference. If
$\mathbb E[U\mid Z]=0$, the input-linear term disappears in expectation: the reference
backward pass depends only on $Z$, while the discarded perturbation has conditional mean
zero. Independent centered input coordinates satisfy this condition. Mean zero or zero
covariance alone does not.

For a finite conditional input law, `Averaging.mean_error_le` proves the resulting bound
from coordinatewise certificates for every node's remainder. Its right-hand side evaluates
only reference states and their VJPs. For a ReLU, the radius above supplies the local rule
once we have bounded its preactivation change. Later hidden-layer perturbations need not
be centered: we bound their remainders pointwise and cancel only the original input term.

<details>
<summary>Read the conditional-averaging theorem</summary>
<pre data-lean-block="heuristic-averaging"></pre>
</details>

We tried this on constructed three-layer networks with 64 independent sign inputs.
Keeping four signs leaves 16 reference states to evaluate. One network had stable
activation branches throughout the full input cube: the budget was zero, and an independent
affine calculation recovered the same expectation. All 60 discarded signs had nonzero
weights into active first-layer neurons, so they could still change individual outputs.
In a nonlinear example, the first readout's actual mean error was about $0.000405$, within
a conservative bound of $0.195$. Its repeated input weights made a separate exact binomial
calculation possible. We also checked a smaller network exhaustively and a correlated-input
counterexample that violates conditional centering. The
[experiment record](experiments/results-boundary.json) contains the exact fractions.

These instances use exact rational Python calculations. The general graph and averaging
theorems are checked in Lean; the experiment does not emit proof terms for those networks.
We still need operation-specific radius propagation and an executable certificate generator
to make this automatic. Smooth activations need curvature bounds, and multiplication needs
its cross-perturbation term. The same graph identity gives us somewhere to connect those
rules, but the ReLU result doesn't establish them.

## An attention score we can calculate exactly

An attention head gives us a useful next target. Let $q(x),k(x)\in\mathbb R^d$ be a query
and key computed from the same random input, and consider their unscaled score $q(x)^{\mathsf T}k(x)$.
When the coordinates have finite second moments,

$$\mathbb E[q^{\mathsf T}k]
  =\mu_q^{\mathsf T}\mu_k+\operatorname{tr}\bigl(\operatorname{Cov}(q,k)\bigr).$$

This is our two-branch example with vectors. It gives us a direct reason to preserve
cross-covariances between the query and key projections. Sampling the two branches
independently would change the distribution and remove precisely the dependency we want
to study. We can write their dot product with the same tensor contractions used in the
[attention example](../torchlean-tensor-programming/#the-two-contractions-inside-attention).

Here are four query–key pairs. Each key scales the first query coordinate by $1/2$ and
the second by $1/4$. Both marginal means vanish. Every dot product is $3/4$, and the
cross-covariance diagonal contains precisely the missing $1/2$ and $1/4$:

<pre data-lean-example="attentionMoments"></pre>

The first output averages the actual scores. The second multiplies the marginal means
and loses the answer. The third evaluates the covariance-aware rule and recovers $3/4$.
`score_mean` proves the same expectation identity for any finite dimension and any query
and key distribution whose coordinates have finite second moments:

<pre data-lean-block="heuristic-attention"></pre>

Supplying the true moments is a substantial premise. This theorem tells us what we can
calculate once we have them; it doesn't make them free to obtain from a trained model.
Affine projections offer one useful case, since their input means and cross-covariances
transform by exact matrix formulas. Beyond that, we need an approximation rule and an
account of the information it loses.

We can also compare a full score with one computed after projecting both vectors into a
smaller subspace. For an orthogonal projector $P$, the pointwise score difference is

$$q^{\mathsf T}k-(Pq)^{\mathsf T}(Pk)=q^{\mathsf T}(I-P)k.$$

The tensor theorem states the right side as the score against $k-Pk$. It requires
$P^{\mathsf T}=P$ and $P^2=P$, and uses TorchLean's existing matrix–vector operations:

<pre data-lean-block="heuristic-attention-projection"></pre>

That gives the projection rule an explicit target. Which subspace keeps the expected score
accurate, and how much does it cost to identify it? Moving to the expected *squared* score
would require fourth-order information, assuming the relevant fourth moments exist.
A sketch adequate for the first question might fail on the second.

We followed this with attention states from our trained 124-million-parameter GPT.
TorchLean replayed 3,072 first-block head queries. Keeping half the query–key coordinates
with an output-aware ordering gave 24.1% relative attention-output error, compared with
38.9% for the seeded random ordering. Packing the retained coordinates cost more than
the smaller contraction saved. A later sweep covered layers 1, 6, and 12 at histories
of 64, 128, and 512 tokens; the quality of the summary depended strongly on both.

Those runs compare attention outputs on fixed supplied activations, before the output
projection. They don't yet construct an expectation estimator from the model and input
law. They do give us larger computations on which to test the proposed summaries and
their costs. The [experiment report](experiments/RESULTS.md) includes the exact scope,
the softmax error certificates, and follow-up checks on a small transformer and a trained
digit classifier.

## Sampling the part we cannot explain

We don't have to discard sampling while developing this. Suppose we have a cheap surrogate
$S(x)$ whose expectation $m_S$ is known, and a target $T(x)$. For independent samples
$X_1,\ldots,X_n$ from the target distribution, consider

$$\widehat m = m_S + \frac1n\sum_{i=1}^n\bigl(T(X_i)-S(X_i)\bigr).$$

If $T$ and $S$ are integrable and $m_S=\mathbb E[S]$, this estimate is unbiased. The
expectation identity is short enough to show in full:

<pre data-lean-block="heuristic-residual"></pre>

Under finite second moments, its variance is $\operatorname{Var}(T-S)/n$. With a sufficiently
correlated surrogate, the residual can be much easier to estimate than the original
output. A poor surrogate can make the variance worse. If the surrogate mean is approximate,
its error becomes a bias; residual sampling doesn't remove it.

We can make that last sentence quantitative. Write $\delta=\widetilde m_S-\mathbb E[S]$
for the error in the surrogate mean we supply. The checked sampling theorem gives

$$\mathbb E\!\left[(\widehat m-\mathbb E[T])^2\right]
  =\frac{\operatorname{Var}(T-S)}{n}+\delta^2.$$

It uses `Tensor.mean` on the residual samples and mathlib's probability definitions.
Pairwise zero covariance between sample residuals is enough for the calculation;
independence implies that condition. These are relationships between separate samples,
not a permission to erase dependencies inside the network.

<pre data-lean-block="heuristic-sampling"></pre>

The averaging construction gives us one way to obtain the surrogate mean: sum over its
retained states. Its boundary certificate bounds the bias if we stop there. If we also
sample the residual, the theorem above describes the correction. A small mean error
doesn't imply a small residual variance, though. Even in a stable affine region, a discarded
mean-zero input can cause large fluctuations that cancel only after averaging.

This suggests a practical intermediate experiment. Let a structural estimator construct a
surrogate with a tractable expectation. Measure whether it reduces the residual variance,
then include the cost of constructing it and evaluating it alongside the target. If we fit
the surrogate using a pilot sample, we'd fix it before drawing the independent correction
sample. Otherwise the elementary fixed-surrogate argument above needs a different analysis.

This hybrid is not automatically a mechanistic estimator in ARC's sense. The final answer
still uses random observations. I'd use it as a baseline and as a way to find which
dependencies are worth modeling, while keeping the purely structural question separate.
It also offers a useful failure report: we can show whether the approximation went wrong
because it missed structure, because its residual estimate was noisy, or because its
known-mean premise was inaccurate.

## What competing with sampling would require

ARC's [*Competing with sampling*](https://www.alignment.org/blog/competing-with-sampling/)
motivates structural estimation partly through events too rare to encounter in ordinary
evaluation. Its matching sampling formulation introduces a context $c$ and advice $\pi$
describing a model's structure. Requiring the same advice across contexts prevents the
advice from merely storing each answer. The comparison is mean squared error averaged
over contexts, with an execution budget comparable to sampling. Matching this general
benchmark and substantially beating it on useful structured models are different goals.

Our four-sign family has a visible context parameter, but proving its mean is not evidence
for the general principle. We can enumerate its entire input space with four evaluations.
Its purpose is to expose a dependency and make the target unambiguous. A serious benchmark
would need larger instances, a specified context distribution, a fixed advice construction,
and an accounting of the work needed to obtain that advice.

I'd report error against total cost, including preprocessing, memory, and any samples used
to choose a sketch. For a reusable explanation, we'd show the number of contexts over
which preprocessing is amortized. We'd also separate time spent finding a candidate from
time spent checking a claimed bound. A fast checker with an impossibly expensive candidate
search may be mathematically interesting, but it hasn't yet given us an efficient estimator.

The comparison should include plain sampling, mean propagation, a covariance-based method,
and a structure-aware sketch at comparable cost. Tiny exact instances would supply ground
truth. Larger experiments could use independent high-budget estimates with their own
uncertainty reported. Rewrites, shared branches, and distributions with matching low-order
moments would be deliberate stress cases, rather than examples chosen only because a
method succeeds on them.

For random wide networks, there is already relevant analysis to engage with.
[Dunbar and Aaronson's wide-network baseline](https://arxiv.org/html/2510.06527v2)
studies decorrelation with conditions on inputs, scaling, and activations, using a
perturbative expansion. I would like to isolate a concrete covariance recurrence and
formalize the conditions under which it contracts, before trying to connect it to a
finite-width model. A limit calculation needs a finite-width error argument before we
can use it as a quantitative statement about a particular trained network.

## Numerical error is another part of the answer

An estimator can implement its chosen rules perfectly and still make a bad statistical
approximation. It can also choose sound rules and accumulate enough numerical error to
spoil the result. TorchLean and FloatLib give us a way to ask both questions without
pretending they are the same question.

Let $m$ be the true expectation, $a$ the ideal real-arithmetic estimator, and
$\widehat a$ its computed value. If we can establish

$$|a-m|\leq\delta_{\mathrm{model}},\qquad
  |\widehat a-a|\leq\delta_{\mathrm{num}},$$

then the triangle inequality gives

$$|\widehat a-m|\leq\delta_{\mathrm{model}}+\delta_{\mathrm{num}}.$$

The first term needs probabilistic or approximation analysis. The second can use arithmetic
contracts, including the accumulation order and the range assumptions required by the
rounding bounds. Increasing precision can improve the numerical term; it cannot recover
the covariance we deliberately threw away. Conversely, an exact moment identity can lose
accuracy when implemented with a long, poorly conditioned floating-point reduction.

For the first finite examples, rational tensors let us avoid rounding while checking the
mathematics. We can then round a sketch's coefficients deliberately and keep track of
what changes. Let $\Delta>0$ be a rational grid spacing. The encoder uses FloatLib's
executable nearest-even integer rounding on $x/\Delta$, then multiplies by $\Delta$:

$$Q_\Delta(x)=\Delta\,\operatorname{roundEven}(x/\Delta),\qquad
  |Q_\Delta(x)-x|\leq\Delta/2.$$

For example, storing $(1/3,1/7)$ on a grid of eighths gives $(3/8,1/8)$:

<pre data-lean-example="encodedSketch"></pre>

The next output is the exact squared encoding error. The final one is the general bound
$d(\Delta/2)^2$, here $1/128$. FloatLib supplies the scalar half-spacing theorem; the
tensor proof sums it over coordinates. This is a rational grid encoder, with no finite
exponent range or packed byte representation. A fixed grid spacing alone doesn't specify
how many bits its unbounded integer coefficients take to store.

We can carry that error through the same downstream certificate. If $r$ is the ideal
sketch residual and $e$ is the coefficient-rounding residual, then

$$ (r+e)^{\mathsf T}K(r+e)
   \leq 2\lambda\left(\|r\|_2^2+d(\Delta/2)^2\right).$$

`encoded_error_le` proves this bound without assuming the two residuals are independent
or orthogonal. The factor two pays for their possible cross-term. In an orthonormal
coordinate basis, coefficient error has this same Euclidean interpretation; a different
basis would need its own norm factor.

<pre data-lean-block="heuristic-encoding"></pre>

This gives us a numerical trade-off to investigate: retain more coefficients, increase
their precision, or spend more work on residual sampling. Proving the reference arithmetic
does not verify a foreign GPU kernel. A GPU result would need a connection to its behavior,
or retain that backend assumption explicitly.

The aim isn't to ask Lean's kernel to evaluate millions of tensor entries every time we
want an estimate. A general theorem can establish how a local error rule composes. An
instance then supplies the premises and any certificates that must be checked. How much
work that saves depends on the certificate format; it is something we would measure.

## An average is not yet a safety guarantee

If our observable is the indicator of a precisely defined failure event, its expectation
is the event's probability under the specified input distribution. But an estimate of the
mean score, loss, or activation doesn't determine a rare-event probability. Distributions
with the same mean and variance can have different tails. A tail bound needs its own
theorem and premises; a point estimate is not an upper bound.

The distribution is a substantive assumption too. A result averaged over one input law
doesn't establish worst-case robustness, or automatically cover a deployment process that
changes that law. I want these distinctions visible in the statement we show a reader,
not left implicit in an experiment script.

MAISI also discusses ARC's computational no-coincidence conjecture. Its proposed verifier
has completeness and a random-circuit soundness condition, not universal soundness for
every accepted argument. Our Lean proofs are ordinary conditional mathematical theorems.
Checking them doesn't establish that complexity-theoretic conjecture. The connection I
want to pursue is more specific: use executable formalizations to investigate candidate
estimators and make their assumptions and failures easier to examine.

## What I want to build from this

The attention calculation and finite sketch give us a common problem to work on: one
tensor program, one input distribution, competing summaries, and exact small-instance
answers. We now have proofs for the expectation, projection error, finite error certificate,
sampling correction, coefficient encoding, and conditional averaging across an autograd
graph. I'd want the next experiments and the formalization to develop together. A failed
approximation should help us choose the next
theorem to prove; a theorem should
tell us which assumptions to check in the next run.

What I'd find most exciting is an estimator that can keep improving its explanation.
It might begin with a cheap summary, identify a discarded interaction that could affect
the answer, refine that part of the graph, and return a tighter interval when the new
premises justify one. An algorithm could propose those refinements; Lean could check the
mathematical implications of the accepted steps. Choosing useful refinements cheaply,
and obtaining the premises without solving the original problem, remain the central
research questions.

That's the contribution I'd like to explore with people working on this agenda. I'd be
glad to work with MAISI or ARC researchers to choose a concrete family where a formally
specified structural estimator would teach us something
new. We'll be adding more examples as we develop this direction :)
