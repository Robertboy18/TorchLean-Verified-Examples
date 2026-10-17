# Estimating model behavior from its structure

The [article](../site/articles/week-06-heuristic-estimators.md) explores how TorchLean could
support the heuristic-estimation research described by MAISI and ARC.

The development uses TorchLean tensors, mathlib probability, and FloatLib rounding:

- `Basic` follows a shared input through two branches and proves its mean and variance.
- `Attention` proves the query–key expectation identity and orthogonal-projection error.
- `Moments` proves affine mean/cross-covariance transforms and a counterexample to
  determining fourth moments from only means and variances.
- `Sketch` relates downstream squared error to a Gram matrix. Its exact row-sum checker
  certifies uniform bounds, including for real residuals, and checks coordinate sketches.
- `Sampling` proves the residual estimator's mean squared error, including surrogate bias.
- `Encoding` bounds coefficient rounding and combines it with a sketch's downstream error.
- `Softmax` carries a uniform score error through attention's softmax-weighted output.
  Its bound has no sequence-length factor and can remove a common score shift first.
- `Boundary` bounds the error left after a ReLU's linear contribution, using the distance
  to its activation boundary. Its tensor theorem uses TorchLean's existing ReLU JVP.
- `Composition` proves an exact discrepancy identity on TorchLean's existing autograd graph.
  Local remainder certificates give a budget evaluated with the reference execution and VJPs.
- `Averaging` proves that conditional centering of the discarded input cancels the input-linear
  term. The resulting expected-error budget evaluates only retained states.

The [experiments](experiments/RESULTS.md) include the original 3,072 attention-head queries,
a nine-setting layer/history sweep, the small sparsemax transformer's finite task, and pixel
removal and parameter quantization in a trained digit classifier. Fitting and evaluation
remain separate. These results do not establish the matching sampling principle or a
measured speedup over sampling.

The [boundary experiment](experiments/results-boundary.json) uses constructed three-layer
ReLU networks with independent sign inputs. With 64 inputs, it evaluates 16 retained states
and checks the expectation against independent affine or binomial calculations. These are
exact rational Python checks, separate from the general Lean proofs. The nonlinear example's
bound is conservative. A correlated counterexample demonstrates why marginal mean zero
does not justify averaging away an input.

The graph theorem requires proved local remainder bounds for every node. We have the ReLU
rule and a general composition theorem; automatic radius propagation, a certificate generator
for arbitrary models, and floating-point execution certificates remain further work. Local
hidden-node perturbations need not be centered: the cancellation theorem applies to the
initial input, separately for every retained state.

From the repository root:

```bash
lake build HeuristicEstimation
```

The runnable rational calculations embedded in the article are in `site/Examples.lean` under
`sharedExpectation`, `discardedSignal`, `attentionMoments`, `sensitiveSketch`, and `encodedSketch`.
Their recorded output and type information are
refreshed with the existing `python3 site/lean.py` workflow. See the root README for building
the complete local site.
