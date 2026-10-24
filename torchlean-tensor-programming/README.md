# Writing tensor programs in Lean

The article lives in
[`site/articles/torchlean-tensor-programming.md`](../site/articles/torchlean-tensor-programming.md).
Its checked examples are the `tensor*` anchors in
[`site/Examples.lean`](../site/Examples.lean) (`tensorFunction` through `tensorChainRule`), and
they use the TorchLean revision pinned in this repository's `lake-manifest.json`.

Beyond those examples, the essay follows `einsum` lowering, model composition, saved runtime
comparisons, PDE derivatives, typed training and checkpoints, and supported JAX/PyTorch graph
imports. Its model-builder and calculus code are checked anchors too.

To refresh compiler type information and the CPU command outputs, then rebuild the site:

```bash
python -m pip install -r site/requirements.txt
lake build BatchInvariantInference VerifiableTransformers TorchLeanGPT \
  KimiK3 Nemotron HeuristicEstimation NN
python3 site/lean.py
python3 site/build.py
```

`site/lean.py` asks Lean's server about modules that are already built, so build the six weeks
and TorchLean's `NN` library first. Output goes to `build/site`.

Every CPU output on the page comes from Lean. Definitions that select `gpu` are typechecked, and
`site/lean.py` rejects one whose GPU compilation failed during elaboration (such a call
typechecks, then fails when it runs). But recording the page doesn't run those definitions or
measure GPU performance. For device setup and the full custom-computation API, see
[TorchLean's guide](https://lean-dojo.github.io/TorchLean/examples/custom-computations/).
