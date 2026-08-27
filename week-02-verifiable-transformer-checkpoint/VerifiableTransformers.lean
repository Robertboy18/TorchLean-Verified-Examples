/-
Top-level module for the Week 2 verifiable-transformer example.

The imports are ordered around the claim I want Lean to check:

1. record the small-GPT operator semantics from Neel Somani's verifier in Lean;
2. define the finite prompt/circuit checks we care about;
3. import generated certificates from real checkpoint and trace outputs; and
4. check the archived TorchLean comparison trace on the same finite task.

The design mirrors `neelsomani/verifiable-transformers`: `scripts/small/config.py`
for dimensions, `scripts/small/train.py` for the custom layers,
`scripts/small/extract_weights.py` for exported weights, and
`scripts/smt/{encoders,trace}.py` plus `scripts/small/{extract,verify}.py` for
the Z3/circuit side. Those Python files create the exported evidence; Lean
checks that the evidence still matches the finite claim.
-/

import VerifiableTransformers.Spec.UpstreamSmallGPT
import VerifiableTransformers.Certificate.Weights
import VerifiableTransformers.Certificate.FiniteEval
import VerifiableTransformers.Certificate.Circuit
import VerifiableTransformers.Generated.UpstreamExportSummary
import VerifiableTransformers.Generated.UpstreamEvalTrace
import VerifiableTransformers.Generated.TorchLeanEvalTrace
