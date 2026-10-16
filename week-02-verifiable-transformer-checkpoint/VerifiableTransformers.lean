import VerifiableTransformers.Spec.UpstreamSmallGPT
import VerifiableTransformers.Certificate.Fingerprint
import VerifiableTransformers.Certificate.Weights
import VerifiableTransformers.Certificate.FiniteEval
import VerifiableTransformers.Certificate.Circuit
import VerifiableTransformers.Generated.UpstreamExportSummary
import VerifiableTransformers.Generated.UpstreamEvalTrace
import VerifiableTransformers.Generated.UpstreamCircuitSummaries
import VerifiableTransformers.Generated.TorchLeanEvalTrace

/-!
# Verifiable transformer checkpoint example

This module imports our transcription of the operators in Neel Somani's small verifiable
transformer, the checkers in `Certificate`, and the generated evidence those checkers accept.
The checkpoint was trained with Neel's code at the upstream commit pinned in
`Certificate.Fingerprint`. The archived TorchLean trace uses the same prompt domain but a
different model and checkpoint.

Python (Neel's scripts and the generators in `tools/`) produces the weights, score traces, and
circuit summaries. Lean checks their recorded dimensions, fingerprints, domain coverage,
margins, and circuit status counts. The compiled checkpoint replay is a separate executable in
`Replay.UpstreamFloatReplay`.
-/
