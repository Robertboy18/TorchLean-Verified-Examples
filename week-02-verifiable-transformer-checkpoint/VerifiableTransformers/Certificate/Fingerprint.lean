/-!
# Pinned checkpoint fingerprints

Each generated certificate records the SHA-256 of the files it was generated from, and each
certificate check compares that record with the strings below. So the export summary, the
replay payload, the score trace, and the circuit summaries all have to come from one training
run. That run is Neel Somani's training code at the upstream commit below, which we ran with
his default settings (`artifacts/upstream-small-gpt/provenance.json`). Retraining changes the
hashes, and the checks fail until this file is updated on purpose.

Lean compares strings here. It does not hash the files.
-/

namespace VerifiableTransformers.Certificate.Fingerprint

/-- `neelsomani/verifiable-transformers` commit whose code trained and exported the checkpoint. -/
def upstreamCommit : String :=
  "8e3c54bad2494c17c91109fdb58cf9e2f1c98efc"

/-- SHA-256 of `artifacts/upstream-small-gpt/smt_weights.json`. -/
def smtWeightsSha256 : String :=
  "ab61ac5dce70292ab9fa4dcd5265d98e04c34c5d89b809384ab297ef9e7abc34"

/-- SHA-256 of `artifacts/upstream-small-gpt/checkpoint-final/model.safetensors`. -/
def modelSafetensorsSha256 : String :=
  "f5b04594e8a0d9f833bd041e3a239bd865fc502819fc2eb0765e8119c3955b07"

/-- SHA-256 of `artifacts/torchlean-cuda-small-gpt.parambits.json`, the archived TorchLean run. -/
def torchLeanParamBitsSha256 : String :=
  "94a86fd6a0e875e1ec54d9c7bcd55ade995950ef7a5f4e6809a4c59763649442"

end VerifiableTransformers.Certificate.Fingerprint
