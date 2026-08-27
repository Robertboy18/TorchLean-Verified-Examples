# A Small CUDA Reduction Certificate

This directory isolates one value-reduction fragment from attention. NVCC compiles the kernel to
PTX, CUBIN, and SASS; the extractor recovers the eight fused multiply-add steps; and Lean checks
that the recovered chain denotes the intended reduction. Keeping the kernel small makes the
certificate and its assumptions possible to inspect directly.

## Files

```text
../BatchInvariantInference/CUDA.lean
  Hand-written Lean checker and semantics for the tiny CUDA value-reduction
  certificate. Read this when you want to know what Lean actually checks.

../BatchInvariantInference/Generated/TinyValueReductionCert.lean
  Generated Lean certificate value for the current CUDA/PTX/SASS build.

tiny_attn_one_row.cu
  The inspected CUDA kernel. It computes only the value-reduction part of
  attention, with softmax weights already provided.

extract_cert.py
  Compiles the CUDA kernel, reads PTX/SASS, extracts the eight-step FMA
  dataflow chain, writes cert_tiny_attn.json, and emits
  ../BatchInvariantInference/Generated/TinyValueReductionCert.lean.

cert_tiny_attn.json
  JSON copy of the extracted certificate for inspection.

build/
  Generated PTX, CUBIN, and SASS files tied to the certificate hashes.
```

## Regenerate

From the repository root:

```bash
python3 week-01-batch-invariant-inference/cuda/extract_cert.py
lake build BatchInvariantInference
```

The checker and the generated evidence have different jobs:

```text
week-01-batch-invariant-inference/BatchInvariantInference/CUDA.lean
  defines the checker and proves checker soundness

week-01-batch-invariant-inference/BatchInvariantInference/Generated/TinyValueReductionCert.lean
  contains the concrete generated certificate value
```

The result covers this extracted FMA chain. It does not give operational semantics to all PTX or
SASS instructions, verify NVCC, or certify the NVIDIA hardware. Those components remain below the
checked boundary.
