# CUDA Certificate Folder

Here is the CUDA part of the example. I kept the kernel deliberately small: one
value-reduction fragment from attention, the PTX/CUBIN/SASS produced by NVCC,
the extractor that reads those files, and the Lean checker that accepts the
resulting certificate.

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

The split to keep in mind is:

```text
week-01-batch-invariant-inference/BatchInvariantInference/CUDA.lean
  defines the checker and proves checker soundness

week-01-batch-invariant-inference/BatchInvariantInference/Generated/TinyValueReductionCert.lean
  contains the concrete generated certificate value
```

This still does not claim full CUDA/PTX/SASS/hardware verification. It checks a
small proof-carrying microkernel certificate and keeps the lower runtime and
hardware refinement boundary visible.
