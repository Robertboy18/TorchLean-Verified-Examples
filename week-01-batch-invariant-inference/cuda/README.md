# A Small CUDA Reduction Certificate

This directory isolates one value-reduction fragment from attention. NVCC compiles the kernel to
PTX, CUBIN, and SASS; the extractor recovers the eight fused multiply-add steps; and Lean checks
that the recovered chain denotes the intended reduction. Keeping the kernel small makes the
certificate and its assumptions possible to inspect directly.

The extractor follows addresses from the kernel's weight, value, and output parameters.
It keeps 32-bit index arithmetic separate from 64-bit pointer arithmetic and checks the
byte offset of each load and the final store. It also follows all four combinations of
the batch and thread guards: every inactive path must return without a global memory
access. This handles both a branch on `b < B && tid < 4` being false and a branch on
`b >= B || tid > 3` being true, without depending on register numbers or label names.
Instructions outside this small supported subset are rejected.

The generated contract names the compiled architecture explicitly. The checker requires
that exact target, together with every dataflow and memory obligation. The original
`sm_70` checker and its theorems remain available for the earlier certificates.

## Launch Requirements

The CUDA source computes element indices with 32-bit unsigned arithmetic. To interpret
those indices as ordinary rows, require `0 ≤ B ≤ 2^27`, or equivalently
`32 * B ≤ 2^32`. For an active row `b < B`, reduction step `t < 8`, and output
coordinate `tid < 4`, the largest value index is `32 * b + 4 * t + tid`.
It is strictly less than `32 * B`, so it fits in 32 bits throughout this range.
The weight and output indices fit as well. Above this batch bound, value indices
can wrap modulo `2^32` and alias an earlier row.

The extractor tracks that unsigned wraparound. Interpreting its symbolic row offsets
as accesses to distinct rows requires the batch bound above and valid device allocations:

* `weights` contains at least `8 * B` floats, `v` at least `32 * B` floats, and
  `out` at least `4 * B` floats. Each float occupies four bytes.
* The three buffers are separate, suitably aligned allocations in the active CUDA
  context, with readable inputs and writable output. Their address ranges must be
  valid, and they must remain allocated until the kernel finishes.
* The grid and blocks have unit `y` and `z` dimensions. Launch at least `B` blocks
  and at least four threads per block, within the device's launch limits. Extra
  blocks and threads exercise the existing bounds guards.

The cluster runtime check uses only `B = 0, 1, 2, 7, 33, 129`, creates separate
buffers of the required sizes, and launches only after all allocations and input
copies succeed. It allocates one input row for the empty-batch case and three extra
output rows for guard checks. These tests therefore stay within the indexing and
allocation preconditions; they do not test batches near the unsigned-index limit.

Lean's address lemmas use natural-number indices. The generated denotation theorem
evaluates the extracted eight-step FMA chain on supplied values; applying that result
to concrete CUDA memory still requires the launch conditions and the connection
between the extracted operands and the live buffers. The full native attention bridge
also carries its own proof of equality with TorchLean's FlashAttention result. The
small reduction certificate leaves that separate refinement obligation in place.

## Files

```text
../BatchInvariantInference/CUDA.lean
  Hand-written Lean checker and semantics for the tiny CUDA value-reduction
  certificate. Read this when you want to know what Lean actually checks.

../BatchInvariantInference/Generated/TinyValueReductionCert.lean
  Generated Lean certificate value for the recorded CUDA/PTX/SASS build.

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

Run CUDA compilation in a cluster job, using a copied checkout and the assigned GPU's
architecture. For an `sm_80` job, run from that copy's repository root:

```bash
python3 week-01-batch-invariant-inference/cuda/extract_cert.py --arch sm_80
lake build BatchInvariantInference
```

Retain the source, PTX, CUBIN, SASS, JSON, and generated Lean together. Check their recorded
hashes and compile the generated certificate before importing the bundle into the shared
checkout. The cluster validation runner additionally executes the CUBIN and compares its
outputs against a binary32 FMA reference; those finite tests supplement the Lean checks.

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
