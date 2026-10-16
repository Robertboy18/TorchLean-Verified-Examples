# A Small CUDA Reduction Certificate

This directory isolates one fragment of attention, the value reduction. NVCC compiles the kernel to
PTX, ptxas assembles that PTX into the CUBIN, and nvdisasm prints its SASS. From there the
extractor recovers the active path (eight fused multiply-add steps), and Lean checks that the
recovered program stores the intended reduction. I kept the kernel small so that you can inspect
the certificate and its assumptions directly.

Starting from the kernel's weight, value, and output parameters, the extractor follows addresses
out, keeping 32-bit index arithmetic separate from 64-bit pointer arithmetic, and checks the byte
offset of each load and of the final store. It also walks all four combinations of the batch and thread guards,
and every inactive path has to return without a global memory access. That covers a branch taken
when `b < B && tid < 4` is false as well as a branch taken when `b >= B || tid > 3` is true, and it
doesn't depend on register numbers or label names. Anything outside this small supported subset of
instructions is rejected. Then the extractor records the active path for Lean: each load with its
byte address, the eight FMAs with their registers, and the store. Lean runs that program
symbolically, so the register chain and the addresses get checked a second time there.

Every checker theorem takes the expected target as an argument (the generated contract names the
compiled architecture), and the generated file instantiates them for `sm_80`. Some checks stay
outside Lean, though. Source, PTX, and SASS flags are only substring tests and instruction counts, and the guard trace runs in the extractor, so for those Lean only requires that
they were recorded as true.

## Launch Requirements

In the CUDA source, element indices are computed with 32-bit unsigned arithmetic. If you want to read
those indices as ordinary rows, you need `0 ≤ B ≤ 2^27` (or equivalently
`32 * B ≤ 2^32`). For an active row `b < B`, reduction step `t < 8`, and output
coordinate `tid < 4`, the largest value index is `32 * b + 4 * t + tid`, which is strictly less
than `32 * B` and so fits in 32 bits throughout this range. So do the weight and output indices.
Go above this batch bound, though, and value indices can wrap modulo `2^32` and alias an earlier row.

The extractor does track that unsigned wraparound. But reading its symbolic row offsets
as accesses to distinct rows takes the batch bound above and valid device allocations:

* `weights` contains at least `8 * B` floats, `v` at least `32 * B` floats, and
  `out` at least `4 * B` floats (four bytes per float).
* All three buffers are separate, suitably aligned allocations in the active CUDA
  context, with readable inputs and a writable output. Their address ranges have to be
  valid, and they have to stay allocated until the kernel finishes.
* Grid and blocks have unit `y` and `z` dimensions. Launch at least `B` blocks
  and at least four threads per block, within the device's launch limits. Any extra
  blocks and threads exercise the existing bounds guards.

The runtime check (`run_cubin.cpp`) only uses `B = 0, 1, 2, 7, 33, 129`. It creates separate
buffers of the required sizes and launches only after every allocation and input copy has
succeeded, with one input row allocated for the empty-batch case and three extra output rows for
the guard checks. So these tests stay within the indexing and allocation preconditions, and none
of them tries a batch near the unsigned-index limit.

Lean's address lemmas use natural-number indices. On buffers laid out as `[B, 8]` and
`[B, 8, 4]`, the generated theorems run the extracted program and show that each active thread
stores the FMA chain over its own row, that its loads and store stay in bounds, and that distinct
threads write distinct words. You still need the conditions above before any of that applies to a
live launch. And the full native attention bridge carries its own proof of equality with
TorchLean's FlashAttention result, because this small reduction certificate leaves that separate
refinement obligation in place.

## Files

```text
../BatchInvariantInference/CUDA.lean
  Hand-written Lean checker and semantics for the tiny CUDA value-reduction
  certificate. Read this when you want to know what Lean actually checks.

../BatchInvariantInference/Generated.lean
  Generated Lean certificate value for the recorded CUDA/PTX/SASS build.

tiny_attn_one_row.cu
  The inspected CUDA kernel. It computes only the value-reduction part of
  attention, with softmax weights already provided.

extract_cert.py
  Compiles the CUDA kernel, reads PTX/SASS, extracts the active-path program
  with its eight-step FMA chain, writes cert_tiny_attn.json, and emits
  ../BatchInvariantInference/Generated.lean.

cert_tiny_attn.json
  JSON copy of the extracted certificate for inspection.

run_cubin.cpp
  Runs the CUBIN on a GPU and compares it with a binary32 FMA reference.

runtime_check.json
  Results of that run.

build/
  Generated PTX, CUBIN, and SASS files tied to the certificate hashes.
```

## Regenerate

Compile CUDA in a cluster job, from a copied checkout and for the assigned GPU's
architecture. For an `sm_80` job, run this from the copy's repository root:

```bash
python3 week-01-batch-invariant-inference/cuda/extract_cert.py --arch sm_80
lake build BatchInvariantInference
```

The committed files came from CUDA 13.2 (V13.2.78). With that toolkit (`--cuda-bin` points at
its `bin` directory, which needs `nvcc`, `ptxas`, and `nvdisasm`) the extractor reproduces the
PTX, CUBIN, and SASS byte for byte. If you only want the JSON and Lean, `--from-build` re-extracts
them from the committed files without compiling.

Keep the source, PTX, CUBIN, SASS, JSON, and generated Lean together. Before you import the
bundle into the shared checkout, check their recorded hashes and compile the generated
certificate. Beyond that, the runtime check actually executes the CUBIN and compares its outputs
against a binary32 FMA reference. Those finite tests supplement the Lean checks:

```bash
g++ -O2 -std=c++17 -ffp-contract=off -I/usr/local/cuda/include \
  week-01-batch-invariant-inference/cuda/run_cubin.cpp -lcuda -lcrypto -o /tmp/run_cubin
/tmp/run_cubin week-01-batch-invariant-inference/cuda/build/tiny_attn_one_row.cubin \
  week-01-batch-invariant-inference/cuda/runtime_check.json
```

On an A100 all 688 outputs matched the FMA reference bit for bit, and no thread wrote past the
batch. More than half of them (360 of the 688) differ from a separately rounded multiply-then-add
([runtime_check.json](runtime_check.json)).

The checker and the generated evidence have different jobs:

```text
week-01-batch-invariant-inference/BatchInvariantInference/CUDA.lean
  defines the checker and proves checker soundness

week-01-batch-invariant-inference/BatchInvariantInference/Generated.lean
  contains the concrete generated certificate value
```

So the result covers this one extracted FMA chain. It doesn't give operational semantics to every
PTX or SASS instruction, verify NVCC or ptxas, or certify the NVIDIA hardware, and all of those
stay below the checked boundary.
