// Small CUDA kernel used as the verification target for the TorchLean
// batch-invariance example.
//
// One CUDA block covers one batch row. Threads 0..3 each own one output
// coordinate. The softmax weights are provided as input so the first
// PTX/SASS certificate can focus on memory ownership and the KV reduction
// schedule before proving the online-softmax recurrence.

extern "C" __global__
void tiny_attn_one_row(
    const float* __restrict__ weights, // [B, 8]
    const float* __restrict__ v,       // [B, 8, 4]
    float* __restrict__ out,           // [B, 4]
    unsigned int B) {
  const unsigned int b = blockIdx.x;
  const unsigned int tid = threadIdx.x;

  if (b >= B) return;

  if (tid < 4) {
    float acc = 0.0f;

    // Fixed request-local KV schedule: 0,1,2,3,4,5,6,7.
    for (unsigned int t = 0; t < 8; ++t) {
      const float w = weights[b * 8 + t];
      const float val = v[(b * 8 + t) * 4 + tid];
      acc = acc + w * val;
    }

    out[b * 4 + tid] = acc;
  }
}
