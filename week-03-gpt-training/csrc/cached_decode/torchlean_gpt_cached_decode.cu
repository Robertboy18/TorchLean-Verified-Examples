#include <lean/lean.h>

#include <cuda_runtime.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "torchlean_cuda_buffer.h"

/*
 * Example-local CUDA support for incremental GPT decoding.
 *
 * This file deliberately implements only state that TorchLean's ordinary eager tape does not own:
 * persistent per-layer key/value rows. Matrix multiplication, parameter storage, checkpoint
 * loading, and host transfer continue to use TorchLean's CUDA buffer runtime.
 */

typedef struct {
  uint32_t layers;
  uint32_t heads;
  uint32_t capacity;
  uint32_t head_dim;
  float* keys;
  float* values;
} torchlean_gpt_kv_cache;

static void check_cuda(cudaError_t error, const char* message) {
  if (error != cudaSuccess) {
    lean_internal_panic(message);
  }
}

static size_t checked_mul(size_t a, size_t b, const char* message) {
  if (a != 0 && b > SIZE_MAX / a) {
    lean_internal_panic(message);
  }
  return a * b;
}

static size_t cache_elements(
    uint32_t layers, uint32_t heads, uint32_t capacity, uint32_t head_dim) {
  size_t count = checked_mul((size_t)layers, (size_t)heads, "KV cache size overflow");
  count = checked_mul(count, (size_t)capacity, "KV cache size overflow");
  return checked_mul(count, (size_t)head_dim, "KV cache size overflow");
}

static void cache_release(torchlean_gpt_kv_cache* cache) {
  if (!cache) {
    return;
  }
  if (cache->keys) {
    (void)cudaFree(cache->keys);
    cache->keys = NULL;
  }
  if (cache->values) {
    (void)cudaFree(cache->values);
    cache->values = NULL;
  }
}

static void cache_finalize(void* pointer) {
  torchlean_gpt_kv_cache* cache = (torchlean_gpt_kv_cache*)pointer;
  if (!cache) {
    return;
  }
  cache_release(cache);
  free(cache);
}

/* A cache owns no Lean references. */
static void cache_foreach(void* _pointer, b_lean_obj_arg _callback) {
  (void)_pointer;
  (void)_callback;
}

static lean_external_class* cache_class = NULL;

static lean_external_class* get_cache_class(void) {
  if (!cache_class) {
    cache_class = lean_register_external_class(cache_finalize, cache_foreach);
  }
  return cache_class;
}

static torchlean_gpt_kv_cache* unbox_cache(b_lean_obj_arg object) {
  lean_object* value = (lean_object*)object;
  if (!lean_is_external(value)) {
    lean_internal_panic("cached decoder: expected a native KV cache");
  }
  return (torchlean_gpt_kv_cache*)lean_get_external_data(value);
}

static lean_obj_res box_cache(torchlean_gpt_kv_cache* cache) {
  return lean_alloc_external(get_cache_class(), cache);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_create(
    uint32_t layers, uint32_t heads, uint32_t capacity, uint32_t head_dim) {
  if (layers == 0 || heads == 0 || capacity == 0 || head_dim == 0) {
    lean_internal_panic("cached decoder: KV cache dimensions must be positive");
  }
  torchlean_gpt_kv_cache* cache =
      (torchlean_gpt_kv_cache*)calloc(1, sizeof(torchlean_gpt_kv_cache));
  if (!cache) {
    lean_internal_panic_out_of_memory();
  }
  cache->layers = layers;
  cache->heads = heads;
  cache->capacity = capacity;
  cache->head_dim = head_dim;

  const size_t elements = cache_elements(layers, heads, capacity, head_dim);
  const size_t bytes = checked_mul(elements, sizeof(float), "KV cache byte size overflow");
  cudaError_t key_error = cudaMalloc((void**)&cache->keys, bytes);
  if (key_error != cudaSuccess) {
    free(cache);
    lean_internal_panic("cached decoder: CUDA key-cache allocation failed");
  }
  cudaError_t value_error = cudaMalloc((void**)&cache->values, bytes);
  if (value_error != cudaSuccess) {
    (void)cudaFree(cache->keys);
    free(cache);
    lean_internal_panic("cached decoder: CUDA value-cache allocation failed");
  }
  check_cuda(cudaMemset(cache->keys, 0, bytes), "cached decoder: key-cache reset failed");
  check_cuda(cudaMemset(cache->values, 0, bytes), "cached decoder: value-cache reset failed");
  return lean_io_result_mk_ok(box_cache(cache));
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_reset(
    b_lean_obj_arg cache_object) {
  torchlean_gpt_kv_cache* cache = unbox_cache(cache_object);
  if (!cache->keys || !cache->values) {
    lean_internal_panic("cached decoder: reset after cache close");
  }
  const size_t elements =
      cache_elements(cache->layers, cache->heads, cache->capacity, cache->head_dim);
  const size_t bytes = checked_mul(elements, sizeof(float), "KV cache byte size overflow");
  check_cuda(cudaMemset(cache->keys, 0, bytes), "cached decoder: key-cache reset failed");
  check_cuda(cudaMemset(cache->values, 0, bytes), "cached decoder: value-cache reset failed");
  return lean_io_result_mk_ok(lean_box(0));
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_close(
    b_lean_obj_arg cache_object) {
  cache_release(unbox_cache(cache_object));
  return lean_io_result_mk_ok(lean_box(0));
}

__global__ static void append_cache_kernel(
    const float* key, const float* value, float* keys, float* values,
    uint32_t layer, uint32_t position, uint32_t heads,
    uint32_t capacity, uint32_t head_dim) {
  const size_t total = (size_t)heads * head_dim;
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += (size_t)blockDim.x * gridDim.x) {
    const uint32_t head = (uint32_t)(index / head_dim);
    const uint32_t coordinate = (uint32_t)(index % head_dim);
    const size_t cache_index =
        ((((size_t)layer * heads + head) * capacity + position) * head_dim) + coordinate;
    keys[cache_index] = key[index];
    values[cache_index] = value[index];
  }
}

/*
 * One CUDA block computes one attention head. Scores occupy `prefix_length` shared floats.
 * Dot products and weighted sums use a fixed per-coordinate order; the implementation is compact
 * enough for a reference-quality cached decoder while avoiding a full context-sized logits tensor.
 */
__global__ static void cached_attention_kernel(
    const float* query, const float* keys, const float* values, float* output,
    uint32_t layer, uint32_t position, uint32_t heads,
    uint32_t capacity, uint32_t head_dim) {
  extern __shared__ float scores[];
  const uint32_t head = blockIdx.x;
  const uint32_t prefix_length = position + 1;
  const float* query_head = query + (size_t)head * head_dim;
  const size_t cache_head =
      ((size_t)layer * heads + head) * capacity * head_dim;
  const float scale = rsqrtf((float)head_dim);

  for (uint32_t row = threadIdx.x; row < prefix_length; row += blockDim.x) {
    const float* key_row = keys + cache_head + (size_t)row * head_dim;
    float dot = 0.0f;
    for (uint32_t coordinate = 0; coordinate < head_dim; ++coordinate) {
      dot += query_head[coordinate] * key_row[coordinate];
    }
    scores[row] = dot * scale;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float maximum = -FLT_MAX;
    for (uint32_t row = 0; row < prefix_length; ++row) {
      maximum = fmaxf(maximum, scores[row]);
    }
    float denominator = 0.0f;
    for (uint32_t row = 0; row < prefix_length; ++row) {
      const float weight = expf(scores[row] - maximum);
      scores[row] = weight;
      denominator += weight;
    }
    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (uint32_t row = 0; row < prefix_length; ++row) {
      scores[row] *= inverse;
    }
  }
  __syncthreads();

  for (uint32_t coordinate = threadIdx.x;
       coordinate < head_dim;
       coordinate += blockDim.x) {
    float total = 0.0f;
    for (uint32_t row = 0; row < prefix_length; ++row) {
      const float* value_row = values + cache_head + (size_t)row * head_dim;
      total += scores[row] * value_row[coordinate];
    }
    output[(size_t)head * head_dim + coordinate] = total;
  }
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_attention(
    b_lean_obj_arg cache_object,
    b_lean_obj_arg query_object,
    b_lean_obj_arg key_object,
    b_lean_obj_arg value_object,
    uint32_t layer,
    uint32_t position) {
  torchlean_gpt_kv_cache* cache = unbox_cache(cache_object);
  torchlean_cuda_buffer* query = torchlean_cuda_buffer_unbox(query_object);
  torchlean_cuda_buffer* key = torchlean_cuda_buffer_unbox(key_object);
  torchlean_cuda_buffer* value = torchlean_cuda_buffer_unbox(value_object);
  if (!cache->keys || !cache->values) {
    lean_internal_panic("cached decoder: attention after cache close");
  }
  if (layer >= cache->layers || position >= cache->capacity) {
    lean_internal_panic("cached decoder: layer or position outside KV cache");
  }
  const size_t width = (size_t)cache->heads * cache->head_dim;
  if (query->size != width || key->size != width || value->size != width) {
    lean_internal_panic("cached decoder: Q/K/V width mismatch");
  }

  torchlean_cuda_buffer* output = torchlean_cuda_buffer_alloc(width);
  const uint32_t threads = 256;
  const uint32_t blocks = (uint32_t)((width + threads - 1) / threads);
  append_cache_kernel<<<blocks, threads>>>(
      key->data, value->data, cache->keys, cache->values,
      layer, position, cache->heads, cache->capacity, cache->head_dim);
  check_cuda(cudaGetLastError(), "cached decoder: cache append kernel failed");
  const size_t shared_bytes = (size_t)(position + 1) * sizeof(float);
  cached_attention_kernel<<<cache->heads, threads, shared_bytes>>>(
      query->data, cache->keys, cache->values, output->data,
      layer, position, cache->heads, cache->capacity, cache->head_dim);
  check_cuda(cudaGetLastError(), "cached decoder: attention kernel failed");
  return lean_io_result_mk_ok(torchlean_cuda_buffer_box(output));
}

__global__ static void layer_norm_kernel(
    const float* input, const float* gamma, const float* beta,
    float* output, uint32_t width, float epsilon) {
  __shared__ float partial[256];
  float local_sum = 0.0f;
  for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    local_sum += input[i];
  }
  partial[threadIdx.x] = local_sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = partial[0] / (float)width;

  float local_variance = 0.0f;
  for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    const float centered = input[i] - mean;
    local_variance += centered * centered;
  }
  partial[threadIdx.x] = local_variance;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse_std = rsqrtf(partial[0] / (float)width + epsilon);
  for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    output[i] = (input[i] - mean) * inverse_std * gamma[i] + beta[i];
  }
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_gpt_layer_norm(
    b_lean_obj_arg input_object,
    b_lean_obj_arg gamma_object,
    b_lean_obj_arg beta_object,
    uint32_t width,
    double epsilon) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(input_object);
  torchlean_cuda_buffer* gamma = torchlean_cuda_buffer_unbox(gamma_object);
  torchlean_cuda_buffer* beta = torchlean_cuda_buffer_unbox(beta_object);
  if (width == 0 || input->size != width || gamma->size != width || beta->size != width) {
    lean_internal_panic("cached decoder: LayerNorm width mismatch");
  }
  torchlean_cuda_buffer* output = torchlean_cuda_buffer_alloc(width);
  layer_norm_kernel<<<1, 256>>>(
      input->data, gamma->data, beta->data, output->data, width, (float)epsilon);
  check_cuda(cudaGetLastError(), "cached decoder: LayerNorm kernel failed");
  return lean_io_result_mk_ok(torchlean_cuda_buffer_box(output));
}

__global__ static void gelu_kernel(const float* input, float* output, size_t count) {
  const float coefficient = 0.044715f;
  const float scale = 0.7978845608028654f; /* sqrt(2/pi) */
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count;
       i += (size_t)blockDim.x * gridDim.x) {
    const float x = input[i];
    /*
     * Keep the same primitive order as TorchLean's GELU definition:
     * x², x³, scale/add, scale, tanh through sigmoid, then the final products.
     * The fused kernel still avoids intermediate allocations, but does not silently replace the
     * runtime expression by a differently rounded algebraic rearrangement.
     */
    const float x2 = x * x;
    const float x3 = x2 * x;
    const float inner = (x + x3 * coefficient) * scale;
    const float sigmoid_two_x = 1.0f / (1.0f + expf(-(inner * 2.0f)));
    const float tanh_value = sigmoid_two_x * 2.0f - 1.0f;
    const float mid = x * (1.0f + tanh_value);
    output[i] = mid * 0.5f;
  }
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_gpt_gelu(
    b_lean_obj_arg input_object, uint32_t count) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(input_object);
  if (input->size != count) {
    lean_internal_panic("cached decoder: GELU size mismatch");
  }
  torchlean_cuda_buffer* output = torchlean_cuda_buffer_alloc(count);
  if (count != 0) {
    const uint32_t threads = 256;
    const uint32_t blocks = (count + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(input->data, output->data, count);
    check_cuda(cudaGetLastError(), "cached decoder: GELU kernel failed");
  }
  return lean_io_result_mk_ok(torchlean_cuda_buffer_box(output));
}
