#include <lean/lean.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "torchlean_cuda_buffer.h"

/*
 * Host-memory parity implementation for ordinary `lake build`.
 *
 * It is intentionally simple and deterministic. The production Week 3 chat path uses the CUDA
 * implementation, while this stub keeps the same Lean modules buildable and testable without an
 * NVIDIA toolchain.
 */

typedef struct {
  uint32_t layers;
  uint32_t heads;
  uint32_t capacity;
  uint32_t head_dim;
  float* keys;
  float* values;
} torchlean_gpt_kv_cache;

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
  free(cache->keys);
  free(cache->values);
  cache->keys = NULL;
  cache->values = NULL;
}

static void cache_finalize(void* pointer) {
  torchlean_gpt_kv_cache* cache = (torchlean_gpt_kv_cache*)pointer;
  if (!cache) {
    return;
  }
  cache_release(cache);
  free(cache);
}

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

LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_create(
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
  cache->keys = (float*)calloc(elements, sizeof(float));
  cache->values = (float*)calloc(elements, sizeof(float));
  if (!cache->keys || !cache->values) {
    cache_release(cache);
    free(cache);
    lean_internal_panic_out_of_memory();
  }
  return lean_io_result_mk_ok(box_cache(cache));
}

LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_reset(b_lean_obj_arg cache_object) {
  torchlean_gpt_kv_cache* cache = unbox_cache(cache_object);
  if (!cache->keys || !cache->values) {
    lean_internal_panic("cached decoder: reset after cache close");
  }
  const size_t elements =
      cache_elements(cache->layers, cache->heads, cache->capacity, cache->head_dim);
  memset(cache->keys, 0, elements * sizeof(float));
  memset(cache->values, 0, elements * sizeof(float));
  return lean_io_result_mk_ok(lean_box(0));
}

LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_close(b_lean_obj_arg cache_object) {
  cache_release(unbox_cache(cache_object));
  return lean_io_result_mk_ok(lean_box(0));
}

LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_attention(
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

  for (uint32_t head = 0; head < cache->heads; ++head) {
    for (uint32_t coordinate = 0; coordinate < cache->head_dim; ++coordinate) {
      const size_t source = (size_t)head * cache->head_dim + coordinate;
      const size_t target =
          ((((size_t)layer * cache->heads + head) * cache->capacity + position) *
            cache->head_dim) + coordinate;
      cache->keys[target] = key->data[source];
      cache->values[target] = value->data[source];
    }
  }

  torchlean_cuda_buffer* output = torchlean_cuda_buffer_alloc(width);
  const uint32_t prefix_length = position + 1;
  float* scores = (float*)malloc((size_t)prefix_length * sizeof(float));
  if (!scores) {
    torchlean_cuda_buffer_drop_unboxed(output);
    lean_internal_panic_out_of_memory();
  }
  const float scale = 1.0f / sqrtf((float)cache->head_dim);
  for (uint32_t head = 0; head < cache->heads; ++head) {
    const float* query_head = query->data + (size_t)head * cache->head_dim;
    const size_t cache_head =
        ((size_t)layer * cache->heads + head) * cache->capacity * cache->head_dim;
    float maximum = -FLT_MAX;
    for (uint32_t row = 0; row < prefix_length; ++row) {
      const float* key_row = cache->keys + cache_head + (size_t)row * cache->head_dim;
      float dot = 0.0f;
      for (uint32_t coordinate = 0; coordinate < cache->head_dim; ++coordinate) {
        dot += query_head[coordinate] * key_row[coordinate];
      }
      scores[row] = dot * scale;
      maximum = fmaxf(maximum, scores[row]);
    }
    float denominator = 0.0f;
    for (uint32_t row = 0; row < prefix_length; ++row) {
      scores[row] = expf(scores[row] - maximum);
      denominator += scores[row];
    }
    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (uint32_t coordinate = 0; coordinate < cache->head_dim; ++coordinate) {
      float total = 0.0f;
      for (uint32_t row = 0; row < prefix_length; ++row) {
        const float* value_row =
            cache->values + cache_head + (size_t)row * cache->head_dim;
        total += scores[row] * inverse * value_row[coordinate];
      }
      output->data[(size_t)head * cache->head_dim + coordinate] = total;
    }
  }
  free(scores);
  return lean_io_result_mk_ok(torchlean_cuda_buffer_box(output));
}

LEAN_EXPORT lean_obj_res torchlean_gpt_layer_norm(
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
  float mean = 0.0f;
  for (uint32_t i = 0; i < width; ++i) {
    mean += input->data[i];
  }
  mean /= (float)width;
  float variance = 0.0f;
  for (uint32_t i = 0; i < width; ++i) {
    const float centered = input->data[i] - mean;
    variance += centered * centered;
  }
  variance /= (float)width;
  const float inverse_std = 1.0f / sqrtf(variance + (float)epsilon);
  for (uint32_t i = 0; i < width; ++i) {
    output->data[i] =
        (input->data[i] - mean) * inverse_std * gamma->data[i] + beta->data[i];
  }
  return lean_io_result_mk_ok(torchlean_cuda_buffer_box(output));
}

LEAN_EXPORT lean_obj_res torchlean_gpt_gelu(b_lean_obj_arg input_object, uint32_t count) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(input_object);
  if (input->size != count) {
    lean_internal_panic("cached decoder: GELU size mismatch");
  }
  torchlean_cuda_buffer* output = torchlean_cuda_buffer_alloc(count);
  const float coefficient = 0.044715f;
  const float scale = 0.7978845608028654f;
  for (uint32_t i = 0; i < count; ++i) {
    const float x = input->data[i];
    const float x2 = x * x;
    const float x3 = x2 * x;
    const float inner = (x + x3 * coefficient) * scale;
    const float sigmoid_two_x = 1.0f / (1.0f + expf(-(inner * 2.0f)));
    const float tanh_value = sigmoid_two_x * 2.0f - 1.0f;
    const float mid = x * (1.0f + tanh_value);
    output->data[i] = mid * 0.5f;
  }
  return lean_io_result_mk_ok(torchlean_cuda_buffer_box(output));
}
