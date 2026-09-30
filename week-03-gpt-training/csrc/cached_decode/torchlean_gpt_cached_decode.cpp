#include "torchlean_libtorch.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

// Only the persistent cache is example-local. ATen owns its storage and computes attention.
struct Cache {
  at::Tensor keys;
  at::Tensor values;
};

void finalize(void* pointer) { delete static_cast<Cache*>(pointer); }
void foreach_reference(void*, b_lean_obj_arg) {}

lean_external_class* cache_class() {
  static auto* result = lean_register_external_class(finalize, foreach_reference);
  return result;
}

Cache& cache(b_lean_obj_arg object) {
  if (!lean_is_external(object) || lean_get_external_class(object) != cache_class())
    throw std::invalid_argument("cached decoder: expected a KV cache");
  return *static_cast<Cache*>(lean_get_external_data(object));
}

void require_open(const Cache& state) {
  if (!state.keys.defined()) throw std::invalid_argument("cached decoder: cache is closed");
}

template <typename F>
lean_obj_res io(F&& body) {
  try {
    torchlean::initialize();
    at::NoGradGuard no_grad;
    c10::DeviceGuard guard(torchlean::device());
    return lean_io_result_mk_ok(body());
  } catch (const c10::OutOfMemoryError& error) {
    return lean_io_result_mk_error(
        lean_mk_io_error_resource_exhausted(0, lean_mk_string(error.what())));
  } catch (const std::bad_alloc& error) {
    return lean_io_result_mk_error(
        lean_mk_io_error_resource_exhausted(0, lean_mk_string(error.what())));
  } catch (const std::exception& error) {
    return lean_io_result_mk_error(
        lean_mk_io_error_other_error(0, lean_mk_string(error.what())));
  }
}

}  // namespace

extern "C" LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_create(
    uint32_t layers, uint32_t heads, uint32_t capacity, uint32_t head_dim) {
  return io([&] {
    int64_t count = 1;
    for (auto dimension : {layers, heads, capacity, head_dim}) {
      if (dimension == 0 || count > std::numeric_limits<int64_t>::max() / dimension)
        throw std::invalid_argument("cached decoder: invalid cache dimensions");
      count *= dimension;
    }
    if (count > std::numeric_limits<int64_t>::max() / static_cast<int64_t>(sizeof(float)))
      throw std::invalid_argument("cached decoder: cache byte size overflow");
    auto state = std::make_unique<Cache>();
    state->keys = at::zeros({layers, heads, capacity, head_dim}, torchlean::options());
    state->values = at::zeros_like(state->keys);
    return lean_alloc_external(cache_class(), state.release());
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_reset(b_lean_obj_arg object) {
  return io([&] {
    auto& state = cache(object);
    require_open(state);
    state.keys.zero_();
    state.values.zero_();
    return lean_box(0);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_close(b_lean_obj_arg object) {
  return io([&] {
    auto& state = cache(object);
    state.keys = at::Tensor();
    state.values = at::Tensor();
    return lean_box(0);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_attention(
    b_lean_obj_arg object, b_lean_obj_arg query, b_lean_obj_arg key, b_lean_obj_arg value,
    uint32_t layer, uint32_t position) {
  return io([&] {
    auto& state = cache(object);
    require_open(state);
    const auto heads = state.keys.size(1);
    const auto width = state.keys.size(3);
    if (layer >= state.keys.size(0) || position >= state.keys.size(2))
      throw std::invalid_argument("cached decoder: cache index out of range");
    const auto& q = torchlean::tensor(query);
    const auto& k = torchlean::tensor(key);
    const auto& v = torchlean::tensor(value);
    for (const auto* input : {&q, &k, &v}) {
      if (input->numel() != heads * width || input->device() != state.keys.device() ||
          input->scalar_type() != at::kFloat)
        throw std::invalid_argument("cached decoder: attention buffer mismatch");
    }
    auto keys = state.keys.select(0, layer);
    auto values = state.values.select(0, layer);
    keys.select(1, position).copy_(k.reshape({heads, width}));
    values.select(1, position).copy_(v.reshape({heads, width}));
    const int64_t length = static_cast<int64_t>(position) + 1;
    auto scores = at::bmm(q.reshape({heads, 1, width}),
        keys.narrow(1, 0, length).transpose(1, 2)) / std::sqrt(static_cast<double>(width));
    auto output = at::bmm(at::softmax(scores, -1), values.narrow(1, 0, length));
    return torchlean::box(output.reshape({heads * width}).contiguous());
  });
}
