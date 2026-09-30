#include <lean/lean.h>
#include <stdint.h>

static lean_obj_res unavailable(void) {
  return lean_io_result_mk_error(lean_mk_io_error_other_error(0,
      lean_mk_string("cached decoder: rebuild with -Kcuda=true and a LibTorch CUDA SDK")));
}

LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_create(
    uint32_t layers, uint32_t heads, uint32_t capacity, uint32_t head_dim) {
  (void)layers; (void)heads; (void)capacity; (void)head_dim;
  return unavailable();
}

LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_reset(b_lean_obj_arg cache) {
  (void)cache;
  return unavailable();
}

LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_close(b_lean_obj_arg cache) {
  (void)cache;
  return unavailable();
}

LEAN_EXPORT lean_obj_res torchlean_gpt_kv_cache_attention(
    b_lean_obj_arg cache, b_lean_obj_arg query, b_lean_obj_arg key, b_lean_obj_arg value,
    uint32_t layer, uint32_t position) {
  (void)cache; (void)query; (void)key; (void)value; (void)layer; (void)position;
  return unavailable();
}
