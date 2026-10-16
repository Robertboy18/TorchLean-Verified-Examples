// Runtime check for the certified CUBIN.
//
// Loads build/tiny_attn_one_row.cubin through the CUDA driver API, launches it
// for B = 0, 1, 2, 7, 33, 129 with random inputs, and compares every output word
// bitwise with a host binary32 FMA chain (std::fmaf, keys 0..7 in order). It also
// counts the outputs where a separately rounded multiply-then-add would differ,
// and checks that the three extra output rows past the batch keep their sentinel.
//
// Build and run from the repository root (commands in cuda/README.md):
//   g++ -O2 -std=c++17 -ffp-contract=off -I/usr/local/cuda/include
//       week-01-batch-invariant-inference/cuda/run_cubin.cpp -lcuda -lcrypto -o /tmp/run_cubin
//   /tmp/run_cubin week-01-batch-invariant-inference/cuda/build/tiny_attn_one_row.cubin
//       week-01-batch-invariant-inference/cuda/runtime_check.json

#include <cuda.h>
#include <openssl/evp.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr unsigned kExtraBlocks = 3;   // blocks past the batch, all guarded
constexpr unsigned kThreads = 32;      // threads per block; only 0..3 are active
constexpr uint32_t kSentinel = 0x7fc0beefu;  // a quiet NaN the kernel never writes

bool check(CUresult result, const char* what) {
  if (result == CUDA_SUCCESS) return true;
  const char* message = nullptr;
  cuGetErrorString(result, &message);
  std::fprintf(stderr, "%s failed: %s\n", what, message ? message : "unknown error");
  return false;
}

uint32_t bits(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof u);
  return u;
}

std::string sha256_file(const char* path) {
  std::ifstream file(path, std::ios::binary);
  std::vector<unsigned char> data((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int length = 0;
  EVP_Digest(data.data(), data.size(), digest, &length, EVP_sha256(), nullptr);
  std::string hex;
  char byte[3];
  for (unsigned int i = 0; i < length; ++i) {
    std::snprintf(byte, sizeof byte, "%02x", digest[i]);
    hex += byte;
  }
  return hex;
}

struct BatchResult {
  unsigned batch = 0;
  unsigned outputs = 0;
  unsigned mismatches = 0;
  unsigned mulAddDiffers = 0;
  unsigned guardRowWrites = 0;
};

bool run_batch(CUfunction kernel, unsigned batch, std::mt19937& rng, BatchResult& result) {
  std::uniform_real_distribution<float> uniform(-4.0f, 4.0f);
  // cuMemAlloc rejects zero bytes, so the empty batch still gets one input row.
  const unsigned inputRows = batch == 0 ? 1 : batch;
  std::vector<float> weights(8 * inputRows), values(32 * inputRows);
  for (float& w : weights) w = uniform(rng);
  for (float& v : values) v = uniform(rng);
  std::vector<float> out(4 * (batch + kExtraBlocks));
  for (float& o : out) std::memcpy(&o, &kSentinel, sizeof o);

  CUdeviceptr dWeights = 0, dValues = 0, dOut = 0;
  bool ok = check(cuMemAlloc(&dWeights, weights.size() * sizeof(float)), "cuMemAlloc(weights)") &&
            check(cuMemAlloc(&dValues, values.size() * sizeof(float)), "cuMemAlloc(v)") &&
            check(cuMemAlloc(&dOut, out.size() * sizeof(float)), "cuMemAlloc(out)") &&
            check(cuMemcpyHtoD(dWeights, weights.data(), weights.size() * sizeof(float)),
                  "copy weights") &&
            check(cuMemcpyHtoD(dValues, values.data(), values.size() * sizeof(float)), "copy v") &&
            check(cuMemcpyHtoD(dOut, out.data(), out.size() * sizeof(float)), "copy out");
  // Launch only after every allocation and copy succeeded.
  if (ok) {
    unsigned b = batch;
    void* args[] = {&dWeights, &dValues, &dOut, &b};
    ok = check(cuLaunchKernel(kernel, batch + kExtraBlocks, 1, 1, kThreads, 1, 1, 0, nullptr,
                              args, nullptr), "cuLaunchKernel") &&
         check(cuCtxSynchronize(), "cuCtxSynchronize") &&
         check(cuMemcpyDtoH(out.data(), dOut, out.size() * sizeof(float)), "copy back");
  }
  if (dWeights) cuMemFree(dWeights);
  if (dValues) cuMemFree(dValues);
  if (dOut) cuMemFree(dOut);
  if (!ok) return false;

  result.batch = batch;
  for (unsigned row = 0; row < batch; ++row) {
    for (unsigned lane = 0; lane < 4; ++lane) {
      float fused = 0.0f, separate = 0.0f;
      for (unsigned t = 0; t < 8; ++t) {
        const float w = weights[row * 8 + t];
        const float v = values[(row * 8 + t) * 4 + lane];
        fused = std::fmaf(w, v, fused);
        volatile float product = w * v;
        separate = separate + product;
      }
      ++result.outputs;
      if (bits(out[row * 4 + lane]) != bits(fused)) ++result.mismatches;
      if (bits(separate) != bits(fused)) ++result.mulAddDiffers;
    }
  }
  for (size_t k = 4 * size_t(batch); k < out.size(); ++k) {
    if (bits(out[k]) != kSentinel) ++result.guardRowWrites;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s tiny_attn_one_row.cubin results.json\n", argv[0]);
    return 2;
  }
  CUdevice device;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;
  if (!check(cuInit(0), "cuInit") || !check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !check(cuCtxCreate(&context, nullptr, 0, device), "cuCtxCreate") ||
      !check(cuModuleLoad(&module, argv[1]), "cuModuleLoad") ||
      !check(cuModuleGetFunction(&kernel, module, "tiny_attn_one_row"), "cuModuleGetFunction")) {
    return 1;
  }
  char name[256] = {};
  int major = 0, minor = 0, driver = 0;
  cuDeviceGetName(name, sizeof name, device);
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  cuDriverGetVersion(&driver);

  std::mt19937 rng(1234);
  std::vector<BatchResult> results;
  for (unsigned batch : {0u, 1u, 2u, 7u, 33u, 129u}) {
    BatchResult result;
    if (!run_batch(kernel, batch, rng, result)) return 1;
    results.push_back(result);
  }
  cuModuleUnload(module);
  cuCtxDestroy(context);

  unsigned outputs = 0, mismatches = 0, mulAddDiffers = 0, guardRowWrites = 0;
  for (const BatchResult& r : results) {
    outputs += r.outputs;
    mismatches += r.mismatches;
    mulAddDiffers += r.mulAddDiffers;
    guardRowWrites += r.guardRowWrites;
  }
  FILE* json = std::fopen(argv[2], "w");
  if (json == nullptr) {
    std::perror(argv[2]);
    return 1;
  }
  std::fprintf(json, "{\n");
  std::fprintf(json, "  \"cubin\": \"%s\",\n", argv[1]);
  std::fprintf(json, "  \"cubinSha256\": \"%s\",\n", sha256_file(argv[1]).c_str());
  std::fprintf(json, "  \"device\": \"%s\",\n", name);
  std::fprintf(json, "  \"computeCapability\": \"%d.%d\",\n", major, minor);
  std::fprintf(json, "  \"cudaDriverVersion\": %d,\n", driver);
  std::fprintf(json, "  \"inputs\": \"uniform binary32 in [-4, 4), std::mt19937 seed 1234\",\n");
  std::fprintf(json, "  \"reference\": \"std::fmaf chain over keys 0..7, starting from +0.0\",\n");
  std::fprintf(json, "  \"launch\": {\"blocks\": \"B + %u\", \"threadsPerBlock\": %u},\n",
               kExtraBlocks, kThreads);
  std::fprintf(json, "  \"batches\": [\n");
  for (size_t i = 0; i < results.size(); ++i) {
    const BatchResult& r = results[i];
    std::fprintf(json,
                 "    {\"B\": %u, \"outputs\": %u, \"bitwiseMismatchesVsFmaChain\": %u, "
                 "\"mulThenAddDiffers\": %u, \"guardRowWrites\": %u}%s\n",
                 r.batch, r.outputs, r.mismatches, r.mulAddDiffers, r.guardRowWrites,
                 i + 1 == results.size() ? "" : ",");
  }
  std::fprintf(json, "  ],\n");
  std::fprintf(json,
               "  \"total\": {\"outputs\": %u, \"bitwiseMismatchesVsFmaChain\": %u, "
               "\"mulThenAddDiffers\": %u, \"guardRowWrites\": %u}\n",
               outputs, mismatches, mulAddDiffers, guardRowWrites);
  std::fprintf(json, "}\n");
  std::fclose(json);
  std::printf("outputs=%u bitwise_mismatches=%u mul_then_add_differs=%u guard_row_writes=%u\n",
              outputs, mismatches, mulAddDiffers, guardRowWrites);
  return mismatches == 0 && guardRowWrites == 0 ? 0 : 1;
}
