#include "quant/gpu_compute_webgpu.h"
#include <iostream>
#include <cstring>
#include <cstdlib>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace quant {
namespace gpu {

static void* load_webgpu_lib() {
#if defined(_WIN32)
    return LoadLibraryA("webgpu.dll");
#else
    return dlopen("libwebgpu.so", RTLD_LAZY | RTLD_LOCAL);
#endif
}

static void close_webgpu_lib(void* handle) {
#if defined(_WIN32)
    if (handle) FreeLibrary((HMODULE)handle);
#else
    if (handle) dlclose(handle);
#endif
}

static void* get_webgpu_sym(void* handle, const char* name) {
#if defined(_WIN32)
    return (void*)GetProcAddress((HMODULE)handle, name);
#else
    return dlsym(handle, name);
#endif
}

GPUComputeWebGPU::~GPUComputeWebGPU() {
    if (handle_) {
        close_webgpu_lib(handle_);
    }
}

bool GPUComputeWebGPU::init(int device_id) {
    if (initialized_) return true;
    handle_ = load_webgpu_lib();
    if (!handle_) return false;
    initialized_ = true;
    return true;
}

void* GPUComputeWebGPU::alloc(size_t size) {
    return std::malloc(size);
}

void GPUComputeWebGPU::free_buf(void* ptr) {
    std::free(ptr);
}

void GPUComputeWebGPU::upload(const void* src, void* dst, size_t size) {
    std::memcpy(dst, src, size);
}

void GPUComputeWebGPU::download(const void* src, void* dst, size_t size) {
    std::memcpy(dst, src, size);
}

void GPUComputeWebGPU::gemm(float alpha, const void* A, const void* B, float beta, void* C, int64_t M, int64_t N, int64_t K) {
    const float* a = static_cast<const float*>(A);
    const float* b = static_cast<const float*>(B);
    float* c = static_cast<float*>(C);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = alpha * sum + (beta == 0.0f ? 0.0f : beta * c[i * N + j]);
        }
    }
}

int64_t GPUComputeWebGPU::memory_free() const {
    return 1024LL * 1024 * 1024;
}

int64_t GPUComputeWebGPU::memory_total() const {
    return 1024LL * 1024 * 1024;
}

void GPUComputeWebGPU::synchronize() {
    // No-op
}

GPUComputeWebGPU& get_webgpu_compute() {
    static GPUComputeWebGPU instance;
    return instance;
}

} // namespace gpu
} // namespace quant
