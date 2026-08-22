#include "quant/gpu_compute_virtgpu.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace quant {
namespace gpu {

GPUComputeVirtGPU::~GPUComputeVirtGPU() {
    if (fd_ >= 0) {
#if defined(__linux__) || defined(__APPLE__)
        close(fd_);
#endif
    }
}

bool GPUComputeVirtGPU::init(int device_id) {
    if (initialized_) return true;
#if defined(__linux__)
    fd_ = open("/dev/dri/renderD128", O_RDWR);
    if (fd_ < 0) {
        return false;
    }
#endif
    initialized_ = true;
    return true;
}

void* GPUComputeVirtGPU::alloc(size_t size) {
    return std::malloc(size); // Fallback memory allocation
}

void GPUComputeVirtGPU::free_buf(void* ptr) {
    std::free(ptr);
}

void GPUComputeVirtGPU::upload(const void* src, void* dst, size_t size) {
    std::memcpy(dst, src, size);
}

void GPUComputeVirtGPU::download(const void* src, void* dst, size_t size) {
    std::memcpy(dst, src, size);
}

void GPUComputeVirtGPU::gemm(float alpha, const void* A, const void* B, float beta, void* C, int64_t M, int64_t N, int64_t K) {
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

int64_t GPUComputeVirtGPU::memory_free() const {
    return 1024LL * 1024 * 1024;
}

int64_t GPUComputeVirtGPU::memory_total() const {
    return 1024LL * 1024 * 1024;
}

void GPUComputeVirtGPU::synchronize() {
    // No-op
}

GPUComputeVirtGPU& get_virtgpu_compute() {
    static GPUComputeVirtGPU instance;
    return instance;
}

} // namespace gpu
} // namespace quant
