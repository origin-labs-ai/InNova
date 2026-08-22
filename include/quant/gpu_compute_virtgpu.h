#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace quant {
namespace gpu {

class GPUComputeVirtGPU {
public:
    GPUComputeVirtGPU() = default;
    ~GPUComputeVirtGPU();

    bool init(int device_id = 0);
    bool is_initialized() const { return initialized_; }

    void* alloc(size_t size);
    void free_buf(void* ptr);

    void upload(const void* src, void* dst, size_t size);
    void download(const void* src, void* dst, size_t size);

    void gemm(float alpha, const void* A, const void* B, float beta, void* C, int64_t M, int64_t N, int64_t K);
    
    int64_t memory_free() const;
    int64_t memory_total() const;
    void synchronize();

private:
    bool initialized_ = false;
    int fd_ = -1;
};

GPUComputeVirtGPU& get_virtgpu_compute();

} // namespace gpu
} // namespace quant
