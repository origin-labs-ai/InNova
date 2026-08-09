#pragma once
#include "quant/tensor.h"
#include "quant/types.h"
#include "quant/math.h"
#include <vector>
#include <string>

namespace quant {
namespace gpu {

// E10: Vulkan backend
class VulkanCompute {
public:
    VulkanCompute();
    ~VulkanCompute();
    bool init(int64_t device_id = 0);
    bool is_initialized() const;
    void* allocate(size_t bytes);
    void free(void* ptr);
    void upload(const Tensor& src, void* dst);
    void download(void* src, Tensor& dst);
    void synchronize();
    int64_t memory_free() const;
private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// E11: Metal backend (Apple GPU)
class MetalCompute {
public:
    MetalCompute();
    ~MetalCompute();
    bool init(int64_t device_id = 0);
    bool is_initialized() const;
    void synchronize();
private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// IGPU shared memory backend (uses DX12 shared heap for zero-copy)
class IGPUSharedBackend {
public:
    IGPUSharedBackend();
    ~IGPUSharedBackend();
    bool init(int64_t device_id = 0);
    bool is_initialized() const;
    void* allocate(size_t bytes);
    void free(void* ptr);
    void upload(const Tensor& src, void* dst);
    void download(void* src, Tensor& dst);
    void synchronize();
    int64_t memory_free() const;
    // Zero-copy: shared memory between CPU and GPU
    Tensor shared_tensor(const Shape& shape);
private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// E12: Multi-GPU manager
class MultiGPUManager {
public:
    MultiGPUManager();
    ~MultiGPUManager();
    int detect_devices();
    bool init_device(int idx);
    void* allocate_on(int idx, size_t bytes);
    void free_on(int idx, void* ptr);
    void transfer(int src_idx, int dst_idx, void* src, void* dst, size_t bytes);
    int device_count() const { return (int)devices_.size(); }
    std::string device_name(int idx) const;
private:
    struct DeviceInfo { bool initialized; std::string name; int64_t total_mem; };
    std::vector<DeviceInfo> devices_;
};

// E14: Auto-tuning for GEMM
struct TunedParams { int block_m = 64; int block_n = 64; int block_k = 16; };
TunedParams auto_tune_gemm(int64_t M, int64_t N, int64_t K);

// E15: Fallback — route GPU ops to CPU when GPU unavailable
class GPUFallback {
public:
    GPUFallback();
    template<typename T> void gemm(float a, const Tensor& A, const Tensor& B, float b, Tensor& C) {
        math::gemm(a, A, B, b, C);
    }
};

} // namespace gpu
} // namespace quant
