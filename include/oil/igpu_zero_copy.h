#pragma once
#include "oil/tensor.h"
#include "oil/gpu_compute.h"
#include <cstdint>
#include <cstddef>
#include <string>

namespace oil {
namespace gpu {

struct IGPUDeviceInfo {
    std::string vendor;
    std::string device_name;
    int device_type = 0;
    uint64_t dedicated_vram = 0;
    uint64_t shared_memory = 0;
    uint32_t compute_queue_family = 0;
};

class IGPUZeroCopyAllocator {
public:
    IGPUZeroCopyAllocator();
    ~IGPUZeroCopyAllocator();

    bool init(int64_t device_id = 0);
    void shutdown();

    void* allocate_unified(size_t bytes);
    void free_unified(void* ptr);
    bool upload_direct(const void* src, void* dst, size_t bytes);
    bool download_direct(void* src, void* dst, size_t bytes);
    size_t available_unified() const;
    size_t total_unified() const;

    bool is_initialized() const;
    const IGPUDeviceInfo& device_info() const;

private:
    struct Impl;
    Impl* impl_;
};

class IGPUZeroCopyTrainer {
public:
    IGPUZeroCopyTrainer();
    ~IGPUZeroCopyTrainer();

    bool init();
    bool is_available() const;

    Tensor forward_linear(const Tensor& input, const Tensor& weight, const Tensor& bias);
    Tensor forward_attention(const Tensor& Q, const Tensor& K, const Tensor& V, bool causal);
    Tensor forward_rms_norm(const Tensor& x, const Tensor& gamma, float eps);
    Tensor forward_swiglu(const Tensor& gate, const Tensor& up);
    float compute_loss(const Tensor& logits, const Tensor& targets);
    bool train_step(const Tensor& input_ids, const Tensor& labels, float& loss_out);

    void synchronize();

private:
    struct Impl;
    Impl* impl_;
};

IGPUZeroCopyTrainer& get_igpu_trainer();

} // namespace gpu
} // namespace oil
