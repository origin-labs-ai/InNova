#pragma once
#include "quant/tensor.h"
#include "quant/gpu_compute.h"
#include <vector>
#include <cstdint>
#include <string>
#include <mutex>

namespace quant {
namespace gpu {

struct KernelProfile {
    std::string name;
    double total_ms = 0.0;
    int64_t call_count = 0;
    double avg_ms() const { return call_count > 0 ? total_ms / (double)call_count : 0.0; }
};

struct GPUBufferPoolStats {
    int64_t allocated_bytes = 0;
    int64_t free_bytes = 0;
    int64_t total_bytes = 0;
    int64_t num_allocations = 0;
    int64_t num_free_blocks = 0;
};

class GPUComputeFull : public GPUBackendInterface {
public:
    GPUComputeFull();
    ~GPUComputeFull() override;

    bool init(int64_t device_id = 0) override;
    bool is_initialized() const override;
    void shutdown() override;
    bool has_gpu() const;

    void* alloc(int64_t bytes);
    void free_buf(void* ptr);
    void upload(const Tensor& src, void* dst) override;
    void download(void* src, Tensor& dst) override;
    void gpu_to_gpu(void* dst, const void* src, int64_t bytes);
    GPUBufferPoolStats buffer_pool_stats() const;
    int64_t memory_free() const override;
    int64_t memory_total() const override;

    void gemm_tiled(float alpha, const void* A, const void* B, float beta, void* C,
                    int64_t M, int64_t N, int64_t K, int64_t tile_size = 16);
    void gemm_async(float alpha, const void* A, const void* B, float beta, void* C,
                    int64_t M, int64_t N, int64_t K);

    void element_add(const void* a, const void* b, void* c, int64_t n);
    void element_sub(const void* a, const void* b, void* c, int64_t n);
    void element_mul(const void* a, const void* b, void* c, int64_t n);
    void element_div(const void* a, const void* b, void* c, int64_t n);
    void element_add_scalar(const void* a, float s, void* c, int64_t n);

    void activation_relu(const void* x, void* y, int64_t n);
    void activation_gelu(const void* x, void* y, int64_t n);
    void activation_silu(const void* x, void* y, int64_t n);
    void activation_tanh(const void* x, void* y, int64_t n);
    void activation_sigmoid(const void* x, void* y, int64_t n);
    void activation_softmax(const void* x, void* y, int64_t rows, int64_t cols);

    void reduce_sum(const float* x, float* out, int64_t n);
    void reduce_max(const float* x, float* out, int64_t n);
    void reduce_mean(const float* x, float* out, int64_t n);
    void reduce_sum_axis(const float* x, float* out, int64_t rows, int64_t cols, int axis);
    void reduce_max_axis(const float* x, float* out, int64_t rows, int64_t cols, int axis);

    void attention_fwd(const void* Q, const void* K, const void* V, void* out,
                       int64_t B, int64_t H, int64_t T, int64_t D, bool causal);
    void attention_cross(const void* Q, const void* KV, void* out,
                         int64_t B, int64_t H, int64_t Tq, int64_t Tk, int64_t D);

    void layer_norm(const void* x, const void* gamma, const void* beta, void* y,
                    float eps, int64_t n, int64_t d) override;
    void rms_norm(const void* x, const void* gamma, void* y,
                  float eps, int64_t n, int64_t d) override;
    void batch_norm(const void* x, const void* gamma, const void* beta,
                    void* y, int64_t n, int64_t c, int64_t hw);

    void softmax_stable(const void* x, void* y, int64_t rows, int64_t cols);

    void synchronize() override;
    void profile_reset();
    std::vector<KernelProfile> profile_dump() const;

    void async_memcpy_h2d(void* dst, const void* src, int64_t bytes);
    void async_memcpy_d2h(void* dst, const void* src, int64_t bytes);
    void stream_synchronize(int stream_idx);
    int create_stream();
    void destroy_stream(int stream_idx);

    void transpose_2d(const void* x, void* y, int64_t rows, int64_t cols);
    void embedding_lookup(const float* table, const int64_t* indices, float* out,
                          int64_t n, int64_t d);

    static GPUComputeFull& instance();

private:
    struct Impl;
    Impl* impl_;

    struct CPUTensorFallback {
        static void gemm_fallback(float alpha, const void* A, const void* B,
                                  float beta, void* C, int64_t M, int64_t N, int64_t K);
        static void softmax_fallback(const float* x, float* y, int64_t rows, int64_t cols);
        static void layernorm_fallback(const float* x, const float* gamma, const float* beta,
                                       float* y, float eps, int64_t n, int64_t d);
        static void rmsnorm_fallback(const float* x, const float* gamma,
                                     float* y, float eps, int64_t n, int64_t d);
    };
};

} // namespace gpu
} // namespace quant
