#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/backend.h"

namespace quant {
namespace gpu {

class GPUComputeRpc {
public:
    GPUComputeRpc();
    ~GPUComputeRpc();

    bool init(const char* host = "127.0.0.1", int port = 9000);
    bool is_initialized() const;
    void shutdown();

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C);
    void relu(const Tensor& x, Tensor& y);
    void gelu(const Tensor& x, Tensor& y);
    void silu(const Tensor& x, Tensor& y);
    void softmax(const Tensor& x, Tensor& y, int axis);
    void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y);
    void add(const Tensor& a, const Tensor& b, Tensor& c);
    void mul(const Tensor& a, const Tensor& b, Tensor& c);

    void synchronize();
    int64_t memory_free() const;
    int64_t memory_total() const;

private:
    struct Impl;
    Impl* impl_;
};

GPUComputeRpc& get_rpc_compute();

} // namespace gpu
} // namespace quant
