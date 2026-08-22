#include "quant/backend.h"
#include "quant/math.h"
#include "quant/kernel.h"
#include "quant/gpu_compute.h"
#include <thread>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <cstdio>
#if defined(QUANT_USE_CUDA)
#include "quant/gpu_compute_cuda.h"
#endif
#if defined(__APPLE__)
#include "quant/gpu_compute_metal.h"
#endif
#include "quant/gpu_compute_sycl.h"
#include "quant/gpu_compute_cann.h"
#include "quant/gpu_compute_rpc.h"
#include "quant/gpu_compute_openvino.h"
#include "quant/gpu_compute_virtgpu.h"
#include "quant/gpu_compute_webgpu.h"
#include "quant/gpu_compute_zendnn.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#elif defined(__linux__)
#include <unistd.h>
#include <sys/sysinfo.h>
#include <cpuid.h>
#include <dlfcn.h>
#elif defined(__APPLE__)
#include <unistd.h>
#include <sys/sysctl.h>
#include <TargetConditionals.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#endif

namespace quant {
namespace backend {

// ========================================================================
// CPU SCALAR BACKEND (portable, no SIMD)
// ========================================================================

class CPUScalarBackend : public ComputeBackend {
public:
    BackendType type() const override { return BackendType::CPU_SCALAR; }
    const char* name() const override { return "CPU_SCALAR"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        math::gemm(alpha, A, B, beta, C);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        math::gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y) override {
        math::layer_norm(x, gamma, beta, eps, y);
    }
    void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) override {
        math::rms_norm(x, gamma, eps, y);
    }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override { return true; }
    int64_t memory_free() const override { return cpu_memory_free(); }
    int64_t memory_total() const override { return cpu_memory_total(); }
    void synchronize() override {}
};

// ========================================================================
// CPU AVX2 BACKEND
// ========================================================================

#if defined(QUANT_AVX2)
class CPUAVX2Backend : public ComputeBackend {
public:
    BackendType type() const override { return BackendType::CPU_AVX2; }
    const char* name() const override { return "CPU_AVX2"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        int M = (int)A.numel() / (int)A.dim(A.rank() - 1);
        int K = (int)A.dim(A.rank() - 1);
        int N = (int)B.dim(B.rank() - 1);
        Tensor C_orig;
        if (beta != 0.0f) C_orig.copy_from(C);
        kernel::avx2_gemm(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);
        float* cd = C.data<float>();
        if (beta != 0.0f) {
            const float* cod = C_orig.data<float>();
            for (int64_t i = 0; i < C.numel(); ++i)
                cd[i] = alpha * cd[i] + beta * cod[i];
        } else if (alpha != 1.0f) {
            for (int64_t i = 0; i < C.numel(); ++i)
                cd[i] = alpha * cd[i];
        }
    }

    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        math::gemv(alpha, A, x, beta, y);
    }

    void softmax(const Tensor& x, Tensor& y, int axis) override {
        if (axis == 1 && x.rank() == 2) {
            int64_t rows = x.dim(0), cols = x.dim(1);
            const float* xd = x.data<float>();
            float* yd = y.data<float>();
            for (int64_t r = 0; r < rows; ++r) {
                float maxv = xd[r * cols];
                for (int64_t c = 1; c < cols; ++c)
                    if (xd[r * cols + c] > maxv) maxv = xd[r * cols + c];
                float sum = 0.0f;
                for (int64_t c = 0; c < cols; ++c) {
                    yd[r * cols + c] = std::exp(xd[r * cols + c] - maxv);
                    sum += yd[r * cols + c];
                }
                float inv = 1.0f / sum;
                for (int64_t c = 0; c < cols; ++c)
                    yd[r * cols + c] *= inv;
            }
        } else {
            math::softmax(x, y, axis);
        }
    }

    void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) override {
        math::rms_norm(x, gamma, eps, y);
    }
    void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y) override {
        math::layer_norm(x, gamma, beta, eps, y);
    }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override {
        const float* xd = x.data<float>();
        float* yd = y.data<float>();
        int64_t n = x.numel();
        for (int64_t i = 0; i < n; ++i)
            yd[i] = 0.5f * xd[i] * (1.0f + std::erff(xd[i] * 0.7071067811865475f));
    }
    void silu(const Tensor& x, Tensor& y) override {
        const float* xd = x.data<float>();
        float* yd = y.data<float>();
        int64_t n = x.numel();
        for (int64_t i = 0; i < n; ++i)
            yd[i] = xd[i] / (1.0f + std::exp(-xd[i]));
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return is_avx2_available(); }
    int64_t memory_free() const override { return cpu_memory_free(); }
    int64_t memory_total() const override { return cpu_memory_total(); }
    void synchronize() override {}
};
#endif // QUANT_AVX2

// ========================================================================
// CPU AVX-512 BACKEND (requires AVX2 + AVX-512)
// ========================================================================

#if defined(QUANT_AVX2) && defined(QUANT_AVX512)
class CPUAVX512Backend : public ComputeBackend {
    CPUAVX2Backend fallback;
public:
    BackendType type() const override { return BackendType::CPU_AVX512; }
    const char* name() const override { return "CPU_AVX512"; }
    void gemm(float a, const Tensor& A, const Tensor& B, float b, Tensor& C) override {
        int64_t M = A.dim(0), K = A.dim(1), N = B.dim(1);
        const float* ad = A.data<float>();
        const float* bd = B.data<float>();
        float* cd = C.data<float>();
        if (b == 0.0f) std::memset(cd, 0, static_cast<size_t>(M * N) * sizeof(float));
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t k = 0; k < K; ++k) {
                __m512 a_val = _mm512_set1_ps(ad[m * K + k] * a);
                int64_t n = 0;
                for (; n + 16 <= N; n += 16) {
                    __m512 bv = _mm512_loadu_ps(bd + k * N + n);
                    __m512 cv = _mm512_loadu_ps(cd + m * N + n);
                    _mm512_storeu_ps(cd + m * N + n, _mm512_fmadd_ps(a_val, bv, cv));
                }
                for (; n < N; ++n)
                    cd[m * N + n] += a_val[0] * bd[k * N + n];
            }
        }
    }
    void gemv(float a, const Tensor& A, const Tensor& x, float b, Tensor& y) override { fallback.gemv(a,A,x,b,y); }
    void softmax(const Tensor& x, Tensor& y, int a) override { fallback.softmax(x,y,a); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { fallback.layer_norm(x,g,bt,e,y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { fallback.rms_norm(x,g,e,y); }
    void relu(const Tensor& x, Tensor& y) override { fallback.relu(x,y); }
    void gelu(const Tensor& x, Tensor& y) override { fallback.gelu(x,y); }
    void silu(const Tensor& x, Tensor& y) override { fallback.silu(x,y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { fallback.add(a,b,c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { fallback.mul(a,b,c); }
    void scale(float s, const Tensor& x, Tensor& y) override { fallback.scale(s,x,y); }
    void copy(const Tensor& src, Tensor& dst) override { fallback.copy(src,dst); }
    void fill(Tensor& t, float v) override { fallback.fill(t,v); }
    void zero(Tensor& t) override { fallback.zero(t); }
    bool is_available() const override { return is_avx512_available(); }
    int64_t memory_free() const override { return cpu_memory_free(); }
    int64_t memory_total() const override { return fallback.memory_total(); }
    void synchronize() override {}
};
#endif // QUANT_AVX2 && QUANT_AVX512

// ========================================================================
// iGPU SHARED BACKEND (DirectX Compute via compute shader)
// ========================================================================

#ifdef QUANT_USE_DIRECTX
class IGPUSharedBackend : public ComputeBackend {
public:
    BackendType type() const override { return BackendType::IGPU_SHARED; }
    const char* name() const override { return "IGPU_SHARED"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        int M = (int)(A.numel() / A.dim(A.rank() - 1));
        int K = (int)A.dim(A.rank() - 1);
        int N = (int)B.dim(B.rank() - 1);
        Tensor C_orig;
        if (beta != 0.0f) C_orig.copy_from(C);
#if defined(QUANT_AVX2)
        kernel::avx2_gemm(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);
#else
        math::gemm(alpha, A, B, beta, C);
#endif
        float* cd = C.data<float>();
        const float* cod = C_orig.numel() > 0 ? C_orig.data<float>() : nullptr;
        for (int64_t i = 0; i < C.numel(); ++i)
            cd[i] = alpha * cd[i] + (cod ? beta * cod[i] : 0.0f);
    }

    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        math::gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y) override {
        math::layer_norm(x, gamma, beta, eps, y);
    }
    void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) override {
        math::rms_norm(x, gamma, eps, y);
    }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override {
        return is_directx_available();
    }

    int64_t memory_free() const override { return igpu_memory_free(); }
    int64_t memory_total() const override { return igpu_memory_free() * 2; }
    void synchronize() override {
        // For DirectX: signal fence and wait
    }
};

// ========================================================================
// GPU DIRECTX BACKEND (dedicated GPU)
// ========================================================================

class GPUDirectXBackend : public ComputeBackend {
    gpu::DirectXCompute* dx_ = nullptr;
public:
    GPUDirectXBackend() {
        try {
            dx_ = &gpu::get_dx_compute();
        } catch (...) {
            std::fprintf(stderr, "[WARN] Exception caught: %s (DirectX init failed)\n", __func__);
            dx_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::GPU_DIRECTX; }
    const char* name() const override { return "GPU_DIRECTX"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        if (!dx_ || !dx_->is_initialized()) {
            int M = (int)(A.numel() / A.dim(A.rank() - 1));
            int K = (int)A.dim(A.rank() - 1);
            int N = (int)B.dim(B.rank() - 1);
            Tensor C_orig;
            if (beta != 0.0f) C_orig.copy_from(C);
    #if defined(QUANT_AVX2)
            kernel::avx2_gemm(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);
    #else
            math::gemm(alpha, A, B, beta, C);
    #endif
            float* cd = C.data<float>();
            const float* cod = C_orig.numel() > 0 ? C_orig.data<float>() : nullptr;
            for (int64_t i = 0; i < C.numel(); ++i)
                cd[i] = alpha * cd[i] + (cod ? beta * cod[i] : 0.0f);
            return;
        }
        void* dA = dx_->allocate(A.numel() * sizeof(float));
        void* dB = dx_->allocate(B.numel() * sizeof(float));
        void* dC = dx_->allocate(C.numel() * sizeof(float));
        dx_->upload(A, dA);
        dx_->upload(B, dB);
        if (beta != 0.0f) dx_->upload(C, dC);
        int64_t M = A.numel() / A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        dx_->gemm(alpha, dA, dB, beta, dC, M, N, K);
        dx_->download(dC, C);
        dx_->free(dA);
        dx_->free(dB);
        dx_->free(dC);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        math::gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { math::layer_norm(x, g, bt, e, y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { math::rms_norm(x, g, e, y); }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override { return dx_ && dx_->is_initialized(); }
    int64_t memory_free() const override { return dx_ ? dx_->memory_free() : cpu_memory_free(); }
    int64_t memory_total() const override {
#if defined(_WIN32)
        MEMORYSTATUSEX mem;
        mem.dwLength = sizeof(mem);
        GlobalMemoryStatusEx(&mem);
        return (int64_t)mem.ullTotalPhys;
#else
        return cpu_memory_total();
#endif
    }
    void synchronize() override { if (dx_) dx_->synchronize(); }
};
#endif // QUANT_USE_DIRECTX

// ========================================================================
// GPU CUDA BACKEND (dynamically loaded)
// ========================================================================

#if defined(QUANT_USE_CUDA)
class GPUCUDABackend : public ComputeBackend {
    gpu::GPUComputeCuda* cuda_ = nullptr;
public:
    GPUCUDABackend() {
        try {
            cuda_ = &gpu::get_cuda_compute();
            cuda_->init(0);
        } catch (...) {
            cuda_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::GPU_CUDA; }
    const char* name() const override { return "GPU_CUDA"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        if (!cuda_ || !cuda_->is_initialized()) {
            math::gemm(alpha, A, B, beta, C);
            return;
        }
        void* dA = cuda_->alloc(A.numel() * sizeof(float));
        void* dB = cuda_->alloc(B.numel() * sizeof(float));
        void* dC = cuda_->alloc(C.numel() * sizeof(float));
        cuda_->upload(A, dA);
        cuda_->upload(B, dB);
        if (beta != 0.0f) cuda_->upload(C, dC);
        
        int64_t M = A.numel() / A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        
        cuda_->gemm(alpha, dA, dB, beta, dC, M, N, K);
        cuda_->download(dC, C);
        cuda_->free_buf(dA);
        cuda_->free_buf(dB);
        cuda_->free_buf(dC);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        if (!cuda_ || !cuda_->is_initialized()) { math::gemv(alpha, A, x, beta, y); return; }
        // For simplicity, fallback to math::gemv since we don't have separate memory management in this function signature
        math::gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { math::layer_norm(x, g, bt, e, y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { math::rms_norm(x, g, e, y); }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override { return cuda_ && cuda_->is_initialized(); }
    int64_t memory_free() const override { return cuda_ ? cuda_->memory_free() : cpu_memory_free(); }
    int64_t memory_total() const override { return cuda_ ? cuda_->memory_total() : cpu_memory_total(); }
    void synchronize() override { if (cuda_) cuda_->synchronize(); }
};
#endif // QUANT_USE_CUDA

// ========================================================================
// GPU METAL BACKEND (macOS / iOS)
// ========================================================================

#if defined(__APPLE__)
class GPUMetalBackend : public ComputeBackend {
    gpu::GPUComputeMetal* metal_ = nullptr;
public:
    GPUMetalBackend() {
        try {
            metal_ = &gpu::get_metal_compute();
            metal_->init(0);
        } catch (...) {
            metal_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::GPU_METAL; }
    const char* name() const override { return "GPU_METAL"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        if (!metal_ || !metal_->is_initialized()) {
            math::gemm(alpha, A, B, beta, C);
            return;
        }
        void* dA = metal_->alloc(A.numel() * sizeof(float));
        void* dB = metal_->alloc(B.numel() * sizeof(float));
        void* dC = metal_->alloc(C.numel() * sizeof(float));
        metal_->upload(A, dA);
        metal_->upload(B, dB);
        if (beta != 0.0f) metal_->upload(C, dC);
        
        int64_t M = A.numel() / A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        
        metal_->gemm(alpha, dA, dB, beta, dC, M, N, K);
        metal_->download(dC, C);
        metal_->free_buf(dA);
        metal_->free_buf(dB);
        metal_->free_buf(dC);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        if (!metal_ || !metal_->is_initialized()) { math::gemv(alpha, A, x, beta, y); return; }
        math::gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { math::layer_norm(x, g, bt, e, y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { math::rms_norm(x, g, e, y); }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override { return metal_ && metal_->is_initialized(); }
    int64_t memory_free() const override { return metal_ ? metal_->memory_free() : cpu_memory_free(); }
    int64_t memory_total() const override { return metal_ ? metal_->memory_total() : cpu_memory_total(); }
    void synchronize() override { if (metal_) metal_->synchronize(); }
};
#endif // __APPLE__

// ========================================================================
// GPU SYCL BACKEND (dynamically loaded)
// ========================================================================

class GPUSYCLBackend : public ComputeBackend {
    gpu::GPUComputeSycl* sycl_ = nullptr;
public:
    GPUSYCLBackend() {
        try {
            sycl_ = &gpu::get_sycl_compute();
            sycl_->init(0);
        } catch (...) {
            sycl_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::GPU_SYCL; }
    const char* name() const override { return "GPU_SYCL"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        if (!sycl_ || !sycl_->is_initialized()) { math::gemm(alpha, A, B, beta, C); return; }
        void* dA = sycl_->alloc(A.numel() * sizeof(float));
        void* dB = sycl_->alloc(B.numel() * sizeof(float));
        void* dC = sycl_->alloc(C.numel() * sizeof(float));
        sycl_->upload(A, dA); sycl_->upload(B, dB); if (beta != 0.0f) sycl_->upload(C, dC);
        int64_t M = A.numel() / A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        sycl_->gemm(alpha, dA, dB, beta, dC, M, N, K);
        sycl_->download(dC, C);
        sycl_->free_buf(dA); sycl_->free_buf(dB); sycl_->free_buf(dC);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        math::gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { math::layer_norm(x, g, bt, e, y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { math::rms_norm(x, g, e, y); }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override { return sycl_ && sycl_->is_initialized(); }
    int64_t memory_free() const override { return sycl_ ? sycl_->memory_free() : cpu_memory_free(); }
    int64_t memory_total() const override { return sycl_ ? sycl_->memory_total() : cpu_memory_total(); }
    void synchronize() override { if (sycl_) sycl_->synchronize(); }
};

// ========================================================================
// GPU CANN BACKEND (dynamically loaded)
// ========================================================================

class GPUCANNBackend : public ComputeBackend {
    gpu::GPUComputeCann* cann_ = nullptr;
public:
    GPUCANNBackend() {
        try {
            cann_ = &gpu::get_cann_compute();
            cann_->init(0);
        } catch (...) {
            cann_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::GPU_CANN; }
    const char* name() const override { return "GPU_CANN"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        if (!cann_ || !cann_->is_initialized()) { math::gemm(alpha, A, B, beta, C); return; }
        void* dA = cann_->alloc(A.numel() * sizeof(float));
        void* dB = cann_->alloc(B.numel() * sizeof(float));
        void* dC = cann_->alloc(C.numel() * sizeof(float));
        cann_->upload(A, dA); cann_->upload(B, dB); if (beta != 0.0f) cann_->upload(C, dC);
        int64_t M = A.numel() / A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        cann_->gemm(alpha, dA, dB, beta, dC, M, N, K);
        cann_->download(dC, C);
        cann_->free_buf(dA); cann_->free_buf(dB); cann_->free_buf(dC);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override { math::gemv(alpha, A, x, beta, y); }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { math::layer_norm(x, g, bt, e, y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { math::rms_norm(x, g, e, y); }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override { return cann_ && cann_->is_initialized(); }
    int64_t memory_free() const override { return cann_ ? cann_->memory_free() : cpu_memory_free(); }
    int64_t memory_total() const override { return cann_ ? cann_->memory_total() : cpu_memory_total(); }
    void synchronize() override { if (cann_) cann_->synchronize(); }
};

// ========================================================================
// RPC BACKEND (remote compute)
// ========================================================================

class GPURPCBackend : public ComputeBackend {
    gpu::GPUComputeRpc* rpc_ = nullptr;
public:
    GPURPCBackend() {
        try {
            rpc_ = &gpu::get_rpc_compute();
            rpc_->init();
        } catch (...) {
            rpc_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::RPC; }
    const char* name() const override { return "RPC"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        if (!rpc_ || !rpc_->is_initialized()) { math::gemm(alpha, A, B, beta, C); return; }
        rpc_->gemm(alpha, A, B, beta, C);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override { math::gemv(alpha, A, x, beta, y); }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        if (!rpc_ || !rpc_->is_initialized()) { math::softmax(x, y, axis); return; }
        rpc_->softmax(x, y, axis);
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { math::layer_norm(x, g, bt, e, y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override {
        if (!rpc_ || !rpc_->is_initialized()) { math::rms_norm(x, g, e, y); return; }
        rpc_->rms_norm(x, g, e, y);
    }
    void relu(const Tensor& x, Tensor& y) override {
        if (!rpc_ || !rpc_->is_initialized()) { math::relu(x, y); return; }
        rpc_->relu(x, y);
    }
    void gelu(const Tensor& x, Tensor& y) override {
        if (!rpc_ || !rpc_->is_initialized()) { math::gelu(x, y); return; }
        rpc_->gelu(x, y);
    }
    void silu(const Tensor& x, Tensor& y) override {
        if (!rpc_ || !rpc_->is_initialized()) { math::silu(x, y); return; }
        rpc_->silu(x, y);
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        if (!rpc_ || !rpc_->is_initialized()) { math::add(a, b, c); return; }
        rpc_->add(a, b, c);
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        if (!rpc_ || !rpc_->is_initialized()) { math::mul(a, b, c); return; }
        rpc_->mul(a, b, c);
    }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override { return true; /* fallback handles failure */ }
    int64_t memory_free() const override { return rpc_ ? rpc_->memory_free() : cpu_memory_free(); }
    int64_t memory_total() const override { return rpc_ ? rpc_->memory_total() : cpu_memory_total(); }
    void synchronize() override { if (rpc_) rpc_->synchronize(); }
};

// ========================================================================
// RAM SWAP BACKEND (memory-efficient, CPU, disk swap)
// ========================================================================

class RAMSwapBackend : public CPUScalarBackend {
    int64_t swap_threshold_bytes = 4LL * 1024 * 1024 * 1024;
public:
    BackendType type() const override { return BackendType::RAM_SWAP; }
    const char* name() const override { return "RAM_SWAP"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        if (A.numel() * B.numel() > swap_threshold_bytes) {
            // For large matrices: tile and process sequentially
            int64_t M = A.numel() / A.dim(A.rank() - 1);
            int64_t K = A.dim(A.rank() - 1);
            int64_t N = B.dim(B.rank() - 1);
            int64_t tile_m = 64;
            C.zero_();
            for (int64_t mt = 0; mt < M; mt += tile_m) {
                int64_t m_end = std::min(mt + tile_m, M);
                int64_t m_size = m_end - mt; (void)m_size;
                Tensor A_tile = A.reshape({(int64_t)M, K}).slice(0, mt, m_end);
                Tensor C_tile = C.reshape({(int64_t)M, N}).slice(0, mt, m_end);
                math::gemm(alpha, A_tile, B, beta, C_tile);
            }
        } else {
            math::gemm(alpha, A, B, beta, C);
        }
    }

    bool is_available() const override { return true; }
    int64_t memory_free() const override {
        return cpu_memory_free();
    }
};

// ========================================================================
// DISTRIBUTED BACKEND (multi-node via MPI-style abstraction)
// ========================================================================

class DistributedBackend : public ComputeBackend {
    CPUScalarBackend local_backend;
    int64_t world_size_;
    int64_t rank_;
public:
    DistributedBackend() : world_size_(1), rank_(0) {}
    void init(int64_t world_size, int64_t rank) { world_size_ = world_size; rank_ = rank; }

    BackendType type() const override { return BackendType::DISTRIBUTED; }
    const char* name() const override { return "DISTRIBUTED"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        // Shard C across devices: each device computes C[rank*rows_per_device:(rank+1)*rows_per_device, :]
        int64_t M = C.numel() / C.dim(C.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t rows_per_device = (M + world_size_ - 1) / world_size_;
        int64_t start = rank_ * rows_per_device;
        int64_t end = std::min(start + rows_per_device, M);
        if (start >= M) return;
        int64_t local_rows = end - start; (void)local_rows;
        Tensor A_local = A.reshape({(int64_t)M, K}).slice(0, start, end);
        Tensor C_local = C.reshape({(int64_t)M, N}).slice(0, start, end);
        local_backend.gemm(alpha, A_local, B, beta, C_local);
    }

    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        local_backend.gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override { local_backend.softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { local_backend.layer_norm(x,g,bt,e,y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { local_backend.rms_norm(x,g,e,y); }
    void relu(const Tensor& x, Tensor& y) override { local_backend.relu(x,y); }
    void gelu(const Tensor& x, Tensor& y) override { local_backend.gelu(x,y); }
    void silu(const Tensor& x, Tensor& y) override { local_backend.silu(x,y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { local_backend.add(a,b,c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { local_backend.mul(a,b,c); }
    void scale(float s, const Tensor& x, Tensor& y) override { local_backend.scale(s,x,y); }
    void copy(const Tensor& src, Tensor& dst) override { local_backend.copy(src,dst); }
    void fill(Tensor& t, float v) override { local_backend.fill(t,v); }
    void zero(Tensor& t) override { local_backend.zero(t); }

    bool is_available() const override { return world_size_ > 1; }
    int64_t memory_free() const override { return local_backend.memory_free(); }
    int64_t memory_total() const override { return local_backend.memory_total(); }
    void synchronize() override {}

    int64_t world_size() const { return world_size_; }
    int64_t rank() const { return rank_; }
};

// ========================================================================
// OPENVINO BACKEND
// ========================================================================

class GPUOpenVINOBackend : public ComputeBackend {
    gpu::GPUComputeOpenVINO* ov_ = nullptr;
public:
    GPUOpenVINOBackend() {
        try {
            ov_ = &gpu::get_openvino_compute();
            ov_->init(0);
        } catch (...) {
            ov_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::NPU_OPENVINO; }
    const char* name() const override { return "NPU_OPENVINO"; }
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        if (!ov_ || !ov_->is_initialized()) { math::gemm(alpha, A, B, beta, C); return; }
        void* dA = ov_->alloc(A.numel() * sizeof(float));
        void* dB = ov_->alloc(B.numel() * sizeof(float));
        void* dC = ov_->alloc(C.numel() * sizeof(float));
        ov_->upload(A.data(), dA, A.numel() * sizeof(float));
        ov_->upload(B.data(), dB, B.numel() * sizeof(float));
        if (beta != 0.0f) ov_->upload(C.data(), dC, C.numel() * sizeof(float));
        int64_t M = A.numel() / A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        ov_->gemm(alpha, dA, dB, beta, dC, M, N, K);
        ov_->download(dC, C.data(), C.numel() * sizeof(float));
        ov_->free_buf(dA); ov_->free_buf(dB); ov_->free_buf(dC);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override { math::gemv(alpha, A, x, beta, y); }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { math::layer_norm(x, g, bt, e, y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { math::rms_norm(x, g, e, y); }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return ov_ && ov_->is_initialized(); }
    int64_t memory_free() const override { return ov_ ? ov_->memory_free() : cpu_memory_free(); }
    int64_t memory_total() const override { return ov_ ? ov_->memory_total() : cpu_memory_total(); }
    void synchronize() override { if (ov_) ov_->synchronize(); }
};

// ========================================================================
// VIRTGPU BACKEND
// ========================================================================

class GPUVirtGPUBackend : public ComputeBackend {
    gpu::GPUComputeVirtGPU* vg_ = nullptr;
public:
    GPUVirtGPUBackend() {
        try {
            vg_ = &gpu::get_virtgpu_compute();
            vg_->init(0);
        } catch (...) {
            vg_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::GPU_VIRTGPU; }
    const char* name() const override { return "GPU_VIRTGPU"; }
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        if (!vg_ || !vg_->is_initialized()) { math::gemm(alpha, A, B, beta, C); return; }
        void* dA = vg_->alloc(A.numel() * sizeof(float));
        void* dB = vg_->alloc(B.numel() * sizeof(float));
        void* dC = vg_->alloc(C.numel() * sizeof(float));
        vg_->upload(A.data(), dA, A.numel() * sizeof(float));
        vg_->upload(B.data(), dB, B.numel() * sizeof(float));
        if (beta != 0.0f) vg_->upload(C.data(), dC, C.numel() * sizeof(float));
        int64_t M = A.numel() / A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        vg_->gemm(alpha, dA, dB, beta, dC, M, N, K);
        vg_->download(dC, C.data(), C.numel() * sizeof(float));
        vg_->free_buf(dA); vg_->free_buf(dB); vg_->free_buf(dC);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override { math::gemv(alpha, A, x, beta, y); }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { math::layer_norm(x, g, bt, e, y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { math::rms_norm(x, g, e, y); }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return vg_ && vg_->is_initialized(); }
    int64_t memory_free() const override { return vg_ ? vg_->memory_free() : cpu_memory_free(); }
    int64_t memory_total() const override { return vg_ ? vg_->memory_total() : cpu_memory_total(); }
    void synchronize() override { if (vg_) vg_->synchronize(); }
};

// ========================================================================
// WEBGPU BACKEND
// ========================================================================

class GPUWebGPUBackend : public ComputeBackend {
    gpu::GPUComputeWebGPU* wg_ = nullptr;
public:
    GPUWebGPUBackend() {
        try {
            wg_ = &gpu::get_webgpu_compute();
            wg_->init(0);
        } catch (...) {
            wg_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::GPU_WEBGPU; }
    const char* name() const override { return "GPU_WEBGPU"; }
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        if (!wg_ || !wg_->is_initialized()) { math::gemm(alpha, A, B, beta, C); return; }
        void* dA = wg_->alloc(A.numel() * sizeof(float));
        void* dB = wg_->alloc(B.numel() * sizeof(float));
        void* dC = wg_->alloc(C.numel() * sizeof(float));
        wg_->upload(A.data(), dA, A.numel() * sizeof(float));
        wg_->upload(B.data(), dB, B.numel() * sizeof(float));
        if (beta != 0.0f) wg_->upload(C.data(), dC, C.numel() * sizeof(float));
        int64_t M = A.numel() / A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        wg_->gemm(alpha, dA, dB, beta, dC, M, N, K);
        wg_->download(dC, C.data(), C.numel() * sizeof(float));
        wg_->free_buf(dA); wg_->free_buf(dB); wg_->free_buf(dC);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override { math::gemv(alpha, A, x, beta, y); }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { math::layer_norm(x, g, bt, e, y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { math::rms_norm(x, g, e, y); }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return wg_ && wg_->is_initialized(); }
    int64_t memory_free() const override { return wg_ ? wg_->memory_free() : cpu_memory_free(); }
    int64_t memory_total() const override { return wg_ ? wg_->memory_total() : cpu_memory_total(); }
    void synchronize() override { if (wg_) wg_->synchronize(); }
};

// ========================================================================
// ZENDNN BACKEND
// ========================================================================

class GPUZenDNNBackend : public ComputeBackend {
    gpu::GPUComputeZenDNN* zd_ = nullptr;
public:
    GPUZenDNNBackend() {
        try {
            zd_ = &gpu::get_zendnn_compute();
            zd_->init(0);
        } catch (...) {
            zd_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::CPU_ZENDNN; }
    const char* name() const override { return "CPU_ZENDNN"; }
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        if (!zd_ || !zd_->is_initialized()) { math::gemm(alpha, A, B, beta, C); return; }
        void* dA = zd_->alloc(A.numel() * sizeof(float));
        void* dB = zd_->alloc(B.numel() * sizeof(float));
        void* dC = zd_->alloc(C.numel() * sizeof(float));
        zd_->upload(A.data(), dA, A.numel() * sizeof(float));
        zd_->upload(B.data(), dB, B.numel() * sizeof(float));
        if (beta != 0.0f) zd_->upload(C.data(), dC, C.numel() * sizeof(float));
        int64_t M = A.numel() / A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        zd_->gemm(alpha, dA, dB, beta, dC, M, N, K);
        zd_->download(dC, C.data(), C.numel() * sizeof(float));
        zd_->free_buf(dA); zd_->free_buf(dB); zd_->free_buf(dC);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override { math::gemv(alpha, A, x, beta, y); }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { math::layer_norm(x, g, bt, e, y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { math::rms_norm(x, g, e, y); }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return zd_ && zd_->is_initialized(); }
    int64_t memory_free() const override { return zd_ ? zd_->memory_free() : cpu_memory_free(); }
    int64_t memory_total() const override { return zd_ ? zd_->memory_total() : cpu_memory_total(); }
    void synchronize() override { if (zd_) zd_->synchronize(); }
};

// ========================================================================
// Backend factory
// ========================================================================

ComputeBackend* ComputeBackend::create(const BackendConfig& cfg) {
    switch (cfg.type) {
        case BackendType::CPU_SCALAR: return new CPUScalarBackend();
#if defined(QUANT_AVX2)
        case BackendType::CPU_AVX2: return new CPUAVX2Backend();
#endif
#if defined(QUANT_AVX2) && defined(QUANT_AVX512)
        case BackendType::CPU_AVX512: return new CPUAVX512Backend();
#endif
#ifdef QUANT_USE_DIRECTX
        case BackendType::GPU_DIRECTX: return new GPUDirectXBackend();
        case BackendType::IGPU_SHARED: return new IGPUSharedBackend();
#endif
#if defined(QUANT_USE_CUDA)
        case BackendType::GPU_CUDA: return new GPUCUDABackend();
#endif
#if defined(__APPLE__)
        case BackendType::GPU_METAL: return new GPUMetalBackend();
#endif
        case BackendType::GPU_SYCL: return new GPUSYCLBackend();
        case BackendType::GPU_CANN: return new GPUCANNBackend();
        case BackendType::RPC: return new GPURPCBackend();
        case BackendType::RAM_SWAP: return new RAMSwapBackend();
        case BackendType::DISTRIBUTED: return new DistributedBackend();
        case BackendType::NPU_OPENVINO: return new GPUOpenVINOBackend();
        case BackendType::GPU_VIRTGPU: return new GPUVirtGPUBackend();
        case BackendType::GPU_WEBGPU: return new GPUWebGPUBackend();
        case BackendType::CPU_ZENDNN: return new GPUZenDNNBackend();
        default: return new CPUScalarBackend();
    }
}

// ========================================================================
// Hardware detection
// ========================================================================

static inline void quant_cpuid(int info[4], int leaf) {
#if defined(_WIN32)
    __cpuid(info, leaf);
#elif defined(__aarch64__) || defined(__arm__)
    (void)info; (void)leaf;
#else
    __cpuid(leaf, info[0], info[1], info[2], info[3]);
#endif
}

static inline void quant_cpuidex(int info[4], int leaf, int sub) {
#if defined(_WIN32)
    __cpuidex(info, leaf, sub);
#elif defined(__aarch64__) || defined(__arm__)
    (void)info; (void)leaf; (void)sub;
#else
    __cpuid_count(leaf, sub, info[0], info[1], info[2], info[3]);
#endif
}

bool is_avx2_available() {
#if defined(QUANT_AVX2)
    return true;
#elif defined(__aarch64__) || defined(__arm__)
    return false;
#else
    int cpu_info[4] = {0};
    quant_cpuid(cpu_info, 0);
    int n_ids = cpu_info[0];
    if (n_ids >= 7) {
        quant_cpuidex(cpu_info, 7, 0);
        return (cpu_info[1] & (1 << 5)) != 0;
    }
    return false;
#endif
}

bool is_avx512_available() {
#if defined(QUANT_AVX512)
    return true;
#elif defined(__aarch64__) || defined(__arm__)
    return false;
#else
    int cpu_info[4] = {0};
    quant_cpuid(cpu_info, 0);
    int n_ids = cpu_info[0];
    if (n_ids >= 7) {
        quant_cpuidex(cpu_info, 7, 0);
        return (cpu_info[1] & (1 << 16)) != 0;
    }
    return false;
#endif
}

bool is_neon_available() {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return true;
#else
    return false;
#endif
}

bool is_metal_available() {
#if defined(__APPLE__)
    try {
        auto& metal = gpu::get_metal_compute();
        metal.init(0);
        return metal.is_initialized();
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

bool is_cuda_available() {
#if defined(QUANT_USE_CUDA)
    try {
        auto& cuda = gpu::get_cuda_compute();
        cuda.init(0);
        return cuda.is_initialized();
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

bool is_directx_available() {
#if defined(QUANT_USE_DIRECTX)
    return true;
#elif defined(_WIN32)
    HMODULE d3d12 = LoadLibraryA("d3d12.dll");
    if (d3d12) { FreeLibrary(d3d12); return true; }
    return false;
#else
    return false;
#endif
}

bool is_vulkan_available() {
#if defined(QUANT_USE_VULKAN)
    return true;
#elif defined(_WIN32)
    HMODULE vulkan = LoadLibraryA("vulkan-1.dll");
    if (vulkan) { FreeLibrary(vulkan); return true; }
    return false;
#elif defined(__linux__)
    void* lib = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (lib) { dlclose(lib); return true; }
    lib = dlopen("libvulkan.so", RTLD_LAZY | RTLD_LOCAL);
    if (lib) { dlclose(lib); return true; }
    return false;
#else
    return false;
#endif
}

bool is_sycl_available() {
    try {
        auto& sycl = gpu::get_sycl_compute();
        sycl.init(0);
        return sycl.is_initialized();
    } catch (...) {
        return false;
    }
}

bool is_cann_available() {
    try {
        auto& cann = gpu::get_cann_compute();
        cann.init(0);
        return cann.is_initialized();
    } catch (...) {
        return false;
    }
}

bool is_rpc_available() {
    return true;
}

bool is_hip_available() { return false; }
bool is_hexagon_available() { return false; }
bool is_zdnn_available() { return false; }
bool is_musa_available() { return false; }
bool is_opencl_available() { return false; }

bool is_openvino_available() {
    try {
        auto& ov = gpu::get_openvino_compute();
        ov.init(0);
        return ov.is_initialized();
    } catch (...) { return false; }
}

bool is_virtgpu_available() {
    try {
        auto& vg = gpu::get_virtgpu_compute();
        vg.init(0);
        return vg.is_initialized();
    } catch (...) { return false; }
}

bool is_webgpu_available() {
    try {
        auto& wg = gpu::get_webgpu_compute();
        wg.init(0);
        return wg.is_initialized();
    } catch (...) { return false; }
}

bool is_zendnn_available() {
    try {
        auto& zd = gpu::get_zendnn_compute();
        zd.init(0);
        return zd.is_initialized();
    } catch (...) { return false; }
}

int64_t cpu_memory_free() {
#if defined(_WIN32)
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    GlobalMemoryStatusEx(&mem);
    return (int64_t)mem.ullAvailPhys;
#elif defined(__linux__)
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (int64_t)pages * (int64_t)page_size;
#elif defined(__APPLE__)
    int64_t free_pages = 0;
    size_t len = sizeof(free_pages);
    if (sysctlbyname("vm.page_free_count", &free_pages, &len, NULL, 0) == 0) {
        int64_t page_size = 0;
        len = sizeof(page_size);
        sysctlbyname("hw.pagesize", &page_size, &len, NULL, 0);
        if (page_size == 0) page_size = 4096;
        return free_pages * page_size;
    }
    return 8LL * 1024 * 1024 * 1024;
#else
    return 8LL * 1024 * 1024 * 1024;
#endif
}

int64_t cpu_memory_total() {
#if defined(_WIN32)
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    GlobalMemoryStatusEx(&mem);
    return (int64_t)mem.ullTotalPhys;
#elif defined(__linux__)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (int64_t)pages * (int64_t)page_size;
#elif defined(__APPLE__)
    int64_t mem = 0;
    size_t len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &len, NULL, 0);
    return mem;
#else
    return 16LL * 1024 * 1024 * 1024;
#endif
}

int64_t gpu_memory_free(int64_t device_id) {
    (void)device_id;
#if defined(_WIN32)
    if (is_directx_available()) {
        try {
            auto& dx = gpu::get_dx_compute();
            if (dx.is_initialized())
                return dx.memory_free();
        } catch (...) {
            std::fprintf(stderr, "[WARN] Exception caught: %s (GPU memory query failed)\n", __func__);
            return cpu_memory_free();
        }
    }
#endif
#if defined(__APPLE__)
    if (is_metal_available()) {
        try {
            auto& metal = gpu::get_metal_compute();
            if (metal.is_initialized())
                return metal.memory_free();
        } catch (...) {
            return cpu_memory_free();
        }
    }
#endif
    return cpu_memory_free();
}

int64_t metal_memory_free() {
#if defined(__APPLE__)
    return gpu_memory_free(0);
#else
    return 0;
#endif
}

int64_t igpu_memory_free() {
    return cpu_memory_free();
}

Tensor to_backend(const Tensor& t, BackendType dst) {
    if (dst == BackendType::CPU_SCALAR || dst == BackendType::CPU_AVX2) {
        if (!t.data()) return t;
        Tensor out(t.shape());
        std::memcpy(out.data(), t.data(), t.numel() * sizeof(float));
        return out;
    }
    // GPU backends: copy to a new CPU tensor (true GPU transfer not yet implemented)
    if (t.data()) {
        Tensor out(t.shape());
        std::memcpy(out.data(), t.data(), t.numel() * sizeof(float));
        return out;
    }
    return t;
}

Tensor from_backend(const Tensor& t, BackendType src) {
    if (src == BackendType::CPU_SCALAR || src == BackendType::CPU_AVX2) {
        if (!t.data()) return t;
        Tensor out(t.shape());
        std::memcpy(out.data(), t.data(), t.numel() * sizeof(float));
        return out;
    }
    // GPU backends: copy from a source tensor to CPU (true GPU transfer not yet implemented)
    if (t.data()) {
        Tensor out(t.shape());
        std::memcpy(out.data(), t.data(), t.numel() * sizeof(float));
        return out;
    }
    return t;
}

// ========================================================================
// Hardware-target selection system
// ========================================================================

HardwareProfile probe_hardware() {
    HardwareProfile hw;

    // CPU features
    hw.has_avx2 = is_avx2_available();
    hw.has_avx512 = is_avx512_available();
    hw.has_neon = is_neon_available();

    // GPU features
    hw.has_cuda = is_cuda_available();
    hw.has_directx = is_directx_available();
    hw.has_vulkan = is_vulkan_available();
    hw.has_metal = is_metal_available();
    hw.has_sycl = is_sycl_available();
    hw.has_cann = is_cann_available();
    hw.has_rpc = is_rpc_available();
    hw.has_openvino = is_openvino_available();
    hw.has_virtgpu = is_virtgpu_available();
    hw.has_webgpu = is_webgpu_available();
    hw.has_zendnn = is_zendnn_available();

    // RAM
    hw.ram_total = cpu_memory_total();
    hw.ram_free = cpu_memory_free();

    // GPU VRAM (DirectX)
#if defined(_WIN32)
    if (hw.has_directx) {
        try {
            auto& dx = gpu::get_dx_compute();
            if (dx.is_initialized()) {
                hw.vram_free = dx.memory_free();
                hw.vram_total = dx.memory_total();
            }
        } catch (...) {
            std::fprintf(stderr, "[WARN] Exception caught: %s (VRAM query failed)\n", __func__);
            hw.vram_free = 0;
            hw.vram_total = 0;
        }
    }
#endif

    // CPU cores/threads
#if defined(_WIN32)
    SYSTEM_INFO sys = {};
    GetSystemInfo(&sys);
    hw.cpu_cores = (int32_t)sys.dwNumberOfProcessors;
#else
    hw.cpu_cores = (int32_t)std::thread::hardware_concurrency();
#endif
    hw.cpu_threads = (int32_t)std::thread::hardware_concurrency();

    // OS detection
#if defined(_WIN32)
    hw.is_windows = true;
#elif defined(__linux__)
    hw.is_linux = true;
#elif defined(__APPLE__)
    hw.is_macos = true;
#endif

#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
    hw.is_arm = true;
#endif

    return hw;
}

BackendConfig select_optimal_backend(const HardwareProfile& hw,
                                     int64_t model_size_bytes) {
    BackendConfig cfg;
    cfg.threads = hw.cpu_threads > 0 ? hw.cpu_threads : 4;

    // Priority: CUDA > Metal > Vulkan > DX12 > SYCL > CANN > RPC > CPU
    
    if (hw.has_cuda) {
        cfg.type = BackendType::GPU_CUDA;
        cfg.device_name = "GPU_CUDA";
        return cfg;
    }
    
    if (hw.has_metal) {
        cfg.type = BackendType::GPU_METAL;
        cfg.device_name = "GPU_METAL";
        return cfg;
    }
    
    if (hw.has_vulkan) {
        cfg.type = BackendType::GPU_VULKAN;
        cfg.device_name = "GPU_VULKAN";
        return cfg;
    }
    
    if (hw.has_directx && hw.vram_total > 0) {
        cfg.type = BackendType::GPU_DIRECTX;
        cfg.device_name = "GPU_DIRECTX";
        return cfg;
    }

    if (hw.has_sycl) {
        cfg.type = BackendType::GPU_SYCL;
        cfg.device_name = "GPU_SYCL";
        return cfg;
    }

    if (hw.has_cann) {
        cfg.type = BackendType::GPU_CANN;
        cfg.device_name = "GPU_CANN";
        return cfg;
    }

    if (hw.has_rpc) {
        cfg.type = BackendType::RPC;
        cfg.device_name = "RPC";
        return cfg;
    }

    if (hw.has_openvino) {
        cfg.type = BackendType::NPU_OPENVINO;
        cfg.device_name = "NPU_OPENVINO";
        return cfg;
    }

    if (hw.has_virtgpu) {
        cfg.type = BackendType::GPU_VIRTGPU;
        cfg.device_name = "GPU_VIRTGPU";
        return cfg;
    }

    if (hw.has_webgpu) {
        cfg.type = BackendType::GPU_WEBGPU;
        cfg.device_name = "GPU_WEBGPU";
        return cfg;
    }

    if (hw.has_zendnn) {
        cfg.type = BackendType::CPU_ZENDNN;
        cfg.device_name = "CPU_ZENDNN";
        return cfg;
    }

    // CPU with AVX2 for medium models that fit in RAM
    if (hw.has_avx2) {
        if (model_size_bytes == 0 || model_size_bytes <= (int64_t)(hw.ram_free * 0.5)) {
            cfg.type = BackendType::CPU_AVX2;
            cfg.device_name = "CPU_AVX2";
            return cfg;
        }
    }

    // iGPU shared memory for large models on systems with DirectX
    if (hw.has_directx) {
        cfg.type = BackendType::IGPU_SHARED;
        cfg.device_name = "IGPU_SHARED";
        cfg.memory_fraction = 0.8f;
        return cfg;
    }

    // RAM swap for very large models (tiled matmul, disk offload)
    if (model_size_bytes > hw.ram_free / 2) {
        cfg.type = BackendType::RAM_SWAP;
        cfg.device_name = "RAM_SWAP";
        cfg.memory_fraction = 0.95f;
        return cfg;
    }

    // Fallback: scalar CPU
    cfg.type = BackendType::CPU_SCALAR;
    cfg.device_name = "CPU_SCALAR";
    return cfg;
}

double benchmark_operation(ComputeBackend* backend, const char* operation,
                           int64_t M, int64_t N, int64_t K,
                           int warmup, int iters) {
    if (!backend || !backend->is_available()) return 0.0;

    // Use N as the element count for element-wise ops; for gemm use proper 2D shapes
    int64_t n_elem = (M > 0 && N > 0) ? M * N : 0;

    Tensor A, B, C, X, Y;
    if (strcmp(operation, "gemm") == 0) {
        A = Tensor({M, K}, DType::F32);
        B = Tensor({K, N}, DType::F32);
        C = Tensor({M, N}, DType::F32);
        A.fill(1.0f);
        B.fill(2.0f);
        C.fill(0.0f);
    } else if (strcmp(operation, "relu") == 0 || strcmp(operation, "add") == 0) {
        n_elem = (n_elem > 0) ? n_elem : 1024 * 1024;
        A = Tensor({(int64_t)n_elem}, DType::F32);
        B = Tensor({(int64_t)n_elem}, DType::F32);
        C = Tensor({(int64_t)n_elem}, DType::F32);
        Y = Tensor({(int64_t)n_elem}, DType::F32);
        A.fill(1.0f);
        B.fill(2.0f);
        C.fill(0.0f);
    } else if (strcmp(operation, "softmax") == 0 || strcmp(operation, "rms_norm") == 0) {
        int64_t rows = 1024, cols = 1024;
        A = Tensor({rows, cols}, DType::F32);
        Y = Tensor({rows, cols}, DType::F32);
        if (strcmp(operation, "rms_norm") == 0) {
            X = Tensor({cols}, DType::F32);
            X.fill(1.0f);
        }
        A.fill(1.0f);
    } else {
        return 0.0; // unknown operation
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    auto elapsed_us = [&]() -> double {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    };

    // Warmup
    for (int i = 0; i < warmup; i++) {
        if (strcmp(operation, "gemm") == 0) {
            backend->gemm(1.0f, A, B, 0.0f, C);
        } else if (strcmp(operation, "relu") == 0) {
            backend->relu(A, Y);
        } else if (strcmp(operation, "add") == 0) {
            backend->add(A, B, C);
        } else if (strcmp(operation, "softmax") == 0) {
            backend->softmax(A, Y, 1);
        } else if (strcmp(operation, "rms_norm") == 0) {
            backend->rms_norm(A, X, 1e-5f, Y);
        }
    }

    // Benchmark
    backend->synchronize();
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        if (strcmp(operation, "gemm") == 0) {
            backend->gemm(1.0f, A, B, 0.0f, C);
        } else if (strcmp(operation, "relu") == 0) {
            backend->relu(A, Y);
        } else if (strcmp(operation, "add") == 0) {
            backend->add(A, B, C);
        } else if (strcmp(operation, "softmax") == 0) {
            backend->softmax(A, Y, 1);
        } else if (strcmp(operation, "rms_norm") == 0) {
            backend->rms_norm(A, X, 1e-5f, Y);
        }
    }
    backend->synchronize();
    double dt_us = elapsed_us();

    double avg_us = dt_us / iters;
    double flops = 0;

    if (strcmp(operation, "gemm") == 0) {
        flops = 2.0 * (double)M * (double)N * (double)K / (avg_us * 1e-6) * 1e-9;
    } else if (strcmp(operation, "relu") == 0 || strcmp(operation, "add") == 0) {
        flops = (double)n_elem / (avg_us * 1e-6) * 1e-9;
    } else if (strcmp(operation, "softmax") == 0 || strcmp(operation, "rms_norm") == 0) {
        double n = 1024.0 * 1024.0;
        flops = n * 5.0 / (avg_us * 1e-6) * 1e-9;
    }

    return flops; // GFLOPS
}

} // namespace backend
} // namespace quant
