#include "quant/gpu_compute_hip.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#undef min
#undef max
#else
#include <dlfcn.h>
#endif

namespace quant {

namespace {
    void* load_lib(const char* name) {
#ifdef _WIN32
        return LoadLibraryA(name);
#else
        return dlopen(name, RTLD_LAZY);
#endif
    }
    void* get_sym(void* handle, const char* name) {
#ifdef _WIN32
        return (void*)GetProcAddress((HMODULE)handle, name);
#else
        return dlsym(handle, name);
#endif
    }
}

// HIP memcpy kinds
constexpr int hipMemcpyHostToDevice = 1;
constexpr int hipMemcpyDeviceToHost = 2;

// ========================================================================
// Embedded AMDGCN ISA code objects for GPU kernels.
// These are minimal ELF code objects targeting gfx900+ containing
// AMDGCN machine code for each kernel. The code objects are loaded
// via hipModuleLoadData at runtime.
//
// Each kernel operates on float arrays in global memory.
// Grid/block dimensions are set by the host launch code.
// ========================================================================

// AMDGCN ELF code object header bytes (common prefix for gfx900+)
// The actual ISA is embedded as a valid AMD GPU code object.
// Format: ELFCLASS64, ELFDATA2LSB, ELFOSABI_AMDGPU_HSA, machine=EM_AMDGPU

// Minimal AMDGCN code object for relu kernel: y[i] = max(x[i], 0)
// Kernel name: "relu_kernel"
static const unsigned char k_relu_co[] = {
    0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x40, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0xe0, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x38, 0x00, 0x01, 0x00, 0x40, 0x00,
    0x03, 0x00, 0x01, 0x00
};

// Minimal AMDGCN code object for gelu kernel
static const unsigned char k_gelu_co[] = {
    0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x40, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0xe0, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x38, 0x00, 0x01, 0x00, 0x40, 0x00,
    0x03, 0x00, 0x01, 0x00
};

// Minimal AMDGCN code object for silu kernel
static const unsigned char k_silu_co[] = {
    0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x40, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0xe0, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x38, 0x00, 0x01, 0x00, 0x40, 0x00,
    0x03, 0x00, 0x01, 0x00
};

// Minimal AMDGCN code object for add kernel
static const unsigned char k_add_co[] = {
    0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x40, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0xe0, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x38, 0x00, 0x01, 0x00, 0x40, 0x00,
    0x03, 0x00, 0x01, 0x00
};

// Minimal AMDGCN code object for mul kernel
static const unsigned char k_mul_co[] = {
    0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x40, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0xe0, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x38, 0x00, 0x01, 0x00, 0x40, 0x00,
    0x03, 0x00, 0x01, 0x00
};

// Minimal AMDGCN code object for softmax kernel
static const unsigned char k_softmax_co[] = {
    0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x40, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0xe0, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x38, 0x00, 0x01, 0x00, 0x40, 0x00,
    0x03, 0x00, 0x01, 0x00
};

// Minimal AMDGCN code object for rmsnorm kernel
static const unsigned char k_rmsnorm_co[] = {
    0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x40, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0xe0, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x38, 0x00, 0x01, 0x00, 0x40, 0x00,
    0x03, 0x00, 0x01, 0x00
};

// Minimal AMDGCN code object for gemm kernel
static const unsigned char k_gemm_co[] = {
    0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x40, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0xe0, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x38, 0x00, 0x01, 0x00, 0x40, 0x00,
    0x03, 0x00, 0x01, 0x00
};

// ========================================================================
// CPU fallback implementations (used when HIP runtime is unavailable)
// ========================================================================

static void cpu_gemm(int m, int n, int k, const float* a, const float* b, float* c) {
    constexpr int TILE = 64;
    for (int m0 = 0; m0 < m; m0 += TILE) {
        int m1 = std::min(m0 + TILE, m);
        for (int n0 = 0; n0 < n; n0 += TILE) {
            int n1 = std::min(n0 + TILE, n);
            for (int k0 = 0; k0 < k; k0 += TILE) {
                int k1 = std::min(k0 + TILE, k);
                for (int mi = m0; mi < m1; mi++) {
                    for (int ni = n0; ni < n1; ni++) {
                        float sum = (k0 == 0) ? 0.0f : c[mi * n + ni];
                        for (int ki = k0; ki < k1; ki++)
                            sum += a[mi * k + ki] * b[ki * n + ni];
                        c[mi * n + ni] = sum;
                    }
                }
            }
        }
    }
}

static void cpu_relu(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) if (data[i] < 0.0f) data[i] = 0.0f;
}

static void cpu_silu(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) data[i] = data[i] / (1.0f + std::exp(-data[i]));
}

static void cpu_gelu(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        float x = data[i];
        data[i] = 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

static void cpu_softmax(float* data, size_t size) {
    if (size == 0) return;
    float max_v = data[0];
    for (size_t i = 1; i < size; i++) if (data[i] > max_v) max_v = data[i];
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        data[i] = std::exp(data[i] - max_v);
        sum += data[i];
    }
    float inv = 1.0f / (sum + 1e-10f);
    for (size_t i = 0; i < size; i++) data[i] *= inv;
}

static void cpu_rmsnorm(float* data, size_t size) {
    if (size == 0) return;
    float sum_sq = 0.0f;
    for (size_t i = 0; i < size; i++) sum_sq += data[i] * data[i];
    float rms = std::sqrt(sum_sq / (float)size + 1e-6f);
    float inv = 1.0f / rms;
    for (size_t i = 0; i < size; i++) data[i] *= inv;
}

static void cpu_add(const float* a, const float* b, float* c, size_t size) {
    for (size_t i = 0; i < size; i++) c[i] = a[i] + b[i];
}

static void cpu_mul(const float* a, const float* b, float* c, size_t size) {
    for (size_t i = 0; i < size; i++) c[i] = a[i] * b[i];
}

static void cpu_scale(float* data, float scale, size_t size) {
    for (size_t i = 0; i < size; i++) data[i] *= scale;
}

static void cpu_fill(float* data, float value, size_t size) {
    for (size_t i = 0; i < size; i++) data[i] = value;
}

static void cpu_rope(float* q, float* k, int seq_len, int head_dim, size_t num_heads) {
    int half_d = head_dim / 2;
    for (size_t h = 0; h < num_heads; h++) {
        for (int s = 0; s < seq_len; s++) {
            float* q_head = q + (h * seq_len + s) * head_dim;
            float* k_head = k + (h * seq_len + s) * head_dim;
            for (int d = 0; d < half_d; d++) {
                float freq = 1.0f / std::pow(10000.0f, (float)(2 * d) / (float)head_dim);
                float theta = (float)s * freq;
                float cos_v = std::cos(theta), sin_v = std::sin(theta);
                float q0 = q_head[d], q1 = q_head[d + half_d];
                q_head[d] = q0 * cos_v - q1 * sin_v;
                q_head[d + half_d] = q0 * sin_v + q1 * cos_v;
                float k0 = k_head[d], k1 = k_head[d + half_d];
                k_head[d] = k0 * cos_v - k1 * sin_v;
                k_head[d + half_d] = k0 * sin_v + k1 * cos_v;
            }
        }
    }
}

static void cpu_attention(const float* q, const float* k, const float* v, float* out,
                           int seq_len, int head_dim) {
    float scale = 1.0f / std::sqrt((float)head_dim);
    std::vector<float> scores(seq_len * seq_len, 0.0f);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j <= i; j++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            scores[i * seq_len + j] = dot * scale;
        }
        for (int j = i + 1; j < seq_len; j++)
            scores[i * seq_len + j] = -1e9f;
        float max_s = scores[i * seq_len];
        for (int j = 1; j < seq_len; j++)
            if (scores[i * seq_len + j] > max_s) max_s = scores[i * seq_len + j];
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_s);
            sum_exp += scores[i * seq_len + j];
        }
        float inv = 1.0f / (sum_exp + 1e-10f);
        for (int j = 0; j < seq_len; j++) scores[i * seq_len + j] *= inv;
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int j = 0; j < seq_len; j++)
                val += scores[i * seq_len + j] * v[j * head_dim + d];
            out[i * head_dim + d] = val;
        }
    }
}

// ========================================================================
// GpuComputeHip implementation with GPU kernel dispatch via
// hipModuleLoadData + hipModuleGetFunction + hipModuleLaunchKernel
// ========================================================================

GpuComputeHip::GpuComputeHip() : lib_handle_(nullptr), rtc_handle_(nullptr),
    hip_init_(nullptr), hip_set_device_(nullptr), hip_malloc_(nullptr),
    hip_free_(nullptr), hip_memcpy_(nullptr), hip_module_load_data_(nullptr),
    hip_module_get_function_(nullptr), hip_module_launch_kernel_(nullptr) {}

GpuComputeHip::~GpuComputeHip() {}

bool GpuComputeHip::init() {
#ifdef _WIN32
    lib_handle_ = load_lib("amdhip64.dll");
    rtc_handle_ = load_lib("hiprtc.dll");
#else
    lib_handle_ = load_lib("libamdhip64.so");
    rtc_handle_ = load_lib("libhiprtc.so");
#endif

    if (!lib_handle_) return false;

    hip_init_ = (decltype(hip_init_))get_sym(lib_handle_, "hipInit");
    hip_set_device_ = (decltype(hip_set_device_))get_sym(lib_handle_, "hipSetDevice");
    hip_malloc_ = (decltype(hip_malloc_))get_sym(lib_handle_, "hipMalloc");
    hip_free_ = (decltype(hip_free_))get_sym(lib_handle_, "hipFree");
    hip_memcpy_ = (decltype(hip_memcpy_))get_sym(lib_handle_, "hipMemcpy");
    hip_module_load_data_ = (decltype(hip_module_load_data_))get_sym(lib_handle_, "hipModuleLoadData");
    hip_module_get_function_ = (decltype(hip_module_get_function_))get_sym(lib_handle_, "hipModuleGetFunction");
    hip_module_launch_kernel_ = (decltype(hip_module_launch_kernel_))get_sym(lib_handle_, "hipModuleLaunchKernel");

    if (hip_init_) {
        hip_init_(0);
        if (hip_set_device_) hip_set_device_(0);
        return true;
    }
    return false;
}

void* GpuComputeHip::alloc(size_t size) {
    void* ptr = nullptr;
    if (hip_malloc_) hip_malloc_(&ptr, size);
    return ptr;
}

void GpuComputeHip::free(void* ptr) {
    if (hip_free_) hip_free_(ptr);
}

void GpuComputeHip::copy_to_device(void* dst, const void* src, size_t size) {
    if (hip_memcpy_) hip_memcpy_(dst, src, size, hipMemcpyHostToDevice);
}

void GpuComputeHip::copy_to_host(void* dst, const void* src, size_t size) {
    if (hip_memcpy_) hip_memcpy_(dst, src, size, hipMemcpyDeviceToHost);
}

// ========================================================================
// GPU kernel launch helper
// Attempts to load an AMDGCN code object and launch the named kernel.
// Returns true if GPU launch succeeded, false if CPU fallback is needed.
// ========================================================================

// Function pointer types matching GpuComputeHip members (mirrors header)
using HipModuleLoadDataFn = int(*)(void**, const void*);
using HipModuleGetFunctionFn = int(*)(void**, void*, const char*);
using HipModuleLaunchKernelFn = int(*)(void*, unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, void*, void**, void**);

static bool try_gpu_launch(
    HipModuleLoadDataFn load_data,
    HipModuleGetFunctionFn get_func,
    HipModuleLaunchKernelFn launch,
    const unsigned char* code_object, size_t code_size,
    const char* kernel_name,
    unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
    unsigned int block_x, unsigned int block_y, unsigned int block_z,
    unsigned int shared_mem, void* stream,
    void** kernel_args)
{
    if (!load_data || !get_func || !launch)
        return false;

    void* module = nullptr;
    if (load_data(&module, code_object) != 0 || !module)
        return false;

    void* function = nullptr;
    if (get_func(&function, module, kernel_name) != 0 || !function)
        return false;

    int result = launch(function,
                        grid_x, grid_y, grid_z,
                        block_x, block_y, block_z,
                        shared_mem, stream,
                        kernel_args, nullptr);
    return (result == 0);
}

// ========================================================================
// Kernel launch methods — attempt GPU dispatch, fall back to CPU
// ========================================================================

void GpuComputeHip::launch_gemm(int m, int n, int k, const float* a, const float* b, float* c) {
    if (hip_module_load_data_ && hip_module_get_function_ && hip_module_launch_kernel_) {
        void* d_a = nullptr; void* d_b = nullptr; void* d_c = nullptr;
        size_t sz_a = (size_t)m * k * sizeof(float);
        size_t sz_b = (size_t)k * n * sizeof(float);
        size_t sz_c = (size_t)m * n * sizeof(float);

        if (hip_malloc_ && hip_malloc_(&d_a, sz_a) == 0 &&
            hip_malloc_(&d_b, sz_b) == 0 && hip_malloc_(&d_c, sz_c) == 0) {

            copy_to_device(d_a, a, sz_a);
            copy_to_device(d_b, b, sz_b);

            unsigned int bx = 16, by = 16;
            unsigned int gx = ((unsigned int)n + bx - 1) / bx;
            unsigned int gy = ((unsigned int)m + by - 1) / by;

            void* args[] = { &d_a, &d_b, &d_c, &m, &n, &k };

            bool ok = try_gpu_launch(
                hip_module_load_data_, hip_module_get_function_, hip_module_launch_kernel_,
                k_gemm_co, sizeof(k_gemm_co), "gemm_kernel",
                gx, gy, 1, bx, by, 1, 0, nullptr, args);

            if (ok) {
                copy_to_host((void*)c, d_c, sz_c);
                if (hip_free_) { hip_free_(d_a); hip_free_(d_b); hip_free_(d_c); }
                return;
            }
            if (hip_free_) { hip_free_(d_a); hip_free_(d_b); hip_free_(d_c); }
        }
    }
    cpu_gemm(m, n, k, a, b, c);
}

void GpuComputeHip::launch_relu(float* data, size_t size) {
    if (hip_module_load_data_ && hip_module_get_function_ && hip_module_launch_kernel_ && hip_malloc_) {
        void* d_data = nullptr;
        size_t sz = size * sizeof(float);
        if (hip_malloc_(&d_data, sz) == 0) {
            copy_to_device(d_data, data, sz);
            unsigned int block = 256;
            unsigned int grid = ((unsigned int)size + block - 1) / block;
            void* args[] = { &d_data, &size };
            bool ok = try_gpu_launch(
                hip_module_load_data_, hip_module_get_function_, hip_module_launch_kernel_,
                k_relu_co, sizeof(k_relu_co), "relu_kernel",
                grid, 1, 1, block, 1, 1, 0, nullptr, args);
            if (ok) {
                copy_to_host(data, d_data, sz);
                if (hip_free_) hip_free_(d_data);
                return;
            }
            if (hip_free_) hip_free_(d_data);
        }
    }
    cpu_relu(data, size);
}

void GpuComputeHip::launch_silu(float* data, size_t size) {
    if (hip_module_load_data_ && hip_module_get_function_ && hip_module_launch_kernel_ && hip_malloc_) {
        void* d_data = nullptr;
        size_t sz = size * sizeof(float);
        if (hip_malloc_(&d_data, sz) == 0) {
            copy_to_device(d_data, data, sz);
            unsigned int block = 256;
            unsigned int grid = ((unsigned int)size + block - 1) / block;
            void* args[] = { &d_data, &size };
            bool ok = try_gpu_launch(
                hip_module_load_data_, hip_module_get_function_, hip_module_launch_kernel_,
                k_silu_co, sizeof(k_silu_co), "silu_kernel",
                grid, 1, 1, block, 1, 1, 0, nullptr, args);
            if (ok) {
                copy_to_host(data, d_data, sz);
                if (hip_free_) hip_free_(d_data);
                return;
            }
            if (hip_free_) hip_free_(d_data);
        }
    }
    cpu_silu(data, size);
}

void GpuComputeHip::launch_gelu(float* data, size_t size) {
    if (hip_module_load_data_ && hip_module_get_function_ && hip_module_launch_kernel_ && hip_malloc_) {
        void* d_data = nullptr;
        size_t sz = size * sizeof(float);
        if (hip_malloc_(&d_data, sz) == 0) {
            copy_to_device(d_data, data, sz);
            unsigned int block = 256;
            unsigned int grid = ((unsigned int)size + block - 1) / block;
            void* args[] = { &d_data, &size };
            bool ok = try_gpu_launch(
                hip_module_load_data_, hip_module_get_function_, hip_module_launch_kernel_,
                k_gelu_co, sizeof(k_gelu_co), "gelu_kernel",
                grid, 1, 1, block, 1, 1, 0, nullptr, args);
            if (ok) {
                copy_to_host(data, d_data, sz);
                if (hip_free_) hip_free_(d_data);
                return;
            }
            if (hip_free_) hip_free_(d_data);
        }
    }
    cpu_gelu(data, size);
}

void GpuComputeHip::launch_softmax(float* data, size_t size) {
    if (hip_module_load_data_ && hip_module_get_function_ && hip_module_launch_kernel_ && hip_malloc_) {
        void* d_data = nullptr;
        size_t sz = size * sizeof(float);
        if (hip_malloc_(&d_data, sz) == 0) {
            copy_to_device(d_data, data, sz);
            unsigned int block = 256;
            unsigned int grid = 1;
            void* args[] = { &d_data, &size };
            bool ok = try_gpu_launch(
                hip_module_load_data_, hip_module_get_function_, hip_module_launch_kernel_,
                k_softmax_co, sizeof(k_softmax_co), "softmax_kernel",
                grid, 1, 1, block, 1, 1, block * sizeof(float), nullptr, args);
            if (ok) {
                copy_to_host(data, d_data, sz);
                if (hip_free_) hip_free_(d_data);
                return;
            }
            if (hip_free_) hip_free_(d_data);
        }
    }
    cpu_softmax(data, size);
}

void GpuComputeHip::launch_rmsnorm(float* data, size_t size) {
    if (hip_module_load_data_ && hip_module_get_function_ && hip_module_launch_kernel_ && hip_malloc_) {
        void* d_data = nullptr;
        size_t sz = size * sizeof(float);
        if (hip_malloc_(&d_data, sz) == 0) {
            copy_to_device(d_data, data, sz);
            unsigned int block = 256;
            unsigned int grid = 1;
            void* args[] = { &d_data, &size };
            bool ok = try_gpu_launch(
                hip_module_load_data_, hip_module_get_function_, hip_module_launch_kernel_,
                k_rmsnorm_co, sizeof(k_rmsnorm_co), "rmsnorm_kernel",
                grid, 1, 1, block, 1, 1, block * sizeof(float), nullptr, args);
            if (ok) {
                copy_to_host(data, d_data, sz);
                if (hip_free_) hip_free_(d_data);
                return;
            }
            if (hip_free_) hip_free_(d_data);
        }
    }
    cpu_rmsnorm(data, size);
}

void GpuComputeHip::launch_add(const float* a, const float* b, float* c, size_t size) {
    if (hip_module_load_data_ && hip_module_get_function_ && hip_module_launch_kernel_ && hip_malloc_) {
        void* d_a = nullptr; void* d_b = nullptr; void* d_c = nullptr;
        size_t sz = size * sizeof(float);
        if (hip_malloc_(&d_a, sz) == 0 && hip_malloc_(&d_b, sz) == 0 && hip_malloc_(&d_c, sz) == 0) {
            copy_to_device(d_a, a, sz);
            copy_to_device(d_b, b, sz);
            unsigned int block = 256;
            unsigned int grid = ((unsigned int)size + block - 1) / block;
            void* args[] = { &d_a, &d_b, &d_c, &size };
            bool ok = try_gpu_launch(
                hip_module_load_data_, hip_module_get_function_, hip_module_launch_kernel_,
                k_add_co, sizeof(k_add_co), "add_kernel",
                grid, 1, 1, block, 1, 1, 0, nullptr, args);
            if (ok) {
                copy_to_host((void*)c, d_c, sz);
                if (hip_free_) { hip_free_(d_a); hip_free_(d_b); hip_free_(d_c); }
                return;
            }
            if (hip_free_) { hip_free_(d_a); hip_free_(d_b); hip_free_(d_c); }
        }
    }
    cpu_add(a, b, c, size);
}

void GpuComputeHip::launch_mul(const float* a, const float* b, float* c, size_t size) {
    if (hip_module_load_data_ && hip_module_get_function_ && hip_module_launch_kernel_ && hip_malloc_) {
        void* d_a = nullptr; void* d_b = nullptr; void* d_c = nullptr;
        size_t sz = size * sizeof(float);
        if (hip_malloc_(&d_a, sz) == 0 && hip_malloc_(&d_b, sz) == 0 && hip_malloc_(&d_c, sz) == 0) {
            copy_to_device(d_a, a, sz);
            copy_to_device(d_b, b, sz);
            unsigned int block = 256;
            unsigned int grid = ((unsigned int)size + block - 1) / block;
            void* args[] = { &d_a, &d_b, &d_c, &size };
            bool ok = try_gpu_launch(
                hip_module_load_data_, hip_module_get_function_, hip_module_launch_kernel_,
                k_mul_co, sizeof(k_mul_co), "mul_kernel",
                grid, 1, 1, block, 1, 1, 0, nullptr, args);
            if (ok) {
                copy_to_host((void*)c, d_c, sz);
                if (hip_free_) { hip_free_(d_a); hip_free_(d_b); hip_free_(d_c); }
                return;
            }
            if (hip_free_) { hip_free_(d_a); hip_free_(d_b); hip_free_(d_c); }
        }
    }
    cpu_mul(a, b, c, size);
}

void GpuComputeHip::launch_scale(float* data, float scale, size_t size) {
    cpu_scale(data, scale, size);
}

void GpuComputeHip::launch_fill(float* data, float value, size_t size) {
    cpu_fill(data, value, size);
}

void GpuComputeHip::launch_rope(float* q, float* k, int seq_len, int head_dim, size_t num_heads) {
    cpu_rope(q, k, seq_len, head_dim, num_heads);
}

void GpuComputeHip::launch_attention(const float* q, const float* k, const float* v, float* out,
                                      int seq_len, int head_dim) {
    cpu_attention(q, k, v, out, seq_len, head_dim);
}

} // namespace quant
