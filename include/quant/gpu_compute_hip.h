#pragma once
#include <cstdint>
#include <cstddef>

namespace quant {

class GpuComputeHip {
public:
    GpuComputeHip();
    ~GpuComputeHip();

    bool init();
    void* alloc(size_t size);
    void free(void* ptr);
    void copy_to_device(void* dst, const void* src, size_t size);
    void copy_to_host(void* dst, const void* src, size_t size);

    void launch_gemm(int m, int n, int k, const float* a, const float* b, float* c);
    void launch_relu(float* data, size_t size);
    void launch_silu(float* data, size_t size);
    void launch_gelu(float* data, size_t size);
    void launch_softmax(float* data, size_t size);
    void launch_rmsnorm(float* data, size_t size);
    void launch_add(const float* a, const float* b, float* c, size_t size);
    void launch_mul(const float* a, const float* b, float* c, size_t size);
    void launch_scale(float* data, float scale, size_t size);
    void launch_fill(float* data, float value, size_t size);
    void launch_rope(float* q, float* k, int seq_len, int head_dim, size_t num_heads);
    void launch_attention(const float* q, const float* k, const float* v, float* out, int seq_len, int head_dim);

private:
    void* lib_handle_;
    void* rtc_handle_;
    
    // Function pointers
    int (*hip_init_)(unsigned int);
    int (*hip_set_device_)(int);
    int (*hip_malloc_)(void**, size_t);
    int (*hip_free_)(void*);
    int (*hip_memcpy_)(void*, const void*, size_t, int);
    int (*hip_module_load_data_)(void**, const void*);
    int (*hip_module_get_function_)(void**, void*, const char*);
    int (*hip_module_launch_kernel_)(void*, unsigned int, unsigned int, unsigned int, 
                                     unsigned int, unsigned int, unsigned int, 
                                     unsigned int, void*, void**, void**);
};

} // namespace quant
