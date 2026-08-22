#pragma once
#include <cstdint>
#include <cstddef>

namespace quant {

class GpuComputeMusa {
public:
    GpuComputeMusa();
    ~GpuComputeMusa();

    bool init();
    void* alloc(size_t size);
    void free(void* ptr);
    void copy_to_device(void* dst, const void* src, size_t size);
    void copy_to_host(void* dst, const void* src, size_t size);

    void launch_gemm(int m, int n, int k, const float* a, const float* b, float* c);

private:
    void* lib_handle_;
    
    // Function pointers
    int (*musa_init_)(unsigned int);
    int (*musa_malloc_)(void**, size_t);
    int (*musa_free_)(void*);
    int (*musa_memcpy_)(void*, const void*, size_t, int);
    int (*musa_launch_kernel_)(const void*, int, int, void**, size_t, void*);
};

} // namespace quant
