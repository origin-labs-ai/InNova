#pragma once
#include <cstdint>
#include <cstddef>

namespace quant {

class GpuComputeHexagon {
public:
    GpuComputeHexagon();
    ~GpuComputeHexagon();

    bool init();
    void* alloc(size_t size);
    void free(void* ptr);
    void copy_to_device(void* dst, const void* src, size_t size);
    void copy_to_host(void* dst, const void* src, size_t size);

    void launch_gemm(int m, int n, int k, const float* a, const float* b, float* c);

private:
    void* lib_handle_;
    
    // Function pointers
    int (*hexagon_nn_init_)(void);
    int (*hexagon_nn_prepare_)(void);
    int (*hexagon_nn_execute_)(void);
};

} // namespace quant
