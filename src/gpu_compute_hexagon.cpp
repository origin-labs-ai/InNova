#include "quant/gpu_compute_hexagon.h"
#include <iostream>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
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

GpuComputeHexagon::GpuComputeHexagon() : lib_handle_(nullptr) {}

GpuComputeHexagon::~GpuComputeHexagon() {}

bool GpuComputeHexagon::init() {
#ifdef _WIN32
    lib_handle_ = load_lib("libhexagon_nn_skel.dll");
#else
    lib_handle_ = load_lib("libhexagon_nn_skel.so");
#endif

    if (!lib_handle_) return false;

    hexagon_nn_init_ = (decltype(hexagon_nn_init_))get_sym(lib_handle_, "hexagon_nn_init");
    
    if (hexagon_nn_init_) {
        hexagon_nn_init_();
        return true;
    }
    return false;
}

void* GpuComputeHexagon::alloc(size_t size) {
    return new uint8_t[size];
}

void GpuComputeHexagon::free(void* ptr) {
    delete[] static_cast<uint8_t*>(ptr);
}

void GpuComputeHexagon::copy_to_device(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
}

void GpuComputeHexagon::copy_to_host(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
}

void GpuComputeHexagon::launch_gemm(int m, int n, int k, const float* a, const float* b, float* c) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

} // namespace quant
