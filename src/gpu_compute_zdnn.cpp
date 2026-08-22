#include "quant/gpu_compute_zdnn.h"
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

GpuComputeZDnn::GpuComputeZDnn() : lib_handle_(nullptr) {}

GpuComputeZDnn::~GpuComputeZDnn() {}

bool GpuComputeZDnn::init() {
#ifdef _WIN32
    lib_handle_ = load_lib("zdnn.dll");
#else
    lib_handle_ = load_lib("libzdnn.so");
#endif

    if (!lib_handle_) return false;

    zdnn_init_ = (decltype(zdnn_init_))get_sym(lib_handle_, "zdnn_init");
    zdnn_matmul_ = (decltype(zdnn_matmul_))get_sym(lib_handle_, "zdnn_matmul");
    zdnn_add_ = (decltype(zdnn_add_))get_sym(lib_handle_, "zdnn_add");
    zdnn_relu_ = (decltype(zdnn_relu_))get_sym(lib_handle_, "zdnn_relu");

    if (zdnn_init_) {
        zdnn_init_();
        return true;
    }
    return false;
}

void* GpuComputeZDnn::alloc(size_t size) {
    return new uint8_t[size];
}

void GpuComputeZDnn::free(void* ptr) {
    delete[] static_cast<uint8_t*>(ptr);
}

void GpuComputeZDnn::copy_to_device(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
}

void GpuComputeZDnn::copy_to_host(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
}

void GpuComputeZDnn::launch_gemm(int m, int n, int k, const float* a, const float* b, float* c) {
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
