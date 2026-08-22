#include "quant/gpu_compute_opencl.h"
#include <iostream>
#include <vector>
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

    const char* kOpenCLGEMMKernel = R"(
__kernel void gemm_kernel(const int M, const int N, const int K,
                          __global const float* A,
                          __global const float* B,
                          __global float* C) {
        int row = get_global_id(0);
        int col = get_global_id(1);

        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
)";
}

GpuComputeOpenCL::GpuComputeOpenCL() : lib_handle_(nullptr), context_(nullptr), command_queue_(nullptr) {}

GpuComputeOpenCL::~GpuComputeOpenCL() {}

bool GpuComputeOpenCL::init() {
#ifdef _WIN32
    lib_handle_ = load_lib("OpenCL.dll");
#else
    lib_handle_ = load_lib("libOpenCL.so");
#endif

    if (!lib_handle_) return false;

    clGetPlatformIDs_ = (decltype(clGetPlatformIDs_))get_sym(lib_handle_, "clGetPlatformIDs");
    clGetDeviceIDs_ = (decltype(clGetDeviceIDs_))get_sym(lib_handle_, "clGetDeviceIDs");
    clCreateContext_ = (decltype(clCreateContext_))get_sym(lib_handle_, "clCreateContext");
    clCreateCommandQueue_ = (decltype(clCreateCommandQueue_))get_sym(lib_handle_, "clCreateCommandQueue");
    clCreateProgramWithSource_ = (decltype(clCreateProgramWithSource_))get_sym(lib_handle_, "clCreateProgramWithSource");
    clBuildProgram_ = (decltype(clBuildProgram_))get_sym(lib_handle_, "clBuildProgram");
    clCreateKernel_ = (decltype(clCreateKernel_))get_sym(lib_handle_, "clCreateKernel");
    clEnqueueNDRangeKernel_ = (decltype(clEnqueueNDRangeKernel_))get_sym(lib_handle_, "clEnqueueNDRangeKernel");
    clCreateBuffer_ = (decltype(clCreateBuffer_))get_sym(lib_handle_, "clCreateBuffer");
    clEnqueueWriteBuffer_ = (decltype(clEnqueueWriteBuffer_))get_sym(lib_handle_, "clEnqueueWriteBuffer");
    clEnqueueReadBuffer_ = (decltype(clEnqueueReadBuffer_))get_sym(lib_handle_, "clEnqueueReadBuffer");
    clReleaseMemObject_ = (decltype(clReleaseMemObject_))get_sym(lib_handle_, "clReleaseMemObject");

    if (!clGetPlatformIDs_ || !clGetDeviceIDs_ || !clCreateContext_) return false;

    uint32_t num_platforms = 0;
    void* platform = nullptr;
    if (clGetPlatformIDs_(1, &platform, &num_platforms) != 0 || num_platforms == 0) return false;

    uint32_t num_devices = 0;
    void* device = nullptr;
    if (clGetDeviceIDs_(platform, 0xFFFFFFFF, 1, &device, &num_devices) != 0 || num_devices == 0) return false;

    int err = 0;
    context_ = clCreateContext_(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != 0 || !context_) return false;

    command_queue_ = clCreateCommandQueue_(context_, device, 0, &err);
    return (err == 0 && command_queue_ != nullptr);
}

void* GpuComputeOpenCL::alloc(size_t size) {
    if (clCreateBuffer_ && context_) {
        int err = 0;
        return clCreateBuffer_(context_, 4, size, nullptr, &err); // CL_MEM_READ_WRITE = 4
    }
    return new uint8_t[size];
}

void GpuComputeOpenCL::free(void* ptr) {
    if (clReleaseMemObject_ && context_) {
        clReleaseMemObject_(ptr);
    } else {
        delete[] static_cast<uint8_t*>(ptr);
    }
}

void GpuComputeOpenCL::copy_to_device(void* dst, const void* src, size_t size) {
    if (clEnqueueWriteBuffer_ && command_queue_) {
        clEnqueueWriteBuffer_(command_queue_, dst, 1, 0, size, src, 0, nullptr, nullptr);
    } else {
        std::memcpy(dst, src, size);
    }
}

void GpuComputeOpenCL::copy_to_host(void* dst, const void* src, size_t size) {
    if (clEnqueueReadBuffer_ && command_queue_) {
        clEnqueueReadBuffer_(command_queue_, dst, 1, 0, size, const_cast<void*>(src), 0, nullptr, nullptr);
    } else {
        std::memcpy(dst, src, size);
    }
}

void GpuComputeOpenCL::launch_gemm(int m, int n, int k, const float* a, const float* b, float* c) {
    if (!clCreateProgramWithSource_ || !clBuildProgram_ || !clCreateKernel_ || !clEnqueueNDRangeKernel_) {
        // Fallback standard C++ SIMD GEMM
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        return;
    }

    int err = 0;
    const char* src = kOpenCLGEMMKernel;
    size_t length = std::strlen(src);
    void* program = clCreateProgramWithSource_(context_, 1, &src, &length, &err);
    if (err != 0 || !program) return;

    clBuildProgram_(program, 0, nullptr, nullptr, nullptr, nullptr);
    void* kernel = clCreateKernel_(program, "gemm_kernel", &err);
    if (err != 0 || !kernel) return;

    void* d_a = alloc(m * k * sizeof(float));
    void* d_b = alloc(k * n * sizeof(float));
    void* d_c = alloc(m * n * sizeof(float));

    copy_to_device(d_a, a, m * k * sizeof(float));
    copy_to_device(d_b, b, k * n * sizeof(float));

    size_t global_work_size[2] = {(size_t)m, (size_t)n};
    clEnqueueNDRangeKernel_(command_queue_, kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);

    copy_to_host(c, d_c, m * n * sizeof(float));

    free(d_a);
    free(d_b);
    free(d_c);
}

} // namespace quant
