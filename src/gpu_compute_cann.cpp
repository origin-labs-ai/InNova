#include "quant/gpu_compute_cann.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace quant {
namespace gpu {

typedef int aclError;
typedef void* aclrtStream;

typedef aclError (*PFN_aclInit)(const char* configPath);
typedef aclError (*PFN_aclFinalize)();
typedef aclError (*PFN_aclrtSetDevice)(int32_t deviceId);
typedef aclError (*PFN_aclrtResetDevice)(int32_t deviceId);
typedef aclError (*PFN_aclrtCreateStream)(aclrtStream* stream);
typedef aclError (*PFN_aclrtDestroyStream)(aclrtStream stream);
typedef aclError (*PFN_aclrtMalloc)(void** devPtr, size_t size, int32_t policy);
typedef aclError (*PFN_aclrtFree)(void* devPtr);
typedef aclError (*PFN_aclrtMemcpy)(void* dst, size_t destMax, const void* src, size_t count, int32_t kind);
typedef aclError (*PFN_aclrtSynchronizeStream)(aclrtStream stream);
typedef aclError (*PFN_aclopExecuteV2)(const char* opType, int numInputs, void* inputDesc, void* inputs, int numOutputs, void* outputDesc, void* outputs, void* attr, aclrtStream stream);

struct CANNAPI {
    PFN_aclInit aclInit = nullptr;
    PFN_aclFinalize aclFinalize = nullptr;
    PFN_aclrtSetDevice aclrtSetDevice = nullptr;
    PFN_aclrtResetDevice aclrtResetDevice = nullptr;
    PFN_aclrtCreateStream aclrtCreateStream = nullptr;
    PFN_aclrtDestroyStream aclrtDestroyStream = nullptr;
    PFN_aclrtMalloc aclrtMalloc = nullptr;
    PFN_aclrtFree aclrtFree = nullptr;
    PFN_aclrtMemcpy aclrtMemcpy = nullptr;
    PFN_aclrtSynchronizeStream aclrtSynchronizeStream = nullptr;
    PFN_aclopExecuteV2 aclopExecuteV2 = nullptr;
    void* handle = nullptr;
};

struct GPUComputeCann::Impl {
    CANNAPI api;
    bool initialized = false;
    aclrtStream stream = nullptr;
    int32_t device_id = 0;

    bool load_cann() {
        if (api.handle) return true;
#if defined(_WIN32)
        api.handle = LoadLibraryA("libascendcl.dll");
        if (!api.handle) return false;
        #define LOAD_SYM(name) api.name = (PFN_##name)GetProcAddress((HMODULE)api.handle, #name)
#else
        api.handle = dlopen("libascendcl.so", RTLD_LAZY | RTLD_LOCAL);
        if (!api.handle) return false;
        #define LOAD_SYM(name) api.name = (PFN_##name)dlsym(api.handle, #name)
#endif
        LOAD_SYM(aclInit);
        LOAD_SYM(aclFinalize);
        LOAD_SYM(aclrtSetDevice);
        LOAD_SYM(aclrtResetDevice);
        LOAD_SYM(aclrtCreateStream);
        LOAD_SYM(aclrtDestroyStream);
        LOAD_SYM(aclrtMalloc);
        LOAD_SYM(aclrtFree);
        LOAD_SYM(aclrtMemcpy);
        LOAD_SYM(aclrtSynchronizeStream);
        LOAD_SYM(aclopExecuteV2);
        return true;
    }

    void execute_op(const char* op_type) {
        if (!api.aclopExecuteV2) return;
        api.aclopExecuteV2(op_type, 0, nullptr, nullptr, 0, nullptr, nullptr, nullptr, stream);
    }
};

GPUComputeCann::GPUComputeCann() : impl_(new Impl()) {}
GPUComputeCann::~GPUComputeCann() {
    shutdown();
    delete impl_;
}

bool GPUComputeCann::init(int64_t device_id) {
    if (impl_->initialized) return true;
    if (!impl_->load_cann()) return false;
    if (!impl_->api.aclInit) return false;

    if (impl_->api.aclInit(nullptr) != 0) return false;
    if (impl_->api.aclrtSetDevice((int32_t)device_id) != 0) return false;
    if (impl_->api.aclrtCreateStream(&impl_->stream) != 0) return false;
    
    impl_->device_id = (int32_t)device_id;
    impl_->initialized = true;
    return true;
}

bool GPUComputeCann::is_initialized() const { return impl_->initialized; }

void GPUComputeCann::shutdown() {
    if (impl_->initialized) {
        if (impl_->api.aclrtDestroyStream) impl_->api.aclrtDestroyStream(impl_->stream);
        if (impl_->api.aclrtResetDevice) impl_->api.aclrtResetDevice(impl_->device_id);
        if (impl_->api.aclFinalize) impl_->api.aclFinalize();
        impl_->initialized = false;
    }
}

void* GPUComputeCann::alloc(int64_t bytes) {
    if (!impl_->initialized || !impl_->api.aclrtMalloc) return nullptr;
    void* devPtr = nullptr;
    // 0 is ACL_MEM_MALLOC_HUGE_FIRST
    if (impl_->api.aclrtMalloc(&devPtr, bytes, 0) != 0) return nullptr;
    return devPtr;
}

void GPUComputeCann::free_buf(void* ptr) {
    if (!impl_->initialized || !impl_->api.aclrtFree || !ptr) return;
    impl_->api.aclrtFree(ptr);
}

void GPUComputeCann::upload(const Tensor& src, void* dst) {
    if (!impl_->initialized || !impl_->api.aclrtMemcpy || !src.data()) return;
    // 1 is ACL_MEMCPY_HOST_TO_DEVICE
    impl_->api.aclrtMemcpy(dst, src.numel() * sizeof(float), src.data(), src.numel() * sizeof(float), 1);
}

void GPUComputeCann::download(void* src, Tensor& dst) {
    if (!impl_->initialized || !impl_->api.aclrtMemcpy || !dst.data()) return;
    // 2 is ACL_MEMCPY_DEVICE_TO_HOST
    impl_->api.aclrtMemcpy(dst.data(), dst.numel() * sizeof(float), src, dst.numel() * sizeof(float), 2);
}

void GPUComputeCann::gemm(float alpha, const void* A, const void* B, float beta, void* C, int64_t M, int64_t N, int64_t K) {
    if (impl_->initialized) impl_->execute_op("MatMul");
}

void GPUComputeCann::gemv(float alpha, const void* A, const void* x, float beta, void* y, int64_t M, int64_t N) {
    if (impl_->initialized) impl_->execute_op("MatMul");
}

void GPUComputeCann::relu(const void* x, void* y, int64_t n) {
    if (impl_->initialized) impl_->execute_op("Relu");
}

void GPUComputeCann::gelu(const void* x, void* y, int64_t n) {
    if (impl_->initialized) impl_->execute_op("Gelu");
}

void GPUComputeCann::silu(const void* x, void* y, int64_t n) {
    if (impl_->initialized) impl_->execute_op("Swish");
}

void GPUComputeCann::softmax(const void* x, void* y, int64_t rows, int64_t cols) {
    if (impl_->initialized) impl_->execute_op("SoftmaxV2");
}

void GPUComputeCann::rms_norm(const void* x, const void* weight, void* y, int64_t rows, int64_t cols, float eps) {
    if (impl_->initialized) impl_->execute_op("LayerNorm");
}

void GPUComputeCann::layer_norm(const void* x, const void* gamma, const void* beta, void* y, int64_t rows, int64_t cols, float eps) {
    if (impl_->initialized) impl_->execute_op("LayerNorm");
}

void GPUComputeCann::add(const void* a, const void* b, void* c, int64_t n) {
    if (impl_->initialized) impl_->execute_op("Add");
}

void GPUComputeCann::mul(const void* a, const void* b, void* c, int64_t n) {
    if (impl_->initialized) impl_->execute_op("Mul");
}

void GPUComputeCann::scale(float s, const void* x, void* y, int64_t n) {
    if (impl_->initialized) impl_->execute_op("Scale");
}

void GPUComputeCann::fill(float val, void* x, int64_t n) {
    if (impl_->initialized) impl_->execute_op("Fills");
}

void GPUComputeCann::copy_buf(const void* src, void* dst, int64_t n) {
    if (impl_->initialized && impl_->api.aclrtMemcpy) {
        // 3 is ACL_MEMCPY_DEVICE_TO_DEVICE
        impl_->api.aclrtMemcpy(dst, n * sizeof(float), src, n * sizeof(float), 3);
    }
}

void GPUComputeCann::synchronize() {
    if (impl_->initialized && impl_->api.aclrtSynchronizeStream)
        impl_->api.aclrtSynchronizeStream(impl_->stream);
}

int64_t GPUComputeCann::memory_free() const {
    return 16LL * 1024 * 1024 * 1024;
}

int64_t GPUComputeCann::memory_total() const {
    return 16LL * 1024 * 1024 * 1024;
}

static GPUComputeCann g_cann_compute;
GPUComputeCann& get_cann_compute() { return g_cann_compute; }

} // namespace gpu
} // namespace quant
