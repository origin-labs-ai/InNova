#include "quant/gpu_compute_cuda.h"
#include <string>
#include <vector>
#include <mutex>
#include <stdexcept>
#include <cstring>
#include <cstdio>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace quant {
namespace gpu {

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0

typedef CUresult (*PFN_cuInit)(unsigned int);
typedef CUresult (*PFN_cuDeviceGet)(CUdevice*, int);
typedef CUresult (*PFN_cuDeviceTotalMem)(size_t*, CUdevice);
typedef CUresult (*PFN_cuCtxCreate)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*PFN_cuCtxDestroy)(CUcontext);
typedef CUresult (*PFN_cuModuleLoadData)(CUmodule*, const void*);
typedef CUresult (*PFN_cuModuleUnload)(CUmodule);
typedef CUresult (*PFN_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult (*PFN_cuMemAlloc)(CUdeviceptr*, size_t);
typedef CUresult (*PFN_cuMemFree)(CUdeviceptr);
typedef CUresult (*PFN_cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult (*PFN_cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream);
typedef CUresult (*PFN_cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
typedef CUresult (*PFN_cuStreamCreate)(CUstream*, unsigned int);
typedef CUresult (*PFN_cuStreamDestroy)(CUstream);
typedef CUresult (*PFN_cuStreamSynchronize)(CUstream);
typedef CUresult (*PFN_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);

static const char* CUDA_PTX_CODE = R"(
.version 6.5
.target sm_30
.address_size 64

.visible .entry gemm_kernel(
	.param .f32 alpha, .param .u64 A, .param .u64 B, .param .f32 beta, .param .u64 C,
	.param .u32 M, .param .u32 N, .param .u32 K) {
    .reg .u32 %tx, %ty, %bx, %by, %rCol, %rRow, %rM, %rN, %rK, %k, %aIdx, %bIdx, %cIdx;
    .reg .f32 %fA, %fB, %fSum, %fAlpha, %fBeta, %fC;
    .reg .pred %pExit, %pK;
    .reg .u64 %ptr, %ptrA, %ptrB, %ptrC, %offset;

    mov.u32 %tx, %tid.x; mov.u32 %ty, %tid.y;
    mov.u32 %bx, %ctaid.x; mov.u32 %by, %ctaid.y;
    mad.lo.u32 %rCol, %bx, 16, %tx;
    mad.lo.u32 %rRow, %by, 16, %ty;
    ld.param.u32 %rM, [M]; ld.param.u32 %rN, [N]; ld.param.u32 %rK, [K];
    setp.ge.u32 %pExit, %rCol, %rN; @%pExit bra L_EXIT;
    setp.ge.u32 %pExit, %rRow, %rM; @%pExit bra L_EXIT;

    mov.f32 %fSum, 0.0; mov.u32 %k, 0;
    ld.param.u64 %ptrA, [A]; ld.param.u64 %ptrB, [B];

L_LOOP:
    setp.ge.u32 %pK, %k, %rK; @%pK bra L_END_LOOP;
    mad.lo.u32 %aIdx, %rRow, %rK, %k;
    cvt.u64.u32 %offset, %aIdx; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrA, %offset;
    ld.global.f32 %fA, [%ptr];
    mad.lo.u32 %bIdx, %k, %rN, %rCol;
    cvt.u64.u32 %offset, %bIdx; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrB, %offset;
    ld.global.f32 %fB, [%ptr];
    fma.rn.f32 %fSum, %fA, %fB, %fSum;
    add.u32 %k, %k, 1;
    bra L_LOOP;
L_END_LOOP:
    ld.param.f32 %fAlpha, [alpha]; ld.param.f32 %fBeta, [beta]; ld.param.u64 %ptrC, [C];
    mul.f32 %fSum, %fSum, %fAlpha;
    mad.lo.u32 %cIdx, %rRow, %rN, %rCol;
    cvt.u64.u32 %offset, %cIdx; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrC, %offset;
    setp.eq.f32 %pExit, %fBeta, 0.0; @%pExit bra L_STORE;
    ld.global.f32 %fC, [%ptr];
    fma.rn.f32 %fSum, %fC, %fBeta, %fSum;
L_STORE:
    st.global.f32 [%ptr], %fSum;
L_EXIT:
    ret;
}

.visible .entry relu_kernel(.param .u64 X, .param .u64 Y, .param .u32 N) {
    .reg .u32 %idx, %rN; .reg .pred %p; .reg .f32 %fX, %fZ; .reg .u64 %ptrX, %ptrY, %offset, %ptr;
    mov.u32 %idx, %ctaid.x; mov.u32 %rN, %ntid.x; mad.lo.u32 %idx, %idx, %rN, %tid.x;
    ld.param.u32 %rN, [N]; setp.ge.u32 %p, %idx, %rN; @%p bra L_EXIT;
    ld.param.u64 %ptrX, [X]; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fX, [%ptr];
    mov.f32 %fZ, 0.0; max.f32 %fX, %fX, %fZ;
    ld.param.u64 %ptrY, [Y]; add.u64 %ptr, %ptrY, %offset; st.global.f32 [%ptr], %fX;
L_EXIT: ret;
}

.visible .entry silu_kernel(.param .u64 X, .param .u64 Y, .param .u32 N) {
    .reg .u32 %idx, %rN; .reg .pred %p; .reg .f32 %fX, %fZ; .reg .u64 %ptrX, %ptrY, %offset, %ptr;
    mov.u32 %idx, %ctaid.x; mov.u32 %rN, %ntid.x; mad.lo.u32 %idx, %idx, %rN, %tid.x;
    ld.param.u32 %rN, [N]; setp.ge.u32 %p, %idx, %rN; @%p bra L_EXIT;
    ld.param.u64 %ptrX, [X]; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fX, [%ptr];
    mul.f32 %fZ, %fX, -1.44269504; ex2.approx.f32 %fZ, %fZ; add.f32 %fZ, %fZ, 1.0; rcp.approx.f32 %fZ, %fZ; mul.f32 %fX, %fX, %fZ;
    ld.param.u64 %ptrY, [Y]; add.u64 %ptr, %ptrY, %offset; st.global.f32 [%ptr], %fX;
L_EXIT: ret;
}

.visible .entry gelu_kernel(.param .u64 X, .param .u64 Y, .param .u32 N) {
    .reg .u32 %idx, %rN; .reg .pred %p; .reg .f32 %fX, %fZ; .reg .u64 %ptrX, %ptrY, %offset, %ptr;
    mov.u32 %idx, %ctaid.x; mov.u32 %rN, %ntid.x; mad.lo.u32 %idx, %idx, %rN, %tid.x;
    ld.param.u32 %rN, [N]; setp.ge.u32 %p, %idx, %rN; @%p bra L_EXIT;
    ld.param.u64 %ptrX, [X]; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fX, [%ptr];
    mul.f32 %fZ, %fX, -2.4554649; ex2.approx.f32 %fZ, %fZ; add.f32 %fZ, %fZ, 1.0; rcp.approx.f32 %fZ, %fZ; mul.f32 %fX, %fX, %fZ;
    ld.param.u64 %ptrY, [Y]; add.u64 %ptr, %ptrY, %offset; st.global.f32 [%ptr], %fX;
L_EXIT: ret;
}

.visible .entry add_kernel(.param .u64 A, .param .u64 B, .param .u64 C, .param .u32 N) {
    .reg .u32 %idx, %rN; .reg .pred %p; .reg .f32 %fA, %fB; .reg .u64 %ptrA, %ptrB, %ptrC, %offset, %ptr;
    mov.u32 %idx, %ctaid.x; mov.u32 %rN, %ntid.x; mad.lo.u32 %idx, %idx, %rN, %tid.x;
    ld.param.u32 %rN, [N]; setp.ge.u32 %p, %idx, %rN; @%p bra L_EXIT;
    cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    ld.param.u64 %ptrA, [A]; add.u64 %ptr, %ptrA, %offset; ld.global.f32 %fA, [%ptr];
    ld.param.u64 %ptrB, [B]; add.u64 %ptr, %ptrB, %offset; ld.global.f32 %fB, [%ptr];
    add.f32 %fA, %fA, %fB;
    ld.param.u64 %ptrC, [C]; add.u64 %ptr, %ptrC, %offset; st.global.f32 [%ptr], %fA;
L_EXIT: ret;
}

.visible .entry mul_kernel(.param .u64 A, .param .u64 B, .param .u64 C, .param .u32 N) {
    .reg .u32 %idx, %rN; .reg .pred %p; .reg .f32 %fA, %fB; .reg .u64 %ptrA, %ptrB, %ptrC, %offset, %ptr;
    mov.u32 %idx, %ctaid.x; mov.u32 %rN, %ntid.x; mad.lo.u32 %idx, %idx, %rN, %tid.x;
    ld.param.u32 %rN, [N]; setp.ge.u32 %p, %idx, %rN; @%p bra L_EXIT;
    cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    ld.param.u64 %ptrA, [A]; add.u64 %ptr, %ptrA, %offset; ld.global.f32 %fA, [%ptr];
    ld.param.u64 %ptrB, [B]; add.u64 %ptr, %ptrB, %offset; ld.global.f32 %fB, [%ptr];
    mul.f32 %fA, %fA, %fB;
    ld.param.u64 %ptrC, [C]; add.u64 %ptr, %ptrC, %offset; st.global.f32 [%ptr], %fA;
L_EXIT: ret;
}

.visible .entry scale_kernel(.param .f32 S, .param .u64 X, .param .u64 Y, .param .u32 N) {
    .reg .u32 %idx, %rN; .reg .pred %p; .reg .f32 %fX, %fS; .reg .u64 %ptrX, %ptrY, %offset, %ptr;
    mov.u32 %idx, %ctaid.x; mov.u32 %rN, %ntid.x; mad.lo.u32 %idx, %idx, %rN, %tid.x;
    ld.param.u32 %rN, [N]; setp.ge.u32 %p, %idx, %rN; @%p bra L_EXIT;
    ld.param.u64 %ptrX, [X]; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fX, [%ptr];
    ld.param.f32 %fS, [S]; mul.f32 %fX, %fX, %fS;
    ld.param.u64 %ptrY, [Y]; add.u64 %ptr, %ptrY, %offset; st.global.f32 [%ptr], %fX;
L_EXIT: ret;
}

.visible .entry fill_kernel(.param .f32 V, .param .u64 X, .param .u32 N) {
    .reg .u32 %idx, %rN; .reg .pred %p; .reg .f32 %fV; .reg .u64 %ptrX, %offset, %ptr;
    mov.u32 %idx, %ctaid.x; mov.u32 %rN, %ntid.x; mad.lo.u32 %idx, %idx, %rN, %tid.x;
    ld.param.u32 %rN, [N]; setp.ge.u32 %p, %idx, %rN; @%p bra L_EXIT;
    ld.param.u64 %ptrX, [X]; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.param.f32 %fV, [V]; st.global.f32 [%ptr], %fV;
L_EXIT: ret;
}

.visible .entry softmax_kernel(.param .u64 X, .param .u64 Y, .param .u32 rows, .param .u32 cols) {
    .reg .u32 %row, %col, %rRows, %rCols, %idx; .reg .pred %p; .reg .f32 %fMax, %fVal, %fSum, %fInv;
    .reg .u64 %ptrX, %ptrY, %offset, %ptr;
    mov.u32 %row, %ctaid.x; ld.param.u32 %rRows, [rows]; setp.ge.u32 %p, %row, %rRows; @%p bra L_EXIT;
    ld.param.u32 %rCols, [cols]; ld.param.u64 %ptrX, [X]; ld.param.u64 %ptrY, [Y];
    mov.u32 %col, 0; mov.f32 %fMax, -1000000.0;
L_MAX_LOOP:
    setp.ge.u32 %p, %col, %rCols; @%p bra L_MAX_END;
    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fVal, [%ptr]; max.f32 %fMax, %fMax, %fVal;
    add.u32 %col, %col, 1; bra L_MAX_LOOP;
L_MAX_END:
    mov.u32 %col, 0; mov.f32 %fSum, 0.0;
L_EXP_LOOP:
    setp.ge.u32 %p, %col, %rCols; @%p bra L_EXP_END;
    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fVal, [%ptr]; sub.f32 %fVal, %fVal, %fMax;
    mul.f32 %fVal, %fVal, 1.44269504; ex2.approx.f32 %fVal, %fVal;
    add.u64 %ptr, %ptrY, %offset; st.global.f32 [%ptr], %fVal; add.f32 %fSum, %fSum, %fVal;
    add.u32 %col, %col, 1; bra L_EXP_LOOP;
L_EXP_END:
    rcp.approx.f32 %fInv, %fSum; mov.u32 %col, 0;
L_NORM_LOOP:
    setp.ge.u32 %p, %col, %rCols; @%p bra L_EXIT;
    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrY, %offset; ld.global.f32 %fVal, [%ptr]; mul.f32 %fVal, %fVal, %fInv;
    st.global.f32 [%ptr], %fVal; add.u32 %col, %col, 1; bra L_NORM_LOOP;
L_EXIT: ret;
}

.visible .entry rmsnorm_kernel(.param .u64 X, .param .u64 W, .param .u64 Y, .param .u32 rows, .param .u32 cols, .param .f32 eps) {
    .reg .u32 %row, %col, %rRows, %rCols, %idx; .reg .pred %p; .reg .f32 %fSum, %fVal, %fW, %fEps, %fScale;
    .reg .u64 %ptrX, %ptrW, %ptrY, %offset, %ptr;
    mov.u32 %row, %ctaid.x; ld.param.u32 %rRows, [rows]; setp.ge.u32 %p, %row, %rRows; @%p bra L_EXIT;
    ld.param.u32 %rCols, [cols]; ld.param.u64 %ptrX, [X]; ld.param.u64 %ptrW, [W]; ld.param.u64 %ptrY, [Y];
    mov.u32 %col, 0; mov.f32 %fSum, 0.0;
L_SUM_LOOP:
    setp.ge.u32 %p, %col, %rCols; @%p bra L_SUM_END;
    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fVal, [%ptr]; fma.rn.f32 %fSum, %fVal, %fVal, %fSum;
    add.u32 %col, %col, 1; bra L_SUM_LOOP;
L_SUM_END:
    cvt.rn.f32.u32 %fScale, %rCols; rcp.approx.f32 %fScale, %fScale; mul.f32 %fSum, %fSum, %fScale;
    ld.param.f32 %fEps, [eps]; add.f32 %fSum, %fSum, %fEps; rsqrt.approx.f32 %fSum, %fSum;
    mov.u32 %col, 0;
L_NORM_LOOP:
    setp.ge.u32 %p, %col, %rCols; @%p bra L_EXIT;
    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fVal, [%ptr];
    cvt.u64.u32 %offset, %col; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrW, %offset; ld.global.f32 %fW, [%ptr];
    mul.f32 %fVal, %fVal, %fSum; mul.f32 %fVal, %fVal, %fW;
    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrY, %offset; st.global.f32 [%ptr], %fVal; add.u32 %col, %col, 1; bra L_NORM_LOOP;
L_EXIT: ret;
}

.visible .entry rope_kernel(.param .u64 X, .param .u32 N) { ret; }
.visible .entry attention_kernel(.param .u64 X, .param .u32 N) { ret; }
.visible .entry reduce_sum_kernel(.param .u64 X, .param .u32 N) { ret; }
.visible .entry reduce_max_kernel(.param .u64 X, .param .u32 N) { ret; }
.visible .entry moe_gather_kernel(.param .u64 X, .param .u32 N) { ret; }
.visible .entry moe_scatter_kernel(.param .u64 X, .param .u32 N) { ret; }
)";

struct GPUComputeCuda::Impl {
    bool ok = false;
    void* lib = nullptr;

    PFN_cuInit cuInit = nullptr;
    PFN_cuDeviceGet cuDeviceGet = nullptr;
    PFN_cuDeviceTotalMem cuDeviceTotalMem = nullptr;
    PFN_cuCtxCreate cuCtxCreate = nullptr;
    PFN_cuCtxDestroy cuCtxDestroy = nullptr;
    PFN_cuModuleLoadData cuModuleLoadData = nullptr;
    PFN_cuModuleUnload cuModuleUnload = nullptr;
    PFN_cuModuleGetFunction cuModuleGetFunction = nullptr;
    PFN_cuMemAlloc cuMemAlloc = nullptr;
    PFN_cuMemFree cuMemFree = nullptr;
    PFN_cuMemcpyHtoDAsync cuMemcpyHtoDAsync = nullptr;
    PFN_cuMemcpyDtoHAsync cuMemcpyDtoHAsync = nullptr;
    PFN_cuMemcpyDtoDAsync cuMemcpyDtoDAsync = nullptr;
    PFN_cuStreamCreate cuStreamCreate = nullptr;
    PFN_cuStreamDestroy cuStreamDestroy = nullptr;
    PFN_cuStreamSynchronize cuStreamSynchronize = nullptr;
    PFN_cuLaunchKernel cuLaunchKernel = nullptr;

    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUstream compute_stream;
    CUstream copy_stream;

    CUfunction f_gemm, f_relu, f_silu, f_gelu, f_add, f_mul, f_scale, f_fill, f_softmax, f_rmsnorm;
    
    std::vector<CUdeviceptr> bufs;
    std::mutex mtx;

    void* load_func(const char* name) {
#if defined(_WIN32)
        return (void*)GetProcAddress((HMODULE)lib, name);
#else
        return dlsym(lib, name);
#endif
    }

    bool load_lib() {
#if defined(_WIN32)
        lib = LoadLibraryA("nvcuda.dll");
#else
        lib = dlopen("libcuda.so", RTLD_NOW);
        if (!lib) lib = dlopen("libcuda.so.1", RTLD_NOW);
#endif
        return lib != nullptr;
    }

    bool init(int64_t device_id) {
        if (!load_lib()) return false;
        
        cuInit = (PFN_cuInit)load_func("cuInit");
        cuDeviceGet = (PFN_cuDeviceGet)load_func("cuDeviceGet");
        cuDeviceTotalMem = (PFN_cuDeviceTotalMem)load_func("cuDeviceTotalMem");
        cuCtxCreate = (PFN_cuCtxCreate)load_func("cuCtxCreate");
        cuCtxDestroy = (PFN_cuCtxDestroy)load_func("cuCtxDestroy");
        cuModuleLoadData = (PFN_cuModuleLoadData)load_func("cuModuleLoadData");
        cuModuleUnload = (PFN_cuModuleUnload)load_func("cuModuleUnload");
        cuModuleGetFunction = (PFN_cuModuleGetFunction)load_func("cuModuleGetFunction");
        cuMemAlloc = (PFN_cuMemAlloc)load_func("cuMemAlloc");
        cuMemFree = (PFN_cuMemFree)load_func("cuMemFree");
        cuMemcpyHtoDAsync = (PFN_cuMemcpyHtoDAsync)load_func("cuMemcpyHtoDAsync");
        cuMemcpyDtoHAsync = (PFN_cuMemcpyDtoHAsync)load_func("cuMemcpyDtoHAsync");
        cuMemcpyDtoDAsync = (PFN_cuMemcpyDtoDAsync)load_func("cuMemcpyDtoDAsync");
        cuStreamCreate = (PFN_cuStreamCreate)load_func("cuStreamCreate");
        cuStreamDestroy = (PFN_cuStreamDestroy)load_func("cuStreamDestroy");
        cuStreamSynchronize = (PFN_cuStreamSynchronize)load_func("cuStreamSynchronize");
        cuLaunchKernel = (PFN_cuLaunchKernel)load_func("cuLaunchKernel");

        if (!cuInit || !cuLaunchKernel) return false;

        if (cuInit(0) != CUDA_SUCCESS) return false;
        if (cuDeviceGet(&device, (int)device_id) != CUDA_SUCCESS) return false;
        if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) return false;
        
        if (cuStreamCreate(&compute_stream, 0) != CUDA_SUCCESS) return false;
        if (cuStreamCreate(&copy_stream, 0) != CUDA_SUCCESS) return false;

        if (cuModuleLoadData(&module, CUDA_PTX_CODE) != CUDA_SUCCESS) return false;
        
        cuModuleGetFunction(&f_gemm, module, "gemm_kernel");
        cuModuleGetFunction(&f_relu, module, "relu_kernel");
        cuModuleGetFunction(&f_silu, module, "silu_kernel");
        cuModuleGetFunction(&f_gelu, module, "gelu_kernel");
        cuModuleGetFunction(&f_add, module, "add_kernel");
        cuModuleGetFunction(&f_mul, module, "mul_kernel");
        cuModuleGetFunction(&f_scale, module, "scale_kernel");
        cuModuleGetFunction(&f_fill, module, "fill_kernel");
        cuModuleGetFunction(&f_softmax, module, "softmax_kernel");
        cuModuleGetFunction(&f_rmsnorm, module, "rmsnorm_kernel");

        ok = true;
        return true;
    }

    void shutdown() {
        if (!ok) return;
        for (auto b : bufs) cuMemFree(b);
        bufs.clear();
        if (compute_stream) cuStreamDestroy(compute_stream);
        if (copy_stream) cuStreamDestroy(copy_stream);
        if (module) cuModuleUnload(module);
        if (context) cuCtxDestroy(context);
#if defined(_WIN32)
        if (lib) FreeLibrary((HMODULE)lib);
#else
        if (lib) dlclose(lib);
#endif
        ok = false;
    }
};

GPUComputeCuda::GPUComputeCuda() : impl_(new Impl()) {}
GPUComputeCuda::~GPUComputeCuda() { delete impl_; }

bool GPUComputeCuda::init(int64_t device_id) {
    if (impl_->ok) return true;
    return impl_->init(device_id);
}
bool GPUComputeCuda::is_initialized() const { return impl_->ok; }
void GPUComputeCuda::shutdown() { impl_->shutdown(); }

void* GPUComputeCuda::alloc(int64_t bytes) {
    if (!impl_->ok || bytes <= 0) return nullptr;
    CUdeviceptr ptr = 0;
    if (impl_->cuMemAlloc(&ptr, bytes) == CUDA_SUCCESS) {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        impl_->bufs.push_back(ptr);
        return (void*)ptr;
    }
    return nullptr;
}

void GPUComputeCuda::free_buf(void* ptr) {
    if (!impl_->ok || !ptr) return;
    CUdeviceptr dptr = (CUdeviceptr)ptr;
    impl_->cuMemFree(dptr);
    std::lock_guard<std::mutex> lk(impl_->mtx);
    for (auto it = impl_->bufs.begin(); it != impl_->bufs.end(); ++it) {
        if (*it == dptr) {
            impl_->bufs.erase(it);
            break;
        }
    }
}

void GPUComputeCuda::upload(const Tensor& src, void* dst) {
    if (!impl_->ok || !dst || src.numel() == 0) return;
    impl_->cuMemcpyHtoDAsync((CUdeviceptr)dst, src.data(), src.size_bytes(), impl_->copy_stream);
    impl_->cuStreamSynchronize(impl_->copy_stream);
}

void GPUComputeCuda::download(void* src, Tensor& dst) {
    if (!impl_->ok || !src || dst.numel() == 0) return;
    impl_->cuMemcpyDtoHAsync(dst.data(), (CUdeviceptr)src, dst.size_bytes(), impl_->copy_stream);
    impl_->cuStreamSynchronize(impl_->copy_stream);
}

void GPUComputeCuda::gemm(float alpha, const void* A, const void* B, float beta, void* C, int64_t M, int64_t N, int64_t K) {
    if (!impl_->ok || !impl_->f_gemm) return;
    CUdeviceptr dA = (CUdeviceptr)A, dB = (CUdeviceptr)B, dC = (CUdeviceptr)C;
    uint32_t uM = M, uN = N, uK = K;
    void* args[] = { &alpha, &dA, &dB, &beta, &dC, &uM, &uN, &uK };
    unsigned int gx = (N + 15) / 16;
    unsigned int gy = (M + 15) / 16;
    impl_->cuLaunchKernel(impl_->f_gemm, gx, gy, 1, 16, 16, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::gemv(float alpha, const void* A, const void* x, float beta, void* y, int64_t M, int64_t N) {
    gemm(alpha, A, x, beta, y, M, 1, N);
}

void GPUComputeCuda::relu(const void* x, void* y, int64_t n) {
    if (!impl_->ok || !impl_->f_relu) return;
    CUdeviceptr dX = (CUdeviceptr)x, dY = (CUdeviceptr)y;
    uint32_t uN = n;
    void* args[] = { &dX, &dY, &uN };
    impl_->cuLaunchKernel(impl_->f_relu, (n + 255) / 256, 1, 1, 256, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::gelu(const void* x, void* y, int64_t n) {
    if (!impl_->ok || !impl_->f_gelu) return;
    CUdeviceptr dX = (CUdeviceptr)x, dY = (CUdeviceptr)y;
    uint32_t uN = n;
    void* args[] = { &dX, &dY, &uN };
    impl_->cuLaunchKernel(impl_->f_gelu, (n + 255) / 256, 1, 1, 256, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::silu(const void* x, void* y, int64_t n) {
    if (!impl_->ok || !impl_->f_silu) return;
    CUdeviceptr dX = (CUdeviceptr)x, dY = (CUdeviceptr)y;
    uint32_t uN = n;
    void* args[] = { &dX, &dY, &uN };
    impl_->cuLaunchKernel(impl_->f_silu, (n + 255) / 256, 1, 1, 256, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::softmax(const void* x, void* y, int64_t rows, int64_t cols) {
    if (!impl_->ok || !impl_->f_softmax) return;
    CUdeviceptr dX = (CUdeviceptr)x, dY = (CUdeviceptr)y;
    uint32_t uRows = rows, uCols = cols;
    void* args[] = { &dX, &dY, &uRows, &uCols };
    impl_->cuLaunchKernel(impl_->f_softmax, rows, 1, 1, 1, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::rms_norm(const void* x, const void* weight, void* y, int64_t rows, int64_t cols, float eps) {
    if (!impl_->ok || !impl_->f_rmsnorm) return;
    CUdeviceptr dX = (CUdeviceptr)x, dW = (CUdeviceptr)weight, dY = (CUdeviceptr)y;
    uint32_t uRows = rows, uCols = cols;
    void* args[] = { &dX, &dW, &dY, &uRows, &uCols, &eps };
    impl_->cuLaunchKernel(impl_->f_rmsnorm, rows, 1, 1, 1, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::layer_norm(const void* x, const void* gamma, const void* beta, void* y, int64_t rows, int64_t cols, float eps) {
    // simplified fallback for now
    rms_norm(x, gamma, y, rows, cols, eps);
}

void GPUComputeCuda::add(const void* a, const void* b, void* c, int64_t n) {
    if (!impl_->ok || !impl_->f_add) return;
    CUdeviceptr dA = (CUdeviceptr)a, dB = (CUdeviceptr)b, dC = (CUdeviceptr)c;
    uint32_t uN = n;
    void* args[] = { &dA, &dB, &dC, &uN };
    impl_->cuLaunchKernel(impl_->f_add, (n + 255) / 256, 1, 1, 256, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::mul(const void* a, const void* b, void* c, int64_t n) {
    if (!impl_->ok || !impl_->f_mul) return;
    CUdeviceptr dA = (CUdeviceptr)a, dB = (CUdeviceptr)b, dC = (CUdeviceptr)c;
    uint32_t uN = n;
    void* args[] = { &dA, &dB, &dC, &uN };
    impl_->cuLaunchKernel(impl_->f_mul, (n + 255) / 256, 1, 1, 256, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::scale(float s, const void* x, void* y, int64_t n) {
    if (!impl_->ok || !impl_->f_scale) return;
    CUdeviceptr dX = (CUdeviceptr)x, dY = (CUdeviceptr)y;
    uint32_t uN = n;
    void* args[] = { &s, &dX, &dY, &uN };
    impl_->cuLaunchKernel(impl_->f_scale, (n + 255) / 256, 1, 1, 256, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::fill(float val, void* x, int64_t n) {
    if (!impl_->ok || !impl_->f_fill) return;
    CUdeviceptr dX = (CUdeviceptr)x;
    uint32_t uN = n;
    void* args[] = { &val, &dX, &uN };
    impl_->cuLaunchKernel(impl_->f_fill, (n + 255) / 256, 1, 1, 256, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::copy_buf(const void* src, void* dst, int64_t n) {
    if (!impl_->ok || !src || !dst) return;
    impl_->cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, n * sizeof(float), impl_->compute_stream);
}

void GPUComputeCuda::synchronize() {
    if (impl_->ok) {
        impl_->cuStreamSynchronize(impl_->compute_stream);
        impl_->cuStreamSynchronize(impl_->copy_stream);
    }
}

int64_t GPUComputeCuda::memory_free() const {
    // simplified
    return 1LL * 1024 * 1024 * 1024;
}

int64_t GPUComputeCuda::memory_total() const {
    if (!impl_->ok) return 0;
    size_t mem = 0;
    impl_->cuDeviceTotalMem(&mem, impl_->device);
    return mem;
}

static GPUComputeCuda g_cuda_compute;
GPUComputeCuda& get_cuda_compute() { return g_cuda_compute; }

} // namespace gpu
} // namespace quant
