#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include "oil/tensor.h"
#include "oil/math.h"
#include "oil/moe_variants.h"
#include "oil/backend.h"
#include "oil/gpu_compute.h"
#include "oil/test.h"

using namespace oil;
using namespace oil::moe;
using namespace oil::backend;

void test_tensor_basics() {
    TEST_SUITE("TENSOR BASICS");

    Tensor t({2, 3});
    TEST_CHECK((t.dim(0) == 2 && t.dim(1) == 3), "shape ok");
    TEST_CHECK((t.numel() == 6), "numel ok");

    t.fill(3.0f);
    for (int64_t i = 0; i < 6; ++i)
        TEST_CHECK((t.data<float>()[i] == 3.0f), "fill ok");

    Tensor t2({2, 3});
    t2.fill(2.0f);
    Tensor t3({2, 3});
    math::add(t, t2, t3);
    for (int64_t i = 0; i < 6; ++i)
        TEST_CHECK((t3.data<float>()[i] == 5.0f), "add ok");

    Tensor t4({2, 3});
    math::mul(t, t2, t4);
    for (int64_t i = 0; i < 6; ++i)
        TEST_CHECK((t4.data<float>()[i] == 6.0f), "mul ok");

    Tensor t5({2, 3});
    math::scale(0.5f, t, t5);
    for (int64_t i = 0; i < 6; ++i)
        TEST_CHECK((t5.data<float>()[i] == 1.5f), "scale ok");
}

void test_math_ops() {
    TEST_SUITE("MATH OPS");

    Tensor x({2, 4});
    x.fill(2.0f);
    Tensor y({2, 4});
    math::relu(x, y);
    for (int64_t i = 0; i < 8; ++i) TEST_CHECK((y.data<float>()[i] == 2.0f), "relu ok");

    x.fill(-1.0f);
    math::relu(x, y);
    for (int64_t i = 0; i < 8; ++i) TEST_CHECK((y.data<float>()[i] == 0.0f), "relu neg ok");

    Tensor a({2, 3});
    Tensor b({3, 4});
    a.fill(1.0f);
    b.fill(2.0f);
    Tensor c({2, 4});
    math::gemm(1.0f, a, b, 0.0f, c);
    for (int64_t i = 0; i < 8; ++i)
        TEST_CHECK((c.data<float>()[i] == 6.0f), "gemm ok");

    Tensor ln({2, 4});
    ln.fill(1.0f);
    Tensor gamma({4});
    Tensor beta({4});
    gamma.fill(1.0f);
    beta.fill(0.0f);
    Tensor ln_out({2, 4});
    math::layer_norm(ln, gamma, beta, 1e-5f, ln_out);
    for (int64_t i = 0; i < 8; ++i)
        TEST_CHECK((std::fabs(ln_out.data<float>()[i] - 0.0f) < 1e-5f), "layer_norm ok");

    Tensor soft_x({2, 3});
    soft_x.data<float>()[0] = 1.0f; soft_x.data<float>()[1] = 2.0f; soft_x.data<float>()[2] = 3.0f;
    soft_x.data<float>()[3] = 1.0f; soft_x.data<float>()[4] = 2.0f; soft_x.data<float>()[5] = 3.0f;
    Tensor soft_y({2, 3});
    math::softmax(soft_x, soft_y, 1);
    float sum0 = soft_y.data<float>()[0] + soft_y.data<float>()[1] + soft_y.data<float>()[2];
    float sum1 = soft_y.data<float>()[3] + soft_y.data<float>()[4] + soft_y.data<float>()[5];
    TEST_CHECK((std::fabs(sum0 - 1.0f) < 1e-5f), "softmax row sum 0");
    TEST_CHECK((std::fabs(sum1 - 1.0f) < 1e-5f), "softmax row sum 1");
}

void test_sparse_moe() {
    TEST_SUITE("SPARSE MoE");
    MoEAllConfig cfg;
    cfg.num_experts = 4;
    cfg.top_k = 2;
    cfg.expert_hidden_size = 32;
    SparseMoE moe(16, cfg);
    Tensor x({2, 4, 16});
    x.fill(0.5f);
    auto out = moe.forward(x);
    TEST_CHECK((out.output.rank() == 3 && out.output.dim(0) == 2 && out.output.dim(1) == 4 && out.output.dim(2) == 16), "sparse output shape");
    TEST_CHECK((std::isfinite(out.load_balance_loss)), "sparse load balance computed");
    TEST_CHECK((out.expert_indices.numel() > 0), "expert indices populated");
    TEST_CHECK((out.expert_weights.numel() > 0), "expert weights populated");
}

void test_soft_moe() {
    TEST_SUITE("SOFT MoE");
    MoEAllConfig cfg;
    cfg.num_experts = 2;
    cfg.num_slots_per_expert = 2;
    cfg.expert_hidden_size = 32;
    SoftMoE moe(16, cfg);
    Tensor x({1, 4, 16});
    x.fill(0.3f);
    auto out = moe.forward(x);
    TEST_CHECK((out.output.rank() == 3 && out.output.dim(0) == 1 && out.output.dim(1) == 4 && out.output.dim(2) == 16), "softmoe output shape");
}

void test_hierarchical_moe() {
    TEST_SUITE("HIERARCHICAL MoE");
    MoEAllConfig cfg;
    cfg.num_groups = 2;
    cfg.experts_per_group = 2;
    cfg.top_groups = 1;
    cfg.top_experts_per_group = 1;
    cfg.expert_hidden_size = 32;
    HierarchicalMoE moe(16, cfg);
    Tensor x({1, 4, 16});
    x.fill(0.5f);
    auto out = moe.forward(x);
    TEST_CHECK((out.output.rank() == 3 && out.output.dim(0) == 1 && out.output.dim(1) == 4 && out.output.dim(2) == 16), "hierarchical output shape");
}

void test_momoe() {
    TEST_SUITE("MoMoE");
    MoEAllConfig cfg;
    cfg.num_groups = 2;
    cfg.experts_per_group = 2;
    cfg.top_groups = 1;
    cfg.top_experts_per_group = 1;
    cfg.expert_hidden_size = 32;
    MoMoE moe(16, cfg);
    Tensor x({1, 4, 16});
    x.fill(0.5f);
    auto out = moe.forward(x);
    TEST_CHECK((out.output.rank() == 3 && out.output.dim(0) == 1 && out.output.dim(1) == 4 && out.output.dim(2) == 16), "momoe output shape");
}

void test_expert_choice_moe() {
    TEST_SUITE("EXPERT CHOICE MoE");
    MoEAllConfig cfg;
    cfg.num_experts = 4;
    cfg.capacity_factor = 1.0f;
    cfg.expert_hidden_size = 32;
    ExpertChoiceMoE moe(16, cfg);
    Tensor x({1, 4, 16});
    x.fill(0.5f);
    auto out = moe.forward(x);
    TEST_CHECK((out.output.rank() == 3 && out.output.dim(0) == 1 && out.output.dim(1) == 4 && out.output.dim(2) == 16), "expert_choice output shape");
}

void test_hash_moe() {
    TEST_SUITE("HASH MoE");
    MoEAllConfig cfg;
    cfg.num_experts = 4;
    cfg.hash_bucket_size = 1;
    cfg.expert_hidden_size = 32;
    HashMoE moe(16, cfg);
    Tensor x({1, 4, 16});
    x.fill(0.5f);
    Tensor token_ids({1, 4}, DType::I64);
    token_ids.fill(0);
    auto out = moe.forward(x, token_ids);
    TEST_CHECK((out.output.rank() == 3 && out.output.dim(0) == 1 && out.output.dim(1) == 4 && out.output.dim(2) == 16), "hash_moe output shape");
}

void test_cross_layer_moe() {
    TEST_SUITE("CROSS-LAYER MoE");
    MoEAllConfig cfg;
    cfg.num_experts = 4;
    cfg.top_k = 1;
    cfg.expert_hidden_size = 32;
    CrossLayerMoE moe(16, cfg);
    Tensor x({1, 4, 16});
    x.fill(0.5f);
    auto out = moe.forward(x, 0);
    TEST_CHECK((out.output.rank() == 3 && out.output.dim(0) == 1 && out.output.dim(1) == 4 && out.output.dim(2) == 16), "cross_layer output shape");
    auto out2 = moe.forward(x, 3);
    TEST_CHECK((out2.output.rank() == 3), "cross_layer reuse shared experts");
}

void test_multimodal_moe() {
    TEST_SUITE("MULTIMODAL MoE");
    MoEAllConfig cfg;
    cfg.num_experts = 8;
    cfg.top_k = 2;
    cfg.expert_hidden_size = 32;
    MultiModalMoE moe(16, cfg);
    Tensor x({1, 4, 16});
    x.fill(0.5f);
    Tensor modal({1, 4}, DType::I64);
    modal.fill(0);
    auto out = moe.forward(x, modal);
    TEST_CHECK((out.output.rank() == 3 && out.output.dim(0) == 1 && out.output.dim(1) == 4 && out.output.dim(2) == 16), "multimodal output shape");
}

void test_utility_fns() {
    TEST_SUITE("UTILITY FUNCTIONS");
    Tensor logits({2, 4});
    logits.fill(1.0f);
    Tensor indices, weights;
    Tensor probs = softmax_with_topk(logits, 2, indices, weights);
    TEST_CHECK((probs.dim(0) == 2 && probs.dim(1) == 4), "softmax_with_topk probs shape");
    TEST_CHECK((indices.dtype() == DType::I64), "indices dtype I64");
    TEST_CHECK((indices.dim(0) == 2 && indices.dim(1) == 2), "indices shape");
    TEST_CHECK((weights.dim(0) == 2 && weights.dim(1) == 2), "weights shape");
    float lb = compute_load_balance_loss(logits, indices, 4);
    TEST_CHECK((lb >= 0.0f), "load balance loss computed");
    float zl = compute_z_loss(probs);
    TEST_CHECK((zl > 0.0f), "z-loss computed");
    int64_t h = hash_token(42, 4);
    TEST_CHECK((h >= 0 && h < 4), "hash_token in range");
}

static void test_backend_ops(ComputeBackend* be, const char* name) {
    // Basic add
    Tensor a({2, 3}), b({2, 3}), c({2, 3});
    a.fill(2.0f);
    b.fill(3.0f);
    be->add(a, b, c);
    {
        char _olab[256];
        snprintf(_olab, sizeof(_olab), "add (%s)", name);
        TEST_CHECK((std::fabs(c.data<float>()[0] - 5.0f) < 1e-5f), _olab);
    }

    // Basic gemm
    Tensor A({2, 3}), B({3, 4}), C({2, 4});
    A.fill(1.0f);
    B.fill(2.0f);
    be->gemm(1.0f, A, B, 0.0f, C);
    {
        char _olab[256];
        snprintf(_olab, sizeof(_olab), "gemm (%s)", name);
        TEST_CHECK((std::fabs(C.data<float>()[0] - 6.0f) < 1e-5f), _olab);
    }

    // Relu
    Tensor rx({2, 4}), ry({2, 4});
    rx.fill(-1.0f);
    be->relu(rx, ry);
    {
        char _olab[256];
        snprintf(_olab, sizeof(_olab), "relu (%s)", name);
        TEST_CHECK((ry.data<float>()[0] == 0.0f), _olab);
    }
    rx.fill(3.0f);
    be->relu(rx, ry);
    {
        char _olab[256];
        snprintf(_olab, sizeof(_olab), "relu pos (%s)", name);
        TEST_CHECK((ry.data<float>()[0] == 3.0f), _olab);
    }

    // Scale
    Tensor sx({2, 3}), sy({2, 3});
    sx.fill(2.0f);
    be->scale(3.0f, sx, sy);
    {
        char _olab[256];
        snprintf(_olab, sizeof(_olab), "scale (%s)", name);
        TEST_CHECK((std::fabs(sy.data<float>()[0] - 6.0f) < 1e-5f), _olab);
    }

    // Memory
    {
        char _olab[256];
        snprintf(_olab, sizeof(_olab), "mem_free (%s)", name);
        TEST_CHECK((be->memory_free() >= 0), _olab);
    }
}

void test_backends() {
    TEST_SUITE("BACKENDS");

    // Test CPU_SCALAR
    {
        auto be = ComputeBackend::create(BackendConfig{BackendType::CPU_SCALAR});
        TEST_CHECK((be != nullptr), "CPU_SCALAR created");
        test_backend_ops(be, "CPU_SCALAR");
    }

    // Test CPU_AVX2 
    {
        auto be = ComputeBackend::create(BackendConfig{BackendType::CPU_AVX2});
        if (be && backend::is_avx2_available()) {
            TEST_CHECK(true, "CPU_AVX2 created");
            test_backend_ops(be, "CPU_AVX2");
        } else {
            printf("  SKIP CPU_AVX2 (not available on this platform)\n");
        }
    }

    // Test IGPU_SHARED if DirectX available
    {
        auto be = ComputeBackend::create(BackendConfig{BackendType::IGPU_SHARED});
        TEST_CHECK((be != nullptr), "IGPU_SHARED created");
        if (be && backend::is_directx_available()) {
            test_backend_ops(be, "IGPU_SHARED");
        } else {
            printf("  SKIP IGPU_SHARED ops (DirectX not available)\n");
        }
    }

    // Test GPU_DIRECTX if DirectX available
    {
        auto be = ComputeBackend::create(BackendConfig{BackendType::GPU_DIRECTX});
        TEST_CHECK((be != nullptr), "GPU_DIRECTX created");
        if (be && backend::is_directx_available()) {
            test_backend_ops(be, "GPU_DIRECTX");
        } else {
            printf("  SKIP GPU_DIRECTX ops (DirectX not available)\n");
        }
    }

    // Test RAM_SWAP
    {
        auto be = ComputeBackend::create(BackendConfig{BackendType::RAM_SWAP});
        TEST_CHECK((be != nullptr), "RAM_SWAP created");
        test_backend_ops(be, "RAM_SWAP");
    }

    // Test DISTRIBUTED (single-node)
    {
        BackendConfig dcfg;
        dcfg.type = BackendType::DISTRIBUTED;
        dcfg.num_devices = 1;
        auto be = ComputeBackend::create(dcfg);
        TEST_CHECK((be != nullptr), "DISTRIBUTED created");
        test_backend_ops(be, "DISTRIBUTED");
    }
}

void test_gpu_detection() {
    TEST_SUITE("GPU DETECTION");
    bool dx = backend::is_directx_available();
    bool cuda = backend::is_cuda_available();
    bool vk = backend::is_vulkan_available();
    bool avx2 = backend::is_avx2_available();
    int64_t gpu_mem = backend::gpu_memory_free(0);

    printf("  DirectX available: %s\n", dx ? "yes" : "no");
    printf("  CUDA available: %s\n", cuda ? "yes" : "no");
    printf("  Vulkan available: %s\n", vk ? "yes" : "no");
    printf("  AVX2 available: %s\n", avx2 ? "yes" : "no");
    printf("  GPU mem free: %lld\n", (long long)gpu_mem);

    // Verify detection functions return consistent results (call twice, must match)
    TEST_CHECK(backend::is_avx2_available() == avx2, "is_avx2_available() is deterministic");
    TEST_CHECK(backend::is_directx_available() == dx, "is_directx_available() is deterministic");
    TEST_CHECK(backend::is_cuda_available() == cuda, "is_cuda_available() is deterministic");
    TEST_CHECK(backend::is_vulkan_available() == vk, "is_vulkan_available() is deterministic");
    TEST_CHECK(gpu_mem >= 0, "gpu_memory_free() returns >= 0");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("MYTHOS.cpp — MoE Variants + Backend Test Suite\n");
    printf("==============================================\n");

    test_tensor_basics();
    test_math_ops();
    test_sparse_moe();
    test_soft_moe();
    test_hierarchical_moe();
    test_momoe();
    test_expert_choice_moe();
    test_hash_moe();
    test_cross_layer_moe();
    test_multimodal_moe();
    test_utility_fns();
    test_backends();
    test_gpu_detection();

    return TEST_REPORT() > 0 ? 1 : 0;
}
