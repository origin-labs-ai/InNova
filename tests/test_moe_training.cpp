#define NOMINMAX
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "oil/moe_trainer.h"
#include "oil/moe_model.h"
#include "oil/math.h"
#include "oil/optimizer.h"
#include "oil/tokenizer.h"
#include "oil/random.h"

using namespace oil;

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { std::cout << "  " << name << " ... "; tests_run++; } while(0)
#define PASS() do { std::cout << "PASS" << std::endl; tests_passed++; } while(0)
#define FAIL(msg) do { std::cout << "FAIL: " << msg << std::endl; return 1; } while(0)

static MoEModel create_test_moe_model() {
    TransformerConfig cfg;
    cfg.vocab_size = 128;
    cfg.hidden_size = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.ffn_hidden_size = 128;
    cfg.norm_eps = 1e-5f;
    cfg.rope_theta = 10000.0f;
    cfg.max_seq_len = 128;
    cfg.activation = Activation::SiLU;
    cfg.num_kv_heads = 2;

    moe::MoEAllConfig moe_cfg;
    moe_cfg.variant = moe::MoEVariant::SPARSE_TOPK;
    moe_cfg.num_experts = 4;
    moe_cfg.top_k = 2;
    moe_cfg.expert_hidden_size = 64;
    moe_cfg.load_balance_coef = 0.01f;
    moe_cfg.z_loss_coef = 0.001f;
    moe_cfg.use_shared_expert = true;
    moe_cfg.num_shared_experts = 1;
    moe_cfg.capacity_strategy = moe::CapacityStrategy::TOKEN_DROP;
    moe_cfg.capacity_factor = 1.5f;

    return MoEModel(cfg, moe_cfg);
}

static int test_moe_trainer_construct() {
    TEST("MoETrainer construction");
    MoEModel model = create_test_moe_model();
    MoETrainer trainer(&model, nullptr);
    PASS();
    return 0;
}

static int test_moe_trainer_compile() {
    TEST("MoETrainer compile with optimizer");
    MoEModel model = create_test_moe_model();
    MoETrainer trainer(&model, nullptr);

    AdamW opt(1e-3f);
    MoETrainConfig cfg;
    cfg.batch_size = 2;
    cfg.seq_length = 32;
    cfg.train_steps = 10;
    cfg.gradient_accumulation_steps = 1;
    cfg.load_balance_coef = 0.01f;
    cfg.z_loss_coef = 0.001f;
    cfg.aux_loss_coef = 0.01f;

    trainer.compile(&opt, cfg);
    PASS();
    return 0;
}

static int test_moe_trainer_forward_pass() {
    TEST("MoETrainer forward pass");
    MoEModel model = create_test_moe_model();
    MoETrainer trainer(&model, nullptr);

    AdamW opt(1e-3f);
    MoETrainConfig cfg;
    cfg.batch_size = 2;
    cfg.seq_length = 16;
    cfg.train_steps = 5;
    trainer.compile(&opt, cfg);

    RNG rng(12345);
    Tensor input_ids({2, 16});
    Tensor labels({2, 16});
    float* id = input_ids.data<float>();
    float* lb = labels.data<float>();
    for (int i = 0; i < 32; i++) {
        id[i] = (float)rng.uniform_int(0, 127);
        lb[i] = (float)rng.uniform_int(0, 127);
    }

    Tensor logits = model.forward(input_ids, Tensor(), nullptr);
    assert(logits.dim(0) == 2);
    assert(logits.dim(1) == 16);
    assert(logits.dim(2) == 128);

    float loss = trainer.micro_step(input_ids, labels);
    assert(std::isfinite(loss));
    assert(loss > 0);

    PASS();
    return 0;
}

static int test_moe_trainer_params_collected() {
    TEST("MoETrainer collects model parameters");
    MoEModel model = create_test_moe_model();
    MoETrainer trainer(&model, nullptr);

    AdamW opt(1e-3f);
    MoETrainConfig cfg;
    trainer.compile(&opt, cfg);

    auto params = trainer.get_model_params();
    assert(params.size() > 10);

    int64_t total_params = 0;
    for (auto* p : params) total_params += p->numel();
    assert(total_params > 0);
    assert(model.param_count() > 0);

    PASS();
    return 0;
}

static int test_moe_trainer_param_count_consistency() {
    TEST("MoETrainer param count consistency");
    MoEModel model = create_test_moe_model();
    MoETrainer trainer(&model, nullptr);

    AdamW opt(1e-3f);
    MoETrainConfig cfg;
    trainer.compile(&opt, cfg);

    auto params = trainer.get_model_params();
    int64_t collected = 0;
    for (auto* p : params) collected += p->numel();

    int64_t stored = model.stored_param_count();
    int64_t activated = model.param_count();

    assert(collected >= activated);
    assert(stored >= activated);
    assert(activated > 0);

    PASS();
    return 0;
}

static int test_moe_trainer_mixed_precision() {
    TEST("MoETrainer mixed precision init");
    MoEModel model = create_test_moe_model();
    MoETrainer trainer(&model, nullptr);

    AdamW opt(1e-3f);
    MoETrainConfig cfg;
    cfg.mixed_precision = true;
    cfg.loss_scale = 128.0f;

    trainer.compile(&opt, cfg);
    assert(trainer.mixed_precision_active());

    trainer.mp_quantize_forward();
    trainer.mp_restore_master();

    PASS();
    return 0;
}

static int test_moe_trainer_ema() {
    TEST("MoETrainer EMA");
    MoEModel model = create_test_moe_model();
    MoETrainer trainer(&model, nullptr);

    AdamW opt(1e-3f);
    MoETrainConfig cfg;
    trainer.compile(&opt, cfg);

    trainer.ema_init(0.999f);

    for (int i = 0; i < 10; i++) {
        trainer.ema_step();
    }

    trainer.ema_apply();
    trainer.ema_swap();
    trainer.ema_apply();

    PASS();
    return 0;
}

static int test_moe_trainer_gradient_clip() {
    TEST("MoETrainer gradient clipping");
    MoEModel model = create_test_moe_model();
    MoETrainer trainer(&model, nullptr);

    AdamW opt(1e-3f);
    MoETrainConfig cfg;
    cfg.max_grad_norm = 0.5f;
    trainer.compile(&opt, cfg);

    auto params = trainer.get_model_params();
    for (auto* p : params) {
        if (p->numel() > 0) {
            p->set_grad(*p);
        }
    }

    float grad_norm = trainer.clip_gradients(0.5f);
    assert(std::isfinite(grad_norm));
    assert(grad_norm > 0 || grad_norm == 0);

    PASS();
    return 0;
}

static int test_moe_training_pipeline_construct() {
    TEST("MoETrainingPipeline construction");
    MoEModel model = create_test_moe_model();
    MoETrainingPipeline pipeline(&model, nullptr, "data/tinyshakespeare.txt");
    PASS();
    return 0;
}

static int test_moe_training_pipeline_configure() {
    TEST("MoETrainingPipeline configure");
    MoEModel model = create_test_moe_model();
    MoETrainingPipeline pipeline(&model, nullptr, "data/tinyshakespeare.txt");

    MoETrainConfig cfg;
    cfg.batch_size = 2;
    cfg.seq_length = 16;
    cfg.train_steps = 5;
    cfg.num_epochs = 1;
    cfg.gradient_accumulation_steps = 1;
    cfg.load_balance_coef = 0.01f;

    pipeline.configure(cfg);
    PASS();
    return 0;
}

static int test_moe_expert_utilization() {
    TEST("MoETrainer expert utilization");
    MoEModel model = create_test_moe_model();
    MoETrainer trainer(&model, nullptr);

    AdamW opt(1e-3f);
    MoETrainConfig cfg;
    trainer.compile(&opt, cfg);

    float util = trainer.compute_expert_utilization();
    assert(std::isfinite(util));
    assert(util >= 0);

    PASS();
    return 0;
}

static int test_moe_metrics_defaults() {
    TEST("MoEMetrics default values");
    MoEMetrics m;
    assert(m.loss == 0.0f);
    assert(m.load_balance_loss == 0.0f);
    assert(m.z_loss == 0.0f);
    assert(m.aux_loss == 0.0f);
    assert(m.total_loss == 0.0f);
    assert(m.grad_norm == 0.0f);
    assert(m.expert_utilization == 0.0f);
    assert(m.step == 0);
    assert(m.epoch == 0);
    PASS();
    return 0;
}

int main() {
    std::cout << "MoE Training Tests" << std::endl;
    std::cout << "==================" << std::endl;

    int result = 0;
    result += test_moe_trainer_construct();
    result += test_moe_trainer_compile();
    result += test_moe_trainer_forward_pass();
    result += test_moe_trainer_params_collected();
    result += test_moe_trainer_param_count_consistency();
    result += test_moe_trainer_mixed_precision();
    result += test_moe_trainer_ema();
    result += test_moe_trainer_gradient_clip();
    result += test_moe_training_pipeline_construct();
    result += test_moe_training_pipeline_configure();
    result += test_moe_expert_utilization();
    result += test_moe_metrics_defaults();

    std::cout << "\n============================================" << std::endl;
    std::cout << "Results: " << tests_passed << " / " << tests_run << " tests passed";
    if (result == 0) std::cout << " -- ALL PASSED";
    std::cout << std::endl;

    return result;
}
