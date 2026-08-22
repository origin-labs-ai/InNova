// test_moe_training.cpp — MoETrainer train_step + metrics
#include "quant/moe_trainer.h"
#include "quant/moe_model.h"
#include "quant/optimizer.h"
#include "quant/tokenizer.h"
#include "quant/tensor.h"
#include "quant/test.h"
#include <vector>
#include <cmath>
#include <cstdio>

using namespace quant;

int main() {
    TEST_SUITE("moe_training");
    printf("=== MoE training test ===\n\n");

    TransformerConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = cfg.hidden_size / cfg.num_heads;
    cfg.ffn_hidden_size = 64;
    cfg.norm_eps = 1e-5f;
    cfg.max_seq_len = 32;

    moe::MoEAllConfig moe_cfg;
    moe_cfg.num_experts = 4;
    moe_cfg.top_k = 2;
    moe_cfg.expert_hidden_size = 64;

    MoEModel model(cfg, moe_cfg);
    TEST_CHECK(model.param_count() > 0, "MoE model has parameters");

    BPETokenizer tokenizer;
    MoETrainer trainer(&model, &tokenizer);

    MoETrainConfig tcfg;
    tcfg.learning_rate = 1e-3f;
    tcfg.batch_size = 2;
    tcfg.seq_length = 8;
    tcfg.aux_loss_coef = 0.01f;

    AdamW optimizer(tcfg.learning_rate);
    trainer.compile(&optimizer, tcfg);
    TEST_CHECK(trainer.get_model_params().size() > 0, "trainer collected model params");

    // Learnable periodic next-token task
    const int64_t B = 2, S = 8;
    Tensor ids(Shape{B, S}, DType::F32);
    Tensor labels(Shape{B, S}, DType::F32);
    for (int64_t i = 0; i < B * S; i++) {
        int64_t tok = (i + 1) % (cfg.vocab_size - 1);
        ids.data<float>()[i] = (float)(tok + 1);
        labels.data<float>()[i] = (float)((tok + 2) % cfg.vocab_size);
    }

    float l0 = trainer.train_step(ids, labels);
    float l1 = trainer.train_step(ids, labels);
    printf("  train_step loss: %.4f -> %.4f\n", l0, l1);
    TEST_CHECK(std::isfinite(l1), "MoE train_step loss finite");

    const MoEMetrics& m = trainer.metrics();
    TEST_CHECK(m.step > 0, "metrics step advanced");
    TEST_CHECK(std::isfinite(m.total_loss), "metrics total_loss finite");

    // Load-balance loss computed on real router output
    TEST_CHECK(m.load_balance_loss >= 0.0f, "load balance loss non-negative");
    TEST_CHECK(m.z_loss >= 0.0f, "z loss non-negative");

    int failures = TEST_REPORT();
    printf("\nMOE TRAINING TEST %s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures > 0 ? 1 : 0;
}
