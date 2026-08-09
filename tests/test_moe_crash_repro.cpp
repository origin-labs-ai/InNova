// test_moe_crash_repro.cpp — MoE stability test
#include "quant/moe_model.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[Test] Running MoE Crash Repro test..." << std::endl;
    quant::TransformerConfig cfg;
    cfg.vocab_size = 100;
    cfg.hidden_size = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.max_seq_len = 64;

    quant::moe::MoEAllConfig moe_cfg;
    moe_cfg.num_experts = 4;
    moe_cfg.top_k = 2;
    moe_cfg.expert_hidden_size = 128;

    quant::MoEModel model(cfg, moe_cfg);
    assert(model.param_count() > 0);
    std::cout << "MoE Crash Repro test passed!" << std::endl;
    return 0;
}
