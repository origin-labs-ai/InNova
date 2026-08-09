// test_model.cpp — Transformer model tests
#include "quant/model.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[Test] Running Model test..." << std::endl;
    quant::TransformerConfig cfg;
    cfg.vocab_size = 1000;
    cfg.hidden_size = 128;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.head_dim = 32;
    cfg.max_seq_len = 128;

    quant::DenseModel model(cfg);
    assert(model.param_count() > 0);
    std::cout << "  -> Model created with " << model.param_count() << " parameters." << std::endl;
    std::cout << "Model test passed!" << std::endl;
    return 0;
}
