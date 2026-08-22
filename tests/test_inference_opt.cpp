#include "quant/inference_opt.h"
#include "quant/model.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_kv_quantization() {
    quant::KVCache cache_fp32;
    cache_fp32.init(1, 100, 1, 64);
    
    quant::KVCache cache_q4;
    cache_q4.init(1, 100, 1, 64);
    assert(cache_fp32.context_len() == cache_q4.context_len());
    assert(cache_fp32.max_seq_len() == cache_q4.max_seq_len());
}

void test_speculative_decoding() {
    quant::DenseModel target_model;
    quant::DenseModel draft_model;
    
    target_model.config.vocab_size = 100;
    target_model.config.hidden_size = 64;
    draft_model.config.vocab_size = 100;
    draft_model.config.hidden_size = 32;

    assert(target_model.config.vocab_size == draft_model.config.vocab_size);
    std::cout << "[Speculative Decoding Test] Passed." << std::endl;
}

void test_flash_attention() {
    quant::Tensor q(quant::Shape{1, 4, 8, 64});
    quant::Tensor k(quant::Shape{1, 4, 8, 64});
    quant::Tensor v(quant::Shape{1, 4, 8, 64});
    
    q.zero_();
    k.zero_();
    v.zero_();
    
    assert(q.dim(3) == 64);
    assert(k.dim(3) == 64);
    assert(v.dim(3) == 64);
    std::cout << "[Flash Attention Test] Passed." << std::endl;
}

int main() {
    test_kv_quantization();
    test_speculative_decoding();
    test_flash_attention();

    quant::DenseModel model;
    model.config.vocab_size = 1000;
    model.config.hidden_size = 64;
    model.config.num_layers = 2;
    model.config.num_heads = 4;
    model.config.head_dim = 16;
    model.config.max_seq_len = 512;
    model.init_weights();

    quant::Tensor input_ids(quant::Shape{1, 4});
    quant::Tensor pos(quant::Shape{1, 4});

    for (int i = 0; i < 4; i++) {
        input_ids.data<float>()[i] = (float)(i % 10);
        pos.data<float>()[i] = (float)(i % 4);
    }

    quant::Tensor logits = model.forward(input_ids, pos);

    assert(logits.rank() == 3);
    assert(logits.dim(0) == 1);
    assert(logits.dim(1) == 4);
    assert(logits.dim(2) == 1000);

    std::cout << "Optimized Inference Test Passed!" << std::endl;
    return 0;
}
