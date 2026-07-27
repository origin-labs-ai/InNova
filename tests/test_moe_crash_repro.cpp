#include "oil/native_oil_moe.h"
#include <cstdio>
#include "oil/test.h"

using namespace oil;

int main() {
    printf("Test 1: Just create model\n");
    native::NativeMoEConfig cfg;
    cfg.hidden_size = 128;
    cfg.num_heads = 4;
    cfg.ffn_hidden = 256;
    cfg.num_layers = 2;
    cfg.num_experts = 4;
    cfg.top_k = 2;
    cfg.vocab_size = 64;
    cfg.max_seq_len = 16;
    cfg.block_size = 32;

    native::NativeOILMoEModel model(cfg);
    printf("Model created. Params: %zu\n", model.total_params());
    printf("Test 1 PASSED\n");
    return 0;
}
