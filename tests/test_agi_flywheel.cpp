// test_agi_flywheel.cpp — Flywheel construction, history, log path
#include "quant/agi.h"
#include "quant/transformer.h"
#include "quant/model.h"
#include "quant/trainer.h"
#include "quant/test.h"
#include <cstdio>
#include <string>

using namespace quant;

int main() {
    TEST_SUITE("agi_flywheel");
    printf("=== AGI flywheel test ===\n\n");

    TransformerConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.ffn_hidden_size = 32;
    cfg.max_seq_len = 32;

    DenseModel model(cfg);
    Trainer trainer(&model, nullptr);
    agi::SafetyGuardrails safety;
    agi::CodeGenSelfImprover codegen(&model);
    agi::SelfVerifier verifier(&model);
    agi::CapabilityAmplifier amplifier(&model);

    // Construct flywheel with all subsystems
    agi::Flywheel flywheel(&model, &trainer, &codegen, &verifier, &amplifier, &safety);

    TEST_CHECK(flywheel.get_history().empty(), "flywheel history empty at start");
    TEST_CHECK(flywheel.get_no_improvement_count() == 0, "no improvements at start");

    std::string log_path = flywheel.get_log_path();
    TEST_CHECK(!log_path.empty(), "flywheel log path non-empty");
    printf("  flywheel log path: %s\n", log_path.c_str());

    // Subsystem verification (used inside the loop)
    TEST_CHECK(codegen.compile_and_test("int main(){return 0;}\n"),
               "codegen compiles trivial program");

    int failures = TEST_REPORT();
    printf("\nAGI FLYWHEEL TEST %s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures > 0 ? 1 : 0;
}
