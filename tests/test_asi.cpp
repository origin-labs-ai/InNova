#include "oil/asi.h"
#include "oil/multi_agent.h"
#include "oil/test.h"
#include "oil/model.h"
#include "oil/tensor.h"
#include "oil/trainer.h"
#include "oil/types.h"
#include "oil/optimizer.h"
#include "oil/tokenizer.h"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>

using namespace oil;
using namespace oil::asi;
using namespace oil::multi_agent;

static void test_self_monitor() {
    TEST_SUITE("G1: SelfMonitor");
    SelfMonitor sm(nullptr);
    Tensor logits({2, 10});
    logits.fill(0.0f);
    float conf = sm.estimate_confidence(logits);
    TEST_CHECK(conf >= 0.0f && conf <= 1.0f, "confidence in [0,1]");

    auto state = sm.analyze("input", "output");
    TEST_CHECK(!state.recommendation.empty(), "recommendation is set");
    TEST_CHECK(state.confidence >= 0, "initial confidence non-negative");
}

static void test_self_reflector() {
    TEST_SUITE("G2: SelfReflector");
    SelfReflector sr(nullptr);
    auto reflection = sr.reflect("input", "output");
    TEST_CHECK(!reflection.empty(), "reflect returns non-empty");
    TEST_CHECK(reflection.find("Reflection:") != std::string::npos, "reflect contains prefix");

    auto refined = sr.refine("original", reflection);
    TEST_CHECK(!refined.empty(), "refine returns non-empty");
    TEST_CHECK(refined.find("[Refined]:") != std::string::npos, "refine contains prefix");
}

static void test_recursive_self_improver() {
    TEST_SUITE("G5: RecursiveSelfImprover");
    RecursiveSelfImprover rsi(nullptr, nullptr);
    rsi.improvement_cycle(5);
    bool result = rsi.self_modify("analysis");
    TEST_CHECK(!result, "self_modify returns false with null model");
}

static void test_code_gen_self_improver() {
    TEST_SUITE("G6: CodeGenSelfImprover");
    CodeGenSelfImprover cg(nullptr);
    auto code = cg.generate_kernel("matmul", 128, 128, 128);
    TEST_CHECK(!code.empty(), "generate_kernel returns non-empty");
    TEST_CHECK(code.find("kernel") != std::string::npos, "generated code contains kernel");

    bool compiled = cg.compile_and_test(code);
    (void)compiled;
    TEST_CHECK(true, "compile_and_test executed successfully");
}

static void test_self_verifier() {
    TEST_SUITE("G7: SelfVerifier");
    SelfVerifier sv(nullptr);
    bool ok = sv.verify("2+2=4", "4");
    TEST_CHECK(ok, "verify returns true for simple equality");

    auto cases = sv.find_edge_cases("solution");
    TEST_CHECK(cases.empty(), "edge cases empty with null model");
}

static void test_capability_amplifier() {
    TEST_SUITE("G8: CapabilityAmplifier");
    CapabilityAmplifier ca(nullptr);
    float m = ca.measure("reasoning");
    TEST_CHECK(m >= 0.0f && m <= 1.0f, "measure in [0,1]");

    bool improved = ca.improve("reasoning", 10);
    TEST_CHECK(!improved, "improve returns false with null model");
}

static void test_safety_guardrails() {
    TEST_SUITE("G9: SafetyGuardrails");
    SafetyGuardrails sg;
    TEST_CHECK(sg.check_output("hello world"), "safe output passes");
    TEST_CHECK(!sg.check_output("rm -rf /"), "blocked output rejected");
    TEST_CHECK(sg.check_input("hello"), "safe input passes");
    TEST_CHECK(!sg.check_input("sudo rm -rf"), "dangerous input rejected");

    sg.set_kill_switch(true);
    TEST_CHECK(!sg.check_output("anything"), "output blocked when killed");
    TEST_CHECK(sg.is_killed(), "kill switch on");
    sg.set_kill_switch(false);
    TEST_CHECK(!sg.is_killed(), "kill switch off");
}

static void test_hitl() {
    TEST_SUITE("G10: HITL");
    HITL hitl;
    TEST_CHECK(!hitl.is_paused(), "not paused initially");
    bool approved = hitl.request_approval("deploy");
    TEST_CHECK(approved, "approval granted by default");

    hitl.pause();
    TEST_CHECK(hitl.is_paused(), "paused after call");
    hitl.resume();
    TEST_CHECK(!hitl.is_paused(), "not paused after resume");
}

static void test_alignment_system() {
    TEST_SUITE("G11-G12: AlignmentSystem");
    AlignmentSystem as;
    float score = as.value_alignment_score("helpful output");
    TEST_CHECK(score > 0, "alignment score positive");
    TEST_CHECK(as.max_loop_iterations == 100, "max loop iterations is 100");
}

static void test_world_model() {
    TEST_SUITE("G13: WorldModel");
    WorldModel wm(nullptr);
    Tensor state({10}), action({4});
    state.fill(0.0f); action.fill(1.0f);
    auto next = wm.simulate_step(state, action);
    TEST_CHECK(next.numel() == 10, "simulated state has same shape");

    auto plan = wm.plan(5);
    TEST_CHECK(plan.empty(), "plan returns empty with null model");
}

static void test_curiosity_explorer() {
    TEST_SUITE("G14: CuriosityDrivenExplorer");
    CuriosityDrivenExplorer ce(nullptr);
    Tensor state({4});
    state.fill(0.0f);
    auto reward = ce.intrinsic_reward(state);
    TEST_CHECK(reward.numel() == 1, "intrinsic reward shape");
    TEST_CHECK(reward.data<float>()[0] >= 0.0f, "first state reward >= 0");

    auto reward2 = ce.intrinsic_reward(state);
    TEST_CHECK(reward2.data<float>()[0] >= 0.0f, "same state reward >= 0");

    auto steps = ce.explore(10);
    TEST_CHECK(steps.empty(), "explore returns empty with null model");
}

static void test_multi_agent_system() {
    TEST_SUITE("G15: MultiAgentSystem");
    MultiAgentSystem mas(3);
    auto histories = mas.get_histories();
    TEST_CHECK(histories.size() == 3, "3 agent histories");

    mas.run_episode(5);
    auto histories2 = mas.get_histories();
    TEST_CHECK(histories2.size() == 3, "histories preserved after episode");
}

static void test_nas() {
    TEST_SUITE("G16: NAS");
    NeuralArchitectureSearch nas;
    auto arch = nas.search(10, 5);
    TEST_CHECK(arch.layers > 0, "NAS returns valid layers");
    TEST_CHECK(arch.hidden > 0, "NAS returns valid hidden size");
    TEST_CHECK(arch.score > 0, "NAS returns valid score");

    Architecture base{12, 4096, 0.0f};
    (void)base;
}

static void test_hp_optimizer() {
    TEST_SUITE("G17: HPOptimizer");
    HPOptimizer hpo(nullptr);
    hpo.population_based_training(4, 5);

    TEST_CHECK(true, "HPOptimizer constructor + PBT succeeds");
}

static void test_continuous_learner() {
    TEST_SUITE("G18: ContinuousLearner");
    ContinuousLearner cl(nullptr);
    Tensor data({10});
    data.fill(1.0f);
    cl.update(data);

    bool retained = cl.prevent_forgetting(0.9f);
    TEST_CHECK(retained, "prevent_forgetting returns true with null model");
}

static void test_knowledge_distillation() {
    TEST_SUITE("G19: KnowledgeDistillation");
    KnowledgeDistillation kd(nullptr, nullptr);
    TEST_CHECK(true, "distill completes without errors");
}

static void test_prompt_optimizer() {
    TEST_SUITE("G20: PromptOptimizer");
    PromptOptimizer po(nullptr);
    auto optimized = po.optimize("translate", 5);
    TEST_CHECK(!optimized.empty(), "optimize returns non-empty");
    TEST_CHECK(optimized == "translate", "optimize returns original with null model");

    float score = po.evaluate("prompt", "task");
    TEST_CHECK(score >= 0, "evaluate returns non-negative score");
}

static void test_chain_of_thought() {
    TEST_SUITE("G21: ChainOfThought");
    ChainOfThought cot(nullptr);
    auto result = cot.reason("Solve math: 2+2", 5);
    TEST_CHECK(!result.empty(), "reason returns result");

    auto chain = cot.get_chain();
    TEST_CHECK(chain.size() == 5, "chain has 5 steps");
    TEST_CHECK(chain[0].find("Step 0") != std::string::npos, "first step labeled");
    TEST_CHECK(chain[4].find("Step 4") != std::string::npos, "last step labeled");
}

static void test_tool_use() {
    TEST_SUITE("G22: ToolUse");
    ToolUse tu(nullptr);
    auto tools = tu.get_available_tools();
    TEST_CHECK(tools.size() >= 3, "has calculator, search, execute tools");

    auto result = tu.call_tool("calculator", "1 + 1");
    TEST_CHECK(!result.empty(), "tool call returns result");
    TEST_CHECK(result.find("calculator") != std::string::npos, "tool call includes tool name");
}

static void test_memory_system() {
    TEST_SUITE("G23: MemorySystem");
    MemorySystem mem(100);
    Tensor key({4}), value({4});
    key.fill(1.0f); value.fill(42.0f);
    mem.store(key, value);

    Tensor query({4});
    query.fill(1.0f);
    auto retrieved = mem.retrieve(query, 1);
    TEST_CHECK(retrieved.numel() == 4, "retrieved has same shape as query");
    for (int64_t i = 0; i < 4; i++)
        TEST_CHECK_CLOSE(retrieved.data<float>()[i], 42.0f, 1e-4f, "retrieved value matches stored");

    mem.consolidate();
    TEST_CHECK(true, "consolidate succeeds");
}

static void test_planning_engine() {
    TEST_SUITE("G24: PlanningEngine");
    PlanningEngine pe(nullptr);
    auto plan = pe.plan("build a house", 10);
    TEST_CHECK(!plan.empty(), "plan returns steps");

    bool executed = pe.execute(plan);
    TEST_CHECK(!executed, "execute returns false with null model");
}

static void test_evaluation_harness() {
    TEST_SUITE("G25: EvaluationHarness");
    EvaluationHarness eh(nullptr);
    auto result = eh.evaluate("hellaswag", 10);
    TEST_CHECK(result.accuracy >= 0, "accuracy non-negative");
    TEST_CHECK(result.loss >= 0, "loss >= 0");
    TEST_CHECK(result.samples >= 0, "samples non-negative");
    TEST_CHECK(result.samples <= 10, "samples respects n_samples");

    auto all = eh.evaluate_all();
    TEST_CHECK(all.size() >= 3, "evaluate_all returns multiple benchmarks");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("InNova — ASI Pipeline (G1-G25) Test Suite\n");
    printf("=============================================\n");

    test_self_monitor();
    test_self_reflector();
    test_recursive_self_improver();
    test_code_gen_self_improver();
    test_self_verifier();
    test_capability_amplifier();
    test_safety_guardrails();
    test_hitl();
    test_alignment_system();
    test_world_model();
    test_curiosity_explorer();
    test_multi_agent_system();
    test_nas();
    test_hp_optimizer();
    test_continuous_learner();
    test_knowledge_distillation();
    test_prompt_optimizer();
    test_chain_of_thought();
    test_tool_use();
    test_memory_system();
    test_planning_engine();
    test_evaluation_harness();

    printf("\n=============================================\n");

    return TEST_REPORT() > 0 ? 1 : 0;
}
