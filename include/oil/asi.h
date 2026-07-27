#pragma once
#include "oil/tensor.h"
#include "oil/model.h"
#include "oil/trainer.h"
#include "oil/reward.h"
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <chrono>
#include <fstream>

namespace oil {
namespace asi {

// ========================================================================
// G1-G5: Meta-cognition and RSI
// ========================================================================

struct MetaCognitionState {
    float confidence = 0;
    float uncertainty = 0;
    int reasoning_depth = 0;
    bool needs_reanalysis = false;
    std::string recommendation;
    std::vector<std::string> reasoning_chain;
    std::vector<float> token_confidences;
};

class SelfMonitor {
public:
    SelfMonitor(Model* model);
    MetaCognitionState analyze(const std::string& input, const std::string& output);
    float estimate_confidence(const Tensor& logits);
private:
    Model* model_;
};

class SelfReflector {
public:
    SelfReflector(Model* model);
    std::string reflect(const std::string& input, const std::string& output);
    std::string refine(const std::string& original, const std::string& reflection);
private:
    Model* model_;
};

class RecursiveSelfImprover {
public:
    RecursiveSelfImprover(Model* model, Trainer* trainer);
    void improvement_cycle(int iterations = 10);
    bool self_modify(const std::string& analysis);
private:
    Model* model_;
    Trainer* trainer_;
};

// G6: Code generation self-improvement
class CodeGenSelfImprover {
public:
    CodeGenSelfImprover(Model* model);
    std::string generate_kernel(const std::string& op, int64_t M, int64_t N, int64_t K);
    bool compile_and_test(const std::string& code);
    bool replace_kernel(const std::string& op, const std::string& new_code);
private:
    Model* model_;
};

// G7: Self-verification
class SelfVerifier {
public:
    SelfVerifier(Model* model);
    bool verify(const std::string& problem, const std::string& solution);
    std::vector<std::string> find_edge_cases(const std::string& solution);
private:
    Model* model_;
};

// G8: Capability amplification
class CapabilityAmplifier {
public:
    CapabilityAmplifier(Model* model);
    float measure(const std::string& capability);
    bool improve(const std::string& capability, int steps = 100);
private:
    Model* model_;
};

// G9: Safety guardrails
class SafetyGuardrails {
public:
    SafetyGuardrails();
    bool check_output(const std::string& output);
    bool check_input(const std::string& input);
    void set_kill_switch(bool kill) { kill_switch_ = kill; }
    bool is_killed() const { return kill_switch_; }
private:
    bool kill_switch_ = false;
    std::vector<std::string> blocked_patterns_;
};

// G10: Human-in-the-loop
class HITL {
public:
    HITL();
    bool request_approval(const std::string& action);
    void pause() { paused_ = true; }
    void resume() { paused_ = false; }
    bool is_paused() const { return paused_; }
private:
    bool paused_ = false;
};

// G11-G12: Alignment
class AlignmentSystem {
public:
    AlignmentSystem();
    float value_alignment_score(const std::string& output);
    static constexpr int max_loop_iterations = 100;
};

// G13: World model
class WorldModel {
public:
    WorldModel(Model* model);
    Tensor simulate_step(const Tensor& state, const Tensor& action);
    std::vector<Tensor> plan(int64_t horizon);
private:
    Model* model_;
};

// G14: Curiosity exploration
class CuriosityDrivenExplorer {
public:
    CuriosityDrivenExplorer(Model* model);
    Tensor intrinsic_reward(const Tensor& state);
    std::vector<int> explore(int64_t n_steps);
private:
    Model* model_;
    std::vector<Tensor> visited_states_;
};

// G15: Multi-agent system
class MultiAgentSystem {
public:
    MultiAgentSystem(int n_agents);
    void run_episode(int steps);
    std::vector<std::string> get_histories() const;
private:
    struct Agent { std::vector<std::string> history; std::string state; };
    std::vector<Agent> agents_;
    void communicate(int from, int to);
};

// G16: NAS — Neural Architecture Search
class NeuralArchitectureSearch {
public:
    NeuralArchitectureSearch();
    struct Architecture { int layers; int hidden; float score; };
    Architecture search(int population = 50, int generations = 20);
private:
    Architecture mutate(const Architecture& arch);
    float evaluate(const Architecture& arch);
};

// G17: Hyperparameter auto-tuning
class HPOptimizer {
public:
    HPOptimizer(Trainer* trainer);
    void population_based_training(int n_population = 8, int n_generations = 10);
private:
    Trainer* trainer_;
};

// G18: Continuous learning
class ContinuousLearner {
public:
    ContinuousLearner(Model* model);
    void update(const Tensor& new_data);
    bool prevent_forgetting(float threshold = 0.95f);
private:
    Model* model_;
    std::vector<Tensor> exemplars_;
};

// G19: Knowledge distillation
class KnowledgeDistillation {
public:
    KnowledgeDistillation(Model* teacher, Model* student);
    void distill(const DataLoader& data, int steps);
private:
    Model* teacher_, *student_;
};

// G20: Auto prompt engineering
class PromptOptimizer {
public:
    PromptOptimizer(Model* model);
    std::string optimize(const std::string& task, int n_iterations = 20);
    float evaluate(const std::string& prompt, const std::string& task);
private:
    Model* model_;
};

// G21: Reasoning chains (CoT)
class ChainOfThought {
public:
    ChainOfThought(Model* model);
    std::string reason(const std::string& problem, int max_steps = 10);
    std::vector<std::string> get_chain() const { return chain_; }
private:
    Model* model_;
    std::vector<std::string> chain_;
};

// G22: Tool use
class ToolUse {
public:
    ToolUse(Model* model);
    struct Tool { std::string name; std::string description; };
    std::string call_tool(const std::string& tool_name, const std::string& args);
    std::vector<std::string> get_available_tools() const;
private:
    Model* model_;
    std::vector<Tool> tools_;
};

// G23: Memory system
class MemorySystem {
public:
    MemorySystem(int64_t capacity = 10000);
    void store(const Tensor& key, const Tensor& value);
    Tensor retrieve(const Tensor& query, int k = 5);
    void consolidate();
private:
    std::vector<std::pair<Tensor, Tensor>> memory_;
    int64_t capacity_;
};

// G24: Planning engine
class PlanningEngine {
public:
    PlanningEngine(Model* model);
    struct PlanStep { std::string action; std::vector<std::string> dependencies; };
    std::vector<PlanStep> plan(const std::string& goal, int max_steps = 20);
    bool execute(const std::vector<PlanStep>& plan);
private:
    Model* model_;
};

// G25: Evaluation harness
class EvaluationHarness {
public:
    EvaluationHarness(Model* model);
    struct Result { float accuracy; float loss; int samples; };
    Result evaluate(const std::string& benchmark_name, int n_samples = 100);
    std::vector<Result> evaluate_all();
private:
    Model* model_;
};

// ========================================================================
// Self-Improving Flywheel + ASI Sandbox
// ========================================================================

struct SandboxResult {
    bool compiled = false;
    bool passed = false;
    float score = 0.0f;
    std::string stdout_capture;
    std::string stderr_capture;
    double runtime_ms = 0.0;
    int exit_code = -1;
};

struct FlywheelIteration {
    int iter = 0;
    std::string task;
    std::string solution;
    bool compiled = false;
    bool verified = false;
    float delta = 0.0f;
    bool applied = false;
    bool rolled_back = false;
    std::string file;
    int line = 0;
    std::string proof;
    double runtime_ms = 0.0;
};

class Flywheel {
public:
    Flywheel(Model* model, Trainer* trainer, CodeGenSelfImprover* codegen = nullptr,
             SelfVerifier* verifier = nullptr, CapabilityAmplifier* amplifier = nullptr,
             SafetyGuardrails* safety = nullptr);

    void run(int max_iters = 100);
    const std::vector<FlywheelIteration>& get_history() const { return history_; }
    int get_no_improvement_count() const { return no_improvement_count_; }
    std::string get_log_path() const;

private:
    std::string self_play();
    SandboxResult sandbox_compile_and_test(const std::string& code, const std::string& task);
    float measure_improvement(const std::string& task, const std::string& solution);
    bool apply_improvement(const std::string& original, const std::string& improved, const std::string& target_file);
    bool rollback(const std::string& file_path, const std::string& backup_path);
    void log_iteration(const FlywheelIteration& iter);
    std::string generate_fallback_solution(const std::string& task);
    std::string generate_test_program(const std::string& code, const std::string& task);
    std::string extract_proof(const std::string& solution);
    bool run_with_timeout(const std::string& binary, double timeout_sec, std::string& stdout_out, std::string& stderr_out, int& exit_code);
    std::string make_diff(const std::string& original, const std::string& improved);
    bool check_convergence();
    std::string sandbox_path() const;
    int count_tests_passed(const std::string& stdout_str);
    std::string generate_benchmark_harness(const std::string& code, const std::string& task, int n_iterations = 1000);
    int calculate_cyclomatic_complexity(const std::string& code);
    int measure_nesting_depth(const std::string& code);
    float estimate_code_quality(const std::string& code);
    bool sandbox_benchmark(const std::string& code, const std::string& task, double& ops_per_sec, double& avg_latency_ms);
    std::string generate_multi_file_test(const std::vector<std::pair<std::string, std::string>>& files, const std::string& task);

    Model* model_;
    Trainer* trainer_;
    CodeGenSelfImprover* codegen_;
    SelfVerifier* verifier_;
    CapabilityAmplifier* amplifier_;
    SafetyGuardrails* safety_;
    int no_improvement_count_ = 0;
    int converged_count_ = 0;
    std::vector<FlywheelIteration> history_;
};

// G26: RLHF integration — wire the RLHF pipeline into the ASI flywheel
struct RLHFIntegration {
    static void run_with_flywheel(Model* model, Model* ref_model,
                                   Tokenizer* tokenizer, Trainer* trainer,
                                   RewardModel* reward_model,
                                   Optimizer* policy_opt, Optimizer* rm_opt,
                                   int n_rounds = 3, int n_prompts = 50);
};

} // namespace asi
} // namespace oil
