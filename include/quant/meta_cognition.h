#pragma once
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/trainer.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <chrono>

namespace quant {
namespace agi {

// ========================================================================
// Meta-Cognition Engine (DIFFUSION T43)
// ========================================================================

enum class ReasoningStrategy {
    FORWARD,
    BACKWARD,
    ANALOGICAL,
    DECOMPOSITION,
    HYPOTHETICAL
};

struct ReasoningStep {
    std::string description;
    float confidence = 0.0f;
    float entropy = 0.0f;
    bool consistent = true;
    int64_t step_index = 0;
    ReasoningStrategy strategy_used = ReasoningStrategy::FORWARD;
};

struct ReasoningChain {
    std::vector<ReasoningStep> steps;
    float overall_confidence = 0.0f;
    float consistency_score = 0.0f;
    int inconsistency_count = 0;
    int64_t chain_id = 0;
    std::string goal;
};

struct ConfidenceRecord {
    float predicted_confidence = 0.0f;
    bool was_correct = false;
    std::string task_description;
    int64_t timestamp = 0;
};

struct CalibrationBin {
    float bin_lower = 0.0f;
    float bin_upper = 0.0f;
    int64_t total_count = 0;
    int64_t correct_count = 0;
    float avg_confidence = 0.0f;
};

struct SubGoal {
    std::string description;
    float estimated_difficulty = 0.0f;
    std::vector<int64_t> depends_on;
    bool completed = false;
    bool failed = false;
    std::string result;
    int64_t subgoal_id = 0;
};

struct GoalDecomposition {
    std::string original_goal;
    std::vector<SubGoal> subgoals;
    int64_t total_subgoals = 0;
    int64_t completed_subgoals = 0;
    float progress = 0.0f;
};

struct MetacognitiveInsight {
    std::string insight_text;
    float relevance_score = 0.0f;
    int64_t timestamp = 0;
    std::string source_task;
};

struct SelfAssessment {
    float accuracy_score = 0.0f;
    float speed_score = 0.0f;
    float resource_efficiency = 0.0f;
    float overall_score = 0.0f;
    int64_t tasks_evaluated = 0;
    float avg_confidence = 0.0f;
    float avg_calibration_error = 0.0f;
};

struct FeedbackEntry {
    std::string task;
    std::string strategy_used;
    float performance_score = 0.0f;
    bool succeeded = false;
    std::string failure_reason;
    int64_t duration_ms = 0;
};

class MetaCognitionEngine {
public:
    MetaCognitionEngine(Model* model);

    ReasoningChain analyze_reasoning(const std::string& input, const std::string& output);
    bool check_step_consistency(const ReasoningStep& prev, const ReasoningStep& curr);
    float calculate_chain_consistency(const ReasoningChain& chain);

    void record_confidence(float predicted, bool correct, const std::string& task);
    float get_expected_calibration_error();
    std::vector<CalibrationBin> get_calibration_bins(int num_bins = 10);
    float get_current_confidence();

    void identify_reasoning_errors(ReasoningChain& chain);
    int64_t find_first_error_step(const ReasoningChain& chain);
    std::string diagnose_error(const ReasoningChain& chain, int64_t error_step);

    GoalDecomposition decompose_goal(const std::string& goal, int64_t max_subgoals = 20);
    bool check_subgoal_completion(const GoalDecomposition& decomp);
    std::vector<int64_t> get_ready_subgoals(const GoalDecomposition& decomp);

    ReasoningStrategy select_strategy(const std::string& task, float difficulty);
    ReasoningStrategy get_best_strategy_for_task_type(const std::string& task_type);

    void record_feedback(const FeedbackEntry& entry);
    std::vector<FeedbackEntry> get_recent_feedback(int64_t count = 50);
    std::string generate_feedback_summary();
    std::vector<std::string> extract_improvement_patterns();

    SelfAssessment assess_self(int64_t lookback_count = 100);
    MetacognitiveInsight generate_insight(const std::string& task, const std::string& result);

    std::vector<ReasoningChain> get_history();
    std::vector<ConfidenceRecord> get_confidence_history();
    void reset();

    static constexpr int64_t max_history = 10000;
    static constexpr int64_t max_confidence_records = 50000;

private:
    Model* model_;
    std::vector<ReasoningChain> chain_history_;
    std::vector<ConfidenceRecord> confidence_records_;
    std::vector<FeedbackEntry> feedback_history_;
    std::vector<MetacognitiveInsight> insights_;
    int64_t next_chain_id_ = 0;
    int64_t next_subgoal_id_ = 0;

    float compute_entropy(const float* logits, int64_t vocab_size);
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
    float extract_task_type_score(const std::string& task, ReasoningStrategy strategy);
    std::string task_type_classify(const std::string& task);
};

} // namespace agi
} // namespace quant
