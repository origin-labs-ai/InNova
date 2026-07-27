#pragma once
#include "oil/tensor.h"
#include "oil/model.h"
#include "oil/meta_cognition.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace oil {
namespace asi {

// ========================================================================
// Self-Evaluation Benchmark Suite (DIFFUSION T44)
// ========================================================================

struct AccuracyResult {
    std::string benchmark_name;
    int64_t total_problems = 0;
    int64_t correct_count = 0;
    float accuracy = 0.0f;
    double runtime_ms = 0.0;
};

struct CalibrationResult {
    float expected_calibration_error = 0.0f;
    float max_calibration_error = 0.0f;
    int64_t total_predictions = 0;
    float avg_confidence = 0.0f;
    float avg_accuracy = 0.0f;
    std::vector<std::pair<float, float>> confidence_vs_accuracy;
};

struct ConsistencyResult {
    std::string problem_description;
    int64_t total_asks = 0;
    int64_t agreement_count = 0;
    float consistency_score = 0.0f;
    std::vector<std::string> responses;
};

struct RobustnessResult {
    std::string problem_description;
    float baseline_accuracy = 0.0f;
    float perturbed_accuracy = 0.0f;
    float degradation = 0.0f;
    int64_t perturbations_tested = 0;
};

struct SpeedResult {
    float ops_per_second = 0.0f;
    float p50_latency_ms = 0.0f;
    float p90_latency_ms = 0.0f;
    float p99_latency_ms = 0.0f;
    float mean_latency_ms = 0.0f;
    int64_t total_operations = 0;
    double total_time_ms = 0.0;
};

struct MemoryResult {
    float peak_ram_mb = 0.0f;
    float avg_ram_mb = 0.0f;
    int64_t cache_hits = 0;
    int64_t cache_misses = 0;
    float cache_efficiency = 0.0f;
};

struct PassAtKResult {
    int64_t k = 0;
    float pass_at_k = 0.0f;
    int64_t total_problems = 0;
    int64_t problems_with_at_least_one_correct = 0;
};

struct VersionComparison {
    std::string old_version;
    std::string new_version;
    float accuracy_delta = 0.0f;
    float calibration_delta = 0.0f;
    float speed_delta = 0.0f;
    float memory_delta = 0.0f;
    bool overall_improved = false;
};

struct EvaluationReport {
    std::string version_id;
    int64_t timestamp = 0;
    AccuracyResult accuracy;
    CalibrationResult calibration;
    std::vector<ConsistencyResult> consistency_results;
    std::vector<RobustnessResult> robustness_results;
    SpeedResult speed;
    MemoryResult memory;
    std::vector<PassAtKResult> pass_at_k_results;
    float overall_score = 0.0f;
};

class SelfEvalSuite {
public:
    SelfEvalSuite(Model* model);

    AccuracyResult run_accuracy_benchmark(const std::string& benchmark_name, int64_t n_samples = 100);
    AccuracyResult run_accuracy_benchmark_mmlu(int64_t n_samples);
    AccuracyResult run_accuracy_benchmark_arc(int64_t n_samples);
    AccuracyResult run_accuracy_benchmark_hellaswag(int64_t n_samples);
    AccuracyResult run_accuracy_benchmark_gsm8k(int64_t n_samples);

    CalibrationResult run_calibration_benchmark(int64_t n_samples = 500);
    float compute_expected_calibration_error(const std::vector<CalibrationBin>& bins);

    ConsistencyResult run_consistency_benchmark(const std::string& problem, int64_t n_asks = 10);
    std::vector<ConsistencyResult> run_consistency_suite(int64_t n_problems = 20, int64_t n_asks = 10);

    RobustnessResult run_robustness_benchmark(const std::string& problem, int64_t n_perturbations = 50);
    std::vector<RobustnessResult> run_robustness_suite(int64_t n_problems = 20);

    SpeedResult run_speed_benchmark(int64_t n_operations = 1000);

    MemoryResult run_memory_benchmark(int64_t n_allocations = 10000);

    PassAtKResult run_pass_at_k(const std::string& benchmark_name, int64_t k, int64_t n_problems = 100);
    std::vector<PassAtKResult> run_pass_at_k_suite(int64_t max_k = 10);

    EvaluationReport generate_report(const std::string& version_id);
    VersionComparison compare_versions(const EvaluationReport& old_report, const EvaluationReport& new_report);

    void store_report(const EvaluationReport& report);
    std::vector<EvaluationReport> get_report_history();
    EvaluationReport get_latest_report();

    static constexpr int64_t max_reports = 100;

private:
    Model* model_;
    std::vector<EvaluationReport> report_history_;

    float evaluate_answer(const std::string& problem, const std::string& answer);
    std::string generate_perturbation(const std::string& original);
    std::vector<std::string> generate_answer_samples(const std::string& problem, int64_t n);
    std::vector<float> compute_logits_confidence(const std::string& input);
    int64_t get_vocab_size();
};

} // namespace asi
} // namespace oil
