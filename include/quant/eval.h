#pragma once
#include "quant/tensor.h"
#include "quant/model.h"
#include <vector>
#include <string>
#include <utility>

namespace quant {

struct EvalResult {
    double perplexity = 0.0;
    double accuracy = 0.0;
    double f1_score = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double bleu = 0.0;
    double rouge_l = 0.0;
    double tokens_per_sec = 0.0;
    double memory_mb = 0.0;
    int64_t total_tokens = 0;
    int64_t correct = 0;
    int64_t total = 0;
    std::string task_name;
};

// Perplexity evaluation over tokenized text
EvalResult eval_perplexity(Model* model, const std::vector<int>& tokens,
                           int context_size = 512, int stride = 256);

// Next-token prediction accuracy
EvalResult eval_accuracy(Model* model, const std::vector<int>& tokens,
                         int context_size = 512);

// Classification eval: given logits and labels, compute accuracy + F1
EvalResult eval_classification(const Tensor& predictions,
                                const std::vector<int>& labels);

// BLEU score for generated vs reference text (n-gram overlap)
double compute_bleu(const std::vector<int>& candidate,
                    const std::vector<int>& reference, int max_n = 4);

// ROUGE-L: longest common subsequence based
double compute_rouge_l(const std::vector<int>& candidate,
                       const std::vector<int>& reference);

// Generation benchmark: tokens/sec
double eval_generation_speed(Model* model, int prompt_tokens, int gen_tokens);

// Full model evaluation suite
class ModelEvaluator {
public:
    explicit ModelEvaluator(Model* model);

    EvalResult evaluate_perplexity(const std::vector<int>& tokens,
                                    const std::string& name = "perplexity",
                                    int context = 512, int stride = 256);

    EvalResult evaluate_accuracy(const std::vector<int>& tokens,
                                  const std::string& name = "accuracy",
                                  int context = 512);

    EvalResult evaluate_generation(int prompt_len = 256, int gen_len = 128,
                                    const std::string& name = "generation");

    std::vector<EvalResult> run_all(const std::vector<int>& eval_tokens);

    void set_batch_size(int bs) { batch_size_ = bs; }
    int batch_size() const { return batch_size_; }

private:
    Model* model_;
    int batch_size_ = 1;
};

// HellaSwag-style: choose correct continuation
EvalResult eval_hellaswag(Model* model,
                           const std::vector<std::vector<int>>& contexts,
                           const std::vector<std::vector<int>>& correct_endings,
                           const std::vector<std::vector<int>>& wrong_endings);

// Load tokens from file (one int per line)
std::vector<int> load_eval_tokens(const std::string& path, int max_tokens = 0);

} // namespace quant