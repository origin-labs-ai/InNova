#pragma once
#include "oil/tensor.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_map>

namespace oil {

// ============================================================
// Accuracy — Top-1, Top-5, Top-10 classification accuracy
// ============================================================
class Accuracy {
public:
    explicit Accuracy(int top_k = 1);
    double update(const Tensor& predictions, const Tensor& targets);
    void reset();
    int64_t correct() const { return correct_; }
    int64_t total() const { return total_; }
    double value() const;
    int top_k() const { return top_k_; }
private:
    int top_k_;
    int64_t correct_ = 0, total_ = 0;
    bool topk_match(const float* scores, int n_classes, int target) const;
};

// ============================================================
// Perplexity — exp(cross_entropy) for language models
// ============================================================
class Perplexity {
public:
    Perplexity() = default;
    double update(const Tensor& logits, const Tensor& targets);
    void reset();
    double value() const;
    double nll() const { return nll_; }
    int64_t count() const { return count_; }
private:
    double nll_ = 0.0;
    int64_t count_ = 0;
};

// ============================================================
// F1Score — Precision/Recall/F1 for classification
// ============================================================
class F1Score {
public:
    explicit F1Score(int num_classes = 2);
    void update(const Tensor& predictions, const Tensor& targets);
    void reset();
    double precision(int class_idx = -1) const;
    double recall(int class_idx = -1) const;
    double f1(int class_idx = -1) const;
    double macro_f1() const;
    double weighted_f1() const;
    int num_classes() const { return num_classes_; }
private:
    int num_classes_;
    std::vector<int64_t> tp_, fp_, fn_;
    int64_t total_samples_ = 0;
};

// ============================================================
// BLEUScore — N-gram precision with brevity penalty
// ============================================================
class BLEUScore {
public:
    explicit BLEUScore(int max_n = 4);
    double compute(const std::vector<int>& candidate,
                   const std::vector<int>& reference);
    double compute(const Tensor& candidate, const Tensor& reference);
    int max_n() const { return max_n_; }
private:
    int max_n_;
    static std::vector<int> get_ngrams(const std::vector<int>& seq, int n);
    static std::unordered_map<int, int> count_ngrams(
        const std::vector<int>& seq, int n);
};

// ============================================================
// ROUGELScore — Longest common subsequence based
// ============================================================
class ROUGELScore {
public:
    double compute(const std::vector<int>& candidate,
                   const std::vector<int>& reference);
    double compute(const Tensor& candidate, const Tensor& reference);
};

// ============================================================
// MeanAveragePrecision (mAP) — For ranking tasks
// ============================================================
class MeanAveragePrecision {
public:
    MeanAveragePrecision() = default;
    double compute(const Tensor& scores, const Tensor& labels);
    void reset();
    double value() const;
private:
    std::vector<double> ap_scores_;
};

// ============================================================
// ConfusionMatrix — Classification confusion matrix
// ============================================================
class ConfusionMatrix {
public:
    explicit ConfusionMatrix(int num_classes);
    void update(const Tensor& predictions, const Tensor& targets);
    void update(const Tensor& logits, const Tensor& targets, bool from_logits);
    void reset();
    int64_t at(int pred_class, int true_class) const;
    int64_t true_positive(int class_idx) const;
    int64_t false_positive(int class_idx) const;
    int64_t false_negative(int class_idx) const;
    int64_t true_negative(int class_idx) const;
    int total_samples() const { return static_cast<int>(total_); }
    int num_classes() const { return num_classes_; }
    const std::vector<int64_t>& data() const { return matrix_; }
private:
    int num_classes_;
    std::vector<int64_t> matrix_;
    int64_t total_ = 0;
};

} // namespace oil
