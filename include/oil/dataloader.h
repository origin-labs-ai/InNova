#pragma once
#include "oil/tensor.h"
#include "oil/tokenizer.h"
#include "oil/dataset.h"
#include <string>
#include <vector>
#include <deque>
#include <random>
#include <memory>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <atomic>

namespace oil {

struct DataLoaderConfig {
    std::string data_path;
    int64_t batch_size = 8;
    int64_t seq_length = 2048;
    int64_t shuffle_buffer = 10000;
    int64_t num_epochs = 1;
    int64_t num_workers = 1;
    int64_t seed = 42;
    bool drop_last = true;
    int64_t world_size = 1;
    int64_t rank = 0;
};

class TextPreprocessor {
public:
    TextPreprocessor();

    void set_lowercase(bool val) { lowercase_ = val; }
    void set_accent_removal(bool val) { remove_accents_ = val; }
    void set_html_stripping(bool val) { strip_html_ = val; }
    void set_url_removal(bool val) { remove_urls_ = val; }
    void set_email_removal(bool val) { remove_emails_ = val; }
    void set_whitespace_normalization(bool val) { normalize_whitespace_ = val; }
    void set_pii_removal(bool val) { remove_pii_ = val; }
    void set_deduplication(bool val) { deduplicate_ = val; }
    void set_fuzzy_dedup_threshold(double threshold) { fuzzy_threshold_ = threshold; }
    void set_min_length(int64_t min_len) { min_length_ = min_len; }
    void set_max_length(int64_t max_len) { max_length_ = max_len; }

    std::string process(const std::string& text) const;
    std::vector<std::string> process_batch(const std::vector<std::string>& texts) const;

    std::string detect_language(const std::string& text) const;

    std::vector<std::string> deduplicate(const std::vector<std::string>& texts) const;
    double fuzzy_similarity(const std::string& a, const std::string& b) const;

    std::vector<std::string> process_with_metadata(const std::vector<std::string>& texts,
                                                    std::vector<std::string>& languages,
                                                    std::vector<int64_t>& lengths) const;

private:
    bool lowercase_ = false;
    bool remove_accents_ = false;
    bool strip_html_ = false;
    bool remove_urls_ = false;
    bool remove_emails_ = false;
    bool normalize_whitespace_ = true;
    bool remove_pii_ = true;
    bool deduplicate_ = false;
    double fuzzy_threshold_ = 0.95;
    int64_t min_length_ = 0;
    int64_t max_length_ = -1;

    std::string do_lowercase(const std::string& text) const;
    std::string do_remove_accents(const std::string& text) const;
    std::string do_strip_html(const std::string& text) const;
    std::string do_remove_urls(const std::string& text) const;
    std::string do_remove_emails(const std::string& text) const;
    std::string do_normalize_whitespace(const std::string& text) const;
    std::string do_remove_pii(const std::string& text) const;
    std::string filter_by_length(const std::string& text) const;

    std::string strip_html_regex(const std::string& text) const;
    std::string remove_urls_regex(const std::string& text) const;
    std::string remove_emails_regex(const std::string& text) const;
    std::string remove_pii_regex(const std::string& text) const;
};

class DataPipeline {
public:
    using StageFn = std::function<std::vector<Tensor>(const std::vector<std::string>&)>;

    DataPipeline();

    void add_stage(const std::string& name, StageFn fn);
    void set_preprocessor(std::shared_ptr<TextPreprocessor> preprocessor);
    void set_tokenizer(Tokenizer* tokenizer);
    void set_batch_size(int64_t batch_size) { batch_size_ = batch_size; }
    void set_seq_length(int64_t seq_length) { seq_length_ = seq_length; }
    void set_num_workers(int64_t n) { num_workers_ = n; }

    std::vector<Tensor> process_texts(const std::vector<std::string>& texts);
    std::vector<Tensor> process_file(const std::string& path);

    std::vector<std::string> run_stage1(const std::vector<std::string>& raw_texts);
    std::vector<std::vector<int>> run_stage2(const std::vector<std::string>& preprocessed);
    std::vector<Tensor> run_stage3(const std::vector<std::vector<int>>& tokenized);

    void clear_pipeline();

    int64_t total_processed() const { return total_processed_; }

private:
    std::shared_ptr<TextPreprocessor> preprocessor_;
    Tokenizer* tokenizer_ = nullptr;
    int64_t batch_size_ = 32;
    int64_t seq_length_ = 2048;
    int64_t num_workers_ = 1;
    int64_t total_processed_ = 0;
    std::vector<std::pair<std::string, StageFn>> custom_stages_;
};

class DataLoader {
public:
    explicit DataLoader(const DataLoaderConfig& cfg, BPETokenizer* tokenizer = nullptr);

    Tensor next_batch();
    std::pair<Tensor, Tensor> next_batch_with_labels();

    bool has_next() const;
    int64_t num_batches() const;
    void reset();

    DataLoaderConfig& config() { return config_; }

private:
    DataLoaderConfig config_;
    BPETokenizer* tokenizer_;
    std::unique_ptr<StreamingDataset> dataset_;
    int64_t batch_index_;
    std::mt19937 rng_;
    std::deque<int64_t> shuffle_buffer_;

    void shuffle_and_tokenize(const std::string& text,
                               std::vector<int64_t>& tokens);
};

} // namespace oil
