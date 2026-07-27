#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <functional>

namespace oil {

struct ParquetShard {
    std::string path;
    int64_t num_rows;
    int64_t num_tokens;
};

class ParquetReader {
public:
    ParquetReader();
    ~ParquetReader();

    bool open(const std::string& path);
    bool next_batch(float* tokens, int64_t max_tokens, int64_t& actual_tokens);
    void reset();
    void close();
    int64_t total_tokens() const { return total_tokens_; }
    int64_t rows_read() const { return rows_read_; }

private:
    FILE* fp_;
    std::string buffer_;
    int64_t total_tokens_;
    int64_t rows_read_;
    int64_t file_size_;
    std::vector<float> token_buffer_;
};

class StreamingDataset {
public:
    StreamingDataset(const std::string& data_dir, int64_t vocab_size);
    ~StreamingDataset();

    void add_shard(const std::string& parquet_path);
    bool next_batch(float* input_ids, float* labels, int64_t batch_size, int64_t seq_len);
    void reset();
    void shuffle_shards();

    int64_t total_tokens() const { return total_tokens_; }
    int64_t shards_loaded() const { return shards_.size(); }
    int64_t epochs_completed() const { return epochs_; }

private:
    std::vector<ParquetShard> shards_;
    ParquetReader reader_;
    int64_t vocab_size_;
    int64_t total_tokens_;
    int64_t current_shard_;
    int64_t epochs_;
    std::vector<float> text_buffer_;
    int64_t buffer_pos_;
    int64_t buffer_len_;

    bool fill_buffer();
    void tokenize_and_fill(float* dst, int64_t needed);
};

} // namespace oil
