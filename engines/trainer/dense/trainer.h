#pragma once
#include <string>
#include <vector>
#include <functional>
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/optimizer.h"
#include "quant/tokenizer.h"
#include "quant/autograd.h"

namespace quant {
namespace dense {

struct TrainConfig {
    int64_t batch_size = 8;
    int64_t seq_length = 512;
    int num_epochs = 3;
    float learning_rate = 3e-4f;
    float weight_decay = 1e-2f;
    int warmup_steps = 100;
    int log_interval = 10;
    int save_interval = 1000;
    float grad_clip = 1.0f;
    std::string output_path = "model.quant";
};

struct TrainMetrics {
    float loss = 0;
    float perplexity = 0;
    float learning_rate = 0;
    float grad_norm = 0;
    int tokens_per_sec = 0;
    int step = 0;
    float epoch_progress = 0;
};

class DataLoader {
public:
    DataLoader(Tokenizer* tokenizer, const std::string& data_path,
               int64_t batch_size = 8, int64_t seq_length = 512);
    bool next_batch(Tensor& input_ids, Tensor& labels);
    void shuffle();
    void reset();
    int64_t num_batches() const;
private:
    Tokenizer* tokenizer_;
    std::vector<int> data_;
    int64_t batch_size_, seq_length_;
    int64_t pos_ = 0;
    int64_t num_batches_ = 0;
};

class DenseTrainer {
public:
    DenseTrainer(DenseModel* model, Tokenizer* tokenizer);
    void compile(const TrainConfig& cfg);
    void fit(DataLoader& loader);
    float train_step(const Tensor& input_ids, const Tensor& labels);
    void zero_grad();
    void clip_gradients(float max_norm);
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    const TrainMetrics& metrics() const;
    using LogCallback = std::function<void(const TrainMetrics&)>;
    void set_log_callback(LogCallback cb);
private:
    DenseModel* model_;
    Tokenizer* tokenizer_;
    TrainConfig config_;
    TrainMetrics metrics_;
    AdamW optimizer_;
    int step_ = 0;
    LogCallback log_cb_;
    std::vector<Tensor*> get_parameters();
};

} // namespace dense
} // namespace quant
