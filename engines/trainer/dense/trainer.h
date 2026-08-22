#pragma once
#include <string>
#include <vector>
#include <functional>
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/optimizer.h"
#include "quant/tokenizer.h"
#include "quant/autograd.h"
#include "quant/qat.h"
#include "quant/continual_engine.h"

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
    // QAT
    bool use_qat = false;
    int qat_bits = 8;
    bool qat_symmetric = true;
    bool qat_use_lsq = false;
    float qat_init_scale = 0.1f;
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
    // QAT hooks
    void enable_qat(int bits = 8, bool symmetric = true, bool use_lsq = false, float init_scale = 0.1f);
    void disable_qat();
    bool qat_enabled() const { return qat_enabled_; }
    Tensor qat_fake_quantize(const Tensor& w);
    Tensor qat_fake_quantize_lsq(const Tensor& w, Tensor& scale_param);
    // Continual
    void enable_continual(const ContinualEngineConfig& cfg = {});
    void disable_continual();
    bool continual_enabled() const { return continual_enabled_; }
    ContinualEngine* continual_engine() { return continual_engine_.get(); }
    void on_task_boundary(uint32_t new_task_id, const float* eval_inputs=nullptr, const float* eval_targets=nullptr, size_t eval_count=0);
private:
    DenseModel* model_;
    Tokenizer* tokenizer_;
    TrainConfig config_;
    TrainMetrics metrics_;
    AdamW optimizer_;
    int step_ = 0;
    LogCallback log_cb_;
    std::vector<Tensor*> get_parameters();
    // QAT state
    bool qat_enabled_ = false;
    int qat_bits_ = 8;
    bool qat_symmetric_ = true;
    bool qat_use_lsq_ = false;
    float qat_init_scale_ = 0.1f;
    std::vector<Tensor> qat_scales_;
    // Continual
    bool continual_enabled_ = false;
    std::unique_ptr<ContinualEngine> continual_engine_;
    ContinualEngineConfig continual_cfg_;
    float ewc_lambda_ = 1.0f;
    float replay_ratio_ = 0.25f;
    uint32_t current_task_id_ = 0;
};

} // namespace dense
} // namespace quant
