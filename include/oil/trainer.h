#pragma once
#include "oil/types.h"
#include "oil/tensor.h"
#include "oil/model.h"
#include "oil/optimizer.h"
#include "oil/tokenizer.h"
#include "oil/autograd.h"
#include "oil/codebook.h"
#include "oil/random.h"
#include "oil/reward.h"
#include <vector>
#include <string>
#include <functional>
#include <fstream>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>

namespace oil {

struct TrainConfig {
    int64_t batch_size = 8;
    int64_t seq_length = 512;
    int num_epochs = 3;
    int train_steps = 10000;
    float learning_rate = 3e-4f;
    float weight_decay = 1e-2f;
    int warmup_steps = 100;
    int log_interval = 10;
    int save_interval = 1000;
    int val_interval = 500;
    std::string output_path = "model.oil";
    bool use_ste = true;
    float max_grad_norm = 1.0f;
    int gradient_accumulation_steps = 1;
    AdamW::Schedule schedule = AdamW::Schedule::WARMUP_COSINE;
    bool mixed_precision = false;
    float loss_scale = 128.0f;
    int loss_scale_interval = 2000;
    // C17: Label smoothing
    float label_smoothing = 0.0f;
    // C19: R-Drop consistency
    bool use_rdrop = false;
    float rdrop_alpha = 1.0f;
    // C8: Data augmentation
    bool data_augmentation = false;
    float aug_noise_std = 0.01f;
    float aug_mask_prob = 0.0f;
    // C9: Curriculum learning
    bool curriculum = false;
    int curriculum_epochs = 3;
    // Gradient noise injection (Neelakantan et al. ICLR 2016)
    float grad_noise_eta = 0.0f;
    float grad_noise_gamma = 0.55f;
};

struct AugmentConfig {
    bool enabled = false;
    float mask_prob = 0.0f;
    float noise_std = 0.01f;
    float replace_prob = 0.0f;
};

class DataLoader {
public:
    DataLoader(Tokenizer* tokenizer, const std::string& data_path,
               int64_t batch_size, int64_t seq_length,
               bool stream_from_disk = false);
    DataLoader(Tokenizer* tokenizer, const std::string& data_path,
               int64_t batch_size, int64_t seq_length,
               bool stream_from_disk, int num_workers,
               int64_t prefetch_capacity, bool use_mmap);
    ~DataLoader();
    
    bool next_batch(Tensor& input_ids, Tensor& labels);
    void shuffle(int epoch = 0);
    void reset();
    int64_t num_batches() const;
    int64_t batch_size() const { return batch_size_; }
    int64_t seq_length() const { return seq_length_; }
    
    // C8: Data augmentation
    void set_augmentation(const AugmentConfig& cfg) { aug_cfg_ = cfg; }
    void apply_augmentation(Tensor& input_ids, Tensor& labels);
    
    // C9: Curriculum learning
    struct CurriculumState {
        int current_epoch = 0;
        int total_epochs = 3;
        float seq_len_multiplier = 0.5f;
        int64_t effective_seq_length() const {
            float t = (float)current_epoch / (float)std::max(total_epochs, 1);
            int64_t min_len = 64;
            return std::max(min_len, (int64_t)(512 * (min_len / 512.0f + t * (1.0f - min_len / 512.0f))));
        }
    };
    void set_curriculum(int total_epochs) { cur_state_.total_epochs = total_epochs; }
    void curriculum_step(int epoch) { cur_state_.current_epoch = epoch; }
    
private:
    Tokenizer* tokenizer_;
    std::vector<int> tokenized_data_;
    int64_t batch_size_;
    int64_t seq_length_;
    int64_t current_pos_ = 0;
    int64_t num_batches_ = 0;
    bool streaming_ = false;
    std::string data_path_;
    std::ifstream file_stream_;
    std::vector<int> stream_chunk_;
    int64_t stream_file_offset_ = 0;
    static constexpr int64_t STREAM_CHUNK_TOKENS = 1024 * 1024;
    
    void tokenize_chunk();
    
    // C3: Memory-mapped I/O
    bool use_mmap_ = false;
    void* mmap_ptr_ = nullptr;
    size_t mmap_size_ = 0;
#ifdef _WIN32
    void* mmap_handle_ = nullptr;
    void* file_handle_ = nullptr;
#else
    int mmap_fd_ = -1;
#endif
    void open_mmap(const std::string& path);
    void close_mmap();
    
    // C2/C16: Multi-worker prefetch
    int num_workers_ = 0;
    int64_t prefetch_capacity_ = 0;
    std::thread prefetch_thread_;
    std::mutex prefetch_mutex_;
    std::queue<std::pair<Tensor, Tensor>> prefetch_queue_;
    std::atomic<bool> prefetch_running_{false};
    void prefetch_worker();
    void start_prefetch();
    void stop_prefetch();
    
    // C8: Data augmentation
    AugmentConfig aug_cfg_;
    
    // C9: Curriculum learning
    CurriculumState cur_state_;
};

// Standalone Exponential Moving Average of model parameters (Task 122 / EMA R-Drop).
class ModelEMA {
public:
    ModelEMA(const std::vector<Tensor*>& params, float decay = 0.999f)
        : params_(params), decay_(decay) {
        for (auto* p : params_) shadow_.push_back(p->clone());
    }

    void update() {
        for (size_t i = 0; i < params_.size(); i++) {
            float* e = shadow_[i].data<float>();
            const float* p = params_[i]->data<float>();
            int64_t n = params_[i]->numel();
            for (int64_t j = 0; j < n; j++)
                e[j] = decay_ * e[j] + (1.0f - decay_) * p[j];
        }
    }

    void apply() const {
        for (size_t i = 0; i < params_.size(); i++)
            std::memcpy(params_[i]->data<float>(), shadow_[i].data<float>(),
                        params_[i]->numel() * sizeof(float));
    }

    void swap() {
        for (size_t i = 0; i < params_.size(); i++) {
            float* p = params_[i]->data<float>();
            float* e = shadow_[i].data<float>();
            int64_t n = params_[i]->numel();
            for (int64_t j = 0; j < n; j++) std::swap(p[j], e[j]);
        }
    }

    float decay() const { return decay_; }
    void set_decay(float d) { decay_ = d; }
    const std::vector<Tensor>& shadow() const { return shadow_; }

private:
    std::vector<Tensor*> params_;
    std::vector<Tensor> shadow_;
    float decay_;
};

// Streaming DataLoader (Task 118): yields batches from a file by tokenizing
// fixed-size chunks on demand, never loading the whole file into memory.
class StreamingDataLoader {
public:
    StreamingDataLoader(Tokenizer* tokenizer, const std::string& data_path,
                       int64_t batch_size, int64_t seq_length);
    bool next_batch(Tensor& input_ids, Tensor& labels);
    void reset();
    int64_t num_batches() const { return num_batches_; }
    int64_t batch_size() const { return batch_size_; }
    int64_t seq_length() const { return seq_length_; }

private:
    Tokenizer* tokenizer_;
    int64_t batch_size_;
    int64_t seq_length_;
    std::ifstream file_;
    std::vector<int> buffer_;      // rolling token buffer
    int64_t num_batches_ = 0;
    int64_t chunk_bytes_ = 1 << 20;  // 1 MiB read per chunk
    bool eof_ = false;
    void fill_buffer();
};

struct TrainMetrics {
    float loss = 0;
    float perplexity = 0;
    float val_loss = 0;
    float val_perplexity = 0;
    float learning_rate = 0;
    float grad_norm = 0;
    int tokens_per_sec = 0;
    int step = 0;
    int epoch = 0;
};

class Trainer {
public:
    Trainer(Model* model, Tokenizer* tokenizer);
    
    void compile(AdamW* optimizer, const TrainConfig& cfg = TrainConfig{});
    void compile(Adafactor* optimizer, const TrainConfig& cfg = TrainConfig{});
    void fit(DataLoader& train_dl, const TrainConfig& cfg,
             DataLoader* val_dl = nullptr);
    
    float train_step(const Tensor& input_ids, const Tensor& labels);
    float micro_step(const Tensor& input_ids, const Tensor& labels, float loss_scale = 1.0f);
    float eval_loss(DataLoader& val_dl, int64_t max_batches = 20);
    float clip_gradients(float max_norm);
    void unscale_gradients(float scale);
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    
    // Callbacks
    using LogCallback = std::function<void(const TrainMetrics&)>;
    using EpochCallback = std::function<void(int epoch, const TrainMetrics&)>;
    using StepCallback = std::function<void(int step, const TrainMetrics&)>;
    void set_log_callback(LogCallback cb);
    void set_epoch_callback(EpochCallback cb);
    void set_step_callback(StepCallback cb);
    
    const TrainMetrics& metrics() const;
    const std::vector<Tensor*>& get_model_params() const { return model_params_; }
    std::vector<Codebook*> get_codebooks() const { return codebooks_; }
    void set_codebooks(const std::vector<Codebook*>& cbs) { codebooks_ = cbs; }
    
    // C20: EMA tracking
    void ema_init(float decay = 0.999f);
    void ema_step();
    void ema_apply();
    void ema_swap();
    
    // C14: Enhanced mixed precision
    void dynamic_loss_scale(float grad_norm, float max_grad_norm);

    // Task 121: master FP32 weights + FP16 forward
    void init_mixed_precision();
    void mp_quantize_forward();   // snapshot FP32 master, quantize model weights to FP16
    void mp_restore_master();     // restore FP32 master weights for optimizer step
    bool mixed_precision_active() const { return mp_active_; }
    
    // C17: Label smoothing cross-entropy
    Tensor label_smoothing_loss(const Tensor& logits, const Tensor& labels, float smoothing);
    // C19: R-Drop consistency loss
    float rdrop_loss(const Tensor& input_ids, const Tensor& labels, float alpha);
    // Gradient noise injection (Neelakantan et al. ICLR 2016)
    void inject_gradient_noise(int step);
    
private:
    Model* model_;
    Tokenizer* tokenizer_;
    Optimizer* optimizer_ = nullptr;
    TrainMetrics metrics_;
    LogCallback log_cb_;
    EpochCallback epoch_cb_;
    StepCallback step_cb_;
    std::vector<Tensor*> model_params_;
    std::vector<Codebook*> codebooks_;
    
    int step_ = 0;
    float loss_scale_ = 1.0f;
    int loss_scale_interval_ = 2000;
    int steps_since_scale_update_ = 0;
    
    // C20: EMA state
    std::vector<Tensor> ema_params_;
    float ema_decay_ = 0.999f;
    bool ema_enabled_ = false;

    // Task 121: master FP32 weight copies for mixed precision
    std::vector<Tensor> mp_master_;
    bool mp_active_ = false;

    float grad_noise_eta_ = 0.0f;
    float grad_noise_gamma_ = 0.55f;
    RNG grad_noise_rng_;
    float label_smoothing_ = 0.0f;
};

// ===========================================================================
// PPO Trainer — Proximal Policy Optimization for RLHF
// ===========================================================================
class PPOTrainer {
public:
    PPOTrainer(Model* policy, Model* ref_model, float clip_epsilon = 0.2f,
               float value_coef = 0.5f, float entropy_coef = 0.01f,
               float gamma = 0.99f, float gae_lambda = 0.95f,
               float kl_target = 0.02f);

    void train_step(const Tensor& states, const Tensor& actions,
                    const Tensor& old_logprobs, const Tensor& advantages,
                    const Tensor& returns);

    Tensor compute_gae(const Tensor& rewards, const Tensor& values,
                       const Tensor& dones);
    float compute_kl_divergence(const Tensor& logits_a, const Tensor& logits_b);

    Tensor compute_log_probs(const Tensor& logits, const Tensor& ids);

    std::vector<Tensor*> critic_parameters();
    void set_log_callback(std::function<void(float pl, float vl, float ent, float kl)> cb) {
        log_cb_ = cb;
    }
    float kl_alpha() const { return kl_alpha_; }
    float last_kl() const { return last_kl_; }

private:
    Model* policy_;
    Model* ref_model_;
    float clip_epsilon_;
    float value_coef_;
    float entropy_coef_;
    float gamma_;
    float gae_lambda_;
    float kl_target_;
    float kl_alpha_;
    float last_kl_ = 0.0f;

    Tensor v_fc1_weight_, v_fc1_bias_;
    Tensor v_fc2_weight_, v_fc2_bias_;

    Tensor critic_forward(const Tensor& hidden);
    std::function<void(float, float, float, float)> log_cb_;
};

// ===========================================================================
// DPO Trainer — Direct Preference Optimization
// ===========================================================================
class DPOTrainer {
public:
    DPOTrainer(Model* policy, Model* ref_model, float beta = 0.1f,
               Optimizer* optimizer = nullptr);

    float train_step(const Tensor& chosen_logits, const Tensor& rejected_logits,
                     const Tensor& chosen_ids, const Tensor& rejected_ids);

    void set_log_callback(std::function<void(float loss, float kl)> cb) {
        log_cb_ = cb;
    }
    float last_loss() const { return last_loss_; }
    float beta() const { return beta_; }
    void set_beta(float b) { beta_ = b; }

private:
    Model* policy_;
    Model* ref_model_;
    float beta_;
    Optimizer* optimizer_;
    float last_loss_ = 0.0f;

    float compute_log_probs(const Tensor& logits, const Tensor& ids, Tensor* out_probs = nullptr);
    std::function<void(float, float)> log_cb_;
};

// ===========================================================================
// RLHF Pipeline — Complete preference-based alignment loop
// ===========================================================================
class RLHFPipeline {
public:
    RLHFPipeline(Model* model, Model* ref_model, Tokenizer* tokenizer,
                 RewardModel* reward_model, Trainer* trainer,
                 Optimizer* policy_opt, Optimizer* rm_opt);

    void generate_comparisons(const std::vector<std::string>& prompts, int max_new_tokens = 64);
    void train_reward_model(const std::vector<Comparison>& data, int epochs = 3, int batch_size = 8);
    void ppo_finetune(const std::vector<std::string>& prompts, int n_ppo_steps = 100,
                      int ppo_batch_size = 8, int max_new_tokens = 64);
    void dpo_finetune(const std::vector<Comparison>& comparisons, int n_steps = 10,
                      int batch_size = 4);

    void run(int n_rounds = 3, int n_prompts = 50, int n_ppo_steps = 200);

    const std::vector<Comparison>& comparisons() const { return comparison_buffer_; }
    const RLHFMetrics& metrics() const { return metrics_; }

    void set_log_callback(std::function<void(const RLHFMetrics&)> cb) { log_cb_ = cb; }
    void set_verbose(bool v) { verbose_ = v; }

    // GAE computation helpers
    Tensor compute_gae(const Tensor& rewards, const Tensor& values,
                       float gamma = 0.99f, float lam = 0.95f);
    Tensor compute_returns_discounted(const Tensor& rewards, float gamma = 0.99f);
    float compute_kl_between_policies(const Tensor& logits_pi, const Tensor& logits_ref,
                                       const Tensor& actions);
    float compute_reward_accuracy(const std::vector<Comparison>& data);
    float train_reward_model_epoch_with_accuracy(
        const std::vector<Comparison>& train_data,
        const std::vector<Comparison>& val_data,
        int epochs = 3, int batch_size = 8,
        const char* split_name = nullptr);

private:
    Model* model_;
    Model* ref_model_;
    Tokenizer* tokenizer_;
    RewardModel* reward_model_;
    Trainer* trainer_;
    Optimizer* policy_opt_;
    Optimizer* rm_opt_;
    std::vector<Comparison> comparison_buffer_;
    RLHFMetrics metrics_;
    bool verbose_ = true;
    std::function<void(const RLHFMetrics&)> log_cb_;

    Tensor extract_hidden(Tensor& logits, int64_t hidden_size);
    Tensor get_reward_for_sequence(Model* model, const std::vector<int>& ids);
};

} // namespace oil

namespace oil {

void collect_dense_params(DenseModel* dm, std::vector<Tensor*>& params);

} // namespace oil
