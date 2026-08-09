#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/trainer.h"
#include "quant/ste_quantizer.h"
#include "quant/optimizer.h"
#include <vector>
#include <string>

namespace quant {

struct FineTuneConfig {
    float learning_rate = 1e-5f;
    int num_epochs = 1;
    int64_t batch_size = 4;
    int64_t seq_length = 512;
    int warmup_steps = 10;
    int log_interval = 10;
    int save_interval = 100;
    bool update_codebooks = true;
    float grad_threshold = 1e-6f;
    float max_grad_norm = 1.0f;
    std::string output_path = "finetuned.quant";
    std::string optimizer_name = "adafactor";
};

class FineTuner {
public:
    FineTuner(Model* model, Tokenizer* tokenizer);
    
    // Configure fine-tuning
    void configure(const FineTuneConfig& cfg);
    
    // Fine-tune on data
    void fine_tune(const std::string& data_path);
    
    // Fine-tune with provided data loader
    void fine_tune(DataLoader& dataloader);
    
    // Find which weight blocks need updating based on gradient magnitude
    std::vector<int> find_trainable_blocks(const Tensor& gradients, float threshold);
    
    // Apply gradient update in QUANT format via STE
    void apply_quant_update(const Tensor& fp32_grad, Tensor& quant_weight, Format fmt);
    
    // Save fine-tuned model
    void save(const std::string& path);
    
private:
    Model* model_;
    Tokenizer* tokenizer_;
    FineTuneConfig cfg_;
    std::unique_ptr<Optimizer> optimizer_;
    
    void freeze_base_model();
    void unfreeze_for_finetune();
};

} // namespace quant
