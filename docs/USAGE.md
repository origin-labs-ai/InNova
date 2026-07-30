# Usage Guide

> **How to Use InNova for Inference and Training**

---

## 🎯 Quick Start Examples

### Run Inference

```bash
# Basic inference with a model
./build/bin/oil-infer --model path/to/model.oil --prompt "Hello, world!"

# With more options
./build/bin/oil-infer \
    --model path/to/model.oil \
    --prompt "Write a poem about AI" \
    --max-tokens 100 \
    --temperature 0.7 \
    --top-k 50 \
    --output result.txt
```

### Train a Model

```bash
# Train from scratch
./build/bin/oil-train \
    --config path/to/config.json \
    --data path/to/training.txt \
    --output path/to/trained.oil \
    --batch-size 4 \
    --seq-length 128 \
    --epochs 10 \
    --learning-rate 0.001
```

### Fine-tune a Model

```bash
# Fine-tune an existing model
./build/bin/oil-finetune \
    --model path/to/base.oil \
    --data path/to/fine-tune.txt \
    --output path/to/fine-tuned.oil \
    --epochs 3 \
    --learning-rate 1e-5
```

### Convert Model Formats

```bash
# Convert HuggingFace safetensors to OIL
./build/bin/oil-convert \
    --input model.safetensors \
    --output model.oil \
    --target-bpw 1.50

# Get model info
./build/bin/oil-info --model model.oil
```

### Run Benchmarks

```bash
# Benchmark inference speed
./build/bin/oil-bench \
    --model model.oil \
    --prompts bench/prompts.txt \
    --iterations 100 \
    --warmup 10
```

---

## 📚 Command-Line Tools Reference

### oil-infer - Run Inference

**Description:** Run inference with a trained InNova model.

**Usage:**
```bash
oil-infer [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model`, `-m` | string | required | Path to the model file (.oil) |
| `--prompt`, `-p` | string | required | Input prompt for inference |
| `--max-tokens`, `-t` | int | 256 | Maximum number of tokens to generate |
| `--temperature` | float | 1.0 | Temperature for sampling (0.0-2.0) |
| `--top-k` | int | 50 | Top-k sampling |
| `--top-p` | float | 1.0 | Top-p (nucleus) sampling |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--batch-size` | int | 1 | Batch size for inference |
| `--output`, `-o` | string | stdout | Output file for generated text |
| `--verbose`, `-v` | flag | false | Verbose output |
| `--gpu` | flag | false | Use GPU acceleration (if available) |
| `--precision` | string | auto | Precision: auto, fp32, fp16, int8 |

**Examples:**

```bash
# Basic inference
oil-infer -m model.oil -p "Once upon a time"

# Creative writing with temperature
oil-infer -m model.oil -p "Write a haiku" --temperature 0.8 --max-tokens 50

# Batch inference
oil-infer -m model.oil -p "Prompt 1" -p "Prompt 2" --batch-size 2

# Save output to file
oil-infer -m model.oil -p "Tell me a story" -o story.txt --max-tokens 200
```

---

### oil-train - Train a Model from Scratch

**Description:** Train a new model from scratch.

**Usage:**
```bash
oil-train [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config`, `-c` | string | required | Path to training configuration JSON file |
| `--data`, `-d` | string | required | Path to training data (text file) |
| `--output`, `-o` | string | required | Path to save the trained model (.oil) |
| `--batch-size`, `-b` | int | 4 | Batch size for training |
| `--seq-length`, `-s` | int | 128 | Sequence length for training |
| `--epochs`, `-e` | int | 1 | Number of training epochs |
| `--learning-rate`, `-lr` | float | 0.001 | Learning rate |
| `--optimizer` | string | adamw | Optimizer: sgd, adam, adamw |
| `--weight-decay` | float | 0.01 | Weight decay for optimizer |
| `--warmup-steps` | int | 100 | Learning rate warmup steps |
| `--log-interval` | int | 10 | Log training metrics every N steps |
| `--save-interval` | int | 1000 | Save checkpoint every N steps |
| `--gpu` | flag | false | Use GPU acceleration |
| `--resume` | string | | Path to checkpoint to resume training |
| `--seed` | int | 42 | Random seed |
| `--verbose`, `-v` | flag | false | Verbose output |

**Configuration File Example (`config.json`):**

```json
{
  "model": {
    "type": "dense",
    "dim": 512,
    "n_layers": 6,
    "n_heads": 8,
    "vocab_size": 50257,
    "norm_eps": 1e-6
  },
  "format": {
    "target_bpw": 1.50,
    "format_planner": "awq"
  },
  "training": {
    "batch_size": 4,
    "seq_length": 128,
    "epochs": 10,
    "learning_rate": 0.001,
    "optimizer": "adamw"
  }
}
```

**Examples:**

```bash
# Train with configuration file
oil-train -c config.json -d data/tinyshakespeare.txt -o model.oil

# Train with command-line options
oil-train -d data.txt -o model.oil --dim 512 --n-layers 6 --epochs 5

# Train with GPU
oil-train -c config.json -d data.txt -o model.oil --gpu
```

---

### oil-finetune - Fine-tune an Existing Model

**Description:** Fine-tune a pre-trained model on new data.

**Usage:**
```bash
oil-finetune [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model`, `-m` | string | required | Path to base model (.oil) |
| `--data`, `-d` | string | required | Path to fine-tuning data (text file) |
| `--output`, `-o` | string | required | Path to save fine-tuned model (.oil) |
| `--epochs`, `-e` | int | 1 | Number of fine-tuning epochs |
| `--learning-rate`, `-lr` | float | 1e-5 | Learning rate (typically smaller than training) |
| `--batch-size`, `-b` | int | 4 | Batch size |
| `--seq-length`, `-s` | int | 128 | Sequence length |
| `--method` | string | full | Fine-tuning method: full, quantized |
| `--target-bpw` | float | 1.58 | Target bits-per-weight for quantized fine-tuning |
| `--target-modules` | string | all | Modules to fine-tune (comma-separated) |
| `--freeze-base` | flag | false | Freeze base model weights |
| `--log-interval` | int | 10 | Log every N steps |
| `--save-interval` | int | 100 | Save checkpoint every N steps |
| `--gpu` | flag | false | Use GPU acceleration |
| `--seed` | int | 42 | Random seed |
| `--verbose`, `-v` | flag | false | Verbose output |

**Examples:**

```bash
# Full fine-tuning
oil-finetune -m base.oil -d data.txt -o fine-tuned.oil --epochs 3

# Quantized fine-tuning (native OIL, target 2.0 BPW)
oil-finetune -m base.oil -d data.txt -o quant.oil --method quantized --target-bpw 2.0
```

---

### oil-convert - Convert Model Formats

**Description:** Convert models between different formats.

**Usage:**
```bash
oil-convert [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input`, `-i` | string | required | Input model file |
| `--output`, `-o` | string | required | Output model file (.oil) |
| `--input-format` | string | auto | Input format: auto, safetensors, gguf, pytorch |
| `--target-bpw` | float | 1.50 | Target bits per weight for OIL format |
| `--format-planner` | string | awq | Format planner: awq, uniform, custom |
| `--codebook-size` | int | 256 | Codebook size for OIL8 (256) or OIL4 (16) |
| `--verbose`, `-v` | flag | false | Verbose output |

**Supported Input Formats:**
- HuggingFace `safetensors`
- `gguf` (GGML format)
- PyTorch `.pt` files (limited support)
- ONNX (planned)

**Examples:**

```bash
# Convert safetensors to OIL
oil-convert -i model.safetensors -o model.oil --target-bpw 1.50

# Convert with different BPW
oil-convert -i model.safetensors -o model_small.oil --target-bpw 2.0

# Convert GGUF to OIL
oil-convert -i model.gguf -o model.oil --input-format gguf
```

---

### oil-info - Display Model Information

**Description:** Display information about a InNova model.

**Usage:**
```bash
oil-info [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model`, `-m` | string | required | Path to the model file (.oil) |
| `--json` | flag | false | Output in JSON format |
| `--verbose`, `-v` | flag | false | Verbose output |

**Output Includes:**
- Model architecture (type, dimensions, layers)
- Format information (BPW, formats used)
- Parameter count
- Memory usage
- Supported operations
- Metadata

**Examples:**

```bash
# Basic info
oil-info -m model.oil

# JSON output
oil-info -m model.oil --json > model_info.json

# Verbose output
oil-info -m model.oil -v
```

**Example Output:**

```
Model: model.oil
Type: Dense Transformer
Dimensions: 512
Layers: 6
Heads: 8
Vocab Size: 50257
Parameters: 12,345,678

Format: OIL (Mixed)
Average BPW: 1.50
Formats:
  - OIL8: 1% of weights (salient)
  - OIL4: 4% of weights (moderately important)
  - Ternary: 95% of weights (least important)

Codebook:
  - OIL8: 256 entries (FP32)
  - OIL4: 16 entries (FP16)

Memory Usage:
  - Model: 45.2 MB
  - Codebooks: 1.1 MB
  - Total: 46.3 MB
```

---

### oil-bench - Run Performance Benchmarks

**Description:** Benchmark model performance.

**Usage:**
```bash
oil-bench [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model`, `-m` | string | required | Path to the model file (.oil) |
| `--prompts`, `-p` | string | required | Path to file with benchmark prompts |
| `--iterations`, `-n` | int | 100 | Number of iterations |
| `--warmup` | int | 10 | Number of warmup iterations |
| `--batch-size` | int | 1 | Batch size for benchmarking |
| `--max-tokens` | int | 256 | Maximum tokens to generate |
| `--gpu` | flag | false | Use GPU acceleration |
| `--csv` | flag | false | Output results in CSV format |
| `--output`, `-o` | string | stdout | Output file for results |
| `--verbose`, `-v` | flag | false | Verbose output |

**Prompts File Format:**

```text
# Comments start with #
Once upon a time
Write a poem about AI
The quick brown fox jumps over the lazy dog
Explain quantum computing to a 5-year-old
```

**Output Metrics:**
- Tokens per second
- Time per token (ms)
- Prefill time
- Generation time
- Memory usage

**Examples:**

```bash
# Basic benchmark
oil-bench -m model.oil -p bench/prompts.txt -n 100

# With warmup and CSV output
oil-bench -m model.oil -p prompts.txt -n 1000 --warmup 50 --csv -o results.csv

# GPU benchmark
oil-bench -m model.oil -p prompts.txt -n 100 --gpu
```

---

## 📖 Using the C++ API

For programmatic access to InNova functionality, you can use the C++ API directly.

### Basic Setup

```cpp
#include <oil/oil.h>
#include <iostream>

using namespace oil;

int main() {
    // Initialize the OIL engine
    BackendConfig config;
    config.device = Device::CPU;
    config.precision = Precision::FP32;
    
    Backend backend(config);
    
    // Load a model
    Model* model = Model::load("model.oil", backend);
    
    // Create a tokenizer
    Tokenizer tokenizer("tokenizer.json");
    
    // Tokenize input
    std::vector<int> tokens = tokenizer.encode("Hello, world!");
    
    // Run inference
    Tensor input(Shape{1, (int)tokens.size()}, DType::I64);
    std::copy(tokens.begin(), tokens.end(), input.data<int64_t>());
    
    Tensor output = model->forward(input);
    
    // Sample from output
    int next_token = sampler.sample(output);
    
    // Decode token
    std::string text = tokenizer.decode({next_token});
    
    std::cout << "Generated: " << text << std::endl;
    
    // Cleanup
    delete model;
    
    return 0;
}
```

### Training Example

```cpp
#include <oil/oil.h>
#include <iostream>

using namespace oil;

int main() {
    // Create a model
    TransformerConfig config;
    config.dim = 512;
    config.n_layers = 6;
    config.n_heads = 8;
    config.vocab_size = 50257;
    
    DenseModel model(config);
    
    // Create tokenizer
    Tokenizer tokenizer("tokenizer.json");
    
    // Load training data
    DataLoader dataloader(&tokenizer, "data.txt", 4, 128);
    
    // Create optimizer
    AdamW optimizer(0.001, 0.01);
    
    // Create trainer
    Trainer trainer(&model, &tokenizer);
    trainer.compile(&optimizer);
    
    // Training configuration
    TrainConfig train_config;
    train_config.batch_size = 4;
    train_config.seq_length = 128;
    train_config.num_epochs = 10;
    train_config.learning_rate = 0.001;
    train_config.log_interval = 10;
    train_config.save_interval = 100;
    train_config.output_path = "trained.oil";
    
    // Set logging callback
    trainer.set_log_callback([](const TrainMetrics& metrics) {
        std::cout << "Step: " << metrics.step
                  << " Loss: " << metrics.loss
                  << " PPL: " << metrics.perplexity
                  << std::endl;
    });
    
    // Train!
    trainer.fit(dataloader, train_config);
    
    return 0;
}
```

---

### Inference with GPU Acceleration

```cpp
#include <oil/oil.h>
#include <oil/gpu_compute.h>
#include <iostream>

using namespace oil;

int main() {
    // Initialize GPU backend
    BackendConfig config;
    config.device = Device::GPU;
    config.precision = Precision::FP16;
    
    Backend backend(config);
    
    // Load model
    Model* model = Model::load("model.oil", backend);
    
    // Tokenize
    Tokenizer tokenizer("tokenizer.json");
    std::vector<int> tokens = tokenizer.encode("Hello");
    
    // Create input tensor
    Tensor input(Shape{1, (int)tokens.size()}, DType::I64);
    std::copy(tokens.begin(), tokens.end(), input.data<int64_t>());
    
    // Run inference on GPU
    Tensor output = model->forward(input);
    
    // Sample
    Sampler sampler(0.7f, 50);
    int next_token = sampler.sample(output);
    
    std::cout << "Generated token: " << next_token << std::endl;
    
    delete model;
    return 0;
}
```

---

## 🔧 Configuration Files

InNova uses JSON files for configuration.

### Model Configuration

```json
{
  "model": {
    "type": "dense",
    "dim": 512,
    "n_layers": 6,
    "n_heads": 8,
    "vocab_size": 50257,
    "norm_eps": 1e-6,
    "dropout": 0.1
  }
}
```

### Training Configuration

```json
{
  "training": {
    "batch_size": 4,
    "seq_length": 128,
    "epochs": 10,
    "learning_rate": 0.001,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "log_interval": 10,
    "save_interval": 1000
  },
  "format": {
    "target_bpw": 1.50,
    "format_planner": "awq"
  }
}
```

### Quantization Configuration

```json
{
  "quantization": {
    "target_bpw": 1.50,
    "method": "awq",
    "calibration_data": "calib.txt",
    "codebook_size_oil8": 256,
    "codebook_size_oil4": 16,
    "salient_percentile": 0.01,
    "moderate_percentile": 0.04
  }
}
```

---

## 🎯 Best Practices

### 1. Model Selection

| Task | Recommended Model Size | Notes |
|------|----------------------|-------|
| Testing/Development | 10M-50M parameters | Fast to train and run |
| Small Applications | 50M-200M parameters | Good balance of quality and speed |
| Production | 200M-2B parameters | Best quality, requires GPU |
| Research | 2B+ parameters | Experimental, requires significant resources |

### 2. Quantization Settings

| Use Case | Target BPW | Formats Used |
|----------|-----------|--------------|
| Maximum Quality | 3.0+ | Mostly OIL8 |
| Balanced | 1.50-2.0 | OIL8 + OIL4 + Ternary |
| Compact | 1.0-1.5 | OIL4 + Ternary + Binary |
| Minimum Size | < 1.0 | Binary + Ternary |

### 3. Training Tips

- Start with a small model for testing
- Use a learning rate finder to determine optimal LR
- Warmup is important for Adam optimizers
- Monitor loss and perplexity
- Save checkpoints regularly

### 4. Inference Tips

- Use `--max-tokens` to limit generation length
- Adjust `--temperature` for creativity vs. coherence
- Use `--top-k` and `--top-p` together for best results
- Lower temperature = more deterministic
- Higher temperature = more creative/random

---

## 🐛 Common Issues

### 1. Out of Memory

**Solution:**
- Reduce batch size
- Reduce sequence length
- Use smaller model
- Enable quantization to reduce memory usage
- Use CPU instead of GPU (if applicable)

### 2. Slow Inference

**Solution:**
- Enable AVX2 optimizations (`-DOIL_AVX2=ON`)
- Use GPU acceleration (`--gpu`)
- Quantize the model to lower BPW
- Reduce batch size

### 3. Model Fails to Load

**Solution:**
- Check file path is correct
- Verify model format is supported
- Check for file corruption
- Ensure all required files are present

### 4. Training Loss Doesn't Decrease

**Solution:**
- Check learning rate (try 1e-4 to 1e-3)
- Verify data is properly tokenized
- Increase batch size or sequence length
- Try different optimizer
- Monitor gradients for vanishing/exploding

### 5. Poor Quality Output

**Solution:**
- Increase model size
- Train for more epochs
- Use higher quality training data
- Adjust quantization settings
- Try different sampling parameters

---

## 📚 Additional Resources

- **[Wiki Usage Guide](../wiki/Usage-Guide.md)** — Extended usage examples and tips
- **[Wiki Inference Guide](../wiki/Inference.md)** — In-depth inference documentation
- **[Wiki Training Guide](../wiki/Training.md)** — Detailed training instructions
- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Architecture](ARCHITECTURE.md)** - System architecture overview
- **[Research](RESEARCH.md)** - Research papers and algorithms
- **[Examples](EXAMPLES/)** - Code examples and tutorials

---

*Last updated: July 12, 2026*
