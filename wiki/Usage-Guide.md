# Usage Guide

## Model Conversion

Convert HuggingFace safetensors models to OIL format:

```bash
# Basic conversion (FP16)
oil-convert --input model.safetensors --output model.oil

# Quantized conversion (OIL4, ~1.50 BPW)
oil-convert --input model.safetensors --output model.oil --target-bpw 1.50

# OIL8 quantization (~0.85 BPW)
oil-convert --input model.safetensors --output model.oil --target-bpw 0.85

# With custom config
oil-convert --input model.safetensors --output model.oil --config config.json

# Verbose output
oil-convert --input model.safetensors --output model.oil --verbose
```

## Inference

```bash
# Basic
oil-infer --model model.oil --prompt "Hello, world!"

# With generation parameters
oil-infer --model model.oil \
    --prompt "Once upon a time" \
    --max-tokens 512 \
    --temperature 0.8 \
    --top-p 0.95 \
    --top-k 40 \
    --seed 42

# Streaming output
oil-infer --model model.oil --prompt "Hello" --stream

# Interactive chat mode
oil-infer --model model.oil --interactive

# Benchmark mode (no output, just tokens/sec)
oil-infer --model model.oil --prompt "Hello" --bench
```

## Training

```bash
# Minimal
oil-train --config config.json --data data.txt --output model.oil

# Full options
oil-train --config config.json \
    --data data/tinyshakespeare.txt \
    --output trained.oil \
    --lr 3e-4 \
    --batch-size 8 \
    --epochs 10 \
    --log-dir ./logs

# Resume from checkpoint
oil-train --config config.json \
    --data data.txt \
    --output model.oil \
    --resume checkpoints/step_1000.oil

# MoE training
oil-train --config config_moe.json \
    --data data.txt \
    --output moe_model.oil
```

## Fine-Tuning

```bash
# Full fine-tune
oil-finetune --model base.oil \
    --data finetune.txt \
    --output finetuned.oil \
    --lr 1e-5 \
    --epochs 3

# Quantized fine-tune (target 2.0 BPW)
oil-finetune --model base.oil \
    --data finetune.txt \
    --output quant.oil \
    --target-bpw 2.0

# Freeze specific layers
oil-finetune --model base.oil \
    --data finetune.txt \
    --output partial.oil \
    --freeze-layers "tok_embeddings,norm,lm_head"
```

## Model Info

```bash
# Basic info
oil-info --model model.oil

# Verbose (tensor-level details)
oil-info --model model.oil --verbose

# Summary only
oil-info --model model.oil --summary
```

## Benchmarking

```bash
# All benchmarks
oil-bench --all

# Kernel benchmarks only
oil-bench --kernels

# Inference benchmark
oil-bench --inference --model model.oil

# Quality evaluation
oil-bench --quality --model model.oil --dataset test.txt
```
