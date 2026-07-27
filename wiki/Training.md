# Training

## Overview

MYTHOS.cpp supports training models from scratch using AdamW optimizer with cosine LR schedule.

## Quick Start

```bash
# Train a small model
oil-train --config config.json --data data/tinyshakespeare.txt --output trained.oil
```

## Configuration

Create `config.json`:
```json
{
  "model": {
    "vocab_size": 32000,
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "intermediate_size": 3072,
    "max_seq_len": 2048
  },
  "training": {
    "learning_rate": 3e-4,
    "batch_size": 8,
    "epochs": 10,
    "warmup_steps": 100,
    "weight_decay": 0.1,
    "grad_clip": 1.0
  }
}
```

## Training Types

### Dense Training
Standard transformer training — all parameters updated.

```bash
oil-train --config config_dense.json --data data.txt --output dense.oil
```

### MoE Training
Sparse MoE training with expert routing and load balancing.

```bash
oil-train --config config_moe.json --data data.txt --output moe.oil
```

MoE config:
```json
{
  "model": {
    "use_moe": true,
    "num_experts": 8,
    "top_k": 2
  },
  "training": {
    "learning_rate": 2e-4,
    "load_balancing_coef": 0.01
  }
}
```

## Training Loop

```
for epoch = 1..epochs:
    for batch in dataloader:
        1. Forward: logits = model(input_ids)
        2. Loss: cross_entropy(logits, targets)
        3. Backward: loss.backward()
        4. Clip: gradient_norm <= grad_clip
        5. Optimize: optimizer.step()
        6. Log: loss, lr, tokens/sec
    Save: checkpoint_{step}.oil
```

## Checkpointing

```bash
# Checkpoint structure
checkpoints/
├── step_1000.oil     # Model weights at step 1000
├── step_1000.opt     # Optimizer state
├── step_2000.oil
├── step_2000.opt
└── best.oil          # Best by validation loss

# Resume training
oil-train --config config.json --data data.txt \
    --output trained.oil --resume checkpoints/step_1000.oil
```

## Fine-Tuning

For adapting pre-trained models, see [Fine-Tuning docs](files/tool-finetune.cpp.md).

```bash
# Full fine-tune
oil-finetune --model base.oil --data data.txt --output finetuned.oil --lr 1e-5

# Quantized fine-tune (native OIL, target 2.0 BPW)
oil-finetune --model base.oil --data data.txt --output quant.oil --target-bpw 2.0
```

## Distributed Training

See [distributed.h](files/distributed.h.md) for multi-device training support.
