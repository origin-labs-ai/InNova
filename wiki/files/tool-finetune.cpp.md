# `oil-finetune` — Fine-Tuning CLI

**Path:** `tools/finetune.cpp`

Fine-tuning tool for adapting pre-trained OIL models to new data.

## Usage

```
oil-finetune --model <base.oil> --data <data.txt> --output <output.oil> [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Base OIL model path |
| `--data` | required | Fine-tuning data |
| `--output` | required | Output OIL model |
| `--lr` | 1e-5 | Learning rate (lower for finetuning) |
| `--epochs` | 3 | Number of epochs |
| `--batch-size` | 4 | Batch size |
| `--target-bpw` | 1.58 | Target bits-per-weight for quantized fine-tuning |
| `--freeze-layers` | - | Comma-separated layers to freeze |

### Example

```bash
# Full fine-tune
oil-finetune --model base.oil --data data.txt --output finetuned.oil --lr 1e-5

# Quantized fine-tune
oil-finetune --model base.oil --data data.txt --output quant.oil --target-bpw 2.0
```
