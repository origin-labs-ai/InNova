# âš¡ InNova Wiki

> **M**ixed-format **Y**our-own **T**ensor **H**andcrafted **O**ptimized **S**ystem

Welcome to the InNova wiki! This is a **zero-dependency C++20 AI engine** that lets you train from scratch, fine-tune in native OIL format, quantize, and run inference â€” all within a single `.oil` binary format.

## ðŸš€ Quick Start

```bash
# Configure & Build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure

# Convert a model to OIL format
build/tools/oil-convert --input model.safetensors --output model.oil --target-bpw 1.50

# Run inference
build/tools/oil-infer --model model.oil --prompt "Hello" --max-tokens 256

# Train from scratch
build/tools/oil-train --config config.json --data data/tinyshakespeare.txt --output trained.oil
```

## ðŸ“– Wiki Sections

| Section | Description |
|---------|-------------|
| [Architecture](Architecture) | System design, philosophy, and component overview |
| [Build Guide](Build-Guide) | Build instructions for all platforms |
| [Usage Guide](Usage-Guide) | How to use the tools and engines |
| [API Reference](Api-Reference) | Complete C++ API documentation |
| [Modules](Modules) | Detailed module breakdown |
| [OIL Format](OIL-Format) | The OIL binary format specification |
| [Training](Training) | Training from scratch guide |
| [Inference](Inference) | Inference and generation guide |
| [Research](Research) | Research papers and background |
| [Contributing](Contributing) | How to contribute |
| [File Docs](files/_index) | Per-file documentation index |

## ðŸ—ï¸ Project Structure

```
InNova/
â”œâ”€â”€ include/oil/          # Public headers
â”œâ”€â”€ src/                  # Core implementation
â”œâ”€â”€ engines/              # Inference & training engines
â”‚   â”œâ”€â”€ OIL8/            # OIL8 codec
â”‚   â”œâ”€â”€ inference/       # Inference engine
â”‚   â””â”€â”€ trainer/         # Training engine (dense + MoE)
â”œâ”€â”€ tools/                # CLI tools
â”œâ”€â”€ tests/                # Test suite
â”œâ”€â”€ bench/                # Benchmarks
â”œâ”€â”€ docs/                 # Documentation
â””â”€â”€ wiki/                 # This wiki
```

## âœ… Build Status

| Platform | Compiler | Status |
|----------|----------|--------|
| Windows 11 | Clang 22.1.7 (clang-cl) | âœ… All 82 targets build (25 libs + 25 executables + 32 tests), 0 errors, 0 warnings |
| Linux | GCC â‰¥ 12 / Clang â‰¥ 16 | âœ… All 82 targets build, tests pass |
| macOS (target) | Apple Clang | â³ Pending |

## ðŸ“„ Whitepaper

The full InNova whitepaper (128 pages) is available at **[publication/whitepaper/](../publication/whitepaper/)**.

## ðŸ”— Links

- [GitHub Repository](https://github.com/origin-labs-ai/InNova)
- [Research Papers](.research/)
- [API Headers](include/oil/)
