# ⚡ MYTHOS.cpp Wiki

> **M**ixed-format **Y**our-own **T**ensor **H**andcrafted **O**ptimized **S**ystem

Welcome to the MYTHOS.cpp wiki! This is a **zero-dependency C++20 AI engine** that lets you train from scratch, fine-tune in native OIL format, quantize, and run inference — all within a single `.oil` binary format.

## 🚀 Quick Start

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

## 📖 Wiki Sections

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

## 🏗️ Project Structure

```
MYTHOS.cpp/
├── include/oil/          # Public headers
├── src/                  # Core implementation
├── engines/              # Inference & training engines
│   ├── OIL8/            # OIL8 codec
│   ├── inference/       # Inference engine
│   └── trainer/         # Training engine (dense + MoE)
├── tools/                # CLI tools
├── tests/                # Test suite
├── bench/                # Benchmarks
├── docs/                 # Documentation
└── wiki/                 # This wiki
```

## ✅ Build Status

| Platform | Compiler | Status |
|----------|----------|--------|
| Windows 11 | Clang 22.1.7 (clang-cl) | ✅ All 82 targets build (25 libs + 25 executables + 32 tests), 0 errors, 0 warnings |
| Linux | GCC ≥ 12 / Clang ≥ 16 | ✅ All 82 targets build, tests pass |
| macOS (target) | Apple Clang | ⏳ Pending |

## 📄 Whitepaper

The full MYTHOS.cpp whitepaper (128 pages) is available at **[publication/whitepaper/](../publication/whitepaper/)**.

## 🔗 Links

- [GitHub Repository](https://github.com/xprimesamx/MYTHOS.cpp)
- [Research Papers](.research/)
- [API Headers](include/oil/)
