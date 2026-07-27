// ============================================================================
// oil_import.cpp — CLI: auto-detect format & import -> OIL mixed-precision
// ============================================================================
// Usage:
//   oil_import --input <path> --output <out.oil> [--bpw 1.58] [--block-size 256] [--verbose]
//
// Detects input format by magic bytes / extension, then dispatches to the
// appropriate bridge (GGUF, Safetensors, raw FP32/FP16/FP8, or re-quant OIL).
// ============================================================================
#include "adapters/adapter_core.h"
#include "adapters/ptq_bridge.h"
#include "adapters/gguf_bridge.h"
#include "adapters/safetensors_bridge.h"

#include <iostream>
#include <cstring>
#include <cstdio>

using namespace oil::adapters;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "OIL IMPORT — Auto-detect format & import to OIL mixed-precision\n\n"
            "Usage:\n"
            "  oil_import --input <path> --output <out.oil> [options]\n\n"
            "Options:\n"
            "  --input <path>        Input model path (any supported format)\n"
            "  --output <path>       Output OIL mixed-precision file\n"
            "  --bpw <float>         Target bits-per-weight (default: 1.58)\n"
            "  --block-size <N>      Block size (default: 256)\n"
            "  --verbose             Print per-tensor stats\n"
            "  -h, --help            Show this help\n\n"
            "Supported input formats (auto-detected):\n"
            "  GGUF, Safetensors, raw FP32/FP16, raw FP8 (E4M3/E5M2), existing .oil\n");
        return 0;
    }

    BridgeConfig cfg;
    cfg.target_bpw = 1.58f;
    cfg.block_size = 256;
    std::string input_path;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)   input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) cfg.output_path = argv[++i];
        else if (strcmp(argv[i], "--bpw") == 0 && i + 1 < argc) cfg.target_bpw = (float)std::atof(argv[++i]);
        else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) cfg.block_size = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--verbose") == 0) cfg.verbose = true;
    }

    if (input_path.empty() || cfg.output_path.empty()) {
        std::fprintf(stderr, "Error: --input and --output are required.\n");
        return 1;
    }

    ExternalFormat fmt = detect_format(input_path);
    std::fprintf(stdout, "Detected format: %s\n", external_format_name(fmt));

    bool ok = false;
    switch (fmt) {
        case ExternalFormat::GGUF:         ok = gguf_to_oil(input_path, cfg);    break;
        case ExternalFormat::SAFETENSORS:  ok = safetensors_to_oil(input_path, cfg); break;
        case ExternalFormat::OIL:          ok = ptq_requant_oil(input_path, cfg); break;
        case ExternalFormat::RAW_FP16:
        case ExternalFormat::RAW_FP8_E4M3:
        case ExternalFormat::RAW_FP8_E5M2:
        case ExternalFormat::RAW_FP32:
        default:                           ok = ptq_raw(input_path, fmt, cfg);   break;
    }

    if (!ok) {
        std::fprintf(stderr, "Error: import failed for %s\n", input_path.c_str());
        return 1;
    }
    std::fprintf(stdout, "Success: imported to OIL mixed-precision model at %s\n", cfg.output_path.c_str());
    return 0;
}
