// ============================================================================
// adapter_core.h — Common types + OIL mixed-precision funnel for ADAPTER-EDITION
// ----------------------------------------------------------------------------
// Every external-format bridge (PTQ, GGUF, Safetensors) loads weights into
// AdapterTensor (always FP32), then funnels through write_oil_mixed() which
// assigns a per-block mixed-precision format (OIL1/SPARK_Q0/OIL4/OIL8/OIL16)
// via the parent project's FormatRegistry + codebooks and writes a .oil file.
//
// Design rule: NOTHING leaves the adapter edition in a foreign format. The only
// on-disk product is an OIL mixed-precision (.oil) file — so a user who shows
// up with GGUF, safetensors, FP32/FP16/FP8 model always gets a native OIL model.
// ============================================================================
#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace oil {
namespace adapters {

// A weight tensor loaded from any external format, held as FP32 internally.
struct AdapterTensor {
    std::string name;
    std::vector<float> data;       // FP32 weights, row-major, flattened
    std::vector<int64_t> shape;    // e.g. {out_features, in_features} for Linear

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
    bool empty() const { return data.empty(); }
};

// Configuration for the OIL mixed-precision conversion funnel.
struct BridgeConfig {
    // Target average bits-per-weight for the mixed-precision allocation.
    //   1.00 = all OIL1, 1.50 = all SPARK_Q0, 4.0 = all OIL4, 8.0 = all OIL8,
    //   16.0 = OIL16, 32.0 = OIL32. Values in between produce a genuine mix.
    float target_bpw = 1.50f;

    // Weights per block (block-shared scale / codebook granularity).
    int block_size = 256;

    // Adaptive mode: if true, use calibration activations to guide format allocation
    // (more OIL8/OIL4 for blocks with high activation×weight importance).
    bool adaptive = false;

    // Clamp: never go below OIL1 (1.0 bpw) or above OIL32 (32.0 bpw).
    bool verbose = false;

    // Output .oil path (write_oil_mixed writes here).
    std::string output_path;
};

// Result of writing one tensor as mixed-precision blocks.
struct MixedWriteResult {
    uint32_t block_start = 0;
    uint32_t num_blocks = 0;
    float achieved_bpw = 0.0f;
};

// Single conversion funnel: take FP32 AdapterTensors, allocate a per-block
// mixed-precision format by importance (magnitude), quantize each block with
// the matching codebook/OIL format logic, and write a .oil file.
// Returns true on success. This is the ONLY exit point of the adapter edition.
bool write_oil_mixed(const std::vector<AdapterTensor>& tensors,
                     const BridgeConfig& cfg);

// Estimate the achieved bpw for a tensor at a given target (pre-write preview).
float estimate_mixed_bpw(int64_t num_weights, int block_size, float target_bpw);

// ── External-format dequantization helpers (foreign dtype -> FP32) ─────────

float fp16_to_float(uint16_t h);     // IEEE 754 binary16
float bf16_to_float(uint16_t b);     // bfloat16
float fp8_e4m3_to_float(uint8_t x);  // FP8 E4M3 (OCP)
float fp8_e5m2_to_float(uint8_t x);  // FP8 E5M2 (OCP)

// ── Format detection (by magic bytes / extension) ───────────────────────────

enum class ExternalFormat {
    RAW_FP32,
    RAW_FP16,
    RAW_FP8_E4M3,
    RAW_FP8_E5M2,
    GGUF,
    SAFETENSORS,
    OIL,
    UNKNOWN
};

ExternalFormat detect_format(const std::string& path);
const char* external_format_name(ExternalFormat f);

} // namespace adapters
} // namespace oil
