// ============================================================================
// gguf_bridge.h — GGUF loader: GGUF -> AdapterTensors (FP32) -> QUANT mixed
// ============================================================================
#pragma once
#include "adapters/adapter_core.h"
#include <string>
#include <vector>

namespace quant {
namespace adapters {

// Parse a GGUF file and dequantize every tensor to FP32 AdapterTensors.
// Supports GGML types: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q6_K.
std::vector<AdapterTensor> load_gguf(const std::string& path, bool verbose = false);

// GGUF -> QUANT mixed-precision file.
bool gguf_to_quant(const std::string& input_path, const BridgeConfig& cfg);

} // namespace adapters
} // namespace quant
