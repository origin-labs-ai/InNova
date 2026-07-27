// ============================================================================
// safetensors_bridge.h — Safetensors loader -> AdapterTensors (FP32) -> OIL mixed
// ============================================================================
#pragma once
#include "adapters/adapter_core.h"
#include <string>
#include <vector>

namespace oil {
namespace adapters {

// Parse a safetensors file and dequantize every tensor to FP32 AdapterTensors.
// Supports dtypes: F32, F16, BF16, I64, I32, I16, I8, U8, BOOL, F8_E4M3, F8_E5M2.
std::vector<AdapterTensor> load_safetensors(const std::string& path, bool verbose = false);

// Safetensors -> OIL mixed-precision file.
bool safetensors_to_oil(const std::string& input_path, const BridgeConfig& cfg);

} // namespace adapters
} // namespace oil
