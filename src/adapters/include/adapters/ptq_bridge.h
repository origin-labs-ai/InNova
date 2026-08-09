// ============================================================================
// ptq_bridge.h — Post-Training Quantization: raw FP32/FP16/FP8 -> QUANT mixed
// ----------------------------------------------------------------------------
// Also handles re-quantization of an existing .quant file (dequantize to FP32,
// then re-allocate mixed-precision formats at a new target bpw).
// ============================================================================
#pragma once
#include "adapters/adapter_core.h"
#include <string>
#include <vector>

namespace quant {
namespace adapters {

// Load a raw weight blob (FP32/FP16/BF16/FP8-E4M3/FP8-E5M2) as a single
// AdapterTensor. `element_bytes` selects the dtype (4/2/2/1/1). `name` is the
// tensor name written to the QUANT file.
AdapterTensor load_raw_blob(const std::string& path, ExternalFormat fmt,
                            const std::string& tensor_name = "weights");

// PTQ entry point: convert a raw weight file (any foreign dtype) to an QUANT
// mixed-precision file at cfg.target_bpw. Returns true on success.
bool ptq_raw(const std::string& input_path, ExternalFormat fmt,
             const BridgeConfig& cfg);

// Re-quantize an existing .quant file: dequantize every tensor to FP32, then
// re-write at cfg.target_bpw. Useful to change compression level of an QUANT model.
bool ptq_requant_quant(const std::string& input_quant, const BridgeConfig& cfg);

} // namespace adapters
} // namespace quant
