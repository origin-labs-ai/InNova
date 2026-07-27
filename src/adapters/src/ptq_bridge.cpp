// ============================================================================
// ptq_bridge.cpp — raw foreign-dtype weights -> OIL mixed-precision
// ============================================================================
#include "adapters/ptq_bridge.h"
#include "adapters/adapter_core.h"

#include "oil/oil_format.h"
#include "oil/codebook.h"
#include "oil/types.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <cstdint>

namespace oil {
namespace adapters {

AdapterTensor load_raw_blob(const std::string& path, ExternalFormat fmt,
                            const std::string& tensor_name) {
    AdapterTensor t;
    t.name = tensor_name;
    std::ifstream f(path, std::ios::binary);
    if (!f) return t;
    f.seekg(0, std::ios::end);
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<uint8_t> raw((size_t)size);
    if (size > 0) f.read(reinterpret_cast<char*>(raw.data()), size);

    size_t count = 0;
    switch (fmt) {
        case ExternalFormat::RAW_FP32: {
            count = raw.size() / 4;
            t.data.resize(count);
            for (size_t i = 0; i < count; i++) {
                std::memcpy(&t.data[i], &raw[i * 4], 4);
            }
            t.shape = { (int64_t)count };
            break;
        }
        case ExternalFormat::RAW_FP16: {
            count = raw.size() / 2;
            t.data.resize(count);
            for (size_t i = 0; i < count; i++)
                t.data[i] = fp16_to_float(*(uint16_t*)(&raw[i * 2]));
            t.shape = { (int64_t)count };
            break;
        }
        case ExternalFormat::RAW_FP8_E4M3: {
            count = raw.size();
            t.data.resize(count);
            for (size_t i = 0; i < count; i++)
                t.data[i] = fp8_e4m3_to_float(raw[i]);
            t.shape = { (int64_t)count };
            break;
        }
        case ExternalFormat::RAW_FP8_E5M2: {
            count = raw.size();
            t.data.resize(count);
            for (size_t i = 0; i < count; i++)
                t.data[i] = fp8_e5m2_to_float(raw[i]);
            t.shape = { (int64_t)count };
            break;
        }
        default:
            // Treat unknown as raw FP32 (already loaded).
            count = raw.size() / 4;
            t.data.resize(count);
            for (size_t i = 0; i < count; i++)
                std::memcpy(&t.data[i], &raw[i * 4], 4);
            t.shape = { (int64_t)count };
            break;
    }
    return t;
}

bool ptq_raw(const std::string& input_path, ExternalFormat fmt,
             const BridgeConfig& cfg) {
    AdapterTensor t = load_raw_blob(input_path, fmt);
    if (t.empty()) return false;
    BridgeConfig c = cfg;
    return write_oil_mixed({ t }, c);
}

bool ptq_requant_oil(const std::string& input_oil, const BridgeConfig& cfg) {
    OILReader reader(input_oil);
    if (!reader.valid()) return false;

    auto names = reader.tensor_names();
    std::vector<AdapterTensor> tensors;
    tensors.reserve(names.size());
    for (const auto& n : names) {
        oil::Tensor ot = reader.read_tensor(n);
        if (ot.numel() == 0) continue;
        AdapterTensor at;
        at.name = n;
        at.data.resize((size_t)ot.numel());
        const float* src = ot.data<float>();
        std::memcpy(at.data.data(), src, (size_t)ot.numel() * sizeof(float));
        at.shape = { (int64_t)ot.numel() };
        tensors.push_back(std::move(at));
    }
    BridgeConfig c = cfg;
    return write_oil_mixed(tensors, c);
}

} // namespace adapters
} // namespace oil
