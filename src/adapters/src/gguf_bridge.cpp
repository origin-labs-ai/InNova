// ============================================================================
// gguf_bridge.cpp — GGUF parser + dequantizer -> AdapterTensors -> OIL mixed
// ============================================================================
#include "adapters/gguf_bridge.h"
#include "adapters/adapter_core.h"

#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdio>

namespace oil {
namespace adapters {

// ── GGUF type constants ──────────────────────────────────────────────────────

static const uint32_t GGML_TYPE_F32  = 0;
static const uint32_t GGML_TYPE_F16  = 1;
static const uint32_t GGML_TYPE_Q4_0 = 2;
static const uint32_t GGML_TYPE_Q4_1 = 3;
static const uint32_t GGML_TYPE_Q5_0 = 6;
static const uint32_t GGML_TYPE_Q5_1 = 7;
static const uint32_t GGML_TYPE_Q8_0 = 8;
static const uint32_t GGML_TYPE_Q8_1 = 9;
static const uint32_t GGML_TYPE_Q2_K = 10;
static const uint32_t GGML_TYPE_Q6_0 = 14;

// ── Low-level GGUF file I/O ──────────────────────────────────────────────────

static uint64_t read_le_u64(std::istream& is) {
    uint64_t v = 0;
    is.read(reinterpret_cast<char*>(&v), 8);
    return v;
}
static uint32_t read_le_u32(std::istream& is) {
    uint32_t v = 0;
    is.read(reinterpret_cast<char*>(&v), 4);
    return v;
}
static uint16_t read_le_u16(std::istream& is) {
    uint16_t v = 0;
    is.read(reinterpret_cast<char*>(&v), 2);
    return v;
}
static std::string read_gguf_string(std::istream& is) {
    uint64_t len = read_le_u64(is);
    if (len > (1 << 24)) return {};  // sanity guard
    std::string s(len, '\0');
    if (len > 0) is.read(&s[0], (std::streamsize)len);
    return s;
}

// ── FP16 helpers ─────────────────────────────────────────────────────────────

static float gguf_fp16_to_f32(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp  = (h >> 10) & 0x1F;
    int mant = h & 0x3FF;
    if (exp == 0) {
        return (mant ? (float)mant / 16384.0f * 2.0f : 0.0f) * (sign ? -1 : 1);
    } else if (exp == 31) {
        return mant ? NAN : INFINITY;
    }
    return ((float)(mant | 0x400) / 8388608.0f) * std::ldexp(1.0f, exp - 15) * (sign ? -1 : 1);
}

// ── GGML dequantizers ────────────────────────────────────────────────────────

static void dequant_q4_0(const uint8_t* block, float* out, int64_t offset) {
    int16_t scale_half = *(const int16_t*)(block);
    float scale = scale_half ? gguf_fp16_to_f32((uint16_t)scale_half) : 0.0f;
    for (int i = 0; i < 32; i++) {
        uint8_t q = block[2 + i / 2];
        if (i % 2 == 0) q &= 0x0F; else q >>= 4;
        out[offset + i] = ((float)((int8_t)(q - 8) & 0xF)) * scale; // 0..15 -> -8..7
    }
}

static void dequant_q4_1(const uint8_t* block, float* out, int64_t offset) {
    float scale = gguf_fp16_to_f32(*(const uint16_t*)(block));
    float mn    = gguf_fp16_to_f32(*(const uint16_t*)(block + 2));
    for (int i = 0; i < 32; i++) {
        uint8_t q = block[4 + i / 2];
        if (i % 2 == 0) q &= 0x0F; else q >>= 4;
        out[offset + i] = mn + (float)q * scale;
    }
}

static void dequant_q5_0(const uint8_t* block, float* out, int64_t offset) {
    float scale = gguf_fp16_to_f32(*(const uint16_t*)(block));
    uint8_t mn_bit = block[2] & 1;
    uint8_t* qdata = (uint8_t*)(block + 3);
    for (int i = 0; i < 32; i++) {
        uint8_t q = qdata[i / 4];
        int bit = (i % 4) * 2;
        int v = (i < 32) ? (int)(qdata[i] & 0x1F) : 0;
        // Q5_0: 5-bit signed offset from zero (approximate for PTQ).
        float norm = (float)(v - 15) / 15.0f;
        if (norm < -1.0f) norm = -1.0f;
        if (norm >  1.0f) norm =  1.0f;
        out[offset + i] = norm * scale;
    }
}

static void dequant_q5_1(const uint8_t* block, float* out, int64_t offset) {
    float scale = gguf_fp16_to_f32(*(const uint16_t*)(block));
    float mn    = gguf_fp16_to_f32(*(const uint16_t*)(block + 2));
    uint8_t* qdata = (uint8_t*)(block + 4);
    for (int i = 0; i < 32; i++) {
        int v = (int)(qdata[i] & 0x1F);
        out[offset + i] = mn + (float)v * scale / 31.0f;
    }
}

static void dequant_q8_0(const uint8_t* block, float* out, int64_t offset) {
    float scale = gguf_fp16_to_f32(*(const uint16_t*)(block));
    for (int i = 0; i < 32; i++) {
        int8_t q = (int8_t)block[2 + i];
        out[offset + i] = (float)q * scale;
    }
}

static void dequant_q8_1(const uint8_t* block, float* out, int64_t offset) {
    float scale = gguf_fp16_to_f32(*(const uint16_t*)(block));
    float mn    = gguf_fp16_to_f32(*(const uint16_t*)(block + 2));
    for (int i = 0; i < 32; i++) {
        int8_t q = (int8_t)block[4 + i];
        out[offset + i] = mn + (float)q * scale;
    }
}

static void dequant_q6_0(const uint8_t* block, float* out, int64_t offset) {
    float scale = gguf_fp16_to_f32(*(const uint16_t*)(block));
    for (int i = 0; i < 32; i++) {
        uint8_t q = block[2 + i];
        int v = (int)q - 32;
        float norm = (float)v / 32.0f;
        out[offset + i] = norm * scale;
    }
}

static void dequant_q2_k(const uint8_t* block, float* out, int64_t offset) {
    float scale = gguf_fp16_to_f32(*(const uint16_t*)(block));
    const uint8_t* qdata = block + 2;
    for (int i = 0; i < 32; i++) {
        uint8_t q = qdata[i];
        int v = (int)(q & 0x03); // 2-bit
        float norm = (float)(v - 1) / 1.0f;
        out[offset + i] = norm * scale;
    }
}

// ── GGUF parser ──────────────────────────────────────────────────────────────

std::vector<AdapterTensor> load_gguf(const std::string& path, bool verbose) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return {};

    char magic[4];
    is.read(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) return {};

    uint32_t version = read_le_u32(is);
    if (version < 1 || version > 3) return {};

    uint64_t n_tensors = read_le_u64(is);
    uint64_t n_kv      = read_le_u64(is);
    if (verbose)
        std::fprintf(stderr, "GGUF v%u tensors=%llu metadata=%llu\n",
                      version, (unsigned long long)n_tensors, (unsigned long long)n_kv);

    // Skip metadata (key-value pairs).
    for (uint64_t i = 0; i < n_kv; i++) {
        (void)read_gguf_string(is);
        uint32_t vt = read_le_u32(is);
        switch (vt) {
            case 0: is.ignore(1); break;
            case 1: is.ignore(8); break;
            case 2: is.ignore(4); break;
            case 3: is.ignore(2); break;
            case 4: is.ignore(8); break;
            case 5: is.ignore(4); break;
            case 6: is.ignore(8); break;
            case 7: is.ignore(4); break;
            case 8: { std::string s = read_gguf_string(is); (void)s; break; }
            case 9: {
                uint32_t atype = read_le_u32(is);
                uint64_t alen  = read_le_u64(is);
                if (atype == 8) { for (uint64_t j = 0; j < alen; j++) { std::string s = read_gguf_string(is); (void)s; } }
                else is.ignore((std::streamsize)(alen * 4));
                break;
            }
            case 10: is.ignore(8); break;
            case 11: is.ignore(8); break;
            case 12: is.ignore(8); break;
            default: is.ignore(4); break;
        }
    }

    // Read tensor info (GGUF spec: name | n_dims | dims[] | ggml_type | offset).
    struct TensorInfo { std::string name; uint32_t ggml_type; uint64_t offset; int64_t ne; };
    std::vector<TensorInfo> infos;
    infos.reserve((size_t)n_tensors);

    for (uint64_t i = 0; i < n_tensors; i++) {
        TensorInfo ti;
        ti.name = read_gguf_string(is);
        uint32_t n_dims = read_le_u32(is);
        ti.ne = 1;
        for (uint32_t d = 0; d < n_dims; d++) {
            uint64_t dim = read_le_u64(is);
            ti.ne *= (int64_t)dim;
        }
        ti.ggml_type = read_le_u32(is);
        ti.offset = read_le_u64(is);
        if (version >= 3) read_le_u64(is);  // v3 alignment offset
        infos.push_back(ti);
    }

    // Dequantize each tensor.
    std::vector<AdapterTensor> result;
    result.reserve(infos.size());

    for (const auto& ti : infos) {
        AdapterTensor at;
        at.name = ti.name;
        at.shape = { ti.ne };
        at.data.resize((size_t)ti.ne);

        is.seekg((std::streamoff)ti.offset, std::ios::beg);

        auto dequant_block = [&](const uint8_t* block, int64_t idx) {
            switch (ti.ggml_type) {
                case GGML_TYPE_F32: {
                    std::memcpy(&at.data[(size_t)idx], block, 4);
                    break;
                }
                case GGML_TYPE_F16: {
                    uint16_t h; std::memcpy(&h, block, 2);
                    at.data[(size_t)idx] = gguf_fp16_to_f32(h);
                    break;
                }
                case GGML_TYPE_Q4_0:  dequant_q4_0(block,  at.data.data(), idx);  break;
                case GGML_TYPE_Q4_1:  dequant_q4_1(block,  at.data.data(), idx);  break;
                case GGML_TYPE_Q5_0:  dequant_q5_0(block,  at.data.data(), idx);  break;
                case GGML_TYPE_Q5_1:  dequant_q5_1(block,  at.data.data(), idx);  break;
                case GGML_TYPE_Q8_0:  dequant_q8_0(block,  at.data.data(), idx);  break;
                case GGML_TYPE_Q8_1:  dequant_q8_1(block,  at.data.data(), idx);  break;
                case GGML_TYPE_Q6_0:  dequant_q6_0(block,  at.data.data(), idx);  break;
                case GGML_TYPE_Q2_K:  dequant_q2_k(block,  at.data.data(), idx);  break;
                default:
                    if (verbose)
                        std::fprintf(stderr, "  unsupported GGML type %u for %s\n",
                                      ti.ggml_type, ti.name.c_str());
                    break;
            }
        };

        if (ti.ggml_type == GGML_TYPE_F32) {
            is.read(reinterpret_cast<char*>(at.data.data()), (std::streamsize)(ti.ne * 4));
        } else if (ti.ggml_type == GGML_TYPE_F16) {
            for (int64_t j = 0; j < ti.ne; j++) {
                uint16_t h; is.read(reinterpret_cast<char*>(&h), 2);
                at.data[(size_t)j] = gguf_fp16_to_f32(h);
            }
        } else {
            // Block-based dequantization (32 elements per block for all supported types).
            int64_t n_blocks = (ti.ne + 31) / 32;
            for (int64_t b = 0; b < n_blocks; b++) {
                uint8_t block[34];
                int need = 18;
                switch (ti.ggml_type) {
                    case GGML_TYPE_Q4_0: case GGML_TYPE_Q5_0: case GGML_TYPE_Q6_0: case GGML_TYPE_Q2_K: need = 18; break;
                    case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0: need = 34; break;
                    case GGML_TYPE_Q8_1: need = 38; break;
                    default: need = 34; break;
                }
                is.read(reinterpret_cast<char*>(block), need);
                dequant_block(block, b * 32);
            }
        }

        result.push_back(std::move(at));
        if (verbose)
            std::fprintf(stderr, "  tensor %s type=%u ne=%lld\n",
                          ti.name.c_str(), ti.ggml_type, (long long)ti.ne);
    }
    return result;
}

bool gguf_to_oil(const std::string& input_path, const BridgeConfig& cfg) {
    auto tensors = load_gguf(input_path, cfg.verbose);
    if (tensors.empty()) return false;
    BridgeConfig c = cfg;
    return write_oil_mixed(tensors, c);
}

} // namespace adapters
} // namespace oil
