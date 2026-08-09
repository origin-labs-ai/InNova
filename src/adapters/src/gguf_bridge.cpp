// ============================================================================
// gguf_bridge.cpp — GGUF parser + dequantizer -> AdapterTensors -> QUANT mixed
// ============================================================================
#include "adapters/gguf_bridge.h"
#include "adapters/adapter_core.h"

#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>
#include <cstdio>

namespace quant {
namespace adapters {

// ── GGUF type constants ──────────────────────────────────────────────────────
// (ggml_type enum values, see ggml.h / docs/gguf.md)

static const uint32_t GGML_TYPE_F32  = 0;
static const uint32_t GGML_TYPE_F16  = 1;
static const uint32_t GGML_TYPE_Q4_0 = 2;
static const uint32_t GGML_TYPE_Q4_1 = 3;
static const uint32_t GGML_TYPE_Q5_0 = 6;
static const uint32_t GGML_TYPE_Q5_1 = 7;
static const uint32_t GGML_TYPE_Q8_0 = 8;
static const uint32_t GGML_TYPE_Q8_1 = 9;
static const uint32_t GGML_TYPE_Q2_K = 10;
static const uint32_t GGML_TYPE_Q6_K = 14;

// Per-type quantization block layout: {elements per block, bytes per block}.
// Sizes follow ggml-quants.c block structs.
struct TypeLayout { uint32_t elems; uint32_t bytes; };
static TypeLayout type_layout(uint32_t t) {
    switch (t) {
        case GGML_TYPE_Q4_0: return {32, 18};
        case GGML_TYPE_Q4_1: return {32, 20};
        case GGML_TYPE_Q5_0: return {32, 22};
        case GGML_TYPE_Q5_1: return {32, 24};
        case GGML_TYPE_Q8_0: return {32, 34};
        case GGML_TYPE_Q8_1: return {32, 36};
        case GGML_TYPE_Q2_K: return {256, 84};
        case GGML_TYPE_Q6_K: return {256, 210};
        default:             return {0, 0};
    }
}

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
    if (len > (1 << 24)) {            // sanity guard; too long to be a sane key/name
        is.setstate(std::ios::failbit);
        return {};
    }
    std::string s(len, '\0');
    if (len > 0) is.read(&s[0], (std::streamsize)len);
    return s;
}

// ── FP16 helpers ─────────────────────────────────────────────────────────────

static float gguf_fp16_to_f32(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp  = (h >> 10) & 0x1F;
    int mant = h & 0x3FF;
    float v;
    if (exp == 0) {
        v = (float)mant * 5.9604644775390625e-08f;   // mant * 2^-24 (subnormal)
    } else if (exp == 31) {
        v = mant ? NAN : INFINITY;
    } else {
        v = (float)(mant | 0x400) / 1024.0f;         // (1 + mant/1024) * 2^(exp-15)
        v = std::ldexp(v, exp - 15);
    }
    return sign ? -v : v;
}

// ── GGML dequantizers ────────────────────────────────────────────────────────
// Bit layouts copied from llama.cpp ggml-quants.c dequantize_row_* functions.

static void dequant_q4_0(const uint8_t* block, float* out, int64_t offset) {
    float d = gguf_fp16_to_f32(*(const uint16_t*)block);
    for (int j = 0; j < 16; j++) {
        int x0 = (block[2 + j] & 0x0F) - 8;
        int x1 = (block[2 + j] >>   4) - 8;
        out[offset + j      ] = (float)x0 * d;
        out[offset + j + 16 ] = (float)x1 * d;
    }
}

static void dequant_q4_1(const uint8_t* block, float* out, int64_t offset) {
    float d = gguf_fp16_to_f32(*(const uint16_t*)block);
    float m = gguf_fp16_to_f32(*(const uint16_t*)(block + 2));
    for (int j = 0; j < 16; j++) {
        int x0 = block[4 + j] & 0x0F;
        int x1 = block[4 + j] >>   4;
        out[offset + j      ] = (float)x0 * d + m;
        out[offset + j + 16 ] = (float)x1 * d + m;
    }
}

static void dequant_q5_0(const uint8_t* block, float* out, int64_t offset) {
    float d = gguf_fp16_to_f32(*(const uint16_t*)block);
    uint32_t qh; std::memcpy(&qh, block + 2, 4);
    const uint8_t* qs = block + 6;
    for (int j = 0; j < 16; j++) {
        int xh0 = ((qh >> (j +  0)) << 4) & 0x10;
        int xh1 = ((qh >> (j + 12))     ) & 0x10;
        int x0 = ((qs[j] & 0x0F) | xh0) - 16;
        int x1 = ((qs[j] >>   4) | xh1) - 16;
        out[offset + j      ] = (float)x0 * d;
        out[offset + j + 16 ] = (float)x1 * d;
    }
}

static void dequant_q5_1(const uint8_t* block, float* out, int64_t offset) {
    float d = gguf_fp16_to_f32(*(const uint16_t*)block);
    float m = gguf_fp16_to_f32(*(const uint16_t*)(block + 2));
    uint32_t qh; std::memcpy(&qh, block + 4, 4);
    const uint8_t* qs = block + 8;
    for (int j = 0; j < 16; j++) {
        int xh0 = ((qh >> (j +  0)) << 4) & 0x10;
        int xh1 = ((qh >> (j + 12))     ) & 0x10;
        int x0 = (qs[j] & 0x0F) | xh0;
        int x1 = (qs[j] >>   4) | xh1;
        out[offset + j      ] = (float)x0 * d + m;
        out[offset + j + 16 ] = (float)x1 * d + m;
    }
}

static void dequant_q8_0(const uint8_t* block, float* out, int64_t offset) {
    float d = gguf_fp16_to_f32(*(const uint16_t*)block);
    for (int j = 0; j < 32; j++)
        out[offset + j] = (float)(int8_t)block[2 + j] * d;
}

static void dequant_q8_1(const uint8_t* block, float* out, int64_t offset) {
    float d = gguf_fp16_to_f32(*(const uint16_t*)block);
    float s = gguf_fp16_to_f32(*(const uint16_t*)(block + 2));
    for (int j = 0; j < 32; j++)
        out[offset + j] = (float)(int8_t)block[4 + j] * d + s;
}

static void dequant_q2_k(const uint8_t* block, float* out, int64_t offset) {
    // 256 elements per block: d(2) dmin(2) scales[16] qs[64].
    const float d   = gguf_fp16_to_f32(*(const uint16_t*)block);
    const float min = gguf_fp16_to_f32(*(const uint16_t*)(block + 2));
    const uint8_t* scales = block + 4;
    const uint8_t* q      = block + 20;
    int is = 0;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            uint8_t sc = scales[is++];
            float dl = d * (sc & 0xF);
            float ml = min * (sc >> 4);
            for (int l = 0; l < 16; ++l)
                out[offset + n + j*32 + l] = dl * (float)(int8_t)((q[l] >> shift) & 3) - ml;

            sc = scales[is++];
            dl = d * (sc & 0xF);
            ml = min * (sc >> 4);
            for (int l = 0; l < 16; ++l)
                out[offset + n + j*32 + 16 + l] = dl * (float)(int8_t)((q[l + 16] >> shift) & 3) - ml;

            shift += 2;
        }
        q += 32;
    }
}

static void dequant_q6_k(const uint8_t* block, float* out, int64_t offset) {
    // 256 elements per block: d(2) ql[128] qh[64] scales[16] (int8).
    const float d = gguf_fp16_to_f32(*(const uint16_t*)block);
    const uint8_t* ql = block + 2;
    const uint8_t* qh = block + 130;
    const int8_t*  sc = (const int8_t*)(block + 194);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[offset + n + l +  0] = d * sc[is + 0] * q1;
            out[offset + n + l + 32] = d * sc[is + 2] * q2;
            out[offset + n + l + 64] = d * sc[is + 4] * q3;
            out[offset + n + l + 96] = d * sc[is + 6] * q4;
        }
        ql += 64; qh += 32; sc += 8;
    }
}

// ── GGUF parser ──────────────────────────────────────────────────────────────

std::vector<AdapterTensor> load_gguf(const std::string& path, bool verbose) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return {};

    is.seekg(0, std::ios::end);
    const std::streamoff file_size = is.tellg();
    if (file_size < 24) return {};
    is.seekg(0, std::ios::beg);

    // Header: magic(4) | version(u32) | tensor_count(u64) | kv_count(u64).
    // No version-dependent fields exist in the header (GGUF spec).
    char magic[4];
    is.read(magic, 4);
    if (!is.good() || std::memcmp(magic, "GGUF", 4) != 0) return {};

    uint32_t version = read_le_u32(is);
    if (version < 2 || version > 3) return {};   // v1 unsupported upstream; v3 current

    uint64_t n_tensors = read_le_u64(is);
    uint64_t n_kv      = read_le_u64(is);
    if (!is.good()) return {};

    // Corrupt-header guards: every tensor/kv entry needs at least ~8 bytes.
    if (n_tensors > (uint64_t)file_size / 8 + 1) return {};
    if (n_kv      > (uint64_t)file_size / 8 + 1) return {};

    if (verbose)
        std::fprintf(stderr, "GGUF v%u tensors=%llu metadata=%llu\n",
                      version, (unsigned long long)n_tensors, (unsigned long long)n_kv);

    uint32_t alignment = 32;   // GGUF default; overridden by general.alignment

    // Skip metadata (key-value pairs); capture general.alignment.
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = read_gguf_string(is);
        if (!is.good()) return {};
        uint32_t vt = read_le_u32(is);
        if (!is.good()) return {};
        switch (vt) {
            case 0: case 1: case 7: is.ignore(1); break;          // u8 / i8 / bool
            case 2: case 3: is.ignore(2); break;                  // u16 / i16
            case 4:                                               // u32 (general.alignment)
                if (key == "general.alignment") {
                    uint32_t a = read_le_u32(is);
                    if (a != 0 && (a & (a - 1)) == 0) alignment = a;
                } else {
                    is.ignore(4);
                }
                break;
            case 5: case 6: is.ignore(4); break;                  // i32 / f32
            case 8: { std::string s = read_gguf_string(is); (void)s; break; }
            case 9: {                                             // array
                uint32_t atype = read_le_u32(is);
                uint64_t alen  = read_le_u64(is);
                if (atype == 8) {
                    for (uint64_t j = 0; j < alen; j++) {
                        std::string s = read_gguf_string(is);
                        (void)s;
                        if (!is.good()) return {};
                    }
                } else {
                    size_t elem = 4;
                    switch (atype) {
                        case 0: case 1: case 7: elem = 1; break;
                        case 2: case 3: elem = 2; break;
                        case 4: case 5: case 6: elem = 4; break;
                        case 10: case 11: case 12: elem = 8; break;
                        default: elem = 4; break;
                    }
                    if (alen != 0 && alen > std::numeric_limits<uint64_t>::max() / elem) return {};
                    is.ignore((std::streamsize)(alen * elem));
                }
                break;
            }
            case 10: case 11: case 12: is.ignore(8); break;       // u64 / i64 / f64
            default: is.ignore(4); break;
        }
        if (!is.good()) return {};
    }

    // Tensor info: name | n_dims | dims[] | ggml_type | offset.
    // No version-dependent fields exist in tensor info (GGUF spec).
    struct TensorInfo { std::string name; uint32_t ggml_type; uint64_t offset; int64_t ne; };
    std::vector<TensorInfo> infos;
    infos.reserve((size_t)n_tensors);

    for (uint64_t i = 0; i < n_tensors; i++) {
        TensorInfo ti;
        ti.name = read_gguf_string(is);
        if (!is.good() || ti.name.empty()) return {};
        uint32_t n_dims = read_le_u32(is);
        if (n_dims > 4) return {};
        ti.ne = 1;
        for (uint32_t d = 0; d < n_dims; d++) {
            uint64_t dim = read_le_u64(is);
            if (dim != 0 && ti.ne > INT64_MAX / (int64_t)dim) return {};
            ti.ne *= (int64_t)dim;
        }
        ti.ggml_type = read_le_u32(is);
        ti.offset = read_le_u64(is);
        if (!is.good()) return {};
        infos.push_back(ti);
    }

    // The tensor data section starts at the aligned end of the tensor-info list.
    std::streamoff after_info = is.tellg();
    if (after_info < 0) return {};
    uint64_t data_start = (uint64_t)after_info;
    uint64_t rem = data_start % alignment;
    if (rem) data_start += alignment - rem;
    if (data_start > (uint64_t)file_size) return {};

    // Dequantize each tensor. Tensor offsets are relative to `data_start`.
    std::vector<AdapterTensor> result;
    result.reserve(infos.size());

    for (const auto& ti : infos) {
        const TypeLayout lay = type_layout(ti.ggml_type);
        AdapterTensor at;
        at.name = ti.name;
        at.shape = { ti.ne };
        at.data.resize((size_t)ti.ne);

        if (ti.ggml_type != GGML_TYPE_F32 && ti.ggml_type != GGML_TYPE_F16 && lay.elems == 0) {
            if (verbose)
                std::fprintf(stderr, "  unsupported GGML type %u for %s\n",
                              ti.ggml_type, ti.name.c_str());
            result.push_back(std::move(at));
            continue;
        }

        // Validate the tensor's byte range before reading.
        uint64_t abs_off = data_start + ti.offset;
        uint64_t nbytes;
        if (ti.ggml_type == GGML_TYPE_F32) {
            nbytes = (uint64_t)ti.ne * 4;
        } else if (ti.ggml_type == GGML_TYPE_F16) {
            nbytes = (uint64_t)ti.ne * 2;
        } else {
            if (ti.ne % lay.elems != 0) {   // invalid per GGUF spec (ne[0] % block_size == 0)
                if (verbose)
                    std::fprintf(stderr, "  tensor %s ne=%lld not divisible by block size %u\n",
                                  ti.name.c_str(), (long long)ti.ne, lay.elems);
                result.push_back(std::move(at));
                continue;
            }
            nbytes = ((uint64_t)ti.ne / lay.elems) * lay.bytes;
        }
        if (abs_off > (uint64_t)file_size || nbytes > (uint64_t)file_size - abs_off) {
            if (verbose)
                std::fprintf(stderr, "  tensor %s offset %llu out of range\n",
                              ti.name.c_str(), (unsigned long long)ti.offset);
            result.push_back(std::move(at));
            continue;
        }

        is.clear();
        is.seekg((std::streamoff)abs_off, std::ios::beg);
        if (!is.good()) {
            result.push_back(std::move(at));
            continue;
        }

        if (ti.ggml_type == GGML_TYPE_F32) {
            is.read(reinterpret_cast<char*>(at.data.data()), (std::streamsize)(ti.ne * 4));
        } else if (ti.ggml_type == GGML_TYPE_F16) {
            for (int64_t j = 0; j < ti.ne; j++) {
                uint16_t h; is.read(reinterpret_cast<char*>(&h), 2);
                if (is.fail()) break;   // eofbit alone (exact-EOF read) is not an error
                at.data[(size_t)j] = gguf_fp16_to_f32(h);
            }
        } else {
            std::vector<uint8_t> block(lay.bytes);
            int64_t n_blocks = ti.ne / lay.elems;
            for (int64_t b = 0; b < n_blocks; b++) {
                is.read(reinterpret_cast<char*>(block.data()), (std::streamsize)lay.bytes);
                if (is.fail()) break;   // eofbit alone (exact-EOF read) is not an error
                const int64_t base = b * lay.elems;
                switch (ti.ggml_type) {
                    case GGML_TYPE_Q4_0: dequant_q4_0(block.data(), at.data.data(), base); break;
                    case GGML_TYPE_Q4_1: dequant_q4_1(block.data(), at.data.data(), base); break;
                    case GGML_TYPE_Q5_0: dequant_q5_0(block.data(), at.data.data(), base); break;
                    case GGML_TYPE_Q5_1: dequant_q5_1(block.data(), at.data.data(), base); break;
                    case GGML_TYPE_Q8_0: dequant_q8_0(block.data(), at.data.data(), base); break;
                    case GGML_TYPE_Q8_1: dequant_q8_1(block.data(), at.data.data(), base); break;
                    case GGML_TYPE_Q2_K: dequant_q2_k(block.data(), at.data.data(), base); break;
                    case GGML_TYPE_Q6_K: dequant_q6_k(block.data(), at.data.data(), base); break;
                    default: break;
                }
            }
        }

        result.push_back(std::move(at));
        if (verbose)
            std::fprintf(stderr, "  tensor %s type=%u ne=%lld\n",
                          ti.name.c_str(), ti.ggml_type, (long long)ti.ne);
    }
    return result;
}

bool gguf_to_quant(const std::string& input_path, const BridgeConfig& cfg) {
    auto tensors = load_gguf(input_path, cfg.verbose);
    if (tensors.empty()) return false;
    BridgeConfig c = cfg;
    return write_quant_mixed(tensors, c);
}

} // namespace adapters
} // namespace quant
