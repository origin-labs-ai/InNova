// ============================================================================
// adapter_core.cpp — OIL mixed-precision funnel + foreign dtype dequantization
// ============================================================================
#include "adapters/adapter_core.h"
#include "oil/oil_format.h"
#include "oil/codebook.h"
#include "oil/format_planner.h"
#include "oil/types.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cstdint>

namespace oil {
namespace adapters {

// ── Foreign dtype -> FP32 dequantization ────────────────────────────────────

float fp16_to_float(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp  = (h >> 10) & 0x1F;
    int mant = h & 0x3FF;
    float f;
    if (exp == 0) {
        f = (float)mant / 16384.0f * 2.0f;       // subnormal
    } else if (exp == 31) {
        f = mant ? NAN : INFINITY;
    } else {
        f = (float)((mant | 0x400) << 13) / 8388608.0f * (float)(1 << (exp - 15));
    }
    return sign ? -f : f;
}

float bf16_to_float(uint16_t b) {
    // bfloat16 = top 16 bits of FP32 (8 exp, 7 mant)
    uint32_t bits = ((uint32_t)b) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

float fp8_e4m3_to_float(uint8_t x) {
    // E4M3: 1 sign | 4 exponent (bias 7) | 3 mantissa
    int sign = (x >> 7) & 1;
    int exp  = (x >> 3) & 0xF;
    int mant = x & 0x7;
    float f;
    if (exp == 0 && mant == 0) {
        f = 0.0f;                                // zero
    } else if (exp == 15) {
        f = mant ? NAN : INFINITY;               // 15 = NaN/Inf in E4M3 (no Inf really, but guard)
    } else if (exp == 0) {
        f = (float)mant / 64.0f * 2.0f;          // subnormal
    } else {
        f = (float)((mant | 0x8) << (3 + 20)) / 8388608.0f * (float)(1 << (exp - 7));
    }
    return sign ? -f : f;
}

float fp8_e5m2_to_float(uint8_t x) {
    // E5M2: 1 sign | 5 exponent (bias 15) | 2 mantissa
    int sign = (x >> 7) & 1;
    int exp  = (x >> 2) & 0x1F;
    int mant = x & 0x3;
    float f;
    if (exp == 0 && mant == 0) {
        f = 0.0f;
    } else if (exp == 31) {
        f = mant ? NAN : INFINITY;
    } else if (exp == 0) {
        f = (float)mant / 256.0f * 2.0f;        // subnormal
    } else {
        f = (float)((mant | 0x4) << (2 + 20)) / 8388608.0f * (float)(1 << (exp - 15));
    }
    return sign ? -f : f;
}

// ── Format detection ────────────────────────────────────────────────────────

ExternalFormat detect_format(const std::string& path) {
    // Read first 8 magic bytes.
    std::ifstream f(path, std::ios::binary);
    uint8_t magic[8] = {0};
    std::streamsize got = 0;
    if (f) {
        f.read(reinterpret_cast<char*>(magic), 8);
        got = f.gcount();
    }

    // GGUF: "GGUF" (4 bytes) + version u32
    if (got >= 4 && std::memcmp(magic, "GGUF", 4) == 0)
        return ExternalFormat::GGUF;

    // Safetensors: 8-byte LE header length, then JSON starting with '{'
    if (got >= 8) {
        uint64_t hdr_len = 0;
        std::memcpy(&hdr_len, magic, 8);
        if (hdr_len > 0 && hdr_len < (1ULL << 30)) {
            std::vector<char> hdr((size_t)hdr_len);
            if (f) f.read(hdr.data(), (std::streamsize)hdr_len);
            if (f && f.gcount() == (std::streamsize)hdr_len && hdr_len > 0 && hdr[0] == '{')
                return ExternalFormat::SAFETENSORS;
        }
    }

    // OIL: "OIL1"
    if (got >= 4 && std::memcmp(magic, "OIL1", 4) == 0)
        return ExternalFormat::OIL;

    // Fallback: extension-based detection (works even for non-existent files).
    auto ends_with = [](const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() &&
               s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    };
    if (ends_with(path, ".fp8e4m3") || ends_with(path, ".fp8_e4m3"))
        return ExternalFormat::RAW_FP8_E4M3;
    if (ends_with(path, ".fp8e5m2") || ends_with(path, ".fp8_e5m2"))
        return ExternalFormat::RAW_FP8_E5M2;
    if (ends_with(path, ".fp16") || ends_with(path, ".f16") || ends_with(path, ".half"))
        return ExternalFormat::RAW_FP16;
    if (ends_with(path, ".gguf"))
        return ExternalFormat::GGUF;
    if (ends_with(path, ".safetensors"))
        return ExternalFormat::SAFETENSORS;
    if (ends_with(path, ".oil"))
        return ExternalFormat::OIL;
    if (ends_with(path, ".fp32") || ends_with(path, ".f32") || ends_with(path, ".bin") ||
        ends_with(path, ".raw") || ends_with(path, ".weights"))
        return ExternalFormat::RAW_FP32;

    return ExternalFormat::UNKNOWN;
}

const char* external_format_name(ExternalFormat f) {
    switch (f) {
        case ExternalFormat::RAW_FP32:     return "raw_fp32";
        case ExternalFormat::RAW_FP16:     return "raw_fp16";
        case ExternalFormat::RAW_FP8_E4M3: return "raw_fp8_e4m3";
        case ExternalFormat::RAW_FP8_E5M2: return "raw_fp8_e5m2";
        case ExternalFormat::GGUF:         return "gguf";
        case ExternalFormat::SAFETENSORS:   return "safetensors";
        case ExternalFormat::OIL:          return "oil";
        default:                           return "unknown";
    }
}

// ── Per-block mixed-precision quantization ───────────────────────────────────
//
// For a block of `n` FP32 weights and an assigned Format, produce the codebook
// (if any) and the packed index bytes exactly as the OIL reader expects them.

static float block_max_abs(const float* w, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) m = std::max(m, std::fabs(w[i]));
    return m;
}

static void quant_oil1(const float* w, int n,
                         std::vector<uint8_t>& indices, std::vector<uint8_t>& codebook) {
    float scale = block_max_abs(w, n);
    if (scale < 1e-12f) scale = 1.0f;
    indices.assign((size_t)(n + 7) / 8, 0);
    for (int i = 0; i < n; i++)
        if (w[i] >= 0.0f) indices[(size_t)i / 8] |= (uint8_t)(1 << (i % 8));
    // codebook stores {-1.0f, +1.0f} as FP32 so a reader can reconstruct scale*cv
    codebook.resize(2 * sizeof(float));
    float neg = -1.0f, pos = 1.0f;
    std::memcpy(codebook.data(), &neg, 4);
    std::memcpy(codebook.data() + 4, &pos, 4);
    (void)scale;
}

static void quant_spark(const float* w, int n,
                          std::vector<uint8_t>& indices, std::vector<uint8_t>& codebook) {
    float scale = block_max_abs(w, n);
    if (scale < 1e-12f) scale = 1.0f;
    // 2 bits per weight, pack 4 per byte; values map -1->0, 0->1, +1->2
    indices.assign((size_t)(n + 3) / 4, 0);
    for (int i = 0; i < n; i++) {
        float norm = w[i] / scale;
        uint8_t q = (norm > 0.5f) ? 2 : (norm < -0.5f ? 0 : 1);
        indices[(size_t)i / 4] |= (uint8_t)(q << ((i % 4) * 2));
    }
    // codebook stores {-1, 0, +1} as FP32
    codebook.resize(3 * sizeof(float));
    float v[3] = {-1.0f, 0.0f, 1.0f};
    std::memcpy(codebook.data(), v, sizeof(v));
    (void)scale;
}

static void quant_oil4(const float* w, int n,
                       std::vector<uint8_t>& indices, std::vector<uint8_t>& codebook) {
    CodebookOIL4 cb;
    cb.train(w, (size_t)n);
    codebook.resize(cb.serialized_size());
    cb.serialize(codebook.data());
    indices.assign((size_t)(n + 1) / 2, 0);
    for (int i = 0; i < n; i++) {
        uint8_t idx = cb.quantize(w[i]);
        if (i % 2 == 0) indices[(size_t)i / 2] = (uint8_t)(idx & 0x0F);
        else            indices[(size_t)i / 2] |= (uint8_t)(idx << 4);
    }
}

static void quant_oil8(const float* w, int n,
                        std::vector<uint8_t>& indices, std::vector<uint8_t>& codebook) {
    CodebookOIL8 cb;
    cb.train(w, (size_t)n);
    codebook.resize(cb.serialized_size());
    cb.serialize(codebook.data());
    indices.assign((size_t)n, 0);
    for (int i = 0; i < n; i++) indices[(size_t)i] = cb.quantize(w[i]);
}

static void quant_fp16(const float* w, int n,
                        std::vector<uint8_t>& indices, std::vector<uint8_t>& codebook) {
    indices.assign((size_t)n * 2, 0);
    for (int i = 0; i < n; i++) {
        // FP32 -> FP16 (round-to-nearest, ties to even)
        uint32_t bits; std::memcpy(&bits, &w[i], 4);
        int sign = (bits >> 31) & 1;
        int exp  = (int)((bits >> 23) & 0xFF) - 127;   // unbiased
        int mant = bits & 0x7FFFFF;
        uint16_t h;
        if (exp > 15) {                                // overflow -> Inf
            h = (uint16_t)((sign << 15) | 0x7C00);
        } else if (exp >= -14) {                        // normal
            int m = (mant >> 13) | 0x400;
            // round to nearest, ties to even
            int round_bit = (mant >> 12) & 1;
            int sticky = (mant & 0xFFF) ? 1 : 0;
            int lsb = m & 1;
            if (round_bit && (sticky || lsb)) m += 1;
            if (m >= 0x800) { m >>= 1; exp += 1; if (exp > 15) { h = (uint16_t)((sign << 15) | 0x7C00); continue; } }
            h = (uint16_t)((sign << 15) | ((exp + 15) << 10) | (m & 0x3FF));
        } else {                                        // subnormal or zero
            if (exp < -24) {
                h = (uint16_t)(sign << 15);
            } else {
                int m = mant | 0x800000;
                int shift = -exp - 14 + 13;
                int mf = m >> shift;
                int round_bit = (m >> (shift - 1)) & 1;
                int sticky = ((m & ((1 << (shift - 1)) - 1)) ? 1 : 0);
                int lsb = mf & 1;
                if (round_bit && (sticky || lsb)) mf += 1;
                if (mf >= 0x400) {
                    h = (uint16_t)((sign << 15) | (1 << 10) | (mf & 0x3FF)); // became normal
                } else {
                    h = (uint16_t)((sign << 15) | mf);
                }
            }
        }
        std::memcpy(&indices[(size_t)i * 2], &h, 2);
    }
    codebook.clear();
}

static void quant_fp32(const float* w, int n,
                        std::vector<uint8_t>& indices, std::vector<uint8_t>& codebook) {
    indices.assign((size_t)n * 4, 0);
    std::memcpy(indices.data(), w, (size_t)n * 4);
    codebook.clear();
}

// Quantize one block to `fmt`, filling codebook + indices.
static void quant_block(Format fmt, const float* w, int n,
                        std::vector<uint8_t>& indices, std::vector<uint8_t>& codebook) {
    switch (fmt) {
        case Format::OIL1:  quant_oil1(w, n, indices, codebook);  break;
        case Format::SPARK_Q0:  quant_spark(w, n, indices, codebook);  break;
        case Format::OIL4:     quant_oil4(w, n, indices, codebook);     break;
        case Format::OIL8:     quant_oil8(w, n, indices, codebook);     break;
        case Format::OIL16:     quant_fp16(w, n, indices, codebook);     break;
        case Format::OIL32:
        default:               quant_fp32(w, n, indices, codebook);    break;
    }
}

// Score block importance by mean |weight| (no calibration data available for PTQ).
static float block_importance(const float* w, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += std::fabs((double)w[i]);
    return (float)(s / (double)n);
}

// Adaptive importance: weight magnitude × non-linearity proxy for activation sensitivity.
// Without real calibration activations, we use |w|^p * max(|w|, eps) as a proxy that
// up-weights blocks with both high mean magnitude and high peak magnitude.
static float block_importance_adaptive(const float* w, int n) {
    double s = 0.0;
    double peak = 0.0;
    for (int i = 0; i < n; i++) {
        double a = std::fabs((double)w[i]);
        s += a;
        if (a > peak) peak = a;
    }
    double mean = s / (double)n;
    double eps = 1e-3;
    // Proxy: mean^0.7 * max(peak, eps) — up-weights salient blocks
    return (float)(std::pow(mean, 0.7) * std::max(peak, eps));
}

float estimate_mixed_bpw(int64_t num_weights, int block_size, float target_bpw) {
    (void)target_bpw;
    int num_blocks = (int)((num_weights + block_size - 1) / block_size);
    if (num_blocks == 0) return 0.0f;
    int n_oil8 = std::max(1, (int)((double)num_blocks * 0.01 + 0.5));
    int n_oil4 = std::max(1, (int)((double)num_blocks * 0.04 + 0.5));
    int n_spark = num_blocks - n_oil8 - n_oil4;
    if (n_spark < 1) { n_spark = 1; n_oil4 = std::max(0, n_oil4 - 1); }
    float total = (float)(n_oil8 * 8 + n_oil4 * 4 + n_spark * 1.5f);
    return total / (float)num_blocks;
}

// ── The funnel: AdapterTensors (FP32) -> OIL mixed-precision .oil file ──────
//
// OIL on-disk layout: HEADER | FORMAT_TABLE | TENSOR_TABLE | BLOCK_DATA
// Blocks MUST be written last so that OILReader can compute data_offset_
// correctly from the sizes of the preceding tables.

bool write_oil_mixed(const std::vector<AdapterTensor>& tensors,
                     const BridgeConfig& cfg) {
    if (cfg.output_path.empty()) return false;

    OILWriter writer(cfg.output_path);
    OILHeader hdr;
    std::memcpy(hdr.magic, "OIL1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = 0;
    writer.write_header(hdr, nullptr);

    const int bs = cfg.block_size > 0 ? cfg.block_size : 256;
    float target = cfg.target_bpw;
    if (target < 1.0f) target = 1.0f;
    if (target > 32.0f) target = 32.0f;

    std::vector<FormatBlockEntry> ft_entries;
    std::vector<TensorEntry> tensor_entries;
    std::vector<std::string> names;
    std::vector<BlockData> all_blocks;
    all_blocks.reserve(tensors.size() * 4);
    uint32_t block_id = 0;
    double total_bpw_weighted = 0.0;
    int64_t total_blocks = 0;

    for (const auto& t : tensors) {
        names.push_back(t.name);
        int64_t numel = (int64_t)t.data.size();
        if (numel == 0) {
            TensorEntry te; te.name_len = 0; te.block_start = block_id; te.num_blocks = 0;
            tensor_entries.push_back(te);
            continue;
        }

        int num_blocks = (int)((numel + bs - 1) / bs);
        if (num_blocks == 0) continue;

        std::vector<std::pair<float, int>> imp(num_blocks);
        for (int b = 0; b < num_blocks; b++) {
            int start = b * bs;
            int n = (int)std::min<int64_t>(bs, numel - start);
            if (cfg.adaptive)
                imp[b] = { block_importance_adaptive(t.data.data() + start, n), b };
            else
                imp[b] = { block_importance(t.data.data() + start, n), b };
        }
        std::sort(imp.begin(), imp.end(),
                  [](const std::pair<float,int>& a, const std::pair<float,int>& b) {
                      return a.first > b.first;
                  });

        int n_oil8 = std::max(1, (int)((double)num_blocks * 0.01 + 0.5));
        int n_oil4 = std::max(1, (int)((double)num_blocks * 0.04 + 0.5));
        int n_tern = num_blocks - n_oil8 - n_oil4;
        if (n_tern < 1) { n_tern = 1; n_oil4 = std::max(0, n_oil4 - 1); }

        std::vector<Format> fmt_of(num_blocks, Format::SPARK_Q0);
        int rank = 0;
        for (int i = 0; i < n_oil8 && rank < num_blocks; i++, rank++)
            fmt_of[imp[rank].second] = Format::OIL8;
        for (int i = 0; i < n_oil4 && rank < num_blocks; i++, rank++)
            fmt_of[imp[rank].second] = Format::OIL4;
        for (int i = rank; i < num_blocks; i++)
            fmt_of[imp[i].second] = Format::SPARK_Q0;

        uint32_t block_start = block_id;

        for (int b = 0; b < num_blocks; b++) {
            int start = b * bs;
            int n = (int)std::min<int64_t>(bs, numel - start);
            Format fmt = fmt_of[b];

            BlockData block;
            block.format = fmt;
            block.num_weights = (uint32_t)n;
            quant_block(fmt, t.data.data() + start, n, block.indices, block.codebook);
            all_blocks.push_back(std::move(block));

            FormatBlockEntry fe;
            fe.block_id = block_id++;
            fe.format = (uint8_t)fmt;
            fe.cb_bytes = (uint32_t)all_blocks.back().codebook.size();
            ft_entries.push_back(fe);

            total_bpw_weighted += (double)format_bpw(fmt);
            total_blocks++;
        }

        TensorEntry te;
        te.name_len = (uint16_t)t.name.size();
        te.block_start = block_start;
        te.num_blocks = (uint32_t)num_blocks;
        tensor_entries.push_back(te);

        if (cfg.verbose) {
            std::printf("  %-48s blocks=%-5d bpw~%.2f raw=%lldKB\n",
                        t.name.c_str(), num_blocks,
                        estimate_mixed_bpw(numel, bs, target),
                        (long long)((numel * 4) / 1024));
        }
    }

    writer.write_format_table(ft_entries);
    writer.write_tensor_table(tensor_entries, names);
    for (auto& blk : all_blocks) writer.write_block(blk);
    writer.close();

    if (cfg.verbose && total_blocks > 0) {
        std::printf("  achieved avg bpw = %.3f across %lld blocks\n",
                    total_bpw_weighted / (double)total_blocks, (long long)total_blocks);
    }
    return true;
}

} // namespace adapters
} // namespace oil
