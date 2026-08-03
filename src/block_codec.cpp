#include "oil/block_codec.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <array>

namespace oil {

namespace {

constexpr int kMeanBlock = 32;      // OIL1 block-mean window
constexpr int kSparkSubBlock = 32;  // SPARK_Q0 scale window

constexpr std::array<float, 4> kOIL2Levels = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f,
};

constexpr std::array<float, 16> kOIL4Levels = {
    -2.7178f, -2.0522f, -1.5995f, -1.2392f,
    -0.9275f, -0.6508f, -0.3976f, -0.1260f,
     0.1260f,  0.3976f,  0.6508f,  0.9275f,
     1.2392f,  1.5995f,  2.0522f,  2.7178f,
};

// ---- FP16 conversion (self-contained, no math dependency) ------------------
uint16_t f32_to_f16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    const int sign = (int)(bits >> 31) & 1;
    int exp = (int)((bits >> 23) & 0xFF) - 127;
    int mant = (int)(bits & 0x7FFFFF);
    uint16_t h;
    if (exp > 15) {
        h = (uint16_t)((sign << 15) | 0x7C00);
    } else if (exp >= -14) {
        int m = (mant >> 13) | 0x400;
        const int round_bit = (mant >> 12) & 1;
        const int sticky = (mant & 0xFFF) ? 1 : 0;
        const int lsb = m & 1;
        if (round_bit && (sticky || lsb)) m += 1;
        if (m >= 0x800) { m >>= 1; exp += 1; if (exp > 15) { h = (uint16_t)((sign << 15) | 0x7C00); return h; } }
        h = (uint16_t)((sign << 15) | ((exp + 15) << 10) | (m & 0x3FF));
    } else {
        if (exp < -24) {
            h = (uint16_t)(sign << 15);
        } else {
            int m = mant | 0x800000;
            const int shift = -exp - 14 + 13;
            int mf = m >> shift;
            const int round_bit = (m >> (shift - 1)) & 1;
            const int sticky = ((m & ((1 << (shift - 1)) - 1)) ? 1 : 0);
            const int lsb = mf & 1;
            if (round_bit && (sticky || lsb)) mf += 1;
            if (mf >= 0x400) h = (uint16_t)((sign << 15) | (1 << 10) | (mf & 0x3FF));
            else h = (uint16_t)((sign << 15) | mf);
        }
    }
    return h;
}

float f16_to_f32(uint16_t h) {
    const int sign = (h >> 15) & 1;
    const int exp = (h >> 10) & 0x1F;
    const int mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = (uint32_t)sign << 31; }
        else {
            // subnormal: normalize
            int e = -14;
            int m = mant;
            while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3FF;
            bits = ((uint32_t)sign << 31) | (uint32_t)(e + 127) << 23 | ((uint32_t)m << 13);
        }
    } else if (exp == 31) {
        bits = ((uint32_t)sign << 31) | 0x7F800000 | ((uint32_t)mant << 13);
    } else {
        bits = ((uint32_t)sign << 31) | (uint32_t)(exp - 15 + 127) << 23 | ((uint32_t)mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// ---- LSB-first bit packing --------------------------------------------------
struct BitWriter {
    std::vector<uint8_t>& bytes;
    size_t bit = 0;
    explicit BitWriter(std::vector<uint8_t>& b) : bytes(b) {}

    void put(uint32_t value, int count) {
        for (int i = 0; i < count; ++i, ++bit) {
            if (bit >= bytes.size() * 8) return;
            if ((value & (1u << i)) != 0)
                bytes[bit / 8] |= (uint8_t)(1u << (bit % 8));
        }
    }
    size_t bits() const { return bit; }
};

struct BitReader {
    const uint8_t* bytes;
    size_t size; // in bytes
    size_t bit = 0;
    BitReader(const uint8_t* b, size_t s) : bytes(b), size(s) {}

    uint32_t get(int count) {
        uint32_t result = 0;
        for (int i = 0; i < count; ++i, ++bit) {
            if (bit >= size * 8) break;
            if ((bytes[bit / 8] & (uint8_t)(1u << (bit % 8))) != 0)
                result |= 1u << i;
        }
        return result;
    }
    size_t bits() const { return bit; }
};

// Slot positions fund the in-budget scale: `slot_count` weights per group are
// zeroed on decode and skipped on encode.
bool is_slot(int64_t local_index, int64_t group_size, int slot_count) {
    if (slot_count <= 0 || group_size <= 0) return false;
    return ((local_index + 1) * slot_count) / group_size !=
           (local_index * slot_count) / group_size;
}

float level_value(int bits, uint32_t index) {
    if (bits == 2) return kOIL2Levels[std::min<size_t>(index, kOIL2Levels.size() - 1)];
    if (bits == 4) return kOIL4Levels[std::min<size_t>(index, kOIL4Levels.size() - 1)];
    return -8.0f + 16.0f * (float)std::min<uint32_t>(index, 255u) / 255.0f;
}

uint32_t nearest_level(float value, int bits) {
    if (bits == 8) {
        const float clamped = std::max(-8.0f, std::min(8.0f, value));
        return (uint32_t)std::lround((clamped + 8.0f) * 255.0f / 16.0f);
    }
    const int count = bits == 2 ? 4 : 16;
    uint32_t best = 0;
    float best_distance = std::fabs(value - level_value(bits, 0));
    for (int i = 1; i < count; ++i) {
        const float distance = std::fabs(value - level_value(bits, (uint32_t)i));
        if (distance < best_distance) { best_distance = distance; best = (uint32_t)i; }
    }
    return best;
}

float rms_scale(const float* data, int n) {
    if (n <= 0) return 0.0f;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double)data[i] * data[i];
    return (float)std::sqrt(sum / (double)n);
}

// ---- lattice: scale + slots + fixed-level indices ---------------------------
void quant_lattice(Format fmt, const float* w, int n, int bits,
                   std::vector<uint8_t>& indices) {
    const int slots = 16 / bits; // 16-bit scale funded by these slots
    const size_t total_bits = (size_t)bits * (size_t)n;
    indices.assign((total_bits + 7) / 8, 0);
    BitWriter bw(indices);
    const float scale = rms_scale(w, n);
    bw.put(f32_to_f16(scale), 16);
    const float s = (scale <= 1e-12f) ? 0.0f : scale;
    for (int i = 0; i < n; ++i) {
        if (is_slot(i, n, slots)) continue;
        const float normalized = s == 0.0f ? 0.0f : w[i] / s;
        bw.put(nearest_level(normalized, bits), bits);
    }
}

void dequant_lattice(const uint8_t* bytes, size_t size, int n, int bits,
                     float* out) {
    const int slots = 16 / bits;
    BitReader br(bytes, size);
    const float scale = f16_to_f32((uint16_t)br.get(16));
    for (int i = 0; i < n; ++i) {
        if (is_slot(i, n, slots)) { out[i] = 0.0f; continue; }
        out[i] = level_value(bits, br.get(bits)) * scale;
    }
}

// ---- grouped lattice: REAL per-group scale + zero-point --------------------
// Every GRP format below 16 bits stores per-64-weight groups, each carrying a
// FP16 zero-point + FP16 scale followed by `bits` lattice indices per weight.
// The overhead (32 bits per 64 weights = 0.5 BPW) is part of the honest BPW
// claim, so a stored block never exceeds format_bpw().
constexpr int kGrpSize = 64;  // GRP per-group window

void quant_grouped_lattice(int bits, const float* w, int n,
                           std::vector<uint8_t>& indices) {
    // Tail policy so the honest per-block claim (ceil(bpw*n/8) + 1 byte) is
    // never exceeded:
    //   * full 64-weight groups: FP16 zero-point + FP16 scale + indices;
    //   * a lone partial group (n < 64): scale-only (16 bits) when n >= 8,
    //     no header at all when n < 8 (implicit scale=1, zp=0);
    //   * a partial tail (n >= 64): reuses the previous group's scale/zp.
    // 8-bit groups use min/max (range) normalization so all 256 levels are
    // used (zp = min, scale = max-min); 2/4-bit groups use the fixed
    // Gaussian lattice with RMS scale (zp = mean).
    const int nfull = n / kGrpSize;
    const size_t hdr_bits = (nfull == 0) ? ((n < 8) ? 0 : 16) : (size_t)nfull * 32;
    const size_t total_bits = hdr_bits + (size_t)n * (size_t)bits;
    indices.assign((total_bits + 7) / 8, 0);
    BitWriter bw(indices);
    float prev_zp = 0.0f, prev_scale = 1.0f;
    const bool is8 = (bits == 8);
    for (int g = 0; g < nfull; ++g) {
        const int start = g * kGrpSize;
        float zp, scale;
        if (is8) {
            float mn = w[start], mx = w[start];
            for (int i = start + 1; i < start + kGrpSize; ++i) {
                if (w[i] < mn) mn = w[i];
                if (w[i] > mx) mx = w[i];
            }
            zp = mn;
            scale = mx - mn;
        } else {
            double mu = 0.0, ss = 0.0;
            for (int i = start; i < start + kGrpSize; ++i) mu += w[i];
            mu /= (double)kGrpSize;
            for (int i = start; i < start + kGrpSize; ++i) {
                const double d = (double)w[i] - mu;
                ss += d * d;
            }
            zp = (float)mu;
            scale = (ss > 0.0) ? (float)std::sqrt(ss / (double)kGrpSize) : 0.0f;
        }
        bw.put(f32_to_f16(zp), 16);
        bw.put(f32_to_f16(scale), 16);
        prev_zp = zp;
        prev_scale = scale;
        if (scale == 0.0f) {
            for (int i = 0; i < kGrpSize; ++i) bw.put(0, bits);
            continue;
        }
        if (is8) {
            for (int i = start; i < start + kGrpSize; ++i) {
                float t = (w[i] - zp) / scale;           // 0..1
                uint32_t idx = (uint32_t)std::lround(t * 255.0f);
                bw.put(idx > 255 ? 255u : idx, 8);
            }
        } else {
            for (int i = start; i < start + kGrpSize; ++i)
                bw.put(nearest_level((float)(((double)w[i] - zp) / (double)scale), bits), bits);
        }
    }
    const int tail_start = nfull * kGrpSize;
    const int tail_count = n - tail_start;
    if (tail_count > 0) {
        if (nfull == 0) {
            // Lone partial group: the whole block is smaller than one group.
            if (n < 8) {
                if (is8) {
                    for (int i = 0; i < n; ++i) {
                        uint32_t idx = (uint32_t)std::lround((w[i] + 1.0f) * 127.5f);
                        bw.put(idx > 255 ? 255u : idx, 8);
                    }
                } else {
                    for (int i = 0; i < n; ++i) bw.put(nearest_level(w[i], bits), bits);
                }
            } else {
                float scale;
                if (is8) {
                    float mx = std::fabs(w[0]);
                    for (int i = 1; i < n; ++i) mx = std::max(mx, std::fabs(w[i]));
                    scale = mx;
                } else {
                    double ss = 0.0;
                    for (int i = 0; i < n; ++i) ss += (double)w[i] * (double)w[i];
                    scale = (ss > 0.0) ? (float)std::sqrt(ss / (double)n) : 0.0f;
                }
                bw.put(f32_to_f16(scale), 16);
                prev_scale = scale;
                prev_zp = 0.0f;
                if (scale == 0.0f) {
                    for (int i = 0; i < n; ++i) bw.put(0, bits);
                    return;
                }
                if (is8) {
                    for (int i = 0; i < n; ++i) {
                        uint32_t idx = (uint32_t)std::lround((w[i] / scale + 1.0f) * 127.5f);
                        bw.put(idx > 255 ? 255u : idx, 8);
                    }
                } else {
                    for (int i = 0; i < n; ++i)
                        bw.put(nearest_level(w[i] / scale, bits), bits);
                }
            }
        } else {
            // Partial tail: reuse the previous full group's scale/zp.
            for (int i = tail_start; i < n; ++i) {
                if (prev_scale == 0.0f) { bw.put(0, bits); continue; }
                if (is8) {
                    float t = (w[i] - prev_zp) / prev_scale;
                    uint32_t idx = (uint32_t)std::lround(t * 255.0f);
                    bw.put(idx > 255 ? 255u : idx, 8);
                } else {
                    bw.put(nearest_level((float)(((double)w[i] - prev_zp) / (double)prev_scale), bits), bits);
                }
            }
        }
    }
}

void dequant_grouped_lattice(const uint8_t* bytes, size_t size, int bits, int n,
                             float* out) {
    BitReader br(bytes, size);
    const int nfull = n / kGrpSize;
    const bool is8 = (bits == 8);
    float prev_zp = 0.0f, prev_scale = 1.0f;
    for (int g = 0; g < nfull; ++g) {
        const int start = g * kGrpSize;
        const float zp = f16_to_f32((uint16_t)br.get(16));
        const float scale = f16_to_f32((uint16_t)br.get(16));
        prev_zp = zp;
        prev_scale = scale;
        if (scale == 0.0f) {
            for (int i = start; i < start + kGrpSize; ++i) { out[i] = zp; br.get(bits); }
            continue;
        }
        if (is8) {
            for (int i = start; i < start + kGrpSize; ++i)
                out[i] = zp + (float)br.get(8) / 255.0f * scale;
        } else {
            for (int i = start; i < start + kGrpSize; ++i)
                out[i] = level_value(bits, br.get(bits)) * scale + zp;
        }
    }
    const int tail_start = nfull * kGrpSize;
    const int tail_count = n - tail_start;
    if (tail_count > 0) {
        if (nfull == 0) {
            if (n < 8) {
                if (is8) {
                    for (int i = 0; i < n; ++i) out[i] = (float)br.get(8) / 127.5f - 1.0f;
                } else {
                    for (int i = 0; i < n; ++i) out[i] = level_value(bits, br.get(bits));
                }
            } else {
                const float scale = f16_to_f32((uint16_t)br.get(16));
                if (scale == 0.0f) {
                    for (int i = 0; i < n; ++i) { out[i] = 0.0f; br.get(bits); }
                    return;
                }
                if (is8) {
                    for (int i = 0; i < n; ++i) out[i] = ((float)br.get(8) / 127.5f - 1.0f) * scale;
                } else {
                    for (int i = 0; i < n; ++i) out[i] = level_value(bits, br.get(bits)) * scale;
                }
            }
        } else {
            for (int i = tail_start; i < n; ++i) {
                if (prev_scale == 0.0f) { out[i] = prev_zp; br.get(bits); continue; }
                if (is8)
                    out[i] = prev_zp + (float)br.get(8) / 255.0f * prev_scale;
                else
                    out[i] = level_value(bits, br.get(bits)) * prev_scale + prev_zp;
            }
        }
    }
}

} // namespace

bool quantize_block_all(Format fmt, const float* w, int n,
                        std::vector<uint8_t>& indices, std::vector<uint8_t>& codebook) {
    if (!w || n <= 0) return false;
    indices.clear();
    codebook.clear();

    switch (fmt) {
        case Format::OIL32:
            indices.resize((size_t)n * 4);
            std::memcpy(indices.data(), w, (size_t)n * 4);
            return true;
        case Format::OIL16:
        case Format::OIL16_GRP:
            indices.resize((size_t)n * 2);
            for (int i = 0; i < n; ++i) {
                const uint16_t h = f32_to_f16(w[i]);
                indices[(size_t)i * 2] = (uint8_t)(h & 0xFF);
                indices[(size_t)i * 2 + 1] = (uint8_t)(h >> 8);
            }
            return true;
        case Format::OIL8:
            quant_lattice(fmt, w, n, 8, indices);
            return true;
        case Format::OIL8_GRP:
            quant_grouped_lattice(8, w, n, indices);
            return true;
        case Format::OIL4:
            quant_lattice(fmt, w, n, 4, indices);
            return true;
        case Format::OIL4_GRP:
            quant_grouped_lattice(4, w, n, indices);
            return true;
        case Format::OIL2:
            quant_lattice(fmt, w, n, 2, indices);
            return true;
        case Format::OIL2_GRP:
            quant_grouped_lattice(2, w, n, indices);
            return true;
        case Format::OIL1: {
            // FP16 block means (codebook channel): always <= 1.0 BPW.
            const int blocks = (n + kMeanBlock - 1) / kMeanBlock;
            codebook.resize((size_t)blocks * 2);
            for (int b = 0; b < blocks; ++b) {
                const int start = b * kMeanBlock;
                const int end = std::min(start + kMeanBlock, n);
                double sum = 0.0;
                for (int i = start; i < end; ++i) sum += w[i];
                const uint16_t h = f32_to_f16((float)(sum / (double)(end - start)));
                codebook[(size_t)b * 2] = (uint8_t)(h & 0xFF);
                codebook[(size_t)b * 2 + 1] = (uint8_t)(h >> 8);
            }
            return true;
        }
        case Format::OIL1_GRP: {
            // Scale + signs can only be funded within the 1.0 BPW budget when
            // there are at least 32 weights (16-bit scale + (n-16) signs == n
            // bits). Smaller tails fall back to bare sign bits (decode ±1),
            // exactly like SPARK_Q0_GRP — signs stay intact and the payload
            // stays within ceil(n/8) bytes.
            if (n < 32) {
                indices.assign((size_t)(n + 7) / 8, 0);
                BitWriter bw(indices);
                for (int i = 0; i < n; ++i)
                    bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
                return true;
            }
            indices.assign((size_t)(n + 7) / 8, 0);
            BitWriter bw(indices);
            const float scale = rms_scale(w, n);
            bw.put(f32_to_f16(scale), 16);
            for (int i = 0; i < n; ++i) {
                if (is_slot(i, n, 16)) continue;
                bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
            }
            return true;
        }
        case Format::SPARK_Q0: {
            indices.assign(((size_t)n * 3 + 15) / 16, 0);
            BitWriter bw(indices);
            for (int start = 0; start < n; start += kSparkSubBlock) {
                const int end = std::min(start + kSparkSubBlock, n);
                const int count = end - start;
                const bool has_scale = count == kSparkSubBlock;
                const float scale = has_scale ? rms_scale(w + start, count) : 1.0f;
                if (has_scale) bw.put(f32_to_f16(scale), 16);
                const float s = scale <= 1e-12f ? 0.0f : scale;
                for (int i = start; i < end; ++i)
                    bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
            }
            return true;
        }
        case Format::SPARK_Q0_GRP: {
            // Scale + n signs exceed the 1.5 BPW budget below 32 weights
            // (16 + n > 1.5n for n < 32), so tiny tails are sign-only.
            if (n < 32) {
                // Tiny tail: sign bits only (no scale, no refinement).
                indices.assign((size_t)(n + 7) / 8, 0);
                BitWriter bw(indices);
                for (int i = 0; i < n; ++i)
                    bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
                return true;
            }
            indices.assign((size_t)(n * 3 + 15) / 16, 0);
            BitWriter bw(indices);
            const float scale = rms_scale(w, n);
            bw.put(f32_to_f16(scale), 16);
            for (int i = 0; i < n; ++i)
                bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
            const int refined = std::max(0, n / 2 - 16);
            const float threshold = scale * 1.23f;
            for (int i = 0; i < n; ++i) {
                if (!is_slot(i, n, refined)) continue;
                bw.put(std::fabs(w[i]) > threshold ? 1u : 0u, 1);
            }
            return true;
        }
        case Format::SPARK_SPARSE: {
            // 4-byte FP32 scale + k * 3-byte records (uint16 rel + int8 q).
            // Budget 0.25 bytes/weight; k = (n/4 - 4) / 3 exactly fills it.
            const size_t budget = (size_t)n / 4;
            if (budget < 4) return true; // cannot store scale -> all zeros
            const int keep = std::min(20, (int)((budget - 4) / 3));

            float scale = 0.0f;
            std::vector<std::pair<float, int>> mag((size_t)n);
            for (int i = 0; i < n; ++i) {
                const float a = std::fabs(w[i]);
                mag[(size_t)i] = { a, i };
                scale = std::max(scale, a);
            }
            if (scale <= 1e-12f) {
                indices.resize(4, 0);
                return true;
            }
            if (keep > 0) {
                std::partial_sort(mag.begin(), mag.begin() + keep, mag.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
            }

            indices.resize((size_t)4 + (size_t)keep * 3);
            uint32_t scale_bits;
            std::memcpy(&scale_bits, &scale, 4);
            indices[0] = (uint8_t)(scale_bits & 0xFF);
            indices[1] = (uint8_t)((scale_bits >> 8) & 0xFF);
            indices[2] = (uint8_t)((scale_bits >> 16) & 0xFF);
            indices[3] = (uint8_t)((scale_bits >> 24) & 0xFF);
            for (int i = 0; i < keep; ++i) {
                const int idx = mag[(size_t)i].second;
                int q = (int)std::lround(w[idx] / scale * 127.0f);
                q = std::max(-127, std::min(127, q));
                const uint16_t rel = (uint16_t)idx;
                const size_t base = 4 + (size_t)i * 3;
                indices[base] = (uint8_t)(rel & 0xFF);
                indices[base + 1] = (uint8_t)(rel >> 8);
                indices[base + 2] = (uint8_t)q;
            }
            return true;
        }
        case Format::SPARK_SPARSE_GRP: {
            // Real grouped sparsity inside the 2.0 BPW budget: 4-byte header
            // (two FP16 scales, one per block half) + k*3-byte records.
            // Each record's group is inferred from its relative index, so no
            // per-record group byte is needed and the payload is exactly
            // budget = n/4 bytes (2.0 BPW) when n is a multiple of 4.
            const size_t budget = (size_t)n / 4;
            if (budget < 8) return true; // header + >=1 record needs 7 bytes
            const int keep = (int)((budget - 4) / 3);
            if (keep <= 0) return true;
            const int half = n / 2;

            float scale0 = 0.0f, scale1 = 0.0f;
            std::vector<std::pair<float, int>> mag((size_t)n);
            for (int i = 0; i < n; ++i) {
                const float a = std::fabs(w[i]);
                mag[(size_t)i] = { a, i };
                if (i < half) scale0 = std::max(scale0, a);
                else scale1 = std::max(scale1, a);
            }
            if (scale0 <= 1e-12f) scale0 = 1.0f;
            if (scale1 <= 1e-12f) scale1 = 1.0f;
            std::partial_sort(mag.begin(), mag.begin() + keep, mag.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });

            indices.resize((size_t)4 + (size_t)keep * 3);
            const uint16_t h0 = f32_to_f16(scale0);
            const uint16_t h1 = f32_to_f16(scale1);
            indices[0] = (uint8_t)(h0 & 0xFF);
            indices[1] = (uint8_t)(h0 >> 8);
            indices[2] = (uint8_t)(h1 & 0xFF);
            indices[3] = (uint8_t)(h1 >> 8);
            for (int i = 0; i < keep; ++i) {
                const int idx = mag[(size_t)i].second;
                const float s = (idx < half) ? scale0 : scale1;
                int q = (int)std::lround(w[idx] / s * 127.0f);
                q = std::max(-127, std::min(127, q));
                const uint16_t rel = (uint16_t)idx;
                const size_t base = 4 + (size_t)i * 3;
                indices[base] = (uint8_t)(rel & 0xFF);
                indices[base + 1] = (uint8_t)(rel >> 8);
                indices[base + 2] = (uint8_t)q;
            }
            return true;
        }
        default:
            return false;
    }
}

void dequantize_block_all(Format fmt,
                          const uint8_t* indices, size_t idx_bytes,
                          const uint8_t* codebook, size_t cb_bytes,
                          uint32_t nw, float* out) {
    if (!out || nw == 0) return;
    std::fill(out, out + nw, 0.0f);
    const int n = (int)std::min<uint32_t>(nw, 1u << 24); // sanity cap

    switch (fmt) {
        case Format::OIL32:
            if (idx_bytes >= (size_t)n * 4) std::memcpy(out, indices, (size_t)n * 4);
            return;
        case Format::OIL16:
        case Format::OIL16_GRP:
            for (int i = 0; i < n; ++i) {
                const size_t o = (size_t)i * 2;
                if (o + 1 >= idx_bytes) break;
                const uint16_t h = (uint16_t)(indices[o]) | ((uint16_t)(indices[o + 1]) << 8);
                out[i] = f16_to_f32(h);
            }
            return;
        case Format::OIL8:
            dequant_lattice(indices, idx_bytes, n, 8, out);
            return;
        case Format::OIL8_GRP:
            dequant_grouped_lattice(indices, idx_bytes, 8, n, out);
            return;
        case Format::OIL4:
            dequant_lattice(indices, idx_bytes, n, 4, out);
            return;
        case Format::OIL4_GRP:
            dequant_grouped_lattice(indices, idx_bytes, 4, n, out);
            return;
        case Format::OIL2:
            dequant_lattice(indices, idx_bytes, n, 2, out);
            return;
        case Format::OIL2_GRP:
            dequant_grouped_lattice(indices, idx_bytes, 2, n, out);
            return;
        case Format::OIL1: {
            for (int b = 0; b * kMeanBlock < n; ++b) {
                const size_t o = (size_t)b * 2;
                if (o + 1 >= cb_bytes) break;
                const uint16_t h = (uint16_t)(codebook[o]) | ((uint16_t)(codebook[o + 1]) << 8);
                const float mean = f16_to_f32(h);
                const int start = b * kMeanBlock;
                const int end = std::min(start + kMeanBlock, n);
                std::fill(out + start, out + end, mean);
            }
            return;
        }
        case Format::OIL1_GRP: {
            if (n < 32) {
                // Sign-only tail (mirrors the encoder): no scale, decode ±1.
                BitReader br(indices, idx_bytes);
                for (int i = 0; i < n; ++i)
                    out[i] = br.get(1) == 0 ? -1.0f : 1.0f;
                return;
            }
            BitReader br(indices, idx_bytes);
            const float scale = f16_to_f32((uint16_t)br.get(16));
            for (int i = 0; i < n; ++i) {
                if (is_slot(i, n, 16)) { out[i] = 0.0f; continue; }
                out[i] = br.get(1) == 0 ? -scale : scale;
            }
            return;
        }
        case Format::SPARK_Q0: {
            BitReader br(indices, idx_bytes);
            for (int start = 0; start < n; start += kSparkSubBlock) {
                const int end = std::min(start + kSparkSubBlock, n);
                const int count = end - start;
                float scale = 1.0f;
                if (count == kSparkSubBlock) scale = f16_to_f32((uint16_t)br.get(16));
                for (int i = start; i < end; ++i)
                    out[i] = br.get(1) == 0 ? -scale : scale;
            }
            return;
        }
        case Format::SPARK_Q0_GRP: {
            if (n < 32) {
                // Sign-only tail (mirrors the encoder): no scale, decode ±1.
                BitReader br(indices, idx_bytes);
                for (int i = 0; i < n; ++i)
                    out[i] = br.get(1) == 0 ? -1.0f : 1.0f;
                return;
            }
            BitReader br(indices, idx_bytes);
            const float scale = f16_to_f32((uint16_t)br.get(16));
            const int refined = std::max(0, n / 2 - 16);
            std::vector<uint8_t> signs((size_t)n);
            for (int i = 0; i < n; ++i)
                signs[(size_t)i] = (uint8_t)br.get(1);
            for (int i = 0; i < n; ++i) {
                float magnitude = scale;
                if (is_slot(i, n, refined))
                    magnitude = scale * (br.get(1) == 0 ? 0.567f : 1.893f);
                out[i] = signs[(size_t)i] == 0 ? -magnitude : magnitude;
            }
            return;
        }
        case Format::SPARK_SPARSE: {
            if (idx_bytes < 4) return;
            uint32_t scale_bits = (uint32_t)indices[0] | ((uint32_t)indices[1] << 8) |
                                  ((uint32_t)indices[2] << 16) | ((uint32_t)indices[3] << 24);
            float scale;
            std::memcpy(&scale, &scale_bits, 4);
            if (scale == 0.0f) return;
            const size_t records = (idx_bytes - 4) / 3;
            for (size_t r = 0; r < records; ++r) {
                const size_t base = 4 + r * 3;
                if (base + 2 >= idx_bytes) break;
                const uint32_t rel = (uint32_t)indices[base] | ((uint32_t)indices[base + 1] << 8);
                const int8_t q = (int8_t)indices[base + 2];
                if (rel < (uint32_t)n) out[rel] = (float)q / 127.0f * scale;
            }
            return;
        }
        case Format::SPARK_SPARSE_GRP: {
            // Per-half-block FP16 scales in a 4-byte header; records carry
            // uint16 rel index + int8 q and the group is inferred from the
            // relative index (< n/2 -> group 0, else group 1). A block of
            // 32..39 weights keeps exactly one record (7 bytes).
            if (idx_bytes < 7) return;
            const uint16_t s0 = (uint16_t)indices[0] | ((uint16_t)indices[1] << 8);
            const uint16_t s1 = (uint16_t)indices[2] | ((uint16_t)indices[3] << 8);
            const float scale0 = f16_to_f32(s0);
            const float scale1 = f16_to_f32(s1);
            if (scale0 == 0.0f && scale1 == 0.0f) return;
            const size_t records = (idx_bytes - 4) / 3;
            const int half = n / 2;
            for (size_t r = 0; r < records; ++r) {
                const size_t base = 4 + r * 3;
                if (base + 2 >= idx_bytes) break;
                const uint32_t rel = (uint32_t)indices[base] | ((uint32_t)indices[base + 1] << 8);
                const int8_t q = (int8_t)indices[base + 2];
                const float s = ((int)rel < half) ? scale0 : scale1;
                if (rel < (uint32_t)n) out[rel] = (float)q / 127.0f * s;
            }
            return;
        }
        default:
            return;
    }
}

float block_actual_bpw(uint32_t nw, size_t idx_bytes, size_t cb_bytes) {
    if (nw == 0) return 0.0f;
    return (float)((idx_bytes + cb_bytes) * 8.0 / (double)nw);
}

size_t block_claimed_bytes(Format fmt, uint32_t n) {
    const float bpw = format_bpw(fmt);
    return (size_t)std::ceil(bpw * (float)n / 8.0f);
}

} // namespace oil
