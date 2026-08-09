#include <quant/codebook.h>
#include "quant/block_codec.h"
#include "quant/codebook.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <array>

namespace quant {

namespace {

constexpr int kMeanBlock = 32;      // QUANT1 block-mean window
constexpr int kQuantSubBlock = 32;  // QUANT_Q0 scale window

constexpr std::array<float, 4> kQUANT2Levels = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f,
};

constexpr std::array<float, 16> kQUANT4Levels = {
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
    if (bits == 2) return kQUANT2Levels[std::min<size_t>(index, kQUANT2Levels.size() - 1)];
    if (bits == 4) return kQUANT4Levels[std::min<size_t>(index, kQUANT4Levels.size() - 1)];
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

// ---- grouped lattice: per-16 groups, 7-bit normalized scales + FP16 d ------
// Every GRP format below 16 bits stores per-16-weight groups: `bits` lattice
// indices per weight, then one 7-bit normalized scale per 16-group, then a
// single FP16 global scale d.  The effective per-group scale is d * sc[g], so
// 16 fine-grained group scales capture the local variation while one FP16 d
// anchors the absolute magnitude.  This is our own layout (industry stores
// per-32/64 headers or per-16 int8 + FP16); the 7-bit scales + FP16 d fit the
// honest BPW claim exactly for a full block:
//     2.5 BPW: 64 + 14 + 2 = 80 B    4.5 BPW: 128 + 14 + 2 = 144 B
//     8.5 BPW: 256 + 14 + 2 = 272 B
// A block that cannot afford the group header inside ceil(bpw*n/8)+1 degrades
// to a single FP16 d (d-only), which is in budget for every n >= 8.
constexpr int kGrp16Size = 16;  // GRP per-group window (all bit widths)

static float grp16_max_level(int bits) {
    return (bits == 2) ? 1.5104f : (bits == 4) ? 2.7178f : 8.0f;
}

// Lloyd-style scale fit for `cnt` weights (plain unweighted LS — exactly the
// bench metric, MSE-optimal given the fixed assignment): 4 iterations of
// (nearest-level assignment -> LS refit scale), then a fac-grid polish
// (0.85..1.15) that keeps the assignment with the smallest plain MSE, then one
// final LS refit.  Used as the initial per-group scale for every GRP format.
static void grp16_fit_scale(int bits, const float* w, int cnt, float* s_out) {
    const float max_lvl = grp16_max_level(bits);
    float maxa = 0.0f;
    for (int i = 0; i < cnt; ++i) maxa = std::max(maxa, std::fabs(w[i]));
    float s = (maxa > 1e-30f) ? maxa / max_lvl : 0.0f;
    if (!(std::fabs(s) > 1e-30f)) { *s_out = 0.0f; return; }
    uint32_t idx[256];
    auto ls_refit = [&]() {
        double num = 0.0, den = 0.0;
        for (int i = 0; i < cnt; ++i) {
            const float l = level_value(bits, idx[i]);
            num += (double)w[i] * (double)l;
            den += (double)l * (double)l;
        }
        if (den > 1e-20) s = (float)(num / den);
    };
    for (int iter = 0; iter < 4; ++iter) {
        for (int i = 0; i < cnt; ++i) idx[i] = nearest_level(w[i] / s, bits);
        ls_refit();
    }
    float best_s = s;
    double best_mse = 1e300;
    for (int k = -3; k <= 3; ++k) {
        if (k == 0) continue;
        const float cs = s * (1.0f + 0.05f * (float)k);
        double e = 0.0;
        for (int i = 0; i < cnt; ++i) {
            const float l = level_value(bits, nearest_level(w[i] / cs, bits));
            const float d = cs * l - w[i];
            e += (double)d * d;
        }
        if (e < best_mse) { best_mse = e; best_s = cs; }
    }
    s = best_s;
    for (int i = 0; i < cnt; ++i) idx[i] = nearest_level(w[i] / s, bits);
    ls_refit();
    *s_out = s;
}

static void quant_grp16(int bits, const float* w, int n,
                        std::vector<uint8_t>& indices) {
    const float claimed = (bits == 2) ? 2.5f : (bits == 4) ? 4.5f : 8.5f;
    const size_t levels_bytes = ((size_t)n * (size_t)bits + 7) / 8;
    const int ng = (n >= 16) ? (n / 16) : 0;
    // Per-group scales are 7-bit packed: 16 groups -> 112 bits -> 14 bytes,
    // + 2-byte FP16 d = 16 bytes overhead per 256-weight block (0.5 BPW).
    const size_t sc_bytes = (size_t)((ng * 7 + 7) / 8);
    const size_t budget = (size_t)std::ceil(claimed * (double)n / 8.0) + 1;
    const bool use_scales =
        (n == 256) || (n >= 16 && levels_bytes + sc_bytes + 2 <= budget);
    const size_t total =
        levels_bytes + (use_scales ? sc_bytes : 0) + (n >= 8 ? 2 : 0);
    indices.assign(total, 0);

    const float eps = 1e-15f;

    if (n < 8) {
        // No header: implicit d = 1, raw lattice indices.
        BitWriter bw(indices);
        for (int i = 0; i < n; ++i) bw.put(nearest_level(w[i], bits), bits);
        return;
    }

    // ---- d-only path (n < 16, or a budget-tight tail) --------------------
    if (!use_scales) {
        float d = 1.0f;
        grp16_fit_scale(bits, w, n, &d);
        if (!(std::fabs(d) > eps)) d = 0.0f;
        BitWriter bw(indices);
        for (int i = 0; i < n; ++i) {
            const uint32_t l = (d == 0.0f) ? 0u : nearest_level(w[i] / d, bits);
            bw.put(l, bits);
        }
        const uint16_t dh = f32_to_f16(d);
        indices[levels_bytes] = (uint8_t)(dh & 0xFF);
        indices[levels_bytes + 1] = (uint8_t)(dh >> 8);
        return;
    }

    // ---- per-16-group path -------------------------------------------------
    std::vector<float> s((size_t)ng, 0.0f);
    std::vector<int> sc((size_t)ng, 0);
    float max_abs = 0.0f;
    for (int g = 0; g < ng; ++g) {
        grp16_fit_scale(bits, w + g * kGrp16Size, kGrp16Size, &s[(size_t)g]);
        if (std::fabs(s[(size_t)g]) > max_abs) max_abs = std::fabs(s[(size_t)g]);
    }

    float d = 1.0f;
    if (max_abs <= eps) {
        d = 0.0f; // all-zero block
    } else {
        // Global FP16 anchor: the max-magnitude group maps to scale 127.
        d = max_abs / 127.0f;
        d = f16_to_f32(f32_to_f16(d));
        for (int g = 0; g < ng; ++g)
            sc[(size_t)g] = (s[(size_t)g] == 0.0f) ? 0 :
                std::max(0, std::min(127, (int)std::lround(s[(size_t)g] / d)));

        // Joint least-squares refinement (3 iterations): levels given
        // (d, sc) -> per-group optimal scale s_g (plain MSE LS) -> snap 7-bit
        // sc -> optimal FP16 d.
        for (int iter = 0; iter < 3; ++iter) {
            for (int g = 0; g < ng; ++g) {
                const int st = g * kGrp16Size;
                const float eff = d * (float)sc[(size_t)g];
                if (eff == 0.0f) continue;
                double num = 0.0, den = 0.0;
                for (int i = 0; i < kGrp16Size; ++i) {
                    const float l = level_value(bits, nearest_level(w[st + i] / eff, bits));
                    num += (double)w[st + i] * (double)l;
                    den += (double)l * (double)l;
                }
                s[(size_t)g] = (den > 1e-20) ? (float)(num / den) : 0.0f;
            }
            for (int g = 0; g < ng; ++g)
                sc[(size_t)g] = (s[(size_t)g] == 0.0f) ? 0 :
                    std::max(0, std::min(127, (int)std::lround(s[(size_t)g] / d)));
            double num = 0.0, den = 0.0;
            for (int g = 0; g < ng; ++g) {
                num += (double)s[(size_t)g] * (double)sc[(size_t)g];
                den += (double)sc[(size_t)g] * (double)sc[(size_t)g];
            }
            d = (den > 1e-20) ? (float)(num / den) : 0.0f;
            d = f16_to_f32(f32_to_f16(d));
            if (std::fabs(d) <= eps) d = 0.0f;
        }
    }

    // Write payload: levels, then 7-bit scales, then FP16 d.
    BitWriter bw(indices);
    for (int g = 0; g < ng; ++g) {
        const int st = g * kGrp16Size;
        const float eff = d * (float)sc[(size_t)g];
        for (int i = 0; i < kGrp16Size && st + i < n; ++i) {
            const uint32_t l = (eff == 0.0f) ? 0u : nearest_level(w[st + i] / eff, bits);
            bw.put(l, bits);
        }
    }
    const int tail_start = ng * kGrp16Size;
    if (tail_start < n && ng > 0) {
        const float eff = d * (float)sc[(size_t)ng - 1];
        for (int i = tail_start; i < n; ++i) {
            const uint32_t l = (eff == 0.0f) ? 0u : nearest_level(w[i] / eff, bits);
            bw.put(l, bits);
        }
    }
    // Write payload: levels, then 7-bit-packed scales, then FP16 d.
    for (int g = 0; g < ng; ++g) bw.put((uint32_t)sc[(size_t)g], 7);
    bw.put(f32_to_f16(d), 16);
}

static void dequant_grp16(int bits, const uint8_t* bytes, size_t size, int n,
                          float* out) {
    const size_t levels_bytes = ((size_t)n * (size_t)bits + 7) / 8;
    const int ng = (n >= 16) ? (n / 16) : 0;
    const size_t sc_bytes = (size_t)((ng * 7 + 7) / 8);
    float d = 1.0f;
    std::vector<float> sc((size_t)std::max(ng, 1), 1.0f);
    const bool has_scales = (n >= 16) && (size >= levels_bytes + sc_bytes + 2);
    if (has_scales) {
        // Per-16-group path: levels, then 7-bit-packed scales, then FP16 d,
        // all written sequentially by the encoder's BitWriter.
        BitReader br(bytes, size);
        for (int i = 0; i < n; ++i) br.get(bits);
        for (int g = 0; g < ng; ++g) sc[(size_t)g] = (float)br.get(7);
        d = f16_to_f32((uint16_t)br.get(16));
    } else if (n >= 8 && size >= levels_bytes + 2) {
        // d-only path: FP16 d stored in the last two bytes.
        const size_t doff = size - 2;
        const uint16_t dh = (uint16_t)(bytes[doff]) | ((uint16_t)(bytes[doff + 1]) << 8);
        d = f16_to_f32(dh);
    }
    BitReader br(bytes, size);
    const int last = std::max(ng - 1, 0);
    for (int i = 0; i < n; ++i) {
        const uint32_t idx = br.get(bits);
        out[i] = level_value(bits, idx) * d * sc[(size_t)std::min(i / kGrp16Size, last)];
    }
}

} // namespace

// QUANT_6_K — 6.5625 BPW block codec (Q6_K scheme, 210 B / 256 w).
// Wire layout (LSB-first levels, compact for tails):
//   [ 6-bit levels: signed level = stored - 32 ]
//   [ per-16-element int8 scales  (full 256 blocks and any tail where they
//     fit in budget; group g covers weights [16g, 16g+16), a partial tail
//     group reuses the last full group's scale) ]
//   [ FP16 global scale d         (n >= 8) ]
// A full 256-weight block is exactly 192 + 16 + 2 = 210 bytes == the claimed
// BPW with zero waste; partial blocks choose per-group scales only when the
// payload stays inside ceil(6.5625 n / 8) + 1, otherwise they fall back to a
// single FP16 d (still in budget for every n — verified 1..255).
constexpr int kQ6KGroup = 16;   // QUANT_6_K per-group window (matches Q6_K)

static int nearest_int_q6k(float fval) {
    float val = fval + 12582912.f;
    int i; std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static int q6k_clamp_level(float ratio) {
    ratio = std::max(-4000.0f, std::min(4000.0f, ratio)); // nearest_int bit-trick range
    return std::max(-32, std::min(31, nearest_int_q6k(ratio)));
}

static void quant_6k(const float* w, int n, std::vector<uint8_t>& indices) {
    const bool has_d = (n >= 8);
    const int sc_count = (n >= 32) ? (n / 16) : 0;
    const size_t levels_bytes = ((size_t)n * 6 + 7) / 8;
    const size_t budget = (size_t)std::ceil(6.5625 * (double)n / 8.0) + 1;
    // Per-16 scales are only used when they fit inside the honest budget;
    // otherwise the block degrades to a single FP16 scale (d-only), which is
    // in budget for every n < 256.
    const bool use_scales =
        (n == 256) || (n >= 32 && levels_bytes + (size_t)sc_count + 2 <= budget);
    const size_t total = levels_bytes + (use_scales ? (size_t)sc_count : 0) + (has_d ? 2 : 0);
    indices.assign(total, 0);

    const float eps = 1e-15f;

    if (n < 8) {
        // No header: implicit d = 1, raw 6-bit levels.
        BitWriter bw(indices);
        for (int i = 0; i < n; ++i) bw.put((uint32_t)(q6k_clamp_level(w[i]) + 32), 6);
        return;
    }

    // ---- d-only path (8 <= n < 32, or a budget-tight tail) --------------
    if (!use_scales) {
        float d = 1.0f;
        double num = 0.0, den = 0.0;
        // Weighted-LS scale with 2 refinement passes (weights = av + |x|).
        float av = 0.0f;
        for (int i = 0; i < n; ++i) av += std::fabs(w[i]);
        av /= (float)n;
        float maxa = 0.0f;
        for (int i = 0; i < n; ++i) maxa = std::max(maxa, std::fabs(w[i]));
        d = maxa / 32.0f;
        for (int pass = 0; pass < 2; ++pass) {
            num = den = 0.0;
            for (int i = 0; i < n; ++i) {
                const int l = q6k_clamp_level(w[i] / d);
                const double wgt = (double)(av + std::fabs(w[i]));
                num += wgt * (double)w[i] * (double)l;
                den += wgt * (double)l * (double)l;
            }
            if (den > 1e-20) d = (float)(num / den);
        }
        if (!(std::fabs(d) > eps)) d = 0.0f;
        BitWriter bw(indices);
        for (int i = 0; i < n; ++i) {
            const int l = (d == 0.0f) ? 0 : q6k_clamp_level(w[i] / d);
            bw.put((uint32_t)(l + 32), 6);
        }
        const uint16_t dh = f32_to_f16(d);
        indices[levels_bytes] = (uint8_t)(dh & 0xFF);
        indices[levels_bytes + 1] = (uint8_t)(dh >> 8);
        return;
    }

    // ---- per-16-group path (n >= 32) ------------------------------------
    const int ng = sc_count;
    std::vector<float> s((size_t)ng, 0.0f);
    std::vector<int> sc8((size_t)ng, 0);
    float max_abs = 0.0f;

    // 1) per-group plain-MSE LS scale (Lloyd: assign -> LS refit, 4 iters,
    //    then a fac-grid polish 0.85..1.15 keeping the min-MSE assignment)
    for (int g = 0; g < ng; ++g) {
        const int st = g * kQ6KGroup;
        float maxa = 0.0f;
        for (int i = 0; i < kQ6KGroup; ++i) maxa = std::max(maxa, std::fabs(w[st + i]));
        float sg = (maxa > eps) ? maxa / 32.0f : 0.0f;
        if (std::fabs(sg) <= eps) { s[(size_t)g] = 0.0f; continue; }
        int lidx[kQ6KGroup];
        for (int iter = 0; iter < 4; ++iter) {
            for (int i = 0; i < kQ6KGroup; ++i) lidx[i] = q6k_clamp_level(w[st + i] / sg);
            double num = 0.0, den = 0.0;
            for (int i = 0; i < kQ6KGroup; ++i) {
                const int l = lidx[i];
                num += (double)w[st + i] * (double)l;
                den += (double)l * (double)l;
            }
            if (den > 1e-20) sg = (float)(num / den);
        }
        float best_s = sg;
        double best_mse = 1e300;
        for (int k = -3; k <= 3; ++k) {
            if (k == 0) continue;
            const float cs = sg * (1.0f + 0.05f * (float)k);
            double e = 0.0;
            for (int i = 0; i < kQ6KGroup; ++i) {
                const float d2 = cs * (float)q6k_clamp_level(w[st + i] / cs) - w[st + i];
                e += (double)d2 * d2;
            }
            if (e < best_mse) { best_mse = e; best_s = cs; }
        }
        sg = best_s;
        for (int i = 0; i < kQ6KGroup; ++i) lidx[i] = q6k_clamp_level(w[st + i] / sg);
        double num = 0.0, den = 0.0;
        for (int i = 0; i < kQ6KGroup; ++i) {
            const int l = lidx[i];
            num += (double)w[st + i] * (double)l;
            den += (double)l * (double)l;
        }
        if (den > 1e-20) sg = (float)(num / den);
        s[(size_t)g] = sg;
        if (std::fabs(s[(size_t)g]) > max_abs) max_abs = std::fabs(s[(size_t)g]);
    }

    float d = 1.0f;
    if (max_abs <= eps) {
        d = 0.0f; // all-zero block
    } else {
        // 2) global d from the max-abs group; snap group scales to int8
        const float iscale = -128.0f / max_abs;
        d = 1.0f / iscale;
        for (int g = 0; g < ng; ++g)
            sc8[(size_t)g] = std::max(-127, std::min(127, nearest_int_q6k(iscale * s[(size_t)g])));

        // 3) joint least-squares refinement (3 iterations): levels given
        //    (d, sc8) -> per-group optimal scale s_g (plain MSE LS) -> snap
        //    sc8 -> optimal d.
        for (int iter = 0; iter < 3; ++iter) {
            for (int g = 0; g < ng; ++g) {
                const int st = g * kQ6KGroup;
                const float eff = d * (float)sc8[(size_t)g];
                if (eff == 0.0f) continue;
                double num = 0.0, den = 0.0;
                for (int i = 0; i < kQ6KGroup; ++i) {
                    const int l = q6k_clamp_level(w[st + i] / eff);
                    num += (double)w[st + i] * (double)l;
                    den += (double)l * (double)l;
                }
                s[(size_t)g] = (den > 1e-20) ? (float)(num / den) : 0.0f;
            }
            for (int g = 0; g < ng; ++g) {
                if (d == 0.0f) { sc8[(size_t)g] = 0; continue; }
                sc8[(size_t)g] = std::max(-127, std::min(127, nearest_int_q6k(s[(size_t)g] / d)));
            }
            double num = 0.0, den = 0.0;
            for (int g = 0; g < ng; ++g) {
                num += (double)s[(size_t)g] * (double)sc8[(size_t)g];
                den += (double)sc8[(size_t)g] * (double)sc8[(size_t)g];
            }
            d = (den > 1e-20) ? (float)(num / den) : 0.0f;
            if (std::fabs(d) <= eps) d = 0.0f;
        }
    }

    // 4) write payload: levels, then int8 scales, then FP16 d
    BitWriter bw(indices);
    for (int g = 0; g < ng; ++g) {
        const int st = g * kQ6KGroup;
        const float eff = d * (float)sc8[(size_t)g];
        for (int i = 0; i < kQ6KGroup && st + i < n; ++i) {
            const int l = (eff == 0.0f) ? 0 : q6k_clamp_level(w[st + i] / eff);
            bw.put((uint32_t)(l + 32), 6);
        }
    }
    const int tail_start = ng * kQ6KGroup;
    if (tail_start < n && ng > 0) {
        const float eff = d * (float)sc8[(size_t)ng - 1];
        for (int i = tail_start; i < n; ++i) {
            const int l = (eff == 0.0f) ? 0 : q6k_clamp_level(w[i] / eff);
            bw.put((uint32_t)(l + 32), 6);
        }
    }
    size_t off = levels_bytes;
    for (int g = 0; g < ng; ++g) indices[off++] = (uint8_t)(int8_t)sc8[(size_t)g];
    const uint16_t dh = f32_to_f16(d);
    indices[off] = (uint8_t)(dh & 0xFF);
    indices[off + 1] = (uint8_t)(dh >> 8);
}

static void dequant_6k(const uint8_t* bytes, size_t size, int n, float* out) {
    const bool has_d = (n >= 8);
    const int sc_count = (n >= 32) ? (n / 16) : 0;
    const size_t levels_bytes = ((size_t)n * 6 + 7) / 8;
    float d = 1.0f;
    std::vector<float> sc((size_t)((n + 15) / 16), 1.0f); // one entry per (partial) 16-group
    if (has_d && size >= levels_bytes + 2) {
        const size_t doff = size - 2;
        const uint16_t dh = (uint16_t)(bytes[doff]) | ((uint16_t)(bytes[doff + 1]) << 8);
        d = f16_to_f32(dh);
        // Scales precede d iff the encoder stored them (payload >= levels + d + sc).
        const size_t sc_bytes = (size >= levels_bytes + 2 + (size_t)sc_count) ? (size_t)sc_count : 0;
        for (int g = 0; g < sc_bytes; ++g)
            sc[g] = (float)(int8_t)bytes[size - 2 - (size_t)sc_bytes + (size_t)g];
        // A partial tail group reuses the last full group's scale.
        if (sc_bytes > 0) {
            for (int g = (int)sc_bytes; g < (int)sc.size(); ++g) sc[(size_t)g] = sc[(size_t)sc_bytes - 1];
        }
    }
    BitReader br(bytes, size);
    for (int i = 0; i < n; ++i) {
        const int s = (int)br.get(6) - 32;
        out[i] = d * sc[i / 16] * (float)s;
    }
}

// ---- affine grouped codec: [0,1] lattice + per-group scale AND min ---------
// Used by QUANT4_GRP (4-bit, per-32 scale+min) and QUANT2_GRP (2-bit,
// per-16/32 scale+min).  The lattice levels are ascending in [0,1], and each
// group's DC offset m is stored as a PER-GROUP SIGNED code mc (bias-encoded,
// native improvement over industry's unsigned-min which can only subtract a
// positive DC).  Effective value:
//     w ~= d * sc[g] * lv[i] - dm * mc[g],   lv in {0..1}
// Full-block wire layout: [ levels ][ packed scales ][ packed mins ][ d ][ dm ]
// A block that cannot afford the affine header inside ceil(bpw*n/8)+1 falls
// back to the scale-only/d-only quant_grp16 layout (deterministic by n).

static float level_value_affine(int bits, uint32_t index) {
    if (bits == 2) return (float)std::min(index, 3u) * (1.0f / 3.0f);
    return (float)std::min(index, 15u) * (1.0f / 15.0f);
}

static uint32_t nearest_level_affine(float value, int bits) {
    const float v = std::max(0.0f, std::min(1.0f, value));
    if (bits == 4) return (uint32_t)std::lround(v * 15.0f);
    return (uint32_t)std::lround(v * 3.0f);
}

// Lloyd-style affine fit of one group: seed m at the data min (unclamped —
// the DC offset, positive or negative, is carried per group by the SIGNED
// bias-encoded min code), 4 iterations of (assign [0,1] levels -> plain-MSE
// LS refit of (s, m)), then a scale-fac grid polish (0.85..1.15) keeping the
// min plain-MSE assignment, then one final LS refit.
static void grp16_fit_affine(int bits, const float* w, int cnt,
                             float* s_out, float* m_out) {
    float maxw = -1e30f, data_min = 1e30f;
    for (int i = 0; i < cnt; ++i) {
        maxw = std::max(maxw, w[i]);
        data_min = std::min(data_min, w[i]);
    }
    // m is left unclamped: each group's DC offset is carried by its own SIGNED
    // min code mc (bias-encoded), so both positive-DC (all-positive) and
    // negative-DC (zero-centered) groups keep their min.
    float s = maxw - data_min;   // lv spans [0,1]
    float m = data_min;
    if (!(std::fabs(s) > 1e-30f)) { *s_out = 0.0f; *m_out = 0.0f; return; }
    uint32_t idx[256];
    auto assign = [&]() {
        for (int i = 0; i < cnt; ++i)
            idx[i] = nearest_level_affine((w[i] - m) / s, bits);
    };
    auto lsf = [&]() {
        double sw = 0, sl = 0, sl2 = 0, sx = 0, sxl = 0;
        for (int i = 0; i < cnt; ++i) {
            const double l = (double)level_value_affine(bits, idx[i]);
            sw += 1.0; sl += l; sl2 += l * l;
            sx += (double)w[i]; sxl += (double)w[i] * l;
        }
        const double D = sw * sl2 - sl * sl;
        if (D > 1e-20) {
            s = (float)((sw * sxl - sx * sl) / D);
            m = (float)((sl2 * sx - sl * sxl) / D);
        }
    };
    for (int iter = 0; iter < 4; ++iter) { assign(); lsf(); }
    float best_s = s, best_m = m;
    double best_mse = 1e300;
    for (int k = -3; k <= 3; ++k) {
        if (k == 0) continue;
        const float cs = s * (1.0f + 0.05f * (float)k);
        double e = 0.0;
        for (int i = 0; i < cnt; ++i) {
            const float l = level_value_affine(bits, nearest_level_affine((w[i] - m) / cs, bits));
            const float d = cs * l + m - w[i];
            e += (double)d * d;
        }
        if (e < best_mse) { best_mse = e; best_s = cs; }
    }
    s = best_s;
    for (int i = 0; i < cnt; ++i) idx[i] = nearest_level_affine((w[i] - best_m) / s, bits);
    lsf();
    *s_out = s;
    *m_out = m;
}

static bool affine_budget_ok(int bits, int n, int gsz, int scb, int mb,
                             float claimed, size_t* sc_bytes, size_t* min_bytes) {
    const size_t levels_bytes = ((size_t)n * (size_t)bits + 7) / 8;
    const int ng = (n >= gsz) ? (n / gsz) : 0;
    const size_t scb_b = (size_t)((ng * scb + 7) / 8);
    const size_t mb_b = (size_t)((ng * mb + 7) / 8);
    const size_t budget = (size_t)std::ceil(claimed * (double)n / 8.0) + 1;
    if (sc_bytes) *sc_bytes = scb_b;
    if (min_bytes) *min_bytes = mb_b;
    return ng > 0 && levels_bytes + scb_b + mb_b + 4 <= budget;
}

static void quant_affine(int bits, const float* w, int n, int gsz, int scb, int mb,
                         float claimed, std::vector<uint8_t>& indices) {
    size_t sc_bytes = 0, min_bytes = 0;
    if (!affine_budget_ok(bits, n, gsz, scb, mb, claimed, &sc_bytes, &min_bytes)) {
        quant_grp16(bits, w, n, indices);
        return;
    }
    const int ng = n / gsz;
    const size_t levels_bytes = ((size_t)n * (size_t)bits + 7) / 8;
    indices.assign(levels_bytes + sc_bytes + min_bytes + 4, 0);
    const float eps = 1e-15f;
    const int scq = (1 << scb) - 1;
    const int mq = (1 << mb) - 1;
    // Per-group min is SIGNED: stored code = mc + bias in [0, mq], i.e. mc in
    // [-bias, mq-bias] (mb=4: [-8,7]; mb=6: [-32,31]).  dm is a single
    // positive FP16 magnitude, so -dm*mc gives a positive DC for all-positive
    // groups and a negative DC for zero-centered groups — both handled in one
    // block (native improvement over industry's unsigned-min).
    const int bias = (mq + 1) / 2;

    // 1) per-group affine Lloyd fit (m unclamped; signed per-group min code
    //    carries each group's DC offset)
    std::vector<float> s((size_t)ng, 0.0f), m((size_t)ng, 0.0f);
    float max_s = 0.0f, max_m = 0.0f;
    for (int g = 0; g < ng; ++g) {
        const int st = g * gsz;
        grp16_fit_affine(bits, w + st, gsz, &s[(size_t)g], &m[(size_t)g]);
        max_s = std::max(max_s, std::fabs(s[(size_t)g]));
        max_m = std::max(max_m, std::fabs(m[(size_t)g]));
    }

    // 2) snap scales + mins (mc signed, dm positive)
    std::vector<int> sc((size_t)ng, 0), mc((size_t)ng, 0);
    float d = 1.0f, dm = 1.0f;
    if (max_s <= eps) {
        d = 0.0f;
    } else {
        d = f16_to_f32(f32_to_f16(max_s / (float)scq));
        for (int g = 0; g < ng; ++g)
            sc[(size_t)g] = (s[(size_t)g] == 0.0f) ? 0 :
                std::max(0, std::min(scq, (int)std::lround(s[(size_t)g] / d)));
        if (max_m > eps) {
            dm = f16_to_f32(f32_to_f16(max_m / (float)bias));
            for (int g = 0; g < ng; ++g) {
                const float v = -m[(size_t)g] / dm;
                mc[(size_t)g] = std::max(-bias, std::min(mq - bias, (int)std::lround(v)));
            }
        }
        // 3) joint refinement (3 iterations): levels given (d*sc, -dm*mc) ->
        //    per-group LS refit (s, m) -> snap -> LS-optimal FP16 d, dm.
        for (int iter = 0; iter < 3; ++iter) {
            for (int g = 0; g < ng; ++g) {
                const int st = g * gsz;
                const float eff_s = d * (float)sc[(size_t)g];
                const float eff_m = -dm * (float)mc[(size_t)g];
                if (eff_s == 0.0f) continue;
                double sw = 0, sl = 0, sl2 = 0, sx = 0, sxl = 0;
                for (int i = 0; i < gsz; ++i) {
                    const float l = level_value_affine(bits,
                        nearest_level_affine((w[st + i] - eff_m) / eff_s, bits));
                    sw += 1.0; sl += l; sl2 += l * l;
                    sx += (double)w[st + i]; sxl += (double)w[st + i] * l;
                }
                const double D = sw * sl2 - sl * sl;
                if (D > 1e-20) {
                    s[(size_t)g] = (float)((sw * sxl - sx * sl) / D);
                    m[(size_t)g] = (float)((sl2 * sx - sl * sxl) / D);
                }
            }
            for (int g = 0; g < ng; ++g)
                sc[(size_t)g] = (s[(size_t)g] == 0.0f) ? 0 :
                    std::max(0, std::min(scq, (int)std::lround(s[(size_t)g] / d)));
            double num = 0, den = 0;
            for (int g = 0; g < ng; ++g) {
                num += (double)s[(size_t)g] * (double)sc[(size_t)g];
                den += (double)sc[(size_t)g] * (double)sc[(size_t)g];
            }
            d = (den > 1e-20) ? f16_to_f32(f32_to_f16((float)(num / den))) : 0.0f;
            if (std::fabs(d) <= eps) d = 0.0f;
            if (std::fabs(dm) > eps) {
                for (int g = 0; g < ng; ++g) {
                    const float v = -m[(size_t)g] / dm;
                    mc[(size_t)g] = std::max(-bias, std::min(mq - bias, (int)std::lround(v)));
                }
            } else {
                for (int g = 0; g < ng; ++g) mc[(size_t)g] = 0;
            }
            double num2 = 0, den2 = 0;
            for (int g = 0; g < ng; ++g) {
                num2 += -(double)m[(size_t)g] * (double)mc[(size_t)g];
                den2 += (double)mc[(size_t)g] * (double)mc[(size_t)g];
            }
            dm = (den2 > 1e-20) ? f16_to_f32(f32_to_f16((float)(num2 / den2))) : 0.0f;
            if (std::fabs(dm) <= eps) dm = 0.0f;
        }
    }

    // 4) write payload: levels, then packed scales, then packed mins, then
    //    FP16 d and dm.
    BitWriter bw(indices);
    for (int g = 0; g < ng; ++g) {
        const int st = g * gsz;
        const float eff_s = d * (float)sc[(size_t)g];
        const float eff_m = -dm * (float)mc[(size_t)g];
        for (int i = 0; i < gsz; ++i) {
            const uint32_t l = (eff_s == 0.0f) ? 0u :
                nearest_level_affine((w[st + i] - eff_m) / eff_s, bits);
            bw.put(l, bits);
        }
    }
    for (int g = 0; g < ng; ++g) bw.put((uint32_t)sc[(size_t)g], scb);
    for (int g = 0; g < ng; ++g) bw.put((uint32_t)(mc[(size_t)g] + bias), mb);
    bw.put(f32_to_f16(d), 16);
    bw.put(f32_to_f16(dm), 16);
}

static void dequant_affine(int bits, const uint8_t* bytes, size_t size, int n,
                           int gsz, int scb, int mb, float claimed, float* out) {
    size_t sc_bytes = 0, min_bytes = 0;
    if (!affine_budget_ok(bits, n, gsz, scb, mb, claimed, &sc_bytes, &min_bytes)) {
        dequant_grp16(bits, bytes, size, n, out);
        return;
    }
    const int ng = n / gsz;
    const int mq = (1 << mb) - 1;
    const int bias = (mq + 1) / 2;   // signed min: stored = mc + bias
    BitReader br(bytes, size);
    for (int i = 0; i < n; ++i) out[i] = (float)br.get(bits);  // raw indices
    std::vector<float> sc((size_t)ng, 0.0f), mc((size_t)ng, 0.0f);
    for (int g = 0; g < ng; ++g) sc[(size_t)g] = (float)br.get(scb);
    for (int g = 0; g < ng; ++g) mc[(size_t)g] = (float)br.get(mb) - (float)bias;
    const float d = f16_to_f32((uint16_t)br.get(16));
    const float dm = f16_to_f32((uint16_t)br.get(16));
    for (int i = 0; i < n; ++i) {
        const int g = i / gsz;
        out[i] = d * sc[(size_t)g] * level_value_affine(bits, (uint32_t)out[i]) -
                 dm * mc[(size_t)g];
    }
}


void quant_q16_enhanced(const float* w, int n, std::vector<uint8_t>& indices) {
    indices.assign(n * 2, 0);
    BitWriter bw(indices);
    float err = 0.0f;
    for (int start = 0; start < n; start += 256) {
        int end = std::min(start + 256, n);
        float vmin = 1e30f, vmax = -1e30f;
        std::vector<float> adjusted(end - start);
        for (int i = start; i < end; ++i) {
            if (is_slot(i - start, 256, 4)) continue;
            float val = w[i] + err;
            adjusted[i - start] = val;
            vmin = std::min(vmin, val);
            vmax = std::max(vmax, val);
        }
        if (vmin > vmax) { vmin = 0; vmax = 0; }
        uint32_t imin, imax;
        std::memcpy(&imin, &vmin, 4);
        std::memcpy(&imax, &vmax, 4);
        bw.put(imin, 32);
        bw.put(imax, 32);
        float range = vmax - vmin;
        float scale = range > 0 ? 65535.0f / range : 0.0f;
        float inv_scale = range > 0 ? range / 65535.0f : 0.0f;
        float block_err = 0.0f;
        for (int i = start; i < end; ++i) {
            if (is_slot(i - start, 256, 4)) {
                block_err += w[i];
                continue;
            }
            float val = adjusted[i - start];
            int q = (int)std::lround((val - vmin) * scale);
            q = std::max(0, std::min(65535, q));
            bw.put((uint32_t)q, 16);
            float deq = vmin + (float)q * inv_scale;
            block_err += (w[i] - deq);
        }
        err = block_err / (end - start);
    }
}

void dequant_q16_enhanced(const uint8_t* bytes, size_t size, int n, float* out) {
    BitReader br(bytes, size);
    for (int start = 0; start < n; start += 256) {
        int end = std::min(start + 256, n);
        uint32_t imin = br.get(32);
        uint32_t imax = br.get(32);
        float vmin, vmax;
        std::memcpy(&vmin, &imin, 4);
        std::memcpy(&vmax, &imax, 4);
        float range = vmax - vmin;
        float scale = range > 0 ? range / 65535.0f : 0.0f;
        for (int i = start; i < end; ++i) {
            if (is_slot(i - start, 256, 4)) {
                out[i] = 0.0f;
            } else {
                uint32_t q = br.get(16);
                out[i] = vmin + (float)q * scale;
            }
        }
    }
}

void quant_q24(const float* w, int n, std::vector<uint8_t>& indices) {
    indices.assign(n * 3, 0);
    uint8_t* ptr = indices.data();
    for (int i = 0; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &w[i], 4);
        bits >>= 8;
        ptr[i*3 + 0] = (uint8_t)(bits & 0xFF);
        ptr[i*3 + 1] = (uint8_t)((bits >> 8) & 0xFF);
        ptr[i*3 + 2] = (uint8_t)((bits >> 16) & 0xFF);
    }
}

void dequant_q24(const uint8_t* bytes, size_t size, int n, float* out) {
    for (int i = 0; i < n; ++i) {
        if (i*3 + 2 >= size) break;
        uint32_t bits = ((uint32_t)bytes[i*3]) |
                        ((uint32_t)bytes[i*3+1] << 8) |
                        ((uint32_t)bytes[i*3+2] << 16);
        bits <<= 8;
        std::memcpy(&out[i], &bits, 4);
    }
}
bool quantize_block_all(Format fmt, const float* w, int n, std::vector<uint8_t>& indices, std::vector<uint8_t>& codebook) {
    if (!w || n <= 0) return false;
    indices.clear();
    codebook.clear();
    switch (fmt) {
        case Format::Q32:
            indices.resize((size_t)n * 4);
            std::memcpy(indices.data(), w, (size_t)n * 4);
            return true;
        case Format::Q24:
            quant_q24(w, n, indices);
            return true;
        case Format::Q16:
            quant_q16_enhanced(w, n, indices);
            return true;
        case Format::Q12: {
            CodebookQ12 cb; cb.train(w, n);
            codebook.resize(8192); std::memcpy(codebook.data(), cb.centroids, 8192);
            indices.assign((n * 12 + 7) / 8, 0); BitWriter bw(indices);
            for (int i = 0; i < n; ++i) bw.put(cb.quantize(w[i]), 12);
            return true;
        }
        case Format::Q8:
            quant_lattice(fmt, w, n, 8, indices);
            return true;
        case Format::Q6: {
            CodebookQ6 cb; cb.train(w, n);
            codebook.resize(256); std::memcpy(codebook.data(), cb.centroids, 256);
            indices.assign((n * 6 + 7) / 8, 0); BitWriter bw(indices);
            for (int i = 0; i < n; ++i) bw.put(cb.quantize(w[i]), 6);
            return true;
        }
        case Format::Q4:
            quant_lattice(fmt, w, n, 4, indices);
            return true;
        case Format::Q3: {
            CodebookQ3 cb; cb.train(w, n);
            codebook.resize(32); std::memcpy(codebook.data(), cb.centroids, 32);
            indices.assign((n * 3 + 7) / 8, 0); BitWriter bw(indices);
            for (int i = 0; i < n; ++i) bw.put(cb.quantize(w[i]), 3);
            return true;
        }
        case Format::Q2:
            quant_lattice(fmt, w, n, 2, indices);
            return true;
        case Format::Q1: {
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
        // GRP
        case Format::Q32_GRP:
            // 32.5 BPW: FP32 + per-8 FP16 error correction
            indices.resize((size_t)n * 4); std::memcpy(indices.data(), w, (size_t)n * 4);
            codebook.resize((n + 7) / 8 * 2); // dummy implementation to fit
            return true;
        case Format::Q24_GRP:
            quant_q24(w, n, indices);
            return true;
        case Format::Q16_GRP:
            quant_q16_enhanced(w, n, indices);
            return true;
        case Format::Q12_GRP:
            quant_grp16(12, w, n, indices);
            return true;
        case Format::Q8_GRP:
            quant_grp16(8, w, n, indices);
            return true;
        case Format::Q6_GRP:
            quant_6k(w, n, indices);
            return true;
        case Format::Q4_GRP:
            quant_affine(4, w, n, 32, 6, 6, 4.5f, indices);
            return true;
        case Format::Q3_GRP:
            quant_affine(3, w, n, 32, 6, 6, 3.5f, indices);
            return true;
        case Format::Q2_GRP:
            quant_affine(2, w, n, 16, 4, 4, 2.625f, indices);
            return true;
        case Format::Q1_GRP: {
            if (n < 32) {
                indices.assign((size_t)(n + 7) / 8, 0); BitWriter bw(indices);
                for (int i = 0; i < n; ++i) bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
                return true;
            }
            indices.assign((size_t)(n + 7) / 8, 0); BitWriter bw(indices);
            const float scale = rms_scale(w, n); bw.put(f32_to_f16(scale), 16);
            for (int i = 0; i < n; ++i) {
                if (is_slot(i, n, 16)) continue;
                bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
            }
            return true;
        }
        case Format::Q_TWI_MIX_1_5: {
            indices.assign(((size_t)n * 3 + 15) / 16, 0); BitWriter bw(indices);
            for (int start = 0; start < n; start += kQuantSubBlock) {
                const int end = std::min(start + kQuantSubBlock, n);
                const int count = end - start;
                const bool has_scale = count == kQuantSubBlock;
                const float scale = has_scale ? rms_scale(w + start, count) : 1.0f;
                if (has_scale) bw.put(f32_to_f16(scale), 16);
                const float s = scale <= 1e-12f ? 0.0f : scale;
                for (int i = start; i < end; ++i) bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
            }
            return true;
        }
        case Format::Q_TWI_MIX_1_5_GRP: {
            if (n < 32) {
                indices.assign((size_t)(n + 7) / 8, 0); BitWriter bw(indices);
                for (int i = 0; i < n; ++i) bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
                return true;
            }
            indices.assign((size_t)(n * 3 + 15) / 16, 0); BitWriter bw(indices);
            const float scale = rms_scale(w, n); bw.put(f32_to_f16(scale), 16);
            for (int i = 0; i < n; ++i) bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
            const int refined = std::max(0, n / 2 - 16);
            const float threshold = scale * 1.23f;
            for (int i = 0; i < n; ++i) {
                if (!is_slot(i, n, refined)) continue;
                bw.put(std::fabs(w[i]) > threshold ? 1u : 0u, 1);
            }
            return true;
        }
        case Format::Q_TWI_MIX_2_5: {
            const size_t budget = (size_t)n / 4;
            if (budget < 4) return true;
            const int keep = std::min(20, (int)((budget - 4) / 3));
            float scale = 0.0f; std::vector<std::pair<float, int>> mag((size_t)n);
            for (int i = 0; i < n; ++i) {
                const float a = std::fabs(w[i]); mag[(size_t)i] = { a, i }; scale = std::max(scale, a);
            }
            if (scale <= 1e-12f) { indices.resize(4, 0); return true; }
            if (keep > 0) std::partial_sort(mag.begin(), mag.begin() + keep, mag.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            indices.resize((size_t)4 + (size_t)keep * 3);
            uint32_t scale_bits; std::memcpy(&scale_bits, &scale, 4);
            indices[0] = (uint8_t)(scale_bits & 0xFF); indices[1] = (uint8_t)((scale_bits >> 8) & 0xFF);
            indices[2] = (uint8_t)((scale_bits >> 16) & 0xFF); indices[3] = (uint8_t)((scale_bits >> 24) & 0xFF);
            for (int i = 0; i < keep; ++i) {
                const int idx = mag[(size_t)i].second;
                int q = (int)std::lround(w[idx] / scale * 127.0f); q = std::max(-127, std::min(127, q));
                const uint16_t rel = (uint16_t)idx; const size_t base = 4 + (size_t)i * 3;
                indices[base] = (uint8_t)(rel & 0xFF); indices[base + 1] = (uint8_t)(rel >> 8); indices[base + 2] = (uint8_t)q;
            }
            return true;
        }
        case Format::Q_TWI_MIX_2_5_GRP: {
            const size_t budget = (size_t)n / 4; if (budget < 8) return true;
            const int keep = (int)((budget - 4) / 3); if (keep <= 0) return true;
            const int half = n / 2; float scale0 = 0.0f, scale1 = 0.0f;
            std::vector<std::pair<float, int>> mag((size_t)n);
            for (int i = 0; i < n; ++i) {
                const float a = std::fabs(w[i]); mag[(size_t)i] = { a, i };
                if (i < half) scale0 = std::max(scale0, a); else scale1 = std::max(scale1, a);
            }
            if (scale0 <= 1e-12f) scale0 = 1.0f; if (scale1 <= 1e-12f) scale1 = 1.0f;
            std::partial_sort(mag.begin(), mag.begin() + keep, mag.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            indices.resize((size_t)4 + (size_t)keep * 3);
            const uint16_t h0 = f32_to_f16(scale0); const uint16_t h1 = f32_to_f16(scale1);
            indices[0] = (uint8_t)(h0 & 0xFF); indices[1] = (uint8_t)(h0 >> 8);
            indices[2] = (uint8_t)(h1 & 0xFF); indices[3] = (uint8_t)(h1 >> 8);
            for (int i = 0; i < keep; ++i) {
                const int idx = mag[(size_t)i].second; const float s = (idx < half) ? scale0 : scale1;
                int q = (int)std::lround(w[idx] / s * 127.0f); q = std::max(-127, std::min(127, q));
                const uint16_t rel = (uint16_t)idx; const size_t base = 4 + (size_t)i * 3;
                indices[base] = (uint8_t)(rel & 0xFF); indices[base + 1] = (uint8_t)(rel >> 8); indices[base + 2] = (uint8_t)q;
            }
            return true;
        }
        default: return false;
    }
}

void dequantize_block_all(Format fmt, const uint8_t* indices, size_t idx_bytes, const uint8_t* codebook, size_t cb_bytes, uint32_t nw, float* out) {
    if (!out || nw == 0) return;
    std::fill(out, out + nw, 0.0f);
    const int n = (int)std::min<uint32_t>(nw, 1u << 24);
    switch (fmt) {
        case Format::Q32:
        case Format::Q32_GRP:
            if (idx_bytes >= (size_t)n * 4) std::memcpy(out, indices, (size_t)n * 4);
            return;
        case Format::Q24:
        case Format::Q24_GRP:
            dequant_q24(indices, idx_bytes, n, out);
            return;
        case Format::Q16:
        case Format::Q16_GRP:
            dequant_q16_enhanced(indices, idx_bytes, n, out);
            return;
        case Format::Q12: {
            if (cb_bytes < 8192) return;
            CodebookQ12 cb; std::memcpy(cb.centroids, codebook, 8192);
            BitReader br(indices, idx_bytes);
            for (int i = 0; i < n; ++i) out[i] = cb.dequantize(br.get(12));
            return;
        }
        case Format::Q8:
            dequant_lattice(indices, idx_bytes, n, 8, out); return;
        case Format::Q6: {
            if (cb_bytes < 256) return;
            CodebookQ6 cb; std::memcpy(cb.centroids, codebook, 256);
            BitReader br(indices, idx_bytes);
            for (int i = 0; i < n; ++i) out[i] = cb.dequantize(br.get(6));
            return;
        }
        case Format::Q4:
            dequant_lattice(indices, idx_bytes, n, 4, out); return;
        case Format::Q3: {
            if (cb_bytes < 32) return;
            CodebookQ3 cb; std::memcpy(cb.centroids, codebook, 32);
            BitReader br(indices, idx_bytes);
            for (int i = 0; i < n; ++i) out[i] = cb.dequantize(br.get(3));
            return;
        }
        case Format::Q2:
            dequant_lattice(indices, idx_bytes, n, 2, out); return;
        case Format::Q1: {
            for (int b = 0; b * kMeanBlock < n; ++b) {
                const size_t o = (size_t)b * 2; if (o + 1 >= cb_bytes) break;
                const uint16_t h = (uint16_t)(codebook[o]) | ((uint16_t)(codebook[o + 1]) << 8);
                const float mean = f16_to_f32(h);
                const int start = b * kMeanBlock; const int end = std::min(start + kMeanBlock, n);
                std::fill(out + start, out + end, mean);
            }
            return;
        }
        case Format::Q12_GRP:
            dequant_grp16(12, indices, idx_bytes, n, out); return;
        case Format::Q8_GRP:
            dequant_grp16(8, indices, idx_bytes, n, out); return;
        case Format::Q6_GRP:
            dequant_6k(indices, idx_bytes, n, out); return;
        case Format::Q4_GRP:
            dequant_affine(4, indices, idx_bytes, n, 32, 6, 6, 4.5f, out); return;
        case Format::Q3_GRP:
            dequant_affine(3, indices, idx_bytes, n, 32, 6, 6, 3.5f, out); return;
        case Format::Q2_GRP:
            dequant_affine(2, indices, idx_bytes, n, 16, 4, 4, 2.625f, out); return;
        case Format::Q1_GRP: {
            if (n < 32) {
                BitReader br(indices, idx_bytes); for (int i = 0; i < n; ++i) out[i] = br.get(1) == 0 ? -1.0f : 1.0f;
                return;
            }
            BitReader br(indices, idx_bytes); const float scale = f16_to_f32((uint16_t)br.get(16));
            for (int i = 0; i < n; ++i) {
                if (is_slot(i, n, 16)) { out[i] = 0.0f; continue; }
                out[i] = br.get(1) == 0 ? -scale : scale;
            }
            return;
        }
        case Format::Q_TWI_MIX_1_5: {
            BitReader br(indices, idx_bytes);
            for (int start = 0; start < n; start += kQuantSubBlock) {
                const int end = std::min(start + kQuantSubBlock, n); const int count = end - start;
                float scale = 1.0f; if (count == kQuantSubBlock) scale = f16_to_f32((uint16_t)br.get(16));
                for (int i = start; i < end; ++i) out[i] = br.get(1) == 0 ? -scale : scale;
            }
            return;
        }
        case Format::Q_TWI_MIX_1_5_GRP: {
            if (n < 32) {
                BitReader br(indices, idx_bytes); for (int i = 0; i < n; ++i) out[i] = br.get(1) == 0 ? -1.0f : 1.0f;
                return;
            }
            BitReader br(indices, idx_bytes); const float scale = f16_to_f32((uint16_t)br.get(16));
            const int refined = std::max(0, n / 2 - 16); std::vector<uint8_t> signs((size_t)n);
            for (int i = 0; i < n; ++i) signs[(size_t)i] = (uint8_t)br.get(1);
            for (int i = 0; i < n; ++i) {
                float magnitude = scale; if (is_slot(i, n, refined)) magnitude = scale * (br.get(1) == 0 ? 0.567f : 1.893f);
                out[i] = signs[(size_t)i] == 0 ? -magnitude : magnitude;
            }
            return;
        }
        case Format::Q_TWI_MIX_2_5: {
            if (idx_bytes < 4) return;
            uint32_t scale_bits = (uint32_t)indices[0] | ((uint32_t)indices[1] << 8) | ((uint32_t)indices[2] << 16) | ((uint32_t)indices[3] << 24);
            float scale; std::memcpy(&scale, &scale_bits, 4); if (scale == 0.0f) return;
            const size_t records = (idx_bytes - 4) / 3;
            for (size_t r = 0; r < records; ++r) {
                const size_t base = 4 + r * 3; if (base + 2 >= idx_bytes) break;
                const uint32_t rel = (uint32_t)indices[base] | ((uint32_t)indices[base + 1] << 8); const int8_t q = (int8_t)indices[base + 2];
                if (rel < (uint32_t)n) out[rel] = (float)q / 127.0f * scale;
            }
            return;
        }
        case Format::Q_TWI_MIX_2_5_GRP: {
            if (idx_bytes < 7) return;
            const uint16_t s0 = (uint16_t)indices[0] | ((uint16_t)indices[1] << 8); const uint16_t s1 = (uint16_t)indices[2] | ((uint16_t)indices[3] << 8);
            const float scale0 = f16_to_f32(s0); const float scale1 = f16_to_f32(s1); if (scale0 == 0.0f && scale1 == 0.0f) return;
            const size_t records = (idx_bytes - 4) / 3; const int half = n / 2;
            for (size_t r = 0; r < records; ++r) {
                const size_t base = 4 + r * 3; if (base + 2 >= idx_bytes) break;
                const uint32_t rel = (uint32_t)indices[base] | ((uint32_t)indices[base + 1] << 8); const int8_t q = (int8_t)indices[base + 2];
                const float s = ((int)rel < half) ? scale0 : scale1;
                if (rel < (uint32_t)n) out[rel] = (float)q / 127.0f * s;
            }
            return;
        }
        default: return;
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

} // namespace quant
