// ============================================================================
// probe_search.cpp — search-based fitter ceilings + scale-quantization loss.
// Answers the codec design questions on real weights (630M qwen2.5):
//   1. Does a grid-search fitter beat the current 2-pass weighted-LS fitter?
//   2. Do Gaussian (Lloyd-Max-ish) levels beat uniform at 6-bit?
//   3. How much do 7-bit linear / 7-bit sqrt / int8 scale encodings + FP16 d
//      cost vs the unquantized ceiling?
//   4. Is g32 scale+min (affine {0..nmax} codebook) the winning 4-bit / 2-bit
//      structure?
// PSNR is plain MSE-based with peak 1 (identical to bench_industry_real), so
// the numbers are directly comparable to industry: Q4_K=53.56, Q6_K=64.66,
// Q8_0=76.44, Q2_K=41.83.
// Usage: probe_search <model.gguf>
// ============================================================================
#include "adapters/adapter_core.h"
#include "adapters/gguf_bridge.h"

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

using namespace quant;
using namespace quant::adapters;

// ---- FP16 snap (self-contained, matches block_codec) ------------------------
static uint16_t f32_to_f16(float v) {
    uint32_t bits; std::memcpy(&bits, &v, 4);
    const int sign = (int)(bits >> 31) & 1;
    int exp = (int)((bits >> 23) & 0xFF) - 127;
    int mant = (int)(bits & 0x7FFFFF);
    uint16_t h;
    if (exp > 15) h = (uint16_t)((sign << 15) | 0x7C00);
    else if (exp >= -14) {
        int m = (mant >> 13) | 0x400;
        const int round_bit = (mant >> 12) & 1, sticky = (mant & 0xFFF) ? 1 : 0;
        const int lsb = m & 1;
        if (round_bit && (sticky || lsb)) m += 1;
        if (m >= 0x800) { m >>= 1; exp += 1; if (exp > 15) { h = (uint16_t)((sign << 15) | 0x7C00); return h; } }
        h = (uint16_t)((sign << 15) | ((exp + 15) << 10) | (m & 0x3FF));
    } else {
        if (exp < -24) h = (uint16_t)(sign << 15);
        else {
            int m = mant | 0x800000;
            const int shift = -exp - 14 + 13;
            int mf = m >> shift;
            const int round_bit = (m >> (shift - 1)) & 1, sticky = ((m & ((1 << (shift - 1)) - 1)) ? 1 : 0);
            const int lsb = mf & 1;
            if (round_bit && (sticky || lsb)) mf += 1;
            if (mf >= 0x400) h = (uint16_t)((sign << 15) | (1 << 10) | (mf & 0x3FF));
            else h = (uint16_t)((sign << 15) | mf);
        }
    }
    return h;
}
static float f16_to_f32(uint16_t h) {
    const int sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = (uint32_t)sign << 31;
        else { int e = -14, m = mant; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF;
               bits = ((uint32_t)sign << 31) | (uint32_t)(e + 127) << 23 | ((uint32_t)m << 13); }
    } else if (exp == 31) bits = ((uint32_t)sign << 31) | 0x7F800000 | ((uint32_t)mant << 13);
    else bits = ((uint32_t)sign << 31) | (uint32_t)(exp - 15 + 127) << 23 | ((uint32_t)mant << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}
static float snap16(float v) { return f16_to_f32(f32_to_f16(v)); }

// ---- exact inverse error function (Newton on std::erf) ----------------------
static double erfinv_exact(double x) {
    double s = (x < 0) ? -1.0 : 1.0;
    const double a = std::fabs(x);
    if (a >= 1.0) return s * 1e7;
    const double cc = 0.147;
    const double ln = std::log(1.0 - a * a);
    double y = std::sqrt(std::sqrt((2.0 / (3.141592653589793 * cc) + ln / 2.0) *
                                   (2.0 / (3.141592653589793 * cc) + ln / 2.0) - ln / cc) -
                         (2.0 / (3.141592653589793 * cc) + ln / 2.0));
    for (int i = 0; i < 10; ++i) {
        const double e = std::erf(y) - a;
        const double de = 1.1283791670955126 * std::exp(-y * y); // 2/sqrt(pi)
        y -= e / de;
    }
    return s * y;
}

// Symmetric level sets (max |level| = 1).  Gaussian = exact erf quantiles.
static std::vector<double> make_levels(bool gaussian, int n) {
    std::vector<double> lv((size_t)n);
    if (!gaussian) {
        for (int i = 0; i < n; ++i) lv[(size_t)i] = (double)(2 * i - n + 1) / (double)(n - 1);
    } else {
        for (int i = 0; i < n; ++i)
            lv[(size_t)i] = 1.4142135623730951 * erfinv_exact(2.0 * ((double)i + 0.5) / (double)n - 1.0);
        double mx = 0;
        for (double v : lv) mx = std::max(mx, std::fabs(v));
        if (mx > 0) for (auto& v : lv) v /= mx;
    }
    return lv;
}
// Affine codebook {0..n-1}/(n-1) in [0,1] (industry Q2_K/Q4_K style).
static std::vector<double> make_affine_levels(int n) {
    std::vector<double> lv((size_t)n);
    for (int i = 0; i < n; ++i) lv[(size_t)i] = (double)i / (double)(n - 1);
    return lv;
}
// The codec's exact 2/4-bit Gaussian lattices (normalized to max |level| = 1).
static std::vector<double> codec_levels_2() {
    return { -1.0, -0.4528 / 1.5104, 0.4528 / 1.5104, 1.0 };
}
static std::vector<double> codec_levels_4() {
    const double kL4[16] = { -2.7178f, -2.0522f, -1.5995f, -1.2392f, -0.9275f, -0.6508f,
                             -0.3976f, -0.1260f, 0.1260f, 0.3976f, 0.6508f, 0.9275f,
                             1.2392f, 1.5995f, 2.0522f, 2.7178f };
    std::vector<double> lv;
    for (double v : kL4) lv.push_back(v / 2.7178);
    return lv;
}

// ---- combo + encoding descriptor ---------------------------------------------
struct Enc {           // budget-legal scale encoding
    int sc_bits;       // scale bits (7 = 7-bit, 8 = 8-bit, 4 = 4-bit)
    int min_bits;      // affine min bits (0 = none)
    bool has_d;        // FP16 global anchor
    bool has_dm;       // FP16 min anchor
    bool sqrt_map;     // 7-bit sqrt scale mapping
    bool signed_sc;    // int8 scale (6-bit tiers); unsigned otherwise
};
struct Combo {
    const char* name;
    int bits;             // level bits
    int gsz;              // group size
    bool affine;          // scale+min
    bool min_anchor;      // industry data-min anchored assignment
    std::vector<double> levels;
    Enc enc;
    int ng;               // groups per 256-block
};

// Lloyd-style fit of one group: alternate (nearest-level assignment for the
// current (s, m)) and (plain unweighted LS refit of (s, m) — MSE-optimal given
// the fixed assignment, matching the bench metric exactly).  4 iterations, then
// a scale-fac grid polish (0.85..1.15 step 0.05, re-assign + LS, keep the
// assignment with min plain MSE).  `min_anchor` seeds m at the data min; the
// affine min is unclamped (a signed FP16 dm carries a positive DC offset for
// all-positive groups, native improvement over industry's unsigned-min).
static void fit_search(const double* w, int n, const std::vector<double>& lv,
                       const std::vector<int16_t>& tab, bool affine, bool min_anchor,
                       double* s_out, double* m_out, double* mse_out) {
    double av = 0, maxa = 0, maxw = -1e30, data_min = 1e30;
    for (int i = 0; i < n; ++i) {
        const double ax = std::fabs(w[i]);
        av += ax; maxa = std::max(maxa, ax);
        maxw = std::max(maxw, w[i]); data_min = std::min(data_min, w[i]);
    }
    av /= (double)n;
    *s_out = 0; *m_out = 0; *mse_out = 0;
    if (maxa < 1e-30) return;
    const double min_lvl = lv.front(), max_lvl = lv.back();
    double s = affine ? (maxw - data_min) / (max_lvl - min_lvl) : maxa / max_lvl;
    double m = min_anchor ? data_min : 0.0;
    double l[256];

    auto assign = [&](double ss, double mm) {
        for (int i = 0; i < n; ++i) {
            const double v = (w[i] - mm) / ss;
            int bin = (int)((v + 2.0) * 0.25 * 16383.0 + 0.5);
            bin = std::max(0, std::min(16383, bin));
            l[i] = lv[(size_t)tab[(size_t)bin]];
        }
    };
    auto lsf = [&](double& ss, double& mm) {
        double sw = 0, sl = 0, sl2 = 0, sx = 0, sxl = 0;
        for (int i = 0; i < n; ++i) {
            const double li = l[i];
            sw += 1.0; sl += li; sl2 += li * li;
            sx += w[i]; sxl += w[i] * li;
        }
        if (affine) {
            const double D = sw * sl2 - sl * sl;
            if (D > 1e-20) { ss = (sw * sxl - sx * sl) / D; mm = (sl2 * sx - sl * sxl) / D; }
        } else {
            if (sl2 > 1e-20) ss = sxl / sl2;
            mm = 0;
        }
    };
    auto plain_mse = [&](double ss, double mm) {
        assign(ss, mm);
        double e = 0;
        for (int i = 0; i < n; ++i) { const double d = ss * l[i] + mm - w[i]; e += d * d; }
        return e / (double)n;
    };
    for (int iter = 0; iter < 4; ++iter) {
        assign(s, m); lsf(s, m);
    }
    double best_s = s, best_m = m, best_err = plain_mse(s, m);
    for (int k = -3; k <= 3; ++k) {
        if (k == 0) continue;
        const double cs = s * (1.0 + 0.05 * (double)k);
        double s2 = cs, m2 = m;
        assign(cs, m2); lsf(s2, m2);
        const double e = plain_mse(s2, m2);
        if (e < best_err) { best_err = e; best_s = s2; best_m = m2; }
    }
    assign(best_s, best_m);
    double e2 = 0;
    for (int i = 0; i < n; ++i) { const double d = best_s * l[i] + best_m - w[i]; e2 += d * d; }
    *s_out = best_s; *m_out = best_m; *mse_out = e2 / (double)n;
}

// Apply the budget-legal encoding; fills effective per-group (s, m).
// Anchor d at max_s, snap each scale, then refit d by plain LS over the snapped
// codes (d = sum(s_g*sc_g)/sum(sc_g^2), FP16-snapped) — the same joint
// refinement the codec performs.  Affine min is stored as an unsigned magnitude
// (subtracted), dm refit by LS the same way.
static void quantize_scales(int ng, const std::vector<double>& s_in,
                            const std::vector<double>& m_in, const Enc& enc,
                            std::vector<double>& s_out, std::vector<double>& m_out) {
    s_out.assign((size_t)ng, 0.0); m_out.assign((size_t)ng, 0.0);
    double max_s = 0, max_m = 0;
    for (int g = 0; g < ng; ++g) max_s = std::max(max_s, std::fabs(s_in[(size_t)g]));
    for (int g = 0; g < ng; ++g) max_m = std::max(max_m, std::fabs(m_in[(size_t)g]));
    if (max_s < 1e-30) return;
    const int qmax = (1 << enc.sc_bits) - 1;   // 7-bit -> 127, 4-bit -> 15
    std::vector<double> sc((size_t)ng, 0.0);
    if (enc.sqrt_map) {
        const double d = snap16((float)max_s);
        if (d <= 0) return;
        for (int g = 0; g < ng; ++g) {
            const double s = std::fabs(s_in[(size_t)g]);
            int c = (int)std::lround((double)qmax * std::sqrt(s / d));
            c = std::max(0, std::min(qmax, c));
            s_out[(size_t)g] = d * ((double)c / (double)qmax) * ((double)c / (double)qmax);
        }
        return;
    } else {
        const double anchor = enc.signed_sc ? 128.0 : (double)qmax;
        double d = snap16((float)(max_s / anchor));
        if (d <= 0) return;
        for (int g = 0; g < ng; ++g) {
            const double s = std::fabs(s_in[(size_t)g]);
            int c = (int)std::lround(s / d);
            if (enc.signed_sc) c = std::max(-127, std::min(127, c)); // signed int8
            else c = std::max(0, std::min(qmax, c));
            sc[(size_t)g] = (double)c;
        }
        // LS-optimal FP16 d given the snapped codes (codec joint refinement).
        double num = 0, den = 0;
        for (int g = 0; g < ng; ++g) { num += std::fabs(s_in[(size_t)g]) * sc[(size_t)g]; den += sc[(size_t)g] * sc[(size_t)g]; }
        if (den > 1e-20) d = snap16((float)(num / den));
        for (int g = 0; g < ng; ++g) s_out[(size_t)g] = d * sc[(size_t)g];
    }
    if (enc.min_bits > 0 && max_m > 1e-30) {
        const int mq = (1 << enc.min_bits) - 1;
        std::vector<double> mc((size_t)ng, 0.0);
        // signed dm: negative for a dominant positive DC offset (all-positive
        // groups), positive otherwise; the refit below keeps the sign.
        double m_pos = 0, m_neg = 0;
        for (int g = 0; g < ng; ++g) {
            if (m_in[(size_t)g] > 0) m_pos += m_in[(size_t)g]; else m_neg += -m_in[(size_t)g];
        }
        const double dm_sign = (m_pos > m_neg) ? -1.0 : 1.0;
        double dm = dm_sign * snap16((float)(max_m / (double)mq));
        if (std::fabs(dm) <= 1e-30) return;
        for (int g = 0; g < ng; ++g) {
            const double v = -m_in[(size_t)g] / dm;  // positive for the dominant sign
            int c = (v <= 0.0) ? 0 : std::max(0, std::min(mq, (int)std::lround(v)));
            mc[(size_t)g] = (double)c;
        }
        double num = 0, den = 0;
        for (int g = 0; g < ng; ++g) { num += -m_in[(size_t)g] * mc[(size_t)g]; den += mc[(size_t)g] * mc[(size_t)g]; }
        if (den > 1e-20) dm = snap16((float)(num / den));
        for (int g = 0; g < ng; ++g) m_out[(size_t)g] = -dm * mc[(size_t)g]; // signed dm
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: probe_search <model.gguf>\n"); return 2; }
    auto tensors = load_gguf(argv[1], false);
    if (tensors.empty()) { printf("ERROR: no tensors loaded\n"); return 1; }

    const std::vector<double> L2 = codec_levels_2(), L4 = codec_levels_4();
    const std::vector<double> L6u = make_levels(false, 64), L6g = make_levels(true, 64);
    const std::vector<double> L8u = make_levels(false, 256);
    const std::vector<double> A16 = make_affine_levels(16), A4 = make_affine_levels(4);
    const std::vector<double> A64 = make_affine_levels(64);

    auto build_tab = [](const std::vector<double>& lv) {
        std::vector<int16_t> tab(16384);
        for (int bin = 0; bin < 16384; ++bin) {
            const double v = (double)bin / 16383.0 * 4.0 - 2.0;
            int best = 0; double bd = 1e30;
            for (size_t k = 0; k < lv.size(); ++k) {
                const double d = lv[k] - v;
                if (d * d < bd) { bd = d * d; best = (int)k; }
            }
            tab[(size_t)bin] = (int16_t)best;
        }
        return tab;
    };

    // (name, bits, gsz, affine, min_anchor, levels, enc)
    // `*` = 84 B/256w (2.625 bpw, matches Q2_K budget); all others are the
    // honest 80/144/210/272 B claims.
    static const Combo combos[] = {
        { "4b g16 sc gauss sc7",       4, 16, false, false, L2 /*unused*/, Enc{7, 0, true, false, false, false}, 0 },
        { "4b g32 sc+min aff sc7+5",   4, 32, true,  true,  A16,             Enc{7, 5, true, true,  false, false}, 0 },
        { "4b g32 sc+min aff sc6+6",   4, 32, true,  true,  A16,             Enc{6, 6, true, true,  false, false}, 0 },
        { "6b g16 sc unif i8",         6, 16, false, false, L6u,             Enc{8, 0, true, false, false, true }, 0 },
        { "6b g16 sc gauss i8",        6, 16, false, false, L6g,             Enc{8, 0, true, false, false, true }, 0 },
        { "6b g32 sc+min aff sc7+7",   6, 32, true,  true,  A64,             Enc{7, 7, true, true,  false, false}, 0 },
        { "6b g32 sc+min aff sc6+6",   6, 32, true,  true,  A64,             Enc{6, 6, true, true,  false, false}, 0 },
        { "6b g16 sc+min aff sc3+3",   6, 16, true,  true,  A64,             Enc{3, 3, true, true,  false, false}, 0 },
        { "8b g16 sc unif sc7",        8, 16, false, false, L8u,             Enc{7, 0, true, false, false, false}, 0 },
        { "8b g16 sc unif sq7",        8, 16, false, false, L8u,             Enc{7, 0, true, false, true,  false}, 0 },
        { "8b g32 sc unif sc7",        8, 32, false, false, L8u,             Enc{7, 0, true, false, false, false}, 0 },
        { "2b g32 sc+min aff sc7+5",   2, 32, true,  true,  A4,              Enc{7, 5, true, true,  false, false}, 0 },
        { "2b g32 sc+min aff sc6+6",   2, 32, true,  true,  A4,              Enc{6, 6, true, true,  false, false}, 0 },
        { "2b g16 sc+min aff sc4+2",   2, 16, true,  true,  A4,              Enc{4, 2, true, true,  false, false}, 0 },
        { "2b g16 sc+min aff sc3+3",   2, 16, true,  true,  A4,              Enc{3, 3, true, true,  false, false}, 0 },
        { "2b g16 sc+min aff sc5+3",   2, 16, true,  true,  A4,              Enc{5, 3, true, true,  false, false}, 0 },
        { "2b g16 sc+min aff sc4+4*",  2, 16, true,  true,  A4,              Enc{4, 4, true, true,  false, false}, 0 },
    };
    const int ncombos = (int)(sizeof(combos) / sizeof(combos[0]));
    struct Acc { double mse_ceiling = 0, mse_real = 0; int64_t n = 0; };
    std::vector<Acc> accs((size_t)ncombos);
    // level table per combo (patch the placeholder)
    std::vector<std::vector<int16_t>> tabs;
    for (const Combo& c : combos) {
        const std::vector<double>* lv = nullptr;
        if (c.bits == 2) lv = c.affine ? &A4 : &L2;
        else if (c.bits == 4) lv = c.affine ? &A16 : &L4;
        else if (c.bits == 6) lv = c.affine ? &A64 : &c.levels;
        else lv = &c.levels;
        tabs.push_back(build_tab(*lv));
    }
    int64_t nw = 0;

    for (auto& t : tensors) {
        if (t.numel() < 256) continue;
        const int64_t aligned = (t.numel() / 256) * 256;
        for (int64_t b = 0; b < aligned; b += 256) {
            const float* x = t.data.data() + b;
            double xd[256];
            for (int i = 0; i < 256; ++i) xd[i] = (double)x[i];
            for (int ci = 0; ci < ncombos; ++ci) {
                const Combo& c = combos[ci];
                const std::vector<double>& lv = (c.bits == 2) ? (c.affine ? A4 : L2) :
                        (c.bits == 4) ? (c.affine ? A16 : L4) :
                        (c.bits == 6) ? (c.affine ? A64 : c.levels) : c.levels;
                const std::vector<int16_t>& tab = tabs[(size_t)ci];
                const int ng = 256 / c.gsz;
                double mse_ceil = 0, mse_real = 0;
                std::vector<double> s_in((size_t)ng), m_in((size_t)ng);
                for (int g = 0; g < ng; ++g) {
                    double s = 0, m = 0, me = 0;
                    fit_search(xd + (size_t)g * c.gsz, c.gsz, lv, tab, c.affine, c.min_anchor, &s, &m, &me);
                    mse_ceil += me * (double)c.gsz;
                    s_in[(size_t)g] = s; m_in[(size_t)g] = m;
                }
                mse_ceil /= 256.0;
                std::vector<double> sq, mq;
                quantize_scales(ng, s_in, m_in, c.enc, sq, mq);
                double e = 0;
                for (int g = 0; g < ng; ++g) {
                    const double* w = xd + (size_t)g * c.gsz;
                    const double ss = sq[(size_t)g], mm = c.affine ? mq[(size_t)g] : 0.0;
                    if (ss == 0.0) continue;
                    for (int i = 0; i < c.gsz; ++i) {
                        const double v = (w[i] - mm) / ss;
                        int bin = (int)((v + 2.0) * 0.25 * 16383.0 + 0.5);
                        bin = std::max(0, std::min(16383, bin));
                        const double lvl = lv[(size_t)tab[(size_t)bin]];
                        const double d = ss * lvl + mm - w[i];
                        e += d * d;
                    }
                }
                mse_real = e / 256.0;
                accs[(size_t)ci].mse_ceiling += mse_ceil * 256.0;
                accs[(size_t)ci].mse_real += mse_real * 256.0;
                accs[(size_t)ci].n += 256;
            }
            nw += 256;
            if ((nw % 50000000) == 0) {
                printf("  ... %d M weights processed\n", (int)(nw / 1000000));
                fflush(stdout);
            }
        }
    }

    printf("probe_search: %zu tensors, %.1fM weights\n\n", tensors.size(), (double)nw / 1e6);
    printf("%-28s %10s %10s\n", "combo", "ceiling dB", "realized dB");
    printf("%-28s %10s %10s\n", "-----", "----------", "----------");
    for (int i = 0; i < ncombos; ++i) {
        if (accs[(size_t)i].n == 0) continue;
        const double p_c = 10.0 * log10((double)accs[(size_t)i].n / accs[(size_t)i].mse_ceiling);
        const double p_r = 10.0 * log10((double)accs[(size_t)i].n / accs[(size_t)i].mse_real);
        printf("%-28s %10.2f %10.2f\n", combos[i].name, p_c, p_r);
    }
    printf("\nreference: Q4_K=53.56  Q6_K=64.66  Q8_0=76.44  Q2_K=41.83  (bench_industry_real)\n");
    return 0;
}
