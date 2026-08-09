// ============================================================================
// probe_grp.cpp — REAL-WEIGHT per-group statistics for codec design.
// Loads a model, and for every 256-weight block reports:
//   - offset magnitude |mean(w)| relative to the LS scale (group 16 and 32)
//   - scale dynamic range within a block (max_scale / min_scale)
//   - ideal MSE: scale-only LS fit vs scale+min LS fit (unquantized ceiling)
// Usage: probe_grp <model.gguf>
// ============================================================================
#include "adapters/adapter_core.h"
#include "adapters/gguf_bridge.h"

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>

using namespace quant;
using namespace quant::adapters;

// Weighted-LS scale for a group given fixed levels l_i (weighted linear fit).
static double fit_scale_ls(const double* w, int n, double* lv) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) {
        const double l = lv[i];
        const double wgt = 1.0; // uniform weighting for the ceiling study
        num += wgt * w[i] * l;
        den += wgt * l * l;
    }
    return den > 1e-20 ? num / den : 0.0;
}

// Joint scale+min weighted-LS fit: v_i = s*l_i + m. Returns m, sets s.
static void fit_scale_min_ls(const double* w, int n, const double* lv, double* s, double* m) {
    double sum_w = 0, sum_l = 0, sum_l2 = 0, sum_x = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
        const double wgt = 1.0;
        const double l = lv[i];
        sum_w += wgt; sum_l += wgt * l; sum_l2 += wgt * l * l;
        sum_x += wgt * w[i]; sum_xl += wgt * w[i] * l;
    }
    const double D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0) {
        *s = (sum_w * sum_xl - sum_x * sum_l) / D;
        *m = (sum_l2 * sum_x - sum_l * sum_xl) / D;
    }
}

static double mse_block(const double* w, int n, double s, double m, const double* lv) {
    double e = 0;
    for (int i = 0; i < n; ++i) { const double d = s * lv[i] + m - w[i]; e += d * d; }
    return e / n;
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: probe_grp <model.gguf>\n"); return 2; }
    auto tensors = load_gguf(argv[1], false);
    if (tensors.empty()) { printf("ERROR: no tensors loaded\n"); return 1; }

    std::vector<double> off16, off32;   // |offset| / scale
    std::vector<double> dyn16, dyn32;   // max_scale / min_scale per block
    struct Ceil { double mse = 0; int64_t n = 0; };
    Ceil ceilings[8];                   // 2b g16 | 2b g32+min | 4b g16 | 4b g32+min | 6b g16 | 8b g16 | 8b g32
    int64_t nw_ideal = 0;
    int64_t sat64 = 0, sat32 = 0;       // groups with |offset| > 0.5*scale / 1.0*scale

    for (auto& t : tensors) {
        if (t.numel() < 256) continue;
        const int64_t aligned = (t.numel() / 256) * 256;
        for (int64_t b = 0; b < aligned; b += 256) {
            const float* x = t.data.data() + b;
            double xd[256];
            for (int i = 0; i < 256; ++i) xd[i] = (double)x[i];
            // ---- per-16 ----
            double bs_max = 0, bs_min = 1e30;
            for (int g = 0; g < 16; ++g) {
                double av = 0, maxa = 0;
                for (int i = 0; i < 16; ++i) { const double a = std::fabs((double)x[g*16+i]); av += a; if (a > maxa) maxa = a; }
                if (maxa < 1e-30) continue;
                double s = maxa / 1.5104; // 2-bit-style scale (any representative max-level works for ratios)
                const double off = av / 16.0; // mean |w| as offset proxy -> use signed mean below instead
                (void)off;
                double mean = 0; for (int i = 0; i < 16; ++i) mean += (double)x[g*16+i];
                mean /= 16.0;
                off16.push_back(std::fabs(mean) / s);
                if (std::fabs(mean) / s > 0.5) sat64++;
                if (s > bs_max) bs_max = s;
                if (s < bs_min) bs_min = s;
            }
            dyn16.push_back(bs_max / bs_min);
            // ---- per-32 ----
            bs_max = 0; bs_min = 1e30;
            for (int g = 0; g < 8; ++g) {
                double maxa = 0;
                for (int i = 0; i < 32; ++i) { const double a = std::fabs((double)x[g*32+i]); if (a > maxa) maxa = a; }
                if (maxa < 1e-30) continue;
                double s = maxa / 2.7178;
                double mean = 0; for (int i = 0; i < 32; ++i) mean += (double)x[g*32+i];
                mean /= 32.0;
                off32.push_back(std::fabs(mean) / s);
                if (std::fabs(mean) / s > 0.5) sat32++;
                if (s > bs_max) bs_max = s;
                if (s < bs_min) bs_min = s;
            }
            dyn32.push_back(bs_max / bs_min);

            // ---- ideal ceilings per (bits, structure): 2-pass fit, unquantized scales ----
            static const double L2[]  = { -1.5104, -0.4528, 0.4528, 1.5104 };
            static const double L4[]  = { -2.7178, -2.0522, -1.5995, -1.2392, -0.9275, -0.6508,
                                          -0.3976, -0.1260, 0.1260, 0.3976, 0.6508, 0.9275,
                                          1.2392, 1.5995, 2.0522, 2.7178 };
            double L6[64], L8[256];
            for (int i = 0; i < 64; ++i)  L6[i] = (double)(i - 32);
            for (int i = 0; i < 256; ++i) L8[i] = (double)i / 16.0 - 8.0;

            struct Combo { int bits; const double* lv; int nl; int gsz; bool use_min; const char* name; };
            static const Combo combos[] = {
                { 2, L2, 4,  16, false, "2b g16 scale" },
                { 2, L2, 4,  32, true,  "2b g32 scale+min" },
                { 4, L4, 16, 16, false, "4b g16 scale" },
                { 4, L4, 16, 32, true,  "4b g32 scale+min" },
                { 6, L6, 64, 16, false, "6b g16 scale" },
                { 8, L8, 256,16, false, "8b g16 scale" },
                { 8, L8, 256,32, false, "8b g32 scale" },
            };
            for (const auto& c : combos) {
                const int ng = 256 / c.gsz;
                const double step = (c.bits == 6) ? 1.0 : (c.bits == 8) ? (1.0 / 16.0) : 0.0;
                const bool uniform = (c.bits == 6 || c.bits == 8);
                const double lo = (c.bits == 6) ? -32.0 : -8.0;
                double mse = 0;
                for (int g = 0; g < ng; ++g) {
                    const double* w = xd + (size_t)g * c.gsz;
                    double lvl[256];
                    double maxa = 0;
                    for (int i = 0; i < c.gsz; ++i) maxa = std::max(maxa, std::fabs(w[i]));
                    if (maxa < 1e-30) continue;
                    double s = maxa / std::fabs(c.lv[c.nl - 1]);
                    double m = 0;
                    for (int iter = 0; iter < 3; ++iter) {
                        for (int i = 0; i < c.gsz; ++i) {
                            const double v = (c.use_min ? (w[i] - m) : w[i]) / s;
                            if (uniform) {
                                const int k = (int)std::lround((v - lo) / step);
                                const int kk = std::max(0, std::min(c.nl - 1, k));
                                lvl[i] = c.lv[kk];
                            } else {
                                int best = 0; double bd = 1e30;
                                for (int k = 0; k < c.nl; ++k) {
                                    const double d = c.lv[k] - v; if (d * d < bd) { bd = d * d; best = k; }
                                }
                                lvl[i] = c.lv[best];
                            }
                        }
                        if (c.use_min) fit_scale_min_ls(w, c.gsz, lvl, &s, &m);
                        else           s = fit_scale_ls(w, c.gsz, lvl);
                    }
                    mse += mse_block(w, c.gsz, s, c.use_min ? m : 0.0, lvl) * c.gsz;
                }
                mse /= 256.0;
                const int sidx = (c.bits == 2 && c.use_min) ? 1 : (c.bits == 4 && c.use_min) ? 3
                              : (c.bits == 8 && c.gsz == 32) ? 6 : (c.bits == 8) ? 5
                              : (c.bits == 6) ? 4 : (c.bits == 2) ? 0 : 2;
                ceilings[sidx].mse += mse * 256;
                ceilings[sidx].n  += 256;
            }
            nw_ideal += 256;
        }
    }

    auto pct = [](std::vector<double>& v, double p) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const size_t i = (size_t)(p * (double)(v.size() - 1));
        return v[i];
    };

    printf("probe_grp: %zu tensors\n", tensors.size());
    printf("group16 offset/scale: p50=%.3f p90=%.3f p99=%.3f max=%.3f (>0.5x frac %.2f%%)\n",
           pct(off16,0.5), pct(off16,0.9), pct(off16,0.99), pct(off16,1.0),
           100.0*(double)sat64/(double)std::max<size_t>(1,off16.size()));
    printf("group32 offset/scale: p50=%.3f p90=%.3f p99=%.3f max=%.3f (>0.5x frac %.2f%%)\n",
           pct(off32,0.5), pct(off32,0.9), pct(off32,0.99), pct(off32,1.0),
           100.0*(double)sat32/(double)std::max<size_t>(1,off32.size()));
    printf("block scale dyn range: g16 p50=%.1f p99=%.1f   g32 p50=%.1f p99=%.1f\n",
           pct(dyn16,0.5), pct(dyn16,0.99), pct(dyn32,0.5), pct(dyn32,0.99));
    if (nw_ideal > 0) {
        const char* names[8] = { "2b g16 scale", "2b g32 scale+min", "4b g16 scale", "4b g32 scale+min",
                                 "6b g16 scale", "8b g16 scale", "8b g32 scale" };
        printf("ideal ceilings (2-pass fit, unquantized scales):\n");
        for (int i = 0; i < 7; ++i)
            if (ceilings[i].n > 0)
                printf("  %-16s PSNR=%.2f dB\n", names[i],
                       10.0 * log10((double)ceilings[i].n / ceilings[i].mse));
    }
    return 0;
}
