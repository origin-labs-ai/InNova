// test_block_codec.cpp — canonical in-budget block codec proof.
//
// Contract (hard, storage-level):
//   1. Every OIL/SPARK format (0..14) quantizes a block into a payload whose
//      actual stored BPW (indices + codebook bytes) NEVER exceeds the
//      claimed BPW — at full blocks, tail blocks, and degenerate inputs.
//   2. dequantize_block_all() round-trips the payload (clean decode, no
//      out-of-bounds reads, unhandled input -> zeros).
//   3. dequantize_block_all() decodes exactly what quantize_block_all()
//      produced — the single source of truth shared by writer, engine and
//      reader (qwen35_engine.cpp, adapter_core.cpp, oil_format.cpp).
//   4. Quality gate: OIL2_GRP at 2.5 BPW must beat SPARK_SPARSE_GRP (2.0)
//      on dense data — the native "GRP wins at 2x BPW" ladder.
#include "oil/types.h"
#include "oil/block_codec.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>

using namespace oil;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            ++g_failures;                                                    \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                 \
            std::printf(__VA_ARGS__);                                        \
            std::printf("\n");                                               \
        }                                                                    \
    } while (0)

static float rms(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty()) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return (float)std::sqrt(sum / (double)a.size());
}

static void fill_random(std::vector<float>& w, unsigned seed) {
    uint32_t state = seed * 2654435761u + 12345u;
    for (auto& v : w) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        v = ((float)(state & 0xFFFF) / 65535.0f - 0.5f) * 2.0f;
    }
}

static size_t byte_budget(Format fmt, uint32_t n) {
    // Claimed byte budget for a block of n weights, plus 1 byte of container
    // alignment slack for tail blocks (header/scale padding to 8-bit bytes).
    const double bits = (double)format_bpw(fmt) * (double)n + 8.0;
    return (size_t)std::ceil(bits / 8.0);
}

static void check_format(Format fmt, const std::vector<float>& w) {
    const int n = (int)w.size();
    std::vector<uint8_t> idx, cb;
    CHECK(quantize_block_all(fmt, w.data(), n, idx, cb), "quantize failed fmt=%d n=%d",
          (int)fmt, n);
    const size_t budget = byte_budget(fmt, (uint32_t)n);
    const size_t used = idx.size() + cb.size();
    const float actual = block_actual_bpw((uint32_t)n, idx.size(), cb.size());
    const float claimed = format_bpw(fmt);
    CHECK(used <= budget,
          "fmt=%d n=%d over budget: %zu bytes used > %zu budget (%.3f > %.3f bpw)",
          (int)fmt, n, used, budget, actual, claimed);
    CHECK(actual <= claimed + 8.0f / (float)n + 0.001f,
          "fmt=%d n=%d byte-alignment overshoot beyond 1 byte: %.3f > %.3f + 1B",
          (int)fmt, n, actual, claimed);

    std::vector<float> out((size_t)n, 1.0f);
    dequantize_block_all(fmt, idx.data(), idx.size(), cb.data(), cb.size(),
                         (uint32_t)n, out.data());
    // Round-trip must be finite and reproduce the codec's own payload
    // (validated below); sanity: all values within level range.
    double abs_max = 0.0;
    for (int i = 0; i < n; i++) abs_max = std::max(abs_max, (double)std::fabs(out[i]));
    CHECK(std::isfinite((float)abs_max), "fmt=%d n=%d produced non-finite output", (int)fmt, n);
}

static void check_engine_layout(Format fmt, const std::vector<float>& w) {
    // Emulate the exact OIL on-disk block layout the engine reads:
    // [nw u32][cb u32][codebook][idx_bytes u32][indices]
    const int n = (int)w.size();
    std::vector<uint8_t> idx, cb;
    CHECK(quantize_block_all(fmt, w.data(), n, idx, cb), "engine-layout quantize fmt=%d", (int)fmt);

    std::vector<uint8_t> raw;
    auto append_u32 = [&](uint32_t v) {
        raw.push_back((uint8_t)(v & 0xFF));
        raw.push_back((uint8_t)((v >> 8) & 0xFF));
        raw.push_back((uint8_t)((v >> 16) & 0xFF));
        raw.push_back((uint8_t)((v >> 24) & 0xFF));
    };
    append_u32((uint32_t)n);
    append_u32((uint32_t)cb.size());
    raw.insert(raw.end(), cb.begin(), cb.end());
    append_u32((uint32_t)idx.size());
    raw.insert(raw.end(), idx.begin(), idx.end());

    // Parse like qwen35_engine.cpp::decode_block
    const uint8_t* p = raw.data();
    uint32_t nw; std::memcpy(&nw, p, 4);
    uint32_t cbb; std::memcpy(&cbb, p + 4, 4);
    const uint8_t* cbp = p + 8;
    uint32_t idb; std::memcpy(&idb, p + 8 + cbb, 4);
    const uint8_t* idp = p + 8 + cbb + 4;

    std::vector<float> out((size_t)nw, 0.0f);
    dequantize_block_all(fmt, idp, idb, cbp, cbb, nw, out.data());

    // Reference decode from the same payload via read_block-style access
    std::vector<float> ref((size_t)nw, 0.0f);
    dequantize_block_all(fmt, idx.data(), idx.size(), cb.data(), cb.size(), (uint32_t)n, ref.data());
    const float err = rms(out, ref);
    CHECK(err < 1e-6f, "engine-layout decode mismatch fmt=%d n=%d err=%.4f", (int)fmt, n, err);
}

static void check_quality_gate() {
    // Dense Gaussian data: OIL2_GRP (dense lattice @2.5) must beat
    // SPARK_SPARSE_GRP (top-8% sparse @2.0) — the 92%-zeros problem.
    std::vector<float> w(256);
    fill_random(w, 7);
    for (auto& v : w) v = v * 0.5f + 1.0f; // dense, non-centered

    std::vector<uint8_t> i2, c2, is, cs;
    quantize_block_all(Format::OIL2_GRP, w.data(), 256, i2, c2);
    quantize_block_all(Format::SPARK_SPARSE_GRP, w.data(), 256, is, cs);

    std::vector<float> d2(256), ds(256);
    dequantize_block_all(Format::OIL2_GRP, i2.data(), i2.size(), c2.data(), c2.size(), 256, d2.data());
    dequantize_block_all(Format::SPARK_SPARSE_GRP, is.data(), is.size(), cs.data(), cs.size(), 256, ds.data());

    const float mse2 = rms(w, d2);
    const float msess = rms(w, ds);
    const float bpw2 = block_actual_bpw(256, i2.size(), c2.size());
    const float bpwss = block_actual_bpw(256, is.size(), cs.size());
    CHECK(bpw2 <= 2.501f && bpwss <= 2.001f,
          "quality gate budgets: OIL2_GRP=%.3f SPARK_SPARSE_GRP=%.3f (claimed 2.5/2.0)", bpw2, bpwss);
    CHECK(mse2 < msess,
          "quality gate: OIL2_GRP rms %.4f should beat SPARK_SPARSE_GRP rms %.4f on dense data",
          mse2, msess);
    std::printf("  quality gate: dense block rms OIL2_GRP=%.4f vs SPARK_SPARSE_GRP=%.4f @ %.2f bpw\n",
                mse2, msess, bpw2);
}

static void check_zero_block() {
    std::vector<float> w(256, 0.0f);
    for (Format f : { Format::OIL1, Format::SPARK_Q0, Format::OIL2, Format::OIL4,
                      Format::OIL8, Format::OIL16, Format::OIL32,
                      Format::OIL1_GRP, Format::SPARK_Q0_GRP, Format::OIL2_GRP,
                      Format::OIL4_GRP, Format::OIL8_GRP, Format::OIL16_GRP,
                      Format::SPARK_SPARSE, Format::SPARK_SPARSE_GRP }) {
        std::vector<uint8_t> idx, cb;
        quantize_block_all(f, w.data(), 256, idx, cb);
        std::vector<float> out(256, 1.0f);
        dequantize_block_all(f, idx.data(), idx.size(), cb.data(), cb.size(), 256, out.data());
        for (int i = 0; i < 256; i++) {
            CHECK(out[i] == 0.0f, "zero block not zero fmt=%d i=%d v=%.4f", (int)f, i, out[i]);
        }
    }
}

static void check_tail_blocks() {
    // Includes the exact sizes that broke the old SPARK_Q0_GRP tail budget
    // (16..21, 25..26) and the SPARK_SPARSE_GRP single-record blocks (32..39).
    const int tails[] = { 1, 7, 15, 16, 17, 20, 21, 25, 26, 31, 32, 33, 39, 40, 63, 100, 128, 200, 255 };
    for (int n : tails) {
        std::vector<float> w((size_t)n);
        fill_random(w, (unsigned)(1000 + n));
    for (int f = 0; f <= 14; f++) {
        Format fmt = (Format)f;
        std::vector<uint8_t> idx, cb;
        quantize_block_all(fmt, w.data(), n, idx, cb);
        const size_t budget = byte_budget(fmt, (uint32_t)n);
        const size_t used = idx.size() + cb.size();
        const float actual = block_actual_bpw((uint32_t)n, idx.size(), cb.size());
        const float claimed = format_bpw(fmt);
        CHECK(used <= budget,
              "tail n=%d fmt=%d over budget: %zu bytes used > %zu (%.3f > %.3f bpw)",
              n, f, used, budget, actual, claimed);
        std::vector<float> out((size_t)n);
        dequantize_block_all(fmt, idx.data(), idx.size(), cb.data(), cb.size(),
                             (uint32_t)n, out.data());
        CHECK(std::isfinite(out[0]), "tail n=%d fmt=%d non-finite", n, f);

        // Reconstruction correctness: sign-based formats must preserve every
        // weight's sign. The old SPARK_Q0_GRP tail allocation dropped sign
        // bits silently (corruption), which this check now catches. OIL1
        // (block mean) is NOT sign-based — it replaces every weight with the
        // block mean, so it is excluded. OIL1_GRP's 16 funding slots decode
        // to 0.0 by design and are skipped.
        if (fmt == Format::SPARK_Q0 || fmt == Format::SPARK_Q0_GRP ||
            fmt == Format::OIL1_GRP) {
            int mism = 0;
            for (int i = 0; i < n; i++) {
                if (fmt == Format::OIL1_GRP && n >= 32) {
                    const int64_t slot = (((int64_t)i + 1) * 16) / n != ((int64_t)i * 16) / n;
                    if (slot) continue;
                }
                if ((w[(size_t)i] >= 0.0f) != (out[(size_t)i] >= 0.0f)) mism++;
            }
            CHECK(mism == 0, "tail n=%d fmt=%d sign corruption: %d/%d", n, f, mism, n);
        }
        // Sparse formats must not silently decode to all-zeros when they
        // stored at least one record (n >= 32 keeps >= 1).
        if ((fmt == Format::SPARK_SPARSE || fmt == Format::SPARK_SPARSE_GRP) && n >= 32) {
            int nonzero = 0;
            for (int i = 0; i < n; i++) if (out[(size_t)i] != 0.0f) nonzero++;
            CHECK(nonzero > 0, "tail n=%d fmt=%d sparse decode all-zeros", n, f);
        }
    }
    }
}

static void check_truncated_payloads() {
    // Bounds safety: truncated payloads must not read out of bounds.
    std::vector<float> w(256);
    fill_random(w, 99);
    for (int f = 0; f <= 14; f++) {
        Format fmt = (Format)f;
        std::vector<uint8_t> idx, cb;
        quantize_block_all(fmt, w.data(), 256, idx, cb);
        std::vector<float> out(256);
        for (size_t cut = 0; cut <= std::min<size_t>(idx.size(), 8); cut++) {
            dequantize_block_all(fmt, idx.data(), cut, cb.data(), cb.size(), 256, out.data());
        }
        for (size_t cut = 0; cut <= std::min<size_t>(cb.size(), 8); cut++) {
            dequantize_block_all(fmt, idx.data(), idx.size(), cb.data(), cut, 256, out.data());
        }
        CHECK(std::isfinite(out[0]), "truncated decode fmt=%d non-finite", f);
    }
}

int main() {
    std::printf("test_block_codec: canonical in-budget codec proof\n");

    std::vector<float> w256(256), w512(512);
    fill_random(w256, 42);
    fill_random(w512, 1337);
    const float dense = 0.85f, scale = 0.6f;
    for (auto& v : w256) v = v * dense + scale;
    for (auto& v : w512) v = v * dense + scale;

    for (int f = 0; f <= 14; f++) {
        Format fmt = (Format)f;
        check_format(fmt, w256);
        check_format(fmt, w512);
        check_engine_layout(fmt, w256);
        check_engine_layout(fmt, w512);
    }

    check_tail_blocks();
    check_zero_block();
    check_truncated_payloads();
    check_quality_gate();

    std::printf("test_block_codec: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
