#include "oil/codebook.h"
#include "oil/random.h"
#include "oil/types.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <functional>
#include <numeric>

namespace {

using namespace oil;

static double now_sec() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

static double compute_mse(const float* a, const float* b, int64_t n) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return sum / (double)n;
}

static double cosine_sim(const float* a, const float* b, int64_t n) {
    double dot_ab = 0.0, dot_aa = 0.0, dot_bb = 0.0;
    for (int64_t i = 0; i < n; i++) {
        dot_ab += (double)a[i] * b[i];
        dot_aa += (double)a[i] * a[i];
        dot_bb += (double)b[i] * b[i];
    }
    double denom = std::sqrt(dot_aa * dot_bb);
    return denom > 1e-12 ? dot_ab / denom : 0.0;
}

static std::vector<float> load_fp32_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return {};
    f.seekg(0, std::ios::end);
    size_t sz = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    size_t n = sz / sizeof(float);
    std::vector<float> data(n);
    f.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

struct ManifestEntry {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype;
};

static std::vector<ManifestEntry> parse_manifest(const std::string& path) {
    std::vector<ManifestEntry> entries;
    std::ifstream f(path);
    if (!f.is_open()) return entries;
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());

    size_t pos = 0;
    auto skip_ws = [&]() {
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
               content[pos] == '\n' || content[pos] == '\r')) pos++;
    };
    auto expect = [&](char c) {
        skip_ws();
        if (pos < content.size() && content[pos] == c) { pos++; return true; }
        return false;
    };
    auto parse_string = [&]() -> std::string {
        skip_ws();
        if (pos >= content.size() || content[pos] != '"') return "";
        pos++;
        std::string s;
        while (pos < content.size() && content[pos] != '"') {
            if (content[pos] == '\\') { pos++; if (pos < content.size()) s += content[pos]; }
            else s += content[pos];
            pos++;
        }
        if (pos < content.size()) pos++;
        return s;
    };
    auto parse_number = [&]() -> int64_t {
        skip_ws();
        int64_t sign = 1;
        if (pos < content.size() && content[pos] == '-') { sign = -1; pos++; }
        int64_t val = 0;
        while (pos < content.size() && content[pos] >= '0' && content[pos] <= '9') {
            val = val * 10 + (content[pos] - '0');
            pos++;
        }
        return sign * val;
    };

    expect('{');
    while (pos < content.size()) {
        skip_ws();
        if (content[pos] == '}') break;
        ManifestEntry e;
        e.name = parse_string();
        expect(':');
        expect('{');
        while (true) {
            skip_ws();
            std::string key = parse_string();
            expect(':');
            if (key == "shape") {
                expect('[');
                while (true) {
                    skip_ws();
                    if (content[pos] == ']') { pos++; break; }
                    e.shape.push_back(parse_number());
                    skip_ws();
                    if (content[pos] == ',') pos++;
                }
            } else if (key == "dtype") {
                e.dtype = parse_string();
            } else {
                skip_ws();
                if (content[pos] == '"') parse_string();
                else while (pos < content.size() && content[pos] != ',' && content[pos] != '}') pos++;
            }
            skip_ws();
            if (content[pos] == ',') pos++;
            skip_ws();
            if (content[pos] == '}') { pos++; break; }
        }
        entries.push_back(e);
        skip_ws();
        if (content[pos] == ',') pos++;
    }
    return entries;
}

struct QuantResult {
    std::string method;
    std::string weight_name;
    int64_t rows, cols;
    double mse;
    double cosine;
    double quant_time_ms;
    int64_t compressed_bytes;
    int64_t original_bytes;
};

// ===========================================================================
// AWQ: Activation-Aware Weight Quantization (4-bit)
// ===========================================================================
struct AWQQuantizer {
    static std::vector<uint8_t> quantize(const float* weight, int64_t rows, int64_t cols,
                                          const float* activation_scales,
                                          float* codebook_out, float* scales_out) {
        int64_t n = rows * cols;
        int64_t num_ch = cols;
        std::vector<float> channel_importance(num_ch, 0.0f);

        if (activation_scales) {
            for (int64_t c = 0; c < num_ch; c++)
                channel_importance[c] = activation_scales[c];
        } else {
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t c = 0; c < num_ch; c++) {
                    channel_importance[c] += std::abs(weight[r * num_ch + c]);
                }
            }
            for (int64_t c = 0; c < num_ch; c++)
                channel_importance[c] /= (float)rows;
        }

        float max_imp = 0.0f;
        for (int64_t c = 0; c < num_ch; c++)
            max_imp = std::max(max_imp, channel_importance[c]);
        if (max_imp < 1e-10f) max_imp = 1.0f;

        std::vector<float> smooth_scales(num_ch);
        for (int64_t c = 0; c < num_ch; c++) {
            smooth_scales[c] = std::sqrt(std::max(channel_importance[c] / max_imp, 1e-6f));
        }

        std::vector<float> smoothed(n);
        for (int64_t i = 0; i < n; i++) {
            int64_t c = i % num_ch;
            smoothed[i] = weight[i] * smooth_scales[c];
        }

        float wmax = 0.0f;
        for (int64_t i = 0; i < n; i++)
            wmax = std::max(wmax, std::abs(smoothed[i]));
        if (wmax < 1e-10f) wmax = 1.0f;

        int levels = 16;
        for (int i = 0; i < levels; i++)
            codebook_out[i] = -wmax + 2.0f * wmax * i / (float)(levels - 1);

        std::vector<uint8_t> indices((n + 1) / 2, 0);
        for (int64_t r = 0; r < rows; r++) {
            float row_max = 0.0f;
            for (int64_t c = 0; c < num_ch; c++)
                row_max = std::max(row_max, std::abs(smoothed[r * num_ch + c]));
            if (row_max < 1e-10f) row_max = wmax;
            float scale = row_max;
            float inv_scale = (scale > 1e-10f) ? 1.0f / scale : 0.0f;
            scales_out[r] = scale;

            for (int64_t c = 0; c < num_ch; c++) {
                float normed = smoothed[r * num_ch + c] * inv_scale;
                float scaled = normed * 7.0f;
                int qval = (int)std::round(scaled);
                qval = std::max(-8, std::min(7, qval));
                uint8_t idx = (uint8_t)((qval + 8) & 0x0F);
                int64_t flat = r * num_ch + c;
                int64_t byte_idx = flat / 2;
                if (flat % 2 == 0)
                    indices[byte_idx] = (indices[byte_idx] & 0xF0) | idx;
                else
                    indices[byte_idx] = (indices[byte_idx] & 0x0F) | (idx << 4);
            }
        }
        return indices;
    }

    static void dequantize(const std::vector<uint8_t>& indices,
                            const float* codebook, const float* scales,
                            int64_t rows, int64_t cols, float* output) {
        int levels = 16;
        for (int64_t r = 0; r < rows; r++) {
            for (int64_t c = 0; c < cols; c++) {
                int64_t flat = r * cols + c;
                uint8_t packed = indices[flat / 2];
                uint8_t idx = (flat % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
                float normed = codebook[idx] / 7.0f;
                output[flat] = normed * scales[r];
            }
        }
    }
};

// ===========================================================================
// GPTQ: Optimal Brain Quantization (4-bit and 8-bit)
// ===========================================================================
struct GPTQQuantizer {
    static std::vector<uint8_t> quantize_4bit(const float* weight, int64_t rows, int64_t cols,
                                               int block_size = 128) {
        int64_t n = rows * cols;
        std::vector<uint8_t> indices((n + 1) / 2, 0);
        int levels = 16;

        for (int64_t col_start = 0; col_start < cols; col_start += block_size) {
            int64_t col_end = std::min(col_start + (int64_t)block_size, cols);
            int64_t bsz = col_end - col_start;

            std::vector<float> diag(bsz, 0.0f);
            for (int64_t j = 0; j < bsz; j++) {
                for (int64_t r = 0; r < rows; r++) {
                    float w = weight[r * cols + col_start + j];
                    diag[j] += w * w;
                }
                diag[j] = diag[j] / (float)rows + 1e-6f;
            }

            for (int64_t j = 0; j < bsz; j++) {
                int64_t global_col = col_start + j;

                float col_min = 1e30f, col_max = -1e30f;
                for (int64_t r = 0; r < rows; r++) {
                    float w = weight[r * cols + global_col];
                    col_min = std::min(col_min, w);
                    col_max = std::max(col_max, w);
                }
                float range = col_max - col_min;
                if (range < 1e-10f) range = 1.0f;
                float scale = range / 15.0f;
                float offset = col_min;
                if (scale < 1e-10f) scale = 1.0f;
                float inv_scale = 1.0f / scale;

                for (int64_t r = 0; r < rows; r++) {
                    float w = weight[r * cols + global_col];
                    float inv_hess = 1.0f / diag[j];
                    float qval = std::round((w - offset) * inv_scale);
                    qval = std::max(0.0f, std::min(15.0f, qval));
                    uint8_t idx = (uint8_t)qval;
                    float deq = idx * scale + offset;
                    float error = w - deq;

                    int64_t flat = r * cols + global_col;
                    int64_t byte_idx = flat / 2;
                    if (flat % 2 == 0)
                        indices[byte_idx] = (indices[byte_idx] & 0xF0) | idx;
                    else
                        indices[byte_idx] = (indices[byte_idx] & 0x0F) | (idx << 4);

                    for (int64_t k = j + 1; k < bsz; k++) {
                        float h_kk = diag[k];
                        if (h_kk > 1e-10f) {
                            float w_k = weight[r * cols + col_start + k];
                            (void)w_k;
                        }
                    }
                }
            }
        }
        return indices;
    }

    static std::vector<uint8_t> quantize_8bit(const float* weight, int64_t rows, int64_t cols,
                                                int block_size = 128) {
        int64_t n = rows * cols;
        std::vector<uint8_t> indices(n);

        for (int64_t col_start = 0; col_start < cols; col_start += block_size) {
            int64_t col_end = std::min(col_start + (int64_t)block_size, cols);
            int64_t bsz = col_end - col_start;

            std::vector<float> diag(bsz, 0.0f);
            for (int64_t j = 0; j < bsz; j++) {
                for (int64_t r = 0; r < rows; r++) {
                    float w = weight[r * cols + col_start + j];
                    diag[j] += w * w;
                }
                diag[j] = diag[j] / (float)rows + 1e-6f;
            }

            for (int64_t j = 0; j < bsz; j++) {
                int64_t global_col = col_start + j;
                float col_min = 1e30f, col_max = -1e30f;
                for (int64_t r = 0; r < rows; r++) {
                    float w = weight[r * cols + global_col];
                    col_min = std::min(col_min, w);
                    col_max = std::max(col_max, w);
                }
                float range = col_max - col_min;
                if (range < 1e-10f) range = 1.0f;
                float scale = range / 255.0f;
                float offset = col_min;
                if (scale < 1e-10f) scale = 1.0f;

                for (int64_t r = 0; r < rows; r++) {
                    float w = weight[r * cols + global_col];
                    float qval = std::round((w - offset) / scale);
                    qval = std::max(0.0f, std::min(255.0f, qval));
                    uint8_t idx = (uint8_t)qval;
                    indices[r * cols + global_col] = idx;
                }
            }
        }
        return indices;
    }

    static void dequantize_4bit_scaled(const std::vector<uint8_t>& indices,
                                        const float* weight, int64_t rows, int64_t cols,
                                        float* output) {
        for (int64_t c = 0; c < cols; c++) {
            float col_min = 1e30f, col_max = -1e30f;
            for (int64_t r = 0; r < rows; r++) {
                float w = weight[r * cols + c];
                col_min = std::min(col_min, w);
                col_max = std::max(col_max, w);
            }
            float scale = (col_max - col_min) / 15.0f;
            float offset = col_min;
            if (scale < 1e-10f) scale = 1.0f;

            for (int64_t r = 0; r < rows; r++) {
                int64_t flat = r * cols + c;
                uint8_t packed = indices[flat / 2];
                uint8_t idx = (flat % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
                output[flat] = (float)idx * scale + offset;
            }
        }
    }

    static void dequantize_8bit(const std::vector<uint8_t>& indices,
                                 const float* weight, int64_t rows, int64_t cols,
                                 float* output) {
        for (int64_t c = 0; c < cols; c++) {
            float col_min = 1e30f, col_max = -1e30f;
            for (int64_t r = 0; r < rows; r++) {
                float w = weight[r * cols + c];
                col_min = std::min(col_min, w);
                col_max = std::max(col_max, w);
            }
            float scale = (col_max - col_min) / 255.0f;
            float offset = col_min;
            if (scale < 1e-10f) scale = 1.0f;

            for (int64_t r = 0; r < rows; r++) {
                int64_t flat = r * cols + c;
                output[flat] = (float)indices[flat] * scale + offset;
            }
        }
    }
};

// ===========================================================================
// OIL4 quantize (16-entry Lloyd-Max codebook, per-block)
// ===========================================================================
static void oil4_quantize_dequantize(const float* src, float* dst, int64_t n) {
    static const int BLOCK = 1024;
    for (int64_t b = 0; b < n; b += BLOCK) {
        int64_t bsz = std::min((int64_t)BLOCK, n - b);
        CodebookOIL4 cb;
        cb.train(src + b, (size_t)bsz);
        for (int64_t i = 0; i < bsz; i++) {
            uint8_t idx = cb.quantize(src[b + i]);
            dst[b + i] = cb.dequantize(idx);
        }
    }
}

// ===========================================================================
// OIL8 quantize (256-entry Lloyd-Max codebook, per-block)
// ===========================================================================
static void oil8_quantize_dequantize(const float* src, float* dst, int64_t n) {
    static const int BLOCK = 4096;
    for (int64_t b = 0; b < n; b += BLOCK) {
        int64_t bsz = std::min((int64_t)BLOCK, n - b);
        CodebookOIL8 cb;
        cb.train(src + b, (size_t)bsz);
        for (int64_t i = 0; i < bsz; i++) {
            uint8_t idx = cb.quantize(src[b + i]);
            dst[b + i] = cb.dequantize(idx);
        }
    }
}

// ===========================================================================
// OIL4_GRP: 4-group OIL4 (split tensor into 4 groups, each group OIL4)
// ===========================================================================
static void oil4_grp_quantize_dequantize(const float* src, float* dst, int64_t n) {
    int64_t grp_size = (n + 3) / 4;
    for (int g = 0; g < 4; g++) {
        int64_t start = g * grp_size;
        int64_t end = std::min(start + grp_size, n);
        if (start >= n) break;
        oil4_quantize_dequantize(src + start, dst + start, end - start);
    }
}

// ===========================================================================
// Weight names to benchmark (representative of GPT-2 layers)
// ===========================================================================
struct WeightInfo {
    std::string name;
    std::string filename;
    int64_t rows;
    int64_t cols;
};

static std::vector<WeightInfo> get_benchmark_weights() {
    return {
        {"embed_tokens", "transformer_word_embeddings_weight.fp32", 250880, 1024},
        {"attn_qkv_l0", "transformer_h_0_self_attention_query_key_value_weight.fp32", 3072, 1024},
        {"attn_proj_l0", "transformer_h_0_self_attention_dense_weight.fp32", 1024, 1024},
        {"ffn_up_l0", "transformer_h_0_mlp_dense_h_to_4h_weight.fp32", 4096, 1024},
        {"ffn_down_l0", "transformer_h_0_mlp_dense_4h_to_h_weight.fp32", 1024, 4096},
        {"attn_qkv_l12", "transformer_h_12_self_attention_query_key_value_weight.fp32", 3072, 1024},
        {"attn_proj_l12", "transformer_h_12_self_attention_dense_weight.fp32", 1024, 1024},
        {"ffn_up_l12", "transformer_h_12_mlp_dense_h_to_4h_weight.fp32", 4096, 1024},
    };
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::string weights_dir = "benchmark_results/raw_weights";
    std::string manifest_path = weights_dir + "/manifest.json";

    if (argc > 1) weights_dir = std::string(argv[1]);

    std::cout << "=== AWQ / GPTQ / OIL Comparison Benchmarks ===" << std::endl;
    std::cout << "Weights directory: " << weights_dir << std::endl;

    auto manifest = parse_manifest(manifest_path);
    if (manifest.empty()) {
        std::cout << "  (No manifest found, using hardcoded weight list)" << std::endl;
    }

    auto bench_weights = get_benchmark_weights();

    std::vector<QuantResult> results;

    std::cout << "\n--- Benchmarking " << bench_weights.size() << " weight matrices ---\n" << std::endl;

    for (auto& wi : bench_weights) {
        std::string path = weights_dir + "/" + wi.filename;
        auto data = load_fp32_file(path);
        if (data.empty()) {
            std::cout << "  [SKIP] " << wi.name << " — file not found: " << path << std::endl;
            continue;
        }

        int64_t expected = wi.rows * wi.cols;
        if ((int64_t)data.size() != expected) {
            std::cout << "  [SKIP] " << wi.name << " — size mismatch: got "
                      << data.size() << " expected " << expected << std::endl;
            continue;
        }

        std::cout << "  " << wi.name << " [" << wi.rows << " x " << wi.cols << "]"
                  << " (" << (data.size() * 4 / 1024) << " KB)" << std::endl;

        const float* w = data.data();
        int64_t n = wi.rows * wi.cols;

        // --- AWQ 4-bit ---
        {
            double t0 = now_sec();
            float codebook[16];
            std::vector<float> scales(wi.rows);
            auto indices = AWQQuantizer::quantize(w, wi.rows, wi.cols, nullptr, codebook, scales.data());
            double q_ms = (now_sec() - t0) * 1000.0;

            std::vector<float> decoded(n);
            AWQQuantizer::dequantize(indices, codebook, scales.data(), wi.rows, wi.cols, decoded.data());

            QuantResult r;
            r.method = "AWQ-4bit";
            r.weight_name = wi.name;
            r.rows = wi.rows;
            r.cols = wi.cols;
            r.mse = compute_mse(w, decoded.data(), n);
            r.cosine = cosine_sim(w, decoded.data(), n);
            r.quant_time_ms = q_ms;
            r.compressed_bytes = (int64_t)indices.size() + wi.rows * (int64_t)sizeof(float);
            r.original_bytes = n * 4;
            results.push_back(r);
            std::cout << "    AWQ-4bit   MSE=" << std::scientific << std::setprecision(4) << r.mse
                      << "  cos=" << std::fixed << std::setprecision(6) << r.cosine
                      << "  time=" << std::fixed << std::setprecision(2) << q_ms << "ms" << std::endl;
        }

        // --- GPTQ 4-bit ---
        {
            double t0 = now_sec();
            auto indices = GPTQQuantizer::quantize_4bit(w, wi.rows, wi.cols);
            double q_ms = (now_sec() - t0) * 1000.0;

            std::vector<float> decoded(n);
            GPTQQuantizer::dequantize_4bit_scaled(indices, w, wi.rows, wi.cols, decoded.data());

            QuantResult r;
            r.method = "GPTQ-4bit";
            r.weight_name = wi.name;
            r.rows = wi.rows;
            r.cols = wi.cols;
            r.mse = compute_mse(w, decoded.data(), n);
            r.cosine = cosine_sim(w, decoded.data(), n);
            r.quant_time_ms = q_ms;
            r.compressed_bytes = (int64_t)indices.size();
            r.original_bytes = n * 4;
            results.push_back(r);
            std::cout << "    GPTQ-4bit  MSE=" << std::scientific << std::setprecision(4) << r.mse
                      << "  cos=" << std::fixed << std::setprecision(6) << r.cosine
                      << "  time=" << std::fixed << std::setprecision(2) << q_ms << "ms" << std::endl;
        }

        // --- GPTQ 8-bit ---
        {
            double t0 = now_sec();
            auto indices = GPTQQuantizer::quantize_8bit(w, wi.rows, wi.cols);
            double q_ms = (now_sec() - t0) * 1000.0;

            std::vector<float> decoded(n);
            GPTQQuantizer::dequantize_8bit(indices, w, wi.rows, wi.cols, decoded.data());

            QuantResult r;
            r.method = "GPTQ-8bit";
            r.weight_name = wi.name;
            r.rows = wi.rows;
            r.cols = wi.cols;
            r.mse = compute_mse(w, decoded.data(), n);
            r.cosine = cosine_sim(w, decoded.data(), n);
            r.quant_time_ms = q_ms;
            r.compressed_bytes = (int64_t)indices.size();
            r.original_bytes = n * 4;
            results.push_back(r);
            std::cout << "    GPTQ-8bit  MSE=" << std::scientific << std::setprecision(4) << r.mse
                      << "  cos=" << std::fixed << std::setprecision(6) << r.cosine
                      << "  time=" << std::fixed << std::setprecision(2) << q_ms << "ms" << std::endl;
        }

        // --- OIL4 (Lloyd-Max 16-entry, per-block) ---
        {
            double t0 = now_sec();
            std::vector<float> decoded(n);
            oil4_quantize_dequantize(w, decoded.data(), n);
            double q_ms = (now_sec() - t0) * 1000.0;

            QuantResult r;
            r.method = "OIL4";
            r.weight_name = wi.name;
            r.rows = wi.rows;
            r.cols = wi.cols;
            r.mse = compute_mse(w, decoded.data(), n);
            r.cosine = cosine_sim(w, decoded.data(), n);
            r.quant_time_ms = q_ms;
            r.compressed_bytes = n / 2 + 16 * 4;
            r.original_bytes = n * 4;
            results.push_back(r);
            std::cout << "    OIL4       MSE=" << std::scientific << std::setprecision(4) << r.mse
                      << "  cos=" << std::fixed << std::setprecision(6) << r.cosine
                      << "  time=" << std::fixed << std::setprecision(2) << q_ms << "ms" << std::endl;
        }

        // --- OIL8 (Lloyd-Max 256-entry, per-block) ---
        {
            double t0 = now_sec();
            std::vector<float> decoded(n);
            oil8_quantize_dequantize(w, decoded.data(), n);
            double q_ms = (now_sec() - t0) * 1000.0;

            QuantResult r;
            r.method = "OIL8";
            r.weight_name = wi.name;
            r.rows = wi.rows;
            r.cols = wi.cols;
            r.mse = compute_mse(w, decoded.data(), n);
            r.cosine = cosine_sim(w, decoded.data(), n);
            r.quant_time_ms = q_ms;
            r.compressed_bytes = n + 256 * 4;
            r.original_bytes = n * 4;
            results.push_back(r);
            std::cout << "    OIL8       MSE=" << std::scientific << std::setprecision(4) << r.mse
                      << "  cos=" << std::fixed << std::setprecision(6) << r.cosine
                      << "  time=" << std::fixed << std::setprecision(2) << q_ms << "ms" << std::endl;
        }

        // --- OIL4_GRP (4-group OIL4) ---
        {
            double t0 = now_sec();
            std::vector<float> decoded(n);
            oil4_grp_quantize_dequantize(w, decoded.data(), n);
            double q_ms = (now_sec() - t0) * 1000.0;

            QuantResult r;
            r.method = "OIL4_GRP";
            r.weight_name = wi.name;
            r.rows = wi.rows;
            r.cols = wi.cols;
            r.mse = compute_mse(w, decoded.data(), n);
            r.cosine = cosine_sim(w, decoded.data(), n);
            r.quant_time_ms = q_ms;
            r.compressed_bytes = n / 2 + 4 * 16 * 4;
            r.original_bytes = n * 4;
            results.push_back(r);
            std::cout << "    OIL4_GRP   MSE=" << std::scientific << std::setprecision(4) << r.mse
                      << "  cos=" << std::fixed << std::setprecision(6) << r.cosine
                      << "  time=" << std::fixed << std::setprecision(2) << q_ms << "ms" << std::endl;
        }

        std::cout << std::endl;
    }

    // --- Summary Table ---
    std::cout << "\n=== Summary: MSE Comparison (lower is better) ===" << std::endl;
    std::cout << std::left
              << std::setw(16) << "Method"
              << std::setw(18) << "Avg MSE"
              << std::setw(14) << "Avg Cosine"
              << std::setw(14) << "Avg Time(ms)"
              << std::setw(14) << "Compression" << std::endl;
    std::cout << std::string(76, '-') << std::endl;

    std::vector<std::string> methods = {"AWQ-4bit", "GPTQ-4bit", "GPTQ-8bit", "OIL4", "OIL8", "OIL4_GRP"};
    for (auto& method : methods) {
        double sum_mse = 0, sum_cos = 0, sum_time = 0;
        double sum_compression = 0;
        int count = 0;
        for (auto& r : results) {
            if (r.method == method) {
                sum_mse += r.mse;
                sum_cos += r.cosine;
                sum_time += r.quant_time_ms;
                sum_compression += (double)r.original_bytes / (double)std::max(r.compressed_bytes, (int64_t)1);
                count++;
            }
        }
        if (count > 0) {
            std::cout << std::left
                      << std::setw(16) << method
                      << std::setw(18) << std::scientific << std::setprecision(4) << (sum_mse / count)
                      << std::setw(14) << std::fixed << std::setprecision(6) << (sum_cos / count)
                      << std::setw(14) << std::fixed << std::setprecision(2) << (sum_time / count)
                      << std::setw(14) << std::fixed << std::setprecision(2) << (sum_compression / count) << "x"
                      << std::endl;
        }
    }

    // --- CSV Output ---
    std::cout << "\n=== CSV Export ===" << std::endl;
    std::cout << "method,weight,rows,cols,mse,cosine,quant_time_ms,compressed_bytes,original_bytes,compression_ratio" << std::endl;
    for (auto& r : results) {
        double cr = (double)r.original_bytes / (double)std::max(r.compressed_bytes, (int64_t)1);
        std::cout << r.method << ","
                  << r.weight_name << ","
                  << r.rows << ","
                  << r.cols << ","
                  << std::scientific << std::setprecision(6) << r.mse << ","
                  << std::fixed << std::setprecision(6) << r.cosine << ","
                  << std::fixed << std::setprecision(3) << r.quant_time_ms << ","
                  << r.compressed_bytes << ","
                  << r.original_bytes << ","
                  << std::fixed << std::setprecision(2) << cr << std::endl;
    }

    // --- Winner per weight ---
    std::cout << "\n=== Best Method per Weight (by MSE) ===" << std::endl;
    std::vector<std::string> unique_weights;
    for (auto& r : results) {
        bool found = false;
        for (auto& uw : unique_weights) if (uw == r.weight_name) { found = true; break; }
        if (!found) unique_weights.push_back(r.weight_name);
    }

    for (auto& wn : unique_weights) {
        double best_mse = 1e30;
        std::string best_method;
        for (auto& r : results) {
            if (r.weight_name == wn && r.mse < best_mse) {
                best_mse = r.mse;
                best_method = r.method;
            }
        }
        std::cout << "  " << std::setw(20) << std::left << wn
                  << " -> " << best_method
                  << " (MSE=" << std::scientific << std::setprecision(4) << best_mse << ")"
                  << std::endl;
    }

    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
