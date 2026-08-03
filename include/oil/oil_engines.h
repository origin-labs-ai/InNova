#pragma once
#include "oil/tensor.h"
#include "oil/types.h"
#include <vector>
#include <cstdint>
#include <string>

namespace oil {
namespace engines {

// FP8 E4M3: 1 sign + 4 exponent + 3 mantissa
float fp8_e4m3_dequantize(uint8_t bits);
uint8_t fp8_e4m3_quantize(float val);
Tensor fp8_e4m3_dequant_tensor(const uint8_t* data, int64_t n);
Tensor fp8_e4m3_quantize_tensor(const float* data, int64_t n);
Tensor fp8_e4m3_quant_gemm(const Tensor& a, const uint8_t* b_q, int64_t M, int64_t N, int64_t K);
void fp8_e4m3_quantize_per_channel(const Tensor& t, int channel_dim, Tensor& q, Tensor& scales);
void fp8_e4m3_dequantize_per_channel(const Tensor& q, const Tensor& scales, int channel_dim, Tensor& out);
float fp8_e4m3_quant_error(const Tensor& original, const Tensor& reconstructed);
float fp8_e4m3_quant_snr(const Tensor& original, const Tensor& reconstructed);

// FP8 E5M2: 1 sign + 5 exponent + 2 mantissa
float fp8_e5m2_dequantize(uint8_t bits);
uint8_t fp8_e5m2_quantize(float val);
Tensor fp8_e5m2_dequant_tensor(const uint8_t* data, int64_t n);
Tensor fp8_e5m2_quantize_tensor(const float* data, int64_t n);
Tensor fp8_e5m2_quant_gemm(const Tensor& a, const uint8_t* b_q, int64_t M, int64_t N, int64_t K);
void fp8_e5m2_quantize_per_channel(const Tensor& t, int channel_dim, Tensor& q, Tensor& scales);
void fp8_e5m2_dequantize_per_channel(const Tensor& q, const Tensor& scales, int channel_dim, Tensor& out);
float fp8_e5m2_quant_error(const Tensor& original, const Tensor& reconstructed);
float fp8_e5m2_quant_snr(const Tensor& original, const Tensor& reconstructed);

// OIL8 Engine: 256-entry FP32 codebook + per-block scaling
class OIL8Engine {
public:
    OIL8Engine();
    void train_codebook(const float* data, int64_t n);
    void train_codebook_per_block(const float* data, int64_t n, int64_t block_size, int lloyd_iters = 30);
    uint8_t quantize(float val) const;
    float dequantize(uint8_t idx) const;
    Tensor dequant_tensor(const uint8_t* indices, int64_t n) const;
    void quantize_per_block(const float* data, int64_t n, int64_t block_size,
                            uint8_t* indices_out, float* scales_out) const;
    void dequantize_per_block(const uint8_t* indices, const float* scales,
                              int64_t n, int64_t block_size, float* out) const;
    Tensor quantize_tensor(const float* data, int64_t n) const;
    Tensor quant_gemm(const Tensor& a, const uint8_t* b_idx, int64_t M, int64_t N, int64_t K) const;
    void quantize_per_channel(const Tensor& t, int channel_dim, Tensor& q, Tensor& scales) const;
    void dequantize_per_channel(const Tensor& q, const Tensor& scales, int channel_dim, Tensor& out) const;
    float quant_error(const Tensor& original, const Tensor& reconstructed) const;
    float quant_snr(const Tensor& original, const Tensor& reconstructed) const;
    const std::vector<float>& codebook() const { return codebook_; }
    void enable_stochastic_rounding(bool enable, float temp = 1.0f) {
        use_stochastic_ = enable; stoch_temperature_ = temp;
    }
private:
    std::vector<float> codebook_;
    mutable bool use_stochastic_ = false;
    mutable float stoch_temperature_ = 1.0f;
};

// OIL4 Engine: 16-entry FP16 codebook + per-block scaling
class OIL4Engine {
public:
    OIL4Engine();
    void train_codebook(const float* data, int64_t n);
    void train_codebook_per_block(const float* data, int64_t n, int64_t block_size, int lloyd_iters = 30);
    uint8_t quantize(float val) const;
    float dequantize(uint8_t idx) const;
    Tensor dequant_tensor(const uint8_t* indices, int64_t n) const;
    void quantize_per_block(const float* data, int64_t n, int64_t block_size,
                            uint8_t* indices_out, float* scales_out) const;
    void dequantize_per_block(const uint8_t* indices, const float* scales,
                              int64_t n, int64_t block_size, float* out) const;
    Tensor quantize_tensor(const float* data, int64_t n) const;
    Tensor quant_gemm(const Tensor& a, const uint8_t* b_idx, int64_t M, int64_t N, int64_t K) const;
    void quantize_per_channel(const Tensor& t, int channel_dim, Tensor& q, Tensor& scales) const;
    void dequantize_per_channel(const Tensor& q, const Tensor& scales, int channel_dim, Tensor& out) const;
    float quant_error(const Tensor& original, const Tensor& reconstructed) const;
    float quant_snr(const Tensor& original, const Tensor& reconstructed) const;
    const std::vector<float>& codebook() const { return codebook_; }
    void enable_stochastic_rounding(bool enable, float temp = 1.0f) {
        use_stochastic_ = enable; stoch_temperature_ = temp;
    }
private:
    std::vector<float> codebook_;
    mutable bool use_stochastic_ = false;
    mutable float stoch_temperature_ = 1.0f;
};

// Spark Engine: {-1, 0, +1} with per-block scale.
// NOTE: this is an IN-MEMORY engine with its own self-consistent layout
// (2-bit ternary + per-block max-abs scale). It is NOT the canonical wire
// encoding — SPARK_Q0 on disk is produced by quantize_block_all() in
// oil/block_codec.h (per-32 FP16 scale + sign bits).
class SparkEngine {
public:
    explicit SparkEngine(int64_t block_size = 128);
    Tensor quantize(const Tensor& weight);
    Tensor dequantize(const Tensor& packed, const Tensor& scales, int64_t n);
    Tensor quantize_batch(const Tensor& t);
    Tensor dequantize_batch(const Tensor& q);
    Tensor quant_gemm(const Tensor& a, const Tensor& b_packed, const Tensor& b_scales, int64_t M, int64_t N, int64_t K);
    void quantize_per_channel(const Tensor& t, int channel_dim, Tensor& q, Tensor& scales);
    void dequantize_per_channel(const Tensor& q, const Tensor& scales, int channel_dim, Tensor& out);
    float quant_error(const Tensor& original, const Tensor& reconstructed);
    float quant_snr(const Tensor& original, const Tensor& reconstructed);
private:
    int64_t block_size_;
};

// OIL1 Engine: Block mean (1 FP32 centroid per 32 elements).
// NOTE: in-memory engine; the `scale` argument to dequantize() is unused by
// design because the block means are absolute values (not a relative lattice).
// The canonical on-disk OIL1 is produced by quantize_block_all().
class Oil1Engine {
public:
    Oil1Engine();
    Tensor quantize(const Tensor& weight);
    Tensor dequantize(const Tensor& packed, float scale, int64_t n);
    Tensor quantize_batch(const Tensor& t);
    Tensor dequantize_batch(const Tensor& q);
    Tensor quant_gemm(const Tensor& a, const Tensor& b_packed, float b_scale, int64_t M, int64_t N, int64_t K);
    void quantize_per_channel(const Tensor& t, int channel_dim, Tensor& q, Tensor& scales);
    void dequantize_per_channel(const Tensor& q, const Tensor& scales, int channel_dim, Tensor& out);
    float quant_error(const Tensor& original, const Tensor& reconstructed);
    float quant_snr(const Tensor& original, const Tensor& reconstructed);
};

// OIL2 Engine: 4-entry FP32 codebook, 2-bit indices (4 per byte), per-block scaling
class OIL2Engine {
public:
    OIL2Engine();
    void train_codebook(const float* data, int64_t n);
    void train_codebook_per_block(const float* data, int64_t n, int64_t block_size, int lloyd_iters = 30);
    uint8_t quantize(float val) const;
    float dequantize(uint8_t idx) const;
    Tensor dequant_tensor(const uint8_t* indices, int64_t n) const;
    void quantize_per_block(const float* data, int64_t n, int64_t block_size,
                            uint8_t* indices_out, float* scales_out) const;
    void dequantize_per_block(const uint8_t* indices, const float* scales,
                              int64_t n, int64_t block_size, float* out) const;
    Tensor quantize_tensor(const float* data, int64_t n) const;
    Tensor quant_gemm(const Tensor& a, const uint8_t* b_idx, int64_t M, int64_t N, int64_t K) const;
    void quantize_per_channel(const Tensor& t, int channel_dim, Tensor& q, Tensor& scales) const;
    void dequantize_per_channel(const Tensor& q, const Tensor& scales, int channel_dim, Tensor& out) const;
    float quant_error(const Tensor& original, const Tensor& reconstructed) const;
    float quant_snr(const Tensor& original, const Tensor& reconstructed) const;
    const std::vector<float>& codebook() const { return codebook_; }
    void enable_stochastic_rounding(bool enable, float temp = 1.0f) {
        use_stochastic_ = enable; stoch_temperature_ = temp;
    }
private:
    std::vector<float> codebook_;
    mutable bool use_stochastic_ = false;
    mutable float stoch_temperature_ = 1.0f;
};

// OIL16 Engine: FP16 storage (2 bytes per weight), no codebook
class OIL16Engine {
public:
    OIL16Engine() = default;
    Tensor quantize(const Tensor& weight) const;
    Tensor dequantize(const Tensor& packed, int64_t n) const;
    Tensor quantize_batch(const Tensor& t) const;
    Tensor dequantize_batch(const Tensor& q) const;
    Tensor quant_gemm(const Tensor& a, const Tensor& b_packed, int64_t M, int64_t N, int64_t K) const;
    void quantize_per_channel(const Tensor& t, int channel_dim, Tensor& q, Tensor& scales) const;
    void dequantize_per_channel(const Tensor& q, const Tensor& scales, int channel_dim, Tensor& out) const;
    float quant_error(const Tensor& original, const Tensor& reconstructed) const;
    float quant_snr(const Tensor& original, const Tensor& reconstructed) const;
};

// OIL32 Engine: FP32 identity (lossless) — just copies data
class OIL32Engine {
public:
    OIL32Engine() = default;
    Tensor quantize(const Tensor& weight) const;
    Tensor dequantize(const Tensor& packed, int64_t n) const;
    Tensor quantize_batch(const Tensor& t) const;
    Tensor dequantize_batch(const Tensor& q) const;
    Tensor quant_gemm(const Tensor& a, const Tensor& b_packed, int64_t M, int64_t N, int64_t K) const;
    void quantize_per_channel(const Tensor& t, int channel_dim, Tensor& q, Tensor& scales) const;
    void dequantize_per_channel(const Tensor& q, const Tensor& scales, int channel_dim, Tensor& out) const;
    float quant_error(const Tensor& original, const Tensor& reconstructed) const;
    float quant_snr(const Tensor& original, const Tensor& reconstructed) const;
};

// Error computation
float compute_quant_error(const Tensor& original, const Tensor& dequantized);
float compute_quant_mse(const Tensor& original, const Tensor& reconstructed);
float compute_quant_snr(const Tensor& original, const Tensor& reconstructed);

} // namespace engines
} // namespace oil
