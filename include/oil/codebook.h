#pragma once

#include "oil/types.h"

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <limits>

namespace oil {

struct CodebookEntry {
    float val;
    float scale;
};

class Codebook {
public:
    virtual ~Codebook() = default;
    virtual void ema_update(float decay) = 0;
    virtual uint8_t quantize(float val) const = 0;
    virtual float dequantize(uint8_t idx) const = 0;
};

class CodebookOIL8 : public Codebook {
public:
    static constexpr int SIZE = 256;
    float centroids[SIZE];

    CodebookOIL8() {
        for (int i = 0; i < SIZE; i++) centroids[i] = 0.0f;
        for (int i = 0; i < SIZE; i++) batch_counts_[i] = 0.0;
        for (int i = 0; i < SIZE; i++) batch_sums_[i] = 0.0;
        for (int i = 0; i < SIZE; i++) running_counts_[i] = 0.0;
        for (int i = 0; i < SIZE; i++) running_sums_[i] = 0.0;
    }

    void train(const float* data, size_t count);
    void ema_update(const float* data, const uint8_t* assign, size_t count, float lr);
    void ema_update(float decay) override;
    uint8_t quantize(float val) const override;
    float dequantize(uint8_t idx) const override;
    size_t serialized_size() const;
    size_t serialize(uint8_t* dst) const;
    static CodebookOIL8 deserialize(const uint8_t* src, size_t& offset, size_t size);

private:
    mutable double batch_counts_[SIZE];
    mutable double batch_sums_[SIZE];
    double running_counts_[SIZE];
    double running_sums_[SIZE];
};

class CodebookOIL4 : public Codebook {
public:
    static constexpr int SIZE = 16;
    uint16_t centroids[SIZE];  // F16 storage

    CodebookOIL4() {
        for (int i = 0; i < SIZE; i++) centroids[i] = 0;
        for (int i = 0; i < SIZE; i++) batch_counts_[i] = 0.0;
        for (int i = 0; i < SIZE; i++) batch_sums_[i] = 0.0;
        for (int i = 0; i < SIZE; i++) running_counts_[i] = 0.0;
        for (int i = 0; i < SIZE; i++) running_sums_[i] = 0.0;
    }

    void train(const float* data, size_t count);
    void ema_update(const float* data, const uint8_t* assign, size_t count, float lr);
    void ema_update(float decay) override;
    uint8_t quantize(float val) const override;
    float dequantize(uint8_t idx) const override;
    size_t serialized_size() const;
    size_t serialize(uint8_t* dst) const;
    static CodebookOIL4 deserialize(const uint8_t* src, size_t& offset, size_t size);

    static uint16_t float_to_half(float f);
    static float half_to_float(uint16_t h);

private:
    mutable double batch_counts_[SIZE];
    mutable double batch_sums_[SIZE];
    double running_counts_[SIZE];
    double running_sums_[SIZE];
};

struct SparkScale {
    float scale;
};

struct Oil1Scale {
    float scale;
};

} // namespace oil
