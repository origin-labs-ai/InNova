#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cassert>

namespace quant {

enum class Activation : uint8_t { None, ReLU, GELU, SiLU, SwiGLU, GeGLU };

enum class RoPEScalingMode : uint8_t { None, Linear, NTK, YARN };

// ============================================================
// Format enum — Q-series quantization formats
// Base formats: Q1, Q2, Q3, Q4, Q6, Q8, Q12, Q16, Q24, Q32
// GRP variants: per-group scaling for each base format
// MIXED: TWI_MIX (2-tier) and QUAD_MIX (4-tier) mixed-precision
// ============================================================
enum class Format : uint8_t {
    // --- Base formats (10) ---
    Q1              = 0,   // 1.00 BPW, 1×FP32 block mean, 1-bit binary quantization
    Q2              = 1,   // 2.00 BPW, 4×FP32 centroids, 2-bit lattice
    Q3              = 2,   // 3.00 BPW, 8×FP32 centroids, 3-bit lattice
    Q4              = 3,   // 4.00 BPW, 16×FP32 centroids, 4-bit lattice
    Q6              = 4,   // 6.00 BPW, 64×FP32 centroids, 6-bit lattice
    Q8              = 5,   // 8.00 BPW, 256×FP32 centroids, 8-bit lattice
    Q12             = 6,   // 12.00 BPW, 4096×FP16 centroids, 12-bit lattice
    Q16             = 7,   // 16.00 BPW, per-block adaptive 16-bit (beats FP16)
    Q24             = 8,   // 24.00 BPW, per-block FP24 (16b mantissa + 8b exponent)
    Q32             = 9,   // 32.00 BPW, FP32 identity (lossless reference)

    // --- GRP variants (10) — per-group scaling, must beat 2× BPW base ---
    Q1_GRP          = 10,  // 1.00 BPW, block FP16 scale + sign bits (must beat Q2)
    Q2_GRP          = 11,  // 2.625 BPW, 2-bit lattice + per-16 4b sc+4b min + FP16 d/dm (must beat Q4)
    Q3_GRP          = 12,  // 3.50 BPW, 3-bit lattice + per-32 6b sc+6b min + FP16 d (must beat Q6)
    Q4_GRP          = 13,  // 4.50 BPW, 4-bit lattice + per-32 6b sc+6b min + FP16 d/dm (must beat Q8)
    Q6_GRP          = 14,  // 6.5625 BPW, 6-bit lattice + per-16 8b sc + FP16 d (must beat Q12)
    Q8_GRP          = 15,  // 8.50 BPW, 8-bit lattice + per-16 7b sc + FP16 d (must beat Q16)
    Q12_GRP         = 16,  // 12.50 BPW, 12-bit lattice + per-16 FP16 sc + FP16 d (must beat Q24)
    Q16_GRP         = 17,  // 16.50 BPW, 16-bit adaptive + per-16 FP16 sc + FP16 offset (must beat Q32)
    Q24_GRP         = 18,  // 24.50 BPW, FP24 + per-8 FP16 sc + FP16 d (must beat Q32)

    // --- TWI_MIX — 2-tier mixed precision (2 base + 2 GRP = 4) ---
    Q_TWI_MIX_1_5       = 20,  // 1.50 BPW: Q1(95%) + Q4(5%)
    Q_TWI_MIX_2_5       = 21,  // 2.50 BPW: Q2(90%) + Q8(10%)
    Q_TWI_MIX_1_5_GRP   = 22,  // ~1.75 BPW: TWI_MIX@1.5 + per-group scaling
    Q_TWI_MIX_2_5_GRP   = 23,  // ~2.75 BPW: TWI_MIX@2.5 + per-group scaling

    // --- QUAD_MIX — 4-tier mixed precision (7 base + 7 GRP = 14) ---
    Q_QUAD_MIX_3_5      = 24,  // 3.50 BPW: Q1(70%) + Q3(20%) + Q8(8%) + Q32(2%)
    Q_QUAD_MIX_4_5      = 25,  // 4.50 BPW: Q2(60%) + Q4(25%) + Q12(12%) + Q32(3%)
    Q_QUAD_MIX_6_5      = 26,  // 6.50 BPW: Q3(50%) + Q6(30%) + Q16(15%) + Q32(5%)
    Q_QUAD_MIX_8_5      = 27,  // 8.50 BPW: Q4(45%) + Q8(35%) + Q16(15%) + Q32(5%)
    Q_QUAD_MIX_12_5     = 28,  // 12.50 BPW: Q6(40%) + Q12(35%) + Q24(20%) + Q32(5%)
    Q_QUAD_MIX_16_5     = 29,  // 16.50 BPW: Q8(35%) + Q16(40%) + Q24(20%) + Q32(5%)
    Q_QUAD_MIX_24_5     = 30,  // 24.50 BPW: Q12(25%) + Q16(30%) + Q24(35%) + Q32(10%)
    Q_QUAD_MIX_3_5_GRP  = 31,  // ~3.75 BPW: QUAD_MIX@3.5 + per-group scaling
    Q_QUAD_MIX_4_5_GRP  = 32,  // ~4.75 BPW: QUAD_MIX@4.5 + per-group scaling
    Q_QUAD_MIX_6_5_GRP  = 33,  // ~6.75 BPW: QUAD_MIX@6.5 + per-group scaling
    Q_QUAD_MIX_8_5_GRP  = 34,  // ~8.75 BPW: QUAD_MIX@8.5 + per-group scaling
    Q_QUAD_MIX_12_5_GRP = 35,  // ~12.75 BPW: QUAD_MIX@12.5 + per-group scaling
    Q_QUAD_MIX_16_5_GRP = 36,  // ~16.75 BPW: QUAD_MIX@16.5 + per-group scaling
    Q_QUAD_MIX_24_5_GRP = 37,  // ~24.75 BPW: QUAD_MIX@24.5 + per-group scaling
};

// Total number of unique format IDs (excluding legacy aliases)
constexpr int FORMAT_COUNT = 37;

inline const char* format_name(Format f) {
    switch (f) {
        case Format::Q1:              return "Q1";
        case Format::Q2:              return "Q2";
        case Format::Q3:              return "Q3";
        case Format::Q4:              return "Q4";
        case Format::Q6:              return "Q6";
        case Format::Q8:              return "Q8";
        case Format::Q12:             return "Q12";
        case Format::Q16:             return "Q16";
        case Format::Q24:             return "Q24";
        case Format::Q32:             return "Q32";
        case Format::Q1_GRP:          return "Q1_GRP";
        case Format::Q2_GRP:          return "Q2_GRP";
        case Format::Q3_GRP:          return "Q3_GRP";
        case Format::Q4_GRP:          return "Q4_GRP";
        case Format::Q6_GRP:          return "Q6_GRP";
        case Format::Q8_GRP:          return "Q8_GRP";
        case Format::Q12_GRP:         return "Q12_GRP";
        case Format::Q16_GRP:         return "Q16_GRP";
        case Format::Q24_GRP:         return "Q24_GRP";
        case Format::Q_TWI_MIX_1_5:       return "Q_TWI_MIX@1.5";
        case Format::Q_TWI_MIX_2_5:       return "Q_TWI_MIX@2.5";
        case Format::Q_TWI_MIX_1_5_GRP:   return "Q_TWI_MIX@1.5_GRP";
        case Format::Q_TWI_MIX_2_5_GRP:   return "Q_TWI_MIX@2.5_GRP";
        case Format::Q_QUAD_MIX_3_5:      return "Q_QUAD_MIX@3.5";
        case Format::Q_QUAD_MIX_4_5:      return "Q_QUAD_MIX@4.5";
        case Format::Q_QUAD_MIX_6_5:      return "Q_QUAD_MIX@6.5";
        case Format::Q_QUAD_MIX_8_5:      return "Q_QUAD_MIX@8.5";
        case Format::Q_QUAD_MIX_12_5:     return "Q_QUAD_MIX@12.5";
        case Format::Q_QUAD_MIX_16_5:     return "Q_QUAD_MIX@16.5";
        case Format::Q_QUAD_MIX_24_5:     return "Q_QUAD_MIX@24.5";
        case Format::Q_QUAD_MIX_3_5_GRP:  return "Q_QUAD_MIX@3.5_GRP";
        case Format::Q_QUAD_MIX_4_5_GRP:  return "Q_QUAD_MIX@4.5_GRP";
        case Format::Q_QUAD_MIX_6_5_GRP:  return "Q_QUAD_MIX@6.5_GRP";
        case Format::Q_QUAD_MIX_8_5_GRP:  return "Q_QUAD_MIX@8.5_GRP";
        case Format::Q_QUAD_MIX_12_5_GRP: return "Q_QUAD_MIX@12.5_GRP";
        case Format::Q_QUAD_MIX_16_5_GRP: return "Q_QUAD_MIX@16.5_GRP";
        case Format::Q_QUAD_MIX_24_5_GRP: return "Q_QUAD_MIX@24.5_GRP";
        default: return "unknown";
    }
}

inline float format_bpw(Format f) {
    switch (f) {
        // Base formats
        case Format::Q1:       return 1.0f;
        case Format::Q2:       return 2.0f;
        case Format::Q3:       return 3.0f;
        case Format::Q4:       return 4.0f;
        case Format::Q6:       return 6.0f;
        case Format::Q8:       return 8.0f;
        case Format::Q12:      return 12.0f;
        case Format::Q16:      return 16.0f;
        case Format::Q24:      return 24.0f;
        case Format::Q32:      return 32.0f;
        // GRP variants — base BPW + per-group overhead
        // Q1_GRP: block FP16 scale amortized over 256 weights = 0 extra
        // Q2_GRP: per-16 4b scale + 4b min + FP16 d/dm = +0.625 BPW
        // Q3_GRP: per-32 6b scale + 6b min + FP16 d = +0.5 BPW
        // Q4_GRP: per-32 6b scale + 6b min + FP16 d/dm = +0.5 BPW
        // Q6_GRP: per-16 8b scale + FP16 d = +0.5 BPW
        // Q8_GRP: per-16 7b scale + FP16 d = +0.5 BPW
        // Q12_GRP: per-16 FP16 scale + FP16 d = +0.5 BPW
        // Q16_GRP: per-16 FP16 scale + FP16 offset = +0.5 BPW
        // Q24_GRP: per-8 FP16 scale + FP16 d = +0.5 BPW
        case Format::Q1_GRP:   return 1.0f;
        case Format::Q2_GRP:   return 2.625f;
        case Format::Q3_GRP:   return 3.5f;
        case Format::Q4_GRP:   return 4.5f;
        case Format::Q6_GRP:   return 6.5625f;
        case Format::Q8_GRP:   return 8.5f;
        case Format::Q12_GRP:  return 12.5f;
        case Format::Q16_GRP:  return 16.5f;
        case Format::Q24_GRP:  return 24.5f;
        // TWI_MIX — 2-tier mixed
        case Format::Q_TWI_MIX_1_5:       return 1.5f;
        case Format::Q_TWI_MIX_2_5:       return 2.5f;
        case Format::Q_TWI_MIX_1_5_GRP:   return 1.75f;
        case Format::Q_TWI_MIX_2_5_GRP:   return 2.75f;
        // QUAD_MIX — 4-tier mixed
        case Format::Q_QUAD_MIX_3_5:      return 3.5f;
        case Format::Q_QUAD_MIX_4_5:      return 4.5f;
        case Format::Q_QUAD_MIX_6_5:      return 6.5f;
        case Format::Q_QUAD_MIX_8_5:      return 8.5f;
        case Format::Q_QUAD_MIX_12_5:     return 12.5f;
        case Format::Q_QUAD_MIX_16_5:     return 16.5f;
        case Format::Q_QUAD_MIX_24_5:     return 24.5f;
        case Format::Q_QUAD_MIX_3_5_GRP:  return 3.75f;
        case Format::Q_QUAD_MIX_4_5_GRP:  return 4.75f;
        case Format::Q_QUAD_MIX_6_5_GRP:  return 6.75f;
        case Format::Q_QUAD_MIX_8_5_GRP:  return 8.75f;
        case Format::Q_QUAD_MIX_12_5_GRP: return 12.75f;
        case Format::Q_QUAD_MIX_16_5_GRP: return 16.75f;
        case Format::Q_QUAD_MIX_24_5_GRP: return 24.75f;
        default: return 0;
    }
}

// Helper: is this a base format (not GRP, not MIXED)?
inline bool format_is_base(Format f) {
    auto v = static_cast<uint8_t>(f);
    return v <= 9;
}

// Helper: is this a GRP variant?
inline bool format_is_grp(Format f) {
    auto v = static_cast<uint8_t>(f);
    return (v >= 10 && v <= 19) || v == 22 || v == 23 || (v >= 31 && v <= 37);
}

// Helper: is this a mixed format (TWI or QUAD)?
inline bool format_is_mixed(Format f) {
    auto v = static_cast<uint8_t>(f);
    return v >= 20 && v <= 37;
}

// Helper: is this a TWI_MIX format?
inline bool format_is_twi_mix(Format f) {
    auto v = static_cast<uint8_t>(f);
    return v >= 20 && v <= 23;
}

// Helper: is this a QUAD_MIX format?
inline bool format_is_quad_mix(Format f) {
    auto v = static_cast<uint8_t>(f);
    return v >= 24 && v <= 37;
}

// Helper: number of codebook centroids for base formats
inline int format_codebook_size(Format f) {
    switch (f) {
        case Format::Q1:   return 1;      // block mean only
        case Format::Q2:   return 4;
        case Format::Q3:   return 8;
        case Format::Q4:   return 16;
        case Format::Q6:   return 64;
        case Format::Q8:   return 256;
        case Format::Q12:  return 4096;
        case Format::Q16:  return 0;      // per-block adaptive, no codebook
        case Format::Q24:  return 0;      // direct FP24, no codebook
        case Format::Q32:  return 0;      // identity (FP32), no codebook
        default: return 0;
    }
}

enum class DType : uint8_t {
    I64,     // int64_t
    I32,     // int32_t
    U8,      // uint8_t
    U4,      // 4-bit packed (2 per byte)
    U16,     // uint16_t (for 12-bit packed)
    F16,     // half precision
    F32,     // single precision
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::I64: return 8;
        case DType::I32: return 4;
        case DType::U8:  return 1;
        case DType::U4:  return 1;
        case DType::U16: return 2;
        case DType::F16: return 2;
        case DType::F32: return 4;
        default: return 0;
    }
}

inline DType format_to_dtype(Format f) {
    switch (f) {
        case Format::Q1:       return DType::U8;
        case Format::Q2:       return DType::U8;
        case Format::Q3:       return DType::U8;   // 3-bit packed into bytes
        case Format::Q4:       return DType::U4;
        case Format::Q6:       return DType::U8;   // 6-bit packed into bytes
        case Format::Q8:       return DType::U8;
        case Format::Q12:      return DType::U16;  // 12-bit packed into uint16
        case Format::Q16:      return DType::F16;
        case Format::Q24:      return DType::U8;   // 24-bit packed as 3 bytes
        case Format::Q32:      return DType::F32;
        // GRP variants use same index type as their base
        case Format::Q1_GRP:   return DType::U8;
        case Format::Q2_GRP:   return DType::U8;
        case Format::Q3_GRP:   return DType::U8;
        case Format::Q4_GRP:   return DType::U4;
        case Format::Q6_GRP:   return DType::U8;
        case Format::Q8_GRP:   return DType::U8;
        case Format::Q12_GRP:  return DType::U16;
        case Format::Q16_GRP:  return DType::F16;
        case Format::Q24_GRP:  return DType::U8;
        // MIXED formats store a format table + block data, DType is U8 for the container
        default:               return DType::U8;
    }
}

struct Shape {
    int64_t dims[8];
    int rank;

    Shape() : rank(0) { dims[0]=dims[1]=dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0; }
    explicit Shape(int64_t d0) : rank(1) { dims[0]=d0; dims[1]=dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0; }
    Shape(int64_t d0, int64_t d1) : rank(2) { dims[0]=d0; dims[1]=d1; dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0; }
    Shape(int64_t d0, int64_t d1, int64_t d2) : rank(3) { dims[0]=d0; dims[1]=d1; dims[2]=d2; dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0; }
    Shape(std::initializer_list<int64_t> l) : rank((int)l.size()) {
        dims[0]=dims[1]=dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0;
        if (rank > 8) throw std::runtime_error("Shape: rank exceeds maximum of 8");
        int i=0; for (auto x: l) dims[i++] = x;
    }

    int64_t& operator[](int i) { if (i < 0 || i >= rank) throw std::out_of_range("Shape index out of range"); return dims[i]; }
    const int64_t& operator[](int i) const { if (i < 0 || i >= rank) throw std::out_of_range("Shape index out of range"); return dims[i]; }

    int64_t numel() const {
        int64_t n = 1;
        for (int i=0; i<rank; i++) n *= dims[i];
        return n;
    }

    bool operator==(const Shape& o) const {
        if (rank != o.rank) return false;
        for (int i=0; i<rank; i++) if (dims[i] != o.dims[i]) return false;
        return true;
    }

    bool operator!=(const Shape& o) const { return !(*this == o); }

    std::string to_string() const {
        std::string s = "[";
        for (int i=0; i<rank; i++) {
            if (i) s += ",";
            s += std::to_string(dims[i]);
        }
        s += "]";
        return s;
    }
};

struct Status {
    bool ok;
    std::string msg;
    Status() : ok(true) {}
    Status(const std::string& e) : ok(false), msg(e) {}
    static Status error(const std::string& m) { return Status(m); }
    static Status success() { return Status(); }
    explicit operator bool() const { return ok; }
};

struct Config {
    int num_threads = 1;
    uint64_t seed = 42;
    size_t pool_size = 64 * 1024 * 1024; // 64MB temp pool
    bool verbose = false;
};

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

#ifdef QUANT_THROW_ABORT
#define QUANT_CHECK(cond, msg) \
    do { if (!(cond)) { fprintf(stderr, "QUANT_CHECK FAIL: %s\n", std::string(msg).c_str()); fflush(stderr); std::abort(); } } while(0)
#else
#define QUANT_CHECK(cond, msg) \
    do { if (!(cond)) throw quant::Error(msg); } while(0)
#endif

} // namespace quant
