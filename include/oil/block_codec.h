#pragma once

#include "oil/types.h"
#include <cstdint>
#include <cstddef>
#include <vector>

namespace oil {

// ============================================================================
// Canonical OIL block codec — single source of truth for on-disk block payloads.
//
// Every format's payload is self-contained in `indices` (and optionally
// `codebook` for OIL1 block means) and occupies AT MOST the claimed BPW:
//
//   Format               Payload layout (n = block weights)
//   -----                ------------------------------------
//   OIL32                n*4 bytes raw FP32                  -> 32.00 BPW
//   OIL16 / OIL16_GRP    n*2 bytes raw FP16                  -> 16.00 BPW
//   OIL8  / OIL8_GRP     8b*(n) + per-64 grp hdr (2xFP16)    ->  8.00/ 8.50 BPW
//   OIL4  / OIL4_GRP     4b*(n) + per-64 grp hdr (2xFP16)    ->  4.00/ 4.50 BPW
//   OIL2  / OIL2_GRP     2b*(n) + per-64 grp hdr (2xFP16)    ->  2.00/ 2.50 BPW
//   OIL1                 ceil(n/32)*2 bytes FP16 block means ->  0.50 BPW
//   OIL1_GRP             16b FP16 scale + 16 slots + 1b*(n-16)-> 1.00 BPW
//   SPARK_Q0             per 32: 16b FP16 scale + 32 signs   ->  1.50 BPW
//   SPARK_Q0_GRP         16b scale + n signs + (n/2-16) ref. ->  1.50 BPW
//   SPARK_SPARSE         4B FP32 scale + k*3 records         ->  2.00 BPW
//   SPARK_SPARSE_GRP     4B header (2xFP16) + k*3 records    ->  2.00 BPW
//
// Among the GRP variants, only OIL2_GRP / OIL4_GRP / OIL8_GRP carry REAL
// per-64-weight-group state: every 64-weight group stores its own FP16
// zero-point + FP16 scale (32 bits per 64 weights = 0.5 BPW), applied on
// decode, and the overhead is included in the honest claimed BPW
// (OIL2_GRP=2.5, OIL4_GRP=4.5, OIL8_GRP=8.5). OIL1_GRP and SPARK_Q0_GRP
// store a single block-level FP16 scale (funded by slots / refinement bits),
// SPARK_SPARSE_GRP a 4-byte per-half-block FP16 header, and OIL16_GRP is
// plain FP16 with no group state at all. Payload size obeys the budget
// contract below (full blocks exact, tail blocks +1 byte of container
// alignment), so a stored block never exceeds format_bpw() except for that
// documented per-tail-block allowance.
//
// Non-GRP lattice formats fund their single FP16 scale with "slots": a few
// weights per block that decode as 0.0 (skipped on encode too). All bit
// streams are LSB-first, matching FormatRegistry's tensor-level layouts so
// claimed BPW equals stored bytes everywhere (block and tensor level).
// ============================================================================

// Quantize one block into its canonical payload. `codebook` is used only by
// OIL1 (FP16 block means); every other format leaves it empty.
//
// Budget contract (enforced by tests/test_block_codec.cpp):
//   full blocks (n = 256): stored bytes == ceil(claimed_bpw * n / 8) exactly;
//   tail blocks:           stored bytes <= ceil(claimed_bpw * n / 8) + 1
//                          (one byte of container alignment per tail block).
bool quantize_block_all(Format fmt, const float* w, int n,
                        std::vector<uint8_t>& indices, std::vector<uint8_t>& codebook);

// Decode one block payload into `out` (nw floats). Bounds-safe: reads past
// the payload return 0, unhandled formats produce all-zeros.
void dequantize_block_all(Format fmt,
                          const uint8_t* indices, size_t idx_bytes,
                          const uint8_t* codebook, size_t cb_bytes,
                          uint32_t nw, float* out);

// Actual bits-per-weight of a stored payload (0.0 if nw == 0).
float block_actual_bpw(uint32_t nw, size_t idx_bytes, size_t cb_bytes);

// Max payload bytes for the claimed BPW of `fmt` over n weights.
size_t block_claimed_bytes(Format fmt, uint32_t n);

} // namespace oil
