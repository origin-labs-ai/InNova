# Actionable Improvements — Extracted from ASI Papers Research

**Task:** DIFFUSION.txt Phase K, Task 136  
**Date:** 2026-07-16  
**Source:** `.research/asi_papers.md`, `.research/oil_idx_research.md`, `.research/distributed_research.md`  
**Rule:** 5 actionable, each with name, description, LOC estimate, target file, priority. At least 2 are implemented for Task 137 (see `.research/implemented.md`).

---

## Improvement 1 — KV-Cache Prefix Sharing via Paged Refcount + COW

**Source paper:** PagedAttention / vLLM (Kwon 2023) + Speculative reuse.  
**Description:** The existing `PrefixCache` (`src/inference_opt.cpp:432`) computes a longest-prefix-match length (`match_prefix`) but **does not return which cache entry matched**, so a caller cannot actually reuse the cached KV — it stores a fresh `KVCache` per entry, duplicating memory. Make prefix sharing real: change `match_prefix` to return `{cache_id, match_len}`, add an LRU bound + eviction (`evict_lru`), and expose a `lookup` that returns the reusable `KVCache*` and the number of cached layers so the inference path only computes the uncached suffix. This is the foundation for the "Paged KV 1M context" claim (shared system prompts → near-zero recomputation) and directly cuts the redundant allocation.  
**LOC estimate:** ~90 LOC (header decls + impl in `inference_opt.cpp` + `inference_opt.h`)  
**Target file:** `src/inference_opt.cpp`, `include/oil/inference_opt.h`  
**Priority:** HIGH (unblocks 1M-context + serving throughput)  
**Status:** ✅ IMPLEMENTED in Task 137 — see `.research/implemented.md`.

---

## Improvement 2 — Gradient Noise Injection (Decaying Annealing)

**Source paper:** Neelakantan et al., "Adding Gradient Noise Improves Learning for Very Deep Networks" (ICLR 2016); grounding for RSI/training stability in `.research/asi_papers.md` §9.  
**Description:** Add a `inject_gradient_noise(eta, gamma)` step in the Trainer, applied **after** gradient clipping and **before** the optimizer step. It adds Gaussian noise `N(0, σ_t²)` with `σ_t = eta / (1 + step)^gamma` to each gradient element. Decaying noise helps escape sharp minima and plateaus during long pretraining and is a cheap, well-validated regularizer. Wire it into `Trainer::fit`/`train_step` behind a `TrainConfig::grad_noise_eta` flag (default 0 = off, so existing tests are unaffected). Uses the existing `oil::RNG` (`include/oil/random.h`) for deterministic, seedable noise.  
**LOC estimate:** ~70 LOC (header config + `trainer.cpp` method + wiring in `fit`/`train_step`)  
**Target file:** `src/trainer.cpp`, `include/oil/trainer.h`  
**Priority:** MEDIUM (training quality / RSI loop stability)  
**Status:** ✅ IMPLEMENTED in Task 137 — see `.research/implemented.md`.

---

## Improvement 3 — FP8 Two-Stage Residual Accumulation (FlashAttention-3)

**Source paper:** FlashAttention-3 (Dao/Shah 2024), `.research/asi_papers.md` §1.  
**Description:** The FP8 inference path (`src/inference_opt.cpp:665`, `KVCache::quantize_fp8_block`) quantizes per block with a single scale and dequantizes back with no residual correction, so FP8 matmul/attention accumulates rounding error across blocks. FA-3's contribution is **two-stage accumulation**: keep a running FP32 residual `r` that absorbs the FP8 rounding error each block (Welford-style), so the accumulated output is FP8-matmul-speed but FP32-precision. Implement a `flash_decoding_fp8` variant (or augment the existing `FP8Inference::forward`) that, after each FP8 block dequant, adds the residual `r` back into the accumulator before the next block. This makes the FP8 attention path numerically safe for long causal contexts (the "1M context" path) and is the documented FA-3 technique.  
**LOC estimate:** ~150 LOC (new residual accumulation loop + dequant residual in `inference_opt.cpp`)  
**Target file:** `src/inference_opt.cpp`, `include/oil/kv_cache.h`  
**Priority:** HIGH (enables safe FP8 for long context)  
**Status:** Not yet implemented (queued for next diffusion loop).

---

## Improvement 4 — Adaptive Speculative Decoding (Draft γ from Acceptance Rate)

**Source paper:** Speculative Decoding (Leviathan 2023), `.research/asi_papers.md` §3.  
**Description:** `SpeculativeDecoder` (`src/inference_opt.cpp:98`) uses a fixed `gamma_` (draft length). Empirically the optimal γ rises when the draft model is accurate (high recent acceptance) and falls after a rejection. Make γ adaptive: maintain an EWMA of the recent acceptance rate `acc_ema` and set `gamma = clamp(round(base_gamma * acc_ema / (1 - acc_ema + eps)), min, max)` — when the draft is nearly always accepted, draft more; when it misses, draft fewer. Also add a per-call **draft KV cache** so the γ draft tokens don't re-run the draft model from scratch each step. This is a pure throughput win with zero quality change (speculative decoding is distribution-exact by construction).  
**LOC estimate:** ~120 LOC (adaptive gamma state + draft KV reuse in `SpeculativeDecoder`)  
**Target file:** `src/inference_opt.cpp`, `include/oil/inference_opt.h`  
**Priority:** MEDIUM (inference throughput)  
**Status:** Not yet implemented (queued for next diffusion loop).

---

## Improvement 5 — Content-Addressed Dedup + Lazy SHA256 Verify in `oil_idx` Loader

**Source paper:** OIL idx research (`.research/oil_idx_research.md` §1, §4).  
**Description:** The `oil_load` path should (a) compute SHA256 of each tensor **lazily inside the dequant stream** so integrity is verified for free during load and a corrupt tensor fails fast **with its name** (condition 13/15); (b) **deduplicate** blobs by SHA256 on save — if a tensor's hash already exists in the store, skip writing the blob and add an index entry pointing to the existing offset (wins big on MoE with repeated experts + tied embeddings). Unify the cross-platform mmap (`trainer.cpp::open_mmap` Windows `CreateFileMapping` / Linux `mmap`) into one RAII `MemoryMap` used by both the loader and DataLoader. This hardens the integrity gate and cuts checkpoint size.  
**LOC estimate:** ~200 LOC (lazy verify in dequant loop + dedup-on-save + unified `MemoryMap` RAII)  
**Target file:** `src/oil_format.cpp`, `include/oil/oil_format.h`, `src/trainer.cpp` (mmap unification)  
**Priority:** HIGH (integrity gate condition 15 + dedup for MoE)  
**Status:** Not yet implemented (queued for next diffusion loop).

---

## Summary

| # | Improvement | LOC | Target | Priority | Implemented? |
|---|-------------|-----|--------|----------|--------------|
| 1 | KV-Cache Prefix Sharing (lookup + LRU) | ~90 | `inference_opt.cpp/.h` | HIGH | ✅ Yes (Task 137) |
| 2 | Gradient Noise Injection (decaying) | ~70 | `trainer.cpp/.h` | MEDIUM | ✅ Yes (Task 137) |
| 3 | FP8 Two-Stage Residual Acc. (FA-3) | ~150 | `inference_opt.cpp` | HIGH | Queued |
| 4 | Adaptive Speculative γ + draft KV | ~120 | `inference_opt.cpp/.h` | MEDIUM | Queued |
| 5 | Content-Addressed Dedup + Lazy SHA256 | ~200 | `oil_format.cpp/.h` | HIGH | Queued |

Improvements 1 and 2 are the two small, self-contained, high-leverage changes selected for Task 137. They each touch exactly one source + one header, have no cross-file ripple, and are gated behind new config fields so existing tests and protected code are untouched.
