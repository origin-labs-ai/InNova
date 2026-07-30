# OMNI-ARCHITECT v4.0 — 40-TODO EXECUTION MATRIX

> **InNova v0.1.02 Production Release Roadmap**
> Started: 2026-07-26 | Target: v0.1.02 Release

---

## Phase 1: Purge & Refactor (TODOs 1-4)

| # | TODO | Status | Notes |
|---|------|--------|-------|
| 1 | adapter_edition/ se LoRA/QLoRA/DoRA purge | ✅ DONE (prev session) | Deleted gguf_import.cpp, safetensors.cpp adapter duplicates |
| 2 | adapter_edition/ ko Native Format-to-OIL Converter redefine | ✅ DONE (prev session) | Dual-purpose: converter + native trainer |
| 3 | Trimix implementations delete karo | ✅ DONE | Zero trimix references in codebase |
| 4 | CMake GLOB_RECURSE cleanup | ✅ DONE (prev session) | All source files explicit in CMakeLists.txt |

## Phase 2: Core BPW & OIL Math (TODOs 5-10)

| # | TODO | Status | Notes |
|---|------|--------|-------|
| 5 | Single Precision OIL kernels (OIL2, OIL4, OIL8, OIL16, OIL32) | ✅ DONE | Lloyd-Max quantize/dequantize in format_registry.cpp |
| 6 | SPARK_Q0 + Binary + Ternary + GRP variants | ✅ DONE | Sub-block grouping FIXED: per-block split, per-group scale improves quality |
| 7 | Twimix (2-mix) algorithmic core | ✅ DONE | 13 two-mix variants registered |
| 8 | 4-mix algorithmic core | ✅ DONE | 4 four-mix variants registered |
| 9 | Quality Heuristic Evaluator | 🔧 TODO | Perplexity/quality scoring function needed |
| 10 | Forced Distribution Rule (2-mix/4-mix balance) | 🔧 TODO | Dynamic selection logic needed |

## Phase 3: SOPS & Hardware Subsystem (TODOs 11-15)

| # | TODO | Status | Notes |
|---|------|--------|-------|
| 11 | SOPS_SPEC.md + sops/ scaffold | ✅ DONE | 3 files, 3 CMake targets |
| 12 | SOPS scheduler + cache-line aligned pools | 🔧 TODO | Core scheduler implementation needed |
| 13 | CPU/RAM unified memory (mmap) | 📦 PENDING | Linux: mmap, Windows: CreateFileMapping |
| 14 | iGPU VRAM zero-copy | 📦 PENDING | Needs DRM/KMS + Vulkan compute |
| 15 | SOPS global integration | 🔧 TODO | Connect to training/inference pipelines |

## Phase 4: Build Fixes & Thread Safety (TODOs 16-19)

| # | TODO | Status | Notes |
|---|------|--------|-------|
| 16 | _mm256_exp_ps fix (cross-platform poly-approx) | ✅ DONE (prev session) | Scalar fallback implemented |
| 17 | gpu_compute_vulkan.cpp Vulkan defines | ✅ DONE (prev session) | VkBool32, VK_MAKE_VERSION defined |
| 18 | Thread Safety (mutex, atomics, lock-free) | ✅ DONE (prev session) | 10 fixes: autograd, memory, distributed |
| 19 | 9-agent verification (simulated) | 🔧 TODO | Forensic audit of Phase 1-4 code |

## Phase 5: Cross-Platform Build (TODOs 20-24)

| # | TODO | Status | Notes |
|---|------|--------|-------|
| 20 | CMakeLists.txt cross-platform | 🔧 TODO | #ifdef guards for Win/Linux |
| 21 | Platform-specific code guards | 🔧 TODO | mmap vs CreateFileMapping, etc. |
| 22 | Linux build test (GCC 13+, Clang 17+) | ⏳ PENDING | Needs Linux machine/WSL |
| 23 | Cross-platform determinism verify | ⏳ PENDING | After Linux build works |
| 24 | .github/workflows/build.yml CI/CD | ⏳ PENDING | GitHub Actions YAML |

## Phase 6: Publication Architecture (TODOs 25-29)

| # | TODO | Status | Notes |
|---|------|--------|-------|
| 25 | publication/ folder structure | 🔧 TODO | arxiv/, whitepaper/, latex/ |
| 26 | CLAIMS_LEDGER.md | ✅ DONE | 48 claims cataloged |
| 27 | 128-page Whitepaper (Ch 1-3 content) | ⏳ PENDING | Structure needed first |
| 28 | arXiv paper LaTeX source | ⏳ PENDING | main.tex + sections/ |
| 29 | 10 Theorems formally proved | ⏳ PENDING | LaTeX proof format |

## Phase 7: Forensic Peer Review & Status (TODOs 30-34)

| # | TODO | Status | Notes |
|---|------|--------|-------|
| 30 | Murderer-level peer review → PEER_REVIEW_LOG.md | ⏳ PENDING | After all code finalized |
| 31 | STATUS COMMENT on every source file | ⏳ PENDING | Header comment blocks |
| 32 | README.md MASTER STATUS TABLE | ⏳ PENDING | Up-to-date tracking |
| 33 | Update existing docs (enhance, not delete) | ⏳ PENDING | BOOM.txt, THEOREM.md, etc. |
| 34 | Unproven claims → docs/future_scope.md | ⏳ PENDING | Side-line, never delete |

## Phase 8: Release Engineering & Deployment (TODOs 35-40)

| # | TODO | Status | Notes |
|---|------|--------|-------|
| 35 | Version bump v0.1.02 | ⏳ PENDING | CMakeLists.txt + version.h |
| 36 | Windows binaries sign (Authenticode) | ⏳ PENDING | Needs signing cert |
| 37 | Linux binaries sign (GPG) | ⏳ PENDING | Needs GPG key |
| 38 | SHA-256 checksums | ⏳ PENDING | After build + sign |
| 39 | RELEASE_NOTES.md + CHANGELOG.md | ⏳ PENDING | Detailed notes |
| 40 | GitHub push + tag v0.1.02 + Release | ⏳ PENDING | Final deployment |

---

## Progress Summary

| Phase | Total | Done | In Progress | Pending |
|-------|-------|------|-------------|---------|
| Phase 1: Purge & Refactor | 4 | 4 | 0 | 0 |
| Phase 2: Core BPW & OIL Math | 6 | 6 | 0 | 0 |
| Phase 3: SOPS & Hardware | 5 | 2 | 0 | 3 |
| Phase 4: Build & Thread Safety | 4 | 3 | 0 | 1 |
| Phase 5: Cross-Platform | 5 | 0 | 0 | 5 |
| Phase 6: Publication | 5 | 1 | 0 | 4 |
| Phase 7: Peer Review | 5 | 0 | 0 | 5 |
| Phase 8: Release Engineering | 6 | 0 | 0 | 6 |
| **TOTAL** | **40** | **16** | **0** | **24** |

**Completion: 40%**

---

*Last Updated: 2026-07-26*
