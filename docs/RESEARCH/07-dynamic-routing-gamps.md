# Research Phase 7: Dynamic Format Routing & Gradient-Aware Mixed Precision

## Dynamic Mixed-Precision Routing — Proven Feasible (2024-2026)

### Key Papers Found

**1. DynaQuant (AAAI 2026)** — arXiv:2511.07903
- Dynamic mixed-precision quantization for learned image compression
- Content-aware quantization: learnable scaling/offset parameters adapt to input
- Dynamic Bit-Width Selector: learns optimal bit precision per layer per input
- Distance-aware Gradient Modulator (DGM): better gradients than standard STE
- Results: R-D performance comparable to full-precision, up to 5.17× speedup

**2. DMR — Dynamic Mixed-Precision Routing (ICML 2026)** — arXiv:2602.02711
- Adaptively selects between high-precision and low-precision LLMs at each decision step
- Two-stage training: KL-divergence supervised learning → GRPO refinement
- Proven effective for agentic LLM tasks (ALFWorld, WebShop)
- Boundary location matters more than threshold value

**3. HGQ — High Granularity Quantization (CERN/ETH)** — arXiv:2405.00645
- Per-weight and per-activation precision via gradient descent
- Ultra-low latency on FPGAs/ASICs
- Resource reduction up to 20×, latency improvement 5× while preserving accuracy

**4. Gradient Knows Best (CVPR 2026)**
- Gradient-guided bit allocation for super-resolution
- Gradients of objective w.r.t. bit-widths → adaptive layer-wise allocation
- Outperforms existing PTQ methods by 1.26 dB PSNR

**5. APreQEL (March 2026)** — arXiv:2603.23575
- Adaptive mixed-precision for edge LLMs
- Balances memory, latency, and accuracy under user-defined priorities
- Layer-wise contribution analysis + hardware-aware quantization type selection

### GAMPS — Gradient-Aware Mixed Precision Scheduler

**Not a named technique, but the CONCEPT is well-established:**
- HGQ: gradient-based per-weight precision optimization
- Gradient Knows Best: gradient-guided bit allocation
- KVmix: gradient-based layer importance for KV cache quantization
- ADQ (2025): sensitivity-aware mixed-precision with EMA codebook adaptation

**All these are "GAMPS" under different names.** The concept of using gradient information to guide mixed-precision allocation is:
1. Well-researched (20+ papers in 2024-2026)
2. Practically proven (CVPR, NeurIPS, ICML publications)
3. NOT a single named method — it's a family of approaches

### Relevance to OIL Format

OIL's format planner currently uses **static** allocation:
- 1% OIL8, 4% OIL4, 95% ternary (hardcoded)

**The research shows DYNAMIC allocation is better:**
1. Gradient-based sensitivity analysis → which layers need high precision
2. Per-input adaptation → some inputs need more precision than others
3. Block-level selection → each block independently chooses its format
4. Hessian-based importance → like GPTQ/NWC, use second-order information

### Implementation Path for OIL

1. **Phase 1:** Static format planner (current) — baseline
2. **Phase 2:** Gradient-sensitivity-based planner — use training gradients to rank blocks
3. **Phase 3:** Hessian-based planner — like NWC, use loss Hessian for importance
4. **Phase 4:** Dynamic per-input router — like DMR, adapt format based on input
5. **Phase 5:** Learnable format policy — train the router end-to-end with STE

### Novelty Assessment

**OIL's contribution would be:**
1. First mixed-precision format with ternary + OIL8 + OIL4 per-block
2. Applying dynamic routing to a VQ-based weight format (not just bit-width)
3. Train-in-format with dynamic format selection (not static)
4. Hardware-efficient: format selection doesn't change storage size

**This IS genuine research territory** — no paper combines ternary training with OIL8/OIL4 VQ in a single format with dynamic routing.
