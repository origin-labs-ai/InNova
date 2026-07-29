# `flash_attention.h` — Flash Attention

**Path:** `include/oil/flash_attention.h`

IO-aware flash attention implementation for efficient attention computation with reduced memory bandwidth.

## FlashAttention Class

```cpp
class FlashAttention {
    FlashAttention(int num_heads, int head_dim);
    
    Tensor forward(const Tensor& q, const Tensor& k, 
                   const Tensor& v, const Tensor& mask = {});
};
```

### Features

| Feature | Benefit |
|---------|---------|
| **Tiled computation** | Processes attention in tiles to fit in SRAM |
| **No materialization** | Never materializes full `N×N` attention matrix |
| **Causal masking** | Built-in causal attention support |
| **Alibi support** | AliBi position encoding compatible |

### Flash Attention Algorithm

1. Split Q, K, V into tiles
2. Load tiles into fast on-chip SRAM
3. Compute partial attention scores per tile
4. Apply softmax in a numerically stable tile-by-tile manner
5. Accumulate weighted values
6. Write final results to global memory

### Performance

- O(N²) compute but with much lower memory bandwidth
- 2-5× speedup over vanilla attention for long sequences
- Particularly beneficial for sequences > 1024 tokens

### GPU Implementation

GPU acceleration via Vulkan compute shaders with tiled shared memory. The CPU fallback uses AVX2 tiled loops.
