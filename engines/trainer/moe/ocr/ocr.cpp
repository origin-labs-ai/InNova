#include "ocr.h"
#include "quant/math.h"
#include "quant/kv_cache.h"

namespace quant {
namespace multimodal {

OCREncoder::OCREncoder(int64_t hidden, int64_t num_layers, int64_t num_heads)
    : hidden_size(hidden)
{
    conv_proj = Tensor({3 * 16 * 16, hidden});
    conv_proj.zero_();
    pos_embed = Tensor({1024, hidden});
    pos_embed.zero_();

    TransformerConfig tcfg;
    tcfg.hidden_size = hidden;
    tcfg.num_layers = num_layers;
    tcfg.num_heads = num_heads;
    tcfg.head_dim = hidden / num_heads;

    blocks.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i)
        blocks.emplace_back(tcfg);
}

Tensor OCREncoder::forward(const Tensor& image) {
    int64_t B = image.dim(0);
    int64_t C = image.dim(1);
    int64_t H_ = image.dim(2);
    int64_t W_ = image.dim(3);
    int64_t D = hidden_size;

    int64_t patch_size = 16;
    int64_t n_h = H_ / patch_size;
    int64_t n_w = W_ / patch_size;
    int64_t n = n_h * n_w;
    int64_t patch_dim = C * patch_size * patch_size;

    Tensor patches({B, n, patch_dim});
    const float* img = image.data<float>();
    float* pat = patches.data<float>();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < n_h; ++i)
            for (int64_t j = 0; j < n_w; ++j) {
                int64_t p_idx = i * n_w + j;
                for (int64_t c = 0; c < C; ++c)
                    for (int64_t pi = 0; pi < patch_size; ++pi)
                        for (int64_t pj = 0; pj < patch_size; ++pj) {
                            int64_t src = ((b * C + c) * H_ + i * patch_size + pi) * W_ + j * patch_size + pj;
                            int64_t dst = (b * n + p_idx) * patch_dim + (c * patch_size + pi) * patch_size + pj;
                            pat[dst] = img[src];
                        }
            }

    Tensor proj({B, n, D});
    Tensor proj_2d = proj.reshape({B * n, D});
    math::gemm(1.0f, patches.reshape({B * n, patch_dim}), conv_proj, 0.0f, proj_2d);

    Tensor h({B, n, D});
    math::gelu(proj, h);

    Tensor output({B, n, D});
    const float* pe = pos_embed.data<float>();
    float* od = output.data<float>();
    float* hd = h.data<float>();
    int64_t max_pos = pos_embed.dim(0);
    for (int64_t b = 0; b < B; ++b)
        for (int64_t p = 0; p < n; ++p) {
            int64_t pe_idx = p < max_pos ? p : max_pos - 1;
            for (int64_t d = 0; d < D; ++d)
                od[(b * n + p) * D + d] = hd[(b * n + p) * D + d] + pe[pe_idx * D + d];
        }

    Tensor positions = Tensor::arange(n);
    Tensor mask({n * n});
    mask.fill(0.0f);
    KVCache dummy_cache;
    for (auto& block : blocks)
        output = block.forward(output, positions, mask, dummy_cache, 0);

    return output;
}

} // namespace multimodal
} // namespace quant
