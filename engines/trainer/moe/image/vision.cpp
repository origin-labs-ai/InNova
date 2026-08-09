#include "vision.h"
#include "quant/math.h"
#include "quant/kv_cache.h"

namespace quant {
namespace multimodal {

ViTEncoder::ViTEncoder(int64_t img_size, int64_t patch_size, int64_t hidden,
                       int64_t num_layers, int64_t num_heads)
    : hidden_size(hidden)
{
    int64_t n_patches = (img_size / patch_size) * (img_size / patch_size);
    num_patches = n_patches;
    int64_t patch_dim = 3 * patch_size * patch_size;

    patch_embed = Tensor({patch_dim, hidden});
    patch_embed.zero_();
    pos_embed = Tensor({n_patches + 1, hidden});
    pos_embed.zero_();
    cls_token = Tensor({1, hidden});
    cls_token.zero_();

    TransformerConfig tcfg;
    tcfg.hidden_size = hidden;
    tcfg.num_layers = num_layers;
    tcfg.num_heads = num_heads;
    tcfg.head_dim = hidden / num_heads;

    blocks.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i)
        blocks.emplace_back(tcfg);
}

Tensor ViTEncoder::forward(const Tensor& images) {
    int64_t B = images.dim(0);
    int64_t C = images.dim(1);
    int64_t H_ = images.dim(2);
    int64_t W_ = images.dim(3);
    int64_t patch_size = (int64_t)std::sqrt((double)(patch_embed.dim(0) / C));
    int64_t n_patches_h = H_ / patch_size;
    int64_t n_patches_w = W_ / patch_size;
    int64_t n = n_patches_h * n_patches_w;
    int64_t D = hidden_size;
    int64_t patch_dim = C * patch_size * patch_size;

    Tensor patches({B, n, patch_dim});
    const float* img = images.data<float>();
    float* pat = patches.data<float>();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < n_patches_h; ++i)
            for (int64_t j = 0; j < n_patches_w; ++j) {
                int64_t p_idx = i * n_patches_w + j;
                for (int64_t c = 0; c < C; ++c)
                    for (int64_t pi = 0; pi < patch_size; ++pi)
                        for (int64_t pj = 0; pj < patch_size; ++pj) {
                            int64_t src = ((b * C + c) * H_ + i * patch_size + pi) * W_ + j * patch_size + pj;
                            int64_t dst = (b * n + p_idx) * patch_dim + (c * patch_size + pi) * patch_size + pj;
                            pat[dst] = img[src];
                        }
            }

    Tensor token_emb({B, n, D});
    Tensor token_emb_2d = token_emb.reshape({B * n, D});
    math::gemm(1.0f, patches.reshape({B * n, patch_dim}), patch_embed, 0.0f, token_emb_2d);

    Tensor seq({B, n + 1, D});
    const float* pe = pos_embed.data<float>();
    const float* cls = cls_token.data<float>();
    float* s = seq.data<float>();
    float* te = token_emb.data<float>();
    for (int64_t b = 0; b < B; ++b) {
        std::memcpy(s + (b * (n + 1)) * D, cls, D * sizeof(float));
        for (int64_t i = 0; i < D; ++i)
            s[(b * (n + 1)) * D + i] += pe[i];
        for (int64_t p = 0; p < n; ++p) {
            std::memcpy(s + (b * (n + 1) + p + 1) * D, te + (b * n + p) * D, D * sizeof(float));
            for (int64_t i = 0; i < D; ++i)
                s[(b * (n + 1) + p + 1) * D + i] += pe[(p + 1) * D + i];
        }
    }

    Tensor h = seq;
    Tensor positions = Tensor::arange(n + 1);
    Tensor mask({(n + 1) * (n + 1)});
    mask.fill(0.0f);  // bidirectional: no mask
    KVCache dummy_cache;
    for (auto& block : blocks)
        h = block.forward(h, positions, mask, dummy_cache, 0);

    return h;
}

} // namespace multimodal
} // namespace quant
