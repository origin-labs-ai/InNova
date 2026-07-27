#include "video.h"
#include "oil/math.h"
#include "oil/kv_cache.h"

namespace oil {
namespace multimodal {

VideoEncoder::VideoEncoder(int64_t tube_size, int64_t img_size, int64_t patch_size,
                           int64_t hidden, int64_t num_layers, int64_t num_heads)
    : hidden_size(hidden), patch_size(patch_size), tube_size(tube_size)
{
    int64_t n_patches = (img_size / patch_size) * (img_size / patch_size);
    num_tubes = n_patches;
    int64_t tube_dim = 3 * tube_size * patch_size * patch_size;

    conv3d_proj = Tensor({tube_dim, hidden});
    conv3d_proj.zero_();
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

Tensor VideoEncoder::forward(const Tensor& video) {
    int64_t B = video.dim(0);
    int64_t C = video.dim(1);
    int64_t T_ = video.dim(2);
    int64_t H_ = video.dim(3);
    int64_t W_ = video.dim(4);
    int64_t D = hidden_size;
    int64_t n = num_tubes;

    int64_t tube_dim = (int64_t)conv3d_proj.dim(0);
    int64_t tube_time = tube_dim / (C * (int64_t)std::sqrt((double)(tube_dim / (C * T_))));
    int64_t ps = (int64_t)std::sqrt((double)(tube_dim / (C * T_)));
    if (ps == 0) ps = patch_size;
    int64_t ts = tube_dim / (C * ps * ps);
    if (ts == 0) ts = 1;
    int64_t n_patches_h = H_ / ps;
    int64_t n_patches_w = W_ / ps;
    Tensor tube_flat({B, n, tube_dim});
    tube_flat.zero_();
    float* tf = tube_flat.data<float>();
    const float* vid = video.data<float>();
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t tp = 0; tp < n; ++tp) {
            int64_t ph = tp / n_patches_w;
            int64_t pw = tp % n_patches_w;
            int64_t idx = 0;
            for (int64_t tt = 0; tt < ts; ++tt) {
                int64_t t_idx = std::min((tp * ts + tt) * T_ / n, T_ - 1);
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t pi = 0; pi < ps; ++pi) {
                        for (int64_t pj = 0; pj < ps; ++pj) {
                            int64_t h_idx = ph * ps + pi;
                            int64_t w_idx = pw * ps + pj;
                            if (h_idx < H_ && w_idx < W_) {
                                int64_t src = ((b * C + c) * T_ + t_idx) * H_ * W_ + h_idx * W_ + w_idx;
                                tf[(b * n + tp) * tube_dim + idx] = vid[src];
                            }
                            idx++;
                        }
                    }
                }
            }
        }
    }

    Tensor token_emb({B, n, D});
    Tensor token_emb_2d = token_emb.reshape({B * n, D});
    math::gemm(1.0f, tube_flat.reshape({B * n, conv3d_proj.dim(0)}), conv3d_proj, 0.0f, token_emb_2d);

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
    mask.fill(0.0f);
    KVCache dummy_cache;
    for (auto& block : blocks)
        h = block.forward(h, positions, mask, dummy_cache, 0);

    return h;
}

} // namespace multimodal
} // namespace oil
