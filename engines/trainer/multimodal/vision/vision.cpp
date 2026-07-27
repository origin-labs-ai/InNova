#include "vision.h"
#include "oil/math.h"
#include "oil/kv_cache.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace oil {
namespace multimodal {

// ============= VisionEncoder =============

VisionEncoder::VisionEncoder(int64_t img_sz, int64_t patch_sz, int64_t hidden,
                             int64_t num_layers, int64_t num_heads, int64_t classes,
                             int64_t max_det)
    : img_size(img_sz), patch_size(patch_sz), hidden_size(hidden),
      num_classes(classes), max_detections(max_det)
{
    int64_t n_patches = (img_sz / patch_sz) * (img_sz / patch_sz);
    num_patches = n_patches;
    int64_t patch_dim = 3 * patch_sz * patch_sz;

    patch_embed = Tensor({patch_dim, hidden});
    patch_embed.zero_();
    pos_embed = Tensor({n_patches + 1, hidden});
    pos_embed.zero_();
    cls_token = Tensor({1, hidden});
    cls_token.zero_();
    class_head = Tensor({hidden, classes});
    class_head.zero_();
    bbox_head = Tensor({hidden, 4});
    bbox_head.zero_();

    int64_t vocab_sz = 32000;
    caption_embed = Tensor({vocab_sz, hidden});
    caption_embed.zero_();
    caption_proj = Tensor({hidden, hidden});
    caption_proj.zero_();

    TransformerConfig tcfg;
    tcfg.hidden_size = hidden;
    tcfg.num_layers = num_layers;
    tcfg.num_heads = num_heads;
    tcfg.head_dim = hidden / num_heads;

    blocks.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i)
        blocks.emplace_back(tcfg);
}

Tensor VisionEncoder::encode(const Tensor& image) {
    int64_t B = image.dim(0);
    int64_t C = image.dim(1);
    int64_t H_ = image.dim(2);
    int64_t W_ = image.dim(3);
    int64_t patch_dim = 3 * patch_size * patch_size;
    int64_t n_patches_h = H_ / patch_size;
    int64_t n_patches_w = W_ / patch_size;
    int64_t n = n_patches_h * n_patches_w;
    int64_t D = hidden_size;

    Tensor patches({B, n, patch_dim});
    const float* img = image.data<float>();
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

    Tensor patches_flat = patches.reshape({B * n, patch_dim});
    Tensor token_emb_flat({B * n, D});
    math::gemm(1.0f, patches_flat, patch_embed, 0.0f, token_emb_flat);
    Tensor token_emb = token_emb_flat.reshape({B, n, D});

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
    Tensor flat_mask({(n + 1) * (n + 1)});
    flat_mask.fill(0.0f);
    KVCache dummy_cache;
    for (auto& block : blocks)
        h = block.forward(h, positions, flat_mask, dummy_cache, 0);

    return h;
}

Tensor VisionEncoder::classify(const Tensor& encoded) {
    int64_t B = encoded.dim(0);
    int64_t D = encoded.dim(2);

    Tensor cls_feat({B, D});
    const float* f = encoded.data<float>();
    float* cf = cls_feat.data<float>();
    for (int64_t b = 0; b < B; ++b)
        std::memcpy(cf + b * D, f + b * encoded.dim(1) * D, D * sizeof(float));

    Tensor logits({B, num_classes});
    math::gemm(1.0f, cls_feat, class_head, 0.0f, logits);
    return logits;
}

Tensor VisionEncoder::detect_objects(const Tensor& encoded) {
    int64_t B = encoded.dim(0);
    int64_t S = encoded.dim(1);
    int64_t D = hidden_size;
    int64_t N = max_detections;
    int64_t P = S - 1;

    Tensor spatial({B, P, D});
    const float* enc = encoded.data<float>();
    float* sp = spatial.data<float>();
    for (int64_t b = 0; b < B; ++b)
        std::memcpy(sp + b * P * D, enc + b * S * D + D, P * D * sizeof(float));

    Tensor spatial_flat = spatial.reshape({B * P, D});
    Tensor bbox_raw({B * P, 4});
    math::gemm(1.0f, spatial_flat, bbox_head, 0.0f, bbox_raw);
    Tensor bbox_per_token = bbox_raw.reshape({B, P, 4});

    Tensor class_raw({B * P, num_classes});
    math::gemm(1.0f, spatial_flat, class_head, 0.0f, class_raw);

    Tensor bbox_out({B, N, 4});
    bbox_out.fill(0.0f);

    for (int64_t b = 0; b < B; ++b) {
        std::vector<std::pair<float, int64_t>> scores(P);
        const float* cr = class_raw.data<float>() + b * P * num_classes;
        for (int64_t p = 0; p < P; ++p) {
            float max_score = -INFINITY;
            for (int64_t c = 0; c < num_classes; ++c)
                if (cr[p * num_classes + c] > max_score)
                    max_score = cr[p * num_classes + c];
            scores[p] = {max_score, p};
        }
        std::sort(scores.begin(), scores.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        const float* bb = bbox_per_token.data<float>() + b * P * 4;
        float* bo = bbox_out.data<float>() + b * N * 4;
        int64_t k = std::min(N, P);
        for (int64_t i = 0; i < k; ++i) {
            int64_t idx = scores[i].second;
            std::memcpy(bo + i * 4, bb + idx * 4, 4 * sizeof(float));
        }
    }

    return bbox_out;
}

Tensor VisionEncoder::generate_caption(const Tensor& encoded, int64_t max_len) {
    int64_t B = encoded.dim(0);
    int64_t D = hidden_size;

    Tensor cls_feat({B, D});
    const float* f = encoded.data<float>();
    float* cf = cls_feat.data<float>();
    for (int64_t b = 0; b < B; ++b)
        std::memcpy(cf + b * D, f + b * encoded.dim(1) * D, D * sizeof(float));

    Tensor visual_context({B, D});
    math::gemm(1.0f, cls_feat, caption_proj, 0.0f, visual_context);

    int64_t vocab_sz = caption_embed.dim(0);
    Tensor captions({B, max_len, vocab_sz});
    captions.fill(0.0f);

    int64_t bos_id = 1;
    int64_t eos_id = 2;

    Tensor token_ids({B, 1});
    int* tid = token_ids.data<int>();
    for (int64_t b = 0; b < B; ++b) tid[b] = static_cast<int>(bos_id);

    for (int64_t pos = 0; pos < max_len; ++pos) {
        Tensor embedding({B, D});
        const float* ce = caption_embed.data<float>();
        float* emb = embedding.data<float>();
        for (int64_t b = 0; b < B; ++b) {
            int64_t tok = tid[b];
            if (tok < 0) tok = 0;
            if (tok >= vocab_sz) tok = 0;
            std::memcpy(emb + b * D, ce + tok * D, D * sizeof(float));
        }

        for (int64_t i = 0; i < B * D; ++i)
            emb[i] += visual_context.data<float>()[i];

        Tensor logits({B, vocab_sz});
        const float* emb_ptr = embedding.data<float>();
        float* log_ptr = logits.data<float>();
        for (int64_t b = 0; b < B; ++b)
            for (int64_t v = 0; v < vocab_sz; ++v) {
                float dot = 0.0f;
                for (int64_t d = 0; d < D; ++d)
                    dot += emb_ptr[b * D + d] * ce[v * D + d];
                log_ptr[b * vocab_sz + v] = dot;
            }

        for (int64_t b = 0; b < B; ++b) {
            std::memcpy(captions.data<float>() + (b * max_len + pos) * vocab_sz,
                        log_ptr + b * vocab_sz, vocab_sz * sizeof(float));

            int64_t next_token = 0;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < vocab_sz; ++v) {
                if (log_ptr[b * vocab_sz + v] > max_l) {
                    max_l = log_ptr[b * vocab_sz + v];
                    next_token = v;
                }
            }
            tid[b] = static_cast<int>(next_token);
        }
    }

    return captions;
}

VisionResult VisionEncoder::forward(const Tensor& image, VisionTask task) {
    VisionResult result;
    result.features = encode(image);

    if (task == VisionTask::CLASSIFY) {
        result.class_logits = classify(result.features);
        result.bboxes = Tensor({1, max_detections, 4});
        result.bboxes.fill(0.0f);
        result.captions = Tensor({1, 1, 1});
        result.captions.fill(0.0f);
        result.confidence = 1.0f;
    } else if (task == VisionTask::DETECT) {
        result.bboxes = detect_objects(result.features);
        result.class_logits = classify(result.features);
        result.captions = Tensor({1, 1, 1});
        result.captions.fill(0.0f);
        result.confidence = 1.0f;
    } else if (task == VisionTask::SEGMENT) {
        result.bboxes = detect_objects(result.features);
        result.class_logits = classify(result.features);
        result.captions = Tensor({1, 1, 1});
        result.captions.fill(0.0f);
        result.confidence = 1.0f;
    } else if (task == VisionTask::CAPTION) {
        int64_t max_cap = result.features.dim(1);
        result.captions = generate_caption(result.features, max_cap);
        result.class_logits = Tensor({1, num_classes});
        result.class_logits.fill(0.0f);
        result.bboxes = Tensor({1, max_detections, 4});
        result.bboxes.fill(0.0f);
        result.confidence = 1.0f;
    } else if (task == VisionTask::DEPTH_ESTIMATE) {
        result.class_logits = classify(result.features);
        result.bboxes = detect_objects(result.features);
        result.captions = Tensor({1, 1, 1});
        result.captions.fill(0.0f);
        result.confidence = 1.0f;
    }

    return result;
}

// ============= ViTFeatureExtractor =============

ViTFeatureExtractor::ViTFeatureExtractor(int64_t img_sz, int64_t patch_sz)
    : img_size(img_sz), patch_size(patch_sz) {}

Tensor ViTFeatureExtractor::extract_patches(const Tensor& image) {
    int64_t B = image.dim(0);
    int64_t C = image.dim(1);
    int64_t H_ = image.dim(2);
    int64_t W_ = image.dim(3);
    int64_t patch_dim = C * patch_size * patch_size;
    int64_t n_patches_h = H_ / patch_size;
    int64_t n_patches_w = W_ / patch_size;
    int64_t N = n_patches_h * n_patches_w;

    Tensor patches({B, N, patch_dim});
    const float* img = image.data<float>();
    float* pat = patches.data<float>();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < n_patches_h; ++i)
            for (int64_t j = 0; j < n_patches_w; ++j) {
                int64_t p_idx = i * n_patches_w + j;
                for (int64_t c = 0; c < C; ++c)
                    for (int64_t pi = 0; pi < patch_size; ++pi)
                        for (int64_t pj = 0; pj < patch_size; ++pj) {
                            int64_t src = ((b * C + c) * H_ + i * patch_size + pi) * W_ + j * patch_size + pj;
                            int64_t dst = (b * N + p_idx) * patch_dim + (c * patch_size + pi) * patch_size + pj;
                            pat[dst] = img[src];
                        }
            }

    return patches;
}

Tensor ViTFeatureExtractor::normalize(const Tensor& image) {
    int64_t B = image.dim(0);
    int64_t C = image.dim(1);
    int64_t H_ = image.dim(2);
    int64_t W_ = image.dim(3);

    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float std[3]  = {0.229f, 0.224f, 0.225f};

    Tensor out({B, C, H_, W_});
    const float* img = image.data<float>();
    float* o = out.data<float>();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t i = 0; i < H_; ++i)
                for (int64_t j = 0; j < W_; ++j) {
                    int64_t idx = ((b * C + c) * H_ + i) * W_ + j;
                    o[idx] = (img[idx] - mean[c]) / std[c];
                }

    return out;
}

// ============= Conv2dPatchEmbed =============

Conv2dPatchEmbed::Conv2dPatchEmbed(int64_t in_c, int64_t embed_d, int64_t ps)
    : in_channels(in_c), embed_dim(embed_d), patch_size(ps) {
    int64_t fan_in = in_channels * ps * ps;
    int64_t fan_out = embed_dim * ps * ps;
    float scale = std::sqrt(2.0f / (float)(fan_in + fan_out));

    weight = Tensor({embed_dim, in_channels * ps * ps});
    float* w = weight.data<float>();
    for (int64_t i = 0; i < weight.numel(); ++i)
        w[i] = scale * ((float)std::rand() / (float)RAND_MAX * 2.0f - 1.0f);

    bias = Tensor({embed_dim});
    bias.zero_();
}

Tensor Conv2dPatchEmbed::forward(const Tensor& x) {
    int64_t B = x.dim(0);
    int64_t C = x.dim(1);
    int64_t H = x.dim(2);
    int64_t W = x.dim(3);
    int64_t ps = patch_size;
    int64_t n_h = H / ps;
    int64_t n_w = W / ps;
    int64_t N = n_h * n_w;
    int64_t D = embed_dim;

    Tensor cols({B * N, C * ps * ps});
    float* col = cols.data<float>();
    const float* inp = x.data<float>();

    for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < n_h; ++i)
            for (int64_t j = 0; j < n_w; ++j) {
                int64_t p_idx = i * n_w + j;
                for (int64_t c = 0; c < C; ++c)
                    for (int64_t pi = 0; pi < ps; ++pi)
                        for (int64_t pj = 0; pj < ps; ++pj) {
                            int64_t src = ((b * C + c) * H + i * ps + pi) * W + j * ps + pj;
                            int64_t dst = (b * N + p_idx) * (C * ps * ps)
                                        + (c * ps + pi) * ps + pj;
                            col[dst] = inp[src];
                        }
            }

    Tensor output({B * N, D});
    math::gemm(1.0f, cols, weight, 0.0f, output);

    float* out = output.data<float>();
    const float* b = bias.data<float>();
    for (int64_t i = 0; i < B * N; ++i)
        for (int64_t d = 0; d < D; ++d)
            out[i * D + d] += b[d];

    return output.reshape({B, N, D});
}

// ============= LayerNorm2D =============

LayerNorm2D::LayerNorm2D(int64_t normalized_shape, float eps_val)
    : eps(eps_val) {
    gamma = Tensor({normalized_shape});
    beta = Tensor({normalized_shape});
    float* g = gamma.data<float>();
    float* b = beta.data<float>();
    for (int64_t i = 0; i < normalized_shape; ++i) {
        g[i] = 1.0f;
        b[i] = 0.0f;
    }
}

Tensor LayerNorm2D::forward(const Tensor& x) {
    int64_t B = x.dim(0);
    int64_t N = x.dim(1);
    int64_t D = x.dim(2);
    int64_t norm_dim = gamma.numel();

    Tensor y({B, N, norm_dim});
    const float* inp = x.data<float>();
    float* out = y.data<float>();
    const float* g = gamma.data<float>();
    const float* b = beta.data<float>();

    for (int64_t i = 0; i < B * N; ++i) {
        const float* row = inp + i * D;
        float mean = 0.0f;
        for (int64_t j = 0; j < norm_dim; ++j)
            mean += row[j];
        mean /= (float)norm_dim;

        float var = 0.0f;
        for (int64_t j = 0; j < norm_dim; ++j) {
            float diff = row[j] - mean;
            var += diff * diff;
        }
        var /= (float)norm_dim;

        float inv_std = 1.0f / std::sqrt(var + eps);
        float* o = out + i * norm_dim;
        for (int64_t j = 0; j < norm_dim; ++j)
            o[j] = (row[j] - mean) * inv_std * g[j] + b[j];
    }

    return y;
}

// ============= interpolate_pos_encoding_bilinear =============

Tensor interpolate_pos_encoding_bilinear(const Tensor& pos_embed,
    int64_t old_h, int64_t old_w, int64_t new_h, int64_t new_w,
    int64_t ps, int64_t D) {
    int64_t n_old = old_h * old_w;
    int64_t n_new = new_h * new_w;

    Tensor new_pe({1, n_new + 1, D});
    float* out = new_pe.data<float>();

    const float* pe = pos_embed.data<float>();

    std::memcpy(out, pe, D * sizeof(float));

    for (int64_t d = 0; d < D; ++d) {
        for (int64_t i = 0; i < new_h; ++i) {
            for (int64_t j = 0; j < new_w; ++j) {
                float ih = (float)i * (float)old_h / (float)new_h;
                float iw = (float)j * (float)old_w / (float)new_w;

                int64_t ih0 = (int64_t)std::floor(ih);
                int64_t ih1 = std::min(ih0 + 1, old_h - 1);
                int64_t iw0 = (int64_t)std::floor(iw);
                int64_t iw1 = std::min(iw0 + 1, old_w - 1);

                float dh = ih - (float)ih0;
                float dw = iw - (float)iw0;

                float v00 = pe[(1 + ih0 * old_w + iw0) * D + d];
                float v01 = pe[(1 + ih0 * old_w + iw1) * D + d];
                float v10 = pe[(1 + ih1 * old_w + iw0) * D + d];
                float v11 = pe[(1 + ih1 * old_w + iw1) * D + d];

                float v0 = v00 * (1.0f - dw) + v01 * dw;
                float v1 = v10 * (1.0f - dw) + v11 * dw;
                out[(1 + i * new_w + j) * D + d] = v0 * (1.0f - dh) + v1 * dh;
            }
        }
    }

    return new_pe;
}

Tensor VisionEncoder::interpolate_pos_encoding(int64_t new_h, int64_t new_w) const {
    int64_t old_h = img_size / patch_size;
    int64_t old_w = img_size / patch_size;
    return interpolate_pos_encoding_bilinear(
        pos_embed, old_h, old_w, new_h, new_w, patch_size, hidden_size);
}

Tensor VisionEncoder::encode_multiscale(const std::vector<Tensor>& images) {
    int64_t n_scales = (int64_t)images.size();
    if (n_scales == 0)
        return Tensor::zeros({1, 1, hidden_size});

    std::vector<Tensor> scale_features;
    for (int64_t s = 0; s < n_scales; ++s)
        scale_features.push_back(encode(images[s]));

    int64_t max_t = 0;
    for (auto& f : scale_features)
        max_t = std::max(max_t, f.dim(1));

    int64_t B = scale_features[0].dim(0);
    int64_t D = hidden_size;
    Tensor fused({B, max_t * n_scales, D});
    fused.zero_();
    float* fd = fused.data<float>();

    for (int64_t s = 0; s < n_scales; ++s) {
        int64_t T = scale_features[s].dim(1);
        const float* sf = scale_features[s].data<float>();
        for (int64_t b = 0; b < B; ++b)
            for (int64_t t = 0; t < T; ++t)
                for (int64_t d = 0; d < D; ++d)
                    fd[((b * n_scales + s) * max_t + t) * D + d] = sf[(b * T + t) * D + d];
    }

    return fused;
}

std::vector<Tensor> VisionEncoder::extract_attention_maps(const Tensor& image) {
    int64_t B = image.dim(0);
    int64_t C = image.dim(1);
    int64_t H = image.dim(2);
    int64_t W = image.dim(3);
    int64_t n_patches_h = H / patch_size;
    int64_t n_patches_w = W / patch_size;
    int64_t n = n_patches_h * n_patches_w;
    int64_t D = hidden_size;
    int64_t patch_dim = C * patch_size * patch_size;

    Tensor patches({B, n, patch_dim});
    const float* img = image.data<float>();
    float* pat = patches.data<float>();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < n_patches_h; ++i)
            for (int64_t j = 0; j < n_patches_w; ++j) {
                int64_t p_idx = i * n_patches_w + j;
                for (int64_t c = 0; c < C; ++c)
                    for (int64_t pi = 0; pi < patch_size; ++pi)
                        for (int64_t pj = 0; pj < patch_size; ++pj) {
                            int64_t src = ((b * C + c) * H + i * patch_size + pi) * W + j * patch_size + pj;
                            int64_t dst = (b * n + p_idx) * patch_dim + (c * patch_size + pi) * patch_size + pj;
                            pat[dst] = img[src];
                        }
            }

    Tensor patches_flat = patches.reshape({B * n, patch_dim});
    Tensor token_emb_flat({B * n, D});
    math::gemm(1.0f, patches_flat, patch_embed, 0.0f, token_emb_flat);
    Tensor token_emb = token_emb_flat.reshape({B, n, D});

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
    Tensor flat_mask({(n + 1) * (n + 1)});
    flat_mask.fill(0.0f);
    KVCache dummy_cache;

    std::vector<Tensor> attention_maps;
    for (auto& block : blocks)
        h = block.forward(h, positions, flat_mask, dummy_cache, 0);

    return attention_maps;
}

void VisionEncoder::init_weights(float std_val) {
    float scale = std_val > 0 ? std_val : 0.02f;
    auto init_tensor = [scale](Tensor& t) {
        float* d = t.data<float>();
        for (int64_t i = 0; i < t.numel(); ++i)
            d[i] = scale * ((float)std::rand() / (float)RAND_MAX * 2.0f - 1.0f);
    };
    init_tensor(patch_embed);
    float* pe = pos_embed.data<float>();
    for (int64_t p = 0; p < num_patches + 1; ++p)
        for (int64_t d = 0; d < hidden_size; ++d) {
            float ang = (float)p / std::pow(10000.0f, (float)(2 * (d / 2)) / (float)hidden_size);
            pe[p * hidden_size + d] = (d % 2 == 0) ? std::sin(ang) : std::cos(ang);
        }
    init_tensor(cls_token);
    init_tensor(class_head);
    init_tensor(bbox_head);
}

Tensor VisionEncoder::extract_patch_features(const Tensor& patches) {
    int64_t B = patches.dim(0);
    int64_t N = patches.dim(1);
    int64_t patch_dim = patches.dim(2);
    int64_t D = hidden_size;

    Tensor flat = patches.reshape({B * N, patch_dim});
    Tensor emb_flat({B * N, D});
    math::gemm(1.0f, flat, patch_embed, 0.0f, emb_flat);
    return emb_flat.reshape({B, N, D});
}

Tensor VisionEncoder::compute_self_attention(const Tensor& query, const Tensor& key) {
    int64_t B = query.dim(0);
    int64_t N = query.dim(1);
    int64_t D = query.dim(2);

    Tensor q_flat = query.reshape({B * N, D});
    Tensor k_flat = key.reshape({B * N, D});
    Tensor kt_flat({D, B * N});
    const float* kd = k_flat.data<float>();
    float* kt = kt_flat.data<float>();
    for (int64_t i = 0; i < B * N; ++i)
        for (int64_t j = 0; j < D; ++j)
            kt[j * (B * N) + i] = kd[i * D + j];

    Tensor scores_flat({B * N, B * N});
    math::gemm(1.0f, q_flat, kt_flat, 0.0f, scores_flat);

    float inv_sqrt_d = 1.0f / std::sqrt((float)D);
    float* sc = scores_flat.data<float>();
    for (int64_t i = 0; i < B * N * B * N; ++i)
        sc[i] *= inv_sqrt_d;

    Tensor attn = scores_flat.reshape({B, N, B, N});
    return attn;
}

// ============= ViTFeatureExtractor additions =============

Tensor ViTFeatureExtractor::resize(const Tensor& image, int64_t new_h, int64_t new_w) const {
    int64_t B = image.dim(0);
    int64_t C = image.dim(1);
    int64_t H = image.dim(2);
    int64_t W = image.dim(3);

    Tensor out({B, C, new_h, new_w});
    const float* inp = image.data<float>();
    float* o = out.data<float>();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t i = 0; i < new_h; ++i) {
                for (int64_t j = 0; j < new_w; ++j) {
                    float ih = (float)i * (float)H / (float)new_h;
                    float iw = (float)j * (float)W / (float)new_w;
                    int64_t ih0 = (int64_t)std::floor(ih);
                    int64_t ih1 = std::min(ih0 + 1, H - 1);
                    int64_t iw0 = (int64_t)std::floor(iw);
                    int64_t iw1 = std::min(iw0 + 1, W - 1);
                    float dh = ih - (float)ih0;
                    float dw = iw - (float)iw0;

                    float v00 = inp[((b * C + c) * H + ih0) * W + iw0];
                    float v01 = inp[((b * C + c) * H + ih0) * W + iw1];
                    float v10 = inp[((b * C + c) * H + ih1) * W + iw0];
                    float v11 = inp[((b * C + c) * H + ih1) * W + iw1];
                    float v0 = v00 * (1.0f - dw) + v01 * dw;
                    float v1 = v10 * (1.0f - dw) + v11 * dw;
                    o[((b * C + c) * new_h + i) * new_w + j] = v0 * (1.0f - dh) + v1 * dh;
                }
            }
        }
    }
    return out;
}

Tensor ViTFeatureExtractor::center_crop(const Tensor& image, int64_t size) const {
    int64_t B = image.dim(0);
    int64_t C = image.dim(1);
    int64_t H = image.dim(2);
    int64_t W = image.dim(3);

    int64_t top = (H - size) / 2;
    int64_t left = (W - size) / 2;

    Tensor out({B, C, size, size});
    const float* inp = image.data<float>();
    float* o = out.data<float>();

    for (int64_t b = 0; b < B; ++b)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t i = 0; i < size; ++i)
                for (int64_t j = 0; j < size; ++j) {
                    int64_t src = ((b * C + c) * H + top + i) * W + left + j;
                    int64_t dst = ((b * C + c) * size + i) * size + j;
                    o[dst] = inp[src];
                }

    return out;
}

Tensor ViTFeatureExtractor::random_erasing(const Tensor& image, float prob, float scale) const {
    int64_t B = image.dim(0);
    int64_t C = image.dim(1);
    int64_t H = image.dim(2);
    int64_t W = image.dim(3);

    Tensor out = image.clone();
    float* o = out.data<float>();

    for (int64_t b = 0; b < B; ++b) {
        if ((float)std::rand() / (float)RAND_MAX >= prob) continue;

        int64_t er_h = (int64_t)((float)H * std::sqrt(scale) * ((float)std::rand() / (float)RAND_MAX * 0.4f + 0.8f));
        int64_t er_w = (int64_t)((float)W * std::sqrt(scale) * ((float)std::rand() / (float)RAND_MAX * 0.4f + 0.8f));
        er_h = std::max<int64_t>(1, std::min(er_h, H));
        er_w = std::max<int64_t>(1, std::min(er_w, W));

        int64_t top = std::rand() % (H - er_h + 1);
        int64_t left = std::rand() % (W - er_w + 1);

        float mean_val = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            mean_val = 0.0f;
            for (int64_t i = 0; i < H; ++i)
                for (int64_t j = 0; j < W; ++j)
                    mean_val += o[((b * C + c) * H + i) * W + j];
            mean_val /= (float)(H * W);
            for (int64_t i = 0; i < er_h; ++i)
                for (int64_t j = 0; j < er_w; ++j) {
                    int64_t idx = ((b * C + c) * H + top + i) * W + left + j;
                    o[idx] = mean_val;
                }
        }
    }

    return out;
}

Tensor ViTFeatureExtractor::normalize_quantized(const Tensor& image) const {
    int64_t B = image.dim(0);
    int64_t C = image.dim(1);
    int64_t H = image.dim(2);
    int64_t W = image.dim(3);

    static const float q_means[3] = {123.675f, 116.28f, 103.53f};
    static const float q_stds[3]  = {58.395f, 57.12f, 57.375f};

    Tensor out({B, C, H, W});
    const float* img = image.data<float>();
    float* o = out.data<float>();

    for (int64_t b = 0; b < B; ++b)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t i = 0; i < H; ++i)
                for (int64_t j = 0; j < W; ++j) {
                    int64_t idx = ((b * C + c) * H + i) * W + j;
                    o[idx] = (img[idx] * 255.0f - q_means[c]) / q_stds[c];
                }

    return out;
}

} // namespace multimodal
} // namespace oil
