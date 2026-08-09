#include "vision.h"
#include "quant/math.h"
#include "quant/kv_cache.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace quant {
namespace multimodal {

// ============= ObjectDetector =============

ObjectDetector::ObjectDetector(int64_t hidden, int64_t classes, int64_t max_det)
    : hidden_size(hidden), num_classes(classes), max_detections(max_det)
{
    bbox_head = Tensor({hidden, 4});
    bbox_head.zero_();
    class_head = Tensor({hidden, classes});
    class_head.zero_();
    object_query = Tensor({max_detections, hidden});
    object_query.zero_();
    q_proj = Tensor({hidden, hidden}); q_proj.zero_();
    k_proj = Tensor({hidden, hidden}); k_proj.zero_();
    v_proj = Tensor({hidden, hidden}); v_proj.zero_();
}

Tensor ObjectDetector::forward(const Tensor& visual_features) {
    int64_t B = visual_features.dim(0);
    int64_t D = hidden_size;
    int64_t N = max_detections;
    int64_t S = visual_features.dim(1);
    Tensor queries = object_query.reshape({1, N, D});
    Tensor queries_batch({B, N, D});
    float* qb = queries_batch.data<float>();
    const float* q = queries.data<float>();
    for (int64_t b = 0; b < B; ++b)
        std::memcpy(qb + b * N * D, q, N * D * sizeof(float));

    // Cross-attention with learned Q/K/V projections
    Tensor vis_flat = visual_features.reshape({B * S, D});
    Tensor q_flat = queries_batch.reshape({B * N, D});
    Tensor q_proj_flat({B * N, D});
    Tensor k_proj_flat({B * S, D});
    Tensor v_proj_flat({B * S, D});
    math::gemm(1.0f, q_flat, q_proj, 0.0f, q_proj_flat);
    math::gemm(1.0f, vis_flat, k_proj, 0.0f, k_proj_flat);
    math::gemm(1.0f, vis_flat, v_proj, 0.0f, v_proj_flat);
    Tensor attn_out({B, N, D});
    const float* vis = v_proj_flat.data<float>();
    const float* q_ptr = q_proj_flat.data<float>();
    float* ao = attn_out.data<float>();
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t n = 0; n < N; ++n) {
            std::vector<float> scores(S);
            for (int64_t s = 0; s < S; ++s) {
                float dot = 0.0f;
                for (int64_t d = 0; d < D; ++d)
                    dot += q_ptr[b * N * D + n * D + d] * k_proj_flat.data<float>()[b * S * D + s * D + d];
                scores[s] = dot / std::sqrt((float)D);
            }
            float row_max = -INFINITY;
            for (int64_t s = 0; s < S; ++s)
                if (scores[s] > row_max) row_max = scores[s];
            float sum = 0.0f;
            for (int64_t s = 0; s < S; ++s) {
                scores[s] = std::exp(scores[s] - row_max);
                sum += scores[s];
            }
            float inv_sum = 1.0f / (sum + 1e-10f);
            for (int64_t s = 0; s < S; ++s) scores[s] *= inv_sum;
            for (int64_t d = 0; d < D; ++d) {
                float val = 0.0f;
                for (int64_t s = 0; s < S; ++s)
                    val += scores[s] * vis[b * S * D + s * D + d];
                ao[b * N * D + n * D + d] = val;
            }
        }
    }

    Tensor attn_flat = attn_out.reshape({B * N, D});
    Tensor bbox_raw({B * N, 4});
    Tensor class_raw({B * N, num_classes});
    math::gemm(1.0f, attn_flat, bbox_head, 0.0f, bbox_raw);
    math::gemm(1.0f, attn_flat, class_head, 0.0f, class_raw);
    Tensor bbox_out = bbox_raw.reshape({B, N, 4});
    last_class_scores = class_raw.reshape({B, N, num_classes});

    return bbox_out;
}

// ============= SceneGraphEncoder =============

SceneGraphEncoder::SceneGraphEncoder(int64_t hidden, int64_t num_rel)
    : hidden_size(hidden), num_relation_types(num_rel)
{
    relation_head = Tensor({hidden * 2, num_rel});
    relation_head.zero_();
    spatial_embed = Tensor({4, hidden});
    spatial_embed.zero_();
}

Tensor SceneGraphEncoder::forward(const Tensor& obj_features, const Tensor& spatial_edges) {
    int64_t N = obj_features.dim(0);
    int64_t D = hidden_size;
    int64_t num_pairs = N * N;

    Tensor obj_expanded_a({num_pairs, D});
    Tensor obj_expanded_b({num_pairs, D});
    const float* of = obj_features.data<float>();
    float* oa = obj_expanded_a.data<float>();
    float* ob = obj_expanded_b.data<float>();
    for (int64_t i = 0; i < N; ++i)
        for (int64_t j = 0; j < N; ++j) {
            std::memcpy(oa + (i * N + j) * D, of + i * D, D * sizeof(float));
            std::memcpy(ob + (i * N + j) * D, of + j * D, D * sizeof(float));
        }

    Tensor pair_feats({num_pairs, D * 2});
    float* pf = pair_feats.data<float>();
    for (int64_t p = 0; p < num_pairs; ++p) {
        std::memcpy(pf + p * D * 2, oa + p * D, D * sizeof(float));
        std::memcpy(pf + p * D * 2 + D, ob + p * D, D * sizeof(float));
    }

    const float* se = spatial_embed.data<float>();
    const int64_t spatial_embed_rows = spatial_embed.dim(0);
    for (int64_t p = 0; p < num_pairs; ++p) {
        int64_t edge_idx = (int64_t)std::round(spatial_edges.data<float>()[p]);
        if (edge_idx >= spatial_embed_rows) edge_idx = spatial_embed_rows - 1;
        if (edge_idx < 0) edge_idx = 0;
        for (int64_t d = 0; d < D; ++d)
            pf[p * D * 2 + d] += se[edge_idx * D + d];
    }

    Tensor relations({num_pairs, num_relation_types});
    math::gemm(1.0f, pair_feats, relation_head, 0.0f, relations);
    return relations;
}

// ============= VisionEncoder =============

VisionEncoder::VisionEncoder(int64_t img_sz, int64_t patch_sz, int64_t hidden,
                             int64_t num_layers, int64_t num_heads, int64_t classes,
                             int64_t max_frm)
    : img_size(img_sz), patch_size(patch_sz), hidden_size(hidden),
      num_classes(classes), max_frames(max_frm),
      detector(hidden, classes, 100),
      scene_encoder(hidden, 50)
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
    temporal_pos_embed = Tensor({max_frm, hidden});
    temporal_pos_embed.zero_();
    caption_proj = Tensor({hidden, hidden});
    caption_proj.zero_();
    global_pool = Tensor({hidden, hidden});
    global_pool.zero_();

    TransformerConfig tcfg;
    tcfg.hidden_size = hidden;
    tcfg.num_layers = num_layers;
    tcfg.num_heads = num_heads;
    tcfg.head_dim = hidden / num_heads;

    blocks.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i)
        blocks.emplace_back(tcfg);
}

Tensor VisionEncoder::encode_visual(const Tensor& visual_input) {
    int64_t B = visual_input.dim(0);
    int64_t C = visual_input.dim(1);
    int64_t H_ = visual_input.dim(2);
    int64_t W_ = visual_input.dim(3);
    int64_t patch_dim = (int64_t)patch_embed.dim(0);
    int64_t patch_sz = (int64_t)std::sqrt((double)(patch_dim / C));
    int64_t n_patches_h = H_ / patch_sz;
    int64_t n_patches_w = W_ / patch_sz;
    int64_t n = n_patches_h * n_patches_w;
    int64_t D = hidden_size;

    Tensor patches({B, n, patch_dim});
    const float* img = visual_input.data<float>();
    float* pat = patches.data<float>();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < n_patches_h; ++i)
            for (int64_t j = 0; j < n_patches_w; ++j) {
                int64_t p_idx = i * n_patches_w + j;
                for (int64_t c = 0; c < C; ++c)
                    for (int64_t pi = 0; pi < patch_sz; ++pi)
                        for (int64_t pj = 0; pj < patch_sz; ++pj) {
                            int64_t src = ((b * C + c) * H_ + i * patch_sz + pi) * W_ + j * patch_sz + pj;
                            int64_t dst = (b * n + p_idx) * patch_dim + (c * patch_sz + pi) * patch_sz + pj;
                            pat[dst] = img[src];
                        }
            }

    Tensor token_emb({B, n, D});
    Tensor patches_flat = patches.reshape({B * n, patch_dim});
    Tensor token_emb_flat({B * n, D});
    math::gemm(1.0f, patches_flat, patch_embed, 0.0f, token_emb_flat);
    token_emb = token_emb_flat.reshape({B, n, D});

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

SceneGraph VisionEncoder::understand_image(const Tensor& image) {
    Tensor features = encode_visual(image);
    int64_t B = features.dim(0), N = features.dim(1), D = features.dim(2);

    Tensor cls_features({B, 1, D});
    const float* f = features.data<float>();
    float* cf = cls_features.data<float>();
    for (int64_t b = 0; b < B; ++b)
        std::memcpy(cf + b * D, f + b * N * D, D * sizeof(float));

    Tensor bbox_out = detector.forward(features);
    int64_t n_spatial = N - 1;
    int64_t n_patches_h = (int64_t)std::sqrt((double)n_spatial);
    if (n_patches_h < 1) n_patches_h = 1;
    int64_t n_patches_w = n_spatial / n_patches_h;
    if (n_patches_w < 1) n_patches_w = 1;
    Tensor spatial_edges({N * N});
    float* se = spatial_edges.data<float>();
    for (int64_t i = 0; i < N; ++i)
        for (int64_t j = 0; j < N; ++j) {
            if (i == 0 || j == 0) { se[i * N + j] = 0.0f; continue; }
            int64_t si = i - 1, sj = j - 1;
            int64_t pi = si / n_patches_w, pj = si % n_patches_w;
            int64_t qi = sj / n_patches_w, qj = sj % n_patches_w;
            int64_t di = pi - qi, dj = pj - qj;
            se[i * N + j] = (float)((std::abs(di) > std::abs(dj))
                ? (di > 0 ? 0 : 1) : (dj > 0 ? 2 : 3));
        }
    Tensor scene_rel = scene_encoder.forward(features.reshape({N, D}), spatial_edges);

    SceneGraph graph;
    graph.objects = bbox_out;
    graph.labels = detector.last_class_scores;
    Tensor global_feat({B, D});
    Tensor cls_feat_2d = cls_features.reshape({B, D});
    math::gemm(1.0f, cls_feat_2d, global_pool, 0.0f, global_feat);
    graph.global_features = global_feat.reshape({D});

    Tensor caption_out({B, D});
    math::gemm(1.0f, cls_feat_2d, caption_proj, 0.0f, caption_out);
    graph.caption_embeds = caption_out;
    graph.completed_tasks = {VisualTask::CLASSIFY, VisualTask::DETECT,
                             VisualTask::SCENE_GRAPH, VisualTask::CAPTION};
    return graph;
}

Tensor VisionEncoder::classify(const Tensor& image) {
    Tensor features = encode_visual(image);
    int64_t B = features.dim(0), D = features.dim(2);
    Tensor cls_feat({B, D});
    const float* f = features.data<float>();
    float* cf = cls_feat.data<float>();
    for (int64_t b = 0; b < B; ++b)
        std::memcpy(cf + b * D, f + b * features.dim(1) * D, D * sizeof(float));

    Tensor logits({B, num_classes});
    math::gemm(1.0f, cls_feat, global_pool, 0.0f, logits);
    return logits;
}

Tensor VisionEncoder::detect(const Tensor& image) {
    Tensor features = encode_visual(image);
    return detector.forward(features);
}

Tensor VisionEncoder::caption(const Tensor& image) {
    Tensor features = encode_visual(image);
    int64_t B = features.dim(0), D = features.dim(2);
    Tensor cls_feat({B, D});
    const float* f = features.data<float>();
    float* cf = cls_feat.data<float>();
    for (int64_t b = 0; b < B; ++b)
        std::memcpy(cf + b * D, f + b * features.dim(1) * D, D * sizeof(float));
    Tensor caption_emb({B, D});
    math::gemm(1.0f, cls_feat, caption_proj, 0.0f, caption_emb);
    return caption_emb;
}

SceneGraph VisionEncoder::build_scene_graph(const Tensor& image) {
    return understand_image(image);
}

SceneGraph VisionEncoder::understand_video(const Tensor& video_frames) {
    int64_t F = video_frames.dim(0);
    int64_t C = video_frames.dim(1);
    int64_t H_ = video_frames.dim(2);
    int64_t W_ = video_frames.dim(3);
    int64_t D = hidden_size;
    int64_t patch_dim = (int64_t)patch_embed.dim(0);
    int64_t patch_sz = (int64_t)std::sqrt((double)(patch_dim / C));
    int64_t n_patches_h = H_ / patch_sz;
    int64_t n_patches_w = W_ / patch_sz;
    int64_t n_spatial = n_patches_h * n_patches_w;
    int64_t n_temporal = std::min(F, max_frames);

    Tensor all_features({n_temporal, n_spatial + 1, D});
    for (int64_t t = 0; t < n_temporal; ++t) {
        Tensor frame({1, C, H_, W_});
        const float* src = video_frames.data<float>() + t * C * H_ * W_;
        std::memcpy(frame.data<float>(), src, C * H_ * W_ * sizeof(float));
        Tensor feat = encode_visual(frame);
        const float* f = feat.data<float>();
        float* af = all_features.data<float>() + t * (n_spatial + 1) * D;
        std::memcpy(af, f, (n_spatial + 1) * D * sizeof(float));
    }

    const float* te = temporal_pos_embed.data<float>();
    float* af_data = all_features.data<float>();
    for (int64_t t = 0; t < n_temporal; ++t)
        for (int64_t i = 0; i < (n_spatial + 1); ++i)
            for (int64_t d = 0; d < D; ++d)
                af_data[(t * (n_spatial + 1) + i) * D + d] += te[t * D + d];

    Tensor seq({1, n_temporal * (n_spatial + 1), D});
    std::memcpy(seq.data<float>(), af_data, n_temporal * (n_spatial + 1) * D * sizeof(float));

    Tensor temp_positions = Tensor::arange(n_temporal * (n_spatial + 1));
    Tensor flat_mask({(n_temporal * (n_spatial + 1)) * (n_temporal * (n_spatial + 1))});
    flat_mask.fill(0.0f);
    KVCache dummy_cache;
    Tensor h = seq;
    for (auto& block : blocks)
        h = block.forward(h, temp_positions, flat_mask, dummy_cache, 0);

    Tensor cls_features({1, D});
    std::memcpy(cls_features.data<float>(), h.data<float>(), D * sizeof(float));

    SceneGraph graph;
    graph.objects = detector.forward(h);
    Tensor global_feat2({1, D});
    Tensor cls_2d = cls_features.reshape({1, D});
    math::gemm(1.0f, cls_2d, global_pool, 0.0f, global_feat2);
    graph.global_features = global_feat2.reshape({D});
    graph.completed_tasks = {VisualTask::CLASSIFY, VisualTask::DETECT,
                             VisualTask::SCENE_GRAPH, VisualTask::CAPTION};
    return graph;
}

Tensor VisionEncoder::recognize_action(const Tensor& video_frames) {
    SceneGraph graph = understand_video(video_frames);
    return graph.global_features;
}

Tensor VisionEncoder::track_objects(const Tensor& video_frames) {
    SceneGraph graph = understand_video(video_frames);
    return graph.objects;
}

SceneGraph VisionEncoder::perceive(const Tensor& visual_input, bool is_video) {
    if (is_video) return understand_video(visual_input);
    return understand_image(visual_input);
}

} // namespace multimodal
} // namespace quant
