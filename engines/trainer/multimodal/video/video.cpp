#include "video.h"
#include "quant/math.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace quant {
namespace multimodal {

// TemporalPositionalEncoding
TemporalPositionalEncoding::TemporalPositionalEncoding(int64_t max_len, int64_t dim)
    : max_len(max_len), dim(dim) {
    pe = Tensor::zeros(Shape{max_len, dim});
    float* pd = pe.data<float>();
    for (int64_t pos = 0; pos < max_len; pos++) {
        for (int64_t i = 0; i < dim / 2; i++) {
            float inv_freq = 1.0f / std::pow(10000.0f, (float)(2 * i) / (float)dim);
            float val = (float)pos * inv_freq;
            pd[pos * dim + i] = std::sin(val);
            pd[pos * dim + dim / 2 + i] = std::cos(val);
        }
    }
}

Tensor TemporalPositionalEncoding::forward(int64_t seq_len) const {
    if (seq_len <= max_len)
        return pe.slice(0, 0, seq_len).clone();
    return pe.clone();
}

TemporalTransformer::TemporalTransformer(const VideoConfig& cfg)
    : temp_pos_enc(cfg.max_frames, cfg.hidden_size),
      final_norm(cfg.hidden_size, cfg.norm_eps),
      hidden_size(cfg.hidden_size),
      max_frames(cfg.max_frames) {
    blocks.reserve(cfg.num_layers);
    for (int64_t i = 0; i < cfg.num_layers; i++)
        blocks.emplace_back(
            TransformerConfig{0, cfg.hidden_size, 1,
                              cfg.num_heads, cfg.head_dim, cfg.ffn_hidden,
                              cfg.norm_eps, 10000.0f, cfg.max_frames,
                              Activation::GELU, cfg.num_heads, false});
}

Tensor TemporalTransformer::forward(const std::vector<Tensor>& frame_features) const {
    int64_t T = (int64_t)frame_features.size();
    if (T == 0) return Tensor::zeros(Shape{1, hidden_size});
    int64_t D = hidden_size;
    Tensor seq = Tensor::zeros(Shape{T, D});
    float* sd = seq.data<float>();
    for (int64_t t = 0; t < T; t++) {
        const float* fd = frame_features[(size_t)t].data<float>();
        int64_t len = frame_features[(size_t)t].numel();
        int64_t copy_dim = std::min(len, D);
        std::memcpy(sd + t * D, fd, (size_t)copy_dim * sizeof(float));
    }
    Tensor pos = temp_pos_enc.forward(T);
    const float* pd = pos.data<float>();
    for (int64_t t = 0; t < T; t++)
        for (int64_t d = 0; d < D; d++)
            sd[t * D + d] += pd[t * D + d];
    Tensor batch = seq.reshape(Shape{1, T, D});
    KVCache cache;
    Tensor dummy_pos = Tensor::arange(T).reshape(Shape{1, T});
    Tensor causal_mask({T, T});
    causal_mask.fill(-1e10f);
    float* md = causal_mask.data<float>();
    for (int64_t i = 0; i < T; i++)
        for (int64_t j = 0; j <= i; j++)
            md[i * T + j] = 0.0f;
    for (const auto& block : blocks)
        batch = block.forward(batch, dummy_pos, causal_mask, cache, 0);
    Tensor out = final_norm.forward(batch);
    return out.reshape(Shape{T, D});
}

// SplitVideoEncoder
SplitVideoEncoder::SplitVideoEncoder(const VideoConfig& cfg)
    : tube_depth(cfg.tube_depth), tube_height(cfg.tube_height),
      tube_width(cfg.tube_width), hidden_size(cfg.hidden_size),
      clip_len(cfg.clip_len),
      patch_proj(cfg.tube_depth * cfg.tube_height * cfg.tube_width, cfg.hidden_size),
      vision_proj(cfg.hidden_size, cfg.hidden_size),
      temporal_transformer(cfg) {}

std::vector<Tensor> SplitVideoEncoder::extract_3d_tubes(
        const Tensor& video_frames) const {
    std::vector<Tensor> tubes;
    int64_t F = video_frames.dim(0);
    int64_t H = video_frames.dim(1);
    int64_t W = video_frames.dim(2);
    int64_t Dp = tube_depth, Hp = tube_height, Wp = tube_width;
    int64_t stride_d = Dp / 2, stride_h = Hp / 2, stride_w = Wp / 2;
    if (stride_d < 1) stride_d = 1;
    if (stride_h < 1) stride_h = 1;
    if (stride_w < 1) stride_w = 1;
    for (int64_t fd = 0; fd + Dp <= F; fd += stride_d)
        for (int64_t y = 0; y + Hp <= H; y += stride_h)
            for (int64_t x = 0; x + Wp <= W; x += stride_w) {
                Tensor tube = Tensor::zeros(Shape{Dp * Hp * Wp});
                float* td = tube.data<float>();
                const float* vd = video_frames.data<float>();
                int64_t idx = 0;
                for (int64_t d = 0; d < Dp; d++)
                    for (int64_t h = 0; h < Hp; h++)
                        for (int64_t w = 0; w < Wp; w++) {
                            int64_t fi = fd + d, hi = y + h, wi = x + w;
                            td[idx++] = vd[fi * H * W + hi * W + wi];
                        }
                tubes.push_back(tube);
            }
    return tubes;
}

Tensor SplitVideoEncoder::encode_clip(const Tensor& clip) const {
    auto tubes = extract_3d_tubes(clip);
    if (tubes.empty())
        return Tensor::zeros(Shape{1, hidden_size});
    int64_t N = (int64_t)tubes.size();
    Tensor tube_mat = Tensor::zeros(Shape{N, tube_depth * tube_height * tube_width});
    float* tm = tube_mat.data<float>();
    for (int64_t i = 0; i < N; i++) {
        const float* td = tubes[(size_t)i].data<float>();
        int64_t len = tubes[(size_t)i].numel();
        std::memcpy(tm + i * (tube_depth * tube_height * tube_width),
                    td, (size_t)len * sizeof(float));
    }
    Tensor projected = patch_proj.forward(tube_mat);
    Tensor encoded = vision_proj.forward(projected);
    return encoded;
}

Tensor SplitVideoEncoder::encode(const Tensor& video_frames) const {
    if (video_frames.numel() == 0)
        return Tensor::zeros(Shape{1, hidden_size});
    int64_t F = video_frames.dim(0);
    std::vector<Tensor> clip_features;
    for (int64_t start = 0; start < F; start += clip_len) {
        int64_t end = std::min(start + clip_len, F);
        Tensor clip = video_frames.slice(0, start, end);
        Tensor feat = encode_clip(clip);
        clip_features.push_back(feat);
    }
    Tensor temporal_out = temporal_transformer.forward(clip_features);
    return temporal_out;
}

std::string SplitVideoEncoder::recognize_action(const Tensor& video) const {
    Tensor feat = encode(video);
    if (feat.numel() == 0) return "unknown";
    int64_t T = feat.dim(0);
    int64_t D = feat.dim(1);
    if (T <= 0 || D <= 0) return "unknown";
    const float* fd = feat.data<float>();
    float max_act = -1e30f;
    int64_t best_t = 0;
    for (int64_t t = 0; t < T; t++) {
        float sum = 0;
        for (int64_t d = 0; d < D; d++)
            sum += fd[t * D + d];
        float mean = sum / (float)D;
        if (mean > max_act) { max_act = mean; best_t = t; }
    }
    std::string actions[] = {"running", "walking", "jumping", "sitting", "unknown"};
    int64_t idx = best_t % 4;
    return actions[idx];
}

std::vector<std::pair<int64_t, int64_t>> SplitVideoEncoder::track_objects(
        const Tensor& video_frames) const {
    std::vector<std::pair<int64_t, int64_t>> positions;
    int64_t F = video_frames.dim(0);
    int64_t H = video_frames.dim(1);
    int64_t W = video_frames.dim(2);
    int64_t patch_h = 16, patch_w = 16;
    for (int64_t f = 0; f < F; f++) {
        Tensor frame = video_frames.slice(0, f, f + 1).reshape(Shape{H, W});
        float max_val = 0;
        int64_t best_y = H / 2, best_x = W / 2;
        for (int64_t y = 0; y + patch_h <= H; y += patch_h / 2)
            for (int64_t x = 0; x + patch_w <= W; x += patch_w / 2) {
                float sum = 0;
                const float* fd = frame.data<float>();
                for (int64_t py = 0; py < patch_h; py++)
                    for (int64_t px = 0; px < patch_w; px++)
                        sum += fd[(y + py) * W + (x + px)];
                float mean = sum / (float)(patch_h * patch_w);
                if (mean > max_val) { max_val = mean; best_y = y; best_x = x; }
            }
        positions.emplace_back(best_x, best_y);
    }
    return positions;
}

// ============================================================================
// conv3d_forward
// ============================================================================

Tensor conv3d_forward(const Tensor& input, const Tensor& weight, const Tensor& bias,
                       int64_t stride_d, int64_t stride_h, int64_t stride_w) {
    int64_t B = input.dim(0);
    int64_t C_in = input.dim(1);
    int64_t D = input.dim(2);
    int64_t H = input.dim(3);
    int64_t W = input.dim(4);

    int64_t C_out = weight.dim(0);
    int64_t kd = weight.dim(2);
    int64_t kh = weight.dim(3);
    int64_t kw = weight.dim(4);

    int64_t out_d = (D - kd) / stride_d + 1;
    int64_t out_h = (H - kh) / stride_h + 1;
    int64_t out_w = (W - kw) / stride_w + 1;

    Tensor output({B, C_out, out_d, out_h, out_w});
    output.zero_();
    const float* inp = input.data<float>();
    const float* w = weight.data<float>();
    const float* b = bias.data<float>();
    float* out = output.data<float>();

    for (int64_t n = 0; n < B; ++n) {
        for (int64_t co = 0; co < C_out; ++co) {
            for (int64_t od = 0; od < out_d; ++od) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        float sum = b[co];
                        for (int64_t ci = 0; ci < C_in; ++ci) {
                            for (int64_t kd_i = 0; kd_i < kd; ++kd_i) {
                                for (int64_t kh_i = 0; kh_i < kh; ++kh_i) {
                                    for (int64_t kw_i = 0; kw_i < kw; ++kw_i) {
                                        int64_t in_idx = ((((n * C_in + ci) * D + od * stride_d + kd_i) * H + oh * stride_h + kh_i) * W + ow * stride_w + kw_i);
                                        int64_t w_idx = ((((co * C_in + ci) * kd + kd_i) * kh + kh_i) * kw + kw_i);
                                        sum += inp[in_idx] * w[w_idx];
                                    }
                                }
                            }
                        }
                        int64_t out_idx = ((((n * C_out + co) * out_d + od) * out_h + oh) * out_w + ow);
                        out[out_idx] = sum;
                    }
                }
            }
        }
    }

    return output;
}

// ============================================================================
// max_pool3d
// ============================================================================

Tensor max_pool3d(const Tensor& x, int64_t kernel_d, int64_t kernel_h, int64_t kernel_w,
                  int64_t stride_d, int64_t stride_h, int64_t stride_w) {
    int64_t B = x.dim(0);
    int64_t C = x.dim(1);
    int64_t D = x.dim(2);
    int64_t H = x.dim(3);
    int64_t W = x.dim(4);

    int64_t out_d = (D - kernel_d) / stride_d + 1;
    int64_t out_h = (H - kernel_h) / stride_h + 1;
    int64_t out_w = (W - kernel_w) / stride_w + 1;

    Tensor out({B, C, out_d, out_h, out_w});
    const float* inp = x.data<float>();
    float* o = out.data<float>();

    for (int64_t b = 0; b < B; ++b)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t od = 0; od < out_d; ++od)
                for (int64_t oh = 0; oh < out_h; ++oh)
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        float max_val = -1e30f;
                        for (int64_t kd_i = 0; kd_i < kernel_d; ++kd_i)
                            for (int64_t kh_i = 0; kh_i < kernel_h; ++kh_i)
                                for (int64_t kw_i = 0; kw_i < kernel_w; ++kw_i) {
                                    int64_t src = ((((b * C + c) * D + od * stride_d + kd_i) * H + oh * stride_h + kh_i) * W + ow * stride_w + kw_i);
                                    if (inp[src] > max_val)
                                        max_val = inp[src];
                                }
                        int64_t dst = ((((b * C + c) * out_d + od) * out_h + oh) * out_w + ow);
                        o[dst] = max_val;
                    }

    return out;
}

// ============================================================================
// temporal_pooling
// ============================================================================

Tensor temporal_pooling(const Tensor& features, const std::string& mode) {
    int64_t B = features.dim(0);
    int64_t T = features.dim(1);
    int64_t D = features.dim(2);

    Tensor pooled({B, D});
    const float* feat = features.data<float>();
    float* p = pooled.data<float>();

    if (mode == "mean") {
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                float sum = 0.0f;
                for (int64_t t = 0; t < T; ++t)
                    sum += feat[(b * T + t) * D + d];
                p[b * D + d] = sum / (float)T;
            }
        }
    } else if (mode == "max") {
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                float max_val = -1e30f;
                for (int64_t t = 0; t < T; ++t) {
                    float v = feat[(b * T + t) * D + d];
                    if (v > max_val) max_val = v;
                }
                p[b * D + d] = max_val;
            }
        }
    } else if (mode == "attention") {
        Tensor attn_weights({B, T});
        float* aw = attn_weights.data<float>();
        for (int64_t b = 0; b < B; ++b) {
            float sum_exp = 0.0f;
            for (int64_t t = 0; t < T; ++t) {
                float score = 0.0f;
                for (int64_t d = 0; d < D; ++d)
                    score += feat[(b * T + t) * D + d];
                score /= (float)D;
                aw[b * T + t] = std::exp(score);
                sum_exp += aw[b * T + t];
            }
            for (int64_t t = 0; t < T; ++t)
                aw[b * T + t] /= sum_exp;
        }

        for (int64_t b = 0; b < B; ++b)
            for (int64_t d = 0; d < D; ++d) {
                float weighted = 0.0f;
                for (int64_t t = 0; t < T; ++t)
                    weighted += aw[b * T + t] * feat[(b * T + t) * D + d];
                p[b * D + d] = weighted;
            }
    } else {
        pooled.zero_();
    }

    return pooled;
}

// ============================================================================
// compute_optical_flow
// ============================================================================

Tensor compute_optical_flow(const Tensor& prev_frame, const Tensor& curr_frame,
                            int64_t block_size, int64_t search_radius) {
    int64_t B = prev_frame.dim(0);
    int64_t H = prev_frame.dim(1);
    int64_t W = prev_frame.dim(2);

    Tensor flow({B, H, W, 2});
    flow.zero_();
    float* f = flow.data<float>();
    const float* prev = prev_frame.data<float>();
    const float* curr = curr_frame.data<float>();

    int64_t bs = block_size;
    int64_t sr = search_radius;

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t i = 0; i + bs <= H; i += bs / 2) {
            for (int64_t j = 0; j + bs <= W; j += bs / 2) {
                float best_sad = 1e30f;
                int64_t best_dy = 0, best_dx = 0;

                int64_t iy_start = std::max((int64_t)0, i - sr);
                int64_t iy_end = std::min(H - bs, i + sr);
                int64_t ix_start = std::max((int64_t)0, j - sr);
                int64_t ix_end = std::min(W - bs, j + sr);

                for (int64_t dy = iy_start; dy <= iy_end; ++dy) {
                    for (int64_t dx = ix_start; dx <= ix_end; ++dx) {
                        float sad = 0.0f;
                        for (int64_t bi = 0; bi < bs; ++bi)
                            for (int64_t bj = 0; bj < bs; ++bj) {
                                float pv = prev[(b * H + i + bi) * W + j + bj];
                                float cv = curr[(b * H + dy + bi) * W + dx + bj];
                                sad += std::abs(pv - cv);
                            }
                        if (sad < best_sad) {
                            best_sad = sad;
                            best_dy = dy - i;
                            best_dx = dx - j;
                        }
                    }
                }

                for (int64_t bi = 0; bi < bs && i + bi < H; ++bi)
                    for (int64_t bj = 0; bj < bs && j + bj < W; ++bj) {
                        f[(b * H + i + bi) * W * 2 + (j + bj) * 2 + 0] = (float)best_dx;
                        f[(b * H + i + bi) * W * 2 + (j + bj) * 2 + 1] = (float)best_dy;
                    }
            }
        }
    }

    return flow;
}

// ============================================================================
// InceptionModule
// ============================================================================

InceptionModule::InceptionModule(int64_t in_c, int64_t hidden)
    : in_channels(in_c), hidden_size(hidden)
    , branch1x1(in_c, hidden, 1, 1, 0)
    , branch3x3_reduce(in_c, hidden, 1, 1, 0)
    , branch3x3(hidden, hidden, 3, 1, 1)
    , branch5x5_reduce(in_c, hidden / 4, 1, 1, 0)
    , branch5x5(hidden / 4, hidden, 5, 1, 2)
    , branch_pool(in_c, hidden, 1, 1, 0) {}

Tensor InceptionModule::forward(const Tensor& x) {
    Tensor b1 = branch1x1.forward(x);
    math::relu(b1, b1);

    Tensor b2 = branch3x3_reduce.forward(x);
    math::relu(b2, b2);
    b2 = branch3x3.forward(b2);
    math::relu(b2, b2);

    Tensor b3 = branch5x5_reduce.forward(x);
    math::relu(b3, b3);
    b3 = branch5x5.forward(b3);
    math::relu(b3, b3);

    Tensor pool = max_pool2d(x, 3, 1);
    Tensor b4 = branch_pool.forward(pool);
    math::relu(b4, b4);

    int64_t B = x.dim(0);
    int64_t H = x.dim(2);
    int64_t W = x.dim(3);
    int64_t total_channels = hidden_size * 4;

    Tensor out({B, total_channels, H, W});
    const float* b1d = b1.data<float>();
    const float* b2d = b2.data<float>();
    const float* b3d = b3.data<float>();
    const float* b4d = b4.data<float>();
    float* o = out.data<float>();

    for (int64_t n = 0; n < B; ++n) {
        int64_t off = 0;
        auto copy_planes = [&](const float* src, int64_t ch) {
            for (int64_t c = 0; c < ch; ++c)
                for (int64_t i = 0; i < H; ++i)
                    for (int64_t j = 0; j < W; ++j) {
                        o[((n * total_channels + off + c) * H + i) * W + j] =
                            src[((n * ch + c) * H + i) * W + j];
                    }
            off += ch;
        };
        copy_planes(b1d, hidden_size);
        copy_planes(b2d, hidden_size);
        copy_planes(b3d, hidden_size);
        copy_planes(b4d, hidden_size);
    }

    return out;
}

// ============================================================================
// MotionFeatureExtractor
// ============================================================================

MotionFeatureExtractor::MotionFeatureExtractor(int64_t block_sz, int64_t search_rad, int64_t hidden)
    : block_size(block_sz), search_radius(search_rad), hidden_size(hidden) {}

Tensor MotionFeatureExtractor::extract_motion(const Tensor& video_frames) const {
    int64_t F = video_frames.dim(0);
    int64_t H = video_frames.dim(1);
    int64_t W = video_frames.dim(2);

    if (F < 2) return Tensor::zeros({F, H, W, 2});

    std::vector<Tensor> flow_fields;
    for (int64_t f = 1; f < F; ++f) {
        Tensor prev = video_frames.slice(0, f - 1, f);
        Tensor curr = video_frames.slice(0, f, f + 1);
        Tensor flow = compute_optical_flow(prev, curr, block_size, search_radius);
        flow_fields.push_back(flow);
    }

    return compute_motion_features(flow_fields);
}

Tensor MotionFeatureExtractor::compute_motion_features(const std::vector<Tensor>& flow_fields) const {
    if (flow_fields.empty()) return Tensor::zeros({1, hidden_size});

    int64_t F = (int64_t)flow_fields.size();
    int64_t B = flow_fields[0].dim(0);
    int64_t H = flow_fields[0].dim(1);
    int64_t W = flow_fields[0].dim(2);

    Tensor mag_sum({B, H, W});
    mag_sum.zero_();
    float* ms = mag_sum.data<float>();

    for (int64_t f = 0; f < F; ++f) {
        const float* fd = flow_fields[f].data<float>();
        for (int64_t b = 0; b < B; ++b)
            for (int64_t i = 0; i < H; ++i)
                for (int64_t j = 0; j < W; ++j) {
                    float dx = fd[(b * H + i) * W * 2 + j * 2 + 0];
                    float dy = fd[(b * H + i) * W * 2 + j * 2 + 1];
                    ms[b * H * W + i * W + j] += std::sqrt(dx * dx + dy * dy);
                }
    }

    Tensor mag_flat = mag_sum.reshape({B, H * W});
    Tensor motion_feat({B, hidden_size});
    Tensor weight({H * W, hidden_size});
    float* w = weight.data<float>();
    float scale = std::sqrt(2.0f / (float)(H * W + hidden_size));
    for (int64_t i = 0; i < weight.numel(); ++i)
        w[i] = scale * ((float)std::rand() / (float)RAND_MAX * 2.0f - 1.0f);

    math::gemm(1.0f, mag_flat, weight, 0.0f, motion_feat);

    return motion_feat;
}

// ============================================================================
// SplitVideoEncoder additions
// ============================================================================

Tensor SplitVideoEncoder::encode_with_motion(const Tensor& video_frames) const {
    Tensor appearance = encode(video_frames);

    MotionFeatureExtractor mfe(tube_depth, 4, hidden_size);
    Tensor motion = mfe.extract_motion(video_frames);

    int64_t T = appearance.dim(0);
    int64_t D = hidden_size;

    Tensor combined({T, D});
    const float* a = appearance.data<float>();
    const float* m = motion.data<float>();
    float* c = combined.data<float>();

    for (int64_t t = 0; t < T; ++t)
        for (int64_t d = 0; d < D; ++d)
            c[t * D + d] = a[t * D + d] + m[d % motion.numel()];

    return combined;
}

Tensor SplitVideoEncoder::temporal_aggregate(const Tensor& encoded, const std::string& mode) const {
    int64_t T = encoded.dim(0);
    int64_t D = encoded.dim(1);

    if (T == 0) return Tensor::zeros({1, D});

    if (mode == "mean") {
        Tensor pooled({1, D});
        const float* e = encoded.data<float>();
        float* p = pooled.data<float>();
        for (int64_t d = 0; d < D; ++d) {
            float sum = 0.0f;
            for (int64_t t = 0; t < T; ++t)
                sum += e[t * D + d];
            p[d] = sum / (float)T;
        }
        return pooled;
    } else if (mode == "max") {
        Tensor pooled({1, D});
        const float* e = encoded.data<float>();
        float* p = pooled.data<float>();
        for (int64_t d = 0; d < D; ++d) {
            float max_val = -1e30f;
            for (int64_t t = 0; t < T; ++t) {
                float v = e[t * D + d];
                if (v > max_val) max_val = v;
            }
            p[d] = max_val;
        }
        return pooled;
    } else if (mode == "last") {
        return encoded.slice(0, T - 1, T).reshape({1, D});
    } else {
        return encoded.slice(0, 0, 1).reshape({1, D});
    }
}

} // namespace multimodal
} // namespace quant
