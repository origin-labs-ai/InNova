#include "image.h"
#include "quant/math.h"
#include <cmath>
#include <random>
#include <cstring>
#include <algorithm>

namespace quant {
namespace multimodal {

// ============================================================================
// Helper functions
// ============================================================================

static void compute_cosine_schedule(Tensor* schedule, int64_t T) {
    const float s = 0.008f;
    float* data = schedule->data<float>();
    int64_t stride = schedule->dim(1);
    float alpha_bar_prev = 1.0f;
    for (int64_t t = 0; t < T; ++t) {
        float t_frac = (float)(t + 1) / (float)T;
        float angle = (t_frac + s) / (1.0f + s) * 3.141592653589793f / 2.0f;
        float alpha_bar = std::cos(angle);
        alpha_bar = alpha_bar * alpha_bar;
        if (alpha_bar < 0.001f) alpha_bar = 0.001f;
        if (alpha_bar > 1.0f) alpha_bar = 1.0f;
        float alpha = alpha_bar / alpha_bar_prev;
        float beta = 1.0f - alpha;
        data[t * stride + 0] = alpha_bar;
        data[t * stride + 1] = alpha;
        data[t * stride + 2] = beta;
        alpha_bar_prev = alpha_bar;
    }
}

static void sinusoidal_embed(float* out, int64_t t, int64_t dim) {
    for (int64_t i = 0; i < dim / 2; ++i) {
        float freq = 1.0f / std::pow(10000.0f, (float)(2 * i) / (float)dim);
        out[2 * i] = std::sin((float)t * freq);
        out[2 * i + 1] = std::cos((float)t * freq);
    }
    if (dim % 2 == 1) out[dim - 1] = 0.0f;
}

static void pos_embed(float* out, int64_t pos, int64_t dim) {
    for (int64_t i = 0; i < dim / 2; ++i) {
        float freq = 1.0f / std::pow(10000.0f, (float)(2 * i) / (float)dim);
        out[2 * i] = std::sin((float)pos * freq);
        out[2 * i + 1] = std::cos((float)pos * freq);
    }
    if (dim % 2 == 1) out[dim - 1] = 0.0f;
}

static Tensor randn(const Shape& shape) {
    Tensor t(shape);
    static thread_local std::mt19937 rng(42);
    static thread_local std::normal_distribution<float> dist(0.0f, 1.0f);
    float* data = t.data<float>();
    int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i)
        data[i] = dist(rng);
    return t;
}

static std::pair<int64_t, int64_t> compute_patch_info(int64_t img_size,
                                                       int64_t C) {
    int64_t p_size = std::max(int64_t(4), img_size / 8);
    while (img_size % p_size != 0) --p_size;
    int64_t n_patches = (img_size / p_size) * (img_size / p_size);
    return {p_size, n_patches};
}

// ============================================================================
// ImageEncoder
// ============================================================================

ImageEncoder::ImageEncoder(int64_t img_size, int64_t hidden_size,
                           int64_t num_layers, int64_t num_heads,
                           int64_t latent_dim)
    : img_size(img_size), hidden_size(hidden_size), latent_dim(latent_dim)
{
    int64_t C = 3;
    int64_t p_size;
    std::tie(p_size, std::ignore) = compute_patch_info(img_size, C);
    int64_t patch_dim = C * p_size * p_size;

    conv_in = Tensor({patch_dim, hidden_size});
    conv_in.zero_();

    latent_proj = Tensor({hidden_size, latent_dim});
    latent_proj.zero_();

    TransformerConfig tcfg;
    tcfg.hidden_size = hidden_size;
    tcfg.num_layers = num_layers;
    tcfg.num_heads = num_heads;
    tcfg.head_dim = hidden_size / num_heads;
    tcfg.ffn_hidden_size = 4 * hidden_size;

    blocks.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i)
        blocks.emplace_back(tcfg);
}

Tensor ImageEncoder::encode(const Tensor& image) {
    int64_t B = image.dim(0);
    int64_t C = image.dim(1);
    int64_t H = image.dim(2);
    int64_t W = image.dim(3);
    int64_t patch_dim = conv_in.dim(0);
    int64_t p_size = (int64_t)std::sqrt((double)(patch_dim / C));
    int64_t n_patches_h = H / p_size;
    int64_t n_patches_w = W / p_size;
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
                    for (int64_t pi = 0; pi < p_size; ++pi)
                        for (int64_t pj = 0; pj < p_size; ++pj) {
                            int64_t src = ((b * C + c) * H + i * p_size + pi) * W + j * p_size + pj;
                            int64_t dst = (b * n + p_idx) * patch_dim + (c * p_size + pi) * p_size + pj;
                            pat[dst] = img[src];
                        }
            }

    Tensor patches_flat = patches.reshape({B * n, patch_dim});
    Tensor token_emb_flat({B * n, D});
    math::gemm(1.0f, patches_flat, conv_in, 0.0f, token_emb_flat);
    Tensor token_emb = token_emb_flat.reshape({B, n, D});

    Tensor seq({B, n, D});
    const float* te = token_emb.data<float>();
    float* s = seq.data<float>();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t p = 0; p < n; ++p) {
            std::memcpy(s + (b * n + p) * D, te + (b * n + p) * D, D * sizeof(float));
            float pe_buf[64];
            sinusoidal_embed(pe_buf, p, D);
            for (int64_t d = 0; d < D; ++d)
                s[(b * n + p) * D + d] += pe_buf[d];
        }

    Tensor h = seq;
    Tensor positions = Tensor::arange(n);
    Tensor mask({n * n});
    mask.fill(0.0f);
    KVCache dummy_cache;

    for (auto& block : blocks)
        h = block.forward(h, positions, mask, dummy_cache, 0);

    return h;
}

Tensor ImageEncoder::project_to_latent(const Tensor& features) {
    int64_t B = features.dim(0);
    int64_t n = features.dim(1);
    int64_t D = features.dim(2);

    Tensor pooled({B, D});
    const float* f = features.data<float>();
    float* p = pooled.data<float>();
    std::memset(p, 0, B * D * sizeof(float));

    for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < n; ++i)
            for (int64_t d = 0; d < D; ++d)
                p[b * D + d] += f[(b * n + i) * D + d];

    for (int64_t i = 0; i < B * D; ++i)
        p[i] /= (float)n;

    Tensor latent({B, latent_dim});
    math::gemm(1.0f, pooled, latent_proj, 0.0f, latent);

    return latent;
}

// ============================================================================
// ImageDecoder
// ============================================================================

ImageDecoder::ImageDecoder(int64_t latent_dim, int64_t hidden_size,
                           int64_t num_layers, int64_t num_heads,
                           int64_t img_size)
    : latent_dim(latent_dim), hidden_size(hidden_size), img_size(img_size)
{
    int64_t C = 3;
    int64_t p_size;
    std::tie(p_size, std::ignore) = compute_patch_info(img_size, C);
    int64_t patch_dim = C * p_size * p_size;
    int64_t up_dim = hidden_size * 2;

    latent_in = Tensor({latent_dim, hidden_size});
    latent_in.zero_();

    up_proj = Tensor({hidden_size, up_dim});
    up_proj.zero_();

    conv_out = Tensor({up_dim, patch_dim});
    conv_out.zero_();

    TransformerConfig tcfg;
    tcfg.hidden_size = hidden_size;
    tcfg.num_layers = num_layers;
    tcfg.num_heads = num_heads;
    tcfg.head_dim = hidden_size / num_heads;
    tcfg.ffn_hidden_size = 4 * hidden_size;

    blocks.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i)
        blocks.emplace_back(tcfg);
}

Tensor ImageDecoder::decode(const Tensor& latent) {
    int64_t B = latent.dim(0);
    int64_t D = hidden_size;
    int64_t C = 3;
    int64_t patch_dim = conv_out.dim(1);
    int64_t p_size = (int64_t)std::sqrt((double)(patch_dim / C));
    int64_t n_patches_h = img_size / p_size;
    int64_t n_patches_w = img_size / p_size;
    int64_t n = n_patches_h * n_patches_w;
    int64_t up_dim = up_proj.dim(1);

    Tensor h({B, D});
    math::gemm(1.0f, latent, latent_in, 0.0f, h);

    Tensor seq({B, n, D});
    const float* h_data = h.data<float>();
    float* s = seq.data<float>();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t p = 0; p < n; ++p)
            std::memcpy(s + (b * n + p) * D, h_data + b * D, D * sizeof(float));

    for (int64_t b = 0; b < B; ++b)
        for (int64_t p = 0; p < n; ++p) {
            float pe_buf[64];
            sinusoidal_embed(pe_buf, p, D);
            for (int64_t d = 0; d < D; ++d)
                s[(b * n + p) * D + d] += pe_buf[d];
        }

    Tensor x = seq;
    Tensor positions = Tensor::arange(n);
    Tensor mask({n * n});
    mask.fill(0.0f);
    KVCache dummy_cache;

    for (auto& block : blocks)
        x = block.forward(x, positions, mask, dummy_cache, 0);

    Tensor x_flat = x.reshape({B * n, D});
    Tensor up_flat({B * n, up_dim});
    math::gemm(1.0f, x_flat, up_proj, 0.0f, up_flat);
    Tensor up = up_flat.reshape({B, n, up_dim});

    Tensor up_flat2 = up.reshape({B * n, up_dim});
    Tensor pixels_flat({B * n, patch_dim});
    math::gemm(1.0f, up_flat2, conv_out, 0.0f, pixels_flat);
    Tensor pixels = pixels_flat.reshape({B, n, patch_dim});

    Tensor image({B, C, img_size, img_size});
    image.zero_();
    float* img = image.data<float>();
    const float* pix = pixels.data<float>();

    for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < n_patches_h; ++i)
            for (int64_t j = 0; j < n_patches_w; ++j) {
                int64_t p_idx = i * n_patches_w + j;
                for (int64_t c = 0; c < C; ++c)
                    for (int64_t pi = 0; pi < p_size; ++pi)
                        for (int64_t pj = 0; pj < p_size; ++pj) {
                            int64_t src = (b * n + p_idx) * patch_dim + (c * p_size + pi) * p_size + pj;
                            int64_t dst = ((b * C + c) * img_size + i * p_size + pi) * img_size + j * p_size + pj;
                            img[dst] = pix[src];
                        }
            }

    return image;
}

Tensor ImageDecoder::upscale(const Tensor& features) {
    int64_t B = features.dim(0);
    int64_t n = features.dim(1);
    int64_t D = features.dim(2);
    int64_t up_dim = up_proj.dim(1);

    Tensor features_flat = features.reshape({B * n, D});
    Tensor result_flat({B * n, up_dim});
    math::gemm(1.0f, features_flat, up_proj, 0.0f, result_flat);
    Tensor result = result_flat.reshape({B, n, up_dim});
    return result;
}

// ============================================================================
// ImageGenerator
// ============================================================================

ImageGenerator::ImageGenerator(int64_t img_size, int64_t hidden_size,
                               int64_t num_layers, int64_t num_heads,
                               int64_t latent_dim)
    : encoder(img_size, hidden_size, num_layers, num_heads, latent_dim),
      decoder(latent_dim, hidden_size, num_layers, num_heads, img_size),
      img_size(img_size), hidden_size(hidden_size), latent_dim(latent_dim)
{
    noise_schedule = Tensor({1000, 3});
    compute_cosine_schedule(&noise_schedule, 1000);
    timestep_embed = Tensor({1, hidden_size});
    timestep_embed.zero_();
}

ImageGenResult ImageGenerator::generate(const Tensor& conditioning,
                                        int64_t num_steps) {
    int64_t B = conditioning.dim(0);
    int64_t D = hidden_size;
    int64_t LD = latent_dim;
    int64_t C = 3;
    int64_t patch_dim = encoder.conv_in.dim(0);
    int64_t p_size = (int64_t)std::sqrt((double)(patch_dim / C));
    int64_t n_patches_h = img_size / p_size;
    int64_t n_patches_w = img_size / p_size;
    int64_t n = n_patches_h * n_patches_w;

    Tensor cond_latent;
    if (conditioning.rank() == 4) {
        Tensor features = encoder.encode(conditioning);
        cond_latent = encoder.project_to_latent(features);
    } else {
        cond_latent = conditioning;
    }

    Tensor x_t = randn({B, LD});

    Tensor schedule({num_steps, 3});
    compute_cosine_schedule(&schedule, num_steps);
    const float* sched = schedule.data<float>();
    int64_t s_stride = schedule.dim(1);

    for (int64_t step = num_steps - 1; step >= 0; --step) {
        float alpha_bar = sched[step * s_stride + 0];
        float alpha     = sched[step * s_stride + 1];
        float beta      = sched[step * s_stride + 2];
        float sqrt_alpha = std::sqrt(std::max(1e-8f, alpha));
        float sqrt_1mab = std::sqrt(std::max(1e-8f, 1.0f - alpha_bar));

        Tensor x_hidden({B, D});
        math::gemm(1.0f, x_t, decoder.latent_in, 0.0f, x_hidden);

        Tensor c_hidden({B, D});
        math::gemm(1.0f, cond_latent, decoder.latent_in, 0.0f, c_hidden);

        Tensor t_embed({B, D});
        for (int64_t b = 0; b < B; ++b)
            sinusoidal_embed(t_embed.data<float>() + b * D, step, D);

        Tensor combined({B, D});
        const float* xh = x_hidden.data<float>();
        const float* ch = c_hidden.data<float>();
        const float* te = t_embed.data<float>();
        float* cb = combined.data<float>();
        for (int64_t i = 0; i < B * D; ++i)
            cb[i] = xh[i] + ch[i] + te[i];

        Tensor seq({B, n, D});
        float* seq_data = seq.data<float>();
        for (int64_t b = 0; b < B; ++b)
            for (int64_t p = 0; p < n; ++p)
                std::memcpy(seq_data + (b * n + p) * D, cb + b * D,
                            D * sizeof(float));

        for (int64_t b = 0; b < B; ++b)
            for (int64_t p = 0; p < n; ++p) {
                float pe_buf[64];
                sinusoidal_embed(pe_buf, p, D);
                float* row = seq_data + (b * n + p) * D;
                for (int64_t d = 0; d < D; ++d)
                    row[d] += pe_buf[d];
            }

        Tensor h = seq;
        Tensor positions = Tensor::arange(n);
        Tensor mask({n * n});
        mask.fill(0.0f);
        KVCache dummy_cache;
        for (auto& block : encoder.blocks)
            h = block.forward(h, positions, mask, dummy_cache, 0);

        Tensor pooled({B, D});
        float* pool = pooled.data<float>();
        std::memset(pool, 0, B * D * sizeof(float));
        const float* h_data = h.data<float>();
        for (int64_t b = 0; b < B; ++b)
            for (int64_t p = 0; p < n; ++p)
                for (int64_t d = 0; d < D; ++d)
                    pool[b * D + d] += h_data[(b * n + p) * D + d];
        for (int64_t i = 0; i < B * D; ++i)
            pool[i] /= (float)n;

        Tensor noise_pred({B, LD});
        math::gemm(1.0f, pooled, encoder.latent_proj, 0.0f, noise_pred);

        float noise_scale = beta / sqrt_1mab;
        float inv_sqrt_alpha = 1.0f / sqrt_alpha;
        float* xt = x_t.data<float>();
        const float* np = noise_pred.data<float>();

        if (step > 0) {
            Tensor z = randn({B, LD});
            float sigma = std::sqrt(std::max(1e-8f, beta));
            const float* z_data = z.data<float>();
            for (int64_t i = 0; i < B * LD; ++i)
                xt[i] = inv_sqrt_alpha * (xt[i] - noise_scale * np[i])
                        + sigma * z_data[i];
        } else {
            for (int64_t i = 0; i < B * LD; ++i)
                xt[i] = inv_sqrt_alpha * (xt[i] - noise_scale * np[i]);
        }
    }

    Tensor image = decoder.decode(x_t);

    std::vector<float> scores(B, 0.0f);
    const float* latent_data = x_t.data<float>();
    for (int64_t b = 0; b < B; ++b) {
        float mean = 0.0f;
        for (int64_t i = 0; i < LD; ++i)
            mean += latent_data[b * LD + i];
        mean /= (float)LD;
        float var = 0.0f;
        for (int64_t i = 0; i < LD; ++i) {
            float diff = latent_data[b * LD + i] - mean;
            var += diff * diff;
        }
        var /= (float)LD;
        scores[b] = 1.0f / (1.0f + var);
    }

    return {image, x_t, scores};
}

// ============================================================================
// im2col
// ============================================================================

Tensor im2col(const Tensor& x, int64_t kernel_h, int64_t kernel_w,
              int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w) {
    int64_t B = x.dim(0);
    int64_t C = x.dim(1);
    int64_t H = x.dim(2);
    int64_t W = x.dim(3);

    int64_t H_pad = H + 2 * pad_h;
    int64_t W_pad = W + 2 * pad_w;
    int64_t out_h = (H_pad - kernel_h) / stride_h + 1;
    int64_t out_w = (W_pad - kernel_w) / stride_w + 1;

    std::vector<float> padded(B * C * H_pad * W_pad, 0.0f);
    const float* inp = x.data<float>();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t i = 0; i < H; ++i)
                for (int64_t j = 0; j < W; ++j) {
                    int64_t dst = ((b * C + c) * H_pad + pad_h + i) * W_pad + pad_w + j;
                    int64_t src = ((b * C + c) * H + i) * W + j;
                    padded[dst] = inp[src];
                }

    int64_t N = B * out_h * out_w;
    int64_t K = C * kernel_h * kernel_w;
    Tensor cols({N, K});
    float* col = cols.data<float>();

    int64_t idx = 0;
    for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < out_h; ++i)
            for (int64_t j = 0; j < out_w; ++j)
                for (int64_t c = 0; c < C; ++c)
                    for (int64_t ki = 0; ki < kernel_h; ++ki)
                        for (int64_t kj = 0; kj < kernel_w; ++kj) {
                            int64_t src = ((b * C + c) * H_pad + i * stride_h + ki) * W_pad + j * stride_w + kj;
                            col[idx++] = padded[src];
                        }

    return cols;
}

// ============================================================================
// max_pool2d
// ============================================================================

Tensor max_pool2d(const Tensor& x, int64_t kernel_size, int64_t stride) {
    int64_t B = x.dim(0);
    int64_t C = x.dim(1);
    int64_t H = x.dim(2);
    int64_t W = x.dim(3);

    int64_t out_h = (H - kernel_size) / stride + 1;
    int64_t out_w = (W - kernel_size) / stride + 1;

    Tensor out({B, C, out_h, out_w});
    const float* inp = x.data<float>();
    float* o = out.data<float>();

    for (int64_t b = 0; b < B; ++b)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t i = 0; i < out_h; ++i)
                for (int64_t j = 0; j < out_w; ++j) {
                    float max_val = -1e30f;
                    for (int64_t ki = 0; ki < kernel_size; ++ki)
                        for (int64_t kj = 0; kj < kernel_size; ++kj) {
                            int64_t src = ((b * C + c) * H + i * stride + ki) * W + j * stride + kj;
                            if (inp[src] > max_val)
                                max_val = inp[src];
                        }
                    o[((b * C + c) * out_h + i) * out_w + j] = max_val;
                }

    return out;
}

// ============================================================================
// global_avg_pool2d
// ============================================================================

Tensor global_avg_pool2d(const Tensor& x) {
    int64_t B = x.dim(0);
    int64_t C = x.dim(1);
    int64_t H = x.dim(2);
    int64_t W = x.dim(3);

    Tensor out({B, C});
    const float* inp = x.data<float>();
    float* o = out.data<float>();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int64_t i = 0; i < H; ++i)
                for (int64_t j = 0; j < W; ++j)
                    sum += inp[((b * C + c) * H + i) * W + j];
            o[b * C + c] = sum / (float)(H * W);
        }
    }

    return out;
}

Tensor adaptive_avg_pool2d(const Tensor& x, int64_t out_h, int64_t out_w) {
    int64_t B = x.dim(0);
    int64_t C = x.dim(1);
    int64_t H = x.dim(2);
    int64_t W = x.dim(3);

    Tensor out({B, C, out_h, out_w});
    const float* inp = x.data<float>();
    float* o = out.data<float>();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t i = 0; i < out_h; ++i) {
                for (int64_t j = 0; j < out_w; ++j) {
                    float h_start = (float)i * (float)H / (float)out_h;
                    float h_end = (float)(i + 1) * (float)H / (float)out_h;
                    float w_start = (float)j * (float)W / (float)out_w;
                    float w_end = (float)(j + 1) * (float)W / (float)out_w;

                    int64_t h_s = (int64_t)std::floor(h_start);
                    int64_t h_e = (int64_t)std::ceil(h_end);
                    int64_t w_s = (int64_t)std::floor(w_start);
                    int64_t w_e = (int64_t)std::ceil(w_end);
                    h_e = std::min(h_e, H);
                    w_e = std::min(w_e, W);

                    float sum = 0.0f;
                    int64_t count = 0;
                    for (int64_t hi = h_s; hi < h_e; ++hi)
                        for (int64_t wj = w_s; wj < w_e; ++wj) {
                            sum += inp[((b * C + c) * H + hi) * W + wj];
                            count++;
                        }
                    o[((b * C + c) * out_h + i) * out_w + j] = (count > 0) ? sum / (float)count : 0.0f;
                }
            }
        }
    }

    return out;
}

Tensor conv2d_batch_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                          const Tensor& running_mean, const Tensor& running_var,
                          float eps) {
    Tensor conv_out = [&]() {
        int64_t B = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
        int64_t out_c = weight.dim(0);
        int64_t ksz = (int64_t)std::sqrt((double)(weight.dim(1) / C));
        int64_t pad = ksz / 2;
        Tensor cols = im2col(x, ksz, ksz, 1, 1, pad, pad);
        Tensor gemm_out({cols.dim(0), out_c});
        math::gemm(1.0f, cols, weight, 0.0f, gemm_out);
        return gemm_out.reshape({B, H, W, out_c}).transpose(1, 3).transpose(2, 3);
    }();

    int64_t B = conv_out.dim(0), C = conv_out.dim(1), H = conv_out.dim(2), W = conv_out.dim(3);
    Tensor result({B, C, H, W});
    const float* co = conv_out.data<float>();
    const float* rm = running_mean.data<float>();
    const float* rv = running_var.data<float>();
    const float* b = bias.data<float>();
    float* r = result.data<float>();

    for (int64_t n = 0; n < B; ++n)
        for (int64_t c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(rv[c] + eps);
            for (int64_t i = 0; i < H; ++i)
                for (int64_t j = 0; j < W; ++j) {
                    int64_t idx = ((n * C + c) * H + i) * W + j;
                    r[idx] = (co[idx] - rm[c]) * inv_std * 1.0f + b[c];
                }
        }

    return result;
}

Tensor ImageEncoder::encode_cnn_frontend(const Tensor& image) {
    int64_t B = image.dim(0);
    int64_t C = image.dim(1);
    int64_t H = image.dim(2);
    int64_t W = image.dim(3);

    ConvBlock stem(C, 64, 7, 2, 3);
    Tensor h = stem.forward(image);
    math::relu(h, h);
    h = max_pool2d(h, 3, 2);

    ConvBlock conv2(64, 128, 3, 1, 1);
    h = conv2.forward(h);
    math::relu(h, h);
    h = max_pool2d(h, 3, 2);

    ConvBlock conv3(128, 256, 3, 1, 1);
    h = conv3.forward(h);
    math::relu(h, h);
    h = max_pool2d(h, 3, 2);

    ConvBlock conv4(256, hidden_size, 3, 1, 1);
    h = conv4.forward(h);
    math::relu(h, h);

    Tensor pooled = global_avg_pool2d(h);

    int64_t n_tokens = pooled.dim(1);
    Tensor seq({B, n_tokens, hidden_size});
    const float* pl = pooled.data<float>();
    float* s = seq.data<float>();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < n_tokens; ++i)
            s[b * n_tokens + i] = pl[b * n_tokens + i];

    return seq;
}

Tensor ImageEncoder::extract_features(const Tensor& image, bool use_cnn_frontend) {
    if (use_cnn_frontend)
        return encode_cnn_frontend(image);
    return encode(image);
}

Tensor ImageEncoder::extract_multi_scale_features(const Tensor& image) {
    int64_t B = image.dim(0);

    ConvBlock stem(3, 64, 7, 2, 3);
    Tensor h = stem.forward(image);
    math::relu(h, h);

    ConvBlock conv2(64, 128, 3, 2, 1);
    Tensor f2 = conv2.forward(h);
    math::relu(f2, f2);

    ConvBlock conv3(128, 256, 3, 2, 1);
    Tensor f3 = conv3.forward(f2);
    math::relu(f3, f3);

    ConvBlock conv4(256, hidden_size, 3, 2, 1);
    Tensor f4 = conv4.forward(f3);
    math::relu(f4, f4);

    Tensor p4 = global_avg_pool2d(f4);
    Tensor p3 = adaptive_avg_pool2d(f3, 1, 1).reshape({B, f3.dim(1)});
    Tensor p2 = adaptive_avg_pool2d(f2, 1, 1).reshape({B, f2.dim(1)});

    int64_t total_dim = p4.dim(1) + p3.dim(1) + p2.dim(1);
    Tensor combined({B, total_dim});
    float* cb = combined.data<float>();
    const float* p4d = p4.data<float>();
    const float* p3d = p3.data<float>();
    const float* p2d = p2.data<float>();

    for (int64_t b = 0; b < B; ++b) {
        std::memcpy(cb + b * total_dim, p4d + b * p4.dim(1), p4.dim(1) * sizeof(float));
        std::memcpy(cb + b * total_dim + p4.dim(1), p3d + b * p3.dim(1), p3.dim(1) * sizeof(float));
        std::memcpy(cb + b * total_dim + p4.dim(1) + p3.dim(1), p2d + b * p2.dim(1), p2.dim(1) * sizeof(float));
    }

    Tensor proj({total_dim, hidden_size});
    float* pw = proj.data<float>();
    float scale = std::sqrt(2.0f / (float)(total_dim + hidden_size));
    for (int64_t i = 0; i < proj.numel(); ++i)
        pw[i] = scale * ((float)std::rand() / (float)RAND_MAX * 2.0f - 1.0f);

    Tensor result({B, hidden_size});
    math::gemm(1.0f, combined, proj, 0.0f, result);

    return result.reshape({B, 1, hidden_size});
}

Tensor ImageDecoder::decode_v2(const Tensor& latent, int64_t scale_factor) {
    int64_t B = latent.dim(0);
    int64_t D = hidden_size;
    int64_t C = 3;

    Tensor h({B, D});
    math::gemm(1.0f, latent, latent_in, 0.0f, h);

    int64_t base_size = img_size / (4 * scale_factor);
    if (base_size < 2) base_size = 2;

    Tensor spatial({B, D, base_size, base_size});
    const float* hd = h.data<float>();
    float* sp = spatial.data<float>();
    int64_t spatial_elems = base_size * base_size;
    for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < spatial_elems; ++i)
            for (int64_t d = 0; d < D; ++d)
                sp[((b * D + d) * base_size * base_size + i)] = hd[b * D + d] / (float)spatial_elems;

    Tensor weight_t({D, D, 3, 3});
    float* wt = weight_t.data<float>();
    float w_scale = std::sqrt(2.0f / (float)(D * 9 + D));
    for (int64_t i = 0; i < weight_t.numel(); ++i)
        wt[i] = w_scale * ((float)std::rand() / (float)RAND_MAX * 2.0f - 1.0f);
    Tensor bias_t({D});
    bias_t.zero_();

    auto conv_block = [&](const Tensor& inp, int64_t in_ch, int64_t out_ch) -> Tensor {
        Tensor w({out_ch, in_ch * 3 * 3});
        float* wd = w.data<float>();
        float ws = std::sqrt(2.0f / (float)(in_ch * 9 + out_ch));
        for (int64_t i = 0; i < w.numel(); ++i)
            wd[i] = ws * ((float)std::rand() / (float)RAND_MAX * 2.0f - 1.0f);
        Tensor b({out_ch});
        b.zero_();

        Tensor cols = im2col(inp, 3, 3, 1, 1, 1, 1);
        Tensor conv_o({cols.dim(0), out_ch});
        math::gemm(1.0f, cols, w, 0.0f, conv_o);
        float* co = conv_o.data<float>();
        const float* bd = b.data<float>();
        for (int64_t i = 0; i < conv_o.dim(0); ++i)
            for (int64_t oc = 0; oc < out_ch; ++oc)
                co[i * out_ch + oc] += bd[oc];

        int64_t oh = inp.dim(2), ow = inp.dim(3);
        return conv_o.reshape({B, oh, ow, out_ch}).transpose(1, 3).transpose(2, 3);
    };

    Tensor up1 = adaptive_avg_pool2d(spatial, base_size * 2, base_size * 2);
    Tensor h1 = conv_block(up1, D, D / 2);
    math::relu(h1, h1);

    Tensor up2 = adaptive_avg_pool2d(h1, base_size * 4, base_size * 4);
    Tensor h2 = conv_block(up2, D / 2, D / 4);
    math::relu(h2, h2);

    Tensor up3 = adaptive_avg_pool2d(h2, img_size, img_size);
    Tensor h3 = conv_block(up3, D / 4, C);
    math::relu(h3, h3);

    return h3;
}

// ============================================================================
// ConvBlock
// ============================================================================

ConvBlock::ConvBlock(int64_t in_c, int64_t out_c, int64_t ksz,
                     int64_t strd, int64_t pad)
    : in_channels(in_c), out_channels(out_c), kernel_size(ksz),
      stride(strd), padding(pad) {
    float scale = std::sqrt(2.0f / (float)(in_c * ksz * ksz + out_c * ksz * ksz));
    conv_weight = Tensor({out_c, in_c * ksz * ksz});
    float* w = conv_weight.data<float>();
    for (int64_t i = 0; i < conv_weight.numel(); ++i)
        w[i] = scale * ((float)std::rand() / (float)RAND_MAX * 2.0f - 1.0f);

    conv_bias = Tensor({out_c});
    conv_bias.zero_();
}

Tensor ConvBlock::forward(const Tensor& x) {
    int64_t B = x.dim(0);
    int64_t C = x.dim(1);
    int64_t H = x.dim(2);
    int64_t W = x.dim(3);

    int64_t ksz = kernel_size;
    int64_t strd = stride;
    int64_t pad = padding;

    Tensor cols = im2col(x, ksz, ksz, strd, strd, pad, pad);
    int64_t N = cols.dim(0);
    int64_t K = cols.dim(1);

    Tensor conv_out({N, out_channels});
    math::gemm(1.0f, cols, conv_weight, 0.0f, conv_out);

    float* co = conv_out.data<float>();
    const float* b = conv_bias.data<float>();
    for (int64_t i = 0; i < N; ++i)
        for (int64_t oc = 0; oc < out_channels; ++oc)
            co[i * out_channels + oc] += b[oc];

    int64_t out_h = (H + 2 * pad - ksz) / strd + 1;
    int64_t out_w = (W + 2 * pad - ksz) / strd + 1;
    Tensor result = conv_out.reshape({B, out_h, out_w, out_channels})
                           .transpose(1, 3).transpose(2, 3);
    int64_t C_out = out_channels;
    int64_t OH = out_h;
    int64_t OW = out_w;

    Tensor final({B, C_out, OH, OW});
    const float* r = result.data<float>();
    float* f = final.data<float>();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t c = 0; c < C_out; ++c)
            for (int64_t i = 0; i < OH; ++i)
                for (int64_t j = 0; j < OW; ++j)
                    f[((b * C_out + c) * OH + i) * OW + j] = r[((b * OH + i) * OW + j) * C_out + c];

    return final;
}

// ============================================================================
// ResidualBlock
// ============================================================================

ResidualBlock::ResidualBlock(int64_t ch, int64_t ksz)
    : conv1(ch, ch, ksz, 1, ksz / 2),
      conv2(ch, ch, ksz, 1, ksz / 2),
      channels(ch) {}

Tensor ResidualBlock::forward(const Tensor& x) {
    Tensor h = conv1.forward(x);
    math::relu(h, h);
    h = conv2.forward(h);
    math::add(h, x, h);
    math::relu(h, h);
    return h;
}

} // namespace multimodal
} // namespace quant
