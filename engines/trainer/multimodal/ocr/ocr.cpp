#include "ocr.h"
#include "quant/math.h"
#include "quant/random.h"
#include "quant/kv_cache.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <queue>

namespace quant {
namespace multimodal {

static constexpr float SIGMA = 0.02f;
static constexpr int64_t BLANK_ID = 0;

// GridPositionalEncoding
GridPositionalEncoding::GridPositionalEncoding(int64_t max_h, int64_t max_w, int64_t dim)
    : max_h(max_h), max_w(max_w), dim(dim) {
    pe = Tensor::zeros(Shape{max_h * max_w, dim});
    float* pd = pe.data<float>();
    int64_t half_d = dim / 2;
    for (int64_t y = 0; y < max_h; y++) {
        for (int64_t x = 0; x < max_w; x++) {
            int64_t idx = y * max_w + x;
            for (int64_t i = 0; i < half_d / 2; i++) {
                float inv_h = 1.0f / std::pow(10000.0f, (float)(2 * i) / (float)dim);
                float inv_w = 1.0f / std::pow(10000.0f, (float)(2 * i + 1) / (float)dim);
                pd[idx * dim + i] = std::sin((float)y * inv_h);
                pd[idx * dim + half_d / 2 + i] = std::cos((float)x * inv_w);
            }
            for (int64_t i = half_d / 2; i < half_d; i++) {
                float inv_h = 1.0f / std::pow(10000.0f, (float)(2 * (i - half_d / 2)) / (float)dim);
                float inv_w = 1.0f / std::pow(10000.0f, (float)(2 * (i - half_d / 2) + 1) / (float)dim);
                pd[idx * dim + half_d + i] = std::cos((float)y * inv_h);
                pd[idx * dim + half_d + half_d / 2 + i] = std::sin((float)x * inv_w);
            }
        }
    }
}

Tensor GridPositionalEncoding::forward(int64_t height, int64_t width) const {
    if (height <= max_h && width <= max_w) {
        int64_t n = height * width;
        Tensor out({n, dim});
        const float* pd = pe.data<float>();
        float* od = out.data<float>();
        for (int64_t y = 0; y < height; y++)
            for (int64_t x = 0; x < width; x++)
                std::memcpy(od + (y * width + x) * dim,
                            pd + (y * max_w + x) * dim,
                            (size_t)dim * sizeof(float));
        return out;
    }
    return pe.slice(0, 0, height * width).clone();
}

Tensor GridPositionalEncoding::forward_1d(int64_t seq_len) const {
    if (seq_len <= max_h * max_w)
        return pe.slice(0, 0, seq_len).clone();
    return pe.clone();
}

// AttentionDecoder
AttentionDecoder::AttentionDecoder(int64_t hidden, int64_t vocab, int64_t num_layers,
                                     int64_t num_heads, int64_t max_len)
    : embedding(vocab, hidden), output_proj(hidden, vocab),
      hidden_size(hidden), vocab_size(vocab), max_len(max_len) {
    TransformerConfig tcfg;
    tcfg.hidden_size = hidden;
    tcfg.num_layers = 1;
    tcfg.num_heads = num_heads;
    tcfg.head_dim = hidden / num_heads;
    tcfg.max_seq_len = max_len;
    decoder_blocks.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; i++)
        decoder_blocks.emplace_back(tcfg);
}

Tensor AttentionDecoder::forward(const Tensor& encoder_out, const Tensor& targets,
                                   const Tensor& target_mask) const {
    int64_t B = targets.dim(0), T = targets.dim(1), D = hidden_size;
    Tensor h = embedding.forward(targets.reshape(Shape{B * T}));
    h = h.reshape(Shape{B, T, D});
    KVCache cache;
    Tensor positions = Tensor::arange(T).reshape(Shape{B, T});
    Tensor causal_mask({T, T});
    causal_mask.fill(-1e10f);
    float* md = causal_mask.data<float>();
    for (int64_t i = 0; i < T; i++)
        for (int64_t j = 0; j <= i; j++)
            md[i * T + j] = 0.0f;
    Tensor h_out = h;
    for (const auto& block : decoder_blocks)
        h_out = block.forward(h_out, positions, causal_mask, cache, 0);
    Tensor flat = h_out.reshape(Shape{B * T, D});
    Tensor logits({B * T, vocab_size});
    const float* w = output_proj.weight.data<float>();
    float* ld = logits.data<float>();
    const float* fd = flat.data<float>();
    for (int64_t i = 0; i < B * T; i++)
        for (int64_t v = 0; v < vocab_size; v++) {
            float dot = 0;
            for (int64_t d = 0; d < D; d++)
                dot += fd[i * D + d] * w[v * D + d];
            ld[i * vocab_size + v] = dot;
        }
    if (output_proj.bias.numel() > 0) {
        const float* bd = output_proj.bias.data<float>();
        for (int64_t i = 0; i < B * T; i++)
            for (int64_t v = 0; v < vocab_size; v++)
                ld[i * vocab_size + v] += bd[v];
    }
    return logits.reshape(Shape{B, T, vocab_size});
}

std::string AttentionDecoder::greedy_decode(const Tensor& encoder_out, const Tensor& pos_encoding,
                                              int bos_id, int eos_id) const {
    int64_t D = hidden_size;
    int64_t S = encoder_out.dim(0);
    std::vector<int> tokens = {bos_id};
    KVCache cache;
    Tensor encoder_context = encoder_out;
    for (int step = 0; step < max_len; step++) {
        int64_t cur_len = (int64_t)tokens.size();
        Tensor input({1, 1});
        input.data<float>()[0] = (float)tokens.back();
        Tensor h = embedding.forward(input.reshape(Shape{1}));
        h = h.reshape(Shape{1, 1, D});
        Tensor positions = Tensor::arange(1).reshape(Shape{1, 1});
        Tensor causal_mask({1, 1});
        causal_mask.fill(0.0f);
        Tensor h_out = h;
        for (const auto& block : decoder_blocks)
            h_out = block.forward(h_out, positions, causal_mask, cache, 0);
        Tensor flat = h_out.reshape(Shape{1, D});
        const float* w = output_proj.weight.data<float>();
        float maxv = -1e30f;
        int next = 0;
        for (int64_t v = 0; v < vocab_size; v++) {
            float dot = 0;
            for (int64_t d = 0; d < D; d++)
                dot += flat.data<float>()[d] * w[v * D + d];
            if (dot > maxv) { maxv = dot; next = (int)v; }
        }
        if (next == eos_id) break;
        tokens.push_back(next);
    }
    std::string result;
    for (size_t i = 1; i < tokens.size(); i++) {
        int t = tokens[i];
        if (t >= 32 && t < 127) result += (char)t;
    }
    return result;
}

std::vector<int> AttentionDecoder::beam_search_decode(
    const Tensor& encoder_out, const Tensor& pos_encoding,
    int bos_id, int eos_id, int beam_width, int max_steps) const {
    struct BeamHyp {
        std::vector<int> tokens;
        float score;
        bool finished;
    };
    auto cmp = [](const BeamHyp& a, const BeamHyp& b) { return a.score < b.score; };
    std::priority_queue<BeamHyp, std::vector<BeamHyp>, decltype(cmp)> beam(cmp);
    beam.push({{bos_id}, 0.0f, false});
    for (int step = 0; step < max_steps; step++) {
        std::vector<BeamHyp> candidates;
        while (!beam.empty()) {
            candidates.push_back(beam.top());
            beam.pop();
        }
        if (candidates.empty()) break;
        bool all_finished = true;
        for (const auto& c : candidates)
            if (!c.finished) all_finished = false;
        if (all_finished) {
            for (const auto& c : candidates) beam.push(c);
            break;
        }
        for (const auto& hyp : candidates) {
            if (hyp.finished) { beam.push(hyp); continue; }
            int64_t cur_len = (int64_t)hyp.tokens.size();
            Tensor input({1, 1});
            input.data<float>()[0] = (float)hyp.tokens.back();
            Tensor h = embedding.forward(input.reshape(Shape{1}));
            h = h.reshape(Shape{1, 1, hidden_size});
            Tensor positions = Tensor::arange(1).reshape(Shape{1, 1});
            Tensor causal_mask({1, 1});
            causal_mask.fill(0.0f);
            KVCache cache;
            Tensor h_out = h;
            for (const auto& block : decoder_blocks)
                h_out = block.forward(h_out, positions, causal_mask, cache, 0);
            Tensor flat = h_out.reshape(Shape{1, hidden_size});
            const float* w = output_proj.weight.data<float>();
            const float* fd = flat.data<float>();
            std::vector<std::pair<float, int>> scored;
            for (int64_t v = 0; v < vocab_size; v++) {
                float dot = 0;
                for (int64_t d = 0; d < hidden_size; d++)
                    dot += fd[d] * w[v * hidden_size + d];
                scored.push_back({dot, (int)v});
            }
            std::partial_sort(scored.begin(), scored.begin() + std::min(beam_width, (int)scored.size()),
                              scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (int k = 0; k < beam_width && k < (int)scored.size(); k++) {
                std::vector<int> new_tokens = hyp.tokens;
                new_tokens.push_back(scored[k].second);
                float new_score = hyp.score + scored[k].first;
                bool fin = (scored[k].second == eos_id);
                beam.push({new_tokens, new_score, fin});
            }
        }
        if ((int64_t)beam.size() > beam_width * 2) {
            std::vector<BeamHyp> to_prune;
            while (!beam.empty()) { to_prune.push_back(beam.top()); beam.pop(); }
            std::partial_sort(to_prune.begin(), to_prune.begin() + beam_width, to_prune.end(), cmp);
            for (int i = 0; i < beam_width && i < (int)to_prune.size(); i++)
                beam.push(to_prune[i]);
        }
    }
    std::vector<int> best;
    float best_score = -1e30f;
    while (!beam.empty()) {
        if (beam.top().score > best_score) {
            best_score = beam.top().score;
            best = beam.top().tokens;
        }
        beam.pop();
    }
    return best;
}

// TransformerOCR
TransformerOCR::TransformerOCR(const OCRConfig& cfg)
    : pos_enc(cfg.max_width / 4, cfg.max_width / 4, cfg.hidden_size),
      output_proj(cfg.hidden_size, cfg.num_classes),
      hidden_size(cfg.hidden_size), num_classes(cfg.num_classes),
      patch_size(4) {
    int64_t conv_in = cfg.in_channels;
    int64_t conv_out = cfg.hidden_size / 8;
    input_conv = Tensor::zeros(Shape{conv_out, conv_in, 3, 3});
    conv_bias = Tensor::zeros(Shape{conv_out});
    RNG rng(42);
    float* wd = input_conv.data<float>();
    for (int64_t i = 0; i < input_conv.numel(); i++)
        wd[i] = rng.normal() * 0.02f;
    TransformerConfig tcfg;
    tcfg.hidden_size = cfg.hidden_size;
    tcfg.num_layers = 1;
    tcfg.num_heads = cfg.num_heads;
    tcfg.head_dim = cfg.hidden_size / cfg.num_heads;
    tcfg.max_seq_len = cfg.max_seq_len;
    encoder_blocks.reserve(cfg.num_transformer_layers);
    for (int64_t i = 0; i < cfg.num_transformer_layers; i++)
        encoder_blocks.emplace_back(tcfg);
}

Tensor TransformerOCR::forward(const Tensor& image) const {
    int64_t B = image.dim(0);
    int64_t C = image.dim(1);
    int64_t H = image.dim(2);
    int64_t W = image.dim(3);
    int64_t D = hidden_size;
    int64_t ps = patch_size;
    int64_t n_h = H / ps;
    int64_t n_w = W / ps;
    int64_t N = n_h * n_w;
    int64_t patch_dim = C * ps * ps;
    Tensor patches({B, N, patch_dim});
    const float* img = image.data<float>();
    float* pat = patches.data<float>();
    for (int64_t b = 0; b < B; b++)
        for (int64_t i = 0; i < n_h; i++)
            for (int64_t j = 0; j < n_w; j++) {
                int64_t p_idx = i * n_w + j;
                for (int64_t c = 0; c < C; c++)
                    for (int64_t pi = 0; pi < ps; pi++)
                        for (int64_t pj = 0; pj < ps; pj++) {
                            int64_t src = ((b * C + c) * H + i * ps + pi) * W + j * ps + pj;
                            int64_t dst = (b * N + p_idx) * patch_dim + (c * ps + pi) * ps + pj;
                            pat[dst] = img[src];
                        }
            }
    Tensor flat_patches = patches.reshape({B * N, patch_dim});
    int64_t conv_out = input_conv.dim(0);
    Tensor conv_weight_2d = input_conv.reshape(Shape{conv_out, patch_dim});
    Tensor proj({B * N, D});
    math::gemm(1.0f, flat_patches, conv_weight_2d.transpose(0, 1), 0.0f, proj);
    float* pd = proj.data<float>();
    const float* bd = conv_bias.data<float>();
    for (int64_t i = 0; i < B * N; i++)
        for (int64_t d = 0; d < D; d++)
            pd[i * D + d] += bd[d % conv_out];
    Tensor h = proj.reshape(Shape{B, N, D});
    Tensor pe = pos_enc.forward(n_h, n_w);
    const float* pe_data = pe.data<float>();
    float* hd = h.data<float>();
    for (int64_t b = 0; b < B; b++)
        for (int64_t n = 0; n < N; n++)
            for (int64_t d = 0; d < D; d++)
                hd[b * N * D + n * D + d] += pe_data[n * D + d];
    Tensor positions = Tensor::arange(N).reshape(Shape{B, N});
    Tensor mask({N, N});
    mask.fill(0.0f);
    KVCache cache;
    for (const auto& block : encoder_blocks)
        h = block.forward(h, positions, mask, cache, 0);
    Tensor flat = h.reshape(Shape{B * N, D});
    Tensor logits({B * N, num_classes});
    math::gemm(1.0f, flat, output_proj.weight, 0.0f, logits);
    float* lgd = logits.data<float>();
    if (output_proj.bias.numel() > 0) {
        const float* ob = output_proj.bias.data<float>();
        for (int64_t i = 0; i < B * N; i++)
            for (int64_t v = 0; v < num_classes; v++)
                lgd[i * num_classes + v] += ob[v];
    }
    return logits.reshape(Shape{B, N, num_classes});
}

std::string TransformerOCR::recognize(const Tensor& image) const {
    if (image.numel() == 0) return "";
    Tensor batch = image.rank() == 3
        ? image.reshape(Shape{1, image.dim(0), image.dim(1), image.dim(2)})
        : image;
    Tensor logits = forward(batch);
    int64_t T = logits.dim(1);
    int64_t V = logits.dim(2);
    const float* ld = logits.data<float>();
    std::string result;
    int prev = -1;
    for (int64_t t = 0; t < T; t++) {
        int max_c = 0;
        float maxv = ld[t * V];
        for (int64_t v = 1; v < V; v++)
            if (ld[t * V + v] > maxv) { maxv = ld[t * V + v]; max_c = (int)v; }
        if (max_c == 0) { prev = -1; continue; }
        if (max_c != prev) {
            if (max_c < 128) result += (char)max_c;
        }
        prev = max_c;
    }
    return result;
}

std::vector<std::pair<int64_t, int64_t>> TransformerOCR::detect_text_regions(const Tensor& image) const {
    std::vector<std::pair<int64_t, int64_t>> regions;
    int64_t H = image.dim(0);
    int64_t W = image.dim(1);
    int64_t win_h = 32, win_w = 128;
    int64_t stride = 16;
    for (int64_t y = 0; y + win_h <= H; y += stride)
        for (int64_t x = 0; x + win_w <= W; x += stride) {
            Tensor crop = image.slice(0, y, y + win_h).slice(1, x, x + win_w);
            Tensor batch_crop = crop.reshape(Shape{1, 1, win_h, win_w});
            Tensor logits = forward(batch_crop);
            const float* ld = logits.data<float>();
            float max_act = 0;
            for (int64_t i = 0; i < logits.numel(); i++)
                if (ld[i] > max_act) max_act = ld[i];
            if (max_act > 0.5f)
                regions.emplace_back(x, y);
        }
    return regions;
}

// CRNN
void CRNN::init_weight(Tensor& t, float scale) const {
    float* d = t.data<float>();
    for (int64_t i = 0; i < t.numel(); i++)
        d[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
}

CRNN::CRNN(const OCRConfig& cfg) {
    int64_t C1 = cfg.conv_channels[0];
    int64_t C2 = cfg.conv_channels[1];
    int64_t C3 = cfg.conv_channels[2];
    lstm_hidden = cfg.lstm_hidden;
    num_classes = cfg.num_classes;
    conv1_w = Tensor::zeros(Shape{C1, cfg.in_channels, 3, 3});
    conv1_b = Tensor::zeros(Shape{C1});
    conv2_w = Tensor::zeros(Shape{C2, C1, 3, 3});
    conv2_b = Tensor::zeros(Shape{C2});
    conv3_w = Tensor::zeros(Shape{C3, C2, 3, 3});
    conv3_b = Tensor::zeros(Shape{C3});
    init_weight(conv1_w, SIGMA);
    init_weight(conv2_w, SIGMA);
    init_weight(conv3_w, SIGMA);
    int64_t lstm_in = C3;
    lstm_w_i = Tensor::zeros(Shape{lstm_hidden, lstm_in});
    lstm_u_i = Tensor::zeros(Shape{lstm_hidden, lstm_hidden});
    lstm_b_i = Tensor::zeros(Shape{lstm_hidden});
    lstm_w_h = Tensor::zeros(Shape{lstm_hidden, lstm_in});
    lstm_u_h = Tensor::zeros(Shape{lstm_hidden, lstm_hidden});
    lstm_b_h = Tensor::zeros(Shape{lstm_hidden});
    init_weight(lstm_w_i, SIGMA);
    init_weight(lstm_u_i, SIGMA);
    init_weight(lstm_w_h, SIGMA);
    init_weight(lstm_u_h, SIGMA);
    ctc_w = Tensor::zeros(Shape{num_classes, lstm_hidden * 2});
    ctc_b = Tensor::zeros(Shape{num_classes});
    init_weight(ctc_w, SIGMA);
}

Tensor CRNN::conv2d_matmul(const Tensor& input, const Tensor& weight,
                           const Tensor& bias, int stride) const {
    int64_t B = input.dim(0);
    int64_t C_in = input.dim(1);
    int64_t H = input.dim(2);
    int64_t W = input.dim(3);
    int64_t C_out = weight.dim(0);
    int64_t KH = weight.dim(2);
    int64_t KW = weight.dim(3);
    int64_t OH = (H - KH) / stride + 1;
    int64_t OW = (W - KW) / stride + 1;
    int64_t patch_size = C_in * KH * KW;
    Tensor patches = Tensor::zeros(Shape{B * OH * OW, patch_size});
    float* pd = patches.data<float>();
    int64_t idx = 0;
    for (int64_t b = 0; b < B; b++)
        for (int64_t oh = 0; oh < OH; oh++)
            for (int64_t ow = 0; ow < OW; ow++)
                for (int64_t c = 0; c < C_in; c++)
                    for (int64_t kh = 0; kh < KH; kh++)
                        for (int64_t kw = 0; kw < KW; kw++) {
                            int64_t ih = oh * stride + kh;
                            int64_t iw = ow * stride + kw;
                            pd[idx++] = input.data<float>()[
                                b * C_in * H * W + c * H * W + ih * W + iw];
                        }
    Tensor w2d = weight.reshape(Shape{C_out, patch_size});
    Tensor out({B * OH * OW, C_out});
    math::gemm(1.0f, patches, w2d.transpose(0, 1), 0.0f, out);
    float* od = out.data<float>();
    const float* bd = bias.data<float>();
    for (int64_t i = 0; i < B * OH * OW; i++)
        for (int64_t c = 0; c < C_out; c++)
            od[i * C_out + c] += bd[c];
    return out.reshape(Shape{B, C_out, OH, OW});
}

Tensor CRNN::max_pool_stride(const Tensor& input, int window, int stride) const {
    int64_t B = input.dim(0);
    int64_t C = input.dim(1);
    int64_t H = input.dim(2);
    int64_t W = input.dim(3);
    int64_t OH = (H - window) / stride + 1;
    int64_t OW = (W - window) / stride + 1;
    Tensor out = Tensor::zeros(Shape{B, C, OH, OW});
    const float* id = input.data<float>();
    float* od = out.data<float>();
    for (int64_t b = 0; b < B; b++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t oh = 0; oh < OH; oh++)
                for (int64_t ow = 0; ow < OW; ow++) {
                    float maxv = -1e30f;
                    for (int64_t kh = 0; kh < window; kh++)
                        for (int64_t kw = 0; kw < window; kw++) {
                            int64_t ih = oh * stride + kh;
                            int64_t iw = ow * stride + kw;
                            float v = id[b * C * H * W + c * H * W + ih * W + iw];
                            if (v > maxv) maxv = v;
                        }
                    od[b * C * OH * OW + c * OH * OW + oh * OW + ow] = maxv;
                }
    return out;
}

Tensor CRNN::bilstm_step(const Tensor& x_fwd, const Tensor& x_bwd,
                         const Tensor& h_fwd, const Tensor& h_bwd) const {
    int64_t D = lstm_hidden;
    Tensor h_fwd_next({D});
    Tensor h_bwd_next({D});
    Tensor i_gate({D});
    Tensor h_gate({D});
    const float* xf = x_fwd.data<float>();
    const float* xb = x_bwd.data<float>();
    const float* hf = h_fwd.data<float>();
    const float* hb = h_bwd.data<float>();
    float* ifd = i_gate.data<float>();
    float* hfd = h_gate.data<float>();
    const float* wfi = lstm_w_i.data<float>();
    const float* ufi = lstm_u_i.data<float>();
    const float* bfi = lstm_b_i.data<float>();
    const float* wfh = lstm_w_h.data<float>();
    const float* ufh = lstm_u_h.data<float>();
    const float* bfh = lstm_b_h.data<float>();
    for (int64_t d = 0; d < D; d++) {
        float sum_f = bfi[d];
        for (int64_t k = 0; k < x_fwd.dim(0); k++)
            sum_f += wfi[d * x_fwd.dim(0) + k] * xf[k];
        for (int64_t k = 0; k < D; k++)
            sum_f += ufi[d * D + k] * hf[k];
        ifd[d] = std::tanh(sum_f);
    }
    for (int64_t d = 0; d < D; d++) {
        float sum_b = bfh[d];
        for (int64_t k = 0; k < x_bwd.dim(0); k++)
            sum_b += wfh[d * x_bwd.dim(0) + k] * xb[k];
        for (int64_t k = 0; k < D; k++)
            sum_b += ufh[d * D + k] * hb[k];
        hfd[d] = std::tanh(sum_b);
    }
    for (int64_t d = 0; d < D; d++) {
        h_fwd_next.data<float>()[d] = ifd[d];
        h_bwd_next.data<float>()[d] = hfd[d];
    }
    Tensor combined({D * 2});
    float* cd = combined.data<float>();
    for (int64_t d = 0; d < D; d++) cd[d] = ifd[d];
    for (int64_t d = 0; d < D; d++) cd[D + d] = hfd[d];
    return combined;
}

std::string CRNN::ctc_greedy_decode(const Tensor& logits) const {
    int64_t T = logits.dim(0);
    int64_t V = logits.dim(1);
    const float* ld = logits.data<float>();
    std::string result;
    int prev = -1;
    for (int64_t t = 0; t < T; t++) {
        int max_c = 0;
        float maxv = ld[t * V];
        for (int64_t v = 1; v < V; v++) {
            if (ld[t * V + v] > maxv) {
                maxv = ld[t * V + v];
                max_c = (int)v;
            }
        }
        if (max_c == BLANK_ID) { prev = -1; continue; }
        if (max_c != prev) {
            if (max_c < 128) result += (char)max_c;
        }
        prev = max_c;
    }
    return result;
}

std::string CRNN::ctc_beam_search_decode(const Tensor& logits, int beam_width) const {
    int64_t T = logits.dim(0);
    int64_t V = logits.dim(1);
    struct CTCBeam {
        std::string text;
        float score;
        int last_token;
    };
    auto cmp = [](const CTCBeam& a, const CTCBeam& b) { return a.score < b.score; };
    std::vector<CTCBeam> beam = {{"", 0.0f, -1}};
    for (int64_t t = 0; t < T; t++) {
        const float* ld = logits.data<float>() + t * V;
        std::vector<std::pair<float, int>> top_k;
        for (int64_t v = 0; v < V; v++)
            top_k.push_back({ld[v], (int)v});
        std::partial_sort(top_k.begin(), top_k.begin() + std::min(beam_width, (int)top_k.size()),
                          top_k.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<CTCBeam> new_beam;
        for (const auto& b : beam) {
            for (int k = 0; k < beam_width && k < (int)top_k.size(); k++) {
                int token = top_k[k].first > -1e10f ? top_k[k].second : 0;
                float logp = std::log(std::max(top_k[k].first, 1e-10f));
                if (token == BLANK_ID) {
                    new_beam.push_back({b.text, b.score + logp, b.last_token});
                } else if (token == b.last_token) {
                    new_beam.push_back({b.text, b.score + logp, token});
                } else {
                    std::string new_text = b.text;
                    if (token < 128) new_text += (char)token;
                    new_beam.push_back({new_text, b.score + logp, token});
                }
            }
        }
        std::partial_sort(new_beam.begin(), new_beam.begin() + std::min(beam_width, (int)new_beam.size()),
                          new_beam.end(), cmp);
        beam.clear();
        int n = std::min(beam_width, (int)new_beam.size());
        for (int i = 0; i < n; i++)
            beam.push_back(new_beam[i]);
    }
    std::string best_text = "";
    float best_score = -1e30f;
    for (const auto& b : beam)
        if (b.score > best_score) { best_score = b.score; best_text = b.text; }
    return best_text;
}

Tensor CRNN::forward(const Tensor& patches) const {
    int64_t B = patches.dim(0);
    Tensor h = conv2d_matmul(patches, conv1_w, conv1_b, 1);
    math::relu(h, h);
    h = max_pool_stride(h, 2, 2);
    h = conv2d_matmul(h, conv2_w, conv2_b, 1);
    math::relu(h, h);
    h = max_pool_stride(h, 2, 2);
    h = conv2d_matmul(h, conv3_w, conv3_b, 1);
    math::relu(h, h);
    int64_t C = h.dim(1);
    int64_t Ht = h.dim(2);
    int64_t Wt = h.dim(3);
    int64_t T = Wt;
    Tensor seq = h.reshape(Shape{B, C * Ht, T});
    Tensor h_fwd = Tensor::zeros(Shape{lstm_hidden});
    Tensor h_bwd = Tensor::zeros(Shape{lstm_hidden});
    Tensor logits({B, T, num_classes});
    for (int64_t b = 0; b < B; b++) {
        h_fwd.zero_();
        h_bwd.zero_();
        for (int64_t t = 0; t < T; t++) {
            Tensor x_fwd = seq.slice(2, t, t + 1).reshape(Shape{C * Ht});
            int64_t t_bwd = T - 1 - t;
            Tensor x_bwd = seq.slice(2, t_bwd, t_bwd + 1).reshape(Shape{C * Ht});
            Tensor combined = bilstm_step(x_fwd, x_bwd, h_fwd, h_bwd);
            for (int64_t d = 0; d < lstm_hidden; d++)
                h_fwd.data<float>()[d] = combined.data<float>()[d];
            for (int64_t d = 0; d < lstm_hidden; d++)
                h_bwd.data<float>()[d] = combined.data<float>()[lstm_hidden + d];
            Tensor ctc_logits({num_classes});
            math::gemv(1.0f, ctc_w, combined, 0.0f, ctc_logits);
            const float* cb = ctc_b.data<float>();
            for (int64_t v = 0; v < num_classes; v++)
                ctc_logits.data<float>()[v] += cb[v];
            for (int64_t v = 0; v < num_classes; v++)
                logits.data<float>()[b * T * num_classes + t * num_classes + v] =
                    ctc_logits.data<float>()[v];
        }
    }
    return logits;
}

std::string CRNN::recognize(const Tensor& text_image) const {
    if (text_image.numel() == 0) return "";
    Tensor batch = text_image.rank() == 3 ? text_image.reshape(
        Shape{1, text_image.dim(0), text_image.dim(1), text_image.dim(2)})
        : text_image;
    Tensor logits = forward(batch);
    return ctc_greedy_decode(logits.reshape(
        Shape{logits.dim(1), logits.dim(2)}));
}

std::vector<std::pair<int64_t, int64_t>> CRNN::detect_text_region(
        const Tensor& image) const {
    std::vector<std::pair<int64_t, int64_t>> regions;
    int64_t H = image.dim(0);
    int64_t W = image.dim(1);
    int64_t win_h = 32, win_w = 128;
    int64_t stride = 16;
    for (int64_t y = 0; y + win_h <= H; y += stride)
        for (int64_t x = 0; x + win_w <= W; x += stride) {
            Tensor crop = image.slice(0, y, y + win_h).slice(1, x, x + win_w);
            Tensor batch_crop = crop.reshape(Shape{1, 1, win_h, win_w});
            Tensor logits = forward(batch_crop);
            const float* ld = logits.data<float>();
            float max_act = 0;
            for (int64_t i = 0; i < logits.numel(); i++)
                if (ld[i] > max_act) max_act = ld[i];
            if (max_act > 0.5f)
                regions.emplace_back(x, y);
        }
    return regions;
}

} // namespace multimodal
} // namespace quant
