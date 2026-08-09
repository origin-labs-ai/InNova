#include "audio.h"
#include "quant/math.h"
#include "quant/kv_cache.h"

namespace quant {
namespace multimodal {

AudioEncoder::AudioEncoder(int64_t n_mels, int64_t hidden, int64_t num_layers, int64_t num_heads)
    : hidden_size(hidden)
{
    conv_proj = Tensor({n_mels, hidden});
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

Tensor AudioEncoder::forward(const Tensor& spectrogram) {
    int64_t B = spectrogram.dim(0);
    int64_t n_mels = spectrogram.dim(1);
    int64_t T = spectrogram.dim(2);
    int64_t D = hidden_size;

    Tensor projected({B, T, D});
    for (int64_t b = 0; b < B; ++b) {
        Tensor spec_slice = spectrogram.slice(0, b, b + 1).reshape({n_mels, T});
        Tensor proj_slice = projected.slice(0, b, b + 1).reshape({T, D});
        math::gemm(1.0f, spec_slice.transpose(0, 1), conv_proj, 0.0f, proj_slice);
    }

    Tensor h({B, T, D});
    const float* pe = pos_embed.data<float>();
    float* hd = h.data<float>();
    const float* pd = projected.data<float>();
    int64_t max_pos = pos_embed.dim(0);
    for (int64_t b = 0; b < B; ++b)
        for (int64_t t = 0; t < T; ++t) {
            int64_t pe_idx = t < max_pos ? t : max_pos - 1;
            for (int64_t d = 0; d < D; ++d)
                hd[(b * T + t) * D + d] = pd[(b * T + t) * D + d] + pe[pe_idx * D + d];
        }

    Tensor positions = Tensor::arange(T);
    Tensor mask({T * T});
    mask.fill(0.0f);
    KVCache dummy_cache;
    for (auto& block : blocks)
        h = block.forward(h, positions, mask, dummy_cache, 0);

    return h;
}

} // namespace multimodal
} // namespace quant
