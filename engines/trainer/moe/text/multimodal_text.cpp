#include "multimodal_text.h"
#include "quant/math.h"
#include <cmath>

namespace quant {
namespace multimodal {

TextEncoder::TextEncoder(int64_t vocab_size, int64_t hidden, int64_t max_len_)
    : embed({vocab_size, hidden}),
      pos_embed({max_len_, hidden}),
      hidden_size(hidden),
      max_len(max_len_)
{
    embed.zero_();
    float* pe = pos_embed.data<float>();
    for (int64_t p = 0; p < max_len_; ++p) {
        for (int64_t i = 0; i < hidden; ++i) {
            float exponent = (2.0f * (float)(i / 2)) / (float)hidden;
            float angle = (float)p / std::pow(10000.0f, exponent);
            if (i % 2 == 0)
                pe[p * hidden + i] = std::sin(angle);
            else
                pe[p * hidden + i] = std::cos(angle);
        }
    }
}

Tensor TextEncoder::forward(const Tensor& input_ids) {
    int64_t B = input_ids.dim(0);
    int64_t S = input_ids.dim(1);
    int64_t H = hidden_size;

    Tensor token_emb({B, S, H});
    const int64_t* ids = input_ids.data<int64_t>();
    float* te = token_emb.data<float>();
    const float* w = embed.data<float>();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t s = 0; s < S; ++s) {
            int64_t idx = ids[b * S + s];
            if (idx < 0) idx = 0;
            if (idx >= embed.dim(0)) idx = embed.dim(0) - 1;
            std::memcpy(te + (b * S + s) * H, w + idx * H, H * sizeof(float));
        }

    Tensor output({B, S, H});
    const float* pe = pos_embed.data<float>();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t s = 0; s < S; ++s)
            for (int64_t h = 0; h < H; ++h)
                output.data<float>()[(b * S + s) * H + h] =
                    te[(b * S + s) * H + h] + pe[s * H + h];

    return output;
}

} // namespace multimodal
} // namespace quant
