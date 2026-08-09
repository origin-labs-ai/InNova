#include "embeddings.h"
#include "quant/math.h"

namespace quant {
namespace multimodal {

SentenceEmbedder::SentenceEmbedder(int64_t hidden, int64_t output)
    : projection({hidden, output}),
      hidden_size(hidden),
      output_size(output)
{
    projection.zero_();
}

Tensor SentenceEmbedder::forward(const Tensor& token_embeddings) {
    int64_t B = token_embeddings.dim(0);
    int64_t S = token_embeddings.dim(1);
    int64_t H = token_embeddings.dim(2);
    int64_t D = output_size;

    Tensor pooled({B, H});
    pooled.zero_();
    const float* te = token_embeddings.data<float>();
    float* pl = pooled.data<float>();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t s = 0; s < S; ++s)
            for (int64_t h = 0; h < H; ++h)
                pl[b * H + h] += te[(b * S + s) * H + h];
    for (int64_t b = 0; b < B; ++b)
        for (int64_t h = 0; h < H; ++h)
            pl[b * H + h] /= (float)S;

    Tensor output({B, D});
    math::gemm(1.0f, pooled, projection, 0.0f, output);

    return output;
}

} // namespace multimodal
} // namespace quant
