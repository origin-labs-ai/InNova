#pragma once
#include "oil/tensor.h"
#include "oil/transformer.h"
#include <vector>
#include <cstdint>

namespace oil {
namespace multimodal {

struct ImageGenResult {
    Tensor generated;       // {batch, channels, height, width}
    Tensor latents;         // {batch, latent_dim}
    std::vector<float> scores;
};

class ImageEncoder {
public:
    ImageEncoder(int64_t img_size, int64_t hidden_size, int64_t num_layers,
                 int64_t num_heads, int64_t latent_dim);
    Tensor encode(const Tensor& image);
    Tensor project_to_latent(const Tensor& features);
    Tensor encode_cnn_frontend(const Tensor& image);
    Tensor extract_features(const Tensor& image, bool use_cnn_frontend = false);
    Tensor extract_multi_scale_features(const Tensor& image);
    int64_t img_size, hidden_size, latent_dim;
    Tensor conv_in;
    std::vector<TransformerBlock> blocks;
    Tensor latent_proj;
};

class ImageDecoder {
public:
    ImageDecoder(int64_t latent_dim, int64_t hidden_size, int64_t num_layers,
                 int64_t num_heads, int64_t img_size);
    Tensor decode(const Tensor& latent);
    Tensor upscale(const Tensor& features);
    Tensor decode_v2(const Tensor& latent, int64_t scale_factor = 1);
    int64_t latent_dim, hidden_size, img_size;
    std::vector<TransformerBlock> blocks;
    Tensor latent_in;
    Tensor conv_out;
    Tensor up_proj;
};

class ConvBlock {
public:
    Tensor conv_weight;
    Tensor conv_bias;
    int64_t in_channels;
    int64_t out_channels;
    int64_t kernel_size;
    int64_t stride;
    int64_t padding;

    ConvBlock(int64_t in_c, int64_t out_c, int64_t ksz,
              int64_t stride = 1, int64_t padding = 0);
    Tensor forward(const Tensor& x);
};

class ResidualBlock {
public:
    ConvBlock conv1;
    ConvBlock conv2;
    int64_t channels;

    ResidualBlock(int64_t channels, int64_t kernel_size = 3);
    Tensor forward(const Tensor& x);
};

Tensor im2col(const Tensor& x, int64_t kernel_h, int64_t kernel_w,
              int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w);
Tensor max_pool2d(const Tensor& x, int64_t kernel_size, int64_t stride);
Tensor global_avg_pool2d(const Tensor& x);
Tensor adaptive_avg_pool2d(const Tensor& x, int64_t out_h, int64_t out_w);
Tensor conv2d_batch_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                          const Tensor& running_mean, const Tensor& running_var,
                          float eps = 1e-5f);

class ImageGenerator {
public:
    ImageGenerator(int64_t img_size, int64_t hidden_size, int64_t num_layers,
                   int64_t num_heads, int64_t latent_dim);
    ImageGenResult generate(const Tensor& conditioning, int64_t num_steps = 50);
    ImageEncoder encoder;
    ImageDecoder decoder;
    Tensor noise_schedule;
    Tensor timestep_embed;
    int64_t img_size, hidden_size, latent_dim;
};

} // namespace multimodal
} // namespace oil
