#pragma once
#include "oil/tensor.h"
#include "oil/transformer.h"
#include <vector>
#include <string>
#include <cstdint>

namespace oil {
namespace multimodal {

enum class VisionTask {
    CLASSIFY,
    DETECT,
    SEGMENT,
    CAPTION,
    DEPTH_ESTIMATE
};

struct VisionResult {
    Tensor features;        // {batch, num_patches+1, hidden}
    Tensor class_logits;    // {batch, num_classes}
    Tensor bboxes;          // {batch, max_detections, 4}
    Tensor captions;        // {batch, max_caption_len, vocab_size}
    float confidence;
};

class VisionEncoder {
public:
    VisionEncoder(int64_t img_size, int64_t patch_size, int64_t hidden_size,
                  int64_t num_layers, int64_t num_heads, int64_t num_classes,
                  int64_t max_detections = 50);

    Tensor encode(const Tensor& image);
    Tensor classify(const Tensor& encoded);
    Tensor detect_objects(const Tensor& encoded);
    Tensor generate_caption(const Tensor& encoded, int64_t max_len);
    Tensor interpolate_pos_encoding(int64_t new_h, int64_t new_w) const;
    Tensor encode_multiscale(const std::vector<Tensor>& images);
    std::vector<Tensor> extract_attention_maps(const Tensor& image);
    void init_weights(float std_val = 0.02f);
    Tensor extract_patch_features(const Tensor& patches);
    Tensor compute_self_attention(const Tensor& query, const Tensor& key);
    VisionResult forward(const Tensor& image, VisionTask task);
    Tensor patch_embed;
    Tensor pos_embed;
    Tensor cls_token;
    std::vector<TransformerBlock> blocks;
    Tensor class_head;
    Tensor bbox_head;
    Tensor caption_embed;
    Tensor caption_proj;
    int64_t img_size, patch_size, hidden_size, num_patches;
    int64_t num_classes, max_detections;
};

class Conv2dPatchEmbed {
public:
    Tensor weight;
    Tensor bias;
    int64_t in_channels;
    int64_t embed_dim;
    int64_t patch_size;

    Conv2dPatchEmbed(int64_t in_channels, int64_t embed_dim, int64_t patch_size);
    Tensor forward(const Tensor& x);
};

class LayerNorm2D {
public:
    Tensor gamma;
    Tensor beta;
    float eps;

    LayerNorm2D(int64_t normalized_shape, float eps = 1e-5f);
    Tensor forward(const Tensor& x);
};

Tensor interpolate_pos_encoding_bilinear(const Tensor& pos_embed,
    int64_t old_h, int64_t old_w, int64_t new_h, int64_t new_w,
    int64_t patch_size, int64_t hidden_size);

class ViTFeatureExtractor {
public:
    ViTFeatureExtractor(int64_t img_size, int64_t patch_size);
    Tensor extract_patches(const Tensor& image);
    Tensor normalize(const Tensor& image);
    Tensor resize(const Tensor& image, int64_t new_h, int64_t new_w) const;
    Tensor center_crop(const Tensor& image, int64_t size) const;
    Tensor random_erasing(const Tensor& image, float prob, float scale) const;
    Tensor normalize_quantized(const Tensor& image) const;
    int64_t img_size, patch_size;
};

} // namespace multimodal
} // namespace oil
