#pragma once
#include "oil/tensor.h"
#include "oil/types.h"
#include "oil/transformer.h"
#include "oil/math.h"
#include "image.h"
#include <vector>
#include <string>
#include <utility>
#include <cmath>

namespace oil {
namespace multimodal {

struct VideoConfig {
    int64_t tube_depth = 2;
    int64_t tube_height = 16;
    int64_t tube_width = 16;
    int64_t hidden_size = 768;
    int64_t num_heads = 8;
    int64_t head_dim = 96;
    int64_t num_layers = 4;
    int64_t clip_len = 16;
    int64_t ffn_hidden = 3072;
    int64_t max_frames = 128;
    float norm_eps = 1e-5f;
};

class TemporalPositionalEncoding {
public:
    Tensor pe;
    int64_t max_len;
    int64_t dim;
    TemporalPositionalEncoding(int64_t max_len, int64_t dim);
    Tensor forward(int64_t seq_len) const;
};

class TemporalTransformer {
public:
    TemporalPositionalEncoding temp_pos_enc;
    std::vector<oil::TransformerBlock> blocks;
    oil::RMSNorm final_norm;
    int64_t hidden_size;
    int64_t max_frames;

    explicit TemporalTransformer(const VideoConfig& cfg = VideoConfig());
    Tensor forward(const std::vector<Tensor>& frame_features) const;
};

Tensor conv3d_forward(const Tensor& input, const Tensor& weight, const Tensor& bias,
                       int64_t stride_d, int64_t stride_h, int64_t stride_w);
Tensor max_pool3d(const Tensor& x, int64_t kernel_d, int64_t kernel_h, int64_t kernel_w,
                  int64_t stride_d, int64_t stride_h, int64_t stride_w);
Tensor temporal_pooling(const Tensor& features, const std::string& mode = "mean");
Tensor compute_optical_flow(const Tensor& prev_frame, const Tensor& curr_frame,
                            int64_t block_size = 8, int64_t search_radius = 4);

class InceptionModule {
public:
    ConvBlock branch1x1;
    ConvBlock branch3x3_reduce;
    ConvBlock branch3x3;
    ConvBlock branch5x5_reduce;
    ConvBlock branch5x5;
    ConvBlock branch_pool;
    int64_t in_channels;
    int64_t hidden_size;

    InceptionModule(int64_t in_c, int64_t hidden);
    Tensor forward(const Tensor& x);
};

class MotionFeatureExtractor {
public:
    int64_t block_size;
    int64_t search_radius;
    int64_t hidden_size;

    MotionFeatureExtractor(int64_t block_sz, int64_t search_rad, int64_t hidden);
    Tensor extract_motion(const Tensor& video_frames) const;
    Tensor compute_motion_features(const std::vector<Tensor>& flow_fields) const;
};

class SplitVideoEncoder {
public:
    oil::Linear patch_proj;
    oil::Linear vision_proj;
    TemporalTransformer temporal_transformer;
    int64_t tube_depth, tube_height, tube_width;
    int64_t hidden_size;
    int64_t clip_len;

    explicit SplitVideoEncoder(const VideoConfig& cfg = VideoConfig());
    Tensor encode(const Tensor& video_frames) const;
    Tensor encode_with_motion(const Tensor& video_frames) const;
    Tensor temporal_aggregate(const Tensor& encoded, const std::string& mode = "mean") const;
    std::string recognize_action(const Tensor& video) const;
    std::vector<std::pair<int64_t, int64_t>> track_objects(const Tensor& video_frames) const;

private:
    std::vector<Tensor> extract_3d_tubes(const Tensor& video_frames) const;
    Tensor encode_clip(const Tensor& clip) const;
};

} // namespace multimodal
} // namespace oil
