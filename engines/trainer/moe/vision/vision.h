#pragma once
#include "quant/tensor.h"
#include "quant/transformer.h"
#include <vector>
#include <string>

namespace quant {
namespace multimodal {

enum class VisualTask {
    CLASSIFY,
    DETECT,
    SEGMENT,
    CAPTION,
    SCENE_GRAPH,
    DEPTH_ESTIMATE,
    POSE_ESTIMATE
};

struct SceneGraph {
    Tensor objects;        // {num_objects, 4} — bbox coords
    Tensor labels;         // {num_objects} — class indices
    Tensor relations;      // {num_relations, 3} — subject, predicate, object
    Tensor global_features; // {hidden_size} — whole-scene embedding
    Tensor caption_embeds; // {max_caption_len, hidden_size}
    std::vector<VisualTask> completed_tasks;
};

class ObjectDetector {
public:
    ObjectDetector(int64_t hidden_size, int64_t num_classes, int64_t max_detections);
    Tensor forward(const Tensor& visual_features);
    Tensor bbox_head;
    Tensor class_head;
    Tensor object_query;
    Tensor q_proj, k_proj, v_proj;
    int64_t hidden_size;
    int64_t num_classes;
    int64_t max_detections;
    Tensor last_class_scores;
};

class SceneGraphEncoder {
public:
    SceneGraphEncoder(int64_t hidden_size, int64_t num_relation_types);
    Tensor forward(const Tensor& object_features, const Tensor& spatial_edges);
    Tensor relation_head;
    Tensor spatial_embed;
    int64_t hidden_size;
    int64_t num_relation_types;
};

class VisionEncoder {
public:
    VisionEncoder(int64_t img_size, int64_t patch_size, int64_t hidden_size,
                  int64_t num_layers, int64_t num_heads, int64_t num_classes,
                  int64_t max_frames = 32);

    // Image understanding
    SceneGraph understand_image(const Tensor& image);
    Tensor classify(const Tensor& image);
    Tensor detect(const Tensor& image);
    Tensor caption(const Tensor& image);
    SceneGraph build_scene_graph(const Tensor& image);

    // Video understanding (temporal)
    SceneGraph understand_video(const Tensor& video_frames);
    Tensor recognize_action(const Tensor& video_frames);
    Tensor track_objects(const Tensor& video_frames);

    // Unified perception entry point
    SceneGraph perceive(const Tensor& visual_input, bool is_video = false);

    // Core encoder
    Tensor encode_visual(const Tensor& visual_input);

    Tensor patch_embed;
    Tensor pos_embed;
    Tensor cls_token;
    Tensor temporal_pos_embed;
    std::vector<TransformerBlock> blocks;
    ObjectDetector detector;
    SceneGraphEncoder scene_encoder;
    Tensor caption_proj;
    Tensor global_pool;

    int64_t hidden_size;
    int64_t img_size;
    int64_t patch_size;
    int64_t num_patches;
    int64_t num_classes;
    int64_t max_frames;
};

} // namespace multimodal
} // namespace quant
