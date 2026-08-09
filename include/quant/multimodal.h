#pragma once
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/trainer.h"
#include "quant/tokenizer.h"
#include <string>
#include <vector>

namespace quant {

// H1: Cross-modal attention — attention across text/image/audio modalities
class CrossModalAttention {
public:
    CrossModalAttention(int64_t hidden);
    Tensor forward(const std::vector<Tensor>& modalities);
private:
    int64_t hidden_;
};

// H2: Joint multi-modal model — unified transformer
class JointMultimodalModel {
public:
    JointMultimodalModel(Model* text_encoder, int64_t hidden);
    Tensor forward(const Tensor& text, const Tensor& image, const Tensor& audio);
private:
    Model* text_encoder_;
    CrossModalAttention cross_attn_;
};

// H3: ImageNet classification — ViT training + evaluation
class ImageNetClassifier {
public:
    ImageNetClassifier(Model* vit, int64_t n_classes = 1000);
    Tensor classify(const Tensor& image);
    float evaluate(DataLoader& val_dl);
private:
    Model* vit_;
    Tensor classifier_head_;
};

// H4: Speech recognition — audio encoder + decoder
class SpeechRecognizer {
public:
    SpeechRecognizer(Model* audio_encoder);
    std::string transcribe(const Tensor& audio);
private:
    Model* audio_encoder_;
};

// H5: OCR pipeline
class OCRPipeline {
public:
    OCRPipeline(Model* ocr_encoder);
    std::string recognize(const Tensor& image);
private:
    Model* ocr_encoder_;
};

// H6: Video understanding
class VideoUnderstanding {
public:
    VideoUnderstanding(Model* video_encoder);
    std::string describe(const Tensor& video_frames);
private:
    Model* video_encoder_;
};

// H7: Image captioning
class ImageCaptioning {
public:
    ImageCaptioning(Model* vision_encoder, Model* text_decoder);
    std::string caption(const Tensor& image, int max_tokens = 32);
private:
    Model* vision_encoder_, *text_decoder_;
};

// H8: Visual QA
class VisualQA {
public:
    VisualQA(Model* vision_encoder, Model* text_decoder);
    std::string answer(const Tensor& image, const std::string& question);
private:
    Model* vision_encoder_, *text_decoder_;
};

// H9: Text-to-image — diffusion model
class TextToImage {
public:
    struct UNetBlock {
        Tensor conv_weight;
        Tensor conv_bias;
        Tensor bn_weight;
        Tensor bn_bias;
        int64_t in_channels, out_channels;
    };

    TextToImage(int64_t latent_dim = 64, int64_t image_size = 256);
    Tensor generate(const std::string& prompt, int steps = 50);
private:
    int64_t latent_dim_, image_size_;
    std::vector<UNetBlock> unet_;
};

// H10: Audio synthesis — mel to waveform
class AudioSynthesizer {
public:
    AudioSynthesizer();
    Tensor synthesize(const Tensor& mel_spectrogram);
};

// H11: Mel spectrogram computation
class MelSpectrogram {
public:
    MelSpectrogram(int sample_rate = 22050, int n_mels = 80);
    Tensor compute(const Tensor& waveform); // waveform: {channels, samples}
private:
    int sample_rate_, n_mels_;
    std::vector<float> mel_filterbank(int n_filters, int fft_size);
};

// H12: Audio feature extractor
class AudioFeatureExtractor {
public:
    AudioFeatureExtractor(Model* encoder);
    Tensor extract(const Tensor& audio);
private:
    Model* encoder_;
};

// H13: Multi-modal tokenizer — unified vocabulary
class MultiModalTokenizer {
public:
    MultiModalTokenizer();
    explicit MultiModalTokenizer(BPETokenizer* bpe);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);
    std::vector<int> encode_image(const Tensor& image);
    std::vector<int> encode_audio(const Tensor& audio);
    int vocab_size() const { return bpe_ ? bpe_->vocab_size() : 32000; }
    int image_token_id() const { return 30000; }
    int audio_token_id() const { return 31000; }
private:
    BPETokenizer* bpe_ = nullptr;
};

// H14: Modality encoder
class ModalityEncoder {
public:
    ModalityEncoder(Model* model, const std::string& modality);
    Tensor encode(const Tensor& input);
private:
    Model* model_;
    std::string modality_;
};

// H15: Cross-modal alignment — contrastive learning
class CrossModalAlignment {
public:
    CrossModalAlignment(Model* vision, Model* text, float temperature = 0.07f);
    float contrastive_loss(const Tensor& image_emb, const Tensor& text_emb);
    Tensor align(const Tensor& image_emb, const Tensor& text_emb);
private:
    Model* vision_, *text_;
    float temperature_;
};

// H16: Perceiver — learned query-based cross-attention
class Perceiver {
public:
    Perceiver(int64_t hidden, int64_t num_queries, int64_t num_heads);
    Tensor forward(const Tensor& input, const Tensor& queries);
private:
    int64_t hidden_;
    int64_t num_queries_;
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;
};

// H17: Vision MoE — modality-specific MoE for vision tokens
class VisionMoE {
public:
    VisionMoE(int64_t hidden, int64_t num_experts = 4);
    Tensor forward(const Tensor& image_features);
private:
    int64_t hidden_;
    std::vector<FFN> experts;
    Linear router;
};

// H18: Audio MoE — modality-specific MoE for audio tokens
class AudioMoE {
public:
    AudioMoE(int64_t hidden, int64_t num_experts = 4);
    Tensor forward(const Tensor& audio_features);
private:
    int64_t hidden_;
    std::vector<FFN> experts;
    Linear router;
};

// H19: Vision+Text MoE — cross-modal routing
class VisionTextMoE {
public:
    VisionTextMoE(int64_t hidden, int64_t num_experts = 6);
    Tensor forward(const Tensor& vision_tokens, const Tensor& text_tokens);
private:
    int64_t hidden_;
    std::vector<FFN> experts;
    Linear router;
    Linear modality_bias;
};

// H20: Audio+Text MoE — cross-modal routing
class AudioTextMoE {
public:
    AudioTextMoE(int64_t hidden, int64_t num_experts = 6);
    Tensor forward(const Tensor& audio_tokens, const Tensor& text_tokens);
private:
    int64_t hidden_;
    std::vector<FFN> experts;
    Linear router;
    Linear modality_bias;
};

// H21: All Modality MoE — routes across all modalities
class AllModalityMoE {
public:
    AllModalityMoE(int64_t hidden, int64_t num_experts = 8);
    Tensor forward(const std::vector<Tensor>& modality_tokens);
private:
    int64_t hidden_;
    std::vector<FFN> experts;
    Linear router;
    Linear modality_classifier;
};

// ============================================================================
// Fusion classes — cross-attention fusion and modality projection
// ============================================================================

struct FusionConfig {
    int64_t hidden_size = 768;
    int64_t num_heads = 8;
    int64_t head_dim = 96;
    int64_t num_fusion_layers = 2;
    float dropout = 0.1f;
};

class ModalityProjection {
public:
    Linear vision_proj;
    Linear audio_proj;
    Linear text_proj;
    int64_t hidden_size;

    ModalityProjection(int64_t hidden_size);
    Tensor project_vision(const Tensor& vision_features) const;
    Tensor project_audio(const Tensor& audio_features) const;
    Tensor project_text(const Tensor& text_features) const;
    Tensor project(const Tensor& features, const std::string& modality) const;
};

class CrossAttentionFusion {
public:
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;
    int64_t hidden_size;
    float scale;

    CrossAttentionFusion(int64_t hidden_size);
    Tensor forward(const Tensor& query, const Tensor& key_value, const Tensor& mask = Tensor()) const;
};

class MultimodalFusion {
public:
    ModalityProjection proj;
    std::vector<CrossAttentionFusion> vision_audio_layers;
    std::vector<CrossAttentionFusion> vision_text_layers;
    std::vector<CrossAttentionFusion> audio_text_layers;
    int64_t hidden_size;

    MultimodalFusion(const FusionConfig& cfg = FusionConfig());
    Tensor fuse_vision_audio(const Tensor& vision, const Tensor& audio) const;
    Tensor fuse_vision_text(const Tensor& vision, const Tensor& text) const;
    Tensor fuse_audio_text(const Tensor& audio, const Tensor& text) const;
    Tensor fuse_all(const Tensor& vision, const Tensor& audio, const Tensor& text) const;
    Tensor fuse_all_multistream(const Tensor& vision, const Tensor& audio, const Tensor& text) const;
    Tensor compute_fusion_weights(const Tensor& vision, const Tensor& audio, const Tensor& text) const;
    Tensor fuse_weighted(const Tensor& vision, const Tensor& audio, const Tensor& text, float vw, float aw, float tw) const;
    size_t param_count() const;
};

} // namespace quant
