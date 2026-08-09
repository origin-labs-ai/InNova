#pragma once

#include "quant/tensor.h"
#include "quant/transformer.h"
#include <cstdint>
#include <vector>
#include <string>
#include <memory>

namespace quant {

// Forward declarations
class Tensor;

// ========================================================================
// Audio Encoder — encode raw PCM audio into latent representations
// ========================================================================
class AudioEncoder {
public:
    AudioEncoder() = default;
    ~AudioEncoder() = default;

    struct Config {
        int sample_rate = 16000;
        int n_mels = 80;
        int n_fft = 400;
        int hop_length = 160;
        int n_encoder_layers = 6;
        int d_model = 512;
        int n_heads = 8;
    };

    void init(const Config& cfg);
    Tensor encode(const float* pcm_samples, int64_t n_samples);
    Tensor encode(const std::vector<float>& pcm_samples);

    int sample_rate() const { return config_.sample_rate; }
    int n_mels() const { return config_.n_mels; }
    int d_model() const { return config_.d_model; }
    bool is_initialized() const { return initialized_; }

private:
    Config config_;
    bool initialized_ = false;
};

// ========================================================================
// Mel Spectrogram — extract mel-frequency spectrograms from audio
// ========================================================================
class MelSpectrogram {
public:
    MelSpectrogram() = default;
    ~MelSpectrogram() = default;

    struct Config {
        int sample_rate = 16000;
        int n_mels = 80;
        int n_fft = 400;
        int hop_length = 160;
        bool normalize = true;
    };

    void init(const Config& cfg);
    Tensor extract(const float* pcm_samples, int64_t n_samples);
    Tensor extract(const std::vector<float>& pcm_samples);

    int n_mels() const { return config_.n_mels; }
    int feature_length(int64_t n_samples) const {
        return (int)((n_samples - config_.n_fft) / config_.hop_length + 1);
    }
    bool is_initialized() const { return initialized_; }

private:
    Config config_;
    bool initialized_ = false;

    static std::vector<float> compute_mel_filterbank(int n_mels, int n_fft, int sample_rate);
};

// ========================================================================
// Audio Features — unified feature extraction for multimodal pipeline
// ========================================================================
struct AudioFeatures {
    Tensor mel_spectrogram;
    Tensor encoder_output;
    int64_t n_frames = 0;
    int sample_rate = 16000;
};

class AudioFeatureExtractor {
public:
    AudioFeatureExtractor() = default;
    ~AudioFeatureExtractor() = default;

    void init(int sample_rate = 16000, int n_mels = 80);
    AudioFeatures extract(const float* pcm_samples, int64_t n_samples);
    AudioFeatures extract(const std::vector<float>& pcm_samples);

    bool is_initialized() const { return initialized_; }

private:
    AudioEncoder encoder_;
    MelSpectrogram mel_;
    bool initialized_ = false;
};

} // namespace quant
