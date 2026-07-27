#pragma once
#include "oil/tensor.h"
#include "oil/transformer.h"
#include <vector>
#include <cstdint>

namespace oil {
namespace multimodal {

struct AudioFeatures {
    Tensor mel_spec;        // {batch, n_mels, time}
    Tensor features;        // {batch, time_frames, hidden}
    Tensor logits;          // {batch, time_frames, vocab_size}
    float duration_sec;
};

class MelSpectrogram {
public:
    MelSpectrogram(int64_t n_mels, int64_t sample_rate, int64_t fft_size, int64_t hop_length);
    Tensor forward(const Tensor& waveform);

    Tensor build_mel_filterbank();
    Tensor stft(const Tensor& waveform);
    Tensor magnitude(const Tensor& complex);
    Tensor apply_filterbank(const Tensor& mag_spectrogram);

    int64_t n_mels, sample_rate, fft_size, hop_length;
    Tensor filterbank;
};

class AudioEncoder {
public:
    AudioEncoder(int64_t n_mels, int64_t hidden_size, int64_t num_layers,
                 int64_t num_heads, int64_t max_frames);
    Tensor encode(const Tensor& mel_spec);
    Tensor encode_from_waveform(const Tensor& waveform);

    MelSpectrogram mel;
    Tensor input_proj;
    Tensor pos_embed;
    std::vector<TransformerBlock> blocks;
    Tensor output_proj;
    int64_t hidden_size, n_mels, max_frames;
};

class MFCC {
public:
    int64_t n_mfcc;
    int64_t n_mels;
    int64_t fft_size;
    int64_t sample_rate;
    Tensor dct_matrix;

    MFCC(int64_t n_mfcc, int64_t n_mels, int64_t fft_size, int64_t sample_rate);
    Tensor forward(const Tensor& log_mel_spec);
    Tensor compute_dct(const Tensor& log_mel_spec) const;
};

Tensor pre_emphasis(const Tensor& waveform, float coeff = 0.97f);
Tensor frame_signal(const Tensor& waveform, int64_t frame_len, int64_t hop_len);
Tensor power_to_db(const Tensor& power, float ref_power = 1.0f, float floor_val = 1e-10f);
Tensor compute_spectral_centroid(const Tensor& magnitude_spec, int64_t fft_size, int64_t sample_rate);
Tensor compute_spectral_bandwidth(const Tensor& magnitude_spec, const Tensor& centroid, int64_t fft_size, int64_t sample_rate);
Tensor compute_spectral_rolloff(const Tensor& magnitude_spec, float rolloff_percent, int64_t fft_size, int64_t sample_rate);
Tensor compute_deltas(const Tensor& features, int64_t window = 2);
Tensor spec_augment(const Tensor& mel_spec, int64_t freq_mask_width, int64_t time_mask_width, float prob = 0.5f);
Tensor voice_activity_detection(const Tensor& mel_spec, float threshold_db = -16.0f);

class SpeechRecognizer {
public:
    SpeechRecognizer(int64_t vocab_size, int64_t hidden_size,
                     int64_t num_layers, int64_t num_heads);
    Tensor recognize(const Tensor& audio_features);
    Tensor ctc_decode(const Tensor& logits);
    Tensor beam_search_decode(const Tensor& logits, int64_t beam_width = 5) const;
    float compute_ctc_loss(const Tensor& logits, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths) const;
    Tensor greedy_decode(const Tensor& logits) const;

    AudioEncoder encoder;
    Tensor ctc_head;
    Tensor blank_logit;
    int64_t vocab_size, hidden_size;
};

} // namespace multimodal
} // namespace oil
