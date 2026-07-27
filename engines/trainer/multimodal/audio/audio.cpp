#include "audio.h"
#include "oil/math.h"
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <vector>

namespace oil {
namespace multimodal {

namespace {
    constexpr float kPi = 3.14159265358979323846f;

    float rand_uniform(float lo, float hi) {
        return lo + (hi - lo) * (std::rand() / (float)RAND_MAX);
    }

    void init_weight(Tensor& w, int64_t in_features, int64_t out_features) {
        float scale = std::sqrt(6.0f / (float)(in_features + out_features));
        float* data = w.data<float>();
        for (int64_t i = 0; i < w.numel(); i++)
            data[i] = rand_uniform(-scale, scale);
    }

}

// ============================================================================
// MelSpectrogram
// ============================================================================

MelSpectrogram::MelSpectrogram(int64_t n_mels, int64_t sample_rate,
                               int64_t fft_size, int64_t hop_length)
    : n_mels(n_mels), sample_rate(sample_rate)
    , fft_size(fft_size), hop_length(hop_length) {
    build_mel_filterbank();
}

Tensor MelSpectrogram::build_mel_filterbank() {
    int64_t n_fft_bins = fft_size / 2 + 1;
    filterbank = Tensor::zeros({n_mels, n_fft_bins});
    float* fb = filterbank.data<float>();

    auto mel_to_hz = [](float mel) {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    };

    float mel_min = 0.0f;
    float mel_max = 2595.0f * std::log10(1.0f + (sample_rate / 2.0f) / 700.0f);

    std::vector<int64_t> bins(n_mels + 2);
    for (int64_t i = 0; i < n_mels + 2; i++) {
        float mel = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        float hz = mel_to_hz(mel);
        bins[i] = (int64_t)std::round(hz * (fft_size + 1) / sample_rate);
        bins[i] = std::max<int64_t>(0, std::min<int64_t>(n_fft_bins - 1, bins[i]));
    }

    for (int64_t m = 0; m < n_mels; m++) {
        int64_t l = bins[m], c = bins[m + 1], r = bins[m + 2];
        float* row = fb + m * n_fft_bins;
        for (int64_t k = l; k < c && c > l; k++)
            row[k] = (float)(k - l) / (float)(c - l);
        for (int64_t k = c; k <= r && r > c; k++)
            row[k] = (float)(r - k) / (float)(r - c);
    }

    for (int64_t m = 0; m < n_mels; m++) {
        float* row = fb + m * n_fft_bins;
        float sum = 0.0f;
        for (int64_t k = 0; k < n_fft_bins; k++) sum += row[k];
        if (sum > 0.0f)
            for (int64_t k = 0; k < n_fft_bins; k++) row[k] /= sum;
    }
    return filterbank;
}

Tensor MelSpectrogram::stft(const Tensor& waveform) {
    int64_t batch = waveform.dim(0);
    int64_t samples = waveform.dim(1);
    int64_t n_fft_bins = fft_size / 2 + 1;
    int64_t n_frames = (samples - fft_size) / hop_length + 1;
    if (n_frames < 0) n_frames = 0;

    Tensor result = Tensor::zeros({batch, n_fft_bins, n_frames, 2});
    const float* wav = waveform.data<float>();
    float* out = result.data<float>();

    std::vector<float> window(fft_size);
    for (int64_t n = 0; n < fft_size; n++)
        window[n] = 0.5f * (1.0f - std::cos(2.0f * kPi * n / (fft_size - 1)));

    std::vector<std::vector<float>> cos_t(n_fft_bins, std::vector<float>(fft_size));
    std::vector<std::vector<float>> sin_t(n_fft_bins, std::vector<float>(fft_size));
    for (int64_t k = 0; k < n_fft_bins; k++) {
        for (int64_t n = 0; n < fft_size; n++) {
            float ang = 2.0f * kPi * k * n / fft_size;
            cos_t[k][n] = std::cos(ang);
            sin_t[k][n] = std::sin(ang);
        }
    }

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t t = 0; t < n_frames; t++) {
            int64_t start = t * hop_length;
            for (int64_t k = 0; k < n_fft_bins; k++) {
                float re = 0.0f, im = 0.0f;
                const float* frame = wav + b * samples + start;
                for (int64_t n = 0; n < fft_size; n++) {
                    float x = frame[n] * window[n];
                    re += x * cos_t[k][n];
                    im -= x * sin_t[k][n];
                }
                int64_t idx = ((b * n_fft_bins + k) * n_frames + t) * 2;
                out[idx] = re;
                out[idx + 1] = im;
            }
        }
    }
    return result;
}

Tensor MelSpectrogram::magnitude(const Tensor& complex) {
    int64_t batch = complex.dim(0);
    int64_t n_freq = complex.dim(1);
    int64_t n_time = complex.dim(2);

    Tensor mag = Tensor::zeros({batch, n_freq, n_time});
    const float* c = complex.data<float>();
    float* m = mag.data<float>();
    int64_t N = batch * n_freq * n_time;
    for (int64_t i = 0; i < N; i++)
        m[i] = std::sqrt(c[i * 2] * c[i * 2] + c[i * 2 + 1] * c[i * 2 + 1]);
    return mag;
}

Tensor MelSpectrogram::apply_filterbank(const Tensor& mag_spectrogram) {
    int64_t batch = mag_spectrogram.dim(0);
    int64_t n_freq = mag_spectrogram.dim(1);
    int64_t n_time = mag_spectrogram.dim(2);

    Tensor mel = Tensor::zeros({batch, n_mels, n_time});
    const float* mag = mag_spectrogram.data<float>();
    const float* fb = filterbank.data<float>();
    float* mel_d = mel.data<float>();

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t t = 0; t < n_time; t++) {
            for (int64_t m = 0; m < n_mels; m++) {
                float sum = 0.0f;
                for (int64_t f = 0; f < n_freq; f++)
                    sum += fb[m * n_freq + f] * mag[(b * n_freq + f) * n_time + t];
                mel_d[(b * n_mels + m) * n_time + t] = sum;
            }
        }
    }
    return mel;
}

Tensor MelSpectrogram::forward(const Tensor& waveform) {
    return apply_filterbank(magnitude(stft(waveform)));
}

// ============================================================================
// AudioEncoder
// ============================================================================

AudioEncoder::AudioEncoder(int64_t n_mels, int64_t hidden_size,
                           int64_t num_layers, int64_t num_heads,
                           int64_t max_frames)
    : mel(n_mels, 16000, 512, 160)
    , input_proj(Tensor::zeros({n_mels, hidden_size}))
    , pos_embed(Tensor::zeros({1, max_frames, hidden_size}))
    , output_proj(Tensor::zeros({hidden_size, hidden_size}))
    , hidden_size(hidden_size), n_mels(n_mels), max_frames(max_frames) {

    std::srand(42);
    init_weight(input_proj, n_mels, hidden_size);
    init_weight(output_proj, hidden_size, hidden_size);

    float* pe = pos_embed.data<float>();
    for (int64_t p = 0; p < max_frames; p++) {
        for (int64_t d = 0; d < hidden_size; d++) {
            float ang = (float)p / std::pow(10000.0f, (float)(d - d % 2) / hidden_size);
            pe[p * hidden_size + d] = (d % 2 == 0) ? std::sin(ang) : std::cos(ang);
        }
    }

    for (int64_t i = 0; i < num_layers; i++) {
        TransformerConfig cfg;
        cfg.hidden_size = hidden_size;
        cfg.num_heads = num_heads;
        cfg.num_layers = num_layers;
        cfg.head_dim = hidden_size / num_heads;
        cfg.ffn_hidden_size = hidden_size * 4;
        cfg.max_seq_len = max_frames;
        cfg.activation = Activation::SiLU;
        blocks.emplace_back(cfg);
    }
}

Tensor AudioEncoder::encode(const Tensor& mel_spec) {
    int64_t batch = mel_spec.dim(0);
    int64_t time = mel_spec.dim(2);

    Tensor flat_mel = Tensor::zeros({batch * time, n_mels});
    const float* ms = mel_spec.data<float>();
    float* fm = flat_mel.data<float>();
    for (int64_t b = 0; b < batch; b++)
        for (int64_t t = 0; t < time; t++)
            for (int64_t m = 0; m < n_mels; m++)
                fm[(b * time + t) * n_mels + m] = ms[(b * n_mels + m) * time + t];

    Tensor proj = Tensor::zeros({batch * time, hidden_size});
    math::gemm(1.0f, flat_mel, input_proj, 0.0f, proj);

    Tensor features = proj.reshape({batch, time, hidden_size});
    float* feat = features.data<float>();
    const float* pe = pos_embed.data<float>();
    int64_t t_used = std::min(time, max_frames);
    for (int64_t b = 0; b < batch; b++)
        for (int64_t t = 0; t < t_used; t++)
            for (int64_t h = 0; h < hidden_size; h++)
                feat[(b * time + t) * hidden_size + h] += pe[t * hidden_size + h];

    KVCache cache((int)blocks.size(), max_frames,
                  blocks[0].attention.num_heads,
                  blocks[0].attention.head_dim);

    Tensor positions = Tensor::arange(t_used);
    Tensor mask = Tensor::zeros({1, 1, t_used, t_used});

    for (size_t i = 0; i < blocks.size(); i++)
        features = blocks[i].forward(features, positions, mask, cache, (int)i);

    Tensor flat_feat = features.reshape({batch * time, hidden_size});
    Tensor out_proj = Tensor::zeros({batch * time, hidden_size});
    math::gemm(1.0f, flat_feat, output_proj, 0.0f, out_proj);

    return out_proj.reshape({batch, time, hidden_size});
}

Tensor AudioEncoder::encode_from_waveform(const Tensor& waveform) {
    return encode(mel.forward(waveform));
}

// ============================================================================
// SpeechRecognizer
// ============================================================================

SpeechRecognizer::SpeechRecognizer(int64_t vocab_size, int64_t hidden_size,
                                   int64_t num_layers, int64_t num_heads)
    : encoder(80, hidden_size, num_layers, num_heads, 2048)
    , ctc_head(Tensor::zeros({hidden_size, vocab_size}))
    , blank_logit(Tensor::zeros({1}))
    , vocab_size(vocab_size), hidden_size(hidden_size) {
    std::srand(42);
    init_weight(ctc_head, hidden_size, vocab_size);
    blank_logit.data<float>()[0] = 0.0f;
}

Tensor SpeechRecognizer::recognize(const Tensor& audio_features) {
    int64_t batch = audio_features.dim(0);
    int64_t time = audio_features.dim(1);

    Tensor flat_feat = audio_features.reshape({batch * time, hidden_size});
    Tensor logits = Tensor::zeros({batch * time, vocab_size});
    math::gemm(1.0f, flat_feat, ctc_head, 0.0f, logits);

    return logits.reshape({batch, time, vocab_size});
}

Tensor SpeechRecognizer::ctc_decode(const Tensor& logits) {
    int64_t batch = logits.dim(0);
    int64_t time = logits.dim(1);
    int64_t v = logits.dim(2);

    const float* l = logits.data<float>();
    std::vector<std::vector<int64_t>> decoded(batch);

    for (int64_t b = 0; b < batch; b++) {
        int64_t prev_id = -1;
        for (int64_t t = 0; t < time; t++) {
            int64_t best = 0;
            float best_val = l[(b * time + t) * v];
            for (int64_t j = 1; j < v; j++) {
                float val = l[(b * time + t) * v + j];
                if (val > best_val) { best_val = val; best = j; }
            }
            int64_t blank = 0;
            if (best != blank && best != prev_id)
                decoded[b].push_back(best);
            if (best != blank)
                prev_id = best;
            else
                prev_id = -1;
        }
    }

    int64_t max_len = 0;
    for (auto& d : decoded)
        max_len = std::max(max_len, (int64_t)d.size());

    int64_t cols = max_len == 0 ? 1 : max_len;
    Tensor result(Shape{batch, cols}, DType::I64);
    result.zero_();
    int64_t* r = result.data<int64_t>();
    for (int64_t b = 0; b < batch; b++) {
        for (size_t i = 0; i < decoded[b].size(); i++)
            r[b * max_len + i] = decoded[b][i];
        for (size_t i = decoded[b].size(); i < (size_t)max_len; i++)
            r[b * max_len + i] = -1;
    }
    return result;
}

// ============================================================================
// MFCC
// ============================================================================

MFCC::MFCC(int64_t n_mfcc_val, int64_t n_mels_val, int64_t fft_sz, int64_t sr)
    : n_mfcc(n_mfcc_val), n_mels(n_mels_val), fft_size(fft_sz), sample_rate(sr) {
    dct_matrix = Tensor({n_mfcc, n_mels});
    float* dct = dct_matrix.data<float>();
    for (int64_t i = 0; i < n_mfcc; ++i) {
        for (int64_t j = 0; j < n_mels; ++j) {
            float ang = 3.141592653589793f * (float)(i) * ((float)j + 0.5f) / (float)n_mels;
            float val = std::cos(ang);
            if (i == 0)
                val *= std::sqrt(1.0f / (float)n_mels);
            else
                val *= std::sqrt(2.0f / (float)n_mels);
            dct[i * n_mels + j] = val;
        }
    }
}

Tensor MFCC::compute_dct(const Tensor& log_mel_spec) const {
    int64_t B = log_mel_spec.dim(0);
    int64_t n_m = log_mel_spec.dim(1);
    int64_t T = log_mel_spec.dim(2);

    int64_t n_dct = std::min(n_mfcc, n_mels);

    Tensor mfcc({B, n_dct, T});
    const float* mel = log_mel_spec.data<float>();
    float* mfc = mfcc.data<float>();
    const float* dct = dct_matrix.data<float>();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t i = 0; i < n_dct; ++i) {
                float sum = 0.0f;
                for (int64_t j = 0; j < n_mels; ++j)
                    sum += dct[i * n_mels + j] * mel[(b * n_mels + j) * T + t];
                mfc[(b * n_dct + i) * T + t] = sum;
            }
        }
    }

    return mfcc;
}

Tensor MFCC::forward(const Tensor& log_mel_spec) {
    return compute_dct(log_mel_spec);
}

// ============================================================================
// Audio feature utilities
// ============================================================================

Tensor pre_emphasis(const Tensor& waveform, float coeff) {
    int64_t B = waveform.dim(0);
    int64_t S = waveform.dim(1);

    Tensor out({B, S});
    const float* wav = waveform.data<float>();
    float* o = out.data<float>();

    for (int64_t b = 0; b < B; ++b) {
        o[b * S] = wav[b * S];
        for (int64_t i = 1; i < S; ++i)
            o[b * S + i] = wav[b * S + i] - coeff * wav[b * S + i - 1];
    }

    return out;
}

Tensor frame_signal(const Tensor& waveform, int64_t frame_len, int64_t hop_len) {
    int64_t B = waveform.dim(0);
    int64_t S = waveform.dim(1);
    int64_t n_frames = (S - frame_len) / hop_len + 1;
    if (n_frames < 0) n_frames = 0;

    Tensor frames({B, n_frames, frame_len});
    const float* wav = waveform.data<float>();
    float* f = frames.data<float>();

    for (int64_t b = 0; b < B; ++b)
        for (int64_t t = 0; t < n_frames; ++t)
            std::memcpy(f + (b * n_frames + t) * frame_len,
                        wav + b * S + t * hop_len,
                        frame_len * sizeof(float));

    return frames;
}

Tensor power_to_db(const Tensor& power, float ref_power, float floor_val) {
    int64_t N = power.numel();
    Tensor out = power.clone();
    float* o = out.data<float>();

    for (int64_t i = 0; i < N; ++i) {
        float val = std::max(o[i], floor_val);
        o[i] = 10.0f * std::log10(val / ref_power);
        if (o[i] < -80.0f) o[i] = -80.0f;
    }

    return out;
}

Tensor compute_spectral_centroid(const Tensor& magnitude_spec, int64_t fft_size, int64_t sr) {
    int64_t B = magnitude_spec.dim(0);
    int64_t n_fft_bins = magnitude_spec.dim(1);
    int64_t T = magnitude_spec.dim(2);

    Tensor centroid({B, T});
    const float* mag = magnitude_spec.data<float>();
    float* c = centroid.data<float>();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            float weighted_sum = 0.0f, total_mag = 0.0f;
            for (int64_t k = 0; k < n_fft_bins; ++k) {
                float freq = (float)k * (float)sr / (float)fft_size;
                float m = mag[(b * n_fft_bins + k) * T + t];
                weighted_sum += freq * m;
                total_mag += m;
            }
            c[b * T + t] = (total_mag > 0.0f) ? weighted_sum / total_mag : 0.0f;
        }
    }

    return centroid;
}

Tensor compute_spectral_bandwidth(const Tensor& magnitude_spec, const Tensor& centroid,
                                   int64_t fft_size, int64_t sr) {
    int64_t B = magnitude_spec.dim(0);
    int64_t n_fft_bins = magnitude_spec.dim(1);
    int64_t T = magnitude_spec.dim(2);

    Tensor bandwidth({B, T});
    const float* mag = magnitude_spec.data<float>();
    const float* cent = centroid.data<float>();
    float* bw = bandwidth.data<float>();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            float weighted_sum = 0.0f, total_mag = 0.0f;
            float c_val = cent[b * T + t];
            for (int64_t k = 0; k < n_fft_bins; ++k) {
                float freq = (float)k * (float)sr / (float)fft_size;
                float diff = freq - c_val;
                float m = mag[(b * n_fft_bins + k) * T + t];
                weighted_sum += diff * diff * m;
                total_mag += m;
            }
            bw[b * T + t] = (total_mag > 0.0f) ? std::sqrt(weighted_sum / total_mag) : 0.0f;
        }
    }

    return bandwidth;
}

Tensor compute_spectral_rolloff(const Tensor& magnitude_spec, float rolloff_pct,
                                 int64_t fft_size, int64_t sr) {
    int64_t B = magnitude_spec.dim(0);
    int64_t n_fft_bins = magnitude_spec.dim(1);
    int64_t T = magnitude_spec.dim(2);

    Tensor rolloff({B, T});
    const float* mag = magnitude_spec.data<float>();
    float* ro = rolloff.data<float>();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            float total_energy = 0.0f;
            for (int64_t k = 0; k < n_fft_bins; ++k)
                total_energy += mag[(b * n_fft_bins + k) * T + t];

            float threshold = rolloff_pct * total_energy;
            float cumulative = 0.0f;
            int64_t rolloff_bin = n_fft_bins - 1;

            for (int64_t k = 0; k < n_fft_bins; ++k) {
                cumulative += mag[(b * n_fft_bins + k) * T + t];
                if (cumulative >= threshold) {
                    rolloff_bin = k;
                    break;
                }
            }

            ro[b * T + t] = (float)rolloff_bin * (float)sr / (float)fft_size;
        }
    }

    return rolloff;
}

Tensor compute_deltas(const Tensor& features, int64_t window) {
    int64_t B = features.dim(0);
    int64_t D = features.dim(1);
    int64_t T = features.dim(2);

    Tensor deltas({B, D, T});
    deltas.zero_();
    const float* feat = features.data<float>();
    float* d = deltas.data<float>();

    float denom = 0.0f;
    for (int64_t w = 1; w <= window; ++w)
        denom += (float)(w * w);

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t f = 0; f < D; ++f) {
            for (int64_t t = 0; t < T; ++t) {
                float delta = 0.0f;
                for (int64_t w = 1; w <= window; ++w) {
                    int64_t tp = std::min(t + w, T - 1);
                    int64_t tn = std::max(t - w, (int64_t)0);
                    delta += (float)w * (feat[(b * D + f) * T + tp] - feat[(b * D + f) * T + tn]);
                }
                d[(b * D + f) * T + t] = delta / denom;
            }
        }
    }

    return deltas;
}

Tensor spec_augment(const Tensor& mel_spec, int64_t freq_mask_width,
                     int64_t time_mask_width, float prob) {
    int64_t B = mel_spec.dim(0);
    int64_t n_m = mel_spec.dim(1);
    int64_t T = mel_spec.dim(2);

    Tensor out = mel_spec.clone();
    float* o = out.data<float>();

    for (int64_t b = 0; b < B; ++b) {
        if ((float)std::rand() / (float)RAND_MAX < prob) {
            int64_t f0 = std::rand() % std::max<int64_t>(1, n_m - freq_mask_width);
            for (int64_t f = f0; f < std::min(n_m, f0 + freq_mask_width); ++f)
                for (int64_t t = 0; t < T; ++t)
                    o[(b * n_m + f) * T + t] = 0.0f;
        }

        if ((float)std::rand() / (float)RAND_MAX < prob) {
            int64_t t0 = std::rand() % std::max<int64_t>(1, T - time_mask_width);
            for (int64_t t = t0; t < std::min(T, t0 + time_mask_width); ++t)
                for (int64_t f = 0; f < n_m; ++f)
                    o[(b * n_m + f) * T + t] = 0.0f;
        }
    }

    return out;
}

Tensor voice_activity_detection(const Tensor& mel_spec, float threshold_db) {
    int64_t B = mel_spec.dim(0);
    int64_t n_m = mel_spec.dim(1);
    int64_t T = mel_spec.dim(2);

    Tensor vad({B, T});
    const float* mel = mel_spec.data<float>();
    float* v = vad.data<float>();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            float energy = 0.0f;
            for (int64_t m = 0; m < n_m; ++m)
                energy += mel[(b * n_m + m) * T + t];
            energy /= (float)n_m;
            float db = 10.0f * std::log10(std::max(1e-10f, energy));
            v[b * T + t] = (db > threshold_db) ? 1.0f : 0.0f;
        }
    }

    return vad;
}

// ============================================================================
// SpeechRecognizer additions
// ============================================================================

Tensor SpeechRecognizer::beam_search_decode(const Tensor& logits, int64_t beam_width) const {
    int64_t B = logits.dim(0);
    int64_t T = logits.dim(1);
    int64_t V = logits.dim(2);

    Tensor result({B, T});
    result.fill(-1);
    int64_t* r = result.data<int64_t>();
    const float* l = logits.data<float>();

    for (int64_t b = 0; b < B; ++b) {
        struct Beam {
            std::vector<int64_t> tokens;
            float score;
            int64_t prev_blank;
        };

        std::vector<Beam> beams = {{{}, 0.0f, -1}};

        for (int64_t t = 0; t < T; ++t) {
            std::vector<Beam> candidates;
            for (auto& beam : beams) {
                for (int64_t v = 0; v < V; ++v) {
                    float prob = l[(b * T + t) * V + v];
                    Beam new_beam = beam;
                    if (v != 0 || beam.prev_blank != 0) {
                        if (v != 0 && (new_beam.tokens.empty() || new_beam.tokens.back() != v))
                            new_beam.tokens.push_back(v);
                        new_beam.prev_blank = (v == 0) ? 0 : 1;
                    } else {
                        new_beam.prev_blank = 1;
                    }
                    new_beam.score += std::log(std::max(1e-10f, prob));
                    candidates.push_back(new_beam);
                }
            }

            std::sort(candidates.begin(), candidates.end(),
                      [](const Beam& a, const Beam& b) { return a.score > b.score; });

            beams.clear();
            for (int64_t i = 0; i < std::min((int64_t)candidates.size(), beam_width); ++i)
                beams.push_back(candidates[i]);
        }

        int64_t max_len = std::min((int64_t)beams[0].tokens.size(), T);
        for (int64_t i = 0; i < max_len; ++i)
            r[b * T + i] = beams[0].tokens[i];
    }

    return result;
}

float SpeechRecognizer::compute_ctc_loss(const Tensor& logits, const Tensor& targets,
                                          const Tensor& input_lengths,
                                          const Tensor& target_lengths) const {
    int64_t B = logits.dim(0);
    int64_t T = logits.dim(1);
    int64_t V = logits.dim(2);

    const float* l = logits.data<float>();
    const int64_t* tgt = targets.data<int64_t>();
    const int64_t* in_len = input_lengths.data<int64_t>();
    const int64_t* tgt_len = target_lengths.data<int64_t>();

    float total_loss = 0.0f;

    for (int64_t b = 0; b < B; ++b) {
        int64_t U = tgt_len[b];
        int64_t Ti = in_len[b];

        if (U == 0 || Ti == 0) continue;

        int64_t S = 2 * U + 1;
        std::vector<int64_t> labels_seq(S);
        labels_seq[0] = 0;
        for (int64_t u = 0; u < U; ++u) {
            labels_seq[2 * u + 1] = tgt[b * T + u];
            labels_seq[2 * u + 2] = 0;
        }

        std::vector<std::vector<float>> alpha(Ti, std::vector<float>(S, -1e30f));
        alpha[0][0] = l[(b * T + 0) * V + 0];
        if (S > 1)
            alpha[0][1] = l[(b * T + 0) * V + labels_seq[1]];

        for (int64_t t = 1; t < Ti; ++t) {
            for (int64_t s = 0; s < S; ++s) {
                float log_prob = l[(b * T + t) * V + labels_seq[s]];
                float best = alpha[t - 1][s];
                if (s > 0) best = std::max(best, alpha[t - 1][s - 1]);
                if (s > 1 && labels_seq[s] != labels_seq[s - 2])
                    best = std::max(best, alpha[t - 1][s - 2]);
                alpha[t][s] = best + log_prob;
            }
        }

        float loss = -alpha[Ti - 1][S - 1];
        if (S > 1)
            loss = -std::max(alpha[Ti - 1][S - 1], alpha[Ti - 1][S - 2]);

        total_loss += loss;
    }

    return total_loss / (float)B;
}

Tensor SpeechRecognizer::greedy_decode(const Tensor& logits) const {
    int64_t B = logits.dim(0);
    int64_t T = logits.dim(1);
    int64_t V = logits.dim(2);

    Tensor result({B, T});
    result.fill(-1);
    int64_t* r = result.data<int64_t>();
    const float* l = logits.data<float>();

    for (int64_t b = 0; b < B; ++b) {
        int64_t prev = -1;
        int64_t pos = 0;
        for (int64_t t = 0; t < T; ++t) {
            int64_t best = 0;
            float best_val = l[(b * T + t) * V];
            for (int64_t v = 1; v < V; ++v) {
                float val = l[(b * T + t) * V + v];
                if (val > best_val) { best_val = val; best = v; }
            }
            if (best != 0 && best != prev)
                r[b * T + pos++] = best;
            prev = (best != 0) ? best : -1;
        }
    }

    return result;
}

} // namespace multimodal
} // namespace oil
