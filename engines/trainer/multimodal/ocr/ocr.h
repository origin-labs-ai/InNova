#pragma once
#include "quant/tensor.h"
#include "quant/types.h"
#include "quant/transformer.h"
#include <vector>
#include <string>
#include <utility>

namespace quant {
namespace multimodal {

struct OCRConfig {
    int64_t in_channels = 1;
    int64_t conv_channels[3] = {32, 64, 128};
    int64_t lstm_hidden = 256;
    int64_t num_classes = 128;
    int64_t max_width = 512;
    int64_t hidden_size = 512;
    int64_t num_heads = 8;
    int64_t num_transformer_layers = 4;
    int64_t max_seq_len = 256;
    float dropout = 0.1f;
};

class GridPositionalEncoding {
public:
    Tensor pe;
    int64_t max_h;
    int64_t max_w;
    int64_t dim;

    GridPositionalEncoding(int64_t max_h, int64_t max_w, int64_t dim);
    Tensor forward(int64_t height, int64_t width) const;
    Tensor forward_1d(int64_t seq_len) const;
};

class AttentionDecoder {
public:
    quant::Embedding embedding;
    std::vector<quant::TransformerBlock> decoder_blocks;
    quant::Linear output_proj;
    int64_t hidden_size;
    int64_t vocab_size;
    int64_t max_len;

    AttentionDecoder(int64_t hidden_size, int64_t vocab_size, int64_t num_layers, int64_t num_heads, int64_t max_len = 256);
    Tensor forward(const Tensor& encoder_out, const Tensor& targets, const Tensor& target_mask) const;
    std::string greedy_decode(const Tensor& encoder_out, const Tensor& pos_encoding, int bos_id, int eos_id) const;
    std::vector<int> beam_search_decode(const Tensor& encoder_out, const Tensor& pos_encoding, int bos_id, int eos_id, int beam_width = 5, int max_steps = 128) const;
};

class TransformerOCR {
public:
    Tensor input_conv;
    Tensor conv_bias;
    GridPositionalEncoding pos_enc;
    std::vector<quant::TransformerBlock> encoder_blocks;
    Linear output_proj;
    int64_t hidden_size;
    int64_t num_classes;
    int64_t patch_size;

    TransformerOCR(const OCRConfig& cfg = OCRConfig());
    Tensor forward(const Tensor& image) const;
    std::string recognize(const Tensor& image) const;
    std::vector<std::pair<int64_t, int64_t>> detect_text_regions(const Tensor& image) const;
};

class CRNN {
public:
    Tensor conv1_w, conv1_b;
    Tensor conv2_w, conv2_b;
    Tensor conv3_w, conv3_b;
    Tensor lstm_w_i, lstm_u_i, lstm_b_i;
    Tensor lstm_w_h, lstm_u_h, lstm_b_h;
    Tensor ctc_w, ctc_b;
    int64_t lstm_hidden;
    int64_t num_classes;
    int64_t patch_size = 3;
    int64_t pool_size = 2;

    explicit CRNN(const OCRConfig& cfg = OCRConfig());
    Tensor forward(const Tensor& patches) const;
    std::string recognize(const Tensor& text_image) const;
    std::vector<std::pair<int64_t, int64_t>> detect_text_region(const Tensor& image) const;

private:
    Tensor conv2d_matmul(const Tensor& input, const Tensor& weight,
                         const Tensor& bias, int stride) const;
    Tensor max_pool_stride(const Tensor& input, int window, int stride) const;
    Tensor bilstm_step(const Tensor& x_fwd, const Tensor& x_bwd,
                       const Tensor& h_fwd, const Tensor& h_bwd) const;
    std::string ctc_greedy_decode(const Tensor& logits) const;
    std::string ctc_beam_search_decode(const Tensor& logits, int beam_width = 5) const;
    void init_weight(Tensor& t, float scale) const;
};

} // namespace multimodal
} // namespace quant
