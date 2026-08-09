#pragma once
#include "quant/tensor.h"
#include "quant/model.h"
#include <string>
#include <vector>
#include <algorithm>

namespace quant {
namespace agi {

inline std::vector<int> simple_encode(const std::string& text, int vocab_size) {
    std::vector<int> ids;
    int offset = 5;
    int mod = std::max(1, vocab_size - offset);
    for (char c : text) {
        ids.push_back((int)(unsigned char)c % mod + offset);
    }
    return ids;
}

inline std::string simple_decode(const std::vector<int>& ids) {
    std::string s;
    for (int id : ids) {
        int c = id - 5;
        if (c >= 0 && c < 256) s += (char)c;
        else s += '?';
    }
    return s;
}

inline int greedy_argmax(const float* logits, int n) {
    return (int)(std::max_element(logits, logits + n) - logits);
}

inline std::vector<int> generate_new_tokens(Model* model, const std::vector<int>& prompt_ids,
                                             int vocab_size, int max_new) {
    if (!model) return {};
    std::vector<int> all_ids = prompt_ids;
    int context = 64;

    for (int step = 0; step < max_new; step++) {
        int64_t len = (int64_t)all_ids.size();
        int64_t start = std::max((int64_t)0, len - context);
        int64_t ctx_len = len - start;

        Tensor input_ids({1, ctx_len});
        Tensor positions({1, ctx_len});
        float* idp = input_ids.data<float>();
        float* psp = positions.data<float>();
        for (int64_t i = 0; i < ctx_len; i++) {
            idp[i] = (float)all_ids[start + i];
            psp[i] = (float)(start + i);
        }

        Tensor logits = model->forward(input_ids, positions, nullptr);
        int64_t V = logits.dim(logits.rank() - 1);
        const float* lp = logits.data<float>();

        int next = greedy_argmax(lp + (ctx_len - 1) * V, (int)V);
        all_ids.push_back(next);
        if (next < 2) break;
    }

    if ((int64_t)all_ids.size() <= (int64_t)prompt_ids.size()) return {};
    return std::vector<int>(all_ids.begin() + (int64_t)prompt_ids.size(), all_ids.end());
}

inline std::string to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

} // namespace agi
} // namespace quant
