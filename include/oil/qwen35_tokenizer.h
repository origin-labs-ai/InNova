#pragma once
#include "oil/tokenizer.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace oil {

struct Qwen35Message {
    std::string role;
    std::string content;
    std::string reasoning_content;
};

class Qwen35Tokenizer : public Tokenizer {
public:
    bool load_from_dir(const std::string& model_dir);

    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& ids, bool skip_special) const;
    // Tokenizer interface decode (skip_special=true by default)
    std::string decode(const std::vector<int>& ids) override { return decode(ids, true); }

    std::vector<int> apply_chat_template(
        const std::vector<Qwen35Message>& messages, bool add_generation_prompt);

    int vocab_size() const override { return (int)id_to_token_.size(); }
    int bos_id() const override { return -1; }  // Qwen3.5 has no BOS token
    int eos_id() const override { return 248046; }
    int im_start_id() const { return 248045; }
    int im_end_id() const { return 248046; }
    bool is_special(int id) const;

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::pair<int, int>> merges_;
    std::vector<int> special_ids_;
    std::vector<std::string> special_tokens_;
    std::unordered_map<int, std::string> special_by_id_;
    std::vector<std::string> byte_chars_;

    static int utf8_decode(const std::string& s, size_t& pos);
    static void utf8_encode(std::string& out, int cp);
    static bool is_letter(int cp);
    static bool is_digit(int cp);
    static bool is_ws(int cp);

    void build_byte_map();
    std::vector<std::string> pretokenize(const std::string& text) const;
    std::vector<int> bpe_encode(const std::string& word) const;
    std::string token_to_bytes(const std::string& token) const;
};

} // namespace oil
