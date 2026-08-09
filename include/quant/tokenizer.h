#pragma once
#include "quant/types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <map>

namespace quant {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    virtual std::vector<int> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<int>& ids) = 0;
    virtual int vocab_size() const = 0;
    virtual int bos_id() const = 0;
    virtual int eos_id() const = 0;
};

class BPETokenizer : public Tokenizer {
public:
    BPETokenizer();
    
    // Train from text files
    void train(const std::vector<std::string>& texts, int vocab_size = 32000);
    
    // Encode text to token IDs
    std::vector<int> encode(const std::string& text) override;
    
    // Decode token IDs back to text
    std::string decode(const std::vector<int>& ids) override;
    
    int vocab_size() const override { return (int)vocab_.size(); }
    int bos_id() const override { return bos_id_; }
    int eos_id() const override { return eos_id_; }
    
    // Save/load tokenizer
    void save(const std::string& path) const;
    void load(const std::string& path);
    
private:
    struct MergePair {
        int id;
        int left, right;
        int freq;
    };
    
    std::vector<std::string> vocab_;
    std::map<std::pair<int, int>, int> merges_;
    int bos_id_ = 1;
    int eos_id_ = 2;
    int unk_id_ = 0;
    
    std::vector<int> bpe_merge(const std::vector<int>& ids) const;
    void add_token(const std::string& token);
    int get_token_id(const std::string& token) const;
};

} // namespace quant
