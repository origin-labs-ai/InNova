#include "trainer.h"
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <filesystem>

namespace quant {
namespace dense {

DataLoader::DataLoader(Tokenizer* tokenizer, const std::string& data_path,
                       int64_t batch_size, int64_t seq_length)
    : tokenizer_(tokenizer), batch_size_(batch_size), seq_length_(seq_length)
{
    std::filesystem::path path(data_path);
    path = std::filesystem::absolute(path);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw Error("DataLoader: cannot open " + path.string());
    size_t size = file.tellg();
    file.seekg(0);
    std::string text(size, '\0');
    file.read(text.data(), size);

    std::string clean;
    clean.reserve(text.size());
    for (char c : text) {
        if (c != '\r')
            clean.push_back(c);
    }

    data_ = tokenizer_->encode(clean);
    if (data_.empty())
        throw Error("DataLoader: tokenized data is empty");

    int64_t chunk = batch_size_ * seq_length_;
    num_batches_ = (int64_t)data_.size() / chunk;
}

bool DataLoader::next_batch(Tensor& input_ids, Tensor& labels) {
    int64_t needed = batch_size_ * seq_length_ + 1;
    if (pos_ + needed > (int64_t)data_.size())
        return false;

    int64_t num_tokens = batch_size_ * seq_length_;

    input_ids = Tensor(Shape(batch_size_, seq_length_));
    labels = Tensor(Shape(batch_size_, seq_length_));

    float* inp = input_ids.data<float>();
    float* lbl = labels.data<float>();

    for (int64_t i = 0; i < num_tokens; ++i) {
        inp[i] = (float)data_[pos_ + i];
        lbl[i] = (float)data_[pos_ + i + 1];
    }

    pos_ += num_tokens;
    return true;
}

void DataLoader::shuffle() {
    int64_t chunk = batch_size_ * seq_length_;
    int64_t num_chunks = (int64_t)data_.size() / chunk;
    if (num_chunks < 2) return;

    std::vector<int64_t> indices(num_chunks);
    for (int64_t i = 0; i < num_chunks; ++i)
        indices[i] = i * chunk;

    unsigned seed = (unsigned)std::chrono::steady_clock::now().time_since_epoch().count();
    std::shuffle(indices.begin(), indices.end(), std::mt19937(seed));

    std::vector<int> shuffled(data_.size());
    for (int64_t i = 0; i < num_chunks; ++i) {
        std::copy(data_.begin() + indices[i],
                  data_.begin() + indices[i] + chunk,
                  shuffled.begin() + i * chunk);
    }
    std::copy(data_.begin() + num_chunks * chunk, data_.end(),
              shuffled.begin() + num_chunks * chunk);
    data_ = std::move(shuffled);
    pos_ = 0;
}

void DataLoader::reset() {
    pos_ = 0;
}

int64_t DataLoader::num_batches() const {
    return num_batches_;
}

} // namespace dense
} // namespace quant
