#include "inference.h"
#include <cctype>
#include <memory>

namespace quant {
namespace engine {

namespace {

struct StreamBuffer {
    std::string buffer;
    void flush(std::function<void(const std::string&)> on_token) {
        if (!buffer.empty()) {
            on_token(buffer);
            buffer.clear();
        }
    }
    void append(const std::string& token) {
        buffer += token;
    }
    bool should_flush() const {
        if (buffer.empty()) return false;
        unsigned char c = static_cast<unsigned char>(buffer.back());
        if (c == ' ' || c == '\n' || c == '\t') return true;
        if (c == '.' || c == ',' || c == '!' || c == '?' || c == ';' || c == ':') return true;
        if (c == ')' || c == '}' || c == ']') return true;
        if (c == '\"' || c == '\'' || c == '\r') return true;
        return false;
    }
};

} // anonymous namespace

void InferenceEngine::generate_stream(
    const std::string& prompt,
    std::function<void(const std::string&)> on_token)
{
    if (!generator_) return;

    auto sb = std::make_shared<StreamBuffer>();
    auto buffered = [sb, on_token = std::move(on_token)](const std::string& token) mutable {
        sb->append(token);
        if (sb->should_flush())
            sb->flush(on_token);
    };

    generator_->generate_stream(prompt, config_.sampler, std::move(buffered));
}

} // namespace engine
} // namespace quant
