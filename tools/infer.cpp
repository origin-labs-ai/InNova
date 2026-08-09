#include "quant/model.h"
#include "quant/qwen35_tokenizer.h"
#include "quant/generator.h"
#include "inference.h"

#include <iostream>
#include <string>
#include <filesystem>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: quant_infer <model.quant> [prompt] [--model-dir DIR]" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];
    std::string prompt = argc > 2 ? argv[2] : "Hello, ";

    // Parse optional --model-dir (defaults to parent of model.quant)
    std::string model_dir;
    for (int i = 3; i < argc; i++) {
        if (std::strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        }
    }
    if (model_dir.empty()) {
        model_dir = std::filesystem::path(model_path).parent_path().string();
    }

    quant::DenseModel model;
    try {
        model.load(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }

    quant::Qwen35Tokenizer tokenizer;
    try {
        if (!tokenizer.load_from_dir(model_dir)) {
            std::cerr << "Error: failed to load tokenizer from " << model_dir << std::endl;
            return 1;
        }
        std::cerr << "Tokenizer loaded (" << tokenizer.vocab_size() << " vocab)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading tokenizer: " << e.what() << std::endl;
        return 1;
    }

    quant::Generator gen(&model, &tokenizer);

    quant::SamplerConfig cfg;
    cfg.temperature = 0.7f;
    cfg.top_k = 40;
    cfg.top_p = 0.9f;
    cfg.max_tokens = 512;

    auto result = gen.generate_full(prompt, cfg);
    std::cout << result.text << std::endl;
    std::cerr << "Generated " << result.tokens_per_sec << " tok/s"
              << " (" << result.duration_sec << "s)"
              << std::endl;

    return 0;
}
