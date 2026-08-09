#include "quant/model.h"
#include "quant/transformer.h"
#include "quant/tokenizer.h"
#include "quant/trainer.h"
#include "quant/optimizer.h"

#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include <sstream>

struct TrainArgs {
    std::string model_path = "model.quant";
    std::string data_path = "data.txt";
    std::string config_path;
    int64_t batch_size = 8;
    int64_t seq_length = 512;
    int num_epochs = 3;
    float learning_rate = 3e-4f;
    int vocab_size = 32000;
    int hidden_size = 768;
    int num_layers = 12;
    int num_heads = 12;
    int log_interval = 10;
    int save_interval = 1000;
    std::string optimizer_name = "adafactor";
};

static TrainArgs parse_args(int argc, char** argv) {
    TrainArgs args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            args.model_path = argv[++i];
        else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc)
            args.data_path = argv[++i];
        else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc)
            args.config_path = argv[++i];
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc)
            args.batch_size = std::stoll(argv[++i]);
        else if (strcmp(argv[i], "--seq-length") == 0 && i + 1 < argc)
            args.seq_length = std::stoll(argv[++i]);
        else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc)
            args.num_epochs = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc)
            args.learning_rate = std::stof(argv[++i]);
        else if (strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc)
            args.vocab_size = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--hidden-size") == 0 && i + 1 < argc)
            args.hidden_size = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--num-layers") == 0 && i + 1 < argc)
            args.num_layers = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--num-heads") == 0 && i + 1 < argc)
            args.num_heads = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--log-interval") == 0 && i + 1 < argc)
            args.log_interval = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--save-interval") == 0 && i + 1 < argc)
            args.save_interval = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--optimizer") == 0 && i + 1 < argc)
            args.optimizer_name = argv[++i];
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: quant_train --model model.quant --data data.txt [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --batch-size N    Batch size (default: 8)\n";
            std::cout << "  --seq-length N    Sequence length (default: 512)\n";
            std::cout << "  --epochs N        Number of epochs (default: 3)\n";
            std::cout << "  --lr F            Learning rate (default: 3e-4)\n";
            std::cout << "  --vocab-size N    Vocabulary size (default: 32000)\n";
            std::cout << "  --hidden-size N   Hidden size (default: 768)\n";
            std::cout << "  --num-layers N    Number of layers (default: 12)\n";
            std::cout << "  --num-heads N     Number of heads (default: 12)\n";
            std::cout << "  --optimizer NAME  Optimizer: 'adafactor' or 'adamw' (default: adafactor)\n";
            exit(0);
        }
    }
    return args;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    std::cout << "=== QUANT Training ===\n";
    std::cout << "Model: " << args.model_path << "\n";
    std::cout << "Data: " << args.data_path << "\n";

    quant::TransformerConfig cfg;
    cfg.vocab_size = args.vocab_size;
    cfg.hidden_size = args.hidden_size;
    cfg.num_layers = args.num_layers;
    cfg.num_heads = args.num_heads;

    quant::DenseModel model(cfg);
    std::cout << "Model created: " << model.param_count() << " params\n";

    quant::BPETokenizer tokenizer;
    std::ifstream data_file(args.data_path);
    if (!data_file.is_open()) {
        std::cerr << "Error: cannot open " << args.data_path << std::endl;
        return 1;
    }
    std::stringstream ss;
    ss << data_file.rdbuf();
    std::string corpus = ss.str();

    std::vector<std::string> texts = {corpus};
    tokenizer.train(texts, static_cast<int>(cfg.vocab_size));

    quant::Trainer trainer(&model, &tokenizer);

    quant::Adafactor opt_adafactor(args.learning_rate);
    quant::AdamW opt_adamw(args.learning_rate);

    if (args.optimizer_name == "adamw") {
        std::cout << "Using AdamW optimizer.\n";
        trainer.compile(&opt_adamw);
    } else {
        std::cout << "Using Adafactor optimizer.\n";
        trainer.compile(&opt_adafactor);
    }

    quant::DataLoader dataloader(&tokenizer, args.data_path,
                               args.batch_size, args.seq_length);

    quant::TrainConfig train_cfg;
    train_cfg.batch_size = args.batch_size;
    train_cfg.seq_length = args.seq_length;
    train_cfg.num_epochs = args.num_epochs;
    train_cfg.learning_rate = args.learning_rate;
    train_cfg.log_interval = args.log_interval;
    train_cfg.save_interval = args.save_interval;
    train_cfg.output_path = args.model_path;

    trainer.set_log_callback([](const quant::TrainMetrics& m) {
        std::cout << "Step " << m.step
                  << " | loss: " << m.loss
                  << " | ppl: " << m.perplexity
                  << " | lr: " << m.learning_rate
                  << " | tok/s: " << m.tokens_per_sec
                  << std::endl;
    });

    trainer.fit(dataloader, train_cfg);
    std::cout << "Training complete. Model saved to " << args.model_path << std::endl;
    return 0;
}
