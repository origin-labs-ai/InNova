#include "quant/quant_format.h"
#include "quant/types.h"

#include <iostream>
#include <string>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: quant_info <model.quant>" << std::endl;
        return 1;
    }

    try {
        quant::QUANTReader reader(argv[1]);
        auto& hdr = reader.header();

        std::cout << "QUANT Model: " << argv[1] << std::endl;
        std::cout << "Magic: " << std::string(hdr.magic, 4) << std::endl;
        std::cout << "Version: " << hdr.version << std::endl;
        std::cout << "Flags: 0x" << std::hex << hdr.flags << std::dec << std::endl;
        std::cout << "Config size: " << hdr.config_size << " bytes" << std::endl;

        auto names = reader.tensor_names();
        std::cout << "Tensors: " << names.size() << std::endl;

        int64_t total_params = 0;
        for (size_t i = 0; i < names.size(); i++) {
            auto& n = names[i];
            try {
                quant::Tensor t = reader.read_tensor(n);
                int64_t numel = t.numel();
                total_params += numel;

                auto formats = reader.tensor_formats(n);
                std::cout << "  [" << i << "] " << n
                          << " shape=" << t.shape().to_string()
                          << " params=" << numel;
                if (!formats.empty()) {
                    std::cout << " format=" << quant::format_name(formats[0]);
                    if (formats.size() > 1)
                        std::cout << " (+" << (formats.size() - 1) << " blocks)";
                }
                std::cout << std::endl;
            } catch (const std::exception& e) {
                std::cout << "  [" << i << "] " << n << " (error: " << e.what() << ")" << std::endl;
            }
        }

        std::cout << "Total parameters: " << total_params << std::endl;

        auto ft = reader.read_format_table();
        if (!ft.empty()) {
            std::cout << "Format table:" << std::endl;
            for (auto& entry : ft) {
                std::cout << "  Block " << entry.block_id
                          << ": format=" << (int)entry.format
                          << " (" << quant::format_name((quant::Format)entry.format) << ")"
                          << ", cb_bytes=" << entry.cb_bytes << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
