#include "oil/oil_format.h"
#include "oil/oil_engines.h"
#include "oil/codebook.h"
#include "oil/kernel.h"
#include "oil/types.h"
#include "oil/tensor.h"
#include "oil/block_codec.h"

#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>

using namespace oil;

static Format parse_format(const std::string& s) {
    if (s == "fp32" || s == "FP32" || s == "oil32" || s == "OIL32") return Format::OIL32;
    if (s == "fp16" || s == "FP16" || s == "oil16" || s == "OIL16") return Format::OIL16;
    if (s == "oil1" || s == "OIL1") return Format::OIL1;
    if (s == "oil1_grp" || s == "OIL1_GRP") return Format::OIL1_GRP;
    if (s == "oil2" || s == "OIL2") return Format::OIL2;
    if (s == "oil2_grp" || s == "OIL2_GRP") return Format::OIL2_GRP;
    if (s == "oil4" || s == "OIL4") return Format::OIL4;
    if (s == "oil4_grp" || s == "OIL4_GRP") return Format::OIL4_GRP;
    if (s == "oil8" || s == "OIL8") return Format::OIL8;
    if (s == "oil8_grp" || s == "OIL8_GRP") return Format::OIL8_GRP;
    if (s == "spark" || s == "SPARK" || s == "spark_q0" || s == "SPARK_Q0") return Format::SPARK_Q0;
    if (s == "spark_q0_grp" || s == "SPARK_Q0_GRP") return Format::SPARK_Q0_GRP;
    if (s == "spark_sparse" || s == "SPARK_SPARSE") return Format::SPARK_SPARSE;
    if (s == "spark_sparse_grp" || s == "SPARK_SPARSE_GRP") return Format::SPARK_SPARSE_GRP;
    std::cerr << "Warning: unknown format '" << s << "' — defaulting to oil8\n";
    return Format::OIL8;
}

static Format s_default_format = Format::OIL8;

struct QuantArgs {
    std::string input_path;
    std::string output_path;
    std::string format = "oil8";
    std::string per_layer_format;
    int num_bits = 8;
};

static QuantArgs parse_args(int argc, char** argv) {
    QuantArgs args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            args.input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            args.output_path = argv[++i];
        else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            args.format = argv[++i];
            s_default_format = parse_format(args.format);
        } else if (strcmp(argv[i], "--per-layer") == 0 && i + 1 < argc)
            args.per_layer_format = argv[++i];
        else if (strcmp(argv[i], "--num-bits") == 0 && i + 1 < argc)
            args.num_bits = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: oil_quantize --input model.oil --output quantized.oil [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --format <f>       Target format (default: oil8)\n";
            std::cout << "                     Formats: fp32, fp16, oil1, oil1_grp, oil2,\n";
            std::cout << "                     oil2_grp, oil4, oil4_grp, oil8, oil8_grp,\n";
            std::cout << "                     spark (1.50 bpw), spark_q0_grp,\n";
            std::cout << "                     spark_sparse, spark_sparse_grp\n";
            std::cout << "  --per-layer <csv>  Per-layer formats (name=fmt,name=fmt,...)\n";
            std::cout << "  --num-bits N       Number of bits (default: 8)\n";
            exit(0);
        }
    }
    return args;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    if (args.input_path.empty() || args.output_path.empty()) {
        std::cerr << "Error: --input and --output required\n";
        return 1;
    }

    std::cout << "OIL Quantization Tool\n";
    std::cout << "Input: " << args.input_path << "\n";
    std::cout << "Output: " << args.output_path << "\n";
    std::cout << "Format: " << args.format << "\n";

    std::unordered_map<std::string, Format> per_layer;
    if (!args.per_layer_format.empty()) {
        std::stringstream ss(args.per_layer_format);
        std::string item;
        while (std::getline(ss, item, ',')) {
            auto eq = item.find('=');
            if (eq != std::string::npos) {
                std::string name = item.substr(0, eq);
                std::string fmt = item.substr(eq + 1);
                per_layer[name] = parse_format(fmt);
                std::cout << "  Layer '" << name << "' -> " << fmt << "\n";
            }
        }
    }

    OILReader reader(args.input_path);
    if (!reader.valid()) {
        std::cerr << "Error: cannot open " << args.input_path << "\n";
        return 1;
    }

    auto tensor_names = reader.tensor_names();
    std::cout << "Found " << tensor_names.size() << " tensors\n";

    OILWriter writer(args.output_path);
    OILHeader hdr;
    std::memcpy(hdr.magic, "OIL1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = 0;
    writer.write_header(hdr, nullptr);

    std::vector<FormatBlockEntry> ft_entries;
    std::vector<TensorEntry> tensor_entries;
    std::vector<std::string> names;
    uint32_t block_id = 0;

    for (const auto& name : tensor_names) {
        Tensor tensor = reader.read_tensor(name);
        if (tensor.numel() == 0) continue;

        Format fmt = s_default_format;
        auto it = per_layer.find(name);
        if (it != per_layer.end())
            fmt = it->second;

        int64_t num_blocks = (tensor.numel() + 255) / 256;

        const float* data = tensor.data<float>();
        int64_t total_weights = tensor.numel();

        // Real stored bytes produced by the canonical codec for this tensor.
        size_t qbytes = 0;

        for (int64_t b = 0; b < num_blocks; b++) {
            int64_t block_start = b * 256;
            int64_t block_end = std::min(block_start + 256, total_weights);
            int64_t block_size = block_end - block_start;

            BlockData block;
            block.format = fmt;
            block.num_weights = (uint32_t)block_size;

            // Encode with the CANONICAL block codec — the single source of
            // truth for on-disk payloads. This keeps the tool's output
            // byte-identical in layout to everything the reader decodes
            // (quantize_block_all/dequantize_block_all), for every format.
            if (!quantize_block_all(fmt, data + block_start, (int)block_size,
                                    block.indices, block.codebook)) {
                std::cerr << "Warning: " << name << " block " << b
                          << ": format unsupported by canonical codec, "
                             "writing raw OIL32\n";
                block.format = Format::OIL32;
                quantize_block_all(Format::OIL32, data + block_start,
                                   (int)block_size, block.indices, block.codebook);
            }

            writer.write_block(block);
            qbytes += block.indices.size() + block.codebook.size();

            FormatBlockEntry entry;
            entry.block_id = block_id;
            entry.format = (uint8_t)block.format;
            entry.cb_bytes = (uint32_t)block.codebook.size();
            ft_entries.push_back(entry);

            block_id++;
        }

        TensorEntry te;
        te.name_len = (uint16_t)name.size();
        te.block_start = block_id - (uint32_t)num_blocks;
        te.num_blocks = (uint32_t)num_blocks;
        tensor_entries.push_back(te);
        names.push_back(name);

        size_t mb = (tensor.size_bytes() + 1048575) / 1048576;
        size_t qmb = (qbytes + 1048575) / 1048576;
        std::cout << "  " << name << ": " << mb << "MB -> " << qmb << "MB ("
                  << args.format << ")\n";
    }

    writer.write_format_table(ft_entries);
    writer.write_tensor_table(tensor_entries, names);
    writer.close();

    std::cout << "Quantization complete. Output: " << args.output_path << "\n";
    return 0;
}
