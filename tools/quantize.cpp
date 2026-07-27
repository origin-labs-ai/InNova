#include "oil/oil_format.h"
#include "oil/oil_engines.h"
#include "oil/codebook.h"
#include "oil/kernel.h"
#include "oil/types.h"
#include "oil/tensor.h"

#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>

using namespace oil;

static Format parse_format(const std::string& s) {
    if (s == "fp32" || s == "FP32") return Format::OIL32;
    if (s == "fp16" || s == "FP16") return Format::OIL16;
    if (s == "oil1" || s == "OIL1") return Format::OIL1;
    if (s == "spark_q0" || s == "SPARK_Q0") return Format::SPARK_Q0;
    if (s == "oil4" || s == "OIL4") return Format::OIL4;
    if (s == "oil8" || s == "OIL8") return Format::OIL8;
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
            std::cout << "                     Formats: fp32, fp16, oil1, spark, oil4, oil8\n";
            std::cout << "  --per-layer <csv>  Per-layer formats (name=fmt,name=fmt,...)\n";
            std::cout << "  --num-bits N       Number of bits (default: 8)\n";
            exit(0);
        }
    }
    return args;
}

static void quantize_block_oil8(const float* data, int64_t block_size,
                                 std::vector<uint8_t>& indices,
                                 std::vector<uint8_t>& codebook) {
    CodebookOIL8 cb;
    cb.train(data, (size_t)block_size);
    codebook.resize(cb.serialized_size());
    cb.serialize(codebook.data());
    indices.resize((size_t)block_size);
    for (int64_t i = 0; i < block_size; i++)
        indices[(size_t)i] = cb.quantize(data[(size_t)i]);
}

static void quantize_block_oil4(const float* data, int64_t block_size,
                                 std::vector<uint8_t>& indices,
                                 std::vector<uint8_t>& codebook) {
    CodebookOIL4 cb;
    cb.train(data, (size_t)block_size);
    codebook.resize(cb.serialized_size());
    cb.serialize(codebook.data());
    indices.resize((size_t)block_size);
    for (int64_t i = 0; i < block_size; i++)
        indices[(size_t)i] = cb.quantize(data[(size_t)i]);
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

        FormatBlockEntry fbe;
        fbe.block_id = block_id;
        fbe.format = (uint8_t)fmt;
        fbe.cb_bytes = (fmt == Format::OIL8 || fmt == Format::OIL4) ? 256 * 2 : 0;

        const float* data = tensor.data<float>();
        int64_t total_weights = tensor.numel();

        for (int64_t b = 0; b < num_blocks; b++) {
            int64_t block_start = b * 256;
            int64_t block_end = std::min(block_start + 256, total_weights);
            int64_t block_size = block_end - block_start;

            BlockData block;
            block.format = fmt;
            block.num_weights = (uint32_t)block_size;

            switch (fmt) {
            case Format::OIL8:
                quantize_block_oil8(data + block_start, block_size,
                                    block.indices, block.codebook);
                break;
            case Format::OIL4:
                quantize_block_oil4(data + block_start, block_size,
                                    block.indices, block.codebook);
                break;
            case Format::OIL32:
                block.format = Format::OIL32;
                block.indices.resize((size_t)block_size * sizeof(float));
                std::memcpy(block.indices.data(), data + block_start,
                           (size_t)block_size * sizeof(float));
                break;
            case Format::OIL16:
                block.format = Format::OIL16;
                block.indices.resize((size_t)block_size * 2);
                {
                    const float* src = data + block_start;
                    uint16_t* dst = (uint16_t*)block.indices.data();
                    for (int64_t i = 0; i < block_size; i++) {
                        int sign = src[i] < 0 ? 1 : 0;
                        float v = src[i] < 0 ? -src[i] : src[i];
                        int e = 0;
                        float m = v;
                        if (v > 0) {
                            e = 15;
                            while (m < 1.0f) { m *= 2.0f; e--; }
                            while (m >= 2.0f) { m /= 2.0f; e++; }
                            m -= 1.0f;
                        }
                        uint16_t f16 = (uint16_t)((sign << 15) | ((e + 15) << 10) | (uint16_t)(m * 1024));
                        dst[i] = f16;
                    }
                }
                break;
            default:
                block.format = Format::OIL32;
                block.indices.resize((size_t)block_size * sizeof(float));
                std::memcpy(block.indices.data(), data + block_start,
                           (size_t)block_size * sizeof(float));
                break;
            }

            writer.write_block(block);
            block_id++;
        }

        TensorEntry te;
        te.name_len = (uint16_t)name.size();
        te.block_start = block_id - (uint32_t)num_blocks;
        te.num_blocks = (uint32_t)num_blocks;
        tensor_entries.push_back(te);
        names.push_back(name);

        size_t mb = (tensor.size_bytes() + 1048575) / 1048576;
        size_t qmb = ((size_t)num_blocks * 256 + 1048575) / 1048576;
        std::cout << "  " << name << ": " << mb << "MB -> " << qmb << "MB ("
                  << args.format << ")\n";

        for (uint32_t b = 0; b < (uint32_t)num_blocks; b++) {
            FormatBlockEntry entry;
            entry.block_id = fbe.block_id + b;
            entry.format = fbe.format;
            entry.cb_bytes = fbe.cb_bytes;
            ft_entries.push_back(entry);
        }
    }

    writer.write_format_table(ft_entries);
    writer.write_tensor_table(tensor_entries, names);
    writer.close();

    std::cout << "Quantization complete. Output: " << args.output_path << "\n";
    return 0;
}
