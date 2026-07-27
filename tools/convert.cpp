#include "oil/oil_format.h"
#include "oil/tensor.h"
#include "oil/types.h"
#include "oil/format_planner.h"
#include "oil/codebook.h"

#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <vector>
#include <cstdint>
#include <map>
#include <algorithm>
#include <cmath>

struct ConvertArgs {
    std::string input_path;
    std::string output_path;
    std::string input_fmt = "rawfp32";
    float target_bpw = 0.0f; // 0 = no compression (FP32 passthrough)
    bool verbose = false;
};

static ConvertArgs parse_args(int argc, char** argv) {
    ConvertArgs args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            args.input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            args.output_path = argv[++i];
        else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc)
            args.input_fmt = argv[++i];
        else if (strcmp(argv[i], "--bpw") == 0 && i + 1 < argc)
            args.target_bpw = (float)std::atof(argv[++i]);
        else if (strcmp(argv[i], "--verbose") == 0)
            args.verbose = true;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: oil_convert --input <file> --output <model.oil> [options]\n";
            std::cout << "  --format fmt   Input format: rawfp32 (default), gguf\n";
            std::cout << "  --bpw N        Target bits-per-weight for FormatPlanner (0=no compression)\n";
            std::cout << "  --verbose      Verbose output\n";
            std::cout << "\nFormats: 1.0(oil1) 1.58(spark) 4.0(oil4) 8.0(oil8) 16.0(fp16) 32.0(fp32)\n";
            exit(0);
        }
    }
    return args;
}

// GGUF v1/v2/v3 header parsing and Q4_0/Q4_1/Q8_0/F16 dequantization
struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t ggml_type;
    uint64_t offset;
};

static const uint32_t GGML_TYPE_Q4_0 = 2;
static const uint32_t GGML_TYPE_Q4_1 = 3;
static const uint32_t GGML_TYPE_Q8_0 = 8;
static const uint32_t GGML_TYPE_F16  = 1;
static const uint32_t GGML_TYPE_F32  = 0;

static uint64_t read_le_u64(std::istream& is) {
    uint64_t v = 0;
    is.read(reinterpret_cast<char*>(&v), 8);
    return v;
}
static uint32_t read_le_u32(std::istream& is) {
    uint32_t v = 0;
    is.read(reinterpret_cast<char*>(&v), 4);
    return v;
}
[[maybe_unused]] static uint16_t read_le_u16(std::istream& is) {
    uint16_t v = 0;
    is.read(reinterpret_cast<char*>(&v), 2);
    return v;
}
[[maybe_unused]] static uint8_t read_u8(std::istream& is) {
    uint8_t v = 0;
    is.read(reinterpret_cast<char*>(&v), 1);
    return v;
}
static std::string read_string(std::istream& is) {
    uint64_t len = read_le_u64(is);
    std::string s(len, '\0');
    if (len > 0) is.read(&s[0], (std::streamsize)len);
    return s;
}

// Dequantization helpers
static void dequantize_q4_0(const uint8_t* block, float* out, int offset) {
    // Q4_0 block: 1x fp16 scale (2 bytes) + 32x 4-bit values (16 bytes) = 18 bytes
    int16_t scale_half = *(const int16_t*)(block);
    float scale = (float)scale_half;
    // Convert from IEEE 754 half to float manually
    if (scale_half) {
        int sign = (scale_half >> 15) & 1;
        int exp  = (scale_half >> 10) & 0x1F;
        int mant = scale_half & 0x3FF;
        if (exp == 0) {
            // subnormal
            scale = (float)(mant) / 16384.0f * 2.0f;
            if (sign) scale = -scale;
        } else if (exp == 31) {
            scale = mant ? NAN : INFINITY;
        } else {
            scale = (float)((mant | 0x400) << 13) / 8388608.0f * (1 << (exp - 15));
            if (sign) scale = -scale;
        }
    }
    for (int i = 0; i < 32; i++) {
        uint8_t q = block[2 + i / 2];
        if (i % 2 == 0) q &= 0x0F;
        else            q >>= 4;
        out[offset + i] = ((float)q - 8.0f) * scale;
    }
}

static void dequantize_q4_1(const uint8_t* block, float* out, int offset) {
    // Q4_1 block: 1x fp16 scale (2 bytes) + 1x fp16 min (2 bytes) + 32x 4-bit values (16 bytes) = 20 bytes
    int16_t scale_half = *(const int16_t*)(block);
    int16_t min_half   = *(const int16_t*)(block + 2);
    auto half_to_float = [](int16_t h) -> float {
        int sign = (h >> 15) & 1;
        int exp  = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0) {
            float v = (float)(mant) / 16384.0f * 2.0f;
            return sign ? -v : v;
        } else if (exp == 31) {
            return mant ? NAN : INFINITY;
        } else {
            float v = (float)((mant | 0x400) << 13) / 8388608.0f * (1 << (exp - 15));
            return sign ? -v : v;
        }
    };
    float scale = half_to_float(scale_half);
    float min   = half_to_float(min_half);
    for (int i = 0; i < 32; i++) {
        uint8_t q = block[4 + i / 2];
        if (i % 2 == 0) q &= 0x0F;
        else            q >>= 4;
        out[offset + i] = (float)q * scale + min;
    }
}

static void dequantize_q8_0(const uint8_t* block, float* out, int offset) {
    // Q8_0 block: 1x fp16 scale (2 bytes) + 32x int8 values (32 bytes) = 34 bytes
    int16_t scale_half = *(const int16_t*)(block);
    auto half_to_float = [](int16_t h) -> float {
        int sign = (h >> 15) & 1;
        int exp  = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0) {
            float v = (float)(mant) / 16384.0f * 2.0f;
            return sign ? -v : v;
        } else if (exp == 31) {
            return mant ? NAN : INFINITY;
        } else {
            float v = (float)((mant | 0x400) << 13) / 8388608.0f * (1 << (exp - 15));
            return sign ? -v : v;
        }
    };
    float scale = half_to_float(scale_half);
    for (int i = 0; i < 32; i++) {
        int8_t q = (int8_t)block[2 + i];
        out[offset + i] = (float)q * scale;
    }
}

// Read GGUF model from file and convert to FP32
static bool read_gguf(const std::string& path, std::vector<float>& all_weights,
                      std::vector<int64_t>& all_shapes, std::vector<std::string>& all_names,
                      bool verbose) {
    std::ifstream is(path, std::ios::binary);
    if (!is) { std::cerr << "Cannot open " << path << "\n"; return false; }

    // Read magic: "GGUF" v1/v2/v3
    char magic[4];
    is.read(magic, 4);
    if (memcmp(magic, "GGUF", 4) != 0) {
        std::cerr << "Not a GGUF file (magic: " << magic[0] << magic[1] << magic[2] << magic[3] << ")\n";
        return false;
    }
    uint32_t version = read_le_u32(is);
    if (version < 1 || version > 3) {
        std::cerr << "Unsupported GGUF version: " << version << "\n";
        return false;
    }
    uint64_t n_tensors   = read_le_u64(is);
    uint64_t n_kv        = read_le_u64(is);
    if (verbose)
        std::cout << "GGUF v" << version << " tensors=" << n_tensors << " metadata=" << n_kv << "\n";

    // Skip metadata key-value pairs
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = read_string(is);
        uint32_t val_type = read_le_u32(is);
        switch (val_type) {
            case 0: is.ignore(1); break;                             // uint8
            case 1: is.ignore(8); break;                             // int8
            case 2: is.ignore(4); break;                             // uint16
            case 3: is.ignore(2); break;                             // int16
            case 4: is.ignore(8); break;                             // uint32
            case 5: is.ignore(4); break;                             // int32
            case 6: is.ignore(8); break;                             // float32
            case 7: is.ignore(4); break;                             // bool
            case 8: read_string(is); break;                          // string
            case 9: {                                                // array
                uint32_t atype = read_le_u32(is);
                uint64_t alen  = read_le_u64(is);
                for (uint64_t j = 0; j < alen; j++) {
                    switch (atype) {
                        case 8: read_string(is); break;
                        default: is.ignore(4); break;
                    }
                }
                break;
            }
            case 10: is.ignore(8); break;                            // uint64
            case 11: is.ignore(8); break;                            // int64
            case 12: is.ignore(8); break;                            // float64
            default: is.ignore(4); break;
        }
    }

    // Read tensor info
    struct GGUFTensorInfo info;
    std::vector<GGUFTensorInfo> tensor_infos;
    tensor_infos.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        info.name = read_string(is);
        info.n_dims = read_le_u32(is);
        info.ggml_type = read_le_u32(is);
        info.offset = read_le_u64(is);
        // v3 adds file offset alignment
        if (version >= 3)
            info.offset = read_le_u64(is);
        for (uint32_t d = 0; d < info.n_dims; d++)
            info.ne[d] = read_le_u64(is);
        for (uint32_t d = info.n_dims; d < 4; d++)
            info.ne[d] = 1;
        tensor_infos.push_back(info);
    }

    // Read and dequantize each tensor
    for (auto& ti : tensor_infos) {
        int64_t num_el = 1;
        for (uint32_t d = 0; d < ti.n_dims; d++)
            num_el *= (int64_t)ti.ne[d];

        all_names.push_back(ti.name);
        all_shapes.push_back(num_el);
        size_t start = all_weights.size();
        all_weights.resize(start + num_el);

        is.seekg(ti.offset, std::ios::beg);
        if (verbose)
            std::cout << "  tensor " << ti.name << " type=" << ti.ggml_type << " n_el=" << num_el << " offset=" << ti.offset << "\n";

        if (ti.ggml_type == GGML_TYPE_F32) {
            // Direct FP32 read
            is.read(reinterpret_cast<char*>(&all_weights[start]), (std::streamsize)(num_el * 4));
        } else if (ti.ggml_type == GGML_TYPE_F16) {
            for (int64_t j = 0; j < num_el; j++) {
                int16_t h = 0;
                is.read(reinterpret_cast<char*>(&h), 2);
                int sign = (h >> 15) & 1;
                int exp  = (h >> 10) & 0x1F;
                int mant = h & 0x3FF;
                float f;
                if (exp == 0)
                    f = (float)(mant) / 16384.0f * 2.0f;
                else if (exp == 31)
                    f = mant ? NAN : INFINITY;
                else
                    f = (float)((mant | 0x400) << 13) / 8388608.0f * (1 << (exp - 15));
                all_weights[start + j] = sign ? -f : f;
            }
        } else if (ti.ggml_type == GGML_TYPE_Q4_0) {
            int64_t n_blocks = (num_el + 31) / 32;
            for (int64_t b = 0; b < n_blocks; b++) {
                uint8_t block[18];
                is.read(reinterpret_cast<char*>(block), 18);
                dequantize_q4_0(block, all_weights.data(), (int)(start + b * 32));
            }
        } else if (ti.ggml_type == GGML_TYPE_Q4_1) {
            int64_t n_blocks = (num_el + 31) / 32;
            for (int64_t b = 0; b < n_blocks; b++) {
                uint8_t block[20];
                is.read(reinterpret_cast<char*>(block), 20);
                dequantize_q4_1(block, all_weights.data(), (int)(start + b * 32));
            }
        } else if (ti.ggml_type == GGML_TYPE_Q8_0) {
            int64_t n_blocks = (num_el + 31) / 32;
            for (int64_t b = 0; b < n_blocks; b++) {
                uint8_t block[34];
                is.read(reinterpret_cast<char*>(block), 34);
                dequantize_q8_0(block, all_weights.data(), (int)(start + b * 32));
            }
        } else {
            std::cerr << "Unsupported GGML type " << ti.ggml_type << " for tensor " << ti.name << "\n";
            return false;
        }
    }
    return true;
}

// Apply FormatPlanner to compress weights to target BPW
static bool apply_format_plan(const std::vector<float>& f32_data, size_t num_weights,
                               float target_bpw, oil::FormatPlan& plan,
                               std::vector<uint8_t>& compressed_indices,
                               std::vector<uint8_t>& codebook_data) {
    if (target_bpw <= 0.0f) return false;
    if (num_weights == 0) return false;

    oil::FormatPlanner planner(target_bpw);
    int64_t num_blocks = (int64_t)((num_weights + 255) / 256);
    plan = planner.plan_for_model(num_blocks * 256);

    oil::Format actual_format = oil::Format::OIL32;
    for (auto& blk : plan.blocks) {
        actual_format = blk.assigned_format;
        break;
    }

    if (actual_format == oil::Format::SPARK_Q0) {
        // SPARK: 2 bits per weight, pack 4 per byte
        size_t packed_size = (num_weights + 3) / 4;
        compressed_indices.resize(packed_size, 0);
        codebook_data.resize(3 * 4); // {-1, 0, +1} stored as FP32
        for (int i = 0; i < 3; i++) {
            float val = (float)(i - 1);
            memcpy(&codebook_data[i * 4], &val, 4);
        }
        for (size_t i = 0; i < num_weights; i++) {
            int8_t q = (f32_data[i] > 0.5f) ? 1 : ((f32_data[i] < -0.5f) ? -1 : 0);
            uint8_t packed = (uint8_t)(q + 1); // map -1->0, 0->1, 1->2
            compressed_indices[i / 4] |= (packed << ((i % 4) * 2));
        }
    } else if (actual_format == oil::Format::OIL8) {
        // OIL8: 8-bit codebook quantized
        oil::CodebookOIL8 cb;
        cb.train(f32_data.data(), (int)num_weights);
        compressed_indices.resize(num_weights);
        codebook_data.resize(cb.serialized_size());
        cb.serialize(codebook_data.data());
        for (size_t i = 0; i < num_weights; i++) {
            compressed_indices[i] = cb.quantize(f32_data[i]);
        }
    } else if (actual_format == oil::Format::OIL4) {
        // OIL4: 4-bit codebook quantized, pack 2 per byte
        oil::CodebookOIL4 cb;
        cb.train(f32_data.data(), (int)num_weights);
        compressed_indices.resize((num_weights + 1) / 2);
        codebook_data.resize(cb.serialized_size());
        cb.serialize(codebook_data.data());
        for (size_t i = 0; i < num_weights; i++) {
            uint8_t idx = cb.quantize(f32_data[i]);
            if (i % 2 == 0)
                compressed_indices[i / 2] = idx & 0x0F;
            else
                compressed_indices[i / 2] |= (idx << 4);
        }
    } else {
        // OIL1: 1 bit per weight, pack 8 per byte
        compressed_indices.resize((num_weights + 7) / 8, 0);
        codebook_data.resize(2 * 4); // {-1, +1} stored as FP32
        float neg_one = -1.0f, pos_one = 1.0f;
        memcpy(&codebook_data[0], &neg_one, 4);
        memcpy(&codebook_data[4], &pos_one, 4);
        for (size_t i = 0; i < num_weights; i++) {
            if (f32_data[i] >= 0)
                compressed_indices[i / 8] |= (1 << (i % 8));
        }
    }
    return true;
}

static bool convert_raw_fp32(const std::string& in_path, const std::string& out_path) {
    std::ifstream in(in_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: cannot open input: " << in_path << std::endl;
        return false;
    }
    in.seekg(0, std::ios::end);
    size_t file_size = (size_t)in.tellg();
    in.seekg(0, std::ios::beg);

    if (file_size == 0 || file_size % 4 != 0) {
        std::cerr << "Error: raw FP32 file size must be multiple of 4\n";
        return false;
    }

    size_t num_floats = file_size / 4;
    std::vector<float> data(num_floats);
    in.read(reinterpret_cast<char*>(data.data()), (std::streamsize)file_size);
    in.close();

    oil::Tensor weights(oil::Shape{1, (int64_t)num_floats}, oil::DType::F32);
    memcpy(weights.data(), data.data(), file_size);

    oil::OILWriter writer(out_path);
    oil::OILHeader hdr;
    memcpy(hdr.magic, "OIL1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = 0;
    writer.write_header(hdr, nullptr);

    oil::FormatBlockEntry ft_entry;
    ft_entry.block_id = 0;
    ft_entry.format = (uint8_t)oil::Format::OIL32;
    ft_entry.cb_bytes = 0;
    writer.write_format_table({ft_entry});

    oil::TensorEntry t_entry;
    t_entry.name_len = 0;
    t_entry.block_start = 0;
    t_entry.num_blocks = 1;
    writer.write_tensor_table({t_entry}, {"weights"});

    oil::BlockData block;
    block.format = oil::Format::OIL32;
    block.num_weights = (uint32_t)num_floats;
    block.indices.resize(file_size);
    memcpy(block.indices.data(), data.data(), file_size);
    writer.write_block(block);

    writer.close();
    return true;
}

// High-level GGUF conversion: reads via read_gguf(), optionally compresses with FormatPlanner, writes OIL
static bool convert_gguf(const std::string& in_path, const std::string& out_path, float target_bpw) {
    std::vector<float> all_weights;
    std::vector<int64_t> all_shapes;
    std::vector<std::string> all_names;
    if (!read_gguf(in_path, all_weights, all_shapes, all_names, false))
        return false;

    oil::OILWriter writer(out_path);
    oil::OILHeader hdr;
    memcpy(hdr.magic, "OIL1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = 0;
    writer.write_header(hdr, nullptr);

    // Use FormatPlanner if BPW target given
    oil::FormatPlan format_plan;
    std::vector<uint8_t> compressed_indices;
    std::vector<uint8_t> codebook_data;
    bool use_compression = false;

    if (target_bpw > 0.0f && !all_weights.empty()) {
        use_compression = apply_format_plan(all_weights, all_weights.size(), target_bpw,
                                            format_plan, compressed_indices, codebook_data);
    }

    oil::Format actual_format = oil::Format::OIL32;
    uint32_t codebook_bytes = 0;
    if (use_compression) {
        for (auto& blk : format_plan.blocks) {
            actual_format = blk.assigned_format;
            break;
        }
        codebook_bytes = (uint32_t)codebook_data.size();
    }

    std::vector<oil::FormatBlockEntry> fmt_entries;
    {
        oil::FormatBlockEntry fe;
        fe.block_id = 0;
        fe.format = (uint8_t)actual_format;
        fe.cb_bytes = codebook_bytes;
        fmt_entries.push_back(fe);
    }
    writer.write_format_table(fmt_entries);

    // If compression was applied, write all weights as a single block
    if (use_compression) {
        std::vector<oil::TensorEntry> t_entries;
        std::vector<std::string> t_names;
        size_t cursor = 0; (void)cursor;
        for (size_t i = 0; i < all_names.size(); i++) {
            oil::TensorEntry te;
            te.name_len = (uint32_t)all_names[i].size();
            te.block_start = (uint32_t)i;
            te.num_blocks = 1;
            t_entries.push_back(te);
            t_names.push_back(all_names[i]);
        }
        writer.write_tensor_table(t_entries, t_names);

        for (size_t i = 0; i < all_names.size(); i++) {
            oil::BlockData block;
            block.format = actual_format;
            block.num_weights = (uint32_t)all_shapes[i];
            if (actual_format == oil::Format::SPARK_Q0 || actual_format == oil::Format::OIL1 ||
                actual_format == oil::Format::OIL32) {
                // Shared codebook for first block; subsequent blocks reuse it (zero-length indices)
            }
            block.indices = compressed_indices;
            block.codebook = codebook_data;
            writer.write_block(block);
        }
    } else {
        // No compression: write FP32 data per tensor
        std::vector<oil::TensorEntry> t_entries;
        std::vector<std::string> t_names;
        for (size_t i = 0; i < all_names.size(); i++) {
            oil::TensorEntry te;
            te.name_len = (uint32_t)all_names[i].size();
            te.block_start = (uint32_t)i;
            te.num_blocks = 1;
            t_entries.push_back(te);
            t_names.push_back(all_names[i]);
        }
        writer.write_tensor_table(t_entries, t_names);

        size_t cursor = 0;
        for (size_t i = 0; i < all_names.size(); i++) {
            oil::BlockData block;
            block.format = oil::Format::OIL32;
            block.num_weights = (uint32_t)all_shapes[i];
            size_t byte_size = all_shapes[i] * sizeof(float);
            block.indices.resize(byte_size);
            memcpy(block.indices.data(), &all_weights[cursor], byte_size);
            writer.write_block(block);
            cursor += all_shapes[i];
        }
    }

    writer.close();
    std::cout << "Converted " << all_names.size() << " tensors to OIL: " << out_path;
    if (use_compression)
        std::cout << " (compressed to " << target_bpw << " BPW using " << oil::format_name(actual_format) << ")";
    std::cout << std::endl;
    return true;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    std::cout << "OIL Converter" << std::endl;

    if (args.input_path.empty() || args.output_path.empty()) {
        std::cerr << "Error: --input and --output are required\n";
        return 1;
    }

    bool ok = false;
    if (args.input_fmt == "rawfp32") {
        ok = convert_raw_fp32(args.input_path, args.output_path);
    } else if (args.input_fmt == "gguf") {
        ok = convert_gguf(args.input_path, args.output_path, args.target_bpw);
    } else {
        std::cerr << "Unknown format: " << args.input_fmt << std::endl;
        return 1;
    }

    if (!ok) return 1;
    if (args.verbose) {
        std::cout << "Written: " << args.output_path << std::endl;
    }
    return 0;
}
