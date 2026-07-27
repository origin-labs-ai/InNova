#pragma once
#include "oil/types.h"
#include "oil/tensor.h"
#include "oil/codebook.h"
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>

namespace oil {

class MappedFile;

#pragma pack(push, 1)
struct OILHeader {
    char magic[4];      // "OIL1"
    uint32_t version;   // packed: major<<22 | minor<<12 | patch
    uint32_t flags;
    uint32_t config_size;
};

struct FormatBlockEntry {
    uint32_t block_id;
    uint8_t format;    // 0=OIL1, 1=SPARK, 2=oil4, 3=oil8
    uint32_t cb_bytes; // codebook size in bytes
};

struct TensorEntry {
    uint16_t name_len;
    uint32_t block_start;
    uint32_t num_blocks;
};
#pragma pack(pop)

struct BlockData {
    Format format;
    std::vector<uint8_t> codebook;
    std::vector<uint8_t> indices;
    uint32_t num_weights;
};

class OILWriter {
public:
    explicit OILWriter(const std::string& path);
    ~OILWriter();
    
    void write_header(const OILHeader& hdr, const uint8_t* config_data);
    void write_format_table(const std::vector<FormatBlockEntry>& entries);
    void write_block(const BlockData& block);
    // Content-addressed dedup write: returns offset, skips duplicate blobs
    size_t write_dedup(const uint8_t* data, size_t size);
    void write_tensor_table(const std::vector<TensorEntry>& entries,
                            const std::vector<std::string>& names);
    void close();
    
private:
    std::ofstream file_;
    size_t data_start_;
    struct BlobIndex { size_t offset; size_t size; };
    std::unordered_map<std::string, BlobIndex> blob_index_; // hex_sha256 -> {offset, size}
};

class OILReader {
public:
    explicit OILReader(const std::string& path);
    ~OILReader();
    
    const OILHeader& header() const;
    std::vector<uint8_t> read_config() const;
    std::vector<FormatBlockEntry> read_format_table() const;
    BlockData read_block(uint32_t block_id) const;
    size_t num_blocks() const;
    
    // Read a named tensor's blocks and reconstruct
    Tensor read_tensor(const std::string& name) const;
    std::vector<std::string> tensor_names() const;
    
    // Check if file was successfully opened
    bool valid() const { return data_ != nullptr; }

    // Get format info for a tensor
    std::vector<Format> tensor_formats(const std::string& name) const;
    
private:
    MappedFile* mapped_file_;
    const uint8_t* data_;
    size_t file_size_;
    OILHeader header_;
    size_t format_table_offset_;
    size_t tensor_table_offset_;
    size_t data_offset_;
    uint32_t num_format_blocks_;
    uint32_t num_tensors_;
    mutable std::vector<FormatBlockEntry> cached_ft_;
};

// ===========================================================================
// OIL Idx — SHA256 integrity-checked index file format
// Header: magic "MYTHOSIDX" | version | num_tensors
// Then for each tensor: name_len | name bytes | sha256(name) [32 bytes]
// On read, each tensor name is re-hashed and compared fail-fast; the first
// corrupt name is reported by name.
// ===========================================================================

struct SHA256Hash {
    uint8_t bytes[32];
};

struct IdxTensorEntry {
    std::string name;
    SHA256Hash name_hash;   // sha256(name) stored in file
};

class OILIdxWriter {
public:
    explicit OILIdxWriter(const std::string& path);
    ~OILIdxWriter();

    // Writes the full idx file: header magic "MYTHOSIDX", version,
    // num_tensors, then per-tensor name + computed sha256(name).
    void write_idx(uint32_t version, const std::vector<std::string>& tensor_names);

    void close();

private:
    std::ofstream file_;
};

class OILIdxReader {
public:
    explicit OILIdxReader(const std::string& path);
    ~OILIdxReader();

    // Reads the idx file and recomputes sha256 for every stored tensor name.
    // On the first mismatch throws oil::Error naming the corrupt tensor.
    // Returns the verified list of tensor names on success.
    std::vector<std::string> read_idx();

    bool valid() const { return data_ != nullptr; }
    uint32_t version() const { return version_; }
    uint32_t num_tensors() const { return num_tensors_; }

private:
    MappedFile* mapped_file_;
    const uint8_t* data_;
    size_t file_size_;
    uint32_t version_;
    uint32_t num_tensors_;
    bool checked_;
    std::vector<std::string> names_;
};

} // namespace oil
