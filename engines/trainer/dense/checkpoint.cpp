#include "trainer.h"
#include "quant/quant_format.h"
#include <fstream>
#include <vector>
#include <cstring>
#include <unordered_map>

namespace quant {
namespace dense {

namespace {

void write_tensor_ser(std::ofstream& file, const std::string& name, const Tensor& t) {
    if (name.size() > UINT32_MAX)
        throw Error("checkpoint: name too long");
    uint32_t name_len = (uint32_t)name.size();
    file.write((const char*)&name_len, sizeof(name_len));
    file.write(name.data(), (std::streamsize)name_len);

    size_t ser_size = t.serialized_size();
    if (ser_size > UINT32_MAX)
        throw Error("tensor too large for checkpoint format (>4GB)");
    uint32_t ser_size32 = (uint32_t)ser_size;
    file.write((const char*)&ser_size32, sizeof(ser_size32));

    std::vector<uint8_t> buf(ser_size);
    t.serialize(buf.data());
    file.write((const char*)buf.data(), (std::streamsize)ser_size);
}

Tensor read_tensor_ser(std::ifstream& file, std::string& name_out) {
    uint32_t name_len = 0;
    file.read((char*)&name_len, sizeof(name_len));
    if (!file) throw Error("corrupt checkpoint: name length");

    name_out.resize(name_len);
    file.read(name_out.data(), (std::streamsize)name_len);

    uint32_t ser_size = 0;
    file.read((char*)&ser_size, sizeof(ser_size));
    if (!file) throw Error("corrupt checkpoint: tensor size");

    std::vector<uint8_t> buf(ser_size);
    file.read((char*)buf.data(), (std::streamsize)ser_size);
    if (!file) throw Error("corrupt checkpoint: tensor data");

    size_t offset = 0;
    return Tensor::deserialize(buf.data(), offset);
}

} // anonymous namespace

void DenseTrainer::save_checkpoint(const std::string& path) {
    auto params = get_parameters();

    std::ofstream file(path, std::ios::binary);
    if (!file) throw Error("save_checkpoint: cannot open " + path);

    uint32_t magic = 0x4B504843;
    uint32_t version = 2;
    file.write((const char*)&magic, sizeof(magic));
    file.write((const char*)&version, sizeof(version));

    if (step_ > UINT32_MAX)
        throw Error("checkpoint: step exceeds uint32_t range");
    uint32_t step_saved = (uint32_t)step_;
    file.write((const char*)&step_saved, sizeof(step_saved));

    float opt_lr = optimizer_.get_lr();
    file.write((const char*)&opt_lr, sizeof(opt_lr));

    if (params.size() > UINT32_MAX)
        throw Error("checkpoint: too many parameters");
    uint32_t num_params = (uint32_t)params.size();
    file.write((const char*)&num_params, sizeof(num_params));

    for (uint32_t i = 0; i < num_params; ++i) {
        std::string name = "param_" + std::to_string(i);
        write_tensor_ser(file, name, *params[i]);
    }

    if (params.size() > UINT32_MAX)
        throw Error("checkpoint: too many optimizer state entries");
    uint32_t num_opt = (uint32_t)params.size();
    file.write((const char*)&num_opt, sizeof(num_opt));

    auto& state_map = optimizer_.mutable_state();
    for (uint32_t idx = 0; idx < num_opt; ++idx) {
        auto* p = params[idx];
        auto it = state_map.find(p);
        if (it != state_map.end()) {
            write_tensor_ser(file, "m_" + std::to_string(idx), it->second.m);
            write_tensor_ser(file, "v_" + std::to_string(idx), it->second.v);
        } else {
            write_tensor_ser(file, "m_" + std::to_string(idx), Tensor::zeros(p->shape()));
            write_tensor_ser(file, "v_" + std::to_string(idx), Tensor::zeros(p->shape()));
        }
    }

    file.close();
}

void DenseTrainer::load_checkpoint(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw Error("load_checkpoint: cannot open " + path);

    uint32_t magic = 0, version = 0;
    file.read((char*)&magic, sizeof(magic));
    file.read((char*)&version, sizeof(version));
    if (magic != 0x4B504843)
        throw Error("load_checkpoint: invalid file magic");

    uint32_t step_loaded = 0;
    float opt_lr_loaded = 0;

    file.read((char*)&step_loaded, sizeof(step_loaded));
    file.read((char*)&opt_lr_loaded, sizeof(opt_lr_loaded));

    uint32_t num_params = 0;
    file.read((char*)&num_params, sizeof(num_params));

    auto params = get_parameters();
    if ((uint32_t)params.size() != num_params)
        throw Error("load_checkpoint: param count mismatch " +
                     std::to_string(params.size()) + " vs " + std::to_string(num_params));

    for (uint32_t i = 0; i < num_params; ++i) {
        std::string loaded_name;
        Tensor t = read_tensor_ser(file, loaded_name);
        std::string expected = "param_" + std::to_string(i);
        if (loaded_name != expected)
            throw Error("load_checkpoint: param name mismatch at " + std::to_string(i));
        params[i]->copy_from(t);
        params[i]->requires_grad(true);
    }

    step_ = (int)step_loaded;

    optimizer_.set_lr(opt_lr_loaded);
    optimizer_.scheduler_step(step_);

    auto& state_map = optimizer_.mutable_state();

    if (version >= 2) {
        uint32_t num_opt = 0;
        file.read((char*)&num_opt, sizeof(num_opt));

        if (state_map.empty() && num_opt > 0) {
            for (auto* p : params)
                state_map[p] = {Tensor::zeros(p->shape()), Tensor::zeros(p->shape())};
        }

        for (uint32_t i = 0; i < num_opt && i < (uint32_t)params.size(); ++i) {
            Tensor* p = params[i];

            std::string m_name, v_name;
            Tensor m_t = read_tensor_ser(file, m_name);
            Tensor v_t = read_tensor_ser(file, v_name);

            std::string m_exp = "m_" + std::to_string(i);
            std::string v_exp = "v_" + std::to_string(i);
            if (m_name != m_exp || v_name != v_exp)
                throw Error("load_checkpoint: optimizer state name mismatch at idx " + std::to_string(i));

            auto it = state_map.find(p);
            if (it != state_map.end()) {
                it->second.m.copy_from(m_t);
                it->second.v.copy_from(v_t);
            }
        }
    }
}

} // namespace dense
} // namespace quant
