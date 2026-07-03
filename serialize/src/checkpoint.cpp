#include "tiramisu/serialize/checkpoint.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "tiramisu/core/cuda_memory.hpp"

namespace tiramisu::serialize {

namespace {

constexpr char kMagic[4] = {'T', 'I', 'R', 'A'};
constexpr uint32_t kVersion = 1;

class ByteReader {
 public:
  explicit ByteReader(std::span<const std::byte> data) : data_(data) {}

  void read(void* dst, std::size_t nbytes) {
    if (pos_ + nbytes > data_.size()) {
      throw std::runtime_error("load_gpt_checkpoint: unexpected end of data");
    }
    std::memcpy(dst, data_.data() + pos_, nbytes);
    pos_ += nbytes;
  }

  uint32_t read_u32() {
    uint32_t v = 0;
    read(&v, sizeof(v));
    return v;
  }

  int64_t read_i64() {
    int64_t v = 0;
    read(&v, sizeof(v));
    return v;
  }

  std::string read_string() {
    const uint32_t len = read_u32();
    std::string s(len, '\0');
    read(s.data(), len);
    return s;
  }

 private:
  std::span<const std::byte> data_;
  std::size_t pos_ = 0;
};

void write_u32(std::ofstream& out, uint32_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void write_i64(std::ofstream& out, int64_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

uint32_t read_u32(std::ifstream& in) {
  uint32_t v = 0;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}

int64_t read_i64(std::ifstream& in) {
  int64_t v = 0;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}

void write_string(std::ofstream& out, const std::string& s) {
  write_u32(out, static_cast<uint32_t>(s.size()));
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

std::string read_string(std::ifstream& in) {
  const uint32_t len = read_u32(in);
  std::string s(len, '\0');
  in.read(s.data(), static_cast<std::streamsize>(len));
  return s;
}

GPTCheckpoint parse_checkpoint(ByteReader& reader) {
  char magic[4] = {};
  reader.read(magic, sizeof(magic));
  if (std::string(magic, sizeof(magic)) != std::string(kMagic, sizeof(kMagic))) {
    throw std::runtime_error("load_gpt_checkpoint: invalid magic");
  }

  const uint32_t version = reader.read_u32();
  if (version != kVersion) {
    throw std::runtime_error("load_gpt_checkpoint: unsupported version");
  }

  GPTCheckpoint ckpt;
  ckpt.config.vocab_size = reader.read_i64();
  ckpt.config.d_model = reader.read_i64();
  ckpt.config.num_heads = reader.read_i64();
  ckpt.config.num_layers = reader.read_i64();
  ckpt.config.max_seq_len = reader.read_i64();
  ckpt.config.tie_weights = reader.read_u32() != 0;
  ckpt.step = reader.read_i64();
  ckpt.epoch = reader.read_i64();

  const uint32_t num_params = reader.read_u32();
  ckpt.parameters.reserve(num_params);
  for (uint32_t i = 0; i < num_params; ++i) {
    ParameterEntry entry;
    entry.name = reader.read_string();
    const uint32_t rank = reader.read_u32();
    entry.shape.resize(rank);
    int64_t numel = 1;
    for (uint32_t d = 0; d < rank; ++d) {
      entry.shape[d] = reader.read_i64();
      numel *= entry.shape[d];
    }
    const uint32_t count = reader.read_u32();
    if (static_cast<int64_t>(count) != numel) {
      throw std::runtime_error("load_gpt_checkpoint: tensor size mismatch");
    }
    entry.data.resize(count);
    reader.read(entry.data.data(), count * sizeof(float));
    ckpt.parameters.push_back(std::move(entry));
  }

  return ckpt;
}

void apply_checkpoint_weights(const GPTCheckpoint& ckpt, nn::GPT& model) {
  if (ckpt.config.vocab_size != model.config().vocab_size ||
      ckpt.config.d_model != model.config().d_model ||
      ckpt.config.num_heads != model.config().num_heads ||
      ckpt.config.num_layers != model.config().num_layers ||
      ckpt.config.max_seq_len != model.config().max_seq_len ||
      ckpt.config.tie_weights != model.config().tie_weights) {
    throw std::runtime_error("load_gpt_model: config mismatch");
  }

  std::vector<Tensor*> params = model.parameters();
  if (params.size() != ckpt.parameters.size()) {
    throw std::runtime_error("load_gpt_model: parameter count mismatch");
  }

  for (size_t i = 0; i < params.size(); ++i) {
    if (params[i]->shape() != ckpt.parameters[i].shape) {
      throw std::runtime_error("load_gpt_model: shape mismatch for " +
                               ckpt.parameters[i].name);
    }
    if (params[i]->device() == Device::CUDA) {
      cuda_mem::copy_bytes(ckpt.parameters[i].data.data(),
                           params[i]->data<float>(),
                           ckpt.parameters[i].data.size() * sizeof(float),
                           Device::CPU, Device::CUDA);
    } else {
      std::copy_n(ckpt.parameters[i].data.begin(), params[i]->numel(),
                  params[i]->data<float>());
    }
  }
}

}  // namespace

void save_gpt_checkpoint(const std::string& path, const GPTCheckpoint& ckpt) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("save_gpt_checkpoint: cannot open " + path);
  }

  out.write(kMagic, sizeof(kMagic));
  write_u32(out, kVersion);
  write_i64(out, ckpt.config.vocab_size);
  write_i64(out, ckpt.config.d_model);
  write_i64(out, ckpt.config.num_heads);
  write_i64(out, ckpt.config.num_layers);
  write_i64(out, ckpt.config.max_seq_len);
  write_u32(out, ckpt.config.tie_weights ? 1u : 0u);
  write_i64(out, ckpt.step);
  write_i64(out, ckpt.epoch);
  write_u32(out, static_cast<uint32_t>(ckpt.parameters.size()));

  for (const auto& entry : ckpt.parameters) {
    write_string(out, entry.name);
    write_u32(out, static_cast<uint32_t>(entry.shape.size()));
    for (int64_t dim : entry.shape) {
      write_i64(out, dim);
    }
    write_u32(out, static_cast<uint32_t>(entry.data.size()));
    out.write(reinterpret_cast<const char*>(entry.data.data()),
              static_cast<std::streamsize>(entry.data.size() * sizeof(float)));
  }
}

GPTCheckpoint load_gpt_checkpoint(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("load_gpt_checkpoint: cannot open " + path);
  }

  in.seekg(0, std::ios::end);
  const std::streamsize size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<std::byte> buffer(static_cast<size_t>(size));
  in.read(reinterpret_cast<char*>(buffer.data()), size);
  return load_gpt_checkpoint(buffer);
}

GPTCheckpoint load_gpt_checkpoint(std::span<const std::byte> data) {
  ByteReader reader(data);
  return parse_checkpoint(reader);
}

void save_gpt_model(const std::string& path, nn::GPT& model, int64_t step,
                    int64_t epoch) {
  GPTCheckpoint ckpt;
  ckpt.config = model.config();
  ckpt.step = step;
  ckpt.epoch = epoch;

  const std::vector<Tensor*> params = model.parameters();
  for (size_t i = 0; i < params.size(); ++i) {
    ParameterEntry entry;
    entry.name = "param_" + std::to_string(i);
    entry.shape = params[i]->shape();
    entry.data.resize(params[i]->numel());
    if (params[i]->device() == Device::CUDA) {
      cuda_mem::copy_bytes(params[i]->data<float>(), entry.data.data(),
                           entry.data.size() * sizeof(float), Device::CUDA,
                           Device::CPU);
    } else {
      std::copy_n(params[i]->data<float>(), params[i]->numel(),
                  entry.data.begin());
    }
    ckpt.parameters.push_back(std::move(entry));
  }

  save_gpt_checkpoint(path, ckpt);
}

void load_gpt_model(const std::string& path, nn::GPT& model, int64_t* step,
                    int64_t* epoch) {
  GPTCheckpoint ckpt = load_gpt_checkpoint(path);
  load_gpt_checkpoint_into(ckpt, model);
  if (step) *step = ckpt.step;
  if (epoch) *epoch = ckpt.epoch;
}

void load_gpt_checkpoint_into(const GPTCheckpoint& ckpt, nn::GPT& model) {
  apply_checkpoint_weights(ckpt, model);
}

nn::GPT create_gpt_from_checkpoint(std::span<const std::byte> data) {
  GPTCheckpoint ckpt = load_gpt_checkpoint(data);
  nn::GPT model(ckpt.config);
  load_gpt_checkpoint_into(ckpt, model);
  return model;
}

}  // namespace tiramisu::serialize
