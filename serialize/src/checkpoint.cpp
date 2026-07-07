#include "tiramisu/serialize/checkpoint.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include "tiramisu/core/cuda_memory.hpp"

// Tiramisu GPT checkpoint format ("TIRA").
//
// Common header:
//   [magic "TIRA" | 4 bytes]
//   [version u32]         // 1 or 2
//   [vocab_size i64][d_model i64][num_heads i64][num_layers i64][max_seq_len i64]
//   [tie_weights u32][step i64][epoch i64]
//   [num_params u32]
//
// v1 record (version == 1):
//   [name_len u32][name bytes]
//   [rank u32][shape[0..rank] i64...]
//   [data_count u32][floats: data_count * f32]
//
// v2 record (version == 2):
//   [name_len u32][name bytes]
//   [rank u32][shape[0..rank] i64...]
//   [kind u32]                            // 0 = fp32, 1 = int8 per-channel symmetric
//   if kind == 0:
//     [data_count u32][floats: data_count * f32]
//   if kind == 1:
//     [channel_axis i64]
//     [num_scales u32][scales: num_scales * f32]
//     [data_count u32][int8s: data_count bytes]
//
// All integers are little-endian on the host that produced the file.
namespace tiramisu::serialize {

namespace {

constexpr char kMagic[4] = {'T', 'I', 'R', 'A'};
constexpr uint32_t kVersionV1 = 1;
constexpr uint32_t kVersionV2 = 2;

inline int64_t checked_mul_i64(int64_t a, int64_t b, const char* ctx) {
#if defined(__has_builtin)
#  if __has_builtin(__builtin_mul_overflow)
  int64_t r;
  if (__builtin_mul_overflow(a, b, &r)) {
    throw std::overflow_error(std::string("integer overflow in ") + ctx);
  }
  return r;
#  else
  if (a != 0 && b != 0 && (a * b) / a != b) {
    throw std::overflow_error(std::string("integer overflow in ") + ctx);
  }
  return a * b;
#  endif
#else
  if (a != 0 && b != 0 && (a * b) / a != b) {
    throw std::overflow_error(std::string("integer overflow in ") + ctx);
  }
  return a * b;
#endif
}

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

void write_string(std::ofstream& out, const std::string& s) {
  write_u32(out, static_cast<uint32_t>(s.size()));
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

int64_t shape_numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    n = checked_mul_i64(n, d, "shape_numel");
  }
  return n;
}

// Dequantize int8 per-channel symmetric into a fresh fp32 vector.
// Same layout math as quant/src/quantize.cpp:dequantize.
std::vector<float> dequantize_int8(const std::vector<int8_t>& q,
                                   const std::vector<float>& scales,
                                   const std::vector<int64_t>& shape,
                                   int64_t channel_axis) {
  int64_t outer = 1, channel = shape[channel_axis], inner = 1;
  for (int64_t i = 0; i < channel_axis; ++i) outer *= shape[i];
  for (size_t i = channel_axis + 1; i < shape.size(); ++i) inner *= shape[i];
  if (static_cast<int64_t>(scales.size()) != channel) {
    throw std::runtime_error("checkpoint v2: scales/channel size mismatch");
  }

  std::vector<float> out(static_cast<size_t>(outer * channel * inner));
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t c = 0; c < channel; ++c) {
      const float s = scales[static_cast<size_t>(c)];
      const int64_t base = (o * channel + c) * inner;
      for (int64_t i = 0; i < inner; ++i) {
        out[static_cast<size_t>(base + i)] =
            static_cast<float>(q[static_cast<size_t>(base + i)]) * s;
      }
    }
  }
  return out;
}

ParameterEntry parse_v1_record(ByteReader& reader) {
  ParameterEntry entry;
  entry.name = reader.read_string();
  const uint32_t rank = reader.read_u32();
  entry.shape.resize(rank);
  int64_t numel = 1;
  for (uint32_t d = 0; d < rank; ++d) {
    entry.shape[d] = reader.read_i64();
    numel = checked_mul_i64(numel, entry.shape[d], "parse_v1: shape product");
  }
  const uint32_t count = reader.read_u32();
  if (static_cast<int64_t>(count) != numel) {
    throw std::runtime_error("load_gpt_checkpoint: tensor size mismatch");
  }
  entry.data.resize(count);
  reader.read(entry.data.data(),
              static_cast<std::size_t>(count) * sizeof(float));
  return entry;
}

ParameterEntry parse_v2_record(ByteReader& reader) {
  ParameterEntry entry;
  entry.name = reader.read_string();
  const uint32_t rank = reader.read_u32();
  entry.shape.resize(rank);
  int64_t numel = 1;
  for (uint32_t d = 0; d < rank; ++d) {
    entry.shape[d] = reader.read_i64();
    numel = checked_mul_i64(numel, entry.shape[d], "parse_v2: shape product");
  }

  const uint32_t kind = reader.read_u32();
  if (kind == kFp32) {
    const uint32_t count = reader.read_u32();
    if (static_cast<int64_t>(count) != numel) {
      throw std::runtime_error("load_gpt_checkpoint: fp32 size mismatch");
    }
    entry.data.resize(count);
    reader.read(entry.data.data(),
                static_cast<std::size_t>(count) * sizeof(float));
  } else if (kind == kInt8PerChannelSymmetric) {
    const int64_t channel_axis = reader.read_i64();
    if (channel_axis < 0 ||
        channel_axis >= static_cast<int64_t>(entry.shape.size())) {
      throw std::runtime_error("load_gpt_checkpoint: bad channel_axis");
    }
    const uint32_t num_scales = reader.read_u32();
    if (static_cast<int64_t>(num_scales) != entry.shape[channel_axis]) {
      throw std::runtime_error("load_gpt_checkpoint: scale count mismatch");
    }
    std::vector<float> scales(num_scales);
    reader.read(scales.data(),
                static_cast<std::size_t>(num_scales) * sizeof(float));
    const uint32_t count = reader.read_u32();
    if (static_cast<int64_t>(count) != numel) {
      throw std::runtime_error("load_gpt_checkpoint: int8 size mismatch");
    }
    std::vector<int8_t> qdata(count);
    reader.read(qdata.data(), static_cast<std::size_t>(count));
    entry.data =
        dequantize_int8(qdata, scales, entry.shape, channel_axis);
  } else {
    throw std::runtime_error("load_gpt_checkpoint: unknown kind");
  }
  return entry;
}

GPTCheckpoint parse_checkpoint(ByteReader& reader) {
  char magic[4] = {};
  reader.read(magic, sizeof(magic));
  if (std::string(magic, sizeof(magic)) != std::string(kMagic, sizeof(kMagic))) {
    throw std::runtime_error("load_gpt_checkpoint: invalid magic");
  }

  const uint32_t version = reader.read_u32();
  if (version != kVersionV1 && version != kVersionV2) {
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
    ckpt.parameters.push_back(version == kVersionV1 ? parse_v1_record(reader)
                                                    : parse_v2_record(reader));
  }

  return ckpt;
}

void write_header(std::ofstream& out, const nn::GPTConfig& cfg, int64_t step,
                  int64_t epoch, uint32_t version, uint32_t num_params) {
  out.write(kMagic, sizeof(kMagic));
  write_u32(out, version);
  write_i64(out, cfg.vocab_size);
  write_i64(out, cfg.d_model);
  write_i64(out, cfg.num_heads);
  write_i64(out, cfg.num_layers);
  write_i64(out, cfg.max_seq_len);
  write_u32(out, cfg.tie_weights ? 1u : 0u);
  write_i64(out, step);
  write_i64(out, epoch);
  write_u32(out, num_params);
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

  write_header(out, ckpt.config, ckpt.step, ckpt.epoch, kVersionV1,
               static_cast<uint32_t>(ckpt.parameters.size()));

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

void save_raw_gpt_checkpoint(const std::string& path,
                             const RawGPTCheckpoint& ckpt) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("save_raw_gpt_checkpoint: cannot open " + path);
  }

  write_header(out, ckpt.config, ckpt.step, ckpt.epoch, kVersionV2,
               static_cast<uint32_t>(ckpt.parameters.size()));

  for (const auto& entry : ckpt.parameters) {
    const int64_t numel = shape_numel(entry.shape);
    write_string(out, entry.name);
    write_u32(out, static_cast<uint32_t>(entry.shape.size()));
    for (int64_t dim : entry.shape) {
      write_i64(out, dim);
    }
    write_u32(out, static_cast<uint32_t>(entry.kind));

    if (entry.kind == kFp32) {
      if (static_cast<int64_t>(entry.fp32_data.size()) != numel) {
        throw std::runtime_error("save_raw: fp32 size != shape numel for " +
                                 entry.name);
      }
      write_u32(out, static_cast<uint32_t>(entry.fp32_data.size()));
      out.write(
          reinterpret_cast<const char*>(entry.fp32_data.data()),
          static_cast<std::streamsize>(entry.fp32_data.size() * sizeof(float)));
    } else if (entry.kind == kInt8PerChannelSymmetric) {
      if (entry.channel_axis < 0 ||
          entry.channel_axis >= static_cast<int64_t>(entry.shape.size())) {
        throw std::runtime_error("save_raw: bad channel_axis for " +
                                 entry.name);
      }
      if (static_cast<int64_t>(entry.scales.size()) !=
          entry.shape[entry.channel_axis]) {
        throw std::runtime_error("save_raw: scale count mismatch for " +
                                 entry.name);
      }
      if (static_cast<int64_t>(entry.int8_data.size()) != numel) {
        throw std::runtime_error("save_raw: int8 size != shape numel for " +
                                 entry.name);
      }
      write_i64(out, entry.channel_axis);
      write_u32(out, static_cast<uint32_t>(entry.scales.size()));
      out.write(
          reinterpret_cast<const char*>(entry.scales.data()),
          static_cast<std::streamsize>(entry.scales.size() * sizeof(float)));
      write_u32(out, static_cast<uint32_t>(entry.int8_data.size()));
      out.write(reinterpret_cast<const char*>(entry.int8_data.data()),
                static_cast<std::streamsize>(entry.int8_data.size()));
    } else {
      throw std::runtime_error("save_raw: unknown kind for " + entry.name);
    }
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
