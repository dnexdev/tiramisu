#include "tiramisu/serialize/mnist_checkpoint.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace tiramisu::serialize {
namespace {

constexpr char kMagic[4] = {'M', 'N', 'S', 'T'};

void write_all(std::ostream& out, const float* data, size_t count) {
  out.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(count * sizeof(float)));
}

void read_all(std::istream& in, float* data, size_t count) {
  in.read(reinterpret_cast<char*>(data),
          static_cast<std::streamsize>(count * sizeof(float)));
  if (!in) {
    throw std::runtime_error("Unexpected end of MNIST checkpoint");
  }
}

std::vector<float> read_vec(std::istream& in, size_t count) {
  std::vector<float> out(count);
  read_all(in, out.data(), count);
  return out;
}

void expect_magic(std::istream& in) {
  char magic[4] = {};
  in.read(magic, 4);
  if (!in || std::memcmp(magic, kMagic, 4) != 0) {
    throw std::runtime_error("Invalid MNIST checkpoint magic");
  }
}

void write_magic(std::ostream& out) { out.write(kMagic, 4); }

MNISTCheckpoint checkpoint_from_layers(nn::Linear& layer1, nn::Linear& layer2) {
  MNISTCheckpoint ckpt;
  const auto& w1 = layer1.weight();
  const auto& b1 = layer1.bias();
  const auto& w2 = layer2.weight();
  const auto& b2 = layer2.bias();

  ckpt.w1.assign(w1.data<float>(), w1.data<float>() + w1.numel());
  ckpt.b1.assign(b1.data<float>(), b1.data<float>() + b1.numel());
  ckpt.w2.assign(w2.data<float>(), w2.data<float>() + w2.numel());
  ckpt.b2.assign(b2.data<float>(), b2.data<float>() + b2.numel());
  return ckpt;
}

}  // namespace

void save_mnist_checkpoint(const std::string& path, const MNISTCheckpoint& ckpt) {
  if (ckpt.w1.size() != static_cast<size_t>(MNISTCheckpoint::kInputDim *
                                            MNISTCheckpoint::kHiddenDim) ||
      ckpt.b1.size() != static_cast<size_t>(MNISTCheckpoint::kHiddenDim) ||
      ckpt.w2.size() != static_cast<size_t>(MNISTCheckpoint::kHiddenDim *
                                            MNISTCheckpoint::kOutputDim) ||
      ckpt.b2.size() != static_cast<size_t>(MNISTCheckpoint::kOutputDim)) {
    throw std::runtime_error("MNIST checkpoint tensor sizes do not match MLP");
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Cannot open MNIST checkpoint for write: " + path);
  }
  write_magic(out);
  write_all(out, ckpt.w1.data(), ckpt.w1.size());
  write_all(out, ckpt.b1.data(), ckpt.b1.size());
  write_all(out, ckpt.w2.data(), ckpt.w2.size());
  write_all(out, ckpt.b2.data(), ckpt.b2.size());
}

void save_mnist_checkpoint(const std::string& path, nn::Linear& layer1,
                           nn::Linear& layer2) {
  save_mnist_checkpoint(path, checkpoint_from_layers(layer1, layer2));
}

MNISTCheckpoint load_mnist_checkpoint(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Cannot open MNIST checkpoint: " + path);
  }
  expect_magic(in);
  MNISTCheckpoint ckpt;
  ckpt.w1 = read_vec(in, static_cast<size_t>(MNISTCheckpoint::kInputDim *
                                             MNISTCheckpoint::kHiddenDim));
  ckpt.b1 = read_vec(in, static_cast<size_t>(MNISTCheckpoint::kHiddenDim));
  ckpt.w2 = read_vec(in, static_cast<size_t>(MNISTCheckpoint::kHiddenDim *
                                             MNISTCheckpoint::kOutputDim));
  ckpt.b2 = read_vec(in, static_cast<size_t>(MNISTCheckpoint::kOutputDim));
  return ckpt;
}

MNISTCheckpoint load_mnist_checkpoint(std::span<const std::byte> data) {
  if (data.size() < 4) {
    throw std::runtime_error("MNIST checkpoint too small");
  }
  if (std::memcmp(data.data(), kMagic, 4) != 0) {
    throw std::runtime_error("Invalid MNIST checkpoint magic");
  }

  const auto* floats = reinterpret_cast<const float*>(data.data() + 4);
  const size_t float_count = (data.size() - 4) / sizeof(float);
  const size_t expected = static_cast<size_t>(
      MNISTCheckpoint::kInputDim * MNISTCheckpoint::kHiddenDim +
      MNISTCheckpoint::kHiddenDim +
      MNISTCheckpoint::kHiddenDim * MNISTCheckpoint::kOutputDim +
      MNISTCheckpoint::kOutputDim);
  if (float_count != expected) {
    throw std::runtime_error("MNIST checkpoint size mismatch");
  }

  MNISTCheckpoint ckpt;
  size_t offset = 0;
  ckpt.w1.assign(floats, floats + MNISTCheckpoint::kInputDim * MNISTCheckpoint::kHiddenDim);
  offset = ckpt.w1.size();
  ckpt.b1.assign(floats + offset, floats + offset + MNISTCheckpoint::kHiddenDim);
  offset += ckpt.b1.size();
  ckpt.w2.assign(floats + offset,
                 floats + offset + MNISTCheckpoint::kHiddenDim * MNISTCheckpoint::kOutputDim);
  offset += ckpt.w2.size();
  ckpt.b2.assign(floats + offset, floats + offset + MNISTCheckpoint::kOutputDim);
  return ckpt;
}

void load_mnist_checkpoint_into(const MNISTCheckpoint& ckpt, nn::Linear& layer1,
                                nn::Linear& layer2) {
  auto& w1 = layer1.weight();
  auto& b1 = layer1.bias();
  auto& w2 = layer2.weight();
  auto& b2 = layer2.bias();

  if (ckpt.w1.size() != static_cast<size_t>(w1.numel()) ||
      ckpt.b1.size() != static_cast<size_t>(b1.numel()) ||
      ckpt.w2.size() != static_cast<size_t>(w2.numel()) ||
      ckpt.b2.size() != static_cast<size_t>(b2.numel())) {
    throw std::runtime_error("MNIST checkpoint does not match layer shapes");
  }

  std::memcpy(w1.data<float>(), ckpt.w1.data(), ckpt.w1.size() * sizeof(float));
  std::memcpy(b1.data<float>(), ckpt.b1.data(), ckpt.b1.size() * sizeof(float));
  std::memcpy(w2.data<float>(), ckpt.w2.data(), ckpt.w2.size() * sizeof(float));
  std::memcpy(b2.data<float>(), ckpt.b2.data(), ckpt.b2.size() * sizeof(float));
}

}  // namespace tiramisu::serialize
