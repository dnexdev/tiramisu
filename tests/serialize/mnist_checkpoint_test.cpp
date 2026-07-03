#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>

#include "tiramisu/autograd/grad_mode.hpp"
#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/nn/linear.hpp"
#include "tiramisu/serialize/mnist_checkpoint.hpp"

using namespace tiramisu;
using namespace tiramisu::nn;
using namespace tiramisu::autograd;
using namespace tiramisu::serialize;

namespace {

uint32_t swap_endian(uint32_t val) { return __builtin_bswap32(val); }

Tensor load_mnist_images(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open: " + path);
  }
  uint32_t magic, num, rows, cols;
  file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  file.read(reinterpret_cast<char*>(&num), 4);
  num = swap_endian(num);
  file.read(reinterpret_cast<char*>(&rows), 4);
  rows = swap_endian(rows);
  file.read(reinterpret_cast<char*>(&cols), 4);
  cols = swap_endian(cols);

  Tensor t({static_cast<int64_t>(num), static_cast<int64_t>(rows * cols)},
           DType::Float32);
  float* data = t.data<float>();
  for (uint32_t i = 0; i < num * rows * cols; ++i) {
    unsigned char pixel = 0;
    file.read(reinterpret_cast<char*>(&pixel), 1);
    data[i] = (static_cast<float>(pixel) / 255.0f - 0.5f) / 0.5f;
  }
  return t;
}

Tensor load_mnist_labels(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open: " + path);
  }
  uint32_t magic, num;
  file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  file.read(reinterpret_cast<char*>(&num), 4);
  num = swap_endian(num);

  Tensor t({static_cast<int64_t>(num)}, DType::Float32);
  float* data = t.data<float>();
  for (uint32_t i = 0; i < num; ++i) {
    unsigned char label = 0;
    file.read(reinterpret_cast<char*>(&label), 1);
    data[i] = static_cast<float>(label);
  }
  return t;
}

int predict_digit(Linear& layer1, Linear& layer2, const Tensor& image) {
  NoGradGuard guard;
  Tensor h = relu(layer1.forward(image.reshape({1, 784})));
  Tensor logits = layer2.forward(h);
  const float* data = logits.data<float>();
  int best = 0;
  for (int i = 1; i < 10; ++i) {
    if (data[i] > data[best]) {
      best = i;
    }
  }
  return best;
}

}  // namespace

TEST(MNISTCheckpointTest, RoundTripMatchesLayerWeights) {
  Linear layer1(784, 128);
  Linear layer2(128, 10);
  const std::string path = "mnist_roundtrip.bin";
  save_mnist_checkpoint(path, layer1, layer2);

  Linear loaded1(784, 128);
  Linear loaded2(128, 10);
  load_mnist_checkpoint_into(load_mnist_checkpoint(path), loaded1, loaded2);

  EXPECT_EQ(layer1.weight().numel(), loaded1.weight().numel());
  const float* src = layer1.weight().data<float>();
  const float* dst = loaded1.weight().data<float>();
  for (int64_t i = 0; i < layer1.weight().numel(); ++i) {
    EXPECT_FLOAT_EQ(src[i], dst[i]);
  }
  std::remove(path.c_str());
}

TEST(MNISTCheckpointTest, CommittedWeightsClassifyTestSet) {
  const std::string ckpt_path = "../../checkpoints/mnist_mlp.bin";
  const std::string images_path = "../../data/t10k-images.idx3-ubyte";
  const std::string labels_path = "../../data/t10k-labels.idx1-ubyte";
  if (std::ifstream(images_path).good() == false) {
    GTEST_SKIP() << "MNIST data not available";
  }

  Linear layer1(784, 128);
  Linear layer2(128, 10);
  load_mnist_checkpoint_into(load_mnist_checkpoint(ckpt_path), layer1, layer2);

  Tensor images = load_mnist_images(images_path);
  Tensor labels = load_mnist_labels(labels_path);
  const int num = images.shape()[0];
  int correct = 0;
  for (int i = 0; i < num; ++i) {
    Tensor sample = images.slice(0, i, i + 1);
    const int pred = predict_digit(layer1, layer2, sample);
    if (pred == static_cast<int>(labels.data<float>()[i])) {
      correct++;
    }
  }
  const float accuracy = 100.0f * correct / num;
  EXPECT_GE(accuracy, 95.0f) << "accuracy=" << accuracy;
}
