#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/core/tensor.hpp"
#include "tiramisu/nn/linear.hpp"
#include "tiramisu/nn/loss.hpp"
#include "tiramisu/nn/sequential.hpp"
#include "tiramisu/optim/adam.hpp"

using namespace tiramisu;
using namespace tiramisu::nn;
using namespace tiramisu::optim;
using namespace tiramisu::autograd;

uint32_t swap_endian(uint32_t val) { return __builtin_bswap32(val); }

Tensor load_mnist_images(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) throw std::runtime_error("Cannot open: " + path);
  uint32_t magic, num, rows, cols;
  file.read((char*)&magic, 4);
  magic = swap_endian(magic);
  file.read((char*)&num, 4);
  num = swap_endian(num);
  file.read((char*)&rows, 4);
  rows = swap_endian(rows);
  file.read((char*)&cols, 4);
  cols = swap_endian(cols);

  Tensor t({(int64_t)num, (int64_t)(rows * cols)}, DType::Float32);
  float* data = t.data<float>();

  for (uint32_t i = 0; i < num * rows * cols; ++i) {
    unsigned char pixel;
    file.read((char*)&pixel, 1);
    data[i] = (static_cast<float>(pixel) / 255.0f - 0.5f) / 0.5f;  // [-1, 1]
  }
  return t;
}

Tensor load_mnist_labels(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) throw std::runtime_error("Cannot open: " + path);
  uint32_t magic, num;
  file.read((char*)&magic, 4);
  magic = swap_endian(magic);
  file.read((char*)&num, 4);
  num = swap_endian(num);

  Tensor t({(int64_t)num}, DType::Float32);
  float* data = t.data<float>();
  for (int i = 0; i < (int)num; ++i) {
    unsigned char label;
    file.read((char*)&label, 1);
    data[i] = static_cast<float>(label);
  }
  return t;
}

void evaluate(std::shared_ptr<nn::Linear> layer1,
              std::shared_ptr<nn::Linear> layer2, const Tensor& test_x,
              const Tensor& test_y) {
  int num_test = test_x.shape()[0];
  int correct = 0;
  int batch_size = 64;

  for (int b = 0; b * batch_size < num_test; b++) {
    int start = b * batch_size;
    int end = std::min(start + batch_size, num_test);

    Tensor batch_x = test_x.slice(0, start, end);
    Tensor batch_y = test_y.slice(0, start, end);

    // Forward pass only
    Tensor h = layer1->forward(batch_x);
    h = tiramisu::autograd::relu(h);
    Tensor logits = layer2->forward(h);

    // Find prediction: compare argmax(logits) with batch_y
    float* logit_data = logits.data<float>();
    float* label_data = batch_y.data<float>();

    for (int i = 0; i < (end - start); i++) {
      // Find index of max logit
      int max_idx = 0;
      float max_val = logit_data[i * 10];
      for (int j = 1; j < 10; j++) {
        if (logit_data[i * 10 + j] > max_val) {
          max_val = logit_data[i * 10 + j];
          max_idx = j;
        }
      }
      if (max_idx == (int)label_data[i]) {
        correct++;
      }
    }
  }
  printf("Test Accuracy: %.2f%%\n", (float)correct / num_test * 100.0f);
}

int main() {
  Tensor train_x = load_mnist_images("../../data/train-images.idx3-ubyte");
  Tensor train_y = load_mnist_labels("../../data/train-labels.idx1-ubyte");

  Tensor test_x = load_mnist_images("../../data/t10k-images.idx3-ubyte");
  Tensor test_y = load_mnist_labels("../../data/t10k-labels.idx1-ubyte");

  auto layer1 = std::make_shared<nn::Linear>(784, 128);
  auto layer2 = std::make_shared<nn::Linear>(128, 10);

  std::vector<Tensor*> params;
  for (auto p : layer1->parameters()) params.push_back(p);
  for (auto p : layer2->parameters()) params.push_back(p);

  optim::Adam opt(params, 0.001f);

  const int batch_size = 64;
  int num_train = train_x.shape()[0];

  std::mt19937 gen(42);  // Seeded random engine
  for (Tensor* p : params) {
    if (p->shape().size() == 2) {  // It's a weight matrix
      int64_t rows = p->shape()[0];
      int64_t cols = p->shape()[1];

      // Use He initialization for ReLU layers: stddev = sqrt(2 / fan_in)
      // Adjust fan_in depending on whether shape is [out_features, in_features]
      // or vice versa
      int64_t fan_in = (rows > cols) ? cols : rows;
      std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fan_in));

      float* d = p->data<float>();
      for (int64_t i = 0; i < rows * cols; ++i) {
        d[i] = dist(gen);
      }
    } else if (p->shape().size() == 1) {  // It's a bias vector
      float* d = p->data<float>();
      for (int64_t i = 0; i < p->shape()[0]; ++i) {
        d[i] = 0.0f;  // Initialize biases to 0
      }
    }
  }

  for (int epoch = 0; epoch < 10; epoch++) {
    float total_loss = 0.0f;
    for (int b = 0; b * batch_size < num_train; b++) {
      int start = b * batch_size;
      int end = std::min(start + batch_size, num_train);

      Tensor batch_x = train_x.slice(0, start, end);
      Tensor batch_y = train_y.slice(0, start, end);

      opt.zero_grad();
      Tensor h = layer1->forward(batch_x);
      h = tiramisu::autograd::relu(h);
      Tensor logits = layer2->forward(h);
      Tensor loss = nn::cross_entropy_loss(logits, batch_y);

      autograd::backward(loss);
      opt.step();

      total_loss += loss.data<float>()[0];
    }
    printf("Epoch %d, Avg Loss: %.4f\n", epoch,
           total_loss / (num_train / batch_size));
  }

  printf("Running evaluation on test set...\n");
  evaluate(layer1, layer2, test_x, test_y);
  return 0;
}