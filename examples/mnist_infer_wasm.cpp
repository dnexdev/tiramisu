#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "tiramisu/autograd/grad_mode.hpp"
#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/core/tensor.hpp"
#include "tiramisu/nn/linear.hpp"
#include "tiramisu/serialize/mnist_checkpoint.hpp"

using namespace tiramisu;
using namespace tiramisu::nn;
using namespace tiramisu::autograd;
using namespace tiramisu::serialize;

namespace {

std::shared_ptr<Linear> g_layer1;
std::shared_ptr<Linear> g_layer2;

int argmax10(const float* logits) {
  int best = 0;
  for (int i = 1; i < 10; ++i) {
    if (logits[i] > logits[best]) {
      best = i;
    }
  }
  return best;
}

int forward_mlp(const float* pixels784, float* logits_out, float* hidden_out) {
  Tensor x({1, 784}, DType::Float32);
  std::memcpy(x.data<float>(), pixels784, 784 * sizeof(float));
  Tensor h = relu(g_layer1->forward(x));
  Tensor logits = g_layer2->forward(h);

  const float* h_data = h.data<float>();
  const float* logit_data = logits.data<float>();
  if (hidden_out) {
    std::memcpy(hidden_out, h_data, 128 * sizeof(float));
  }
  std::memcpy(logits_out, logit_data, 10 * sizeof(float));
  return argmax10(logit_data);
}

}  // namespace

extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
int mnist_init(const uint8_t* weights, size_t len) {
  try {
    NoGradGuard guard;
    const auto bytes = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(weights), len);
    MNISTCheckpoint ckpt = load_mnist_checkpoint(bytes);
    g_layer1 = std::make_shared<Linear>(784, 128);
    g_layer2 = std::make_shared<Linear>(128, 10);
    load_mnist_checkpoint_into(ckpt, *g_layer1, *g_layer2);
    return 0;
  } catch (...) {
    g_layer1.reset();
    g_layer2.reset();
    return -1;
  }
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
int mnist_predict(const float* pixels784) {
  if (!g_layer1 || !g_layer2 || !pixels784) {
    return -1;
  }
  try {
    NoGradGuard guard;
    float logits[10];
    return forward_mlp(pixels784, logits, nullptr);
  } catch (...) {
    return -1;
  }
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
int mnist_predict_ex(const float* pixels784, float* logits_out,
                     float* hidden_out) {
  if (!g_layer1 || !g_layer2 || !pixels784 || !logits_out) {
    return -1;
  }
  try {
    NoGradGuard guard;
    return forward_mlp(pixels784, logits_out, hidden_out);
  } catch (...) {
    return -1;
  }
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void mnist_free() {
  g_layer1.reset();
  g_layer2.reset();
}

}  // extern "C"

#ifndef __EMSCRIPTEN__
int main() { return 0; }
#endif
