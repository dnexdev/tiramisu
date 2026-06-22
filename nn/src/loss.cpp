#include "tiramisu/nn/loss.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "tiramisu/core/node.hpp"

namespace tiramisu::nn {

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
  if (logits.shape().size() != 2)
    throw std::invalid_argument(
        "cross_entropy_loss: logits must be 2D (batch, classes)");

  int64_t batch = logits.shape()[0];
  int64_t C = logits.shape()[1];

  Tensor c_logits = logits.contiguous();
  const float* logit_data = c_logits.data<float>();

  std::vector<float> softmax_vals(batch * C);
  float total_loss = 0.0f;

  for (int64_t i = 0; i < batch; i++) {
    const float* row = logit_data + i * C;

    float row_max = *std::max_element(row, row + C);
    float sum_exp = 0.0f;
    for (int64_t c = 0; c < C; c++) {
      softmax_vals[i * C + c] = std::exp(row[c] - row_max);
      sum_exp += softmax_vals[i * C + c];
    }
    for (int64_t c = 0; c < C; c++) {
      softmax_vals[i * C + c] /= sum_exp;
    }
    int64_t target = static_cast<int64_t>(targets.data<float>()[i]);
    total_loss += -std::log(softmax_vals[i * C + target] + 1e-9f);
  }

  Tensor loss({1});
  loss.data<float>()[0] = total_loss / static_cast<float>(batch);

  if (logits.requires_grad()) {
    auto node = std::make_shared<tiramisu::Node>();
    node->inputs = {logits};
    node->backward_fn = [softmax_vals, batch, C,
                         targets](const Tensor& grad_output) {
      float upstream = grad_output.data<float>()[0];

      Tensor grad_logits({batch, C});
      float* gl = grad_logits.data<float>();

      for (int64_t i = 0; i < batch; i++) {
        int64_t target = static_cast<int64_t>(targets.data<float>()[i]);
        for (int64_t c = 0; c < C; c++) {
          float sm = softmax_vals[i * C + c];
          float indicator = (c == target) ? 1.0f : 0.0f;
          gl[i * C + c] =
              upstream * (sm - indicator) / static_cast<float>(batch);
        }
      }
      return std::vector<Tensor>{grad_logits};
    };
    loss.set_requires_grad(true);
    loss.set_grad_fn(node);
  }

  return loss;
}

}  // namespace tiramisu::nn