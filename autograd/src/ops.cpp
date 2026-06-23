#include "tiramisu/autograd/ops.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include "tiramisu/autograd/grad_mode.hpp"
#include "tiramisu/core/node.hpp"
#include "tiramisu/ops/elementwise.hpp"
#include "tiramisu/ops/matmul.hpp"
#include "tiramisu/ops/normalization.hpp"
#include "tiramisu/ops/reduce.hpp"

namespace tiramisu::autograd {

// sum grad_output down to target_shape by summing over any leading
// dims that were broadcast
// ex: grad_output (64, 10), target_shape (10,) -> sum over dim 0 -> (10,)
static Tensor reduce_grad_to(const Tensor& grad,
                             const std::vector<int64_t>& target_shape) {
  if (grad.shape() == target_shape) {
    return grad;
  }

  Tensor result = grad.contiguous();
  while (result.shape().size() > target_shape.size()) {
    int64_t rows = result.shape()[0];
    std::vector<int64_t> out_shape(result.shape().begin() + 1,
                                    result.shape().end());
    int64_t cols = result.numel() / rows;

    Tensor summed(out_shape);
    float* dst = summed.data<float>();
    const float* src = result.data<float>();
    std::fill_n(dst, cols, 0.0f);

    for (int64_t r = 0; r < rows; r++) {
      for (int64_t c = 0; c < cols; c++) {
        dst[c] += src[r * cols + c];
      }
    }
    result = summed;
  }

  return result;
}

Tensor add(const Tensor& a, const Tensor& b) {
  Tensor out = tiramisu::ops::add(a, b);

  if (grad_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto node = std::make_shared<Node>();
    node->inputs = {a, b};
    node->backward_fn = [a, b](const Tensor& grad_output) {
      return std::vector<Tensor>{reduce_grad_to(grad_output, a.shape()),
                                 reduce_grad_to(grad_output, b.shape())};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
  Tensor out = tiramisu::ops::mul(a, b);

  if (grad_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto node = std::make_shared<Node>();
    node->inputs = {a, b};
    node->backward_fn = [a, b](const Tensor& grad_output) {
      return std::vector<Tensor>{tiramisu::ops::mul(grad_output, b),
                                 tiramisu::ops::mul(grad_output, a)};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

void backward(const Tensor& loss) {
  std::vector<Tensor> topo;
  std::unordered_set<Node*> visited;

  std::function<void(const Tensor&)> build_topo = [&](const Tensor& t) {
    auto n = t.grad_fn();
    if (!n || visited.count(n.get())) return;

    visited.insert(n.get());
    for (const auto& input : n->inputs) {
      build_topo(input);
    }
    topo.push_back(t);
  };

  build_topo(loss);

  Tensor ones_grad(loss.shape(), loss.dtype(), loss.device());
  std::fill_n(ones_grad.data<float>(), ones_grad.numel(), 1.0f);
  Tensor loss_mut = loss;
  loss_mut.accumulate_grad(ones_grad);

  NoGradGuard guard;

  for (auto it = topo.rbegin(); it != topo.rend(); it++) {
    Tensor& t = *it;
    auto node = t.grad_fn();
    if (!node || !t.grad()) continue;
    auto grads = node->backward_fn(*t.grad());

    for (size_t i = 0; i < node->inputs.size(); i++) {
      if (node->inputs[i].requires_grad()) {
        Tensor input_mut = node->inputs[i];
        input_mut.accumulate_grad(grads[i]);
      }
    }
  }
}

Tensor sub(const Tensor& a, const Tensor& b) {
  Tensor out = tiramisu::ops::sub(a, b);
  if (grad_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto node = std::make_shared<Node>();
    node->inputs = {a, b};
    node->backward_fn = [a, b](const Tensor& grad_output) {
      return std::vector<Tensor>{
          reduce_grad_to(grad_output, a.shape()),
          reduce_grad_to(tiramisu::ops::neg(grad_output), b.shape())};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
  Tensor out = tiramisu::ops::div(a, b);
  if (grad_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto node = std::make_shared<Node>();
    node->inputs = {a, b};
    node->backward_fn = [a, b](const Tensor& grad_output) {
      // grad_a = grad_output / b
      Tensor grad_a = tiramisu::ops::div(grad_output, b);

      // grad_b = -grad_output * a / (b * b)
      Tensor b_sq = tiramisu::ops::mul(b, b);
      Tensor neg_grad = tiramisu::ops::neg(grad_output);
      Tensor num = tiramisu::ops::mul(neg_grad, a);
      Tensor grad_b = tiramisu::ops::div(num, b_sq);

      return std::vector<Tensor>{grad_a, grad_b};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor neg(const Tensor& t) {
  Tensor out = tiramisu::ops::neg(t);
  if (grad_enabled() && t.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {t};
    node->backward_fn = [](const Tensor& grad_output) {
      // grad_t = -grad_output
      return std::vector<Tensor>{tiramisu::ops::neg(grad_output)};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor exp(const Tensor& t) {
  Tensor out = tiramisu::ops::exp(t);
  if (grad_enabled() && t.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {t};
    Tensor out_val(out.shape(), out.dtype(), out.device());
    std::copy(out.data<float>(), out.data<float>() + out.numel(),
              out_val.data<float>());
    node->backward_fn = [out_val](const Tensor& grad_output) {
      // grad_t = grad_output * (detached out)
      return std::vector<Tensor>{tiramisu::ops::mul(grad_output, out_val)};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor log(const Tensor& t) {
  Tensor out = tiramisu::ops::log(t);
  if (grad_enabled() && t.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {t};
    node->backward_fn = [t](const Tensor& grad_output) {
      return std::vector<Tensor>{tiramisu::ops::div(grad_output, t)};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor relu(const Tensor& t) {
  Tensor out = tiramisu::ops::relu(t);
  if (grad_enabled() && t.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {t};
    node->backward_fn = [t](const Tensor& grad_output) {
      Tensor t_c = t.contiguous();
      Tensor g_c = grad_output.contiguous();
      Tensor grad_t(t_c.shape(), t_c.dtype(), t_c.device());

      const float* t_data = t_c.data<float>();
      const float* g_data = g_c.data<float>();
      float* gt_data = grad_t.data<float>();

      int64_t n = t_c.numel();
      for (int64_t i = 0; i < n; i++) {
        gt_data[i] = t_data[i] > 0.0f ? g_data[i] : 0.0f;
      }

      return std::vector<Tensor>{grad_t};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor sum(const Tensor& t) {
  Tensor out = tiramisu::ops::sum(t);
  if (grad_enabled() && t.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {t};
    auto t_shape = t.shape();
    node->backward_fn = [t_shape](const Tensor& grad_output) {
      Tensor grad_t(t_shape, grad_output.dtype(), grad_output.device());
      float g_val = grad_output.data<float>()[0];
      std::fill_n(grad_t.data<float>(), grad_t.numel(), g_val);

      return std::vector<Tensor>{grad_t};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor mean(const Tensor& t) {
  Tensor out = tiramisu::ops::mean(t);
  if (grad_enabled() && t.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {t};
    auto t_shape = t.shape();
    auto n = t.numel();
    node->backward_fn = [t_shape, n](const Tensor& grad_output) {
      Tensor grad_t(t_shape, grad_output.dtype(), grad_output.device());

      float g_val = grad_output.data<float>()[0] / static_cast<float>(n);
      std::fill_n(grad_t.data<float>(), grad_t.numel(), g_val);

      return std::vector<Tensor>{grad_t};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  Tensor out = tiramisu::ops::matmul(a, b);
  if (grad_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto node = std::make_shared<Node>();
    node->inputs = {a, b};
    node->backward_fn = [a, b](const Tensor& grad_output) {
      auto transpose_last_two = [](const Tensor& t) {
        int64_t rank = static_cast<int64_t>(t.shape().size());
        return t.transpose(rank - 2, rank - 1);
      };

      Tensor grad_a =
          tiramisu::ops::matmul(grad_output, transpose_last_two(b));
      grad_a = reduce_grad_to(grad_a, a.shape());

      Tensor grad_b =
          tiramisu::ops::matmul(transpose_last_two(a), grad_output);
      grad_b = reduce_grad_to(grad_b, b.shape());

      return std::vector<Tensor>{grad_a, grad_b};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}


Tensor softmax(const Tensor& x) {
  Tensor out = tiramisu::ops::softmax(x);
  if (grad_enabled() && x.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {x};
    Tensor out_val(out.shape(), out.dtype(), out.device());
    std::copy(out.data<float>(), out.data<float>() + out.numel(),
              out_val.data<float>());
    node->backward_fn = [out_val](const Tensor& grad_output) {
      auto shape = out_val.shape();
      int64_t N = shape.back();
      int64_t rows = out_val.numel() / N;

      Tensor c_grad = grad_output.contiguous();
      Tensor c_out = out_val.contiguous();
      Tensor grad_x(shape);

      const float* go = c_grad.data<float>();
      const float* out_data = c_out.data<float>();
      float* gx = grad_x.data<float>();

      for (int64_t r = 0; r < rows; r++) {
        const float* go_row = go + r * N;
        const float* out_row = out_data + r * N;
        float* gx_row = gx + r * N;

        float dot = 0.0f;
        for (int64_t i = 0; i < N; i++) {
          dot += go_row[i] * out_row[i];
        }
        for (int64_t i = 0; i < N; i++) {
          gx_row[i] = out_row[i] * (go_row[i] - dot);
        }
      }
      return std::vector<Tensor>{grad_x};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                 float eps) {
  Tensor out = tiramisu::ops::layernorm(x, gamma, beta, eps);
  if (grad_enabled() && (x.requires_grad() || gamma.requires_grad() || beta.requires_grad())) {
    auto node = std::make_shared<Node>();
    node->inputs = {x, gamma, beta};
    node->backward_fn = [x, gamma, beta, eps](const Tensor& grad_output) {
      int64_t N = x.shape().back();
      int64_t rows = x.numel() / N;

      Tensor grad_x(x.shape());
      Tensor grad_gamma(gamma.shape());
      Tensor grad_beta(gamma.shape());

      float* gg = grad_gamma.data<float>();
      float* gb = grad_beta.data<float>();
      std::fill_n(gg, N, 0.0f);
      std::fill_n(gb, N, 0.0f);

      const float* x_data = x.contiguous().data<float>();
      const float* go_data = grad_output.contiguous().data<float>();
      const float* gamma_data = gamma.data<float>();
      float* gx_data = grad_x.data<float>();

      for (int64_t r = 0; r < rows; r++) {
        const float* x_row = x_data + r * N;
        const float* go_row = go_data + r * N;
        float* gx_row = gx_data + r * N;

        float mean = 0.0f;
        for (int64_t i = 0; i < N; i++) mean += x_row[i];
        mean /= N;

        float var = 0.0f;
        for (int64_t i = 0; i < N; i++) {
          float diff = x_row[i] - mean;
          var += diff * diff;
        }
        var /= N;

        float inv_std = 1.0f / std::sqrt(var + eps);

        float grad_var = 0.0f;
        float grad_mean = 0.0f;
        
        for (int64_t i = 0; i < N; i++) {
          float x_hat = (x_row[i] - mean) * inv_std;
          float grad_x_hat = go_row[i] * gamma_data[i];
          
          gg[i] += go_row[i] * x_hat;
          gb[i] += go_row[i];

          grad_var += grad_x_hat * (x_row[i] - mean) * -0.5f * inv_std * inv_std * inv_std;
          grad_mean += grad_x_hat * -inv_std;
        }

        for (int64_t i = 0; i < N; i++) {
          float grad_x_hat = go_row[i] * gamma_data[i];
          gx_row[i] = grad_x_hat * inv_std + 
                      grad_var * 2.0f * (x_row[i] - mean) / N + grad_mean / N;
        }
      }

      return std::vector<Tensor>{grad_x, grad_gamma, grad_beta};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

}  // namespace tiramisu::autograd