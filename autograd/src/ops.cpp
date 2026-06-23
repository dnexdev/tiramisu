#include "tiramisu/autograd/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

#include "tiramisu/autograd/grad_mode.hpp"
#include "tiramisu/core/cuda_memory.hpp"
#include "tiramisu/core/node.hpp"
#include "tiramisu/ops/elementwise.hpp"
#include "tiramisu/ops/embedding.hpp"
#include "tiramisu/ops/matmul.hpp"
#include "tiramisu/ops/normalization.hpp"
#include "tiramisu/ops/reduce.hpp"

#ifdef TIRAMISU_CUDA_ENABLED
#include "tiramisu/ops/cuda_ops.hpp"
#endif

namespace tiramisu::autograd {

static Tensor reduce_grad_to(const Tensor& grad,
                             const std::vector<int64_t>& target_shape) {
  if (grad.shape() == target_shape) {
    return grad;
  }

  Tensor result = grad.contiguous();
  while (result.shape().size() > target_shape.size()) {
#ifdef TIRAMISU_CUDA_ENABLED
    if (result.device() == Device::CUDA) {
      result = ops::cuda::reduce_sum_first_dim(result);
      continue;
    }
#endif
    int64_t rows = result.shape()[0];
    std::vector<int64_t> out_shape(result.shape().begin() + 1,
                                    result.shape().end());
    int64_t cols = result.numel() / rows;

    Tensor summed(out_shape, result.dtype(), result.device());
    float* dst = summed.data<float>();
    const float* src = result.data<float>();
    cuda_mem::fill_f32(dst, 0.0f, cols, result.device());

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
      return std::vector<Tensor>{
          reduce_grad_to(tiramisu::ops::mul(grad_output, b), a.shape()),
          reduce_grad_to(tiramisu::ops::mul(grad_output, a), b.shape())};
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
  cuda_mem::fill_f32(ones_grad.data<float>(), 1.0f, ones_grad.numel(),
                     loss.device());
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
      Tensor grad_a = tiramisu::ops::div(grad_output, b);
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
    cuda_mem::copy_bytes(out.data<float>(), out_val.data<float>(),
                         static_cast<std::size_t>(out.numel()) * sizeof(float),
                         out.device(), out.device());
    node->backward_fn = [out_val](const Tensor& grad_output) {
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
#ifdef TIRAMISU_CUDA_ENABLED
      if (t.device() == Device::CUDA) {
        return std::vector<Tensor>{ops::cuda::relu_backward(grad_output, t)};
      }
#endif
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

Tensor gelu(const Tensor& t) {
  Tensor out = tiramisu::ops::gelu(t);
  if (grad_enabled() && t.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {t};
    Tensor t_val = t.contiguous();
    node->backward_fn = [t_val](const Tensor& grad_output) {
#ifdef TIRAMISU_CUDA_ENABLED
      if (t_val.device() == Device::CUDA) {
        return std::vector<Tensor>{ops::cuda::gelu_backward(grad_output, t_val)};
      }
#endif
      constexpr float kSqrt2OverPi = 0.7978845608f;
      constexpr float kCoeff = 0.044715f;

      Tensor g_c = grad_output.contiguous();
      Tensor grad_t(t_val.shape(), t_val.dtype(), t_val.device());

      const float* x = t_val.data<float>();
      const float* g = g_c.data<float>();
      float* gx = grad_t.data<float>();

      for (int64_t i = 0; i < t_val.numel(); i++) {
        float xi = x[i];
        float x2 = xi * xi;
        float x3 = x2 * xi;
        float inner = kSqrt2OverPi * (xi + kCoeff * x3);
        float tanh_inner = std::tanh(inner);
        float sech2 = 1.0f - tanh_inner * tanh_inner;
        float inner_deriv = kSqrt2OverPi * (1.0f + 3.0f * kCoeff * x2);
        float d_out_dx =
            0.5f * (1.0f + tanh_inner) +
            0.5f * xi * sech2 * inner_deriv;
        gx[i] = g[i] * d_out_dx;
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
#ifdef TIRAMISU_CUDA_ENABLED
      const float g_val = grad_output.device() == Device::CUDA
                              ? cuda_mem::read_scalar_f32(grad_output)
                              : grad_output.data<float>()[0];
#else
      const float g_val = grad_output.data<float>()[0];
#endif
      cuda_mem::fill_f32(grad_t.data<float>(), g_val, grad_t.numel(),
                         grad_t.device());

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
#ifdef TIRAMISU_CUDA_ENABLED
      const float g_val = grad_output.device() == Device::CUDA
                              ? cuda_mem::read_scalar_f32(grad_output)
                              : grad_output.data<float>()[0];
#else
      const float g_val = grad_output.data<float>()[0];
#endif
      cuda_mem::fill_f32(grad_t.data<float>(), g_val / static_cast<float>(n),
                         grad_t.numel(), grad_t.device());

      return std::vector<Tensor>{grad_t};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

namespace {

std::vector<int64_t> inverse_permutation(const std::vector<int64_t>& dims,
                                         int64_t rank) {
  std::vector<int64_t> inv(rank);
  for (int64_t i = 0; i < rank; ++i) {
    int64_t d = dims[i];
    if (d < 0) {
      d += rank;
    }
    inv[d] = i;
  }
  return inv;
}

}  // namespace

Tensor contiguous(const Tensor& x) {
  if (x.is_contiguous()) {
    return x;
  }

  Tensor out = x.contiguous();
  if (grad_enabled() && x.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {x};
    node->backward_fn = [x](const Tensor& grad_output) {
#ifdef TIRAMISU_CUDA_ENABLED
      if (x.device() == Device::CUDA) {
        return std::vector<Tensor>{ops::cuda::contiguous_backward(grad_output, x)};
      }
#endif
      Tensor grad_in(x.shape(), x.dtype(), x.device());
      cuda_mem::fill_f32(grad_in.data<float>(), 0.0f, grad_in.numel(),
                         x.device());

      const float* src = grad_output.data<float>();
      float* dst = grad_in.data<float>();
      const int64_t total_elements = x.numel();
      const int64_t rank = static_cast<int64_t>(x.shape().size());

      for (int64_t i = 0; i < total_elements; ++i) {
        int64_t temp = i;
        int64_t dst_flat_idx = 0;
        for (int64_t d = rank - 1; d >= 0; --d) {
          const int64_t coord = temp % x.shape()[d];
          dst_flat_idx += coord * x.strides()[d];
          temp /= x.shape()[d];
        }
        dst[dst_flat_idx] = src[i];
      }

      return std::vector<Tensor>{grad_in};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor reshape(const Tensor& x, std::vector<int64_t> new_shape) {
  if (!grad_enabled() || !x.requires_grad()) {
    return x.reshape(std::move(new_shape));
  }

  Tensor src = x.is_contiguous() ? x : contiguous(x);
  Tensor out = src.view(new_shape);

  auto node = std::make_shared<Node>();
  node->inputs = {src};
  const std::vector<int64_t> src_shape = src.shape();
  node->backward_fn = [src_shape](const Tensor& grad_output) {
    return std::vector<Tensor>{grad_output.reshape(src_shape)};
  };
  out.set_requires_grad(true);
  out.set_grad_fn(node);
  return out;
}

Tensor permute(const Tensor& x, std::vector<int64_t> dims) {
  if (!grad_enabled() || !x.requires_grad()) {
    return x.permute(dims);
  }

  Tensor out = x.permute(dims);
  auto node = std::make_shared<Node>();
  node->inputs = {x};
  const int64_t rank = static_cast<int64_t>(x.shape().size());
  const std::vector<int64_t> inv = inverse_permutation(dims, rank);
  node->backward_fn = [inv](const Tensor& grad_output) {
    return std::vector<Tensor>{grad_output.permute(inv)};
  };
  out.set_requires_grad(true);
  out.set_grad_fn(node);
  return out;
}

Tensor transpose(const Tensor& x, int64_t dim0, int64_t dim1) {
  const int64_t rank = static_cast<int64_t>(x.shape().size());
  if (dim0 < 0) {
    dim0 += rank;
  }
  if (dim1 < 0) {
    dim1 += rank;
  }

  std::vector<int64_t> dims(rank);
  std::iota(dims.begin(), dims.end(), 0);
  std::swap(dims[dim0], dims[dim1]);
  return permute(x, dims);
}

Tensor merge_heads(const Tensor& x, int64_t d_model) {
  if (x.shape().size() != 4) {
    throw std::invalid_argument("merge_heads: input must be 4D");
  }

  const int64_t batch = x.shape()[0];
  const int64_t heads = x.shape()[1];
  const int64_t seq = x.shape()[2];
  const int64_t d_k = x.shape()[3];
  if (heads * d_k != d_model) {
    throw std::invalid_argument("merge_heads: heads * d_k must equal d_model");
  }

#ifdef TIRAMISU_CUDA_ENABLED
  if (x.device() == Device::CUDA) {
    Tensor out = ops::cuda::merge_heads(x, d_model);
    if (grad_enabled() && x.requires_grad()) {
      auto node = std::make_shared<Node>();
      node->inputs = {x};
      node->backward_fn = [batch, heads, seq, d_k,
                           d_model](const Tensor& grad_output) {
        return std::vector<Tensor>{ops::cuda::merge_heads_backward(
            grad_output, batch, heads, seq, d_k, d_model)};
      };
      out.set_requires_grad(true);
      out.set_grad_fn(node);
    }
    return out;
  }
#endif

  Tensor out({batch, seq, d_model}, x.dtype(), x.device());
  Tensor c_x = x.contiguous();
  const float* src = c_x.data<float>();
  float* dst = out.data<float>();

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t h = 0; h < heads; ++h) {
        for (int64_t d = 0; d < d_k; ++d) {
          const int64_t src_idx = ((b * heads + h) * seq + s) * d_k + d;
          const int64_t dst_idx = (b * seq + s) * d_model + h * d_k + d;
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }

  if (grad_enabled() && x.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {x};
    node->backward_fn = [batch, heads, seq, d_k,
                         d_model](const Tensor& grad_output) {
#ifdef TIRAMISU_CUDA_ENABLED
      if (grad_output.device() == Device::CUDA) {
        return std::vector<Tensor>{ops::cuda::merge_heads_backward(
            grad_output, batch, heads, seq, d_k, d_model)};
      }
#endif
      Tensor grad_in({batch, heads, seq, d_k}, grad_output.dtype(),
                     grad_output.device());
      cuda_mem::fill_f32(grad_in.data<float>(), 0.0f, grad_in.numel(),
                         grad_output.device());

      Tensor c_grad = grad_output.contiguous();
      const float* gsrc = c_grad.data<float>();
      float* gdst = grad_in.data<float>();

      for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
          for (int64_t h = 0; h < heads; ++h) {
            for (int64_t d = 0; d < d_k; ++d) {
              const int64_t src_idx = (b * seq + s) * d_model + h * d_k + d;
              const int64_t dst_idx = ((b * heads + h) * seq + s) * d_k + d;
              gdst[dst_idx] += gsrc[src_idx];
            }
          }
        }
      }

      return std::vector<Tensor>{grad_in};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }

  return out;
}

Tensor embedding(const Tensor& weight, const Tensor& indices) {
  Tensor out = tiramisu::ops::embedding(weight, indices);

  if (grad_enabled() && weight.requires_grad()) {
    const int64_t batch = indices.shape()[0];
    const int64_t seq = indices.shape()[1];
    const int64_t vocab = weight.shape()[0];
    const int64_t dim = weight.shape()[1];
    Tensor c_indices = indices.contiguous();
    auto node = std::make_shared<Node>();
    node->inputs = {weight, indices};
    node->backward_fn = [c_indices, batch, seq, vocab,
                         dim](const Tensor& grad_output) {
#ifdef TIRAMISU_CUDA_ENABLED
      if (grad_output.device() == Device::CUDA) {
        Tensor grad_weight =
            ops::cuda::embedding_backward(grad_output, c_indices, vocab, dim);
        Tensor grad_indices({1});
        return std::vector<Tensor>{grad_weight, grad_indices};
      }
#endif
      Tensor grad_weight({vocab, dim}, grad_output.dtype(), grad_output.device());
      cuda_mem::fill_f32(grad_weight.data<float>(), 0.0f, vocab * dim,
                         grad_output.device());

      Tensor c_grad = grad_output.contiguous();
      const float* g = c_grad.data<float>();
      const float* index_data = c_indices.data<float>();
      float* gw = grad_weight.data<float>();

      for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < seq; s++) {
          const int64_t token = static_cast<int64_t>(index_data[b * seq + s]);
          float* gw_row = gw + token * dim;
          const float* g_row = g + (b * seq + s) * dim;
          for (int64_t d = 0; d < dim; d++) {
            gw_row[d] += g_row[d];
          }
        }
      }

      Tensor grad_indices({1});
      return std::vector<Tensor>{grad_weight, grad_indices};
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
    cuda_mem::copy_bytes(out.data<float>(), out_val.data<float>(),
                         static_cast<std::size_t>(out.numel()) * sizeof(float),
                         out.device(), out.device());
    node->backward_fn = [out_val](const Tensor& grad_output) {
#ifdef TIRAMISU_CUDA_ENABLED
      if (out_val.device() == Device::CUDA) {
        return std::vector<Tensor>{
            ops::cuda::softmax_backward(grad_output, out_val)};
      }
#endif
      auto shape = out_val.shape();
      int64_t N = shape.back();
      int64_t rows = out_val.numel() / N;

      Tensor c_grad = grad_output.contiguous();
      Tensor c_out = out_val.contiguous();
      Tensor grad_x(shape, out_val.dtype(), out_val.device());

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
#ifdef TIRAMISU_CUDA_ENABLED
      if (x.device() == Device::CUDA) {
        Tensor grad_gamma(gamma.shape(), gamma.dtype(), Device::CUDA);
        Tensor grad_beta(gamma.shape(), gamma.dtype(), Device::CUDA);
        Tensor grad_x = ops::cuda::layernorm_backward(
            grad_output, x, gamma, eps, grad_gamma, grad_beta);
        return std::vector<Tensor>{grad_x, grad_gamma, grad_beta};
      }
#endif
      int64_t N = x.shape().back();
      int64_t rows = x.numel() / N;

      Tensor grad_x(x.shape(), x.dtype(), x.device());
      Tensor grad_gamma(gamma.shape(), gamma.dtype(), gamma.device());
      Tensor grad_beta(gamma.shape(), gamma.dtype(), gamma.device());

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