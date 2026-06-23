#pragma once

#include "tiramisu/core/tensor.hpp"

namespace tiramisu::ops::cuda {

void assert_same_device(const Tensor& a, const Tensor& b);

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor neg(const Tensor& t);
Tensor gelu(const Tensor& t);
Tensor relu(const Tensor& t);
Tensor softmax(const Tensor& x);
Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                 float eps);
Tensor sum(const Tensor& t);
Tensor mean(const Tensor& t);
Tensor embedding(const Tensor& weight, const Tensor& indices);
Tensor merge_heads(const Tensor& x, int64_t d_model);

Tensor reduce_sum_first_dim(const Tensor& t);
Tensor gelu_backward(const Tensor& grad_output, const Tensor& input);
Tensor relu_backward(const Tensor& grad_output, const Tensor& input);
Tensor softmax_backward(const Tensor& grad_output, const Tensor& output);
Tensor layernorm_backward(const Tensor& grad_output, const Tensor& x,
                          const Tensor& gamma, float eps, Tensor& grad_gamma,
                          Tensor& grad_beta);
Tensor embedding_backward(const Tensor& grad_output, const Tensor& indices,
                          int64_t vocab, int64_t dim);
Tensor merge_heads_backward(const Tensor& grad_output, int64_t batch,
                            int64_t heads, int64_t seq, int64_t d_k,
                            int64_t d_model);
Tensor contiguous_backward(const Tensor& grad_output, const Tensor& input);
Tensor cross_entropy_forward(const Tensor& logits, const Tensor& targets,
                             Tensor& softmax_out);
Tensor cross_entropy_backward(const Tensor& grad_scalar, const Tensor& softmax,
                              const Tensor& targets, int64_t batch, int64_t C);
void adamw_step(float* param, const float* grad, float* m, float* v, int64_t n,
                float lr, float beta1, float beta2, float eps,
                float weight_decay, int64_t t);
void zero_grad(float* grad, int64_t n);

}  // namespace tiramisu::ops::cuda
