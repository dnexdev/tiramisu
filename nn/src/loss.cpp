#include "tiramisu/nn/loss.hpp"
#include "tiramisu/autograd/ops.hpp"

namespace tiramisu::nn {

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
  //log-sum-exp LSE(x) = max(x) + log(sigma(exp(x - max(x))))

  // max(x)
  Tensor max_logits = tiramisu::autograd::max(logits, 1, true);
  // x - max(x)
  Tensor shifted_logits = tiramisu::autograd::sub(logits, max_logits);
  // exp(x-max(x))
  Tensor exp_logits = tiramisu::autograd::exp(shifted_logits);
  // sigma(exp(x-max(x)))
  Tensor sum_exp = tiramisu::autograd::sum(exp_logits, 1, true);
  // log(sigma(exp(x-max(x)))) + max(x)
  Tensor log_sum_exp = tiramisu::autograd::add(tiramisu::autograd::log(sum_exp), max_logits);
  
  // loss = LSE - logits[target]
  Tensor log_probs = tiramisu::autograd::sub(logits, log_sum_exp);  
  return tiramisu::autograd::nll_loss(log_probs, targets);
}

}