#include "tiramisu/autograd/gradcheck.hpp"

#include <cmath>

#include "tiramisu/autograd/grad_mode.hpp"
#include "tiramisu/autograd/ops.hpp"

namespace tiramisu::autograd {

bool gradcheck(const std::function<Tensor(const Tensor&)>& f,
               const Tensor& input, double epsilon, double tolerance) {
  Tensor x = input;
  x.set_requires_grad(true);
  Tensor out = f(x);
  backward(out);

  if (!x.grad()) return false;
  const float* ana_data = x.grad()->data<float>();

  NoGradGuard guard;
  float* x_data = x.data<float>();
  int64_t numel = x.numel();

  for (int64_t i = 0; i < numel; i++) {
    float orig = x_data[i];

    x_data[i] = orig + epsilon;
    float f_plus = f(x).data<float>()[0];

    x_data[i] = orig - epsilon;
    float f_minus = f(x).data<float>()[0];

    x_data[i] = orig;

    float num_diff = (f_plus - f_minus) / (2.0 * epsilon);

    if (std::abs(num_diff - ana_data[i]) > tolerance) {
      return false;
    }
  }
  return true;
}

}  // namespace tiramisu::autograd