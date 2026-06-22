#include "tiramisu/autograd/grad_mode.hpp"

namespace tiramisu::autograd {

thread_local bool g_grad_enabled = true;

bool grad_enabled() { return g_grad_enabled; }

NoGradGuard::NoGradGuard() : previous_(g_grad_enabled) {
  g_grad_enabled = false;
}

NoGradGuard::~NoGradGuard() { g_grad_enabled = previous_; }

}  // namespace tiramisu::autograd