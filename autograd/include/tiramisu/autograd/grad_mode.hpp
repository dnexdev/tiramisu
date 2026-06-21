#pragma once

namespace tiramisu::autograd {

bool grad_enabled();

class NoGradGuard {
  public:
    NoGradGuard();
    ~NoGradGuard();
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
  
  private:
    bool previous_;
};

}