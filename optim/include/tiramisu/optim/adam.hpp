#pragma once
#include <vector>
#include "tiramisu/core/tensor.hpp"

namespace tiramisu::optim {

class Adam {
  public:
    Adam(const std::vector<Tensor*>& parameters,
         float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    
    void step();
    void zero_grad();

  private:
    std::vector<Tensor*> parameters_;
    std::vector<Tensor> m_; // first moment (momentum)
    std::vector<Tensor> v_; // second moment (adaptive)
    float lr_, beta1_, beta2_, eps_;
    int64_t t_; 
};

}