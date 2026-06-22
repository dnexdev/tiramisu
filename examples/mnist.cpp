#include <cstdio>
#include <vector>
#include "tiramisu/core/tensor.hpp"
#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/nn/sequential.hpp"
#include "tiramisu/nn/linear.hpp"
#include "tiramisu/nn/loss.hpp"
#include "tiramisu/optim/adam.hpp"

using namespace tiramisu;

// TODO: load MNIST data (60k training, 10k test images)
// For now, just a shape/API sketch

int main() {
  // Build a simple 2-layer MLP: 784 -> 128 -> 10
  auto model = std::make_shared<nn::Sequential>(
    std::vector<std::shared_ptr<nn::Module>>{
      std::make_shared<nn::Linear>(784, 128),
      std::make_shared<nn::Linear>(128, 10)
    }
  );
  
  auto params = model->parameters();
  optim::Adam opt(params, 0.001f);
  
  // Dummy training loop — replace with real MNIST later
  for (int epoch = 0; epoch < 10; epoch++) {
    // TODO: iterate over MNIST batches
    // for each batch:
    //   opt.zero_grad();
    //   logits = model->forward(batch_x);
    //   loss = nn::cross_entropy_loss(logits, batch_y);
    //   autograd::backward(loss);
    //   opt.step();
    //   if (epoch % 10 == 0) printf("loss: %f\n", loss.data<float>()[0]);
  }
  
  printf("Training complete\n");
  return 0;
}