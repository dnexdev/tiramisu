#include <cstdio>
#include <vector>
#include <fstream>
#include "tiramisu/core/tensor.hpp"
#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/nn/sequential.hpp"
#include "tiramisu/nn/linear.hpp"
#include "tiramisu/nn/loss.hpp"
#include "tiramisu/optim/adam.hpp"

using namespace tiramisu;
using namespace tiramisu::nn;
using namespace tiramisu::optim;
using namespace tiramisu::autograd;

uint32_t swap_endian(uint32_t val) {
    return __builtin_bswap32(val);
}

Tensor load_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open: " + path);
    uint32_t magic, num, rows, cols;
    file.read((char*)&magic, 4); magic = swap_endian(magic);
    file.read((char*)&num, 4);   num = swap_endian(num);
    file.read((char*)&rows, 4);  rows = swap_endian(rows);
    file.read((char*)&cols, 4);  cols = swap_endian(cols);

    Tensor t({(int64_t)num, (int64_t)(rows * cols)}, DType::Float32);
    float* data = t.data<float>();

    for (uint32_t i = 0; i < num * rows * cols; ++i) {
        unsigned char pixel;
        file.read((char*)&pixel, 1);
        data[i] = (static_cast<float>(pixel) / 255.0f - 0.5f) / 0.5f; // [-1, 1]
    }
    return t;
}

Tensor load_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open: " + path);
    uint32_t magic, num;
    file.read((char*)&magic, 4); magic = swap_endian(magic);
    file.read((char*)&num, 4);   num = swap_endian(num);

    Tensor t({(int64_t)num}, DType::Float32);
    float* data = t.data<float>();
    for (int i = 0; i < (int)num; ++i) {
        unsigned char label;
        file.read((char*)&label, 1);
        data[i] = static_cast<float>(label);
    }
    return t;
}

int main() {
    Tensor train_x = load_mnist_images("../../data/train-images.idx3-ubyte");
    Tensor train_y = load_mnist_labels("../../data/train-labels.idx1-ubyte");

    auto model = std::make_shared<nn::Sequential>(std::vector<std::shared_ptr<nn::Module>>{
        std::make_shared<nn::Linear>(784, 128),
        std::make_shared<nn::Linear>(128, 10)
    });
    optim::Adam opt(model->parameters(), 0.001f);

    const int batch_size = 64;
    int num_train = train_x.shape()[0];

    for (int epoch = 0; epoch < 10; epoch++) {
        float total_loss = 0.0f;
        for (int b = 0; b * batch_size < num_train; b++) {
            int start = b * batch_size;
            int end = std::min(start + batch_size, num_train);

            Tensor batch_x = train_x.slice(0, start, end);
            Tensor batch_y = train_y.slice(0, start, end);

            opt.zero_grad();
            Tensor logits = model->forward(batch_x);
            Tensor loss = nn::cross_entropy_loss(logits, batch_y);
            
            autograd::backward(loss);
            opt.step();
            
            total_loss += loss.data<float>()[0];
        }
        printf("Epoch %d, Avg Loss: %.4f\n", epoch, total_loss / (num_train / batch_size));
    }
    return 0;
}