#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "tiramisu/core/device.hpp"
#include "tiramisu/core/dtype.hpp"
#include "tiramisu/core/storage.hpp"

namespace tiramisu {

// Forward declaration
struct Node;

// A Tensor is a view: shape + strides + offset describe how to walk
// the buffer. The storage owns the bytes; the Tensor just interprets
// them. transpose()/permute() never touch Storage at all.
class Tensor {
 public:
  // Fresh Tensor: allocates a new contiguous Storage sized for shape.
  Tensor(std::vector<int64_t> shape, DType dtype = DType::Float32,
         Device device = Device::CPU);

  // View constructor: wraps an existing Storage with explicit
  // shape/strides/offset. Used internally by view(), permute(), etc.
  Tensor(std::shared_ptr<Storage> storage, std::vector<int64_t> shape,
         std::vector<int64_t> strides, std::size_t offset);

  const std::vector<int64_t>& shape() const;
  const std::vector<int64_t>& strides() const;
  int64_t numel() const;
  bool is_contiguous() const;

  DType dtype() const;
  Device device() const;

  template <typename T>
  T* data();  // raw pointer into storage with offset() applied

  // Slow, bounds-checked - for debugging/tests, not hot paths.
  template <typename T>
  T& at(std::initializer_list<int64_t> indices);

  template <typename T>
  const T* data() const;

  Tensor view(std::vector<int64_t> new_shape) const;  // requires contiguous
  Tensor reshape(std::vector<int64_t> new_shape) const;
  Tensor permute(std::vector<int64_t> dims) const;
  Tensor transpose(int64_t dim0, int64_t dim1) const;  // permute special-case

  // Copies into a fresh contiguous tensor only if not already
  // contiguous; otherwise returns *this.
  Tensor contiguous() const;

  Tensor slice(int64_t dim, int64_t start, int64_t end) const;

  bool requires_grad() const { return autograd_state_->requires_grad; }
  void set_requires_grad(bool r) { autograd_state_->requires_grad = r; }
  const std::shared_ptr<Tensor>& grad() const { return autograd_state_->grad; }
  const std::shared_ptr<Node>& grad_fn() const {
    return autograd_state_->grad_fn;
  }
  void set_grad_fn(std::shared_ptr<Node> fn) {
    autograd_state_->grad_fn = std::move(fn);
  }
  void accumulate_grad(const Tensor& g);

 private:
  std::shared_ptr<Storage> storage_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
  std::size_t offset_ = 0;

  struct AutogradState {
    bool requires_grad = false;
    std::shared_ptr<Tensor> grad;
    std::shared_ptr<Node> grad_fn;
  };
  std::shared_ptr<AutogradState> autograd_state_ =
      std::make_shared<AutogradState>();
};

// Canonical row-major strides for a shape. Pure function, no
std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape);

}  // namespace tiramisu