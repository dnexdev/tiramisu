#include "tiramisu/core/tensor.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace tiramisu {

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size());

  if (shape.empty()) {
    return strides;
  }

  int64_t current_stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
    strides[i] = current_stride;
    current_stride *= shape[i];
  }

  return strides;
}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Device device)
    : shape_(std::move(shape)), offset_(0) {
  strides_ = contiguous_strides(shape_);
  int64_t elements = numel();

  storage_ = std::make_shared<Storage>(elements, dtype, device);
}

Tensor::Tensor(std::shared_ptr<Storage> storage, std::vector<int64_t> shape,
               std::vector<int64_t> strides, std::size_t offset)
    : storage_(std::move(storage)),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      offset_(offset) {
  if (!storage_) {
    throw std::invalid_argument("Tensor view created with null Storage.");
  }
  if (shape_.size() != strides_.size()) {
    throw std::invalid_argument("Shape and strides must have the same rank.");
  }
}

const std::vector<int64_t>& Tensor::shape() const { return shape_; }

const std::vector<int64_t>& Tensor::strides() const { return strides_; }

int64_t Tensor::numel() const {
  // if (shape_.empty()) {
  //   return 1;
  // }
  int64_t count = 1;
  for (int64_t dim : shape_) {
    count *= dim;
  }
  return count;
}

bool Tensor::is_contiguous() const {
  auto expected_strides = contiguous_strides(shape_);
  return strides_ == expected_strides;
}

DType Tensor::dtype() const { return storage_->dtype(); }

Device Tensor::device() const { return storage_->device(); }

template <typename T>
T* Tensor::data() {
  if (tiramisu::dtype_of<T>() != dtype()) {
    throw std::runtime_error(
        "Tensor::data<T>(): T does not match this tensor's dtype.");
  }
  return reinterpret_cast<T*>(storage_->data()) + offset_;
}

template <typename T>
T& Tensor::at(std::initializer_list<int64_t> indices) {
  if (indices.size() != shape_.size()) {
    throw std::out_of_range("Rank mismatch in Tensor::at()");
  }

  int64_t flat_idx = 0;
  auto idx_it = indices.begin();

  for (size_t i = 0; i < shape_.size(); i++, idx_it++) {
    if (*idx_it < 0 || *idx_it >= shape_[i]) {
      throw std::out_of_range("Index out of bounds in Tensor::at()");
    }
    flat_idx += (*idx_it) * strides_[i];
  }

  return data<T>()[flat_idx];
}

Tensor Tensor::view(std::vector<int64_t> new_shape) const {
  if (!is_contiguous()) {
    throw std::runtime_error("Cannot view a non-contiguous tensor.");
  }

  int64_t new_numel = 1;
  for (int64_t dim : new_shape) {
    new_numel *= dim;
  }

  if (new_numel != numel()) {
    throw std::invalid_argument(
        "View shape must have the same number of elements.");
  }

  std::vector<int64_t> new_strides = contiguous_strides(new_shape);
  return Tensor(storage_, std::move(new_shape), std::move(new_strides),
                offset_);
}

Tensor Tensor::permute(std::vector<int64_t> dims) const {
  if (dims.size() != shape_.size()) {
    throw std::invalid_argument("Permute dimensions must match tensor rank.");
  }

  // need to verify that dims is also actually a permutation of current
  uint32_t seen = 0;
  int64_t rank = shape_.size();
  for (int64_t d : dims) {
    if (d < 0 || d >= rank) {
      throw std::runtime_error("Range already exists.");
    }
    uint32_t bit = 1u << d;
    if (seen & bit) {
      throw std::runtime_error("Duplicate dimension in permute.");
    }
    seen |= bit;
  }

  std::vector<int64_t> new_shape(shape_.size());
  std::vector<int64_t> new_strides(strides_.size());

  for (size_t i = 0; i < dims.size(); i++) {
    int64_t d = dims[i];
    if (d < 0 || d >= static_cast<int64_t>(shape_.size())) {
      throw std::out_of_range("Invalid dimension in permute.");
    }

    new_shape[i] = shape_[d];
    new_strides[i] = strides_[d];
  }

  return Tensor(storage_, std::move(new_shape), std::move(new_strides),
                offset_);
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
  int64_t rank = shape_.size();
  if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank) {
    throw std::out_of_range("Invalid dimension in transpose.");
  }

  std::vector<int64_t> dims(rank);
  std::iota(dims.begin(), dims.end(), 0);
  std::swap(dims[dim0], dims[dim1]);

  return permute(dims);
}

Tensor Tensor::contiguous() const {
  if (is_contiguous()) {
    return *this;
  }

  Tensor result(shape_, dtype(), device());

  std::size_t itemsize = tiramisu::dtype_size(dtype());
  std::byte* dst = result.storage_->data();

  const std::byte* src_base = storage_->data() + (offset_ * itemsize);

  int64_t total_elements = numel();
  int64_t rank = shape_.size();

  for (int64_t i = 0; i < total_elements; i++) {
    int64_t temp = i;
    int64_t src_flat_idx = 0;

    for (int d = rank - 1; d >= 0; d--) {
      int64_t coord = temp % shape_[d];
      src_flat_idx += coord * strides_[d];
      temp /= shape_[d];
    }

    std::memcpy(dst + i * itemsize, src_base + src_flat_idx * itemsize,
                itemsize);
  }

  return result;
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end) const {
  if (dim != 0)
    throw std::runtime_error("Only dim 0 slicing supported for now");

  size_t new_offset = offset_ + start * strides_[0];

  std::vector<int64_t> new_shape = shape_;
  new_shape[0] = end - start;

  return Tensor(storage_, new_shape, strides_, new_offset);
}

void Tensor::accumulate_grad(const Tensor& g) {
  Tensor c_g = g.contiguous();

  if (!autograd_state_->grad) {
    Tensor new_grad(c_g.shape(), c_g.dtype(), c_g.device());
    std::copy(c_g.data<float>(), c_g.data<float>() + c_g.numel(),
              new_grad.data<float>());
    autograd_state_->grad = std::make_shared<Tensor>(new_grad);
  } else {
    float* current_grad_data = autograd_state_->grad->data<float>();
    const float* new_grad_data = c_g.data<float>();
    for (int64_t i = 0; i < autograd_state_->grad->numel(); i++) {
      current_grad_data[i] += new_grad_data[i];
    }
  }
}

template float* Tensor::data<float>();
template float& Tensor::at<float>(std::initializer_list<int64_t>);

template int32_t* Tensor::data<int32_t>();
template int32_t& Tensor::at<int32_t>(std::initializer_list<int64_t>);

template <typename T>
const T* Tensor::data() const {
  if (tiramisu::dtype_of<T>() != dtype()) {
    throw std::runtime_error("Tensor::data<T>() const: DType mismatch.");
  }
  return reinterpret_cast<const T*>(storage_->data()) + offset_;
}
template const float* Tensor::data<float>() const;
template const int32_t* Tensor::data<int32_t>() const;

}  // namespace tiramisu