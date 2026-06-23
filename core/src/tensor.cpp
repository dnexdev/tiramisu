#include "tiramisu/core/tensor.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "tiramisu/core/cuda_memory.hpp"

namespace tiramisu {

namespace {

int64_t normalize_dim(int64_t dim, int64_t rank) {
  if (dim < 0) {
    dim += rank;
  }
  if (dim < 0 || dim >= rank) {
    throw std::out_of_range("Dimension index out of range.");
  }
  return dim;
}

}  // namespace

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
  if (device() == Device::CUDA) {
    throw std::runtime_error("Tensor::at() is not supported for CUDA tensors");
  }
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

Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
  return contiguous().view(std::move(new_shape));
}

Tensor Tensor::permute(std::vector<int64_t> dims) const {
  if (dims.size() != shape_.size()) {
    throw std::invalid_argument("Permute dimensions must match tensor rank.");
  }

  const int64_t rank = static_cast<int64_t>(shape_.size());
  std::vector<int64_t> normalized(rank);
  std::vector<bool> seen(rank, false);

  for (size_t i = 0; i < dims.size(); i++) {
    int64_t d = normalize_dim(dims[i], rank);
    if (seen[d]) {
      throw std::runtime_error("Duplicate dimension in permute.");
    }
    seen[d] = true;
    normalized[i] = d;
  }

  std::vector<int64_t> new_shape(shape_.size());
  std::vector<int64_t> new_strides(strides_.size());

  for (size_t i = 0; i < normalized.size(); i++) {
    int64_t d = normalized[i];
    new_shape[i] = shape_[d];
    new_strides[i] = strides_[d];
  }

  return Tensor(storage_, std::move(new_shape), std::move(new_strides),
                offset_);
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
  int64_t rank = static_cast<int64_t>(shape_.size());
  dim0 = normalize_dim(dim0, rank);
  dim1 = normalize_dim(dim1, rank);

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
  cuda_mem::contiguous_copy(*this, result);
  return result;
}

Tensor Tensor::to(Device device) const {
  if (device == this->device()) {
    return *this;
  }
  Tensor result(shape_, dtype(), device);
  cuda_mem::copy_bytes(data<float>(), result.data<float>(),
                       static_cast<std::size_t>(numel()) * sizeof(float),
                       this->device(), device);
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
  const Device dev = c_g.device();

  if (!autograd_state_->grad) {
    Tensor new_grad(c_g.shape(), c_g.dtype(), dev);
    cuda_mem::copy_bytes(c_g.data<float>(), new_grad.data<float>(),
                         static_cast<std::size_t>(c_g.numel()) * sizeof(float),
                         dev, dev);
    autograd_state_->grad = std::make_shared<Tensor>(new_grad);
  } else {
    if (autograd_state_->grad->device() != dev) {
      throw std::runtime_error("accumulate_grad: device mismatch");
    }
    cuda_mem::add_f32(autograd_state_->grad->data<float>(), c_g.data<float>(),
                      autograd_state_->grad->numel(), dev);
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