#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/core/dtype.hpp"
#include "tiramisu/core/tensor.hpp"
#include "tiramisu/nn/gpt.hpp"
#include "tiramisu/nn/layernorm.hpp"
#include "tiramisu/nn/linear.hpp"
#include "tiramisu/nn/loss.hpp"
#include "tiramisu/optim/adam.hpp"

namespace py = pybind11;
using namespace tiramisu;
using namespace tiramisu::autograd;
using namespace tiramisu::nn;
using namespace tiramisu::optim;

namespace {

std::vector<int64_t> to_shape(const py::iterable& shape) {
  std::vector<int64_t> out;
  for (py::handle item : shape) {
    out.push_back(item.cast<int64_t>());
  }
  return out;
}

Tensor tensor_from_list(const std::vector<int64_t>& shape,
                        const std::vector<float>& data) {
  Tensor t(shape);
  if (static_cast<int64_t>(data.size()) != t.numel()) {
    throw std::invalid_argument("data length does not match tensor numel");
  }
  std::memcpy(t.data<float>(), data.data(), data.size() * sizeof(float));
  return t;
}

Tensor tensor_from_numpy(const py::array& arr) {
  py::array_t<float, py::array::c_style | py::array::forcecast> floats(arr);
  if (floats.ndim() == 0) {
    throw std::invalid_argument("from_numpy expects at least 1-D array");
  }

  std::vector<int64_t> shape(static_cast<size_t>(floats.ndim()));
  for (ssize_t i = 0; i < floats.ndim(); ++i) {
    shape[static_cast<size_t>(i)] = floats.shape(i);
  }

  Tensor t(shape);
  std::memcpy(t.data<float>(), floats.data(),
              static_cast<size_t>(t.numel()) * sizeof(float));
  return t;
}

py::array tensor_to_numpy(Tensor& t) {
  if (t.dtype() != DType::Float32) {
    throw std::runtime_error("numpy() only supports float32 tensors");
  }
  Tensor c = t.is_contiguous() ? t : t.contiguous();
  std::vector<ssize_t> shape(c.shape().begin(), c.shape().end());
  std::vector<ssize_t> strides;
  strides.reserve(c.strides().size());
  for (int64_t s : c.strides()) {
    strides.push_back(s * static_cast<ssize_t>(sizeof(float)));
  }
  return py::array(py::buffer_info(
      c.data<float>(), sizeof(float), py::format_descriptor<float>::format(),
      shape.size(), shape, strides));
}

void tensor_backward(Tensor& self) {
  if (self.numel() != 1) {
    throw std::runtime_error("backward() expects a scalar tensor (numel == 1)");
  }
  backward(self);
}

Tensor add_scalar(const Tensor& a, float b) {
  Tensor s({1});
  s.at<float>({0}) = b;
  return add(a, s);
}

Tensor mul_scalar(const Tensor& a, float b) {
  Tensor s({1});
  s.at<float>({0}) = b;
  return mul(a, s);
}

GPTConfig make_gpt_config(int64_t vocab_size, int64_t d_model, int64_t num_heads,
                          int64_t num_layers, int64_t max_seq_len) {
  return GPTConfig{
      .vocab_size = vocab_size,
      .d_model = d_model,
      .num_heads = num_heads,
      .num_layers = num_layers,
      .max_seq_len = max_seq_len,
  };
}

std::shared_ptr<GPT> make_gpt(int64_t vocab_size, int64_t d_model, int64_t num_heads,
                               int64_t num_layers, int64_t max_seq_len) {
  return std::make_shared<GPT>(make_gpt_config(vocab_size, d_model, num_heads,
                                               num_layers, max_seq_len));
}

py::list tensor_ptr_list(const std::vector<Tensor*>& params, py::object owner) {
  py::list out;
  for (Tensor* p : params) {
    out.append(py::cast(*p, py::return_value_policy::reference_internal, owner));
  }
  return out;
}

std::vector<Tensor*> parse_param_list(const py::list& params) {
  std::vector<Tensor*> out;
  out.reserve(params.size());
  for (py::handle item : params) {
    out.push_back(&item.cast<Tensor&>());
  }
  return out;
}

std::shared_ptr<Adam> make_adam(const py::list& params, float lr, float beta1,
                                float beta2, float eps) {
  return std::make_shared<Adam>(parse_param_list(params), lr, beta1, beta2, eps);
}

}  // namespace

PYBIND11_MODULE(_C, m) {
  m.doc() = "tiramisu C++ extension";

  py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
      .def(py::init([](const py::iterable& shape) {
             return Tensor(to_shape(shape));
           }),
           py::arg("shape"))
      .def(py::init(&tensor_from_list), py::arg("shape"), py::arg("data"))
      .def_buffer([](Tensor& t) -> py::buffer_info {
        if (t.dtype() != DType::Float32) {
          throw std::runtime_error("buffer protocol only supports float32");
        }
        if (!t.is_contiguous()) {
          throw std::runtime_error(
              "buffer protocol requires contiguous tensor; call contiguous()");
        }
        std::vector<ssize_t> shape(t.shape().begin(), t.shape().end());
        std::vector<ssize_t> strides;
        strides.reserve(t.strides().size());
        for (int64_t s : t.strides()) {
          strides.push_back(s * static_cast<ssize_t>(sizeof(float)));
        }
        return py::buffer_info(
            t.data<float>(), sizeof(float),
            py::format_descriptor<float>::format(), shape.size(), shape,
            strides);
      })
      .def_property(
          "requires_grad",
          [](Tensor& self) { return self.requires_grad(); },
          [](Tensor& self, bool value) { self.set_requires_grad(value); })
      .def_property_readonly("grad", [](Tensor& self) -> py::object {
        const std::shared_ptr<Tensor>& g = self.grad();
        if (!g) {
          return py::none();
        }
        return py::cast(*g);
      })
      .def("backward", &tensor_backward)
      .def("shape", &Tensor::shape)
      .def("numel", &Tensor::numel)
      .def("is_contiguous", &Tensor::is_contiguous)
      .def("contiguous", &Tensor::contiguous)
      .def("reshape", &Tensor::reshape)
      .def("slice", &Tensor::slice, py::arg("dim"), py::arg("start"),
           py::arg("end"))
      .def("numpy", &tensor_to_numpy)
      .def("__repr__", [](const Tensor& t) {
        std::string s = "Tensor(shape=[";
        const auto& shape = t.shape();
        for (size_t i = 0; i < shape.size(); ++i) {
          if (i > 0) {
            s += ", ";
          }
          s += std::to_string(shape[i]);
        }
        s += "], requires_grad=";
        s += t.requires_grad() ? "True" : "False";
        s += ")";
        return s;
      });

  m.def("from_numpy", &tensor_from_numpy, py::arg("array"));

  m.def("backward", &backward, py::arg("loss"));

  m.def("add", py::overload_cast<const Tensor&, const Tensor&>(&add));
  m.def("add", &add_scalar, py::arg("a"), py::arg("b"));
  m.def("mul", py::overload_cast<const Tensor&, const Tensor&>(&mul));
  m.def("mul", &mul_scalar, py::arg("a"), py::arg("b"));
  m.def("sub", py::overload_cast<const Tensor&, const Tensor&>(&sub));
  m.def("div", py::overload_cast<const Tensor&, const Tensor&>(&div));
  m.def("neg", &neg);
  m.def("relu", &relu);
  m.def("gelu", &gelu);
  m.def("softmax", &softmax);
  m.def("sum", &sum);
  m.def("mean", &mean);
  m.def("matmul", &matmul);
  m.def("reshape", &reshape);
  m.def("contiguous", &contiguous);
  m.def("transpose", &transpose);

  py::module_ nn = m.def_submodule("nn");

  py::class_<Linear, std::shared_ptr<Linear>>(nn, "Linear")
      .def(py::init<int64_t, int64_t>(), py::arg("in_features"),
           py::arg("out_features"))
      .def("forward", &Linear::forward)
      .def("parameters",
           [](Linear& self) { return tensor_ptr_list(self.parameters(), py::cast(self)); });

  py::class_<LayerNorm, std::shared_ptr<LayerNorm>>(nn, "LayerNorm")
      .def(py::init<int64_t, float>(), py::arg("features"),
           py::arg("eps") = 1e-5f)
      .def("forward", &LayerNorm::forward)
      .def("parameters",
           [](LayerNorm& self) { return tensor_ptr_list(self.parameters(), py::cast(self)); });

  py::class_<GPT, std::shared_ptr<GPT>>(nn, "GPT")
      .def(py::init(&make_gpt), py::arg("vocab_size"), py::arg("d_model"),
           py::arg("num_heads") = 2, py::arg("num_layers") = 2,
           py::arg("max_seq_len") = 256)
      .def("forward", &GPT::forward)
      .def("parameters",
           [](GPT& self) { return tensor_ptr_list(self.parameters(), py::cast(self)); })
      .def("config", &GPT::config);

  nn.def("cross_entropy_loss", &cross_entropy_loss, py::arg("logits"),
         py::arg("targets"));

  py::module_ optim = m.def_submodule("optim");

  py::class_<Adam, std::shared_ptr<Adam>>(optim, "Adam")
      .def(py::init(&make_adam), py::arg("parameters"), py::arg("lr") = 1e-3f,
           py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f,
           py::arg("eps") = 1e-8f)
      .def("step", &Adam::step)
      .def("zero_grad", &Adam::zero_grad);
}
