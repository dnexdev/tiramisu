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
#include "tiramisu/core/device.hpp"
#include "tiramisu/core/dtype.hpp"
#include "tiramisu/core/tensor.hpp"
#include "tiramisu/nn/gpt.hpp"
#include "tiramisu/nn/layernorm.hpp"
#include "tiramisu/nn/linear.hpp"
#include "tiramisu/nn/loss.hpp"
#include "tiramisu/optim/adam.hpp"
#include "tiramisu/optim/adamw.hpp"
#include "tiramisu/optim/grad_clip.hpp"
#include "tiramisu/optim/lr_scheduler.hpp"
#include "tiramisu/optim/sgd.hpp"
#include "tiramisu/serialize/checkpoint.hpp"

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

Device coerce_device(const py::object& obj) {
  if (obj.is_none()) {
    return Device::CPU;
  }
  if (py::isinstance<py::str>(obj)) {
    std::string s = obj.cast<std::string>();
    for (char& c : s) c = static_cast<char>(std::tolower(c));
    if (s == "cpu") return Device::CPU;
    if (s == "cuda" || s == "gpu") return Device::CUDA;
    throw std::invalid_argument("device: expected 'cpu' or 'cuda', got '" + s +
                                "'");
  }
  return obj.cast<Device>();
}

Tensor tensor_from_numpy(const py::array& arr, const py::object& device) {
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
  const Device d = coerce_device(device);
  return d == Device::CPU ? t : t.to(d);
}

py::array tensor_to_numpy(Tensor& t) {
  if (t.dtype() != DType::Float32) {
    throw std::runtime_error("numpy() only supports float32 tensors");
  }
  Tensor c = t.is_contiguous() ? t : t.contiguous();
  if (c.device() == Device::CUDA) {
    c = c.to(Device::CPU);
  }
  std::vector<ssize_t> shape(c.shape().begin(), c.shape().end());
  std::vector<ssize_t> strides;
  strides.reserve(c.strides().size());
  for (int64_t s : c.strides()) {
    strides.push_back(s * static_cast<ssize_t>(sizeof(float)));
  }
  // c is a fresh contiguous CPU copy (or was already CPU-contiguous); py::array
  // will copy its buffer here so we don't hand out a dangling pointer.
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
                          int64_t num_layers, int64_t max_seq_len,
                          bool tie_weights) {
  return GPTConfig{
      .vocab_size = vocab_size,
      .d_model = d_model,
      .num_heads = num_heads,
      .num_layers = num_layers,
      .max_seq_len = max_seq_len,
      .tie_weights = tie_weights,
  };
}

std::shared_ptr<GPT> make_gpt(int64_t vocab_size, int64_t d_model, int64_t num_heads,
                               int64_t num_layers, int64_t max_seq_len,
                               bool tie_weights, const py::object& device) {
  return std::make_shared<GPT>(
      make_gpt_config(vocab_size, d_model, num_heads, num_layers, max_seq_len,
                      tie_weights),
      coerce_device(device));
}

std::shared_ptr<Linear> make_linear(int64_t in_features, int64_t out_features,
                                    const py::object& device) {
  return std::make_shared<Linear>(in_features, out_features,
                                  coerce_device(device));
}

std::shared_ptr<LayerNorm> make_layernorm(int64_t features, float eps,
                                          const py::object& device) {
  return std::make_shared<LayerNorm>(features, eps, coerce_device(device));
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

std::shared_ptr<AdamW> make_adamw(const py::list& params, float lr, float beta1,
                                  float beta2, float eps, float weight_decay) {
  return std::make_shared<AdamW>(parse_param_list(params), lr, beta1, beta2, eps,
                                 weight_decay);
}

std::shared_ptr<SGD> make_sgd(const py::list& params, float lr) {
  return std::make_shared<SGD>(parse_param_list(params), lr);
}

float clip_grad_norm_py(const py::list& params, float max_norm) {
  std::vector<Tensor*> ps = parse_param_list(params);
  return clip_grad_norm(ps, max_norm);
}

void save_gpt_py(const std::string& path, GPT& model, int64_t step,
                 int64_t epoch) {
  serialize::save_gpt_model(path, model, step, epoch);
}

py::tuple load_gpt_py(const std::string& path, GPT& model) {
  int64_t step = 0;
  int64_t epoch = 0;
  serialize::load_gpt_model(path, model, &step, &epoch);
  return py::make_tuple(step, epoch);
}

}  // namespace

PYBIND11_MODULE(_C, m) {
  m.doc() = "tiramisu C++ extension";

  py::enum_<Device>(m, "Device")
      .value("CPU", Device::CPU)
      .value("CUDA", Device::CUDA);

  m.def(
      "cuda_available",
      []() {
#ifdef TIRAMISU_CUDA_ENABLED
        return true;
#else
        return false;
#endif
      },
      "True if this build was compiled with CUDA support.");

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
        if (t.device() != Device::CPU) {
          throw std::runtime_error(
              "buffer protocol requires a CPU tensor; call .cpu() or .numpy()");
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
      .def_property_readonly("device", &Tensor::device)
      .def(
          "to",
          [](const Tensor& self, const py::object& device) {
            return self.to(coerce_device(device));
          },
          py::arg("device"),
          "Move the tensor to the given device ('cpu' or 'cuda').")
      .def("cpu", [](const Tensor& self) { return self.to(Device::CPU); })
      .def("cuda", [](const Tensor& self) { return self.to(Device::CUDA); })
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

  m.def("from_numpy", &tensor_from_numpy, py::arg("array"),
        py::arg("device") = py::none(),
        "Copy a numpy array into a tiramisu Tensor on the given device.");

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
      .def(py::init(&make_linear), py::arg("in_features"),
           py::arg("out_features"), py::arg("device") = py::none())
      .def("forward", &Linear::forward)
      .def("parameters",
           [](Linear& self) { return tensor_ptr_list(self.parameters(), py::cast(self)); });

  py::class_<LayerNorm, std::shared_ptr<LayerNorm>>(nn, "LayerNorm")
      .def(py::init(&make_layernorm), py::arg("features"),
           py::arg("eps") = 1e-5f, py::arg("device") = py::none())
      .def("forward", &LayerNorm::forward)
      .def("parameters",
           [](LayerNorm& self) { return tensor_ptr_list(self.parameters(), py::cast(self)); });

  py::class_<GPT, std::shared_ptr<GPT>>(nn, "GPT")
      .def(py::init(&make_gpt), py::arg("vocab_size"), py::arg("d_model"),
           py::arg("num_heads") = 2, py::arg("num_layers") = 2,
           py::arg("max_seq_len") = 256, py::arg("tie_weights") = false,
           py::arg("device") = py::none())
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

  py::class_<AdamW, std::shared_ptr<AdamW>>(optim, "AdamW")
      .def(py::init(&make_adamw), py::arg("parameters"), py::arg("lr") = 3e-4f,
           py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f,
           py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.1f)
      .def("step", static_cast<void (AdamW::*)()>(&AdamW::step))
      .def("zero_grad", &AdamW::zero_grad)
      .def_property("lr", &AdamW::lr, &AdamW::set_lr)
      .def_property(
          "step_count",
          static_cast<int64_t (AdamW::*)() const>(&AdamW::step),
          &AdamW::set_step);

  py::class_<SGD, std::shared_ptr<SGD>>(optim, "SGD")
      .def(py::init(&make_sgd), py::arg("parameters"), py::arg("lr") = 0.01f)
      .def("step", &SGD::step)
      .def("zero_grad", &SGD::zero_grad);

  optim.def("clip_grad_norm_", &clip_grad_norm_py, py::arg("parameters"),
            py::arg("max_norm"),
            "Clip parameter grads in-place to max_norm. Returns pre-clip norm.");

  py::class_<CosineAnnealingLR>(optim, "CosineAnnealingLR")
      .def(py::init<float, int64_t, float>(), py::arg("base_lr"),
           py::arg("total_steps"), py::arg("min_lr") = 0.0f)
      .def("step", &CosineAnnealingLR::step,
           "Advance one step; returns the new learning rate.")
      .def_property_readonly("current_lr", &CosineAnnealingLR::current_lr);

  py::module_ serialize = m.def_submodule("serialize");
  serialize.def("save_gpt", &save_gpt_py, py::arg("path"), py::arg("model"),
                py::arg("step") = 0, py::arg("epoch") = 0,
                "Save a GPT model's parameters + config to `path`.");
  serialize.def("load_gpt", &load_gpt_py, py::arg("path"), py::arg("model"),
                "Load parameters into `model` in-place. Returns (step, epoch).");
}
