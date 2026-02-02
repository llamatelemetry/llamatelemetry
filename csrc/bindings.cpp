#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "core/device.h"
#include "core/tensor.h"
#include "ops/matmul.h"

namespace py = pybind11;
using namespace llamatelemetry;

PYBIND11_MODULE(llamatelemetry_cpp, m) {
    m.doc() = "llamatelemetry native CUDA operations";

    // DType enum
    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float16", DType::Float16)
        .value("BFloat16", DType::BFloat16)
        .value("Int32", DType::Int32)
        .value("Int64", DType::Int64)
        .value("UInt8", DType::UInt8)
        .export_values();

    // DeviceProperties struct
    py::class_<DeviceProperties>(m, "DeviceProperties")
        .def_readonly("device_id", &DeviceProperties::device_id)
        .def_readonly("name", &DeviceProperties::name)
        .def_readonly("total_memory", &DeviceProperties::total_memory)
        .def_readonly("compute_capability_major", &DeviceProperties::compute_capability_major)
        .def_readonly("compute_capability_minor", &DeviceProperties::compute_capability_minor)
        .def_readonly("multiprocessor_count", &DeviceProperties::multiprocessor_count)
        .def_readonly("max_threads_per_block", &DeviceProperties::max_threads_per_block)
        .def_readonly("max_threads_per_multiprocessor", &DeviceProperties::max_threads_per_multiprocessor)
        .def_readonly("warp_size", &DeviceProperties::warp_size)
        .def("__repr__", [](const DeviceProperties& props) {
            return "<DeviceProperties(" + props.name + ", SM " +
                   std::to_string(props.compute_capability_major) + "." +
                   std::to_string(props.compute_capability_minor) + ")>";
        });

    // Device class
    py::class_<Device>(m, "Device")
        .def_static("get_device_count", &Device::get_device_count,
                    "Get the number of available CUDA devices")
        .def_static("get_device_properties", &Device::get_device_properties,
                    py::arg("device_id"),
                    "Get properties of a specific device")
        .def_static("set_device", &Device::set_device,
                    py::arg("device_id"),
                    "Set the active CUDA device")
        .def_static("get_device", &Device::get_device,
                    "Get the currently active device")
        .def_static("synchronize", &Device::synchronize,
                    py::arg("device_id") = -1,
                    "Synchronize device (current device if device_id=-1)")
        .def_static("get_free_memory", &Device::get_free_memory,
                    py::arg("device_id") = -1,
                    "Get free memory in bytes")
        .def_static("get_total_memory", &Device::get_total_memory,
                    py::arg("device_id") = -1,
                    "Get total memory in bytes");

    // Tensor class
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<int64_t>&, DType, int>(),
             py::arg("shape"),
             py::arg("dtype") = DType::Float32,
             py::arg("device_id") = 0,
             "Create a new tensor with given shape and dtype")
        .def_property_readonly("shape", &Tensor::shape,
                              "Tensor shape")
        .def_property_readonly("strides", &Tensor::strides,
                              "Tensor strides")
        .def_property_readonly("dtype", &Tensor::dtype,
                              "Data type")
        .def_property_readonly("device", &Tensor::device,
                              "Device ID")
        .def("numel", &Tensor::numel,
             "Number of elements")
        .def("element_size", &Tensor::element_size,
             "Size of each element in bytes")
        .def("nbytes", &Tensor::nbytes,
             "Total size in bytes")
        .def("ndim", &Tensor::ndim,
             "Number of dimensions")
        .def("to", &Tensor::to,
             py::arg("device_id"),
             "Move tensor to specified device")
        .def("is_contiguous", &Tensor::is_contiguous,
             "Check if tensor is contiguous")
        .def("contiguous", &Tensor::contiguous,
             "Return contiguous version of tensor")
        .def_static("zeros", &Tensor::zeros,
                    py::arg("shape"),
                    py::arg("dtype") = DType::Float32,
                    py::arg("device_id") = 0,
                    "Create zero-filled tensor")
        .def_static("ones", &Tensor::ones,
                    py::arg("shape"),
                    py::arg("dtype") = DType::Float32,
                    py::arg("device_id") = 0,
                    "Create ones-filled tensor")
        .def("__repr__", [](const Tensor& t) {
            std::string shape_str = "[";
            auto shape = t.shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                shape_str += std::to_string(shape[i]);
                if (i < shape.size() - 1) shape_str += ", ";
            }
            shape_str += "]";

            std::string dtype_str;
            switch (t.dtype()) {
                case DType::Float32: dtype_str = "float32"; break;
                case DType::Float16: dtype_str = "float16"; break;
                case DType::BFloat16: dtype_str = "bfloat16"; break;
                case DType::Int32: dtype_str = "int32"; break;
                case DType::Int64: dtype_str = "int64"; break;
                case DType::UInt8: dtype_str = "uint8"; break;
            }

            return "<Tensor(shape=" + shape_str + ", dtype=" + dtype_str +
                   ", device=cuda:" + std::to_string(t.device()) + ")>";
        });

    // Operations module
    auto ops = m.def_submodule("ops", "Tensor operations");

    ops.def("matmul", &ops::matmul,
            py::arg("A"), py::arg("B"),
            "Matrix multiplication: C = A @ B");

    ops.def("batched_matmul", &ops::batched_matmul,
            py::arg("A"), py::arg("B"),
            "Batched matrix multiplication");
}
