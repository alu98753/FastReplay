#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "fastreplay/ring_buffer.hpp"

namespace py = pybind11;

int add(int a, int b) { return a + b; }

PYBIND11_MODULE(fastreplay, m) {
    m.doc() = "FastReplay: atomic-based SPSC ring buffer for RL";
    m.def("add", &add, "Smoke test function",
        py::arg("a"), py::arg("b"));
    py::class_<fastreplay::RingBuffer>(m, "RingBuffer",
        py::buffer_protocol())
        .def(py::init<std::size_t>(), py::arg("capacity"))
        .def("push", &fastreplay::RingBuffer::push, py::arg("value"))
        .def("pop", [](fastreplay::RingBuffer& self) -> py::object {
            int val = 0;
            if(self.pop(&val)) return py::int_(val);
            return py::none();
        })
        .def("size", &fastreplay::RingBuffer::size)
        .def("empty", &fastreplay::RingBuffer::empty)
        .def_buffer([](fastreplay::RingBuffer& self) -> py::buffer_info{
            return py::buffer_info(
                self.data(), //data pointer address
                sizeof(int), //item size
                py::format_descriptor<int>::format(), //numpy format
                1, // one-dimensional
                {self.capacity() + 1}, //shape
                {sizeof(int)} //strides
            );
        })
        .def("capacity", &fastreplay::RingBuffer::capacity);

}
