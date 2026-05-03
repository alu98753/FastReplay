#include <pybind11/pybind11.h>
#include "fastreplay/ring_buffer_mutex.hpp"

namespace py = pybind11;

int add(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(fastreplay_baseline, m) {
    m.doc() = "FastReplay Baseline: Mutex-based SPSC ring buffer";
    m.def("add", &add, "Smoke test function", py::arg("a"), py::arg("b"));
    py::class_<fastreplay_baseline::RingBuffer>(m, "RingBuffer")
        .def(py::init<std::size_t>(), py::arg("capacity"))
        .def("push", &fastreplay_baseline::RingBuffer::push, py::arg("value"))
        .def("pop",
             [](fastreplay_baseline::RingBuffer& self) -> py::object {
                 int val = 0;
                 if (self.pop(&val))
                     return py::int_(val);
                 return py::none();
             })
        .def("size", &fastreplay_baseline::RingBuffer::size)
        .def("empty", &fastreplay_baseline::RingBuffer::empty)
        .def("capacity", &fastreplay_baseline::RingBuffer::capacity);
}
