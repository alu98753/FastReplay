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
        .def("pop_view", [](fastreplay::RingBuffer& self, std::size_t n) -> py::object {
            if(n> self.size()) throw std::runtime_error("Not enough data.");

            std::size_t current_head = self.head();
            std::size_t phys_capacity = self.capacity()+1;

            if(current_head+n <= phys_capacity){
                // if contiguous -> zero copy
                auto result = py::array_t<int>(
                    {n},
                    {sizeof(int)},
                    self.data() + current_head, // start memory location
                    py::cast(self)
                );
                self.advance_head(n);
                return result;
            
            } else {
                // wrap around
                std::size_t first_chunk = phys_capacity - current_head;
                std::size_t second_chunk = n - first_chunk;
                
                auto result = py::array_t<int>(n);
                auto r = result.mutable_unchecked<1>(); // get writable reference

                // copy first part
                for (std::size_t i = 0; i < first_chunk; ++i) {
                    r(i) = self.data()[current_head + i];
                }
                // copy second part
                for (std::size_t i = 0; i < second_chunk; ++i) {
                    r(first_chunk + i) = self.data()[i];
                }
                self.advance_head(n);
                return result;
            }

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
