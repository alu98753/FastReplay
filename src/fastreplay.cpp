#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int a, int b) { return a + b; }

PYBIND11_MODULE(fastreplay, m) {
    m.doc() = "FastReplay: atomic-based SPSC ring buffer for RL";
    m.def("add", &add, "Smoke test function",
        py::arg("a"), py::arg("b"));
}
