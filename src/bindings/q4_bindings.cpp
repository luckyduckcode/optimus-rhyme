#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../kernels/q4_kernel.h"

namespace py = pybind11;

py::bytes pack_q4_binding(py::bytes input_bytes) {
    std::string s = input_bytes;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(s.data());
    size_t n = s.size();
    
    if (n % 2 != 0) {
        throw std::runtime_error("Input size must be even");
    }
    
    std::vector<uint8_t> dst(n / 2);
    q4_pack(src, dst.data(), n);
    
    return py::bytes(reinterpret_cast<const char*>(dst.data()), dst.size());
}

py::bytes unpack_q4_binding(py::bytes input_bytes, size_t n) {
    std::string s = input_bytes;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(s.data());
    
    if (s.size() * 2 != n) {
         // This check is loose because n is passed by user, but strictly packed size is n/2
         if (s.size() < n/2) throw std::runtime_error("Input buffer too small");
    }
    
    std::vector<uint8_t> dst(n);
    q4_unpack(src, dst.data(), n);
    
    return py::bytes(reinterpret_cast<const char*>(dst.data()), dst.size());
}

PYBIND11_MODULE(q4_kernel, m) {
    m.doc() = "Q4 quantization kernel module";
    m.def("pack_q4", &pack_q4_binding, "Pack 8-bit integers to 4-bit packed format");
    m.def("unpack_q4", &unpack_q4_binding, "Unpack 4-bit packed format to 8-bit integers");
}
