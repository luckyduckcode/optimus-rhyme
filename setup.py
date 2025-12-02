import sys
from setuptools import setup, Extension
import pybind11

# Define platform specific flags
extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args = ["/arch:AVX2", "/O2"]
else:
    extra_compile_args = ["-mavx2", "-O3", "-march=native"]

ext_modules = [
    Extension(
        "q4_kernel",
        sorted([
            "src/bindings/q4_bindings.cpp",
            "src/kernels/q4_avx2.cpp", 
            "src/kernels/q4_fallback.cpp"
        ]),
        include_dirs=[
            pybind11.get_include(),
            "src/kernels"
        ],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
]

setup(
    name="q4_kernel",
    version="0.1.0",
    ext_modules=ext_modules,
)
