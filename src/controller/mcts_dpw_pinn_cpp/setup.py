from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "mcts_f1tenth_cpp",
        ["src/bindings.cpp"],
        include_dirs=["include"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="mcts_f1tenth_cpp",
    version="0.1.0",
    description="C++ MCTS DPW for F1TENTH racing with pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<2.0",
        "pybind11>=2.9.0",
    ],
)
