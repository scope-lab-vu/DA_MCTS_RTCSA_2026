import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "mcts_f1tenth_pi_svgp_cpp_c_alpha",
        ["src/bindings.cpp"],
        include_dirs=["include"],
        depends=glob.glob("include/*.hpp"),
        cxx_std=17,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="mcts_f1tenth_pi_svgp_cpp_c_alpha",
    version="0.1.0",
    description="C++ MCTS DPW + PI-SVGP (Pacejka, C_alpha input) dynamics for F1TENTH racing",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<2.0",
        "pybind11>=2.9.0",
    ],
)
