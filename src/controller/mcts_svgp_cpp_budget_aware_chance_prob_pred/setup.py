import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "mcts_dpw_svgp_cpp_budget_aware_mcts_chance_node",
        ["src/bindings.cpp"],
        include_dirs=["include"],
        depends=glob.glob("include/*.hpp"),
        cxx_std=17,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="mcts_dpw_svgp_cpp_budget_aware_mcts_chance_node",
    version="0.1.0",
    description="C++ MCTS DPW + Krasser SVGP (C_alpha input) + GP variance sampling at chance nodes",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<2.0",
        "pybind11>=2.9.0",
    ],
)
