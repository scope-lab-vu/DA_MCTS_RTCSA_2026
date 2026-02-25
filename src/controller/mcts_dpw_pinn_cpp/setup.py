import glob
import torch
import torch.utils.cpp_extension
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Get LibTorch include and library paths from the installed PyTorch
torch_include = torch.utils.cpp_extension.include_paths()
torch_lib = torch.utils.cpp_extension.library_paths()

# Match the C++11 ABI flag used by PyTorch
cxx_abi_flag = "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))

ext_modules = [
    Pybind11Extension(
        "mcts_f1tenth_pinn_cpp",
        ["src/bindings.cpp"],
        include_dirs=["include"] + torch_include,
        depends=glob.glob("include/*.hpp"),
        library_dirs=torch_lib,
        runtime_library_dirs=torch_lib,
        libraries=["torch", "torch_cpu", "c10"],
        cxx_std=17,
        extra_compile_args=["-O3", cxx_abi_flag],
    ),
]

setup(
    name="mcts_f1tenth_pinn_cpp",
    version="0.1.0",
    description="C++ MCTS DPW + PINN dynamics for F1TENTH racing (LibTorch)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<2.0",
        "pybind11>=2.9.0",
        "torch",
    ],
)
