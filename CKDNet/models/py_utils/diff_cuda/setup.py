import sys
import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

CXX_FLAGS = [] if sys.platform == 'win32' else ['-g', '-Werror']


if torch.cuda.is_available() and CUDA_HOME is not None:
    ext_modules = [
        CUDAExtension(
            'zddiff_cuda', ['cuda/diff_cuda.cpp','cuda/diff_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']}),

    ]


setup(
    name='zddiff',
    version='0.1.0',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})