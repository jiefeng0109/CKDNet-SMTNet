import sys
import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

CXX_FLAGS = [] if sys.platform == 'win32' else ['-g', '-Werror']


if torch.cuda.is_available() and CUDA_HOME is not None:
    ext_modules = [
        CUDAExtension(
            'top_pooling', ['src/top_pool/top_pooling_cuda.cpp','src/top_pool/top_pooling_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']}),
        CUDAExtension(
            'bottom_pooling', ['src/bottom_pool/bottom_pooling_cuda.cpp', 'src/bottom_pool/bottom_pooling_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']}),
        CUDAExtension(
            'left_pooling', ['src/left_pool/left_pooling_cuda.cpp', 'src/left_pool/left_pooling_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']}),
        CUDAExtension(
            'right_pooling', ['src/right_pool/right_pooling_cuda.cpp', 'src/right_pool/right_pooling_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']})
    ]


setup(
    name='zpool',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})