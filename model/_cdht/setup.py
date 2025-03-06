# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
#
# setup(
#     name='deep_hough',
#     ext_modules=[
#         CUDAExtension('deep_hough', [
#             'deep_hough_cuda.cpp',
#             'deep_hough_cuda_kernel.cu',
#         ],
#         extra_compile_args={'cxx': ['-g'], 'nvcc': ['-arch=sm_60']})
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })
#


# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# import torch.utils.cpp_extension as cpp_ext
# import os
#
# os.environ["CUDA_HOME"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
# os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
#
# # Monkey-patch the CUDA version check to bypass the mismatch error.
# def no_op(*args, **kwargs):
#     return
# cpp_ext._check_cuda_version = no_op
#
# setup(
#     name='deep_hough',
#     ext_modules=[
#         CUDAExtension(
#             'deep_hough',
#             ['deep_hough_cuda.cpp', 'deep_hough_cuda_kernel.cu'],
#             extra_compile_args={
#                 'cxx': ['/std:c++17'],
#                 # Add the flag to override NVCC's version check:
#                 'nvcc': ['--expt-relaxed-constexpr', '-allow-unsupported-compiler', '-arch=sm_75']
#             }
#         )
#     ],
#     cmdclass={'build_ext': BuildExtension}
# )


########################################################################
#
# import os
#
# # Desired MSVC 14.29 directory (from VS2019)
# msvc_1429_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
#
# # Remove variables that might point to VS2022
# for var in ["VSINSTALLDIR", "VisualStudioVersion", "VCINSTALLDIR"]:
#     if var in os.environ:
#         del os.environ[var]
#
# # Optionally, set them to the VS2019 paths instead:
# os.environ["VSINSTALLDIR"] = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional"
# os.environ["VCINSTALLDIR"] = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC"
# os.environ["VisualStudioVersion"] = "16.0"  # VS2019
#
# # Filter out any PATH entries containing Visual Studio 2022
# old_paths = os.environ.get("PATH", "").split(";")
# filtered_paths = [p for p in old_paths if "Visual Studio\\2022" not in p]
# os.environ["PATH"] = msvc_1429_path + ";" + ";".join(filtered_paths)
#
# # (Optional) Set CUDAHOSTCXX to force NVCC to use the correct cl.exe.
# os.environ["CUDAHOSTCXX"] = os.path.join(msvc_1429_path, "cl.exe")
#
# # Verify which cl.exe is found:
# os.system("where cl.exe")
#
# # Set CUDA 11.8 environment variables:
# os.environ["CUDA_HOME"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
# os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
#
# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension
#
# setup(
#     name='deep_hough',
#     ext_modules=[
#         CUDAExtension(
#             'deep_hough',
#             ['deep_hough_cuda.cpp', 'deep_hough_cuda_kernel.cu'],
#             extra_compile_args={
#                 'cxx': ['/std:c++17'],
#                 'nvcc': [
#                     '--expt-relaxed-constexpr',
#                     '-allow-unsupported-compiler',
#                     f'--compiler-bindir={msvc_1429_path}',
#                     '-gencode=arch=compute_75,code=sm_75'
#                 ]
#             }
#         )
#     ],
#     cmdclass={'build_ext': BuildExtension}
# )







from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Desired MSVC 14.29 directory (from VS2019)
msvc_1429_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"

# Remove variables that might point to VS2022
for var in ["VSINSTALLDIR", "VisualStudioVersion", "VCINSTALLDIR"]:
    if var in os.environ:
        del os.environ[var]

# Optionally, set them to the VS2019 paths instead:
os.environ["VSINSTALLDIR"] = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional"
os.environ["VCINSTALLDIR"] = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC"
os.environ["VisualStudioVersion"] = "16.0"  # VS2019

setup(
    name='deep_hough',
    zip_safe=False,  # <-- add this line
    ext_modules=[
        CUDAExtension(
            'deep_hough',
            ['deep_hough_cuda.cpp', 'deep_hough_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': ['/std:c++17'],
                'nvcc': [
                    '--expt-relaxed-constexpr',
                    '-allow-unsupported-compiler',
                    f'--compiler-bindir={msvc_1429_path}',
                    '-gencode=arch=compute_75,code=sm_75'
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)


