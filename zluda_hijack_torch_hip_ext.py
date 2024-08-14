import importlib
import sys
import os
import torch
import torch.utils

os.environ["ROCM_HOME"] = os.environ["HIP_PATH"]
os.environ["CUDA_HOME"] = "" 
torch.version.hip = "5.7.0"
torch.version.cuda = None
#import torch.utils.cpp_extension
module_name = "torch.utils.cpp_extension"
spec = importlib.util.find_spec(module_name, None)
source: str = spec.loader.get_source(module_name)
module = importlib.util.module_from_spec(spec)

source = source.replace(
    """elif IS_WINDOWS:
        raise OSError('Building PyTorch extensions using '
                      'ROCm and Windows is not supported.')""",
    """#
       #
       #""",
)

source = source.replace(
    """if IS_WINDOWS:
            extra_ldflags.append(f'/LIBPATH:{_join_cuda_home("lib", "x64")}')
            extra_ldflags.append('cudart.lib')
            if CUDNN_HOME is not None:
                extra_ldflags.append(f'/LIBPATH:{os.path.join(CUDNN_HOME, "lib", "x64")}')""",
    """ #
        #
        #
        #
        #""",
)


# source = source.replace(
# '''COMMON_HIPCC_FLAGS = [
#     '-DCUDA_HAS_FP16=1',
#     '-D__HIP_NO_HALF_OPERATORS__=1',
#     '-D__HIP_NO_HALF_CONVERSIONS__=1',
# ]''',
# '''COMMON_HIPCC_FLAGS = [
# ]'''
# )


# pytorch 2.2.1
source = source.replace(
    """COMMON_HIP_FLAGS = [
    '-fPIC',
    '-D__HIP_PLATFORM_HCC__=1',
    '-DUSE_ROCM=1',
]""",
    """COMMON_HIP_FLAGS = [
    '-D__HIP_PLATFORM_HCC__=1',
    #
    #
]""",
)

# pytorch 2.3.0
source = source.replace(
    """COMMON_HIP_FLAGS = [
    '-fPIC',
    '-D__HIP_PLATFORM_AMD__=1',
    '-DUSE_ROCM=1',
]""",
    """COMMON_HIP_FLAGS = [
    '-D__HIP_PLATFORM_AMD__=1',
    #
    #
]""",
)

source = source.replace(
    """COMMON_MSVC_FLAGS = ['/MD', '/wd4819', '/wd4251', '/wd4244', '/wd4267', '/wd4275', '/wd4018', '/wd4190', '/wd4624', '/wd4067', '/wd4068', '/EHsc']""",
    """COMMON_MSVC_FLAGS = ['/MT', '/wd4819', '/wd4251', '/wd4244', '/wd4267', '/wd4275', '/wd4018', '/wd4190', '/wd4624', '/wd4067', '/wd4068', '/EHsc']""",
)

source = source.replace(
    """extra_ldflags.append(f'-L{_join_rocm_home("lib")}')
            extra_ldflags.append('-lamdhip64' if ROCM_VERSION >= (3, 5) else '-lhip_hcc')""",
    """extra_ldflags.append(f'/LIBPATH:{_join_rocm_home("lib")}')
            extra_ldflags.append('amdhip64.lib')""",
)

source = source.replace(
    "_join_rocm_home('bin', 'hipcc')", 
    "_join_rocm_home('bin', 'hipcc.bin.exe')"
)
source = source.replace("cuda_cflags.append(common_cflag)", "")

source = source.replace(
    "command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags",
    "command = cl $cflags -c $in /Fo$out $post_cflags",
)
source = source.replace(
    "cuda_flags = ['-DWITH_HIP'] + cflags + COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS",
    "cuda_flags = ['-DWITH_HIP'] + ['-Wno-ignored-attributes'] + common_cflags + ['-std=c++17'] + extra_cflags + COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS",
)


source = source.replace(
    "hipified_sources.add(hipify_result[s_abs].hipified_path if s_abs in hipify_result else s_abs)", 
    "hipified_sources.add(hipify_result[s_abs].hipified_path if (s_abs in hipify_result and hipify_result[s_abs].hipified_path is not None) else s_abs)"
)


# print(source)
codeobj = compile(source, module.__spec__.origin, "exec")
exec(codeobj, module.__dict__)
sys.modules[module_name] = module
torch.utils.cpp_extension = module
