import zluda_hijack_torch_hip_ext

import math
import torch
import time

torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
import os, sys


if sys.platform.startswith("win32"):
    import zluda_hijack_torch_hip_ext

    torch.utils.cpp_extension.IS_HIP_EXTENSION = True
    torch.version.hip = "5.7.0"
    torch.version.cuda = None
else:
    import torch.utils.cpp_extension
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"  # ;gfx1101;gfx1102;gfx1103"
src_Path = os.path.split(os.path.realpath(__file__))[0]
build_path = os.path.join(src_Path, "build")
os.makedirs(build_path, exist_ok=True)
src_code = ["host.cpp", "kernel.cu"]
src_code = [os.path.join(src_Path, x) for x in src_code]
import torch.utils.cpp_extension

gemmTest1 = torch.utils.cpp_extension.load(
    name="gemmTest1",
    sources=src_code,
    extra_cuda_cflags=[
        "-O3",
        "-save-temps",
        "-DROCWMMA_ARCH_GFX1100=1",
        "-DROCWMMA_ARCH_GFX1101=1",
        "-DROCWMMA_ARCH_GFX1102=1",
        "-DROCWMMA_ARCH_GFX1103=1",
        "-DROCWMMA_ARCH_GFX11=1",
        "-DROCWMMA_WAVE32_MODE=1",
        "-DROCWMMA_BLOCK_DIM_16_SUPPORTED=1",
        "-mcumode",
        "-ffast-math",
    ],
    build_directory=build_path,
)

m, n, k = 128, 64, 64
dtype = torch.float16

A = torch.rand((m, k), dtype=dtype, device="cuda")
B = torch.rand((k, n), dtype=dtype, device="cuda")
C = torch.rand((m, n), dtype=dtype, device="cuda")

D = torch.matmul(A, B) + C
D2 = gemmTest1.forward(A, B, C, m, n, k)

round = 1000

t0 = time.time()
for _ in range(round):
    D = torch.matmul(A, B) + C
    torch.cuda.synchronize()
t1 = time.time() - t0
print("torch:", t1)

t0 = time.time()
for _ in range(round):
    D2 = gemmTest1.forward(A, B, C, m, n, k)
    torch.cuda.synchronize()
t1 = time.time() - t0
print("wmma:", t1)

max_diff = (D2 - D).abs().max().item()
print(D2.cpu())
print(D.cpu())
print(max_diff)
