 
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
src_code = ["host.cpp", "kernel_hgemm_AB_T.cu"]
src_code = [os.path.join(src_Path, x) for x in src_code]
import torch.utils.cpp_extension

hgemmTest1 = torch.utils.cpp_extension.load(
    name="hgemmTest1",
    sources=src_code,
    extra_cuda_cflags=[
        "-Ofast",
        "-save-temps",
        "-mcumode",
        "-ffast-math",
    ],
    build_directory=build_path,
)

m, n, k =  16384, 8192, 128
dtype = torch.float16
matmul_ops = 2*m*n*k
 
A = torch.rand((m, k), dtype=dtype, device="cuda")
B = torch.rand((n, k), dtype=dtype, device="cuda")
C = torch.zeros((m, n), dtype=dtype, device="cuda") 
C2 = C.clone().detach()

D2 = hgemmTest1.forward(A, B, C2, m, n, k)
D = torch.matmul(A, B.T) + C

max_diff = (D2 - D).abs().max().item()
print(D2.cpu())
print(D.cpu())
print("diff",max_diff)


round = 100

for _ in range(10):
    D = torch.matmul(A, B.T) + C
torch.cuda.synchronize()
t0 = time.time()
for i in range(round): 
    C[1,1] += 1
    D = torch.matmul(A, B.T) + C
torch.cuda.synchronize()

t1 = time.time() - t0
print("torch:", t1/round)
print(f"TFlops:{matmul_ops/(t1/round) * 1e-12}")


for _ in range(10):
    D2 = hgemmTest1.forward(A, B, C, m, n, k)
torch.cuda.synchronize()
t0 = time.time()
for i in range(round): 
    C[1,1] += 1
    D2 = hgemmTest1.forward(A, B, C, m, n, k)
torch.cuda.synchronize()
t1 = time.time() - t0
print("wmma:", t1/round)
print(f"TFlops:{matmul_ops/(t1/round) * 1e-12}")
