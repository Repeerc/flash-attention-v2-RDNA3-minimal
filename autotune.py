from einops import rearrange
import matplotlib.pyplot as plt
import torch
import torch.utils

torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
import time
import os
import sys

from sko.PSO import PSO
from sko.GA import GA

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

os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "bin"))

flash_attn_wmma = torch.utils.cpp_extension.load(
    name="flash_attn_wmma",
    sources=src_code,
    extra_cuda_cflags=[
        "-Ofast",
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
        "-fgpu-flush-denormals-to-zero",
    ],
    build_directory=build_path,
)

# B H Nq Nkv d
testcase = [
    #    [2,20,1024,1024,64],
    #[2, 20, 1024, 77, 64]
         [2,10,4096,4096,64]
    #    [2,10,4096,77,64]
]

(B, H, N, D) = (2, 10, 4096, 64)
Nkv = 77
q_shape = (B, H, N, D)
v_shape = (B, H, Nkv, D)
k_shape = (B, H, Nkv, D)
causal = False
dtype = torch.float16

test_round = 100


def count_time(func):
    def wrapper(*args, **kwargs):
        torch.cuda.empty_cache()
        # torch.cuda.reset_peak_memory_stats()

        # warm up
        torch.cuda.synchronize()
        for _ in range(10):
            ret = func(*args, **kwargs)
        torch.cuda.synchronize()

        t1 = time.time()
        for _ in range(test_round):
            ret = func(*args, **kwargs)
        torch.cuda.synchronize()
        t2 = time.time() - t1
        print(f"{func.__name__}:  \texec_time:{t2:.4f}")
        return ret, t2

    return wrapper


@count_time
def fttn_rocwmma(q, k, v, Br, Bc):

    d_qkv = q.shape[-1]
    Nq = q.shape[-2]

    ret = flash_attn_wmma.forward(q, k, v, Br, Bc, causal, d_qkv**-0.5)

    O = (ret[0])[:, :, :Nq, :d_qkv]
    # L = ret[1]

    return O  # , L


def const_leq_sram(x):
    Br = (x[0] + 15.9) // 16 * 16
    Bc = (x[1] + 15.9) // 16 * 16
    d = 512
    return Br * Bc + 2 * Br * d - 32768


def func(x):
    run_t = 0

    Br = ((x[0] + 15.9) // 16) * 16
    Bc = ((x[1] + 15.9) // 16) * 16

    if const_leq_sram([Br, Bc]) >= 0:
        return 100000
    print(Br, Bc)

    for test_n in testcase:
        B = test_n[0]
        H = test_n[1]
        Nq = test_n[2]
        Nkv = test_n[3]
        D = test_n[4]
        q_shape = (B, H, N, D)
        v_shape = (B, H, Nkv, D)
        k_shape = (B, H, Nkv, D)
        q = torch.rand(q_shape, dtype=dtype, device="cuda")  # * 5
        k = torch.rand(k_shape, dtype=dtype, device="cuda")  # * 80
        v = torch.rand(v_shape, dtype=dtype, device="cuda")  # * 30
        r, t = fttn_rocwmma(q, k, v, int(Br), int(Bc))
        run_t += t
    return run_t


# print(func([16,64]))


# s = GA(func, n_dim=2,  size_pop=20, max_iter=10,
#         lb=[16, 16],ub=[512, 512],precision=16,
#         # constraint_eq=const_eq,
#         constraint_ueq=[const_leq_sram]
#         )

s = PSO(
    func,
    2,
    lb=[16, 16],
    ub=[256, 256],
    constraint_ueq=[const_leq_sram],
    pop=20,
    max_iter=10,
)

best_x, best_y = s.run()

Br = ((best_x + 15.9) // 16) * 16
Bc = ((best_y + 15.9) // 16) * 16

print("Br:", Br, "\n", "Bc:", Bc)
print(best_y)
