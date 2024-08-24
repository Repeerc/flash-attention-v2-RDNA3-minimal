import math
import torch
import time

torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
import os, sys


if sys.platform.startswith("win32"):
    import rocwmma_fattn.zluda_hijack_torch_hip_ext as zluda_hijack_torch_hip_ext
    torch.utils.cpp_extension.IS_HIP_EXTENSION = True
    torch.version.hip = "5.7.0"
    torch.version.cuda = None
else:
    import torch.utils.cpp_extension
    
    
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"  # ;gfx1101;gfx1102;gfx1103"
src_Path = os.path.split(os.path.realpath(__file__))[0]
build_path = os.path.join(src_Path, "build")
os.makedirs(build_path, exist_ok=True)
src_code = ["dummy.cpp", "GPU_peak_perf_test.cu"]
src_code = [os.path.join(src_Path, x) for x in src_code]
import torch.utils.cpp_extension

peak_perf_test = torch.utils.cpp_extension.load(
    name="peak_perf_test",
    sources=src_code,
    extra_cuda_cflags=[
        "-Ofast",
        "-save-temps",
        "-mcumode", # CU Mode:-mcumode  WGP Mode:-mno-cumode
        "-ffast-math",
    ],
    build_directory=build_path,
)

# more round and running_blocks would cause driver TDR in windows
round = 4
matmul_ops = 16 * 16 * 16 * 2 # 2MNK ops for a matrix mul add
wmma_insts = round * 500000 # 100000 * 5 for loop in .cu
running_blocks = 96  # 96 in cu mode / 48 in wgp mode (gfx1100)
waves_per_block = 16 # 2 SIMD32 per CU / 4 SIMD32 per WGP (use > 4 for latency hiding)
waves = waves_per_block * running_blocks
total_ops = wmma_insts * matmul_ops * waves

# warm up
for i in range(3):
    peak_perf_test.forward(running_blocks, waves_per_block)
torch.cuda.synchronize()

t0 = time.time()
for i in range(round):
    peak_perf_test.forward(running_blocks, waves_per_block)
torch.cuda.synchronize()
t1 = time.time() - t0

print("time:", t1 / round)
print("TFlops:", total_ops / t1 * 1e-12)
