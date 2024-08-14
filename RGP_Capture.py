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


os.add_dll_directory(os.path.join( os.environ['HIP_PATH'] , 'bin')) 
from ck_fttn import ck_fttn_pyb



(B, H, N, D) = 1, 24, 4096, 64
Nkv = 4096
dtype = torch.float16

sc = D**-0.5
causal = False

q = torch.rand((B,  N,   H, D), dtype=dtype, device="cuda")    
k = torch.rand((B,  Nkv, H, D), dtype=dtype, device="cuda") 
v = torch.rand((B,  Nkv, H, D), dtype=dtype, device="cuda")  

ret =  ck_fttn_pyb.fwd(q,k,v, None, 0, sc, causal, False, None) # BNHD

ret =  ck_fttn_pyb.fwd(q,k,v, None, 0, sc, causal, False, None) # BNHD


# time.sleep(3)
# exit()

def fwd(q,k,v):
    Br = 64
    Bc = 256
    scale = q.shape[-1]**-0.5
    causal = False
    ret = flash_attn_wmma.forward(q, k, v, Br, Bc, causal, scale)
    return ret
    

q = torch.rand((B, H, N, D), dtype=dtype, device="cuda")    
k = torch.rand((B, H, Nkv, D), dtype=dtype, device="cuda") 
v = torch.rand((B, H, Nkv, D), dtype=dtype, device="cuda")  

o, qpad,kpad,vpad,opad,L = fwd(q,k,v)

print(ret)
print(o)

time.sleep(3)
exit()

dO = torch.ones_like(q)
dQ, dK, dV = flash_attn_wmma.backward(
    q, k, v, o, dO, L, N, Nkv, D, 128, 64, causal, q.shape[-1] ** -0.5
)

print(dQ,dK,dV)

time.sleep(5)

