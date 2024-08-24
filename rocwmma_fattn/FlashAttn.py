

import os
import sys

import torch
if sys.platform.startswith("win32"):
    from . import zluda_hijack_torch_hip_ext
    torch.utils.cpp_extension.IS_HIP_EXTENSION = True
    torch.version.hip = "5.7.1"
    torch.version.cuda = None
else:
    import torch.utils.cpp_extension


os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100" #;gfx1101;gfx1102;gfx1103"
src_Path = os.path.split(os.path.realpath(__file__))[0]
build_path = os.path.join(src_Path, "build")
os.makedirs(build_path, exist_ok=True)
src_code = ["host.cpp", "kernel_bf16.cu", "kernel_fp16.cu"]
src_code = [os.path.join(src_Path, x) for x in src_code]

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



class FlashAttentionFunction(torch.autograd.Function):
    
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask=None, causal=None, scale=None, BNHD_fmt=False, *args, **kwargs):
        
        D = q.shape[3]
        
        N = q.shape[2]
        Nkv = k.shape[2]
        
        Br = 64
        Bc = 128
        
        if BNHD_fmt:
            N = q.shape[1]
            Nkv = k.shape[1]
            
        if scale is None:
            scale = D**-0.5
        if D > 384:
            Br = 32
            Bc = 128
           
        ret = flash_attn_wmma.forward(q,k,v,Br,Bc, causal, scale, BNHD_fmt)

        o, q_bwd, k_bwd, v_bwd, o_bwd, L = ret
        
        if q.requires_grad:
            ctx.args = (causal, scale, mask, N, Nkv, D, BNHD_fmt)
            ctx.save_for_backward(q_bwd, k_bwd, v_bwd, o_bwd, L)
        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        causal, scale, mask, N, Nkv, D, BNHD_fmt = ctx.args
        q, k, v, o, L = ctx.saved_tensors
        Br = 128
        Bc = 128
        dQ, dK, dV = flash_attn_wmma.backward(q, 
                                              k,
                                              v,
                                              o,
                                              do,
                                              L,N,Nkv,D,
                                              Br, Bc,causal, scale, BNHD_fmt) 
        return dQ, dK, dV, None, None, None, None




