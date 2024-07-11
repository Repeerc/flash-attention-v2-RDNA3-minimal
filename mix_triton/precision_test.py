import math
import torch


torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
import os,sys


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
        "-fgpu-flush-denormals-to-zero"
    ],
    build_directory=build_path,
)


def pad_to_multiple(tensor, multiple, dim=-1, val = 0):
    length = tensor.size(dim)
    remainder = length % multiple
    if remainder == 0:
        return tensor
    padding_length = multiple - remainder
    padding_shape = list(tensor.shape)
    padding_shape[dim] = padding_length
    padding_tensor = torch.zeros(padding_shape, device=tensor.device, dtype=tensor.dtype) + val
    return torch.cat([tensor, padding_tensor], dim=dim).contiguous()


from triton_fused_attention import _attn_bwd_preprocess, _attn_bwd
class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask=None, causal=None, *args, **kwargs):
        Br = 64
        Bc = 256
        
        B = q.shape[0]
        H = q.shape[1]
        N = q.shape[2]
        D = q.shape[3]
        Nkv = k.shape[2]
        scale = D**-0.5

        q = pad_to_multiple(q, 256, 2, -1)
        k = pad_to_multiple(k, 256, 2, -100)
        v = pad_to_multiple(v, 256, 2)
        o = torch.zeros_like(q)
        
        ret = flash_attn_wmma.forward(q,k,v,64,256, causal)
        o = ret[0]
        L = ret[1]

        ctx.args = (causal, scale, mask, Br, Bc, N, Nkv)
        ctx.save_for_backward(q, k, v, o, L)

        return o[:,:,:N,:]

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        causal, scale, mask, Br, Bc, N, Nkv = ctx.args
        q, k, v, o, L = ctx.saved_tensors
        k = pad_to_multiple(k, q.shape[-2], -2, -1).contiguous()
        v = pad_to_multiple(v, q.shape[-2], -2, -1).contiguous()
        do = pad_to_multiple(do, q.shape[-2], 2).contiguous()
        q = q.contiguous()
        o = o.contiguous()
#        assert do.is_contiguous()
#        print(q.shape,k.shape,v.shape,o.shape,do.shape)
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        HEAD_DIM = q.shape[-1]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(L)
        
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, arg_k, v, scale, do, dq, dk, dv,  #
            L, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            N_HEAD, N_CTX,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )
        
        return dq[:,:,:N,:], dk[:,:,:Nkv,:], dv[:,:,:Nkv,:], None, None,None,None

#(B, H, N, D) = 1, 20, 576, 64
#Nkv = 227


(B, H, N, D) = 1, 20, 1024, 64
Nkv = 1024
dtype = torch.float16
causal = False

if __name__ == "__main__":
    q = torch.rand((B, H, N, D), dtype=dtype, device="cuda")   #  * 5
    k = torch.rand((B, H, Nkv, D), dtype=dtype, device="cuda") #  * 75
    v = torch.rand((B, H, Nkv, D), dtype=dtype, device="cuda") #  * 15

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    q2 = q.clone().detach().to(torch.float32).requires_grad_(True)
    k2 = k.clone().detach().to(torch.float32).requires_grad_(True)
    v2 = v.clone().detach().to(torch.float32).requires_grad_(True)

    fttn = FlashAttentionFunction()
    o1 = fttn.apply(q, k, v,None, causal)

    o2 = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)

    maxdiff = (o2 - o1).abs().max().item()

    print(o1.cpu()[0, 0, :, :])
    print(o2.cpu()[0, 0, :, :])


    dO = torch.ones_like(q)

    o1.backward(dO)
    o2.backward(dO)

    dQ1 = q.grad.clone().detach()
    dK1 = k.grad.clone().detach()
    dV1 = v.grad.clone().detach()

    dQ2 = q2.grad.clone().detach()
    dK2 = k2.grad.clone().detach()
    dV2 = v2.grad.clone().detach()


    print("FTTN dQ",dQ1.cpu()[0,-1,:,:] )
    print('PT dQ',dQ2.cpu()[0,-1,:,:] )
    
    print("FTTN dK",dK1.cpu()[0,-1,:,:] )
    print('PT dK',dK2.cpu()[0,-1,:,:] )
    
    print("FTTN dV",dV1.cpu()[0,-1,:,:] )
    print('PT dV',dV2.cpu()[0,-1,:,:] )
    
    print("fwd diff:", maxdiff)
    print(f"dQ diff:{(dQ1 - dQ2).abs().max().item()}")
    print(f"dK diff:{(dK1 - dK2).abs().max().item()}")
    print(f"dV diff:{(dV1 - dV2).abs().max().item()}")
