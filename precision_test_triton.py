import math
import torch


torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
import os,sys


from rocwmma_fattn.FlashAttn import FlashAttentionFunction


#(B, H, N, D) = 1, 20, 576, 64
#Nkv = 227
from triton_fused_attention import _attention
triton_fttn = _attention.apply

(B, H, N, D) = 1, 24, 1024, 64
Nkv = 1024
dtype = torch.float16

ref_sdp_dtype = torch.bfloat16
causal = False

if __name__ == "__main__":
    
    # qkv for rocwmma fttn, q2k2v2 for sdp as reference, q3k3v3 for triton fttn.
    # 
    q = torch.rand((B, H, N, D), dtype=ref_sdp_dtype, device="cuda")   # * 5
    k = torch.rand((B, H, Nkv, D), dtype=ref_sdp_dtype, device="cuda") # * 75
    v = torch.rand((B, H, Nkv, D), dtype=ref_sdp_dtype, device="cuda") # * 15

    q2 = q.clone().detach().requires_grad_(True)
    k2 = k.clone().detach().requires_grad_(True)
    v2 = v.clone().detach().requires_grad_(True)
    
    q3 = q.clone().detach().to(dtype).requires_grad_(True)
    k3 = k.clone().detach().to(dtype).requires_grad_(True)
    v3 = v.clone().detach().to(dtype).requires_grad_(True)
    
    q = q.to(dtype).requires_grad_(True)
    k = k.to(dtype).requires_grad_(True)
    v = v.to(dtype).requires_grad_(True)

    fttn = FlashAttentionFunction()
    o1 = fttn.apply(q, k, v,None, causal, False)
    o2 = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
    o3 = triton_fttn(q3,k3,v3,causal, q.shape[-1] ** -0.5)
    
    
    maxdiff1 = (o2 - o1).abs().max().item()
    maxdiff2 = (o2 - o3).abs().max().item()

    print(o1.cpu()[0, 0, :, :])
    print(o2.cpu()[0, 0, :, :])
    print(o3.cpu()[0, 0, :, :])


    dO = torch.ones_like(q)

    o1.backward(dO)
    o2.backward(dO)
    o3.backward(dO)

    dQ1 = q.grad.clone().detach()
    dK1 = k.grad.clone().detach()
    dV1 = v.grad.clone().detach()

    dQ2 = q2.grad.clone().detach()
    dK2 = k2.grad.clone().detach()
    dV2 = v2.grad.clone().detach()


    dQ3 = q3.grad.clone().detach()
    dK3 = k3.grad.clone().detach()
    dV3 = v3.grad.clone().detach()

    print('')

    print("rocwmma  dQ",dQ1.cpu()[0,-1,:,:] )
    print('torchsdp dQ',dQ2.cpu()[0,-1,:,:] )
    print("triton   dQ",dQ3.cpu()[0,-1,:,:] )
    print('')
    
    print("rocwmma  dK",dK1.cpu()[0,-1,:,:] )
    print('torchsdp dK',dK2.cpu()[0,-1,:,:] )
    print("triton   dK",dK3.cpu()[0,-1,:,:] )
    print('')
    
    print("rocwmma  dV",dV1.cpu()[0,-1,:,:] )
    print('torchsdp dV',dV2.cpu()[0,-1,:,:] )
    print("triton   dV",dV3.cpu()[0,-1,:,:] )
    print('')
    
    print("fwd diff rocwmma-sdp:", maxdiff1)
    print("fwd diff triton-sdp:", maxdiff2)
    print('')
    
    print(f"dQ diff rocwmma-sdp:{(dQ1 - dQ2).abs().max().item()}")
    print(f"dK diff rocwmma-sdp:{(dK1 - dK2).abs().max().item()}")
    print(f"dV diff rocwmma-sdp:{(dV1 - dV2).abs().max().item()}")
    print('')
    
    print(f"dQ diff triton-sdp:{(dQ3 - dQ2).abs().max().item()}")
    print(f"dK diff triton-sdp:{(dK3 - dK2).abs().max().item()}")
    print(f"dV diff triton-sdp:{(dV3 - dV2).abs().max().item()}")
    
    
    
