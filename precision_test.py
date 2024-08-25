import math
from einops import rearrange
import torch


torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
import os, sys



def pad_to_multiple(tensor, multiple, dim=-1, val=0):
    length = tensor.size(dim)
    remainder = length % multiple
    if remainder == 0:
        return tensor
    padding_length = multiple - remainder
    padding_shape = list(tensor.shape)
    padding_shape[dim] = padding_length
    padding_tensor = (
        torch.zeros(padding_shape, device=tensor.device, dtype=tensor.dtype) + val
    )
    return torch.cat([tensor, padding_tensor], dim=dim)




# (B, H, N, D) = 1, 20, 576, 64
# Nkv = 227
import triton


(B, H, N, D) = 3, 7, 1537, 111
Nkv = 1234

dtype = torch.bfloat16
ref_sdp_dtype = torch.bfloat16
causal = False

from rocwmma_fattn.FlashAttn import FlashAttentionFunction
fttn = FlashAttentionFunction

if __name__ == "__main__":
    q = torch.rand((B,  H, N,   D), dtype=dtype, device="cuda") 
    k = torch.rand((B,  H, Nkv, D), dtype=dtype, device="cuda") 
    v = torch.rand((B,  H, Nkv, D), dtype=dtype, device="cuda") 
    
    # q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d'), (q, k, v))
    
    # print(q.size(), k.size(), v.size())
    # print(q.stride(), k.stride(), v.stride())

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    q2 = q.clone().detach().to(ref_sdp_dtype).requires_grad_(True)
    k2 = k.clone().detach().to(ref_sdp_dtype).requires_grad_(True)
    v2 = v.clone().detach().to(ref_sdp_dtype).requires_grad_(True)

    
    o1 = fttn.apply(q, k, v, None, causal, None, False) # q k v mask causal scale BNHD
    print(o1.shape)
    o2 = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
    maxdiff = (o2 - o1).abs().max().item()
    print(o1.cpu()[-1, -1, :, :])
    print(o2.cpu()[-1, -1, :, :])
    print("fwd diff:", maxdiff)
    # exit()

    dO = 0.5 + torch.rand_like(q)
    dO[:,:,:,0] = -2
    
    o1.backward(dO)
    o2.backward(dO)

    dQ1 = q.grad.clone().detach()
    dK1 = k.grad.clone().detach()
    dV1 = v.grad.clone().detach()

    dQ2 = q2.grad.clone().detach()
    dK2 = k2.grad.clone().detach()
    dV2 = v2.grad.clone().detach()

    print("FTTN dQ", dQ1.cpu()[0, -1, :, :])
    print("PT dQ", dQ2.cpu()[0, -1, :, :])

    print("FTTN dK", dK1.cpu()[0, -1, :, :])
    print("PT dK", dK2.cpu()[0, -1, :, :])

    print("FTTN dV", dV1.cpu()[0, -1, :, :])
    print("PT dV", dV2.cpu()[0, -1, :, :])

    print("fwd diff:", maxdiff)
    print(f"dQ diff:{(dQ1 - dQ2).abs().max().item()}")
    print(f"dK diff:{(dK1 - dK2).abs().max().item()}")
    print(f"dV diff:{(dV1 - dV2).abs().max().item()}")
