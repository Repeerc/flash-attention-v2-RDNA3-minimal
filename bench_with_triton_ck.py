import math
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


os.add_dll_directory(os.path.join( os.environ['HIP_PATH'] , 'bin')) 
from ck_fttn import ck_fttn_pyb

import triton
from triton_fused_attention import _attention


triton_fttn = _attention.apply

test_round = 200
def count_time(func):
    def wrapper(*args, **kwargs):
        # torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        #warm up
        torch.cuda.synchronize()
        for _ in range(50):
            ret = func(*args, **kwargs)
        torch.cuda.synchronize()
        
        torch.cuda.reset_peak_memory_stats()
         
        t1 = time.time()
        for _ in range(test_round):
            ret = func(*args, **kwargs)
        torch.cuda.synchronize()
        t2 = time.time() - t1
        
        #assert torch.nan not in ret.cpu()
        max_memory = torch.cuda.max_memory_allocated() // 2**20
        flops_per_matmul = 2.0 * B * H * N * N * D
        total_flops = 2 * flops_per_matmul
        if causal:
            total_flops *= 0.5
        if 'bwd' in func.__name__:
            total_flops *= 2.5
        speed = total_flops / (t2 / test_round)
        print(
            f"{func.__name__}:  \texec_time:{t2:.4f}, total_tflops:{speed / 1e12:.2f}, max_memory:{max_memory}"
        )

        torch.cuda.empty_cache()
        return ret, speed, max_memory, t2
    return wrapper

#torch.Size([2, 10, 4096, 64]) torch.Size([2, 10, 77, 64]) torch.Size([2, 10, 77, 64])
#torch.Size([2, 10, 4096, 64]) torch.Size([2, 10, 4096, 64]) torch.Size([2, 10, 4096, 64])

(B, H, N, D) = (1, 24, 4096, 64)
causal = False
dtype = torch.float16

from rocwmma_fattn.FlashAttn import FlashAttentionFunction

wmma_fttn = FlashAttentionFunction.apply


def pad_to_multiple(tensor, multiple, dim=-1, val = 0):
    length = tensor.size(dim)
    remainder = length % multiple
    if remainder == 0:
        return tensor, 0
    padding_length = multiple - remainder
    padding_shape = list(tensor.shape)
    padding_shape[dim] = padding_length
    padding_tensor = torch.zeros(padding_shape, device=tensor.device, dtype=tensor.dtype) + val
    return torch.cat([tensor, padding_tensor], dim=dim), padding_length

@count_time
def sdp_pt(q, k, v=None):
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ):
        r0 = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return r0


@count_time
def sdp_bwd(q, k, v, O, dO):
    if q.grad is not None:
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_()
    O.backward(dO, retain_graph=True)
    dQ, dK, dV = (q.grad, k.grad, v.grad)
    return dQ, dK, dV

@count_time
def fttn_rocwmma_bwd(q, k, v, O, dO):
    if q.grad is not None:
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_()
    O.backward(dO, retain_graph=True)
    dQ, dK, dV = (q.grad, k.grad, v.grad)
    
    return dQ, dK, dV

@count_time
def fttn_rocwmma(q, k, v=None):
    O, L = wmma_fttn(q,k,v, None,causal)
    
    return O, L

@count_time
def fttn_ck(q, k, v=None):
    
    d_qkv = q.shape[-1] 
    
    sc = d_qkv ** -0.5
    
    q2, k2, v2 = map(lambda t: t.transpose(1, 2), (q, k, v))
    del q,k,v
    ret =  ck_fttn_pyb.fwd(q2,k2,v2, None, 0, sc, causal, False, None) # BNHD
    #O = (ret[0])[:, :, :, :d_qkv]
    O = ret[0]
    O = O.transpose(1, 2)
    
    L = ret[5]
    
    return O, L

@count_time
def ftt_triton_bwd(q, k, v, O, dO, L):
    
    #dQ, dK, dV = flash_attn_wmma.backward(q, k, v, O, dO, L,256,64, causal)

    if q.grad is not None:
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_()
    O.backward(dO, retain_graph=True)
    dQ, dK, dV = (q.grad, k.grad, v.grad)
    
    return dQ, dK, dV
    

@count_time
def ftt_triton(q, k, v=None):
    
    # def pad_to_multiple(tensor, multiple, dim=-1, val = 0):
    #     length = tensor.size(dim)
    #     remainder = length % multiple
    #     if remainder == 0:
    #         return tensor, 0
    #     padding_length = multiple - remainder
    #     padding_shape = list(tensor.shape)
    #     padding_shape[dim] = padding_length
    #     padding_tensor = torch.zeros(padding_shape, device=tensor.device, dtype=tensor.dtype) + val
    #     return torch.cat([tensor, padding_tensor], dim=dim), padding_length
    
    d_qkv = q.shape[-1]
    q, d_pad_len = pad_to_multiple(q, triton.next_power_of_2(d_qkv))
    k, d_pad_len = pad_to_multiple(k, triton.next_power_of_2(d_qkv))
    v, d_pad_len = pad_to_multiple(v, triton.next_power_of_2(d_qkv))
    
    # Bc_max = 256
    # Br_max = 64
    # d_final = d_qkv + d_pad_len
    # SRAM_SZ_FP16 = 32768
    # Bc_max = SRAM_SZ_FP16//Br_max - 2*d_final
    # while Bc_max <= 0:
    #     Br_max -= 32
    #     Bc_max = SRAM_SZ_FP16//Br_max - 2*d_final
        
    # n_kv = k.shape[2]
    # k, nkv_pad_len = pad_to_multiple(k, 16, -2, -1)
    # v, nkv_pad_len = pad_to_multiple(v, 16, -2)
    
    # Bc = Bc_max
    # Br = Br_max
    
    sc = d_qkv ** -0.5
    
    
    ret = triton_fttn(q, k, v, causal, sc)
    O = ret[:, :, :, :d_qkv]
    #L = None
    return O #, L


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
n_list = []
flops_ft_list = []
maxmem_ft_list = []
flops_sdp_list = []
maxmem_sdp_list = []
flops_triton_list = []
maxmem_triton_list = []
flops_ck_list = []
maxmem_ck_list = []
for i in range(1,20,1):
    N = 256 * i
    q_shape = (B, H, N, D)
    v_shape = (B, H, N, D)
    k_shape = (B, H, N, D)
    print(f'B:{B}, H:{H}, SeqLen:{N}, DimHead:{D}')
    q = torch.rand(q_shape, dtype=dtype, device="cuda")  # * 5
    k = torch.rand(k_shape, dtype=dtype, device="cuda")  # * 80
    v = torch.rand(v_shape, dtype=dtype, device="cuda")  # * 30
    
    r3, flops_ft, max_memory_ft, _ = fttn_rocwmma(q, k, v)
    r0, flops_sdp, max_memory_sdp, _ = sdp_pt(q, k, v)
    r1, flops_triton, max_memory_triton, _ = ftt_triton(q, k, v)
    r4, flops_ck, max_memory_ck, _ = fttn_ck(q,k,v)
     
    L_roc = r3[1]
    L_ck = r4[1] 
     
    r3 = r3[0].cpu()
    r0 = r0.cpu()
    r1 = r1.cpu()
    r4 = r4[0].cpu()

    maxdiff = (r0 - r3).abs().max().item()
    print("max diff sdp-rocwmma: ", maxdiff)
    maxdiff = (r0 - r1).abs().max().item()
    print("max diff sdp-triton: ", maxdiff)
    maxdiff = (r0 - r4).abs().max().item()
    print("max diff sdp-ck: ", maxdiff)
    
    # maxdiff = (L_roc - L_ck).abs().max().item()
    # print("max diff Lse: ", maxdiff)
    
    
    n_list.append(N)
    flops_ft_list.append(flops_ft / 1e12)
    flops_sdp_list.append(flops_sdp / 1e12)
    flops_triton_list.append(flops_triton / 1e12)
    flops_ck_list.append(flops_ck / 1e12)
    maxmem_ft_list.append(max_memory_ft)
    maxmem_sdp_list.append(max_memory_sdp)
    maxmem_triton_list.append(max_memory_triton)
    maxmem_ck_list.append(max_memory_ck)

fig = plt.figure(figsize=[7,9])
plt.subplot(211)
plt.plot(n_list, flops_ft_list, label="Flash attn 2 (rocwmma)")
plt.plot(n_list, flops_sdp_list, label="PyTorch SDPA")
plt.plot(n_list, flops_triton_list, label="Flash attn 2 (Triton)")
plt.plot(n_list, flops_ck_list, label="Flash attn 2 (ck)")
plt.xlabel("Seqlen")
plt.ylabel('TFlops')
plt.legend()
plt.xticks(n_list)
plt.grid(True)

plt.subplot(212)
plt.plot(n_list, maxmem_ft_list, label="Flash attn 2 (rocwmma)")
plt.plot(n_list, maxmem_sdp_list, label="PyTorch SDPA")
plt.plot(n_list, maxmem_triton_list, label="Flash attn 2 (Triton)")
plt.plot(n_list, maxmem_ck_list, label="Flash attn 2 (ck)")
plt.xlabel("Seqlen")
plt.ylabel('VRAM(MB)')
plt.legend()
plt.xticks(n_list)
plt.suptitle(f"Forward B:{B}, H:{H}, D:{D} (BHND Order)")
plt.grid(True)
fig.subplots_adjust(top=0.95,bottom=0.05,right=0.96)
fig.savefig('fwd_scan_N.png')


plt.show()
exit()

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
d_list = []
d_ck_list = []
flops_ft_list = []
maxmem_ft_list = []
flops_sdp_list = []
maxmem_sdp_list = []
flops_triton_list = []
maxmem_triton_list = []
flops_ck_list = []
maxmem_ck_list = []

N = 2048
for i in range(48,256+16,16):
    D = i
    q_shape = (B, H, N, D)
    v_shape = (B, H, N, D)
    k_shape = (B, H, N, D)
    print(f'B:{B}, H:{H}, SeqLen:{N}, DimHead:{D}')
    q = torch.rand(q_shape, dtype=dtype, device="cuda")  # * 5
    k = torch.rand(k_shape, dtype=dtype, device="cuda")  # * 80
    v = torch.rand(v_shape, dtype=dtype, device="cuda")  # * 30


    r3, flops_ft, max_memory_ft, _ = fttn_rocwmma(q, k, v)
    r0, flops_sdp, max_memory_sdp, _ = sdp_pt(q, k, v)
    # r1, flops_triton, max_memory_triton, _ = ftt_triton(q, k, v)
    if D <= 128:
        r4, flops_ck, max_memory_ck, _ = fttn_ck(q,k,v)
        L_ck = r4[1] 
     
    L_roc = r3[1]
     
    r3 = r3[0].cpu()
    r0 = r0.cpu()
    # r1 = r1.cpu()
    if D <= 128:
        r4 = r4[0].cpu()

    maxdiff = (r0 - r3).abs().max().item()
    print("max diff sdp-rocwmma: ", maxdiff)
    # maxdiff = (r0 - r1).abs().max().item()
    # print("max diff sdp-triton: ", maxdiff)
    
    if D <= 128:
        maxdiff = (r0 - r4).abs().max().item()
        print("max diff sdp-ck: ", maxdiff)
        
    # maxdiff = (L_roc - L_ck).abs().max().item()
    # print("max diff Lse: ", maxdiff)
    
    if D <= 128:
        d_ck_list.append(D)
    
    d_list.append(D)
    flops_ft_list.append(flops_ft / 1e12)
    flops_sdp_list.append(flops_sdp / 1e12)
    # flops_triton_list.append(flops_triton / 1e12)
    
    if D <= 128:
        flops_ck_list.append(flops_ck / 1e12)
    
    maxmem_ft_list.append(max_memory_ft)
    maxmem_sdp_list.append(max_memory_sdp)
    # maxmem_triton_list.append(max_memory_triton)
    
    if D <= 128:
        maxmem_ck_list.append(max_memory_ck)

    
fig = plt.figure(figsize=[7,9])
plt.subplot(211)
plt.plot(d_list, flops_ft_list, label="Flash attn 2 (rocwmma)")
plt.plot(d_list, flops_sdp_list, label="PyTorch SDPA")
plt.plot(d_ck_list, flops_ck_list, label="Flash attn 2 (ck)")
# plt.plot(d_list, flops_triton_list, label="Flash attn 2 (Triton)")
plt.xlabel("dim_head")
plt.ylabel('TFlops')
plt.legend()
plt.xticks(d_list)
plt.grid(True)

plt.subplot(212)
plt.plot(d_list, maxmem_ft_list, label="Flash attn 2 (rocwmma)")
plt.plot(d_list, maxmem_sdp_list, label="PyTorch SDPA")
plt.plot(d_ck_list, maxmem_ck_list, label="Flash attn 2 (ck)")
# plt.plot(d_list, maxmem_triton_list, label="Flash attn 2 (Triton)")
plt.xlabel("dim_head")
plt.ylabel('VRAM(MB)')
plt.legend()
plt.xticks(d_list)
plt.suptitle(f"Forward B:{B}, H:{H}, N:{N} (BHND Order)")
plt.grid(True)
fig.subplots_adjust(top=0.95,bottom=0.05,right=0.96)
fig.savefig('fwd_scan_D.png')


plt.show()
# print("max diff: ", (r1 - r0).abs().max().item())
print(r0.cpu()[0, 0, :, :])
print(r1.cpu()[0, 0, :, :])
print(r3.cpu()[0, 0, :, :])


# q = torch.rand((B, N, H, D), dtype=dtype, device="cuda")
# k = torch.rand((B, N, H, D), dtype=dtype, device="cuda")
# v = torch.rand((B, N, H, D), dtype=dtype, device="cuda")


# with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     flash_attn_wmma.forward(q, k, v, 64,512)
# print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=100))

# q, k, v = map(
#        lambda t: t.transpose(1, 2).contiguous(),
#        (q, k, v),
#     )
# with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     torch.nn.functional.scaled_dot_product_attention(q, k, v)
# print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=100))
