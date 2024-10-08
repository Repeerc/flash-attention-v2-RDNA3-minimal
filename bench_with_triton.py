import matplotlib.pyplot as plt
import torch
import torch.utils

torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
import time
import os
import sys



import triton
from triton_fused_attention import _attention

triton_fttn = _attention.apply

test_round = 100
def count_time(func):
    def wrapper(*args, **kwargs):
        # torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # warm up
        torch.cuda.synchronize()
        for _ in range(10):
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
        speed = total_flops / (t2 / test_round) if t2 != 0 else 0
        print(
            f"{func.__name__}:  \texec_time:{t2:.4f}, total_tflops:{speed / 1e12:.2f}, max_memory:{max_memory}"
        )

        torch.cuda.empty_cache()
        return ret, speed, max_memory, t2

    return wrapper


(B, H, N, D) = (1, 24, 2048, 64)
q_shape = (B, H, N, D)
v_shape = (B, H, N, D)
k_shape = (B, H, N, D)
causal = False
dtype = torch.float16


from rocwmma_fattn.FlashAttn import FlashAttentionFunction

wmma_fttn = FlashAttentionFunction.apply

@count_time
def sdp_pt(q, k, v=None):
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ):
        r0 = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return r0


@count_time
def sdp_fwd_bwd(q, k, v, dO):
    
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ):
        O = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    
    if q.grad is not None:
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_() 
    O.backward(dO, retain_graph=True)
    dQ, dK, dV = (q.grad, k.grad, v.grad)
    return dQ, dK, dV

@count_time
def fttn_rocwmma_fwd_bwd(q, k, v, dO):
    
    O = wmma_fttn(q,k,v, None,causal)
    if q.grad is not None:
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_() 
    O.backward(dO, retain_graph=True)
    dQ, dK, dV = (q.grad, k.grad, v.grad)
    
    return dQ, dK, dV

@count_time
def fttn_rocwmma(q, k, v=None):
    O = wmma_fttn(q,k,v, None,causal)
    
    return O


@count_time
def ftt_triton_fwd_bwd(q, k, v, dO):
    
    O = ftt_triton_fwd(q,k,v)
    
    if q.grad is not None:
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_() 
    O.backward(dO, retain_graph=True)
    dQ, dK, dV = (q.grad, k.grad, v.grad)
    
    return dQ, dK, dV
    
def ftt_triton_fwd(q,k,v):
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
    
    d_qkv = q.shape[-1]
    q, d_pad_len = pad_to_multiple(q, triton.next_power_of_2(d_qkv))
    k, d_pad_len = pad_to_multiple(k, triton.next_power_of_2(d_qkv))
    v, d_pad_len = pad_to_multiple(v, triton.next_power_of_2(d_qkv))
    d_final = d_qkv + d_pad_len
    
    sc = d_final ** -0.5
    
    ret = triton_fttn(q, k, v, causal, sc)
    O = ret[:, :, :, :d_qkv] 
    return O  

@count_time
def ftt_triton(q, k, v):
    return ftt_triton_fwd(q,k,v)



torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
maxdiff = 0
n_list = []
flops_ft_list = []
maxmem_ft_list = []
flops_sdp_list = []
maxmem_sdp_list = []
flops_triton_list = []
maxmem_triton_list = []
for i in range(1,11,1):
    N = 512 * i
    q_shape = (B, H, N, D)
    v_shape = (B, H, N, D)
    k_shape = (B, H, N, D)
    print(f'B:{B}, H:{H}, SeqLen:{N}, DimHead:{D}')
    q = torch.rand(q_shape, dtype=dtype, device="cuda") #  * 5
    k = torch.rand(k_shape, dtype=dtype, device="cuda") #  * 80
    v = torch.rand(v_shape, dtype=dtype, device="cuda") #  * 30
    
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    
    q1 = q.clone().detach().requires_grad_(True)
    k1 = k.clone().detach().requires_grad_(True)
    v1 = v.clone().detach().requires_grad_(True)
    
    q2 = q.clone().detach().requires_grad_(True)
    k2 = k.clone().detach().requires_grad_(True)
    v2 = v.clone().detach().requires_grad_(True)
    
    
    flops_per_matmul = 2.0 * B * H * N * N * D
    if causal:
        flops_per_matmul *= 0.5
    fwd_ops = 2.0 * flops_per_matmul
    bwd_ops = 5.0 * flops_per_matmul

    r0, flops_sdp_fwd, max_memory_sdp_fwd, sdp_fwd_time = sdp_pt(q, k, v)
    r1, flops_trition_fwd, max_memory_triton_fwd, triton_fwd_time = ftt_triton(q1, k1, v1)
    r3, flops_ft_fwd, max_memory_ft_fwd, fttn_fwd_time = fttn_rocwmma(q2, k2, v2)

    maxdiff = (r0 - r3).abs().max().item()
    print("fwd max sdp-rocwmma diff: ", maxdiff)
    maxdiff = (r0 - r1).abs().max().item()
    print("fwd max sdp-triton diff: ", maxdiff)
    
    
    dO = torch.ones_like(q) 
    dO1 = dO.clone().detach().requires_grad_(False)
    dO2 = dO.clone().detach()
    
    dQKV_0, _, max_memory_sdp_bwd, sdp_fwd_bwd_time = sdp_fwd_bwd(q,k,v,dO)
    dQKV_2, _, max_memory_triton_bwd, trition_fwd_bwd_time = ftt_triton_fwd_bwd(q1,k1,v1, dO1)
    dQKV_3, _, max_memory_ft_bwd, fttn_fwd_bwd_time = fttn_rocwmma_fwd_bwd(q2,k2,v2,dO2)
    
    max_memory_ft = max(max_memory_ft_fwd, max_memory_ft_bwd)
    max_memory_sdp = max(max_memory_sdp_fwd, max_memory_sdp_bwd)
    max_memory_triton = max(max_memory_triton_fwd, max_memory_triton_bwd)
    
    
    flops_sdp = (fwd_ops+bwd_ops)/((sdp_fwd_bwd_time) / test_round)
    flops_triton = (fwd_ops+bwd_ops)/((trition_fwd_bwd_time) / test_round)
    flops_ft = (fwd_ops+bwd_ops)/((fttn_fwd_bwd_time) / test_round)
    
    n_list.append(N)
    flops_ft_list.append(flops_ft / 1e12)
    flops_sdp_list.append(flops_sdp / 1e12)
    flops_triton_list.append(flops_triton / 1e12)
    
    maxmem_ft_list.append(max_memory_ft)
    maxmem_sdp_list.append(max_memory_sdp)
    maxmem_triton_list.append(max_memory_triton)
    
    # maxdiff = (dQKV_0[0] - dQKV_3[0]).abs().max().item()
    # maxdiff = max(maxdiff, (dQKV_0[1] - dQKV_3[1]).abs().max().item())
    # maxdiff = max(maxdiff, (dQKV_0[2] - dQKV_3[2]).abs().max().item())
    print("bwd DQ diff sdp-rocwmma: ", (dQKV_0[0] - dQKV_3[0]).abs().max().item())
    print("bwd DK diff sdp-rocwmma: ", (dQKV_0[1] - dQKV_3[1]).abs().max().item())
    print("bwd DV diff sdp-rocwmma: ", (dQKV_0[2] - dQKV_3[2]).abs().max().item())
    
    
    # maxdiff = (dQKV_0[0] - dQKV_2[0]).abs().max().item()
    # maxdiff = max(maxdiff, (dQKV_0[1] - dQKV_2[1]).abs().max().item())
    # maxdiff = max(maxdiff, (dQKV_0[2] - dQKV_2[2]).abs().max().item())
    print("bwd DQ sdp-trition: ",  (dQKV_0[0] - dQKV_2[0]).abs().max().item())
    print("bwd DK sdp-trition: ",  (dQKV_0[1] - dQKV_2[1]).abs().max().item())
    print("bwd DV sdp-trition: ",  (dQKV_0[2] - dQKV_2[2]).abs().max().item())
    
    #print(q.grad.shape)
    del q,k,v,q1,k1,v1,q2,k2,v2,r0,r3,r1

fig = plt.figure(figsize=[7,9])
plt.subplot(211)
plt.plot(n_list, flops_ft_list, label="Flash attn 2 (rocwmma)")
plt.plot(n_list, flops_sdp_list, label="PyTorch SDPA")
plt.plot(n_list, flops_triton_list, label="Flash attn 2 (Triton)")
plt.xlabel("Seqlen")
plt.ylabel('TFlops')
plt.legend()
plt.xticks(n_list)
plt.grid(True)

plt.subplot(212)
plt.plot(n_list, maxmem_ft_list, label="Flash attn 2 (rocwmma)")
plt.plot(n_list, maxmem_sdp_list, label="PyTorch SDPA")
plt.plot(n_list, maxmem_triton_list, label="Flash attn 2 (Triton)")
plt.xlabel("Seqlen")
plt.ylabel('VRAM(MB)')
plt.legend()
plt.xticks(n_list)
plt.suptitle(f"Fwd+Bwd B:{B}, H:{H}, D:{D}")
plt.grid(True)
fig.subplots_adjust(top=0.95,bottom=0.05,right=0.96)
fig.savefig('fwd_bwd_scan_N.png')

# plt.show()
# exit()

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
n_list = []
flops_ft_list = []
maxmem_ft_list = []
flops_sdp_list = []
maxmem_sdp_list = []
flops_triton_list = []
maxmem_triton_list = []
for i in range(1,15,1):
    N = 512 * i
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
    
    r3 = r3.cpu()
    r0 = r0.cpu()
    r1 = r1.cpu()

    maxdiff = (r0 - r3).abs().max().item()
    print("max diff sdp-rocwmma: ", maxdiff)
    maxdiff = (r0 - r1).abs().max().item()
    print("max diff sdp-triton: ", maxdiff)
    
    n_list.append(N)
    flops_ft_list.append(flops_ft / 1e12)
    flops_sdp_list.append(flops_sdp / 1e12)
    flops_triton_list.append(flops_triton / 1e12)
    maxmem_ft_list.append(max_memory_ft)
    maxmem_sdp_list.append(max_memory_sdp)
    maxmem_triton_list.append(max_memory_triton)

fig = plt.figure(figsize=[7,9])
plt.subplot(211)
plt.plot(n_list, flops_ft_list, label="Flash attn 2 (rocwmma)")
plt.plot(n_list, flops_sdp_list, label="PyTorch SDPA")
plt.plot(n_list, flops_triton_list, label="Flash attn 2 (Triton)")
plt.xlabel("Seqlen")
plt.ylabel('TFlops')
plt.legend()
plt.xticks(n_list)
plt.grid(True)

plt.subplot(212)
plt.plot(n_list, maxmem_ft_list, label="Flash attn 2 (rocwmma)")
plt.plot(n_list, maxmem_sdp_list, label="PyTorch SDPA")
plt.plot(n_list, maxmem_triton_list, label="Flash attn 2 (Triton)")
plt.xlabel("Seqlen")
plt.ylabel('VRAM(MB)')
plt.legend()
plt.xticks(n_list)
plt.suptitle(f"Forward B:{B}, H:{H}, D:{D}")
plt.grid(True)
fig.subplots_adjust(top=0.95,bottom=0.05,right=0.96)
fig.savefig('fwd_scan_N.png')

# plt.show()
# exit()


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
d_list = []
flops_ft_list = []
maxmem_ft_list = []
flops_sdp_list = []
maxmem_sdp_list = []
flops_triton_list = []
maxmem_triton_list = []

N = 4096
for i in [64, 96 ,128]:
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
    r1, flops_triton, max_memory_triton, _ = ftt_triton(q, k, v)
    
    r3 = r3.cpu() 
    r0 = r0.cpu() 
    r1 = r1.cpu()

    maxdiff = (r0 - r3).abs().max().item()
    print("max diff: ", maxdiff)
    maxdiff = (r0 - r1).abs().max().item()
    print("max diff sdp-triton: ", maxdiff)
    
    
    d_list.append(D)
    flops_ft_list.append(flops_ft / 1e12)
    flops_sdp_list.append(flops_sdp / 1e12)
    flops_triton_list.append(flops_triton / 1e12)
    maxmem_ft_list.append(max_memory_ft)
    maxmem_sdp_list.append(max_memory_sdp)
    maxmem_triton_list.append(max_memory_triton)

    
fig = plt.figure(figsize=[7,9])
plt.subplot(211)
plt.plot(d_list, flops_ft_list, label="Flash attn 2 (rocwmma)")
plt.plot(d_list, flops_sdp_list, label="PyTorch SDPA")
plt.plot(d_list, flops_triton_list, label="Flash attn 2 (Triton)")
plt.xlabel("dim_head")
plt.ylabel('TFlops')
plt.legend()
plt.xticks(d_list)
plt.grid(True)

plt.subplot(212)
plt.plot(d_list, maxmem_ft_list, label="Flash attn 2 (rocwmma)")
plt.plot(d_list, maxmem_sdp_list, label="PyTorch SDPA")
plt.plot(d_list, maxmem_triton_list, label="Flash attn 2 (Triton)")
plt.xlabel("dim_head")
plt.ylabel('VRAM(MB)')
plt.legend()
plt.xticks(d_list)
plt.suptitle(f"Forward B:{B}, H:{H}, N:{N}")
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
