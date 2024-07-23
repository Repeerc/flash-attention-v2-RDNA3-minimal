import matplotlib.pyplot as plt
import torch
import torch.utils

# torch.backends.cuda.enable_math_sdp(True)
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
import time

import os
import sys

test_round = 100
def count_time(func):
    def wrapper(*args, **kwargs):
        # torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        #warm up
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
        speed = total_flops / (t2 / test_round)
        print(
            f"{func.__name__}:  \texec_time:{t2:.4f}, total_tflops:{speed / 1e12:.2f}, max_memory:{max_memory}"
        )

        torch.cuda.empty_cache()
        return ret, speed, max_memory, t2

    return wrapper


(B, H, N, D) = (1, 20, 2048, 64)
q_shape = (B, H, N, D)
v_shape = (B, H, N, D)
k_shape = (B, H, N, D)
causal = False
dtype = torch.float16


@count_time
def sdp_pt(q, k, v=None):
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=True
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


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
maxdiff = 0
n_list = []
flops_ft_list = []
maxmem_ft_list = []
flops_sdp_list = []
maxmem_sdp_list = []
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
    
    q2 = q.clone().detach().requires_grad_(True)
    k2 = k.clone().detach().requires_grad_(True)
    v2 = v.clone().detach().requires_grad_(True)
    flops_per_matmul = 2.0 * B * H * N * N * D
    if causal:
        flops_per_matmul *= 0.5
    fwd_ops = 2.0 * flops_per_matmul
    bwd_ops = 5.0 * flops_per_matmul

    r0, flops_sdp_fwd, max_memory_sdp_fwd, sdp_fwd_time = sdp_pt(q, k, v)
    # r3 = r3.cpu().to(torch.float32).transpose(1, 2)
    # r0 = r0.cpu().to(torch.float32).transpose(1, 2)
    # print(L.cpu())
    dO = torch.ones_like(q) 
    
    dQKV_0, flops_sdp_bwd, max_memory_sdp_bwd, sdp_bwd_time = sdp_bwd(q,k,v,r0, dO)
    
    max_memory_sdp = max(max_memory_sdp_fwd, max_memory_sdp_bwd)
    
    flops_sdp = (fwd_ops+bwd_ops)/((sdp_fwd_time + sdp_bwd_time) / test_round)
    
    n_list.append(N)
    flops_sdp_list.append(flops_sdp / 1e12)
    maxmem_sdp_list.append(max_memory_sdp)
    
    #print(q.grad.shape)
    del q,k,v,q2,k2,v2,r0

fig = plt.figure(figsize=[7,9])
plt.subplot(211)
plt.plot(n_list, flops_sdp_list, label="PyTorch SDPA")
plt.xlabel("Seqlen")
plt.ylabel('TFlops')
plt.legend()
plt.xticks(n_list)
plt.grid(True)

plt.subplot(212)
plt.plot(n_list, maxmem_sdp_list, label="PyTorch SDPA")
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

for i in range(1,15,1):
    N = 512 * i
    q_shape = (B, H, N, D)
    v_shape = (B, H, N, D)
    k_shape = (B, H, N, D)
    print(f'B:{B}, H:{H}, SeqLen:{N}, DimHead:{D}')
    q = torch.rand(q_shape, dtype=dtype, device="cuda")  # * 5
    k = torch.rand(k_shape, dtype=dtype, device="cuda")  # * 80
    v = torch.rand(v_shape, dtype=dtype, device="cuda")  # * 30

    r0, flops_sdp, max_memory_sdp, _ = sdp_pt(q, k, v)

    
    n_list.append(N)
    flops_sdp_list.append(flops_sdp / 1e12)
    maxmem_sdp_list.append(max_memory_sdp)

fig = plt.figure(figsize=[7,9])
plt.subplot(211)
plt.plot(n_list, flops_sdp_list, label="PyTorch SDPA")
plt.xlabel("Seqlen")
plt.ylabel('TFlops')
plt.legend()
plt.xticks(n_list)
plt.grid(True)

plt.subplot(212)
plt.plot(n_list, maxmem_sdp_list, label="PyTorch SDPA")
plt.xlabel("Seqlen")
plt.ylabel('VRAM(MB)')
plt.legend()
plt.xticks(n_list)
plt.suptitle(f"Forward B:{B}, H:{H}, D:{D}")
plt.grid(True)
fig.subplots_adjust(top=0.95,bottom=0.05,right=0.96)
fig.savefig('fwd_scan_N.png')



torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
d_list = []
flops_ft_list = []
maxmem_ft_list = []
flops_sdp_list = []
maxmem_sdp_list = []

N = 4096
for i in range(1,16):
    D = 16*i
    q_shape = (B, H, N, D)
    v_shape = (B, H, N, D)
    k_shape = (B, H, N, D)
    print(f'B:{B}, H:{H}, SeqLen:{N}, DimHead:{D}')
    q = torch.rand(q_shape, dtype=dtype, device="cuda")  # * 5
    k = torch.rand(k_shape, dtype=dtype, device="cuda")  # * 80
    v = torch.rand(v_shape, dtype=dtype, device="cuda")  # * 30

    r0, flops_sdp, max_memory_sdp, _ = sdp_pt(q, k, v)

    
    d_list.append(D)
    flops_sdp_list.append(flops_sdp / 1e12)
    maxmem_sdp_list.append(max_memory_sdp)
    
fig = plt.figure(figsize=[7,9])
plt.subplot(211)
plt.plot(d_list, flops_sdp_list, label="PyTorch SDPA")
plt.xlabel("dim_head")
plt.ylabel('TFlops')
plt.legend()
plt.xticks(d_list)
plt.grid(True)

plt.subplot(212)
plt.plot(d_list, maxmem_sdp_list, label="PyTorch SDPA")
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
