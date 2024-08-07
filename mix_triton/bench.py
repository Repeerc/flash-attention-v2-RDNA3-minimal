import matplotlib.pyplot as plt
import torch
import torch.utils

torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
import time

import os
import sys

if sys.platform.startswith("win32"):
    import zluda_hijack_torch_hip_ext

    torch.utils.cpp_extension.IS_HIP_EXTENSION = True
    torch.version.hip = "5.7.0"
    torch.version.cuda = None
else:
    import torch.utils.cpp_extension
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100" #;gfx1101;gfx1102;gfx1103"
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

from triton_fused_attention import _attn_bwd_preprocess, _attn_bwd

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
def ftt_bwd(q, k, v, o, do, L):
    
     #dQ, dK, dV = flash_attn_wmma.backward(q, k, v, O, dO, L,256,64, causal)
    N = q.shape[-2]
    Nkv = k.shape[-2]
    k, _ = pad_to_multiple(k, q.shape[-2], -2, -1)
    v, _ = pad_to_multiple(v, q.shape[-2], -2, -1)
    do, _ = pad_to_multiple(do, q.shape[-2], 2)
    q = q.contiguous()
    o = o.contiguous()
    
    scale = q.shape[-1] ** -0.5
#     assert do.is_contiguous()
#     print(q.shape,k.shape,v.shape,o.shape,do.shape)
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
    
    return dq[:,:,:N,:], dk[:,:,:Nkv,:], dv[:,:,:Nkv,:]
    

@count_time
def ftt_rocm(q, k, v=None):
    
    
    d_qkv = q.shape[-1]
    q, d_pad_len = pad_to_multiple(q, 16)
    k, d_pad_len = pad_to_multiple(k, 16)
    v, d_pad_len = pad_to_multiple(v, 16)
    
    Bc_max = 256
    Br_max = 64
    d_final = d_qkv + d_pad_len
    SRAM_SZ_FP16 = 32768
    Bc_max = SRAM_SZ_FP16//Br_max - 2*d_final
    while Bc_max <= 0:
        Br_max -= 32
        Bc_max = SRAM_SZ_FP16//Br_max - 2*d_final
        
    n_kv = k.shape[2]
    k, nkv_pad_len = pad_to_multiple(k, 16, -2, -1)
    v, nkv_pad_len = pad_to_multiple(v, 16, -2)
    
    Bc = Bc_max
    Br = Br_max
    
    ret = flash_attn_wmma.forward(q, k, v, Br, Bc, causal)
    
    O = (ret[0])[:, :, :, :d_qkv]
    L = ret[1]
    
    return O, L


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
    
    q2 = q.clone().detach().requires_grad_(False)
    k2 = k.clone().detach().requires_grad_(False)
    v2 = v.clone().detach().requires_grad_(False)
    flops_per_matmul = 2.0 * B * H * N * N * D
    if causal:
        flops_per_matmul *= 0.5
    fwd_ops = 2.0 * flops_per_matmul
    bwd_ops = 5.0 * flops_per_matmul

    r0, flops_sdp_fwd, max_memory_sdp_fwd, sdp_fwd_time = sdp_pt(q, k, v)
    r3, flops_ft_fwd, max_memory_ft_fwd, fttn_fwd_time = ftt_rocm(q2, k2, v2)
    L = r3[1]
    r3 = r3[0]
    # r3 = r3.cpu().to(torch.float32).transpose(1, 2)
    # r0 = r0.cpu().to(torch.float32).transpose(1, 2)
    # print(L.cpu())
    maxdiff = (r0 - r3).abs().max().item()
    print("fwd max diff: ", maxdiff)
    dO = torch.ones_like(q) 
    dO2 = dO.clone().detach().requires_grad_(False) 
    
    dQKV_0, flops_sdp_bwd, max_memory_sdp_bwd, sdp_bwd_time = sdp_bwd(q,k,v,r0, dO)
    dQKV_3, flops_ft_bwd, max_memory_ft_bwd, fttn_bwd_time = ftt_bwd(q2,k2,v2,r3, dO2, L)
    
    max_memory_ft = max(max_memory_ft_fwd, max_memory_ft_bwd)
    max_memory_sdp = max(max_memory_sdp_fwd, max_memory_sdp_bwd)
    
    flops_ft = (fwd_ops+bwd_ops)/((fttn_fwd_time + fttn_bwd_time) / test_round)
    flops_sdp = (fwd_ops+bwd_ops)/((sdp_fwd_time + sdp_bwd_time) / test_round)
    
    n_list.append(N)
    flops_ft_list.append(flops_ft / 1e12)
    flops_sdp_list.append(flops_sdp / 1e12)
    maxmem_ft_list.append(max_memory_ft)
    maxmem_sdp_list.append(max_memory_sdp)
    
    maxdiff = (dQKV_0[0] - dQKV_3[0]).abs().max().item()
    maxdiff = max(maxdiff, (dQKV_0[1] - dQKV_3[1]).abs().max().item())
    maxdiff = max(maxdiff, (dQKV_0[2] - dQKV_3[2]).abs().max().item())
    print("bwd max diff: ", maxdiff)
    #print(q.grad.shape)
    del q,k,v,q2,k2,v2,r0,r3

fig = plt.figure(figsize=[7,9])
plt.subplot(211)
plt.plot(n_list, flops_ft_list, label="Flash Attention v2")
plt.plot(n_list, flops_sdp_list, label="PyTorch SDPA")
plt.xlabel("Seqlen")
plt.ylabel('TFlops')
plt.legend()
plt.xticks(n_list)
plt.grid(True)

plt.subplot(212)
plt.plot(n_list, maxmem_ft_list, label="Flash Attention v2")
plt.plot(n_list, maxmem_sdp_list, label="PyTorch SDPA")
plt.xlabel("Seqlen")
plt.ylabel('VRAM(MB)')
plt.legend()
plt.xticks(n_list)
plt.suptitle(f"Fwd+Bwd B:{B}, H:{H}, D:{D}")
plt.grid(True)
fig.subplots_adjust(top=0.95,bottom=0.05,right=0.96)
fig.savefig('fwd_bwd_scan_N.png')

#plt.show()
#exit()

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

    r3, flops_ft, max_memory_ft, _ = ftt_rocm(q, k, v)
    r0, flops_sdp, max_memory_sdp, _ = sdp_pt(q, k, v)
    L = r3[1]
    r3 = r3[0]
    r3 = r3.cpu().to(torch.float32).transpose(1, 2)
    r0 = r0.cpu().to(torch.float32).transpose(1, 2)

    maxdiff = (r0 - r3).abs().max().item()
    print("max diff: ", maxdiff)
    
    n_list.append(N)
    flops_ft_list.append(flops_ft / 1e12)
    flops_sdp_list.append(flops_sdp / 1e12)
    maxmem_ft_list.append(max_memory_ft)
    maxmem_sdp_list.append(max_memory_sdp)

fig = plt.figure(figsize=[7,9])
plt.subplot(211)
plt.plot(n_list, flops_ft_list, label="Flash Attention v2")
plt.plot(n_list, flops_sdp_list, label="PyTorch SDPA")
plt.xlabel("Seqlen")
plt.ylabel('TFlops')
plt.legend()
plt.xticks(n_list)
plt.grid(True)

plt.subplot(212)
plt.plot(n_list, maxmem_ft_list, label="Flash Attention v2")
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

    r3, flops_ft, max_memory_ft, _ = ftt_rocm(q, k, v)
    r0, flops_sdp, max_memory_sdp, _ = sdp_pt(q, k, v)
    L = r3[1]
    r3 = r3[0]
    r3 = r3.cpu().to(torch.float32).transpose(1, 2)
    r0 = r0.cpu().to(torch.float32).transpose(1, 2)

    maxdiff = (r0 - r3).abs().max().item()
    print("max diff: ", maxdiff)
    
    d_list.append(D)
    flops_ft_list.append(flops_ft / 1e12)
    flops_sdp_list.append(flops_sdp / 1e12)
    maxmem_ft_list.append(max_memory_ft)
    maxmem_sdp_list.append(max_memory_sdp)
    
fig = plt.figure(figsize=[7,9])
plt.subplot(211)
plt.plot(d_list, flops_ft_list, label="Flash Attention v2")
plt.plot(d_list, flops_sdp_list, label="PyTorch SDPA")
plt.xlabel("dim_head")
plt.ylabel('TFlops')
plt.legend()
plt.xticks(d_list)
plt.grid(True)

plt.subplot(212)
plt.plot(d_list, maxmem_ft_list, label="Flash Attention v2")
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
print(r0.cpu()[0, 0, :, :])
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
