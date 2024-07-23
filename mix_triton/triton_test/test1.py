import torch
import triton
import triton.language as tl


@triton.jit
def fwd_kernel(q, k, v, 
               N: tl.constexpr,
               Nk: tl.constexpr,
               D: tl.constexpr,
               Br: tl.constexpr, 
               Bc: tl.constexpr
               ):
    q_off = tl.program_id(0) * N * D + tl.program_id(1) * Br * D
    kv_off = tl.program_id(0) * N * D + tl.program_id(1) * Bc * D
    
    tl.make_block_ptr(
        base=q+q_off, 
        shape=(Br, D), 
        strides=(),
        )
    pass
