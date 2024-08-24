from einops import rearrange
import torch

B, H, N, D = 2, 8, 1024, 64
Br = 64

m1 = torch.rand((B, N, H, D), dtype=torch.float16, device="cuda")
m2 = rearrange(m1, "b n h d -> b h n d")
m3 = rearrange(m1, "b n h d -> b h n d").contiguous()

print(m1.stride(), m2.stride(), m3.stride())

t1 = m1.view(-1)
t3 = m3.view(-1)

fetch_B = 1
fetch_H = 2

fetch_Tr_j = 4

fetch_line_N = Br * fetch_Tr_j

off_t1 = m1.stride(0) * fetch_B + fetch_H * m1.stride(2) # D
ld_t1 = m1.stride(1) # H * D

off_t3 = m3.stride(0) * fetch_B + fetch_H * m3.stride(1)
ld_t3 = m3.stride(2) # D

print(t1[off_t1 + ld_t1 * fetch_line_N : off_t1 + ld_t1 * fetch_line_N + D])
print(t3[off_t3 + ld_t3 * fetch_line_N : off_t3 + ld_t3 * fetch_line_N + D])
