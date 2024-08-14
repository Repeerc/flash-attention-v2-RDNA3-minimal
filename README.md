# flash-attention-v2-RDNA3-minimal
a simple Flash Attention v2 implementation with ROCM (RDNA3 GPU, roc wmma), mainly used for stable diffusion(ComfyUI) in Windows ZLUDA environments.

# Build and Test

### Linux with rocm:
just run ```python bench_with_sdpa.py```

### Windows with zluda

Need MSVC Compiler, AMD HIP SDK and rocWMMA Library.

Install rocwmma library: https://github.com/ROCm/rocWMMA

clone it and copy ```library/include/rocwmma``` to HIP SDK installation path of ```include``` folder

In cmd.exe, run ```vcvars64.bat``` to active MSVC Environment, then run ```zluda -- python bench_with_sdpa.py```

### Pre-build Extension 

Tested work with PyTorch 2.2.1 + cu118 windows zluda, gfx1100 GPU

comfyui: https://github.com/Repeerc/ComfyUI-flash-attention-rdna3-win-zluda

webui: https://github.com/Repeerc/sd-webui-flash-attention-zluda-win

# To do

- [x] backward pass
- [x] causal mask (need more optimization)
- [ ] unaligned 32x seqlen padding optimization
- [ ] Load tile into LDS (for BHND format (rearrange in kernel)) and fix bank conflict
- [ ] attention bias
- [ ] matrix multiplication optimization
- [x] fix poor performance in BF16
- [ ] ...

# Benchmark

OS: Windows 11

GPU: 7900xtx (gfx1100)

PyTorch 2.2.1 + CU118 ZLUDA, Python 3.10

### FP16, causal = False

Triton build from: https://github.com/triton-lang/triton

git hash: [47fc046ff29c9ea2ee90e987c39628a540603c8f](https://github.com/triton-lang/triton/tree/47fc046ff29c9ea2ee90e987c39628a540603c8f)

test use Triton windows pre-build version: https://github.com/Repeerc/triton-windows-amdgpu

Triton offcial version use ```06-fused-attention.py```

CK-based(Composable Kernel) flash attention version compiled from: https://github.com/ROCm/flash-attention/tree/howiejay/navi_support

CK-based flash attention windows porting: https://github.com/Repeerc/flash-attn-composable-kernel-gfx110x-windows-port

### seqlen with 32x aligened 

![f487a9936a4cc0f0f76fedef7eae7fc9](https://github.com/user-attachments/assets/7381b407-eac0-4295-b4d2-baf249254734)

![412d57e698d5325f87d4a0ca1da589f7](https://github.com/user-attachments/assets/b8d13e2a-d9e5-48c2-aec5-24cdecef54b3)

#### [B N H D] format rearrange and contiguous to [B H N D]

![5d3b4ad403dda0d84d430d69435b7c6f](https://github.com/user-attachments/assets/5e9f217b-342e-4195-9791-cc799bd6c3b1)

![ce9f235ada83a6147db9f5fc36040e19](https://github.com/user-attachments/assets/ca123479-4c18-4086-b18f-618f8bd5de13)

### seqlen without 32x aligened 

![d470b660f4018cdb3325a5b1f7489537](https://github.com/user-attachments/assets/f9f6603f-cd9e-4c2d-b6af-152b9de27c44)

#### [B N H D] format rearrange and contiguous to [B H N D]

![87475a48c0bb91aa44cb39564dbc2baf](https://github.com/user-attachments/assets/7652bd9d-aa43-4685-8393-325f92a12ce7)

### fwd+bwd

![84ef4f7d7ec6a1158a0a5c31759aafec](https://github.com/user-attachments/assets/01ad13c9-a383-48ef-abc2-5154751db2c3)

### FP16, causal = True

![fwd_scan_N](https://github.com/user-attachments/assets/121c3b13-f37c-49cc-969d-41be1d305a62)

![fwd_bwd_scan_N](https://github.com/Repeerc/flash-attention-v2-RDNA3-minimal/assets/7540581/529353f0-7478-484b-8ddb-d94052dff13a)

![fwd_scan_D](https://github.com/Repeerc/flash-attention-v2-RDNA3-minimal/assets/7540581/47aaeef8-3064-49a3-b737-64d4f36ef30b)

# Performance in Stable Diffusion (ComfyUI)

OS: Windows 11

GPU: 7900xtx (gfx1100)

PyTorch 2.2.1 + CU118 ZLUDA, Python 3.10

Sampler: Euler

| SD 1.5 | PyTorch SDPA |  Flash Attn minimal |  |
|--|--|--|--|
|512x512x1| 17.32 it/s | 19.20 it/s | +10% |
| VRAM | 3.2 GB | 2.3 GB | |
|--|--|--|--|
|512x512x4| 4.96 it/s | 5.47 it/s | +10% |
| VRAM | 5.4 GB | 2.5 GB | |
|--|--|--|--|
|1024x1024x1| 2.52it/s | 3.53it/s | +40%  | 
| VRAM | 10.7 GB | 2.9 GB | |


| SDXL | PyTorch SDPA |  Flash Attn minimal |  |
|--|--|--|--|
|1536x1024x1| 2.03 it/s | 2.35 it/s | +16% |
| VRAM | 7.4 GB | 6.8 GB | |
|--|--|--|--|
|1024x1024x1| 3.30 it/s | 3.60 it/s | +9% |
| VRAM | 6.5 GB | 6.4 GB | |

### SDXL U-Net Lora training

```
unet_lr = 0.0001
lr_scheduler = "constant"
lr_warmup_steps = 0
optimizer_type = "AdamW"
network_dim = 32
network_alpha = 32
seed = 1337
mixed_precision = "fp16"
full_fp16 = false
full_bf16 = false
fp8_base = true
no_half_vae = false
```

| SDXL | PyTorch SDPA |  Flash Attn minimal |  |
|--|--|--|--|
|1024x1024x1| 1.27 it/s | 1.76 it/s | +39 % |
| VRAM | 21.5 GB | 16.8 GB | |



