# flash-attention-v2-RDNA3-minimal
a simple Flash Attention v2 implementation with ROCM (RDNA3 GPU, roc wmma), mainly used for stable diffusion(ComfyUI) in Windows ZLUDA environments.

# Build and Test

### Linux with rocm:
just run ```python main.py```

### Windows with zluda

Need MSVC Compiler, AMD HIP SDK and rocWMMA Library.

Install rocwmma library: https://github.com/ROCm/rocWMMA

clone it and copy ```library/include/rocwmma``` to HIP SDK installation path of ```include``` folder

In cmd.exe, run ```vcvars64.bat``` to active MSVC Environment, then run ```zluda -- python main.py```

### Pre-build Extension 

Tested work with PyTorch 2.2.1 + cu118 windows zluda, gfx1100 GPU

comfyui: https://github.com/Repeerc/ComfyUI-flash-attention-rdna3-win-zluda

webui: https://github.com/Repeerc/sd-webui-flash-attention-zluda-win

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


# To do

- [x] backward pass
- [x] causal mask (need more optimization)
- [ ] unaligned 32x seqlen padding optimization
- [ ] attention bias
- [ ] matrix multiplication optimization
- [ ] fix poor performance in BF16
- [ ] ...


# Benchmark

OS: Windows 11

GPU: 7900xtx (gfx1100)

PyTorch 2.2.1 + CU118 ZLUDA, Python 3.10

### FP16, causal = False

Triton build from: https://github.com/triton-lang/triton

git hash: [47fc046ff29c9ea2ee90e987c39628a540603c8f](https://github.com/triton-lang/triton/tree/47fc046ff29c9ea2ee90e987c39628a540603c8f)

test use Triton windows pre-build version: https://github.com/Repeerc/triton-windows-amdgpu

Compare with Triton offcial version ```06-fused-attention.py``` (96 dim_head was padded to 128 in triton)

CK-based(Composable Kernel) flash attention version compiled from: https://github.com/ROCm/flash-attention/tree/howiejay/navi_support

windows env ck version: https://github.com/Repeerc/flash-attn-composable-kernel-gfx110x-windows-port

### seqlen with 32x aligened 

![3abbb498ec5bb4c665dac05602c4eb55](https://github.com/user-attachments/assets/38e46c22-9fc8-4364-b734-5cfcf1a17344)

![c018be8263e89a1daf67eab2fabb9dbe](https://github.com/user-attachments/assets/7d844241-b24b-4319-a34f-34236f7bf3cc)


### seqlen without 32x aligened 

![b5f625b6706e6dcf8006c2614badda5a](https://github.com/user-attachments/assets/1d9428fb-fbd9-4ed2-b98d-c1f09ef96e1e)

### fwd+bwd

![e1325f5163f73d96ff3a628a2b52d88b](https://github.com/user-attachments/assets/65881bbc-6fa5-487d-b09a-af7de5489854)


### FP16, causal = True

![fwd_scan_N](https://github.com/user-attachments/assets/121c3b13-f37c-49cc-969d-41be1d305a62)

![fwd_bwd_scan_N](https://github.com/Repeerc/flash-attention-v2-RDNA3-minimal/assets/7540581/529353f0-7478-484b-8ddb-d94052dff13a)

![fwd_scan_D](https://github.com/Repeerc/flash-attention-v2-RDNA3-minimal/assets/7540581/47aaeef8-3064-49a3-b737-64d4f36ef30b)



