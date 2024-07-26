#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp16.h>

#include <rocwmma/rocwmma.hpp>
#include <torch/extension.h>
 
using rocwmma::accumulator;
using rocwmma::col_major;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::row_major;

using rocwmma::bfloat16_t;
using rocwmma::float16_t;
using rocwmma::float32_t;


const int ROCWMMA_M = 16;
const int ROCWMMA_N = 16;
const int ROCWMMA_K = 16;

const int WAVE_SIZE = 32;


#define ComputeType float16_t
#define AT_PTR_TYPE at::Half
#define TORCH_DTYPE torch::kFloat16

typedef _Float16 fp16_frag __attribute__((ext_vector_type(16)));
typedef float fp32_frag __attribute__((ext_vector_type(8)));

__global__ void gemm_kernel(
    float *wb
)
{
    

    fp16_frag fragA;
    fp16_frag fragB;
    fp32_frag fragACC = {};
    fragA[0] = blockIdx.x;
    fragB[0] = (threadIdx.x);

//#pragma unroll 200
    for (int i = 0; i < 100000; i++)
    {
        fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA, fragB, fragACC);
        fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA, fragB, fragACC);
        fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA, fragB, fragACC);
        fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA, fragB, fragACC);
        fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA, fragB, fragACC);
    }
    
    wb[0] = fragACC[0];

    // asm volatile("s_sleep 0");

}


torch::Tensor forward(int n_block, int n_waves
)
{

    cudaError_t err = cudaGetLastError();
    auto optD = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto D = torch::zeros({16}, optD);

    auto gridDim = dim3(n_block, 1, 1);
    auto blockDim = dim3(WAVE_SIZE , n_waves);
    gemm_kernel<<<gridDim, blockDim, 0>>>(
        (float *)D.data_ptr()
    );
    err = cudaGetLastError();
    if (err != hipSuccess)
    {
        printf("=============== Backward Kernel Launch Failed !!! =============\r\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        
    }
    return D;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}
