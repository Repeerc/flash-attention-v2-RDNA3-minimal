#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp16.h>

#include <rocwmma/rocwmma.hpp>

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

const int N_WAVES = 32;
const int WAVE_SIZE = 32;


#define ComputeType float16_t
#define AT_PTR_TYPE at::Half
#define TORCH_DTYPE torch::kFloat16

typedef _Float16 fp16_frag __attribute__((ext_vector_type(16)));
typedef float fp32_frag __attribute__((ext_vector_type(8)));

#define HALF16(pointer) (reinterpret_cast<fp16_frag *>((void *)&(pointer))[0])


__global__ void gemm_kernel(
    float16_t *__restrict__ A,
    float16_t *__restrict__ B, 
    float16_t *__restrict__ C, 
    //float16_t *__restrict__ D,
    int m, int n, int k
)
{

    fp16_frag fragA[2];
    fp16_frag fragB[2]; 


    // asm volatile("s_sleep 0");


    const int wave_id = threadIdx.x / WAVE_SIZE;
    const int lane_id = threadIdx.x % WAVE_SIZE;
    const int wmma_lane = (threadIdx.x % 16);

    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = wave_id + wave_off * N_WAVES;

        int wave_x = wave_xy % (n / ROCWMMA_N);
        int wave_y = wave_xy / (n / ROCWMMA_N);

        int blk_x = wave_x * ROCWMMA_N;
        int blk_y = wave_y * ROCWMMA_M;
        if ((blk_x < n) && (blk_y < m))
        {

            fp32_frag fragACC;

#pragma unroll
            for (int ele = 0; ele < 8; ++ele)
            {
                const int r = ele * 2 + (lane_id / 16);
                fragACC[ele] = (C + (blk_y * n + blk_x))[r * n + wmma_lane];
            }

            for (int i = 0; i < k; i += ROCWMMA_K*2)
            {
                // for(int ele = 0; ele < 16; ele++)
                // {
                //     fragA[ele] = (A + (blk_y * k + i))[wmma_lane * k + ele]; //lda = k
                //     //fragB[ele] = (B + (i * n + blk_x))[ele * n + wmma_lane]; // A @ B 
                //     fragB[ele] = (B + (blk_x * k + i))[wmma_lane * k + ele];   // A @ B^T

                // }

                //fragA = (A + (blk_y * k + i))[wmma_lane * k + ele];

                fragA[0] = HALF16((A + (blk_y * k + i))[wmma_lane * k]);
                fragB[0] = HALF16((B + (blk_x * k + i))[wmma_lane * k]);

                fragA[1] = HALF16((A + (blk_y * k + i + ROCWMMA_K))[wmma_lane * k]);
                fragB[1] = HALF16((B + (blk_x * k + i + ROCWMMA_K))[wmma_lane * k]);


                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA[0], fragB[0], fragACC);
                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA[1], fragB[1], fragACC);
                

            }
            __syncthreads();

#pragma unroll
            for (int ele = 0; ele < 8; ++ele)
            {
                const int r = ele * 2 + (lane_id / 16);
                (C + (blk_y * n + blk_x))[r * n + wmma_lane] =  fragACC[ele];
            }

        }
    }
    __syncthreads();

    // asm volatile("s_sleep 0");

}


torch::Tensor forward(
    torch::Tensor A, 
    torch::Tensor B, 
    torch::Tensor C, 
    int m, int n, int k
)
{

    //auto optD = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    //auto D = torch::zeros({m, n}, optD);

    auto gridDim = dim3(1, 1, 1);
    auto blockDim = dim3(WAVE_SIZE * N_WAVES);
    gemm_kernel<<<gridDim, blockDim, 0>>>(
        (ComputeType *)A.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)B.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)C.data_ptr<AT_PTR_TYPE>(), 
        m,n,k
    );

    return C;
}
