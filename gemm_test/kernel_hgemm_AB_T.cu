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
    float16_t *__restrict__ A_,
    float16_t *__restrict__ B_, 
    float16_t *__restrict__ C_, 
    //float16_t *__restrict__ D,
    int m, int n, int k,
    int lda, int ldb, int ldc,
    int tile_x_sz, int tile_y_sz
)
{

    fp16_frag fragA[2];
    fp16_frag fragB[2]; 
    // asm volatile("s_sleep 0");
    const int gird_x = blockIdx.x;
    const int gird_y = blockIdx.y;
    const int gird_ele_x = gird_x * tile_x_sz;
    const int gird_ele_y = gird_y * tile_y_sz;

    float16_t *A = &A_[gird_ele_y * lda  ];
    float16_t *B = &B_[gird_ele_x * ldb  ];
    float16_t *C = &C_[gird_ele_y * ldc + (gird_ele_x)];

    const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
    const int lane_id = threadIdx.x % WAVE_SIZE;
    const int wmma_lane = (threadIdx.x % 16);

    for (int wave_off = 0; wave_off < ((tile_x_sz * tile_y_sz) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = __builtin_amdgcn_readfirstlane(wave_id + wave_off * N_WAVES);

        int wave_x = __builtin_amdgcn_readfirstlane(wave_xy % (tile_x_sz / ROCWMMA_N));
        int wave_y = __builtin_amdgcn_readfirstlane(wave_xy / (tile_x_sz / ROCWMMA_N));

        int blk_x = __builtin_amdgcn_readfirstlane(wave_x * ROCWMMA_N);
        int blk_y = __builtin_amdgcn_readfirstlane(wave_y * ROCWMMA_M);
        if ((gird_ele_x + blk_x < n) && (gird_ele_y + blk_y < m))
        {

            fp32_frag fragACC;

// #pragma unroll
            for (int ele = 0; ele < 8; ++ele)
            {
                const int r = ele * 2 + (lane_id / 16);
                fragACC[ele] = (C + (blk_y * ldc + blk_x))[r * ldc + wmma_lane];
            }

            for (int i = 0; i < k; i += ROCWMMA_K*2)
            {

                fragA[0] = HALF16((A + (blk_y * lda + i))[wmma_lane * lda]);
                fragB[0] = HALF16((B + (blk_x * ldb + i))[wmma_lane * ldb]);

                fragA[1] = HALF16((A + (blk_y * lda + i + ROCWMMA_K))[wmma_lane * lda]);
                fragB[1] = HALF16((B + (blk_x * ldb + i + ROCWMMA_K))[wmma_lane * ldb]);

                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA[0], fragB[0], fragACC);
                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA[1], fragB[1], fragACC);
 
            }
            __syncthreads();

#pragma unroll
            for (int ele = 0; ele < 8; ++ele)
            {
                const int r = ele * 2 + (lane_id / 16);
                (C + (blk_y * ldc + blk_x))[r * ldc + wmma_lane] = fragACC[ele]; // n
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
    int x_blks = 4;
    int y_blks = N_WAVES / 4;


    auto gridDim = dim3(n / (x_blks * 16), m / (y_blks * 16), 1);
    auto blockDim = dim3(WAVE_SIZE * N_WAVES);
    gemm_kernel<<<gridDim, blockDim, 0>>>(
        (ComputeType *)A.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)B.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)C.data_ptr<AT_PTR_TYPE>(), 
        m,n,k,
        k,k,n,
        x_blks * 16, y_blks * 16
    );

    return C;
}
