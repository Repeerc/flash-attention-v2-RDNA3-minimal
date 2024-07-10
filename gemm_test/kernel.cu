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



__global__ void gemm_kernel(
    float16_t *__restrict__ A,
    float16_t *__restrict__ B, 
    float16_t *__restrict__ C, 
    float16_t *__restrict__ D,
    int m, int n, int k
)
{
    rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragA;
    rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragB;
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType> fragC;
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t> fragACC;
    asm volatile("s_sleep 0");


    const int wave_id = (threadIdx.x / WAVE_SIZE);
    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = wave_id + wave_off * N_WAVES;

        int wave_x = wave_xy % (n / ROCWMMA_N);
        int wave_y = wave_xy / (n / ROCWMMA_N);

        int blk_x = wave_x * ROCWMMA_N;
        int blk_y = wave_y * ROCWMMA_M;
        if ((blk_x < n) && (blk_y < m))
        {
            rocwmma::fill_fragment(fragACC, (float32_t)0.0);
            for (int i = 0; i < k; i += ROCWMMA_K)
            {
                rocwmma::load_matrix_sync(fragA, A + (blk_y * k + i), k);
                rocwmma::load_matrix_sync(fragB, B + (i * n + blk_x), n);
                rocwmma::mma_sync(fragACC, fragA, fragB, fragACC);
            }
            rocwmma::load_matrix_sync(fragC, C + (blk_y * n + blk_x), n, rocwmma::mem_row_major);
            for (int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = fragACC.x[i] + fragC.x[i];
            }
            rocwmma::store_matrix_sync(D + (blk_y * n + blk_x), fragC, n, rocwmma::mem_row_major);
        }
    }
    __syncthreads();

    asm volatile("s_sleep 0");

}


torch::Tensor forward(
    torch::Tensor A, 
    torch::Tensor B, 
    torch::Tensor C, 
    int m, int n, int k
)
{

    auto optD = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto D = torch::zeros({m, n}, optD);

    auto gridDim = dim3(1, 1, 1);
    auto blockDim = dim3(WAVE_SIZE * N_WAVES);
    gemm_kernel<<<gridDim, blockDim, 0>>>(
        (ComputeType *)A.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)B.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)C.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)D.data_ptr<AT_PTR_TYPE>(),
        m,n,k
    );

    return D;
}
