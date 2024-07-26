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
    int m, int n, int k)
{
    rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragA[4];
    rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragB[4];
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType> fragC;
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t> fragACC;
    // asm volatile("s_sleep 0");

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
            for (int i = 0; i < k; i += ROCWMMA_K * 2)
            {
                rocwmma::load_matrix_sync(fragA[0], A + (blk_y * k + i), k);
                // rocwmma::load_matrix_sync(fragB[0], B + (i * n + blk_x), n); // A @ B
                rocwmma::load_matrix_sync(fragB[0], B + (blk_x * k + i), k); // A @ B.T

                rocwmma::load_matrix_sync(fragA[1], A + (blk_y * k + (i + 1 * ROCWMMA_K)), k);
                // rocwmma::load_matrix_sync(fragB[1], B + ((i + 1*ROCWMMA_K) * n + blk_x), n);  // A @ B
                rocwmma::load_matrix_sync(fragB[1], B + (blk_x * k + (i + 1 * ROCWMMA_K)), k); // A @ B.T

                // rocwmma::load_matrix_sync(fragA[2], A + (blk_y * k + (i+2*ROCWMMA_K)), k);
                // rocwmma::load_matrix_sync(fragB[2], B + ((i + 2*ROCWMMA_K) * n + blk_x), n);
                // rocwmma::load_matrix_sync(fragA[3], A + (blk_y * k + (i+3*ROCWMMA_K)), k);
                // rocwmma::load_matrix_sync(fragB[3], B + ((i + 3*ROCWMMA_K) * n + blk_x), n);

                rocwmma::mma_sync(fragACC, fragA[0], fragB[0], fragACC);
                rocwmma::mma_sync(fragACC, fragA[1], fragB[1], fragACC);
                // rocwmma::mma_sync(fragACC, fragA[2], fragB[2], fragACC);
                // rocwmma::mma_sync(fragACC, fragA[3], fragB[3], fragACC);
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

    // asm volatile("s_sleep 0");
}

torch::Tensor forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int m, int n, int k)
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
        m, n, k);

    return D;
}
