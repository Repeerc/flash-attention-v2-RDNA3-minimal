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

const int ROCWMMA_M = 32;
const int ROCWMMA_N = 32;
const int ROCWMMA_K = 32;

const int N_WAVES = 32;
const int WAVE_SIZE = 32;

#define ComputeType float32_t
#define AT_PTR_TYPE float
#define TORCH_DTYPE torch::kFloat32

__global__ void gemm_kernel(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType *__restrict__ C,
    ComputeType *__restrict__ D,
    int m, int n, int k)
{
    asm volatile("s_sleep 0");

    const int wave_id = (threadIdx.x / WAVE_SIZE);
    const int lane_id = (threadIdx.x % WAVE_SIZE);

    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = wave_id + wave_off * N_WAVES;

        int wave_x = wave_xy % (n / ROCWMMA_N);
        int wave_y = wave_xy / (n / ROCWMMA_N);

        int blk_x = wave_x * ROCWMMA_N;
        int blk_y = wave_y * ROCWMMA_M;
        if ((blk_x < n) && (blk_y < m))
        {
#pragma unroll ROCWMMA_N
            for (int col = 0; col < ROCWMMA_N; col++)
            {
                float sum = C[(blk_y + lane_id) * n + blk_x + col];
#pragma unroll ROCWMMA_K
                for (int i = 0; i < k; i++)
                {
                    sum += A[(blk_y + lane_id) * k + (i)] * B[(i)*n + (blk_x + col)];
                }
                D[(blk_y + lane_id) * n + (blk_x + col)] = sum;
            }
        }
    }
    __syncthreads();

    asm volatile("s_sleep 0");
}

torch::Tensor forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int m, int n, int k)
{

    auto optD = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
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
