#include <torch/types.h>
#include <torch/torch.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp16.h>

#include <rocwmma/rocwmma.hpp>
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

using rocwmma::accumulator;
using rocwmma::col_major;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::row_major;

using rocwmma::bfloat16_t;
using rocwmma::float16_t;
using rocwmma::float32_t;

#define USE_HALF 1

#if USE_HALF
#define MAX_NUM 30000.0 // 65504.0
#define ComputeType float16_t
#define AT_PTR_TYPE at::Half
#define TORCH_DTYPE torch::kFloat16
#else

#define MAX_NUM INFINITY
#define ComputeType bfloat16_t
#define AT_PTR_TYPE at::BFloat16
#define TORCH_DTYPE torch::kBFloat16

#endif

constexpr int ROCWMMA_M = 16;
constexpr int ROCWMMA_N = 16;
constexpr int ROCWMMA_K = 16;

constexpr int N_WAVES = 16;
constexpr int WAVE_SIZE = 32;


typedef _Float16 fp16_frag __attribute__((ext_vector_type(16)));
typedef float fp32_frag __attribute__((ext_vector_type(8)));
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
#define HALF16(pointer) (reinterpret_cast<fp16_frag *>((void *)&(pointer))[0])
#define HALF8(pointer) (reinterpret_cast<half8 *>((void *)&(pointer))[0])
typedef float float8 __attribute__((ext_vector_type(8)));
#define FLOAT8(pointer) (reinterpret_cast<float8 *>((void *)&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

//================================ Matrix multiplication ===============================
// C = (A^T)B + C
__device__ void mul_add_AT_B(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType *__restrict__ C,
    const int m, const int n, const int k, const float scale)
{
    rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, col_major> fragA[1];
    rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragB[1];
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType> fragC;
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t> fragACC;

    const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);

    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = __builtin_amdgcn_readfirstlane(wave_id + wave_off * N_WAVES);

        int wave_x = __builtin_amdgcn_readfirstlane(wave_xy % (n / ROCWMMA_N));
        int wave_y = __builtin_amdgcn_readfirstlane(wave_xy / (n / ROCWMMA_N));

        int blk_x = __builtin_amdgcn_readfirstlane(wave_x * ROCWMMA_N);
        int blk_y = __builtin_amdgcn_readfirstlane(wave_y * ROCWMMA_M);

        if ((blk_x < n) && (blk_y < m))
        {
            rocwmma::fill_fragment(fragACC, (float32_t)0.0);
            for (int i = 0; i < k; i += 1*ROCWMMA_K)
            {
                rocwmma::load_matrix_sync(fragA[0], A + (i * m + blk_y), m);
                rocwmma::load_matrix_sync(fragB[0], B + (i * n + blk_x), n);
                // rocwmma::load_matrix_sync(fragA[1], A + ((i+ROCWMMA_K) * m + blk_y), m);
                // rocwmma::load_matrix_sync(fragB[1], B + ((i+ROCWMMA_K) * n + blk_x), n);

                rocwmma::mma_sync(fragACC, fragA[0], fragB[0], fragACC);
                // rocwmma::mma_sync(fragACC, fragA[1], fragB[1], fragACC);
            }
            rocwmma::load_matrix_sync(fragC, C + (blk_y * n + blk_x), n, rocwmma::mem_row_major);
            for (int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = fragACC.x[i] * scale + fragC.x[i];
            }
            rocwmma::store_matrix_sync(C + (blk_y * n + blk_x), fragC, n, rocwmma::mem_row_major);
        }
    }
    //__syncthreads();
}

// C = A @ (B^T)
/*
__device__ void mul_A_BT(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType *__restrict__ C,
    const int m, const int n, const int k)
{
    rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragA[2];
    rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, col_major> fragB[2];
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType> fragC;
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t> fragACC;

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
                rocwmma::load_matrix_sync(fragB[0], B + (blk_x * k + i), k);
                rocwmma::load_matrix_sync(fragA[1], A + (blk_y * k + i + ROCWMMA_K), k);
                rocwmma::load_matrix_sync(fragB[1], B + (blk_x * k + i + ROCWMMA_K), k);

                rocwmma::mma_sync(fragACC, fragA[0], fragB[0], fragACC);
                rocwmma::mma_sync(fragACC, fragA[1], fragB[1], fragACC);
            }
            for (int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = fragACC.x[i];
            }
            rocwmma::store_matrix_sync(C + (blk_y * n + blk_x), fragC, n, rocwmma::mem_row_major);
        }
    }
    //__syncthreads();
}
*/


__device__ void mul_A_BT(
    float16_t *__restrict__ A,
    float16_t *__restrict__ B,
    float16_t *__restrict__ C,
    int m, int n, int k,
    const float scale)
{

    fp16_frag fragA[2];
    fp16_frag fragB[2];

    const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
    const int lane_id = threadIdx.x % WAVE_SIZE;
    const int wmma_lane = (threadIdx.x % 16);


    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = __builtin_amdgcn_readfirstlane(wave_id + wave_off * N_WAVES);

        int wave_x = __builtin_amdgcn_readfirstlane(wave_xy % (n / ROCWMMA_N));
        int wave_y = __builtin_amdgcn_readfirstlane(wave_xy / (n / ROCWMMA_N));

        int blk_x = __builtin_amdgcn_readfirstlane(wave_x * ROCWMMA_N);
        int blk_y = __builtin_amdgcn_readfirstlane(wave_y * ROCWMMA_M);
        if ((blk_x < n) && (blk_y < m))
        {
            fp32_frag fragACC = {};

            for (int i = 0; i < k; i += ROCWMMA_K * 2)
            {

                fragA[0] = HALF16((A + (blk_y * k + i))[wmma_lane * k]);
                fragB[0] = HALF16((B + (blk_x * k + i))[wmma_lane * k]);

                fragA[1] = HALF16((A + (blk_y * k + i + ROCWMMA_K))[wmma_lane * k]);
                fragB[1] = HALF16((B + (blk_x * k + i + ROCWMMA_K))[wmma_lane * k]);

                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA[0], fragB[0], fragACC);
                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA[1], fragB[1], fragACC);
            }
            fragACC = fragACC * scale;
            __syncthreads();

            for (int ele = 0; ele < 8; ++ele)
            {
                const int r = ele * 2 + (lane_id / 16);
                (C + (blk_y * n + blk_x))[r * n + wmma_lane] = fragACC[ele];
            }
        }
    }
    // asm volatile("s_sleep 0");
}

// C = AB + C

__device__ void mul_add_A_B(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType *__restrict__ C,
    const int m, const int n, const int k)
{

    rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragA[2];
    rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragB[2];
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType> fragC;
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t> fragACC;

    const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);

    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = __builtin_amdgcn_readfirstlane(wave_id + wave_off * N_WAVES);

        int wave_x = __builtin_amdgcn_readfirstlane(wave_xy % (n / ROCWMMA_N));
        int wave_y = __builtin_amdgcn_readfirstlane(wave_xy / (n / ROCWMMA_N));

        int blk_x = __builtin_amdgcn_readfirstlane(wave_x * ROCWMMA_N);
        int blk_y = __builtin_amdgcn_readfirstlane(wave_y * ROCWMMA_M);
        if ((blk_x < n) && (blk_y < m))
        {
            rocwmma::fill_fragment(fragACC, (float32_t)0.0);
            for (int i = 0; i < k; i += ROCWMMA_K * 2)
            {
                rocwmma::load_matrix_sync(fragA[0], A + (blk_y * k + i), k);
                rocwmma::load_matrix_sync(fragB[0], B + (i * n + blk_x), n);
                rocwmma::load_matrix_sync(fragA[1], A + (blk_y * k + (i + 1 * ROCWMMA_K)), k);
                rocwmma::load_matrix_sync(fragB[1], B + ((i + 1 * ROCWMMA_K) * n + blk_x), n);

                rocwmma::mma_sync(fragACC, fragA[0], fragB[0], fragACC);
                rocwmma::mma_sync(fragACC, fragA[1], fragB[1], fragACC);
            }
            rocwmma::load_matrix_sync(fragC, C + (blk_y * n + blk_x), n, rocwmma::mem_row_major);
            for (int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = fragACC.x[i] + fragC.x[i];
            }
            rocwmma::store_matrix_sync(C + (blk_y * n + blk_x), fragC, n, rocwmma::mem_row_major);
        }
    }
    //__syncthreads();
}

// C = AB + C
__device__ void mul_add_A_B_mask_k(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType *__restrict__ C,
    const int m, const int n, const int k, const int mask_k_start)
{

    rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragA[2];
    rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragB[2];
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType> fragC;
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t> fragACC;

    const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
    const int tid = threadIdx.x % (WAVE_SIZE);
    const int wmma_lane = (threadIdx.x % 16);
    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = __builtin_amdgcn_readfirstlane(wave_id + wave_off * N_WAVES);

        int wave_x = __builtin_amdgcn_readfirstlane(wave_xy % (n / ROCWMMA_N));
        int wave_y = __builtin_amdgcn_readfirstlane(wave_xy / (n / ROCWMMA_N));

        int blk_x = __builtin_amdgcn_readfirstlane(wave_x * ROCWMMA_N);
        int blk_y = __builtin_amdgcn_readfirstlane(wave_y * ROCWMMA_M);

        int wmma_k_end = (mask_k_start / (ROCWMMA_K * 2)) * ROCWMMA_K * 2;

        if ((blk_x < n) && (blk_y < m))
        {
            rocwmma::fill_fragment(fragACC, (float32_t)0.0);
            for (int i = 0; i < wmma_k_end; i += ROCWMMA_K * 2)
            {
                rocwmma::load_matrix_sync(fragA[0], A + (blk_y * k + i), k);
                rocwmma::load_matrix_sync(fragB[0], B + (i * n + blk_x), n);
                rocwmma::load_matrix_sync(fragA[1], A + (blk_y * k + (i + 1 * ROCWMMA_K)), k);
                rocwmma::load_matrix_sync(fragB[1], B + ((i + 1 * ROCWMMA_K) * n + blk_x), n);

                rocwmma::mma_sync(fragACC, fragA[0], fragB[0], fragACC);
                rocwmma::mma_sync(fragACC, fragA[1], fragB[1], fragACC);
            }
            rocwmma::load_matrix_sync(fragC, C + (blk_y * n + blk_x), n, rocwmma::mem_row_major);
            for (int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = fragACC.x[i] + fragC.x[i];
            }
            rocwmma::store_matrix_sync(C + (blk_y * n + blk_x), fragC, n, rocwmma::mem_row_major);

            {
                for (int y = blk_y; y < blk_y + ROCWMMA_M; y += WAVE_SIZE/ROCWMMA_M)
                {
                    int x = blk_x + (tid % ROCWMMA_N);
                    ComputeType acc0 = 0;
                    ComputeType acc1 = 0;
                    for (int i = wmma_k_end; i < mask_k_start; i++)
                    {
                        acc0 += A[y * k + i] * B[i * n + x];
                        acc1 += A[(y + 1) * k + i] * B[i * n + x];
                    }
                    C[y * n + x] += acc0;
                    C[(y + 1) * n + x] += acc1;
                }
            }
        }
    }
    //__syncthreads();
}

// =========================================================================================

template <bool pad_mask, bool causal>
__global__ void
__launch_bounds__(WAVE_SIZE *N_WAVES)
    fwd_kernel(
        ComputeType *__restrict__ q,
        ComputeType *__restrict__ k,
        ComputeType *__restrict__ v,
        ComputeType *__restrict__ o,
        float *__restrict__ L,
        const int Tr, const int Tc, const int Br, const int Bc,
        const int nq, const int nkv,
        const int d,
        const int q_stride_b, const int q_stride_h,
        const int kv_stride_b, const int kv_stride_h,
        const int L_stride_b, const int L_stride_h,
        const float32_t scale)
{

    const int q_offset = blockIdx.x * q_stride_b + blockIdx.y * q_stride_h;
    const int kv_offset = blockIdx.x * kv_stride_b + blockIdx.y * kv_stride_h;
    const int L_offset = blockIdx.x * L_stride_b + blockIdx.y * L_stride_h;

    const int Tr_i = blockIdx.z;
    if (Tr_i >= Tr)
        return;
    const int ele_y = __builtin_amdgcn_readfirstlane(Tr_i * Br);
    const int yb = __builtin_amdgcn_readfirstlane(ele_y + Br);
    const int tx = threadIdx.x;

    extern __shared__ ComputeType sram[];
    ComputeType *__restrict__ Si = &sram[0];       // Br * Bc
    ComputeType *__restrict__ Oi = &sram[Br * Bc]; // Br * d
    // ComputeType *__restrict__ Qi = &sram[Br * Bc + Br * d]; // Br * d
    // ComputeType *__restrict__ Vj = &sram[Br * Bc + 2*Br * d + 8]; // Bc * d

    if (tx < Br)
    {
#pragma unroll 4
        for (int i = 0; i < d; i += 16)
        {
            // Load Qi into sram, fill 0 to Oi
            // Qi[tx * d + i] = q[q_offset + Tr_i * Br * d + tx * d + i];
            // FLOAT8(Qi[tx * d + i]) = FLOAT8((&(q[q_offset + Tr_i * Br * d]))[tx * d + i]);
            FLOAT8(Oi[tx * d + i]) = {0, 0, 0, 0, 0, 0, 0, 0};
        }
        // #pragma unroll 32
        // for (int i = 0; i < d; i++)
        {
            // pre-scale, Si=(Q @ K^T)*scale
            // Qi[tx * d + i] *= scale * 1.442695f; // 1/ln2=1.442695, exp(x)=exp2f((1/ln2)*x)
        }
        // #pragma unroll 4
        // for (int i = 0; i < Bc; i++)
        // {
        //     FLOAT8(Si[tx * d + i]) = {0, 0, 0, 0, 0, 0, 0, 0};
        // }
    }

    __syncthreads();

    ComputeType *__restrict__ Qi = &q[q_offset + Tr_i * Br * d];
    // ComputeType *__restrict__ Oi = &o[q_offset + Tr_i * Br * d];

    float32_t row_max_old = -INFINITY;
    float32_t l_i = 0;

    for (int j = 0; j < Tc; j++)
    {

        ComputeType *__restrict__ Kj = &k[kv_offset + j * Bc * d];
        ComputeType *__restrict__ Vj = &v[kv_offset + j * Bc * d];
        int ele_x = j * Bc;
        int xr = ele_x + Bc;
        float32_t row_max_new = -INFINITY; // mij
        float32_t row_sum = 0;
        float32_t rowmax_diff_exp = 0; // Sij - mij
        //------------ Sij = Qi @ Kj^T
        if constexpr (!causal)
        {
            mul_A_BT(Qi, Kj, Si, Br, Bc, d, scale);
        }
        else
        {
            if (ele_y >= ele_x)
            {
                mul_A_BT(Qi, Kj, Si, Br, Bc, d, scale);
                __syncthreads();
            }
            if ((ele_y < ele_x + Bc - 1) && (tx < Br))
            {
#pragma unroll 32
                for (int i = 0; i < Bc; i++)
                {
                    if (i >= tx + (ele_y - ele_x + 1))
                        Si[tx * Bc + i] = -MAX_NUM;
                }
            }
        }
        __syncthreads();
        //------------
        if constexpr (pad_mask)
        {
            if (unlikely((xr > nkv) && (tx < Br)))
            {
#pragma unroll 32
                for (int i = nkv - ele_x; i < Bc; i++)
                    Si[tx * Bc + i] = -MAX_NUM;
            }

            if (unlikely((yb > nq) && (tx < Bc)))
            {
#pragma unroll 32
                for (int i = nq - ele_y; i < Br; i++)
                    Si[i * Bc + tx] = -MAX_NUM;
            }
            __syncthreads();
        }
        //------------

        if (tx < Br)
        {
// --------------------- find every row max val in Si[Br * Bc]
#pragma unroll 4
            for (int i = 0; i < Bc; i += 8)
            {
                half8 val = HALF8(Si[(tx * Bc) + i]);
                float8 val_f32;
#pragma unroll
                for (int j = 0; j < 8; j++)
                    val_f32[j] = val[j];

#pragma unroll
                for (int j = 0; j < 8; j++)
                    row_max_new = max(row_max_new, val_f32[j]);
            }


            row_max_new = max(row_max_old, row_max_new);
            rowmax_diff_exp = exp2f(row_max_old - row_max_new);
            row_max_old = row_max_new;

//--------------------Calc Pi = exp(Si - mi) and rowsum
#pragma unroll 4
            for (int i = 0; i < Bc; i += 8)
            {
                half8 val = HALF8(Si[(tx * Bc) + i]);
                float8 val_f32;
#pragma unroll // Load fp16 into VGPRs and convert to FP32
                for (int j = 0; j < 8; j++)
                    val_f32[j] = val[j];
// Si - mi
                val_f32 = val_f32 - row_max_new;
#pragma unroll // exp but using exp2 instead.
                for (int j = 0; j < 8; j++)
                    val_f32[j] = exp2f(val_f32[j]);

#pragma unroll // calc rowsum
                for (int j = 0; j < 8; j++)
                    row_sum += val_f32[j];

#pragma unroll // convert back to fp16
                for (int j = 0; j < 8; j++)
                    val[j] = val_f32[j];

               // write back
                HALF8(Si[(tx * Bc) + i]) = val;
            }

            l_i = rowmax_diff_exp * l_i + row_sum;

// --------------------- calc: Oi *= exp2f(row_max_old - row_max_new)
#pragma unroll 4
            for (int i = 0; i < d; i += 8)
            {
                half8 val = HALF8(Oi[(tx * d) + i]);
                float8 val_f32;
#pragma unroll
                for (int j = 0; j < 8; j++)
                    val_f32[j] = val[j];

                val_f32 = val_f32 * rowmax_diff_exp;

#pragma unroll
                for (int j = 0; j < 8; j++)
                    val[j] = val_f32[j];
                    
                HALF8(Oi[(tx * d) + i]) = val;
            }
// --------------------- 
        }
        __syncthreads();

        if constexpr (!pad_mask)
            mul_add_A_B(Si, Vj, Oi, Br, d, Bc);
        else
        {
            if (unlikely(xr > nkv))
            {
                mul_add_A_B_mask_k(Si, Vj, Oi, Br, d, Bc, Bc - (xr - nkv));
            }
            else
            {
                mul_add_A_B(Si, Vj, Oi, Br, d, Bc);
            }
        }

        __syncthreads();
    }

    if (tx < Br)
    {
// #pragma unroll 32
//         for (int i = 0; i < d; i++)
//             Oi[tx * d + i] = Oi[tx * d + i] / l_i;

//------------------------ Calc: Oi /= li  Write back: Oi
#pragma unroll 4
            for (int i = 0; i < d; i += 8)
            {
                half8 val = HALF8(Oi[(tx * d) + i]);
                float8 val_f32;
#pragma unroll
                for (int j = 0; j < 8; j++)
                    val_f32[j] = val[j];

                val_f32 = val_f32 / l_i;

#pragma unroll
                for (int j = 0; j < 8; j++)
                    val[j] = val_f32[j];
                    
                //HALF8(Oi[(tx * d) + i]) = val;
                HALF8((&(o[q_offset + Tr_i * Br * d]))[tx * d + i]) = val;
            }

// #pragma unroll 4
//         for (int i = 0; i < d; i += 16)
//             // o[q_offset + Tr_i * Br * d + tx * d + i] = Oi[tx * d + i];
//             FLOAT8((&(o[q_offset + Tr_i * Br * d]))[tx * d + i]) = FLOAT8(Oi[tx * d + i]);

        l_i = row_max_old + log2f(l_i);
        L[L_offset + Tr_i * Br + tx] = l_i;
    }
}
// =================================================================================

__global__ void
__launch_bounds__(WAVE_SIZE *N_WAVES)
bwd_kernel(
    ComputeType *__restrict__ q,  // [(b*h) x N x d]
    ComputeType *__restrict__ k,  // [(b*h) x N x d]
    ComputeType *__restrict__ v,  // [(b*h) x N x d]
    ComputeType *__restrict__ O,  // [(b*h) x N x d]
    ComputeType *__restrict__ dO, // [(b*h) x N x d]
    ComputeType *__restrict__ dQ, // [(b*h) x N x d]
    ComputeType *__restrict__ dK, // [(b*h) x N x d]
    ComputeType *__restrict__ dV, // [(b*h) x N x d]
    float *__restrict__ Di,       // [(b*h) * N]
    float *__restrict__ L,        // [(b*h) * N]
    const int Tr, const int Tc,
    const int Br, const int Bc,
    const int nq, const int nkv,
    const int d,
    const int Q_O_dO_stride_b, const int Q_O_dO_stride_h,
    const int kvDkv_stride_b, const int kvdKv_stride_h,
    const int L_stride_b, const int L_stride_h,
    const bool pad_mask,
    const float32_t scale,
    const bool causal)

{

    const int q_offset = Q_O_dO_stride_b * blockIdx.x + Q_O_dO_stride_h * blockIdx.y;
    const int kv_offset = kvDkv_stride_b * blockIdx.x + kvdKv_stride_h * blockIdx.y;
    const int L_offset = L_stride_b * blockIdx.x + L_stride_h * blockIdx.y;

    const int Tc_j = blockIdx.z;
    if (Tc_j >= Tc)
        return;
    const int ele_x = Tc_j * Bc;
    const int tx = threadIdx.x;

    extern __shared__ ComputeType sram[];
    ComputeType *__restrict__ Si = &sram[0]; //[Br x Bc]
    ComputeType *__restrict__ Pi = &sram[0];
    ComputeType *__restrict__ dSi = &sram[0];
    ComputeType *__restrict__ dPi = &sram[Br * Bc];    //[Br x Bc]
    // ComputeType *__restrict__ Kj = &sram[2 * Br * Bc]; // [Bc x d]

    ComputeType *__restrict__ Kj = &k[kv_offset + Tc_j * Bc * d]; // [Bc x d]
    ComputeType *__restrict__ Vj = &v[kv_offset + Tc_j * Bc * d]; // [Bc x d]

    ComputeType *__restrict__ dKj = &dK[kv_offset + Tc_j * Bc * d]; // [Bc x d]
    ComputeType *__restrict__ dVj = &dV[kv_offset + Tc_j * Bc * d]; // [Bc x d]

    for (int n_batch = 0; n_batch < ((nq + (blockDim.x - 1)) / blockDim.x); n_batch++)
    {
        int Di_off = n_batch * (blockDim.x) + tx;
        if (Di_off < nq)
        {
            float32_t val = 0;
#pragma unroll 32
            for (int i = 0; i < d; i++)
            {
                val += (dO[q_offset + Di_off * d + i] * O[q_offset + Di_off * d + i]);
            }
            Di[L_offset + Di_off] = val;
        }
    }

    //     if (tx < d)
    //     {
    // #pragma unroll 32
    //         for (int i = 0; i < Bc; i++)
    //         {
    //             Kj[i * d + tx] = scale * 1.442695f * k[kv_offset + Tc_j * Bc * d + i * d + tx];
    //         }
    //     }

    __syncthreads();

    for (int Tr_i = 0; Tr_i < Tr; Tr_i++)
    {
        ComputeType *__restrict__ Qi = &q[q_offset + Tr_i * Br * d];   // [Br x d]
        ComputeType *__restrict__ Oi = &O[q_offset + Tr_i * Br * d];   // [Br x d]
        ComputeType *__restrict__ dOi = &dO[q_offset + Tr_i * Br * d]; // [Br x d]
        ComputeType *__restrict__ dQi = &dQ[q_offset + Tr_i * Br * d]; // [Br x d]
        float32_t *__restrict__ Li = &L[L_offset + Tr_i * Br];         // [Br]
        float32_t *__restrict__ Di_i = &Di[L_offset + Tr_i * Br];
        int ele_y = Tr_i * Br;
        int yb = ele_y + Br;
        int xr = ele_x + Bc;

        mul_A_BT(Qi, Kj, Si, Br, Bc, d, scale); // Qi[Br x d] Kj[Bc x d]
        __syncthreads();
        if (unlikely(causal))
        {
            if ((ele_y < ele_x + Bc - 1) && (tx < Br))
            {
#pragma unroll 32
                for (int i = 0; i < Bc; i++)
                {
                    if (i >= tx + (ele_y - ele_x + 1))
                        Si[tx * Bc + i] = -MAX_NUM;
                }
            }
            __syncthreads();
        }

        if (pad_mask)
        {
            if (unlikely((xr > nkv) && (tx < Br)))
            {
#pragma unroll 32
                for (int i = nkv - ele_x; i < Bc; i++)
                    Si[tx * Bc + i] = -MAX_NUM;
            }

            if (unlikely((yb > nq) && (tx < Bc)))
            {
#pragma unroll 32
                for (int i = nq - ele_y; i < Br; i++)
                    Si[i * Bc + tx] = -MAX_NUM;
            }
            __syncthreads();
        }

        if (tx < Br)
        {
// #pragma unroll 32
//             for (int i = 0; i < Bc; i++)
//             {
//                 Pi[tx * Bc + i] = exp2f(Si[tx * Bc + i] - Li[tx]); // Pi [Br x Bc]
//             }

            float32_t row_max = Li[tx];
#pragma unroll 4
            for (int i = 0; i < Bc; i += 8)
            {
                half8 val = HALF8(Si[(tx * Bc) + i]);
                float8 val_f32;
#pragma unroll // Load fp16 into VGPRs and convert to FP32
                for (int j = 0; j < 8; j++)
                    val_f32[j] = val[j];
// Si - mi
                val_f32 = val_f32 - row_max;
#pragma unroll // exp but using exp2 instead.
                for (int j = 0; j < 8; j++)
                    val_f32[j] = exp2f(val_f32[j]);

#pragma unroll // convert back to fp16
                for (int j = 0; j < 8; j++)
                    val[j] = val_f32[j];

               // write back
                HALF8(Pi[(tx * Bc) + i]) = val;
            }

        }
        __syncthreads();

        mul_add_AT_B(Pi, dOi, dVj, Bc, d, Br, 1); // Pi[Br x Bc] @ dOi[Br x d]
        mul_A_BT(dOi, Vj, dPi, Br, Bc, d, 1);  // dPi:[Br x Bc]
        __syncthreads();
        if (tx < Br)
        {
#pragma unroll 32
            for (int i = 0; i < Bc; i++)
            {
                dSi[tx * Bc + i] = scale * Pi[tx * Bc + i] * (dPi[tx * Bc + i] - Di_i[tx]);
            }
        }
        __syncthreads();
        mul_add_A_B(dSi, Kj, dQi, Br, d, Bc);  // dSi[Br x Bc] @ Kj[Bc x d]
        mul_add_AT_B(dSi, Qi, dKj, Bc, d, Br, 0.69314718f); // dSi[Br x Bc] @ Qi[Br x d]
        __syncthreads();
    }
}

// =================================================================================

std::vector<torch::Tensor> forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    const int Br, const int Bc,
    const bool causal,
    const float scale)
{

    auto q_pad = q;
    auto k_pad = k;
    auto v_pad = v;

    const int b = q.size(0);
    const int h = q.size(1);
    const int n = q.size(2);
    const int d = q.size(3);
    const int n_kv = k.size(2);

    int Nq_pad_sz = (Br - (n % Br)) % Br;
    int Nkv_pad_sz = (Bc - (n_kv % Bc)) % Bc;
    int d_pad_sz = ((ROCWMMA_K * 2) - (d % (ROCWMMA_K * 2))) % (ROCWMMA_K * 2);

    const bool pad_mask = Nq_pad_sz || Nkv_pad_sz;

    if (Nq_pad_sz || d_pad_sz)
    {
        q_pad = torch::nn::functional::pad(q_pad, torch::nn::functional::PadFuncOptions({0, d_pad_sz, 0, Nq_pad_sz}));
    }
    // if (Nkv_pad_sz || d_pad_sz)
    if (d_pad_sz)
    {
        k_pad = torch::nn::functional::pad(k_pad, torch::nn::functional::PadFuncOptions({0, d_pad_sz, 0, 0}));
        v_pad = torch::nn::functional::pad(v_pad, torch::nn::functional::PadFuncOptions({0, d_pad_sz, 0, 0}));
    }
    // if (q_pad.stride(-1) != 1)
        q_pad = q_pad.contiguous();

    // if (k_pad.stride(-1) != 1)
        k_pad = k_pad.contiguous();

    // if (v_pad.stride(-1) != 1)
        v_pad = v_pad.contiguous();

    const int Tr = ceil((float)n / Br);
    const int Tc = ceil((float)n_kv / Bc);

    // auto opt = torch::TensorOptions().dtype(TORCH_DTYPE).device(torch::kCUDA);
    auto O = torch::zeros_like(q_pad);

    auto opt2 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto L = torch::zeros({b, h, n + Nq_pad_sz}, opt2);

    auto blockDim = dim3(WAVE_SIZE * N_WAVES);
    int nblk = b * h * Tr;
    int trPad = 96 - (nblk % 96); // TODO: 96 CU only for gfx1100

    auto gridDim = dim3(b, h, Tr + trPad);

    const int sram_sz =
        Br * Bc * sizeof(ComputeType)               // Si
        + Br * (d + d_pad_sz) * sizeof(ComputeType) // Oi
        // + Br * (d + d_pad_sz) * sizeof(ComputeType) // Qi
        // + Bc * (d + d_pad_sz) * sizeof(ComputeType) // Vj
        ;

#define para_fwd                                      \
        (ComputeType *)q_pad.data_ptr<AT_PTR_TYPE>(), \
        (ComputeType *)k_pad.data_ptr<AT_PTR_TYPE>(), \
        (ComputeType *)v_pad.data_ptr<AT_PTR_TYPE>(), \
        (ComputeType *)O.data_ptr<AT_PTR_TYPE>(),     \
        (float *)L.data_ptr<float>(),                 \
        Tr, Tc, Br, Bc,                               \
        n, n_kv,                                      \
        d + d_pad_sz,                                 \
        q_pad.stride(0), q_pad.stride(1),             \
        k_pad.stride(0), k_pad.stride(1),             \
        L.stride(0), L.stride(1),                     \
        scale * 1.442695f

    cudaError_t err = cudaGetLastError();

    if (!pad_mask && !causal)
        fwd_kernel<false, false><<<gridDim, blockDim, sram_sz>>>(para_fwd);
    else if (pad_mask && causal)
        fwd_kernel<true, true><<<gridDim, blockDim, sram_sz>>>(para_fwd);
    else if (!pad_mask && causal)
        fwd_kernel<false, true><<<gridDim, blockDim, sram_sz>>>(para_fwd);
    else if (pad_mask && !causal)
        fwd_kernel<true, false><<<gridDim, blockDim, sram_sz>>>(para_fwd);

    err = cudaGetLastError();
    if (err != hipSuccess)
    {
        printf("=============== Kernel Launch Failed !!! =============\r\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        printf("Br:%d, Bc:%d \r\n", Br, Bc);
        printf("Tr:%d, Tc:%d \r\n", Tr, Tc);
        printf("B:%d, H:%d, Qn:%d, KVn:%d, d:%d \r\n", b, h, n, n_kv, d);
        printf("SRAM Requirements:%d \r\n", sram_sz);
    }

    auto O_fwd = O.index({"...",
                          torch::indexing::Slice(torch::indexing::None, n),
                          torch::indexing::Slice(torch::indexing::None, d)});

    return {O_fwd, q_pad, k_pad, v_pad, O, L};
}

std::vector<torch::Tensor> backward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor L,
    const int act_n,
    const int act_nkv,
    const int act_d,
    const int Br,
    const int Bc,
    const bool causal,
    const float scale)
{
    const int b = Q.size(0);
    const int h = Q.size(1);
    const int n = Q.size(2);
    const int d = Q.size(3);
    const int n_kv = K.size(2);

    //    const int dO_Npad_sz = Q.size(2) - dO.size(2);
    const int dO_Dpad_sz = Q.size(3) - dO.size(3);

    int Nq_pad_sz = (Br - (n % Br)) % Br;
    int Nkv_pad_sz = (Bc - (n_kv % Bc)) % Bc;
    const bool pad_mask = (Nkv_pad_sz || Nq_pad_sz || (n_kv != act_nkv));

    dO = torch::nn::functional::pad(dO, torch::nn::functional::PadFuncOptions({0, dO_Dpad_sz, 0, Nq_pad_sz}));
    Q = torch::nn::functional::pad(Q, torch::nn::functional::PadFuncOptions({0, 0, 0, Nq_pad_sz}));
    O = torch::nn::functional::pad(O, torch::nn::functional::PadFuncOptions({0, 0, 0, Nq_pad_sz}));
    L = torch::nn::functional::pad(L, torch::nn::functional::PadFuncOptions({0, Nq_pad_sz}));
    K = torch::nn::functional::pad(K, torch::nn::functional::PadFuncOptions({0, 0, 0, Nkv_pad_sz}));
    V = torch::nn::functional::pad(V, torch::nn::functional::PadFuncOptions({0, 0, 0, Nkv_pad_sz}));

    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    dO = dO.contiguous();
    O = O.contiguous();
    L = L.contiguous();

    const int Tr = ceil((float)act_n / Br);
    const int Tc = ceil((float)act_nkv / Bc);

    auto opt = torch::TensorOptions().dtype(TORCH_DTYPE).device(torch::kCUDA);

    auto dQ = torch::zeros_like(Q, opt);
    auto dK = torch::zeros_like(K, opt);
    auto dV = torch::zeros_like(V, opt);

    auto Di = torch::zeros_like(L);

    int nblk = b * h * Tc;
    int tcPad = 96 - (nblk % 96); // TODO: 96 CU only for gfx1100
    auto gridDim = dim3(b, h, Tc + tcPad);
    auto blockDim = dim3(WAVE_SIZE * N_WAVES);

    const int sram_sz =
        2 * Br * Bc * sizeof(ComputeType) // (Si,Pi,dSi), dPi
        // + Bc * d * sizeof(ComputeType)    // Kj
        ;

    cudaError_t err = cudaGetLastError();

    bwd_kernel<<<gridDim, blockDim, sram_sz>>>(
        (ComputeType *)Q.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)K.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)V.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)O.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)dO.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)dQ.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)dK.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)dV.data_ptr<AT_PTR_TYPE>(),
        (float *)Di.data_ptr<float>(),
        (float *)L.data_ptr<float>(),
        Tr, Tc,
        Br, Bc,
        act_n, act_nkv,
        d,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        L.stride(0), L.stride(1),
        pad_mask,
        scale * 1.442695f, causal);

    err = cudaGetLastError();
    if (err != hipSuccess)
    {
        printf("=============== Backward Kernel Launch Failed !!! =============\r\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        printf("Br:%d, Bc:%d \r\n", Br, Bc);
        printf("Tr:%d, Tc:%d \r\n", Tr, Tc);
        printf("B:%d, H:%d, Qn:%d, KVn:%d, d:%d \r\n", b, h, n, n_kv, d);
        printf("SRAM Requirements:%d \r\n", sram_sz);
    }

    // if (dO_Dpad_sz || pad_mask)
    {
        dQ = dQ.index({"...",
                       torch::indexing::Slice(torch::indexing::None, act_n),
                       torch::indexing::Slice(torch::indexing::None, act_d)});
        dK = dK.index({"...",
                       torch::indexing::Slice(torch::indexing::None, act_nkv),
                       torch::indexing::Slice(torch::indexing::None, act_d)});
        dV = dV.index({"...",
                       torch::indexing::Slice(torch::indexing::None, act_nkv),
                       torch::indexing::Slice(torch::indexing::None, act_d)});
    }

    return {dQ, dK, dV};
}
