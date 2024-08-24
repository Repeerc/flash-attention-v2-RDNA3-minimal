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

#define USE_HALF 0

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

//constexpr int N_WAVES = 16;
constexpr int WAVE_SIZE = 32;


typedef uint16_t bf16_frag __attribute__((ext_vector_type(16)));
typedef float fp32_frag __attribute__((ext_vector_type(8)));
typedef uint16_t bhalf8 __attribute__((ext_vector_type(8)));
typedef uint16_t bhalf16 __attribute__((ext_vector_type(16)));
#define HALF16(pointer) (reinterpret_cast<bhalf16 *>((void *)&(pointer))[0])
#define HALF8(pointer) (reinterpret_cast<bhalf8 *>((void *)&(pointer))[0])
typedef float float8 __attribute__((ext_vector_type(8)));
typedef float float_v16 __attribute__((ext_vector_type(16)));
#define FLOAT8(pointer) (reinterpret_cast<float8 *>((void *)&(pointer))[0])
#define FLOATV16(pointer) (reinterpret_cast<float_v16 *>((void *)&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

__device__ __forceinline__ uint16_t f32_to_bf16(float val)
{
    
    uint16_t res = 0;
    union
    {
        float val_f32;
        uint32_t val_u32;
    } u = {val};
    res = u.val_u32 >> 16;
    return res;
}

__device__ __forceinline__ float bf16_to_f32(uint16_t val)
{
    union
    {
        float val_f32;
        uint32_t val_u32;
    } u;
    u.val_u32 = val << 16;
    return u.val_f32;
}

//================================ Matrix multiplication ===============================
// C = (A^T)B + C
template <int N_WAVES>
__device__ void mul_add_AT_B(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType *__restrict__ C,
    int lda, int ldb, int ldc,
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
                rocwmma::load_matrix_sync(fragA[0], A + (i * lda + blk_y), lda); // m
                rocwmma::load_matrix_sync(fragB[0], B + (i * ldb + blk_x), ldb); // n

                // rocwmma::load_matrix_sync(fragA[1], A + ((i+ROCWMMA_K) * lda + blk_y), lda); 
                // rocwmma::load_matrix_sync(fragB[1], B + ((i+ROCWMMA_K) * ldb + blk_x), ldb);

                rocwmma::mma_sync(fragACC, fragA[0], fragB[0], fragACC);
                // rocwmma::mma_sync(fragACC, fragA[1], fragB[1], fragACC);
            }
            rocwmma::load_matrix_sync(fragC, C + (blk_y * ldc + blk_x), ldc, rocwmma::mem_row_major); // n
            for (int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = fragACC.x[i] * scale + fragC.x[i];
            }
            rocwmma::store_matrix_sync(C + (blk_y * ldc + blk_x), fragC, ldc, rocwmma::mem_row_major);
        }
    }
    //__syncthreads();
}
 
// C = A @ (B^T)
template <int N_WAVES>
__device__ void mul_A_BT(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType *__restrict__ C,
    int lda, int ldb, int ldc,
    int m, int n, int k,
    const float scale)
{

    bf16_frag fragA[2];
    bf16_frag fragB[2];

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

                fragA[0] = HALF16((A + (blk_y * lda + i))[wmma_lane * lda]); // k
                fragB[0] = HALF16((B + (blk_x * ldb + i))[wmma_lane * ldb]); // k

                fragA[1] = HALF16((A + (blk_y * lda + i + ROCWMMA_K))[wmma_lane * lda]);
                fragB[1] = HALF16((B + (blk_x * ldb + i + ROCWMMA_K))[wmma_lane * ldb]);
                // fragA[2] = HALF16((A + (blk_y * k + i + 2*ROCWMMA_K))[wmma_lane * k]);
                // fragB[2] = HALF16((B + (blk_x * k + i + 2*ROCWMMA_K))[wmma_lane * k]);
                // fragA[3] = HALF16((A + (blk_y * k + i + 3*ROCWMMA_K))[wmma_lane * k]);
                // fragB[3] = HALF16((B + (blk_x * k + i + 3*ROCWMMA_K))[wmma_lane * k]);

                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[0], fragB[0], fragACC);
                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[1], fragB[1], fragACC);
                // fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[2], fragB[2], fragACC);
                // fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[3], fragB[3], fragACC);
            }
            fragACC = fragACC * scale;
            __syncthreads();

            for (int ele = 0; ele < 8; ++ele)
            {
                const int r = ele * 2 + (lane_id / 16);
                (C + (blk_y * ldc + blk_x))[r * ldc + wmma_lane] = fragACC[ele]; // n
            }
        }
    }
    // asm volatile("s_sleep 0");
}


template <int N_WAVES>
__device__ void mul_add_A_B(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType *__restrict__ C,
    int lda, int ldb, int ldc,
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
                rocwmma::load_matrix_sync(fragA[0], A + (blk_y * lda + i), lda); //k
                rocwmma::load_matrix_sync(fragB[0], B + (i * ldb + blk_x), ldb); //n
                
                rocwmma::load_matrix_sync(fragA[1], A + (blk_y * lda + (i + 1 * ROCWMMA_K)), lda);
                rocwmma::load_matrix_sync(fragB[1], B + ((i + 1 * ROCWMMA_K) * ldb + blk_x), ldb);
                // rocwmma::load_matrix_sync(fragA[2], A + (blk_y * k + (i + 2 * ROCWMMA_K)), k);
                // rocwmma::load_matrix_sync(fragB[2], B + ((i + 2 * ROCWMMA_K) * n + blk_x), n);
                // rocwmma::load_matrix_sync(fragA[3], A + (blk_y * k + (i + 3 * ROCWMMA_K)), k);
                // rocwmma::load_matrix_sync(fragB[3], B + ((i + 3 * ROCWMMA_K) * n + blk_x), n);

                rocwmma::mma_sync(fragACC, fragA[0], fragB[0], fragACC);
                rocwmma::mma_sync(fragACC, fragA[1], fragB[1], fragACC);
                // rocwmma::mma_sync(fragACC, fragA[2], fragB[2], fragACC);
                // rocwmma::mma_sync(fragACC, fragA[3], fragB[3], fragACC);
            }
            rocwmma::load_matrix_sync(fragC, C + (blk_y * ldc + blk_x), ldc, rocwmma::mem_row_major); //n
            for (int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = fragACC.x[i] + fragC.x[i];
            }
            rocwmma::store_matrix_sync(C + (blk_y * ldc + blk_x), fragC, ldc, rocwmma::mem_row_major); //n
        }
    }
    //__syncthreads();
}

// C = AB + C
template <int N_WAVES>
__device__ void mul_add_A_B_mask_k(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType *__restrict__ C,
    int lda, int ldb, int ldc,
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
                rocwmma::load_matrix_sync(fragA[0], A + (blk_y * lda + i), lda); //k
                rocwmma::load_matrix_sync(fragB[0], B + (i * ldb + blk_x), ldb);  //n
                rocwmma::load_matrix_sync(fragA[1], A + (blk_y * lda + (i + 1 * ROCWMMA_K)), lda);
                rocwmma::load_matrix_sync(fragB[1], B + ((i + 1 * ROCWMMA_K) * ldb + blk_x), ldb);

                rocwmma::mma_sync(fragACC, fragA[0], fragB[0], fragACC);
                rocwmma::mma_sync(fragACC, fragA[1], fragB[1], fragACC);
            }
            rocwmma::load_matrix_sync(fragC, C + (blk_y * ldc + blk_x), ldc, rocwmma::mem_row_major); //n
            for (int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = fragACC.x[i] + fragC.x[i];
            }
            rocwmma::store_matrix_sync(C + (blk_y * ldc + blk_x), fragC, ldc, rocwmma::mem_row_major);

            {
                for (int y = blk_y; y < blk_y + ROCWMMA_M; y += WAVE_SIZE/ROCWMMA_M)
                {
                    int x = blk_x + (tid % ROCWMMA_N);
                    float32_t acc0 = (0);
                    float32_t acc1 = (0);
                    for (int i = wmma_k_end; i < mask_k_start; i++)
                    {
                        acc0 += A[y * lda + i] * B[i * ldb + x];  // k n
                        acc1 += A[(y + 1) * lda + i] * B[i * ldb + x];
                    }
                    C[y * ldc + x] = C[y * ldc + x] + acc0;
                    C[(y + 1) * ldc + x] = C[(y + 1) * ldc + x] + acc1; // n
                }
            }
        }
    }
    //__syncthreads();
}

// =========================================================================================

template <bool pad_mask, bool causal, int N_WAVES>
__global__ void
__launch_bounds__(WAVE_SIZE * N_WAVES)
    fwd_kernel(
        ComputeType *__restrict__ q,
        ComputeType *__restrict__ k,
        ComputeType *__restrict__ v,
        ComputeType *__restrict__ o,
        float *__restrict__ L,
        const int Tr, const int Tc, const int Br, const int Bc,
        const int nq, const int nkv,
        const int d,const int h,
        const int64_t q_stride0,const int64_t q_stride1,const int64_t q_stride2,
        const int64_t kv_stride0,const int64_t kv_stride1,const int64_t kv_stride2,
        const int L_stride_b, const int L_stride_h,
        const float32_t scale, const bool permute_NH)
{

    int q_offset = blockIdx.x * q_stride0 + blockIdx.y * q_stride1;
    int kv_offset = blockIdx.x * kv_stride0 + blockIdx.y * kv_stride1;

    int ld_qkv = q_stride2; //d;
    if(permute_NH)
    {
        q_offset = blockIdx.x * q_stride0 + blockIdx.y * q_stride2;
        kv_offset = blockIdx.x * kv_stride0 + blockIdx.y * kv_stride2;
        ld_qkv = q_stride1; //h * d;
    }

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
    // ComputeType *__restrict__ Vj = &sram[Br * Bc + Br * d]; // Bc * d

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

    ComputeType *__restrict__ Qi = &q[q_offset + (Tr_i * Br) * ld_qkv];
    // ComputeType *__restrict__ Oi = &o[q_offset + Tr_i * Br * d];

    float32_t row_max_old = -INFINITY;
    float32_t l_i = 0;

    for (int j = 0; j < Tc; j++)
    {

        ComputeType *__restrict__ Kj = &k[kv_offset + (j * Bc) * ld_qkv];
        ComputeType *__restrict__ Vj = &v[kv_offset + (j * Bc) * ld_qkv];
        int ele_x = j * Bc;
        int xr = ele_x + Bc;
        float32_t row_max_new = -INFINITY; // mij
        float32_t row_sum = 0;
        float32_t rowmax_diff_exp = 0; // Sij - mij
        //------------ Sij = Qi @ Kj^T
        if constexpr (!causal)
        {
            mul_A_BT<N_WAVES>(Qi, Kj, Si, ld_qkv, ld_qkv, Bc, Br, Bc, d, scale);
        }
        else
        {
            if (ele_y >= ele_x)
            {
                mul_A_BT<N_WAVES>(Qi, Kj, Si,ld_qkv, ld_qkv, Bc,  Br, Bc, d, scale);
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
            float val32 = row_max_new;
#pragma unroll 2
            for (int i = 0; i < Bc; i += 16)
            {
                bhalf16 val = HALF16(Si[(tx * Bc) + i]); 
#pragma unroll
                for (int j = 0; j < 16; j++)
                    val32 = max(val32, bf16_to_f32((val[j]))); // V_PK_MAX_F16
            }
            row_max_new = val32;

            row_max_new = max(row_max_old, row_max_new);
            rowmax_diff_exp = exp2f(row_max_old - row_max_new);
            row_max_old = row_max_new;

//--------------------Calc Pi = exp(Si - mi) and rowsum 
// #pragma unroll 4
            for (int i = 0; i < Bc; i += 16)
            {
                bhalf16 val = HALF16(Si[(tx * Bc) + i]);
                float_v16 val_f32;
#pragma unroll // Load fp16 into VGPRs and convert to FP32
                for (int j = 0; j < 16; j++) 
                    val_f32[j] = bf16_to_f32((val[j]));
// Si - mi
                val_f32 = val_f32 - row_max_new;
#pragma unroll // exp but using exp2 instead.
                for (int j = 0; j < 16; j++)
                    val_f32[j] = exp2f(val_f32[j]);

#pragma unroll // calc rowsum
                for (int j = 0; j < 16; j++)
                    row_sum += val_f32[j];

#pragma unroll // convert back to fp16
                for (int j = 0; j < 16; j++)
                    val[j] =  (f32_to_bf16(val_f32[j]));

               // write back
                HALF16(Si[(tx * Bc) + i]) = val;
            }
            l_i = rowmax_diff_exp * l_i + row_sum;

// --------------------- calc: Oi *= exp2f(row_max_old - row_max_new)
#pragma unroll 4
            for (int i = 0; i < d; i += 16)
            {
                bhalf16 val = HALF16(Oi[(tx * d) + i]);
                float_v16 val_f32;
                for(int j = 0; j < 16; j++)
                    val_f32[j] = bf16_to_f32((val[j]));

                val_f32 *= rowmax_diff_exp;

                for(int j = 0; j < 16; j++)
                    val[j] =  (f32_to_bf16(val_f32[j]));

                HALF16(Oi[(tx * d) + i]) = val;
            }
// --------------------- 
        }
        __syncthreads();

        if constexpr (!pad_mask)
            mul_add_A_B<N_WAVES>(Si, Vj, Oi,   Bc,ld_qkv,d,   Br, d, Bc);
        else
        {
            if (unlikely(xr > nkv))
            {
                mul_add_A_B_mask_k<N_WAVES>(Si, Vj, Oi,   Bc,ld_qkv,d,  Br, d, Bc, Bc - (xr - nkv));
            }
            else
            {
                mul_add_A_B<N_WAVES>(Si, Vj, Oi,   Bc,ld_qkv,d,    Br, d, Bc);
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
// #pragma unroll 4
            for (int i = 0; i < d; i += 16)
            {
                bhalf16 val = HALF16(Oi[(tx * d) + i]);
                float_v16 val_f32;
#pragma unroll
                for (int j = 0; j < 16; j++)
                    val_f32[j] = bf16_to_f32((val[j]));

                val_f32 = val_f32 / l_i;

#pragma unroll
                for (int j = 0; j < 16; j++)
                    val[j] =  (f32_to_bf16(val_f32[j]));
                    
                //HALF8(Oi[(tx * d) + i]) = val;
                HALF16((&(o[q_offset + (Tr_i * Br) * ld_qkv]))[tx * ld_qkv + i]) = val;
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


template <bool pad_mask, bool causal, int N_WAVES>
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
    const int d, const int h,
    const int Q_O_dO_stride_0, const int Q_O_dO_stride_1, const int Q_O_dO_stride_2,
    const int kvDkv_stride_0, const int kvdKv_stride_1, const int kvdKv_stride_2,
    const int L_stride_b, const int L_stride_h,
    const float32_t scale, const bool permute_NH
    )

{
    
    int q_offset = blockIdx.x * Q_O_dO_stride_0 + blockIdx.y * Q_O_dO_stride_1;
    int kv_offset = blockIdx.x * kvDkv_stride_0 + blockIdx.y * kvdKv_stride_1;

    int ld_qkv = Q_O_dO_stride_2; //d;
    if(permute_NH)
    {
        q_offset = blockIdx.x * Q_O_dO_stride_0 + blockIdx.y * Q_O_dO_stride_2;
        kv_offset = blockIdx.x * kvDkv_stride_0 + blockIdx.y * kvdKv_stride_2;
        ld_qkv = Q_O_dO_stride_1; //h * d;
    }

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

    ComputeType *__restrict__ Kj = &k[kv_offset + (Tc_j * Bc) * ld_qkv]; // [Bc x d]
    ComputeType *__restrict__ Vj = &v[kv_offset + (Tc_j * Bc) * ld_qkv]; // [Bc x d]

    ComputeType *__restrict__ dKj = &dK[kv_offset + (Tc_j * Bc) * ld_qkv]; // [Bc x d]
    ComputeType *__restrict__ dVj = &dV[kv_offset + (Tc_j * Bc) * ld_qkv]; // [Bc x d]

    for (int n_batch = 0; n_batch < ((nq + (blockDim.x - 1)) / blockDim.x); n_batch++)
    {
        int Di_off = n_batch * (blockDim.x) + tx;
        if (Di_off < nq)
        {
            float32_t val = 0;
#pragma unroll 2
            for (int i = 0; i < d; i+=16)
            {
                bhalf16 line_16 = HALF16(dO[q_offset + Di_off * ld_qkv + i]);
                bhalf16 line_16_2 = HALF16(O[q_offset + Di_off * ld_qkv + i]);
                float_v16 line_32;
                float_v16 line_32_2;
#pragma unroll 
                for(int j = 0; j < 16; j++)
                {
                    line_32[j] = bf16_to_f32(line_16[j]);
                    line_32_2[j] = bf16_to_f32(line_16_2[j]);
                }
                line_32 = line_32 * line_32_2;
                for(int j = 0; j < 16; j++)
                    val += line_32[j];
                // val += (dO[q_offset + Di_off * d + i] * O[q_offset + Di_off * d + i]);
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
        ComputeType *__restrict__ Qi = &q[q_offset + (Tr_i * Br) * ld_qkv];   // [Br x d]
        ComputeType *__restrict__ Oi = &O[q_offset + (Tr_i * Br) * ld_qkv];   // [Br x d]
        ComputeType *__restrict__ dOi = &dO[q_offset + (Tr_i * Br) * ld_qkv]; // [Br x d]
        ComputeType *__restrict__ dQi = &dQ[q_offset + (Tr_i * Br) * ld_qkv]; // [Br x d]
        float32_t *__restrict__ Li = &L[L_offset + Tr_i * Br];         // [Br]
        float32_t *__restrict__ Di_i = &Di[L_offset + Tr_i * Br];
        int ele_y = Tr_i * Br;
        int yb = ele_y + Br;
        int xr = ele_x + Bc;

        mul_A_BT<N_WAVES>(Qi, Kj, Si,  ld_qkv,ld_qkv,Bc,   Br, Bc, d, scale); // Qi[Br x d] Kj[Bc x d]
        __syncthreads();
        if constexpr (causal)
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

        if (tx < Br)
        {
// #pragma unroll 32
//             for (int i = 0; i < Bc; i++)
//             {
//                 Pi[tx * Bc + i] = exp2f(Si[tx * Bc + i] - Li[tx]); // Pi [Br x Bc]
//             }

            float32_t row_max = Li[tx];
#pragma unroll 2
            for (int i = 0; i < Bc; i += 16)
            {
                bhalf16 val = HALF16(Si[(tx * Bc) + i]);
                float_v16 val_f32;
#pragma unroll // Load fp16 into VGPRs and convert to FP32
                for (int j = 0; j < 16; j++)
                    val_f32[j] = bf16_to_f32(val[j]);
// Si - mi
                val_f32 = val_f32 - row_max;
#pragma unroll // exp but using exp2 instead.
                for (int j = 0; j < 16; j++)
                    val_f32[j] = exp2f(val_f32[j]);

#pragma unroll // convert back to fp16
                for (int j = 0; j < 16; j++)
                    val[j] =  f32_to_bf16(val_f32[j]);

               // write back
                HALF16(Pi[(tx * Bc) + i]) = val;
            }

        }
        __syncthreads();

        mul_add_AT_B<N_WAVES>(Pi, dOi, dVj,Bc,ld_qkv,ld_qkv,    Bc, d, Br, 1); // Pi[Br x Bc] @ dOi[Br x d]
        mul_A_BT<N_WAVES>(dOi, Vj, dPi,ld_qkv,ld_qkv,  Bc,     Br, Bc, d, 1);  // dPi:[Br x Bc]
        __syncthreads();
        if (tx < Br)
        {
#pragma unroll 2
            for(int i = 0; i < Bc; i+=16)
            {
                bhalf16 line_16 = HALF16(dPi[tx * Bc + i]);
                bhalf16 line_16_2 = HALF16(Pi[tx * Bc + i]);
                float_v16 line_32;
                float_v16 line_32_2;
#pragma unroll
                for(int j = 0; j < 16; j++)
                {
                    line_32[j] = bf16_to_f32(line_16[j]);
                    line_32_2[j] = bf16_to_f32(line_16_2[j]);
                }
                
                line_32 -= Di_i[tx];
                line_32_2 = line_32_2 * line_32;
                line_32_2 *= scale;
#pragma unroll
                for(int j = 0; j < 16; j++)
                    line_16[j] = f32_to_bf16(line_32_2[j]);
                HALF16(dSi[tx * Bc + i]) = line_16;

            }

// #pragma unroll 32
//             for (int i = 0; i < Bc; i++)
//             {
//                 dSi[tx * Bc + i] = scale * Pi[tx * Bc + i] * (dPi[tx * Bc + i] - Di_i[tx]);
//             }
        }
        __syncthreads();
        mul_add_A_B<N_WAVES>(dSi, Kj, dQi, Bc, ld_qkv,  ld_qkv,  Br, d, Bc);  // dSi[Br x Bc] @ Kj[Bc x d]
        mul_add_AT_B<N_WAVES>(dSi, Qi, dKj, Bc, ld_qkv,  ld_qkv,  Bc, d, Br, 0.69314718f); // dSi[Br x Bc] @ Qi[Br x d]
        __syncthreads();
    }
}

// =================================================================================

std::vector<torch::Tensor> forward_bf16(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    const int Br, const int Bc,
    const bool causal,
    const float scale, const bool permute_NH
)
{

    auto q_pad = q;
    auto k_pad = k;
    auto v_pad = v;

    const int b = q.size(0);
    const int h = permute_NH ? q.size(2):q.size(1);
    const int n = permute_NH ? q.size(1):q.size(2);
    const int d = q.size(3);

    const int n_kv = permute_NH ? k.size(1):k.size(2);
    

    int Nq_pad_sz = (Br - (n % Br)) % Br;
    int Nkv_pad_sz = (Bc - (n_kv % Bc)) % Bc;
    int d_pad_sz = ((ROCWMMA_K * 2) - (d % (ROCWMMA_K * 2))) % (ROCWMMA_K * 2);

    const bool pad_mask = Nq_pad_sz || Nkv_pad_sz;

    if (Nq_pad_sz || d_pad_sz)
    {
        if(permute_NH)
            q_pad = torch::nn::functional::pad(q_pad, torch::nn::functional::PadFuncOptions({0, d_pad_sz, 0,0, 0, Nq_pad_sz}));
        else
            q_pad = torch::nn::functional::pad(q_pad, torch::nn::functional::PadFuncOptions({0, d_pad_sz, 0, Nq_pad_sz})); 
    }
    // if (Nkv_pad_sz || d_pad_sz)
    if (d_pad_sz)
    {
        k_pad = torch::nn::functional::pad(k_pad, torch::nn::functional::PadFuncOptions({0, d_pad_sz, 0, 0}));
        v_pad = torch::nn::functional::pad(v_pad, torch::nn::functional::PadFuncOptions({0, d_pad_sz, 0, 0}));
    }
    if (q_pad.stride(-1) != 1)
        q_pad = q_pad.contiguous();

    if (k_pad.stride(-1) != 1)
        k_pad = k_pad.contiguous();

    if (v_pad.stride(-1) != 1)
        v_pad = v_pad.contiguous();

    const int Tr = ceil((float)n / Br);
    const int Tc = ceil((float)n_kv / Bc);

    // auto opt = torch::TensorOptions().dtype(TORCH_DTYPE).device(torch::kCUDA);
    auto O = torch::zeros_like(q_pad);

    auto opt2 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto L = torch::zeros({b, h, n + Nq_pad_sz}, opt2);

    int N_WAVES = 16;
    // if(d + d_pad_sz == 128)
    //     N_WAVES = 32;

    auto blockDim = dim3(WAVE_SIZE * N_WAVES);
    int nblk = b * h * Tr;
    int trPad = 96 - (nblk % 96); // TODO: 96 CU only for gfx1100

    auto gridDim = dim3(b, h, Tr + 0);

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
        d + d_pad_sz, h,                              \
        q_pad.stride(0),q_pad.stride(1),q_pad.stride(2),\
        k_pad.stride(0),k_pad.stride(1),k_pad.stride(2),\
        L.stride(0), L.stride(1),                     \
        scale * 1.442695f, permute_NH

    cudaError_t err = cudaGetLastError();

    // if(N_WAVES == 32)
    // {
    //     constexpr int NW = 32;
    //     if (!pad_mask && !causal)
    //         fwd_kernel<false, false,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
    //     else if (pad_mask && causal)
    //         fwd_kernel<true, true,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
    //     else if (!pad_mask && causal)
    //         fwd_kernel<false, true,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
    //     else if (pad_mask && !causal)
    //         fwd_kernel<true, false,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
    // }else 
    if(N_WAVES == 16)
    {
        constexpr int NW = 16;
        if (!pad_mask && !causal)
            fwd_kernel<false, false,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
        else if (pad_mask && causal)
            fwd_kernel<true, true,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
        else if (!pad_mask && causal)
            fwd_kernel<false, true,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
        else if (pad_mask && !causal)
            fwd_kernel<true, false,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
    }


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


    auto O_fwd = permute_NH ? 
                    O.index({"...",
                          torch::indexing::Slice(torch::indexing::None, n),
                          torch::indexing::Slice(torch::indexing::None, torch::indexing::None),
                          torch::indexing::Slice(torch::indexing::None, d)})
                    : 
                    O.index({"...",
                          torch::indexing::Slice(torch::indexing::None, n),
                          torch::indexing::Slice(torch::indexing::None, d)});

    return {O_fwd, q_pad, k_pad, v_pad, O, L};
}

std::vector<torch::Tensor> backward_bf16(
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
    const float scale, const bool permute_NH)
{

    const int b = Q.size(0);
    const int h = permute_NH ? Q.size(2):Q.size(1);
    const int n = permute_NH ? Q.size(1):Q.size(2);
    const int d = Q.size(3);
    const int n_kv = permute_NH ? K.size(1):K.size(2);

    //    const int dO_Npad_sz = Q.size(2) - dO.size(2);
    const int dO_Dpad_sz = Q.size(3) - dO.size(3);

    int Nq_pad_sz = (Br - (n % Br)) % Br;
    int Nkv_pad_sz = (Bc - (n_kv % Bc)) % Bc;
    const bool pad_mask = (Nkv_pad_sz || Nq_pad_sz || (n_kv != act_nkv));

    if(permute_NH)
    {
        dO = torch::nn::functional::pad(dO, torch::nn::functional::PadFuncOptions({0, dO_Dpad_sz,0, 0, 0, Nq_pad_sz}));
        Q = torch::nn::functional::pad(Q, torch::nn::functional::PadFuncOptions({0, 0,0, 0, 0, Nq_pad_sz}));
        O = torch::nn::functional::pad(O, torch::nn::functional::PadFuncOptions({0, 0,0, 0, 0, Nq_pad_sz}));
        L = torch::nn::functional::pad(L, torch::nn::functional::PadFuncOptions({0, Nq_pad_sz}));
        K = torch::nn::functional::pad(K, torch::nn::functional::PadFuncOptions({0, 0,0, 0, 0, Nkv_pad_sz}));
        V = torch::nn::functional::pad(V, torch::nn::functional::PadFuncOptions({0, 0,0, 0, 0, Nkv_pad_sz}));
    }else{
        dO = torch::nn::functional::pad(dO, torch::nn::functional::PadFuncOptions({0, dO_Dpad_sz, 0, Nq_pad_sz}));
        Q = torch::nn::functional::pad(Q, torch::nn::functional::PadFuncOptions({0, 0, 0, Nq_pad_sz}));
        O = torch::nn::functional::pad(O, torch::nn::functional::PadFuncOptions({0, 0, 0, Nq_pad_sz}));
        L = torch::nn::functional::pad(L, torch::nn::functional::PadFuncOptions({0, Nq_pad_sz}));
        K = torch::nn::functional::pad(K, torch::nn::functional::PadFuncOptions({0, 0, 0, Nkv_pad_sz}));
        V = torch::nn::functional::pad(V, torch::nn::functional::PadFuncOptions({0, 0, 0, Nkv_pad_sz}));

    }

    // Q = Q.contiguous();
    // K = K.contiguous();
    // V = V.contiguous();
    // dO = dO.contiguous();
    // O = O.contiguous();
    // L = L.contiguous();

    const int Tr = ceil((float)act_n / Br);
    const int Tc = ceil((float)act_nkv / Bc);

    auto opt = torch::TensorOptions().dtype(TORCH_DTYPE).device(torch::kCUDA);

    auto dQ = torch::zeros_like(Q, opt);
    auto dK = torch::zeros_like(K, opt);
    auto dV = torch::zeros_like(V, opt);

    auto Di = torch::zeros_like(L);

    constexpr int NW = 32;

    int nblk = b * h * Tc;
    int tcPad = 96 - (nblk % 96); // TODO: 96 CU only for gfx1100
    auto gridDim = dim3(b, h, Tc + 0);
    auto blockDim = dim3(WAVE_SIZE * NW);

    const int sram_sz =
        2 * Br * Bc * sizeof(ComputeType) // (Si,Pi,dSi), dPi
        // + Bc * d * sizeof(ComputeType)    // Kj
        ;

    cudaError_t err = cudaGetLastError();

    #define bwd_parm \
        (ComputeType *)Q.data_ptr<AT_PTR_TYPE>(),  \
        (ComputeType *)K.data_ptr<AT_PTR_TYPE>(),  \
        (ComputeType *)V.data_ptr<AT_PTR_TYPE>(),  \
        (ComputeType *)O.data_ptr<AT_PTR_TYPE>(),  \
        (ComputeType *)dO.data_ptr<AT_PTR_TYPE>(), \
        (ComputeType *)dQ.data_ptr<AT_PTR_TYPE>(), \
        (ComputeType *)dK.data_ptr<AT_PTR_TYPE>(), \
        (ComputeType *)dV.data_ptr<AT_PTR_TYPE>(), \
        (float *)Di.data_ptr<float>(), \
        (float *)L.data_ptr<float>(),  \
        Tr, Tc, \
        Br, Bc, \
        act_n, act_nkv, \
        d, h, \
        Q.stride(0), Q.stride(1), Q.stride(2), \
        K.stride(0), K.stride(1), K.stride(2), \
        L.stride(0), L.stride(1), \
        scale * 1.442695f, permute_NH

 
    if (!pad_mask && !causal)
        bwd_kernel<false, false,NW><<<gridDim, blockDim, sram_sz>>>(bwd_parm);
    else if (pad_mask && causal)
        bwd_kernel<true, true,NW><<<gridDim, blockDim, sram_sz>>>(bwd_parm);
    else if (!pad_mask && causal)
        bwd_kernel<false, true,NW><<<gridDim, blockDim, sram_sz>>>(bwd_parm);
    else if (pad_mask && !causal)
        bwd_kernel<true, false,NW><<<gridDim, blockDim, sram_sz>>>(bwd_parm);

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
    if(permute_NH)
    {
        dQ = dQ.index({"...",
                       torch::indexing::Slice(torch::indexing::None, act_n),
                       torch::indexing::Slice(torch::indexing::None, torch::indexing::None),
                       torch::indexing::Slice(torch::indexing::None, act_d)});
        dK = dK.index({"...",
                       torch::indexing::Slice(torch::indexing::None, act_nkv),
                       torch::indexing::Slice(torch::indexing::None, torch::indexing::None),
                       torch::indexing::Slice(torch::indexing::None, act_d)});
        dV = dV.index({"...",
                       torch::indexing::Slice(torch::indexing::None, act_nkv),
                       torch::indexing::Slice(torch::indexing::None, torch::indexing::None),
                       torch::indexing::Slice(torch::indexing::None, act_d)});
    }else
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
