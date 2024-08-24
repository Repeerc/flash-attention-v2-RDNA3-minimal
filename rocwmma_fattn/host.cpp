#include <torch/extension.h>

#define fwd_parm                                       \
    torch::Tensor q, torch::Tensor k, torch::Tensor v, \
        const int Br, const int Bc,                    \
        const bool causal,                             \
        const float scale, const bool permute_NH

#define bwd_parm           \
    torch::Tensor Q,       \
        torch::Tensor K,   \
        torch::Tensor V,   \
        torch::Tensor O,   \
        torch::Tensor dO,  \
        torch::Tensor L,   \
        const int act_n,   \
        const int act_nkv, \
        const int act_d,   \
        const int Br,      \
        const int Bc,      \
        const bool causal, \
        const float scale, const bool permute_NH

std::vector<torch::Tensor> forward_bf16(fwd_parm);
std::vector<torch::Tensor> backward_bf16(bwd_parm);

std::vector<torch::Tensor> forward_fp16(fwd_parm);
std::vector<torch::Tensor> backward_fp16(bwd_parm);

std::vector<torch::Tensor> forward(fwd_parm)
{
    auto q_dtype = q.dtype();
    if (q_dtype == torch::kFloat16)
    {
        return forward_fp16(q, k, v, Br, Bc, causal, scale, permute_NH);
    }
    else if (q_dtype == torch::kBFloat16)
    {
        return forward_bf16(q, k, v, Br, Bc, causal, scale, permute_NH);
    }
    q = q.to(torch::kBFloat16);
    k = k.to(torch::kBFloat16);
    v = v.to(torch::kBFloat16);
    return forward_bf16(q, k, v, Br, Bc, causal, scale, permute_NH);
}

std::vector<torch::Tensor> backward(bwd_parm)
{
    auto dO_dtype = dO.dtype();
    if (dO_dtype == torch::kFloat16)
    {
        return backward_fp16(Q, K, V, O, dO, L, act_n, act_nkv, act_d, Br, Bc, causal, scale, permute_NH);
    }
    else
    {
        return backward_bf16(Q, K, V, O, dO, L, act_n, act_nkv, act_d, Br, Bc, causal, scale, permute_NH);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", torch::wrap_pybind_function(forward), "flash attention forward pass");
    m.def("backward", torch::wrap_pybind_function(backward), "flash attention backward pass");
}
