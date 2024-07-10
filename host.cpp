#include <torch/extension.h>
 
std::vector<torch::Tensor> forward(
    torch::Tensor Q, 
    torch::Tensor K, 
    torch::Tensor V, 
    const int Br,  
    const int Bc,
    const bool causal
    );

std::vector<torch::Tensor> backward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor L,
    const int Br, 
    const int Bc,
    const bool causal
    );

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
    m.def("backward", torch::wrap_pybind_function(backward), "backward");
}
