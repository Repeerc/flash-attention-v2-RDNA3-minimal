#include <torch/extension.h>
 
torch::Tensor forward(
    torch::Tensor A, 
    torch::Tensor B, 
    torch::Tensor C, 
    int m, int n, int k
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}
