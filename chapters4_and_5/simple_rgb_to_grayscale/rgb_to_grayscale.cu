#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1)/b; }

// Computes the luminance-based grayscale conversion (BT.601 standard). It assumes:
//
//     Input in is a flat contiguous RGB tensor with layout: [3, H, W].
//     n = H * W, so R, G, B channels are laid out contiguously.

__global__ void rgb_to_grayscale_kernel(unsigned char* __restrict__ out, unsigned char* __restrict__ in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n) return;

    out[i] = 0.2989f*in[i] + 0.5870f*in[i + n] + 0.1140f*in[i+2*n]; //f postfix is needed to ensure float32 calculations instead
                                                                    // of float64 which are slower on consumer GPUs (none-datacenter)
    
}

torch::Tensor rgb_to_grayscale_out(torch::Tensor output, const torch::Tensor& input) {
    CHECK_INPUT(input);
    int h = input.size(1);
    int w = input.size(2);
    TORCH_CHECK((h == output.size(0)) || (w == output.size(1)) || (output.device() == input.device()) 
    || (output.scalar_type() == input.scalar_type()));
    
    int threads = 256;
    rgb_to_grayscale_kernel<<<cdiv(w*h, threads), threads>>>(
        output.data_ptr<unsigned char>(), input.data_ptr<unsigned char>(), w*h
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor rgb_to_grayscale(const torch::Tensor& input) {
    CHECK_INPUT(input);
    int h = input.size(1);
    int w = input.size(2);
    auto output = torch::empty({h,w}, input.options());
    rgb_to_grayscale_out(output, input);
    return output;
}