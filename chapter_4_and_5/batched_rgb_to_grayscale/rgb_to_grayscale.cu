#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1)/b; }

// Computes the luminance-based grayscale conversion (BT.601 standard). It assumes:
//
//     Input in is a flat contiguous RGB tensor with layout: [N, 3, H, W].
//     There is a batch of N images and each image in the tensor is 3 *H * W, so R, G, B 
//     channels are laid out contiguously.

__global__ void rgb_to_grayscale_kernel(unsigned char* __restrict__ out, unsigned char* __restrict__ in, int N, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = N * W *H; // N is the number of images in the batch
    if (idx >= num_pixels) return;

    int batch_idx = idx / (H * W); // Will be used in calculations of indices of both input and output,
                                   // hence why we use H * W rather than 3* H * W. We adjust for the
                                   // missing 3 , when calculating the offset for the input
    int pixel_idx = idx % (H * W);
    int offset = batch_idx * 3 * H * W;

    int r = in[offset + pixel_idx];
    int g = in[offset + H * W + pixel_idx];
    int b = in[offset + 2 * H * W + pixel_idx];

    // Note that input has an offset of batch_idx * 3 * H * W, whereas the output
    // has an offset of batch_idx * H * W
    out[batch_idx * H * W + pixel_idx] =
        static_cast<unsigned char>(0.2989f * r + 0.5870f * g + 0.1140f * b); // f postfix is needed to ensure float32 calculations instead
                                                                             //  of float64 which are slower on consumer GPUs (none-datacenter)
}

torch::Tensor rgb_to_grayscale_out(torch::Tensor output, const torch::Tensor& input_raw) {
    CHECK_INPUT(input_raw);
    auto input = input_raw.dim() == 3 ? input_raw.unsqueeze(0) : input_raw;
    int N = input.size(0); // Number of images in the batch i.e tensor is [N, 3, H, W]
    int H = input.size(2);
    int W = input.size(3);
    TORCH_CHECK((N == output.size(0)) && (H == output.size(1)) && (W == output.size(2)) && (output.device() == input.device()) 
    && (output.scalar_type() == input.scalar_type()));
    
    int threads = 256;
    rgb_to_grayscale_kernel<<<cdiv(N * W * H, threads), threads>>>(
        output.data_ptr<unsigned char>(), input.data_ptr<unsigned char>(), N, H, W
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor rgb_to_grayscale(const torch::Tensor& input_raw) {
    CHECK_INPUT(input_raw);
    auto input = input_raw.dim() == 3 ? input_raw.unsqueeze(0) : input_raw;
    int N = input.size(0);
    int H = input.size(2);
    int W = input.size(3);
    auto output = torch::empty({N, H,W}, input.options());
    rgb_to_grayscale_out(output, input);
    return output;
}