#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

using uchar = unsigned char;


__global__ void imageBlurKernel(uchar* out, uchar *in, int width, int height, int radius){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int channel = threadIdx.z;

    int baseOffset = channel * height * width;

    if (row >= height || col >= width){
        return;
    }
    auto idx = [&width, &baseOffset](int y, int x){return baseOffset + (y * width + x);};
    int pixelVal = 0;
    int pixelCount = 0;
    for(int rowOffset = -radius ; rowOffset < radius + 1; ++rowOffset){
        for(int colOffset = -radius ; colOffset < radius + 1; ++colOffset){
            int curRow = row + rowOffset;
            int curCol = col + colOffset;
            if(curRow >= 0 && curRow < height && curCol >= 0 && curCol < width){
                pixelVal += in[idx(row + rowOffset, col + colOffset)];
                ++pixelCount;
            }
            
        }
    }
    out[idx(row , col)] = uchar(pixelVal / pixelCount);
}

// Helper function for ceiling unsigned integer division
inline unsigned int cdiv(const unsigned int a, const unsigned int b){
    return (a + b - 1) / b;
}

torch::Tensor imageBlur(torch::Tensor image, int radius){
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);
    assert(radius > 0);

    const auto channels = image.size(0);
    const auto height = image.size(1);
    const auto width = image.size(2);

    auto result = torch::empty_like(image);
    dim3 threadsPerBlock(16, 16, channels);
    dim3 numberOfBlocks(
        cdiv(width, threadsPerBlock.x),
        cdiv(height, threadsPerBlock.y)
    );

    imageBlurKernel<<<numberOfBlocks, threadsPerBlock, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<uchar>(),
        image.data_ptr<uchar>(),
        width,
        height,
        radius
    );

    // Check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}

