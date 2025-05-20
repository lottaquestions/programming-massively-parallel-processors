#include "cuda_runtime.h"
#include "common.h"

#define OUT_TILE_DIM 32

// Constant memory has a max size of 64K.
// Even though constant memory is at the same level as global memory, each 
// SM has a constant cache (separate from normal L1 cache), which caches this
// constant memory. The constant cache in the SM has the below characteristics
//    1. Constant data: easier to build an effient cache i.e.
//       - no need to support tracking changes and writing back.
//       - no need to support cache coherence for multicore access
//    2. Small size of constant memory means we minimize evictions in the constant
//       cache of each SM, and hence these SM constant caches have a low miss rate
__constant__ float mask_c[MASK_DIM][MASK_DIM];

__global__ void convolution_kernel(float *input, float *output, unsigned int width, unsigned int height){
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    auto idxCalc = [&width](int y, int x) { return y * width + x };
    if ( outRow < height && outCol < width) {
        float sum = 0.0f;
        for (int maskRow = 0; maskRow < MASK_DIM ; ++maskRow){
            for (int maskCol = 0; maskCol < MASK_DIM; ++maskCol){
                int rowIdx = outRow + maskRow - MASK_RADIUS;
                int colIdx = outCol + maskCol - MASK_RADIUS;
                if(rowIdx > 0 && rowIdx < height && colIdx > 0 && colIdx < width) {
                    sum += input[idxCalc(rowIdx, colIdx)] * mask_c[maskRow][maskCol];
                }
            }
        }
        output[idxCalc(outRow, outCol)] = sum;
    }
}

void convolution_gpu(float mask[][MASK_DIM], float* input, float* output, unsigned int width, unsigned int height){

    // Allocate GPU memory
    float *input_d, output_d;
    cudaMalloc((void**) &input_d, width * height * sizeof(float));
    cudaMalloc((void**) &output_d, width * height * sizeof(float));
    cudaDeviceSynchronize();

    // Copy data to GPU
    cudaMemcpy(input_d, input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Copy mask to constant memory
    cudaMemcpyToSymbol(mask_c, mask, MASK_DIM * MASK_DIM * sizeof(float));
    cudaDeviceSynchronize();

    // Call Kernel
    dim3 numThreadsPerBlock(OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM - 1)/OUT_TILE_DIM, (height + OUT_TILE_DIM -1) / OUT_TILE_DIM);
    convolution_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, width, height);
    cudaDeviceSynchronize();

    // Copy data from GPU
    cudaMemcpy(output, output_d, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Free GPU memory
    cudaFree(input_d);
    cudaFree(output_d);
}
