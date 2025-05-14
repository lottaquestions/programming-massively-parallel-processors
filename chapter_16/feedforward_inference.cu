#include "cuda_runtime.h"

#define TILE_WIDTH 16


__global__ void ConvLayerForward_Kernel(int C, int W_out, int H_out, int W_grid, int K, float* X, float* Weights, float* Y) {
    int m = blockIdx.x; // Output channel
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int n = blockIdx.z; // Batch sample index

    if (h >= H_out || w >= W_out) return;

    float acc = 0.0f;
    for (int c = 0; c < C; c++){          // Sum over all input channels
        for (int p = 0; p < K; p ++) {    // Loop over K x K filter
            for (int q = 0; q < K ; q++) {
                acc += X [n, c, h + p, w + q] * Weights [m, c, p, q]; // Fix indexing here
            }
        }
    }
    Y[n, m, h, w] = acc;
}

void kernelLauncher(int C, int N, int M, int W_out, int H_out, float* input, float *W, float *output){
    //int W_out = 1024;
    //int H_out = 768;
    auto W_grid = W_out / TILE_WIDTH; // Number of horizontal tiles per output map
    auto H_grid = H_out / TILE_WIDTH; // Number of vertical tiles per output map

    auto T = W_grid * H_grid;

    dim3 blockDim (TILE_WIDTH, TILE_WIDTH, 1);

    // x-component M : output feature maps covered by each block
    // y-component T : reflects the location of a block's output tile inside the output feature map
    // z-component N : samples in the minibatch
    dim3 gridDim  (M, T, N);
    ConvLayerForward_Kernel<<<gridDim, blockDim>>>(C, W_out, H_out, W_grid, H_grid,  K, input, W, output);
}
