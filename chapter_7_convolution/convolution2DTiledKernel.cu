#include "cuda_runtime.h"

#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))

__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1]

// To use this kernel, launch it with a block size whose dimensions match
// the input tile, but with a block count that is calculated from the output tile dimension.
__global__ void convolutionTiled2DConstMemKernel(float *N, float *P, int width, int height){
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    // Load input tile
    __shared__ N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < height && col >= 0 && col < width){
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    // Turning off the threads at the edge of the block
    if(row >= 0 && row < height && col >= 0 && col < width){
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM){
            float Pvalue = 0.0f;
            for (int fRow = 0 ; fRow < 2 * FILTER_RADIUS + 1; ++fRow){
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; ++fCol){
                    Pvalue += F_c[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}