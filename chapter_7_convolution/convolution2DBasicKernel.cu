#include "cuda_runtime.h"

__global__ void convolution2DBasicKernel(float *N, float *F, float *P, int r, int width, int height){
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0.0f;

    for (int fRow = 0; fRow < 2*r + 1; ++fRow){
        for (int fCol = 0; fCol < 2*r + 1; ++fCol){
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width){
                Pvalue += F[fRow][fCol] * N[inRow * width + inCol];
            }
            
        }
    }
    P[outRow * width + outCol] = Pvalue;
}