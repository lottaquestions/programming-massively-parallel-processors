#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

// In the coarsened version, I have increased the BLOCK_DIM from 8 to 32
#define BLOCK_DIM 32
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    // i is in z direction
    // j is in y direction
    // k is in x direction
    int iStart = blockIdx.z * OUT_TILE_DIM; 
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];

    if(iStart -1 > 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart -1)*N*N + j * N + k ];
    }

    if(iStart > 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart*N*N + j * N + k ];
    }
    __syncthreads();

    // constants for the stencil
    const float C0 = 0.85f; // randomly chosen values for demonstration
    const float C1 = 0.15f; // such that corresponding input is heavily weighted

    for(int i = iStart; i < iStart + OUT_TILE_DIM; ++i){
        if (i + 1 > 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();
        if(i > 1 && i < N-1 && j > 1 && j < N -1 && k > 1 && k < N -1 ) {
            if (threadIdx.y >= 1 && threadIdx.y < OUT_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < OUT_TILE_DIM - 1)
            {
                out[i * N * N + j * N + k] = C0 * inCurr_s[threadIdx.y][threadIdx.x] +
                                             C1 * (inCurr_s[threadIdx.y + 1][threadIdx.x] +
                                                   inCurr_s[threadIdx.y - 1][threadIdx.x] +
                                                   inCurr_s[threadIdx.y][threadIdx.x + 1] +
                                                   inCurr_s[threadIdx.y][threadIdx.x - 1] +
                                                   inPrev_s[threadIdx.y][threadIdx.x] +
                                                   inNext_s[threadIdx.y][threadIdx.x]);
            }
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}

void stencil_gpu(float* in, float* out, unsigned int N) {
    // Allocate GPU memory
    float *in_d, *out_d;
    cudaMalloc((void**)&in_d, N * N * N * sizeof(float));
    cudaMalloc((void**)&out_d, N * N * N *sizeof(float));
    cudaDeviceSynchronize();

    // Copy data to GPU
    cudaMemcpy(in_d, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Call kernel
    auto cdiv =  [](int a, int b) { return (a + b -1)/ b;};
    // In each block, we need sufficient threads to load each element of the input tile
    // For the coarsened version IN_TILE_DIM = 32 and 32*32 = 1024 which is the max
    // number of threads per block in my GPU. It means that there is no z-dimension
    // within a block, but I still have a z-dimension within the grid.
    dim3 numThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM, 1);
    dim3 numBlocks(cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM));
    stencil_kernel<<<numBlocks, numThreadsPerBlock>>>(in_d, out_d, N);
    cudaDeviceSynchronize();

    // Copy output from GPU
    cudaMemcpy(out, out_d, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

__host__ void stencil_cpu(const float* in, float* out, unsigned int N) {
    const float C0 = 0.85f;
    const float C1 = 0.15f;

    for (unsigned int z = 1; z < N - 1; ++z) {
        for (unsigned int y = 1; y < N - 1; ++y) {
            for (unsigned int x = 1; x < N - 1; ++x) {
                size_t idx = z * N * N + y * N + x;

                float center = in[idx];
                float top    = in[idx - N];
                float bottom = in[idx + N];
                float left   = in[idx - 1];
                float right  = in[idx + 1];
                float front  = in[idx - N * N];
                float back   = in[idx + N * N];

                out[idx] = C0 * center + C1 * (top + bottom + left + right + front + back);
            }
        }
    }
}


int main() {
    const unsigned int N = 64;  // 64x64x64 cube for testing (fits GPU memory)

    size_t total_elems = N * N * N;

    // Allocate host memory
    std::vector<float> input(total_elems);
    std::vector<float> output(total_elems, 0.0f);

    // Initialize input with some values
    std::srand(static_cast<unsigned>(std::time(0)));
    for (size_t i = 0; i < total_elems; ++i) {
        input[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    // Run GPU stencil computation
    stencil_gpu(input.data(), output.data(), N);

    std::vector<float> cpu_output(total_elems, 0.0f);
    stencil_cpu(input.data(), cpu_output.data(), N);

    // Compare CPU vs GPU values
    float max_diff = 0.0f;
    for (size_t i = 0; i < total_elems; ++i)
    {
        std::cout << "CPU: " << cpu_output[i] << ", GPU: " << output[i] << std::endl;
        float diff = std::abs(cpu_output[i] - output[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    std::cout << "Max absolute difference between CPU and GPU: " << max_diff << "\n";

    return 0;
}