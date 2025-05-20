
#define BLOCK_DIM 8

__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    // i is in z direction
    // j is in y direction
    // k is in x direction
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    // constants for the stencil
    const float C0 = 0.85f; // randomly chosen values for demonstration
    const float C1 = 0.15f; // such that corresponding input is heavily weighted

    // We opt not to calculate the values at the boundaries to ensure we
    // we do not have to check for boundary conditions
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1){
        out[i * N * N + j * N + k] = C0 * in[i * N * N + j * N + k] +
                                     C1 * (in[i * N * N + j * N + k - 1] +
                                           in[i * N * N + j * N + k + 1] +
                                           in[i * N * N + (j + 1) * N + k] +
                                           in[i * N * N + (j - 1) * N + k] +
                                           in[(i + 1) * N * N + j * N + k] +
                                           in[(i - 1) * N * N + j * N + k]);
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
    dim3 numThreadsPerBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 numBlocks(cdiv(N, BLOCK_DIM), cdiv(N, BLOCK_DIM), cdiv(N, BLOCK_DIM));
    stencil_kernel<<<numBlocks, numThreadsPerBlock>>>(in_d, out_d, N);
    cudaDeviceSynchronize();

    // Copy output from GPU
    cudaMemcpy(out, out_d, N * N * N * sizeof(N));
    cudaDeviceSynchronize();
}