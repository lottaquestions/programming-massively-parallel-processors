
#define BLOCK_DIM 8
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    // i is in z direction
    // j is in y direction
    // k is in x direction
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1; // Minus 1 because we start one element earlier for the halo
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    // constants for the stencil
    const float C0 = 0.85f; // randomly chosen values for demonstration
    const float C1 = 0.15f; // such that corresponding input is heavily weighted

    // Shared mem is size of input tile
    __shared__ float input_s [IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N){
        input_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
    }
    __syncthreads();


    // We opt not to calculate the values at the boundaries to ensure we
    // we do not have to check for boundary conditions
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1){
        // For each block we are only calculating values for the output i.e.
        // values less than the halo
        if (threadIdx.x >= 1 && threadIdx < blockDim.x - 1 &&
            threadIdx.y >= 1 && threadIdx.y < blockDim.y - 1 &&
            threadIdx.z >= 1 && threadIdx.z < blockDim.z - 1)
        {
            out[i * N * N + j * N + k] = C0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
                                     C1 * (in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
                                           in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
                                           in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
                                           in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
                                           in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
                                           in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x]);
        }
        
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
    dim3 numThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    // We need sufficient number of blocks to calculate all the output tiles.
    // Since output tiles are smaller than input tiles, they will naturally yield
    // a larger number of tiles and hence MORE blocks for processing
    dim3 numBlocks(cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM));
    stencil_kernel<<<numBlocks, numThreadsPerBlock>>>(in_d, out_d, N);
    cudaDeviceSynchronize();

    // Copy output from GPU
    cudaMemcpy(out, out_d, N * N * N * sizeof(N));
    cudaDeviceSynchronize();
}