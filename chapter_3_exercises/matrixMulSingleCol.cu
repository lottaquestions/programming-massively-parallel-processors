// Question 1b
// Each thread calculates the value of a whole output column instead of a single element.
__global__ void matrixMulSingleColmnPerThread(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int width){
    int rowIdx = (blockDim.y * blockIdx.y  + threadIdx.y) * width; // This line and the next if block ensure we only use rowIdx = 0 threads
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (rowIdx >= width && colIdx >= width){
        return;
    }
    auto idx = [&width](int y, int x){ return y * width + x ;};
    for(int newRowIdx = rowIdx; newRowIdx < rowIdx + width; ++newRowIdx){
        float temp;
        for(i = 0 ; i < width < i ++){
            temp += A[idx(newRowIdx, i)] * B[idx(i, colIdx)]
        }
        C[idx(newRowIdx, colIdx)] = temp;
    }
    
}