/*
    Question: 2
    Peforms a multiplication of a matrix B and vector C to give a matrix A. The input matrix B is assumed
    to be a square matrix
*/
__global__ void matrixVectorMul(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int width){
    int colThread = blockDim.x * blockIdx.x + threadIdx.x;
    int rowThread = blockDim.y * blockIdx.y + threadIdx.y;

    if(colThread >= width && rowThread >= width){
        return;
    }

    auto idx = [&width](int y, int x){ return y * width + x; };
    float temp = 0.0f;
    for (int i = 0; i < width ; ++i){
        temp += B[idx(rowThread, i)] * C[i];
    }
    A[rowThread, colThread] = temp;
}