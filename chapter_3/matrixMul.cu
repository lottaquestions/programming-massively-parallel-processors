__global__ void matrixMul(float *Output,float *M, float *N, int width){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= width || col >= width){
        return;
    }
    auto idx = [&width](int y, int x){y * width + x;}
    float temp = 0.0f;
    for (int i = 0 ; i < width; ++i ){
        temp += M[idx(row,i)] * N[idx(i,col)];
    }
    Output[idx(row, col)] = temp;
}