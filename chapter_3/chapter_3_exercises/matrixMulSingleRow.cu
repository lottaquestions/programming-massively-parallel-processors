// Question 1a
// Each thread calculates the value of a whole output row instead of a single element.
__global__ void matrixMulSingleRowPerThread(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int width)
{
    int rowThread = (blockDim.y * blockIdx.y + threadIdx.y);
    int colThread = (blockDim.x * blockIdx.x + threadIdx.x) * width; // This line and the next line will effectively disable threads
    // with colThread > 0 since the kernel does one whole row at a time per thread, instead of one matrix element per thread.

    if (rowThread >= width && colThread >= width) {
        return;
    }

    auto idx = [&width](int y, int x){ y *width + x; };
    float temp = 0.0f;
    for (int newColThread = colThread; newColThread < colThread + width; ++newColThread)
    {
        for (int i = 0; i < width; i++)
        {
            temp += A[idx(rowThread, i)] * B[idx(i, newColThread)];
        }
        C[idx(rowThread, newColThread)] = temp;
    }
}

