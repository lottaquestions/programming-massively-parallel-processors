using uchar = unsigned char;

template<int BLUR_SIZE>
__global__ void imageBlur(uchar *in, uchar* out, int width, int height){
    int col = blockDim.x * blockIdx.x + theadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= height || col >= width){
        return;
    }
    auto idx = [&width](int y, int x){return y * width + x;};
    int pixelVal = 0;
    int pixelCount = 0;
    for(int rowOffset = -BLUR_SIZE ; rowOffset < BLUR_SIZE + 1; ++rowOffset){
        for(int colOffset = -BLUR_SIZE ; colOffset < BLUR_SIZE + 1; ++colOffset){
            int curRow = row + rowOffset;
            int curCol = col + colOffset;
            if(curRow >= 0 && curRow < height && curCol >= 0 && curCol < width){
                pixelVal += in[idx(row + rowOffset, col + colOffset)];
                ++pixelCount;
            }
            
        }
    }
    out[idx(row , col)] = uchar(pixelVal / pixelCount);
}