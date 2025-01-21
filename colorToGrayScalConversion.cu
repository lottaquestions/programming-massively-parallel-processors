__global__ void colorToGrayScaleConversion(unsigned char *Pout, unsigned char *Pin, int width, int height){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    constexpr int CHANNELS = 3;

    if (row >= height && col >= width){
        return;
    }

    // Get 1D offset of the grayscale image
    int grayOffset = row * width + col;
    // RGB image will have CHANNEL times more columns than the grayscale image
    int rgbOffset = grayOffset * CHANNELS;

    unsigned char red   = Pin[rgbOffset];
    unsigned char green = Pin[rgbOffset + 1];
    unsigned char blue  = Pin[rgbOffset + 2];

    // Perform rescaling using the formula: L = 0.21 * red + 0.72 * green + 0.07 * blue
    Pout[grayOffset] = unsigned char(0.21f * red + 0.72f * green + 0.07f * blue);
}