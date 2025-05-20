#define MASK_RADIUS 2
#define MASK_DIM ((MASK_RADIUS)*2 + 1)

void convolution_gpu(float mask[][MASK_DIM], float* input, float* output, unsigned int width, unsigned int height);