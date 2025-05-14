// Forward propagation path of a convolutional layer

// Input feature maps are stored in a 3D array X[C, H, W] where 
// C = number of input feature maps
// H = height of each input map image
// W = width of each input map image

// Output feature maps of a convolution layer are stored in a 3D array Y[M, H - K + 1, W - K + 1] where
// M = number of output feature maps
// K = height and width of each 2D filter

// Filter banks are stored in a 4D array W_f[M, C, K, K]. 
// There M x C filter banks where filter bank W_f[m, c, _, _] is used with input feature map
// X[c,_,_] to calculate output feature map Y[m,_,_]
// Each output feature map is the sum of the convolutions of all input feature maps
void convLayer_forward(int M, int C, int H, int W, int K, float *X, float *W_f, float *Y) {
    int H_out = H - K + 1;
    int W_out = W - K + 1; 

    for(int m = 0; m < M; m++)          // For each output feature map
        for (int h = 0; h < H_out; h++) // For each output element
            for(int w = 0; w < W_out ; w++) {
                Y[m, h, w] = 0;
                for (int c = 0; c < C; c++) // Sum over all input  feature maps
                    for (int p = 0; p < K; p++) // K x K filter
                        for (int q = 0; q < K; q++) {
                            Y[m, h, w] += X[c, h + p, w + q] * W_f[m, c, p , q];
                        }
            }
}

void subsamplingLayer_forward(int M, int H, int W, int K, float *Y, float *S, float *b, float(*sigmoid)(float)) {
    for (int m = 0 ; m < M; m++)        // For each output feature map
        for (int h = 0; h < H/K ; h++)  // For each output element, we assume 
            for (int w = 0; w < W/K; w++){ // H and W are multiples of K
                S[m, h, w] = 0;
                for (int p = 0; p < K; p++)  // Loop over K x K input samples
                    for (int q = 0; q < K; q++)
                        S[m, h, w] += Y[m, K*h + p, K *w + q] / (K*K);
                // Add bias and apply non-linear activation
                S[m, h, w] = sigmoid (S[m, h, w] + b[m]);
            }
}


void convLayer_batched(int N, int M, int C, int H, int W, int K, float *X, float* W_f, float* Y){
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    for (int  n = 0; n < N ; n++)           // For each sample in the minibatch
        for(int m = 0; m < M; m++)          // For each output feature map
            for (int h = 0; h < H_out; h++) // For each output element
                for(int w = 0; w < W_out ; w++) {
                    Y[m, h, w] = 0;
                    for (int c = 0; c < C; c++) // Sum over all input  feature maps (or channels)
                        for (int p = 0; p < K; p++) // K x K filter
                            for (int q = 0; q < K; q++) {
                                Y[m, h, w] += X[c, h + p, w + q] * W_f[m, c, p , q];
                            }
                }

}