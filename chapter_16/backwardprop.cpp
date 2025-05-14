void convLayer_backward_x_grad(int M, int C, int H_in, int W_in, int K, float* dE_dY, float* W, float* dE_dX) {
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;
    for (int c = 0; c < C; c++)
        for(int h = 0; h < H_in; h++)
            for (int w = 0; w < W_in; w++)
            dE_dX[c, h, w] = 0;

    for (int m = 0; m < M ; m++)
        for(int h = 0; h < H_out - 1; h ++)
          for (int w = 0; w < W_out -1 ; w++)
              for (int c = 0; c< C; c++)
                  for (int p = 0; p < K ; p++)
                      for (int q = 0; q < K; q++)
                          if (h - p >= 0 && w - q >= 0 && h- p < H_out && w - q < W_out)
                              dE_dX[c, h, w] += dE_dY[m, h -p, w -q] * W[m, c, K - p, K -q] ;
        
}


void convLayer_backward_w_grad(int M, int C, int H, int W, int K, float* dE_dY, float* X , float* dE_dW){
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    for (int m = 0; m < M ; m++) 
        for (int c = 0; c < C; c++)
            for (int p = 0; p < K ; p++)
                for (int q = 0; q < K ; q++)
                    dE_dW[m,c,p,q] = 0;

    for (int m = 0; m < M ; m++)
        for(int h = 0; h < H_out; h++)
            for(int w = 0; w < W_out; w++)
                for (int c = 0; c< C; c++)
                    for (int p = 0; p < K ; p++)
                        for (int q = 0; q < K; q++)
                            dE_dW [m, c, p, q] += X[c, h + p, w + q] * dE_dY[m, c, h, w];
}