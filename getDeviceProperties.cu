#include <iostream>
// cuda includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// nvcc -I/home/lottaquestions/nvidia-installers/cuda-samples/Common -G -o getDeviceProperties.bin getDeviceProperties.cu

int main(){
    using std::cout;
    using std::endl;
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    cout << "CUDA devices found: " << devCount << endl;
    cudaDeviceProp devProp;
    for (int i = 0; i < devCount; ++i){
        cudaGetDeviceProperties(&devProp, i);
        cout << "Device " << i << " properties" << endl;
        cout << "Max threads per block: " << devProp.maxThreadsPerBlock << endl;
        cout << "Number of SMs on the device: " << devProp.multiProcessorCount << endl;
        cout << "Max registers per SM: " << devProp.regsPerBlock << endl;
        cout << "Max threads per block in x dimension: " << devProp.maxThreadsDim[0] << endl;
        cout << "Max threads per block in y dimension: " << devProp.maxThreadsDim[1] << endl;
        cout << "Max threads per block in z dimension: " << devProp.maxThreadsDim[2] << endl;
        cout << "Max blocks per grid in x dimension: " << devProp.maxGridSize[0] << endl;
        cout << "Max blocks per grid in y dimension: " << devProp.maxGridSize[1] << endl;
        cout << "Max blocks per grid in z dimension: " << devProp.maxGridSize[2] << endl;
    }
}