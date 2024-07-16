#include <iostream>
#include <cuda_runtime.h>

// Kernel definition
__global__ void MyKernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        data[idx] = idx * 1.0f; // Simple operation for demonstration
    }
}

int main() {
    // Set device 0 as the current device
    cudaSetDevice(0);
    
    // Allocate memory on device 0
    float* p0;
    size_t size = 1024 * sizeof(float);
    cudaMalloc(&p0, size);

    // Launch kernel on device 0
    // MyKernel<<<1000, 128>>>(p0);
    // cudaDeviceSynchronize(); // Ensure kernel execution is complete

    // Set device 1 as the current device
    cudaSetDevice(1);
    
    // Enable peer-to-peer access with device 0
    cudaDeviceEnablePeerAccess(0, 0);

    // Launch kernel on device 1 (accessing memory on device 0)
    MyKernel<<<1000, 128>>>(p0);
    cudaDeviceSynchronize(); // Ensure kernel execution is complete

    // Copy data back to host
    float* h_data = new float[1024];
    cudaSetDevice(0); // Set device 0 as current to copy data
    cudaMemcpy(h_data, p0, size, cudaMemcpyDeviceToHost);

    // Print the values in p0
    std::cout << "Values in p0:" << std::endl;
    for (int i = 0; i < 1024; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(p0);
    delete[] h_data;

    std::cout << "Completed successfully." << std::endl;
    return 0;
}
