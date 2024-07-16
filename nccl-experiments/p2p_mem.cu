#include <iostream>
#include <cuda_runtime.h>

// Kernel to initialize data
__global__ void SetupKernel(float* data, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        data[idx] = idx + offset; // Simple operation for demonstration
    }
}

// Kernel to perform multiplication and allgather operation
__global__ void MultiplyAndAllGatherKernel(float* dest, float* src, int gpu_idx, int elementsPerGPU, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elementsPerGPU) {
        dest[gpu_idx*elementsPerGPU+idx] = src[idx] * factor;
        // dest[gpu_idx*elementsPerGPU+idx] = 1.0;
    }
}

int main() {
    const int numGPUs = 4;
    const int elementsPerGPU = 1024;
    size_t size = elementsPerGPU * sizeof(float);
    float* d_data[numGPUs];
    float* d_allData[numGPUs];
    float* h_data = new float[numGPUs * elementsPerGPU];
    float factor = 1.0f; // Factor to multiply each component

    cudaStream_t streams[numGPUs];

    int32_t threadsPerBlock = 128;
    int32_t blocksPerGrid = (elementsPerGPU + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory on each GPU and initialize data
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        cudaMalloc(&d_data[i], size);
        cudaMalloc(&d_allData[i], numGPUs * size); // Buffer to hold gathered data
        cudaStreamCreate(&streams[i]);
        SetupKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_data[i], i * elementsPerGPU);
    }
    cudaDeviceSynchronize();

    // Enable peer-to-peer access between all GPUs
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        for (int j = 0; j < numGPUs; ++j) {
            if (i != j) {
                cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }

    // Perform multiplication and allgather operation
    for (int i = 0; i < numGPUs; ++i) {
        for (int j = 0; j < numGPUs; ++j) {
            cudaSetDevice(i);
            MultiplyAndAllGatherKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_allData[j], d_data[i], i, elementsPerGPU, factor);
        }
    }
    cudaDeviceSynchronize();

    // Copy data back to host and print
    for (int i = 0; i < numGPUs; ++i) {
        cudaMemcpy(h_data, d_allData[i], numGPUs * size, cudaMemcpyDeviceToHost);
        if (i == 3) {
            std::cout << "Values in gathered data:" << std::endl;
            for (int i = 0; i < numGPUs * elementsPerGPU; ++i) {
                std::cout << h_data[i] << " ";
                if ((i + 1) % elementsPerGPU == 0) std::cout << std::endl;
            }
        }
    }

    // Clean up
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        cudaFree(d_data[i]);
        cudaFree(d_allData[i]);
    }
    delete[] h_data;

    std::cout << "Completed successfully." << std::endl;
    return 0;
}
