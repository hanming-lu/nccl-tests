#include <iostream>
#include <cuda_runtime.h>
#include <cuda/barrier>

#define ITER 1000
#define ThreadsPerBlock 384
#define BlocksPerGrid 32
typedef cuda::barrier<cuda::thread_scope_system> Coordinator;


// Kernel to initialize data
__global__ void SetupKernel(float* data, int offset, int elementsPerGPU) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elementsPerGPU; idx += ThreadsPerBlock*BlocksPerGrid) {
        data[idx] = idx + offset;
    }
}
__global__ void init_coordinator(Coordinator* d_coordinator, int n) {
    new (d_coordinator) Coordinator(n);
}

// Kernel to perform multiplication and allgather operation
__global__ void MultiplyAndAllGatherKernel(float* myDest, float* dest1, float* dest2, float* dest3, float* src, int gpu_idx, int elementsPerGPU, float factor, Coordinator *d_coordinator) {
    for (int iter = 0; iter < 1000; ++iter) {
        // allgather   
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elementsPerGPU; idx += ThreadsPerBlock*BlocksPerGrid) {
            src[idx] *= factor;
            // myDest[gpu_idx*elementsPerGPU+idx] = src[idx]; // not needed
            dest1[gpu_idx*elementsPerGPU+idx] = src[idx];
            dest2[gpu_idx*elementsPerGPU+idx] = src[idx];
            dest3[gpu_idx*elementsPerGPU+idx] = src[idx];
        }

        // sync
        __syncthreads();
        if (threadIdx.x == 0) {
            d_coordinator->arrive_and_wait();
        }
        __syncthreads();

        // operate on data
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elementsPerGPU; idx += ThreadsPerBlock*BlocksPerGrid) {
            // uncomment after sync works -> should see non-zeros in the output now
            src[idx] *= 0.99f;
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("Usage: %s <bs>\n", argv[0]);
        return EXIT_FAILURE;
    }
    int bs = atoi(argv[1]);
    if (bs <= 0) {
        printf("Error: bs must be a positive integer\n");
        return EXIT_FAILURE;
    }
    const int totalElements = 8192 * bs;
    const int numGPUs = 4;
    const int elementsPerGPU = totalElements / numGPUs;
    size_t size = elementsPerGPU * sizeof(float);
    float* d_data[numGPUs];
    float* d_allData[numGPUs];
    float* h_data = new float[numGPUs * elementsPerGPU];
    float factor = 1.0f; // Factor to multiply each component
    Coordinator *d_coordinator;
    cudaMalloc((void**)&d_coordinator, sizeof(Coordinator));
    init_coordinator<<<1, 1>>>(d_coordinator, BlocksPerGrid * numGPUs);

    cudaStream_t streams[numGPUs];

    // Allocate memory on each GPU and initialize data
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        cudaMalloc(&d_data[i], size);
        cudaMalloc(&d_allData[i], numGPUs * size); // Buffer to hold gathered data
        cudaStreamCreate(&streams[i]);
        SetupKernel<<<BlocksPerGrid, ThreadsPerBlock, 0, streams[i]>>>(d_data[i], i * elementsPerGPU, elementsPerGPU);
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

    auto start = std::chrono::high_resolution_clock::now();
    // Perform multiplication and allgather operation
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        MultiplyAndAllGatherKernel<<<BlocksPerGrid, ThreadsPerBlock, 0, streams[i]>>>(d_allData[i], d_allData[(i+1)%numGPUs], d_allData[(i+2)%numGPUs], d_allData[(i+3)%numGPUs], d_data[i], i, elementsPerGPU, factor, d_coordinator);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Time spent: %f seconds\n", elapsed.count());
    
    // Synchronize and check for runtime errors
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "Kernel execution error on GPU " << i << ": " << cudaGetErrorString(err) << std::endl;
        }
    }

    // Copy data back to host and print
    // for (int i = 0; i < numGPUs; ++i) {
    //     cudaMemcpy(h_data, d_allData[i], numGPUs * size, cudaMemcpyDeviceToHost);
    //     if (i == 0) {
    //         std::cout << "Values in gathered data:" << std::endl;
    //         for (int i = 0; i < numGPUs * elementsPerGPU; ++i) {
    //             std::cout << h_data[i] << " ";
    //             if ((i + 1) % elementsPerGPU == 0) std::cout << std::endl;
    //         }
    //     }
    // }

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
