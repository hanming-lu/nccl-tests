#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

// Kernel function
__global__ void kernel1(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 1024 * 1024) { // Ensuring we don't go out of bounds
        data[idx] = idx;
    }
}

__global__ void kernel2(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 1024 * 1024) { // Ensuring we don't go out of bounds
        data[idx] *= 2;
    }
}

// Function to measure execution time without CUDA Graphs
void withoutCudaGraphs(int *d_data, int N) {
    cudaEvent_t estart, estop;
    float milliseconds = 0;
    checkCuda(cudaEventCreate(&estart), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&estop), "cudaEventCreate stop");
    checkCuda(cudaEventRecord(estart), "cudaEventRecord start");
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i) {
        // Launch kernels
        kernel1<<<N/256, 256>>>(d_data);
        kernel2<<<N/256, 256>>>(d_data);
    }

    checkCuda(cudaEventSynchronize(estop), "cudaEventSynchronize stop");
    checkCuda(cudaEventRecord(estop), "cudaEventRecord stop");
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    checkCuda(cudaEventElapsedTime(&milliseconds, estart, estop), "cudaEventElapsedTime");

    std::cout << "Without CUDA Graphs: " << diff.count() << " s; cuda event: " << milliseconds << " ms\n";
}

// Function to measure execution time with CUDA Graphs
void withCudaGraphs(int *d_data, int N) {
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaStream_t stream;
    
    cudaEvent_t estart, estop;
    float milliseconds = 0;
    checkCuda(cudaEventCreate(&estart), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&estop), "cudaEventCreate stop");

    cudaStreamCreate(&stream);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    for (int i = 0; i < 1000; ++i) {
        // Launch kernels
        kernel1<<<N/256, 256, 0, stream>>>(d_data);
        kernel2<<<N/256, 256, 0, stream>>>(d_data);
    }

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    checkCuda(cudaEventRecord(estart), "cudaEventRecord start");
    auto start = std::chrono::high_resolution_clock::now();

    cudaGraphLaunch(instance, stream);
    checkCuda(cudaEventRecord(estop), "cudaEventRecord stop");
    cudaStreamSynchronize(stream);    

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    checkCuda(cudaEventElapsedTime(&milliseconds, estart, estop), "cudaEventElapsedTime");

    std::cout << "With CUDA Graphs: " << diff.count() << " s; cuda event: " << milliseconds << " ms\n";

    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);
    cudaStreamDestroy(stream);
}

int main() {
    const int N = 1024 * 1024;
    int *d_data;

    cudaMalloc(&d_data, N * sizeof(int));

    // Measure execution time without CUDA Graphs
    withoutCudaGraphs(d_data, N);

    // Measure execution time with CUDA Graphs
    withCudaGraphs(d_data, N);

    cudaFree(d_data);
    return 0;
}
