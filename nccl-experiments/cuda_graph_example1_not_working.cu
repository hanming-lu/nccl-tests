#include <iostream>
#include <cuda_runtime.h>

// Define the vector size and number of repetitions
const int N = 1 << 20; // 1M elements
const int REPEATS = 1000; // Number of repetitions

// Kernel for vector addition
__global__ void vecAdd(float* A, float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel for vector addition multiple times
__global__ void vecAddMultipleTimes(float* A, float* B, float* C, int n, int repetitions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < repetitions; ++i) {
        if (idx < n) {
            C[idx] = A[idx] + B[idx];
        }
    }
}

// Kernel for a single vector addition operation
__global__ void vecAddSingleOperation(float* A, float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

int main() {
    // Allocate and initialize host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, N * sizeof(float)), "cudaMalloc A");
    checkCuda(cudaMalloc(&d_B, N * sizeof(float)), "cudaMalloc B");
    checkCuda(cudaMalloc(&d_C, N * sizeof(float)), "cudaMalloc C");

    // Copy data from host to device
    checkCuda(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCuda(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy B");

    // Define kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float milliseconds = 0;

    // Measure latency with CUDA Graphs
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
    for (int i = 0; i < REPEATS; ++i) {
        vecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);
    }
    checkCuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    checkCuda(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0), "cudaGraphInstantiate");

    checkCuda(cudaEventRecord(start), "cudaEventRecord start");
    checkCuda(cudaGraphLaunch(instance, stream), "cudaGraphLaunch");
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");

    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
    std::cout << "Time with 100 kernels, 1 operation each [CUDA Graphs]: " << milliseconds << " ms" << std::endl;
    float overheadWithGraphs = milliseconds / REPEATS;
    std::cout << "Overhead of each kernel: " << overheadWithGraphs << " ms" << std::endl;

    // Measure latency without CUDA Graphs
    checkCuda(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < REPEATS; ++i) {
        vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    }
    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");

    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
    std::cout << "Time with 100 kernels, 1 operation each [no CUDA Graphs]: " << milliseconds << " ms" << std::endl;
    float overheadWithoutGraphs = milliseconds / REPEATS;
    std::cout << "Overhead of each kernel: " << overheadWithoutGraphs << " ms" << std::endl;

    // Measure latency with single kernel doing one operation
    checkCuda(cudaEventRecord(start), "cudaEventRecord start");
    vecAddSingleOperation<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
    std::cout << "Time with 1 kernel, 1 operation each: " << milliseconds << " ms" << std::endl;

    // Measure latency with single kernel doing REPEATS operations
    checkCuda(cudaEventRecord(start), "cudaEventRecord start");
    vecAddMultipleTimes<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, REPEATS);
    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
    std::cout << "Time with 1 kernel, " << REPEATS << " operations each: " << milliseconds << " ms" << std::endl;   

    // Clean up
    checkCuda(cudaGraphDestroy(graph), "cudaGraphDestroy");
    checkCuda(cudaGraphExecDestroy(instance), "cudaGraphExecDestroy");
    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop");

    checkCuda(cudaFree(d_A), "cudaFree A");
    checkCuda(cudaFree(d_B), "cudaFree B");
    checkCuda(cudaFree(d_C), "cudaFree C");

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    std::cout << "Completed successfully!" << std::endl;
    return 0;
}
