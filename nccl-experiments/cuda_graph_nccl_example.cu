#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <nccl.h>

#define CHECK_CUDA(call)                                                    \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error calling \"" #call "\", code is " << err << std::endl; \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    }

#define CHECK_NCCL(call)                                                    \
    {                                                                       \
        ncclResult_t err = call;                                            \
        if (err != ncclSuccess) {                                           \
            std::cerr << "NCCL error calling \"" #call "\", code is " << err << std::endl; \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    }

// Kernel function
__global__ void kernel1(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 1024 * 1024) { // Ensuring we don't go out of bounds
        data[idx] = idx;
    }
}

// Function to measure execution time without CUDA Graphs
void withoutCudaGraphs(int *d_data, int N, ncclComm_t comm, cudaStream_t stream) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i) {
        // Launch kernels
        kernel1<<<N/256, 256, 0, stream>>>(d_data);
        CHECK_NCCL(ncclAllReduce((const void*)d_data, (void*)d_data, N, ncclInt, ncclSum, comm, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    std::cout << "Without CUDA Graphs: " << diff.count() << " ms\n";
}

// Function to measure execution time with CUDA Graphs
void withCudaGraphs(int *d_data, int N, ncclComm_t comm, cudaStream_t stream) {
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    for (int i = 0; i < 1000; ++i) {
        // Launch kernels
        kernel1<<<N/256, 256, 0, stream>>>(d_data);
        CHECK_NCCL(ncclAllReduce((const void*)d_data, (void*)d_data, N, ncclInt, ncclSum, comm, stream));
    }

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    auto start = std::chrono::high_resolution_clock::now();

    cudaGraphLaunch(instance, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    std::cout << "With CUDA Graphs: " << diff.count() << " ms\n";

    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);
}

int main() {
    const int N = 1024 * 1024;
    int *d_data;
    ncclComm_t comm;
    int size = 1, rank = 0;
    cudaStream_t stream;
    ncclUniqueId id;

    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(int)));
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_NCCL(ncclGetUniqueId(&id));
    CHECK_NCCL(ncclCommInitRank(&comm, size, id, rank));

    // Measure execution time without CUDA Graphs
    withoutCudaGraphs(d_data, N, comm, stream);

    // Measure execution time with CUDA Graphs
    withCudaGraphs(d_data, N, comm, stream);

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_NCCL(ncclCommDestroy(comm));

    return 0;
}
