#include <iostream>
#include <nccl.h>
#include <cuda_runtime.h>

#define N 1024  // Size of the array

void checkNcclStatus(ncclResult_t status, const char *msg) {
    if (status != ncclSuccess) {
        std::cerr << "NCCL error: " << msg << " - " << ncclGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCudaStatus(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int nGPUs;
    cudaGetDeviceCount(&nGPUs);

    if (nGPUs < 2) {
        std::cerr << "This example requires at least 2 GPUs" << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate host memory
    float *h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_data[i] = 1.0f;  // Initialize the array with 1.0
    }

    // Allocate device memory and copy data to GPUs
    float *d_data[nGPUs];
    for (int i = 0; i < nGPUs; ++i) {
        cudaSetDevice(i);
        checkCudaStatus(cudaMalloc(&d_data[i], N * sizeof(float)), "cudaMalloc");
        checkCudaStatus(cudaMemcpy(d_data[i], h_data, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
    }

    // Initialize NCCL
    ncclComm_t comms[nGPUs];
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    checkNcclStatus(ncclGroupStart(), "ncclGroupStart");
    for (int i = 0; i < nGPUs; ++i) {
        cudaSetDevice(i);
        checkNcclStatus(ncclCommInitRank(&comms[i], nGPUs, id, i), "ncclCommInitRank");
    }
    checkNcclStatus(ncclGroupEnd(), "ncclGroupEnd");

    // Perform NCCL AllReduce
    checkNcclStatus(ncclGroupStart(), "ncclGroupStart");
    for (int i = 0; i < nGPUs; ++i) {
        cudaSetDevice(i);
        checkNcclStatus(ncclAllReduce(d_data[i], d_data[i], N, ncclFloat, ncclSum, comms[i], cudaStreamDefault), "ncclAllReduce");
    }
    checkNcclStatus(ncclGroupEnd(), "ncclGroupEnd");

    // Copy the results back to host
    for (int i = 0; i < nGPUs; ++i) {
        cudaSetDevice(i);
        checkCudaStatus(cudaMemcpy(h_data, d_data[i], N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
        // Verify the results
        for (int j = 0; j < N; ++j) {
            if (h_data[j] != (float)nGPUs) {
                std::cerr << "Data verification failed at index " << j << ": " << h_data[j] << " != " << nGPUs << std::endl;
                return EXIT_FAILURE;
            }
        }
    }

    std::cout << "NCCL AllReduce test passed." << std::endl;

    // Cleanup
    for (int i = 0; i < nGPUs; ++i) {
        cudaSetDevice(i);
        checkCudaStatus(cudaFree(d_data[i]), "cudaFree");
        ncclCommDestroy(comms[i]);
    }
    free(h_data);

    return EXIT_SUCCESS;
}
