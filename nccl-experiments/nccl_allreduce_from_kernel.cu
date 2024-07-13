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

__global__ void foo_kernel(float *data, int n, ncclComm_t comm, cudaStream_t stream) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 1.0f;
    }

    // Call ncclAllReduce inside the kernel
    __syncthreads();
    if (threadIdx.x == 0) {
        ncclAllReduce(data, data, n, ncclFloat, ncclSum, comm, stream);
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

    // Allocate device memory and copy data to GPUs
    float *d_data[nGPUs];
    cudaStream_t streams[nGPUs];
    for (int i = 0; i < nGPUs; ++i) {
        cudaSetDevice(i);
        checkCudaStatus(cudaMalloc(&d_data[i], N * sizeof(float)), "cudaMalloc");
        checkCudaStatus(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
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

    // Launch the kernel
    for (int i = 0; i < nGPUs; ++i) {
        cudaSetDevice(i);
        foo_kernel<<<(N + 255) / 256, 256, 0, streams[i]>>>(d_data[i], N, comms[i], streams[i]);
        checkCudaStatus(cudaGetLastError(), "Kernel launch");
    }

    // Synchronize and copy the results back to host
    for (int i = 0; i < nGPUs; ++i) {
        cudaSetDevice(i);
        checkCudaStatus(cudaStreamSynchronize(streams[i]), "cudaStreamSynchronize");
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
        checkCudaStatus(cudaStreamDestroy(streams[i]), "cudaStreamDestroy");
    }
    free(h_data);

    return EXIT_SUCCESS;
}


// #include <iostream>
// #include <nccl.h>
// #include <cuda_runtime.h>

// #define N 1024  // Size of the array

// void checkNcclStatus(ncclResult_t status, const char *msg) {
//     if (status != ncclSuccess) {
//         std::cerr << "NCCL error: " << msg << " - " << ncclGetErrorString(status) << std::endl;
//         exit(EXIT_FAILURE);
//     }
// }

// void checkCudaStatus(cudaError_t status, const char *msg) {
//     if (status != cudaSuccess) {
//         std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(status) << std::endl;
//         exit(EXIT_FAILURE);
//     }
// }

// __global__ void foo_kernel(float *data, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         data[idx] = 1.0f;
//     }
// }

// int main() {
//     int nGPUs;
//     cudaGetDeviceCount(&nGPUs);

//     if (nGPUs < 2) {
//         std::cerr << "This example requires at least 2 GPUs" << std::endl;
//         return EXIT_FAILURE;
//     }

//     // Allocate host memory
//     float *h_data = (float*)malloc(N * sizeof(float));
//     // for (int i = 0; i < N; ++i) {
//     //     h_data[i] = 1.0f;  // Initialize the array with 1.0
//     // }

//     // Allocate device memory and copy data to GPUs
//     float *d_data[nGPUs];
//     for (int i = 0; i < nGPUs; ++i) {
//         cudaSetDevice(i);
//         checkCudaStatus(cudaMalloc(&d_data[i], N * sizeof(float)), "cudaMalloc");
//         cudaError_t err = cudaSuccess;
//         foo_kernel<<<(N + 255) / 256, 256>>>(d_data[i], N);
//         err = cudaGetLastError();
//         if (err != cudaSuccess) {
//             fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
//             return -1;
//           }
//         // checkCudaStatus(cudaMemcpy(d_data[i], h_data, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
//     }

//     // Initialize NCCL
//     ncclComm_t comms[nGPUs];
//     ncclUniqueId id;
//     ncclGetUniqueId(&id);
//     checkNcclStatus(ncclGroupStart(), "ncclGroupStart");
//     for (int i = 0; i < nGPUs; ++i) {
//         cudaSetDevice(i);
//         checkNcclStatus(ncclCommInitRank(&comms[i], nGPUs, id, i), "ncclCommInitRank");
//     }
//     checkNcclStatus(ncclGroupEnd(), "ncclGroupEnd");

//     // Perform NCCL AllReduce
//     checkNcclStatus(ncclGroupStart(), "ncclGroupStart");
//     for (int i = 0; i < nGPUs; ++i) {
//         cudaSetDevice(i);
//         checkNcclStatus(ncclAllReduce(d_data[i], d_data[i], N, ncclFloat, ncclSum, comms[i], cudaStreamDefault), "ncclAllReduce");
//     }
//     checkNcclStatus(ncclGroupEnd(), "ncclGroupEnd");

//     // Copy the results back to host
//     for (int i = 0; i < nGPUs; ++i) {
//         cudaSetDevice(i);
//         checkCudaStatus(cudaMemcpy(h_data, d_data[i], N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
//         // Verify the results
//         for (int j = 0; j < N; ++j) {
//             if (h_data[j] != (float)nGPUs) {
//                 std::cerr << "Data verification failed at index " << j << ": " << h_data[j] << " != " << nGPUs << std::endl;
//                 return EXIT_FAILURE;
//             }
//         }
//     }

//     std::cout << "NCCL AllReduce test passed." << std::endl;

//     // Cleanup
//     for (int i = 0; i < nGPUs; ++i) {
//         cudaSetDevice(i);
//         checkCudaStatus(cudaFree(d_data[i]), "cudaFree");
//         ncclCommDestroy(comms[i]);
//     }
//     free(h_data);

//     return EXIT_SUCCESS;
// }
