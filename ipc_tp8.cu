#include <iostream>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <errno.h>

#define ITER 1000
#define ThreadsPerBlock 384
#define BlocksPerGrid 128
typedef cuda::barrier<cuda::thread_scope_system> Coordinator;

// Kernel to initialize data
__global__ void SetupKernel(float* data, int offset, int elementsPerGPU) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elementsPerGPU; idx += ThreadsPerBlock * BlocksPerGrid) {
        data[idx] = idx + offset;
    }
}

__global__ void init_coordinator(Coordinator* d_coordinator, int n) {
    new (d_coordinator) Coordinator(n);
}

// Kernel to perform multiplication and allgather operation
__global__ void MultiplyAndAllGatherKernel(float* myDest, float* dest1, float* dest2, float* dest3, 
    float* dest4, float* dest5, float* dest6, float* dest7, 
    float* src, int gpu_idx, int elementsPerGPU, float factor, Coordinator *d_coordinator) {
    for (int iter = 0; iter < ITER; ++iter) {
        // allgather   
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elementsPerGPU; idx += ThreadsPerBlock * BlocksPerGrid) {
            dest1[gpu_idx*elementsPerGPU+idx] = src[idx];
            dest2[gpu_idx*elementsPerGPU+idx] = src[idx];
            dest3[gpu_idx*elementsPerGPU+idx] = src[idx];
            dest4[gpu_idx*elementsPerGPU+idx] = src[idx];
            dest5[gpu_idx*elementsPerGPU+idx] = src[idx];
            dest6[gpu_idx*elementsPerGPU+idx] = src[idx];
            dest7[gpu_idx*elementsPerGPU+idx] = src[idx];
            myDest[gpu_idx*elementsPerGPU+idx] = src[idx];
        }

        // sync
        __syncthreads();
        if (threadIdx.x == 0) {
            d_coordinator->arrive_and_wait();
        }
        __syncthreads();
    }
}

void write_memhandle_to_shared_mem(const char* name, const cudaIpcMemHandle_t* handle) {
    int shm_fd = shm_open(name, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        printf("Error opening shared memory for writing %s: %s\n", name,  strerror(errno));
    }
    ftruncate(shm_fd, sizeof(cudaIpcMemHandle_t));
    void* ptr = mmap(0, sizeof(cudaIpcMemHandle_t), PROT_WRITE, MAP_SHARED, shm_fd, 0);
    memcpy(ptr, handle, sizeof(cudaIpcMemHandle_t));
    munmap(ptr, sizeof(cudaIpcMemHandle_t));
    close(shm_fd);
}

void read_memhandle_from_shared_mem(const char* name, cudaIpcMemHandle_t* handle) {
    int shm_fd = shm_open(name, O_RDONLY, 0666);
    if (shm_fd == -1) {
        printf("Error opening shared memory for reading %s: %s\n", name, strerror(errno));
    }
    void* ptr = mmap(0, sizeof(cudaIpcMemHandle_t), PROT_READ, MAP_SHARED, shm_fd, 0);
    memcpy(handle, ptr, sizeof(cudaIpcMemHandle_t));
    munmap(ptr, sizeof(cudaIpcMemHandle_t));
    close(shm_fd);
}

void cleanup_shared_mem(const char* name) {
    shm_unlink(name);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <bs> <rank> <primary>\n", argv[0]);
        return EXIT_FAILURE;
    }
    int bs = atoi(argv[1]);
    int rank = atoi(argv[2]);
    int primary = atoi(argv[3]);
    if (bs <= 0 || rank < 0 || rank >= 8) {
        printf("Error: bs must be a positive integer and rank must be between 0 and 7\n");
        return EXIT_FAILURE;
    }

    const int totalElements = 8192 * bs;
    const int numGPUs = 8;
    const int elementsPerGPU = totalElements / numGPUs;
    size_t sizePerGPU = elementsPerGPU * sizeof(float);
    float* d_data;
    float* d_allData;
    float factor = 1.0f; // Factor to multiply each component

    // printf("Process %d: Setting device to GPU %d\n", rank, rank);
    cudaSetDevice(rank);

    // Allocate memory on each GPU and initialize data
    // printf("Process %d: Allocating memory\n", rank);
    cudaMalloc(&d_data, sizePerGPU);
    cudaMalloc(&d_allData, numGPUs * sizePerGPU); // Buffer to hold gathered data
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // printf("Process %d: Initializing data\n", rank);
    SetupKernel<<<BlocksPerGrid, ThreadsPerBlock, 0, stream>>>(d_data, rank * elementsPerGPU, elementsPerGPU);

    cudaStreamSynchronize(stream);

    // Create and exchange IPC memory handles
    // printf("Process %d: Creating IPC memory handles\n", rank);
    cudaIpcMemHandle_t memHandle, allDataHandle, coordHandle;
    cudaIpcGetMemHandle(&memHandle, d_data);
    cudaIpcGetMemHandle(&allDataHandle, d_allData);

    char memHandleName[256], allDataHandleName[256], coordHandleName[256];
    sprintf(memHandleName, "/cuda_memhandle_%d", rank);
    sprintf(allDataHandleName, "/cuda_alldatahandle_%d", rank);
    sprintf(coordHandleName, "/cuda_coordhandle");

    // printf("Process %d: Writing memory handles to shared memory\n", rank);
    write_memhandle_to_shared_mem(memHandleName, &memHandle);
    write_memhandle_to_shared_mem(allDataHandleName, &allDataHandle);

    // Create or open the coordinator handle
    Coordinator* d_coordinator;
    if (primary == 1) {
        // Primary process creates the coordinator
        // printf("Process %d: Creating coordinator\n", rank);
        cudaMalloc((void**)&d_coordinator, sizeof(Coordinator));
        init_coordinator<<<1, 1>>>(d_coordinator, BlocksPerGrid * numGPUs);
        cudaIpcGetMemHandle(&coordHandle, d_coordinator);
        write_memhandle_to_shared_mem(coordHandleName, &coordHandle);
        sleep(5); // This is needed as other processes has this sleep, otherwise primary process will start earlier than other processes, which causes processes time measurement much longer than other processes
    } else {
        // Secondary processes open the coordinator handle
        sleep(5); // wait for primary to write the handle
        // printf("Process %d: Opening coordinator handle\n", rank);
        read_memhandle_from_shared_mem(coordHandleName, &coordHandle);
        cudaIpcOpenMemHandle((void**)&d_coordinator, coordHandle, cudaIpcMemLazyEnablePeerAccess);
    }
    sleep(5);

    // Synchronize processes
    // printf("Process %d: Synchronizing processes\n", rank);
    
    // Open IPC memory handles
    // printf("Process %d: Opening IPC memory handles\n", rank);
    float* d_dataPtrs[numGPUs];
    float* d_allDataPtrs[numGPUs];
    cudaIpcMemHandle_t otherMemHandle, otherAllDataHandle;
    for (int i = 0; i < numGPUs; ++i) {
        if (i != rank) {
            sprintf(memHandleName, "/cuda_memhandle_%d", i);
            sprintf(allDataHandleName, "/cuda_alldatahandle_%d", i);
            // printf("Process %d: Reading memory handle from GPU %d\n", rank, i);
            read_memhandle_from_shared_mem(memHandleName, &otherMemHandle);
            read_memhandle_from_shared_mem(allDataHandleName, &otherAllDataHandle);
            // printf("Process %d: Opening memory handle from GPU %d\n", rank, i);
            cudaIpcOpenMemHandle((void**)&d_dataPtrs[i], otherMemHandle, cudaIpcMemLazyEnablePeerAccess);
            cudaIpcOpenMemHandle((void**)&d_allDataPtrs[i], otherAllDataHandle, cudaIpcMemLazyEnablePeerAccess);
        } else {
            d_dataPtrs[i] = d_data;
            d_allDataPtrs[i] = d_allData;
        }
    }

    // Perform multiplication and allgather operation
    // printf("Process %d: Starting MultiplyAndAllGatherKernel\n", rank);
    auto start = std::chrono::high_resolution_clock::now();
    MultiplyAndAllGatherKernel<<<BlocksPerGrid, ThreadsPerBlock, 0, stream>>>(
        d_allData, d_allDataPtrs[(rank+1)%numGPUs], d_allDataPtrs[(rank+2)%numGPUs], d_allDataPtrs[(rank+3)%numGPUs], 
        d_allDataPtrs[(rank+4)%numGPUs], d_allDataPtrs[(rank+5)%numGPUs], d_allDataPtrs[(rank+6)%numGPUs], d_allDataPtrs[(rank+7)%numGPUs],
        d_data, rank, elementsPerGPU, factor, d_coordinator);

    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Time spent for process %d: %f seconds\n", rank, elapsed.count());

    // // Copy data back to host and print results
    // if (rank == 0) {
    //     float* h_data = new float[numGPUs * elementsPerGPU];
    //     for (int i = 0; i < numGPUs; ++i) {
    //         cudaMemcpy(h_data, d_allDataPtrs[i], numGPUs * sizePerGPU, cudaMemcpyDeviceToHost);
    //         std::cout << "Values in gathered data from GPU " << i << ":" << std::endl;
    //         for (int j = 0; j < numGPUs * elementsPerGPU; ++j) {
    //             std::cout << h_data[j] << " ";
    //             if ((j + 1) % elementsPerGPU == 0) std::cout << std::endl;
    //         }
    //     }
    //     delete[] h_data;
    // }

    // Cleanup IPC handles
    // printf("Process %d: Cleaning up IPC handles\n", rank);
    for (int i = 0; i < numGPUs; ++i) {
        if (i != rank) {
            cudaIpcCloseMemHandle(d_dataPtrs[i]);
            cudaIpcCloseMemHandle(d_allDataPtrs[i]);
        }
    }

    // Cleanup
    // printf("Process %d: Freeing allocated memory\n", rank);
    cudaFree(d_data);
    cudaFree(d_allData);
    if (primary == 1) {
        cudaFree(d_coordinator);
    }
    cudaStreamDestroy(stream);

    // Cleanup shared memory
    // printf("Process %d: Cleaning up shared memory\n", rank);
    sprintf(memHandleName, "/cuda_memhandle_%d", rank);
    sprintf(allDataHandleName, "/cuda_alldatahandle_%d", rank);
    cleanup_shared_mem(memHandleName);
    cleanup_shared_mem(allDataHandleName);

    std::cout << "Process " << rank << " completed successfully." << std::endl;
    return 0;
}

