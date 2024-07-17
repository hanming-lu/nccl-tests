#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <chrono>

#define ITER 1000
typedef cuda::barrier<cuda::thread_scope_device> Coordinator;

// For H100 - NUM_BLOCKS = 16; For V100 - NUM_BLOCKS = 32.
constexpr int32_t NUM_BLOCKS = 32;
constexpr int32_t NUM_THREADS_PER_BLOCK = 384;

__global__ void init_coordinator(Coordinator* d_coordinator, int n) {
    new (d_coordinator) Coordinator(n);
}

__global__ void customized_all_gather(float *embedding, float *block_global_data, int num_elements, Coordinator* coordinator) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    for (int iter = 0; iter < ITER; ++iter) {
        int val = mype + 1;

        int elements_per_pe = num_elements / npes;
        int elements_per_block = elements_per_pe / NUM_BLOCKS;
        if (threadIdx.x < elements_per_block) {
            int idx_with_offset = blockIdx.x * elements_per_block + threadIdx.x;
            block_global_data[idx_with_offset] = val;
        }
        // sync across blocks
        coordinator->arrive_and_wait();
        
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            nvshmem_float_fcollect(NVSHMEMX_TEAM_NODE, embedding, block_global_data, elements_per_pe);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <bs>\n", argv[0]);
        return EXIT_FAILURE;
    }
    int bs = atoi(argv[1]);
    if (bs <= 0) {
        printf("Error: bs must be a positive integer\n");
        return EXIT_FAILURE;
    }
    nvshmem_init();
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    cudaStream_t stream;

    cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE));
    cudaStreamCreate(&stream);

    if (mype == 0) {
        // print NUM_BLOCKS and NUM_THREADS_PER_BLOCK and bs
        printf("NUM_BLOCKS: %d, NUM_THREADS_PER_BLOCK: %d, NUM_BATCH_SIZE: %d\n", NUM_BLOCKS, NUM_THREADS_PER_BLOCK, bs);
    }

    int num_elements = 8192 * bs;
    int elements_per_pe = num_elements / npes;

    float *embedding_all_gather = (float *)nvshmem_calloc(num_elements, sizeof(float));
    if (embedding_all_gather == NULL) {
        fprintf(stderr, "nvshmem_calloc for all gather failed\n");
        exit(EXIT_FAILURE);
    }

    float *block_global_data;
    cudaMalloc((void **)&block_global_data, elements_per_pe * sizeof(float));
    Coordinator *d_coordinator;
    cudaMalloc((void**)&d_coordinator, sizeof(Coordinator));
    init_coordinator<<<1, 1>>>(d_coordinator, NUM_BLOCKS * NUM_THREADS_PER_BLOCK);

    void *args_gather[] = { &embedding_all_gather, &block_global_data, &num_elements,&d_coordinator };
    auto start = std::chrono::high_resolution_clock::now();
    nvshmemx_collective_launch((void *)customized_all_gather, NUM_BLOCKS, NUM_THREADS_PER_BLOCK, args_gather, 0, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after all gather launch: %s\n", cudaGetErrorString(err));
    }
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("PE %d - Time taken for All Gather API (using nvshmem) %d iterations: %f seconds\n", mype, ITER, elapsed.count());

    

    float *host_embedding_all_gather = (float *)malloc(num_elements * sizeof(float));
    cudaMemcpy(host_embedding_all_gather, embedding_all_gather, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // if (mype == 0) {
    //     printf("Results -- All Gather Embedding from GPU: %d\n", mype);
    //     // Sample prints
    //     for (int i = 0; i < num_elements; i += 256) {
    //         printf("%f ", host_embedding_all_gather[i]);
    //     }
    //     printf("\n");
    // }

    free(host_embedding_all_gather);
    cudaFree(block_global_data);
    cudaFree(d_coordinator);
    nvshmem_free(embedding_all_gather);
    nvshmem_barrier_all(); // Ensure all PEs are synchronized before finalizing
    nvshmem_finalize();
    return 0;
}
