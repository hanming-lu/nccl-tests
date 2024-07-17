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
__global__ void get_sum(float *embedding, float* block_sum, int num_elements, Coordinator* coordinator) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    for (int i = 0; i < ITER; ++i) {
        float thread_val = 0.0001 * static_cast<float>(mype + 1);
        // Each gou calculate a sum of all block thread values
        int elements_per_pe = num_elements / npes;
        int elements_per_block = elements_per_pe / NUM_BLOCKS;
        for (int i = threadIdx.x; i < elements_per_block; i += NUM_THREADS_PER_BLOCK) {
            atomicAdd(&block_sum[0], thread_val);
        }
        coordinator->arrive_and_wait();

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            int peer1 = (mype + 1) % npes;
            int peer2 = (mype + 2) % npes;
            int peer3 = (mype + 3) % npes;
            nvshmem_float_p(&embedding[mype], block_sum[0], peer1);
            nvshmem_float_p(&embedding[mype], block_sum[0], peer2);
            nvshmem_float_p(&embedding[mype], block_sum[0], peer3);
            embedding[mype] = block_sum[0];
            nvshmem_barrier_all();
        }
        float final_sum = 0.0;
        for (int i = 0; i < npes; i++) {
            final_sum += embedding[i];
        }
    }
}

int main(int argc, char* argv[]) {
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

    int num_elements = 4096 * bs;

    float *embedding = (float *)nvshmem_calloc(npes, sizeof(float));
    if (embedding == NULL) {
        fprintf(stderr, "nvshmem_calloc for all gather failed\n");
        exit(EXIT_FAILURE);
    }

    float* block_sum;
    cudaMalloc((void **)&block_sum, sizeof(float));
    if (block_sum == NULL) {
        fprintf(stderr, "nvshmem_calloc for all gather failed\n");
        exit(EXIT_FAILURE);
    }

    Coordinator *d_coordinator;
    cudaMalloc((void**)&d_coordinator, sizeof(Coordinator));
    init_coordinator<<<1, 1>>>(d_coordinator, NUM_BLOCKS * NUM_THREADS_PER_BLOCK);

    void *args_sum[] = { &embedding, &block_sum, &num_elements,&d_coordinator };
    auto start = std::chrono::high_resolution_clock::now();
    nvshmemx_collective_launch((void *)get_sum, NUM_BLOCKS, NUM_THREADS_PER_BLOCK, args_sum, 0, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after all gather launch: %s\n", cudaGetErrorString(err));
    }
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("PE %d - Time taken for customized All Gather customized (using nvshmem) %d iterations: %f seconds\n", mype, ITER, elapsed.count());

    

    cudaFree(d_coordinator);
    nvshmem_free(embedding);
    cudaFree(block_sum);
    nvshmem_finalize();

    return 0;
}
