#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <chrono>

#define ITER 1000

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// CUDA kernel to multiply each element by 0.26
__global__ void multiplyByFactor(float* data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] *= 0.251f;
  }
}

void runWithoutCudaGraph(float** buff, cudaStream_t* s, ncclComm_t* comms, int nDev, int size) {
  auto start = std::chrono::high_resolution_clock::now();

  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  for (int iter = 0; iter < ITER; ++iter) {
    // Multiply each element in buff by 0.26
    for (int i = 0; i < nDev; ++i) {
      CUDACHECK(cudaSetDevice(i));
      multiplyByFactor<<<blocksPerGrid, threadsPerBlock>>>(buff[i], size);
    }

    // calling NCCL communication API. Group API is required when using
    // multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i)
      NCCLCHECK(ncclAllReduce((const void*)buff[i], (void*)buff[i], size, ncclFloat, ncclSum,
          comms[i], s[i]));
    NCCLCHECK(ncclGroupEnd());

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaStreamSynchronize(s[i]));
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  printf("Time without CUDA graph: %f seconds\n", elapsed.count());
}

void runWithCudaGraph(float** buff, cudaStream_t* s, ncclComm_t* comms, int nDev, int size) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  cudaGraph_t graph;
  cudaGraphExec_t graphExec;

  cudaStream_t captureStream;
  cudaEvent_t  captureStreamForkEvent;

  cudaEvent_t  privateStreamSyncEvent[nDev];

  CUDACHECK(cudaEventCreate(&captureStreamForkEvent));

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaEventCreate(&privateStreamSyncEvent[i]));
  }

  CUDACHECK(cudaStreamCreate(&captureStream));
  CUDACHECK(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal));
  
  // fork from main
  CUDACHECK(cudaEventRecord(captureStreamForkEvent, captureStream));
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamWaitEvent(s[i], captureStreamForkEvent));
  }

  for (int iter = 0; iter < ITER; ++iter) {
    // Multiply each element in buff by a factor
    for (int i = 0; i < nDev; ++i) {
      CUDACHECK(cudaSetDevice(i));
      multiplyByFactor<<<blocksPerGrid, threadsPerBlock, 0, s[i]>>>(buff[i], size);
    }

    // calling NCCL communication API. Group API is required when using
    // multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i) {
      NCCLCHECK(ncclAllReduce((const void*)buff[i], (void*)buff[i], size, ncclFloat, ncclSum,
          comms[i], s[i]));
    }
    NCCLCHECK(ncclGroupEnd());
  }

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    // put sync event in private stream
    CUDACHECK(cudaEventRecord(privateStreamSyncEvent[i], s[i]));
    // make main stream wait for sync event
    CUDACHECK(cudaStreamWaitEvent(captureStream, privateStreamSyncEvent[i]));
  }
  
  CUDACHECK(cudaStreamEndCapture(captureStream, &graph));
  CUDACHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  // Time measurement with CUDA graph
  auto start = std::chrono::high_resolution_clock::now();

  CUDACHECK(cudaGraphLaunch(graphExec, captureStream));
  CUDACHECK(cudaStreamSynchronize(captureStream));

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  printf("Time with CUDA graph: %f seconds\n", elapsed.count());

  // clean up
  CUDACHECK(cudaGraphExecDestroy(graphExec));
  CUDACHECK(cudaGraphDestroy(graph));
  CUDACHECK(cudaEventDestroy(captureStreamForkEvent));
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaEventDestroy(privateStreamSyncEvent[i]));
  }

  CUDACHECK(cudaStreamDestroy(captureStream));
}

void printBufferValues(float** buff, float* hostBuff, int nDev, int size, const char* label) {
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMemcpy(hostBuff, buff[i], size * sizeof(float), cudaMemcpyDeviceToHost));
    printf("%s - Buff of device %d:\n", label, i);
    for (int j = 0; j < 10; ++j) { // Print only the first 10 elements for brevity
      printf("%f ", hostBuff[j]);
    }
    printf("\n");
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
  ncclComm_t comms[4];

  // managing 4 devices
  int nDev = 4;
  int size = bs * 4096;
  int devs[4] = { 0, 1, 2, 3 };

  // allocating and initializing device buffers
  float** buff = (float**)malloc(nDev * sizeof(float*));
  float* hostBuff = (float*)malloc(size * sizeof(float));
  float* printBuff = (float*)malloc(size * sizeof(float));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  // Initialize hostBuff with 1.0f
  for (int i = 0; i < size; ++i) {
    hostBuff[i] = 1.0f;
  }

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc((void**)&buff[i], size * sizeof(float)));
    CUDACHECK(cudaMemcpy(buff[i], hostBuff, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaStreamCreate(s+i));
  }

  // initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  // Run without CUDA graph
  runWithoutCudaGraph(buff, s, comms, nDev, size);
  // printBufferValues(buff, printBuff, nDev, size, "Without CUDA graph");

  // Reset buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMemcpy(buff[i], hostBuff, size * sizeof(float), cudaMemcpyHostToDevice));
  }

  // Run with CUDA graph
  runWithCudaGraph(buff, s, comms, nDev, size);
  // printBufferValues(buff, printBuff, nDev, size, "With CUDA graph");

  // Free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(buff[i]));
  }

  free(buff);
  free(hostBuff);
  free(printBuff);
  free(s);

  // Finalizing NCCL
  for (int i = 0; i < nDev; ++i) {
    ncclCommDestroy(comms[i]);
  }

  printf("Success \n");
  return 0;
}
