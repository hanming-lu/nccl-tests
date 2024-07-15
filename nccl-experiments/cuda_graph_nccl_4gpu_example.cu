#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <chrono>

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
__global__ void multiplyByPointTwoSix(float* data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] *= 0.26f;
  }
}

int main(int argc, char* argv[])
{
  ncclComm_t comms[4];

  // managing 4 devices
  int nDev = 4;
  int size = 32*1024*1024;
  int devs[4] = { 0, 1, 2, 3 };

  // allocating and initializing device buffers
  float** buff = (float**)malloc(nDev * sizeof(float*));
  float* hostBuff = (float*)malloc(size * sizeof(float));
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

  // Time measurement without CUDA graph
  auto start = std::chrono::high_resolution_clock::now();

  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  for (int iter = 0; iter < 100; ++iter) {
    // Multiply each element in buff by 0.26
    for (int i = 0; i < nDev; ++i) {
      CUDACHECK(cudaSetDevice(i));
      multiplyByPointTwoSix<<<blocksPerGrid, threadsPerBlock>>>(buff[i], size);
      CUDACHECK(cudaDeviceSynchronize());
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

  // Print elements of buff
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMemcpy(hostBuff, buff[i], size * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Buff of device %d:\n", i);
    for (int j = 0; j < 10; ++j) { // Print only the first 10 elements for brevity
      printf("%f ", hostBuff[j]);
    }
    printf("\n");
  }

  
  // Reset hostBuff with 1.0f
  for (int i = 0; i < size; ++i) {
    hostBuff[i] = 1.0f;
  }
  // Reset buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMemcpy(buff[i], hostBuff, size * sizeof(float), cudaMemcpyHostToDevice));
  }

  // CUDA graph setup
  cudaGraph_t* graphs = (cudaGraph_t*)malloc(nDev * sizeof(cudaGraph_t));
  cudaGraphExec_t* graphExecs = (cudaGraphExec_t*)malloc(nDev * sizeof(cudaGraphExec_t));

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamBeginCapture(s[i], cudaStreamCaptureModeGlobal));
    for (int iter = 0; iter < 100; ++iter) {
      multiplyByPointTwoSix<<<blocksPerGrid, threadsPerBlock, 0, s[i]>>>(buff[i], size);
      // CUDACHECK(cudaStreamSynchronize(s[i]));

      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclAllReduce((const void*)buff[i], (void*)buff[i], size, ncclFloat, ncclSum, comms[i], s[i]));
      NCCLCHECK(ncclGroupEnd());
    }
    CUDACHECK(cudaStreamEndCapture(s[i], &graphs[i]));
    CUDACHECK(cudaGraphInstantiate(&graphExecs[i], graphs[i], NULL, NULL, 0));
  }

  // Time measurement with CUDA graph
  start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaGraphLaunch(graphExecs[i], s[i]));
  }
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  printf("Time with CUDA graph: %f seconds\n", elapsed.count());

  // Print elements of buff
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMemcpy(hostBuff, buff[i], size * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Buff of device %d:\n", i);
    for (int j = 0; j < 10; ++j) { // Print only the first 10 elements for brevity
      printf("%f ", hostBuff[j]);
    }
    printf("\n");
  }

  // Free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(buff[i]));
    CUDACHECK(cudaGraphDestroy(graphs[i]));
    CUDACHECK(cudaGraphExecDestroy(graphExecs[i]));
  }

  free(buff);
  free(hostBuff);
  free(s);
  free(graphs);
  free(graphExecs);

  // Finalizing NCCL
  for (int i = 0; i < nDev; ++i)
    ncclCommDestroy(comms[i]);

  printf("Success \n");
  return 0;
}
