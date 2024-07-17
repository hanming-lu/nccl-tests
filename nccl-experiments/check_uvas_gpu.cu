#include <iostream>
#include <cuda_runtime.h>

void checkP2PandUVAS() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error getting device count: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "Number of GPUs: " << deviceCount << std::endl;

    bool allSupportUVAS = true;

    for (int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, i);
        if (err != cudaSuccess) {
            std::cerr << "Error getting device properties for device " << i << ": " << cudaGetErrorString(err) << std::endl;
            allSupportUVAS = false;
            continue;
        }

        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Unified Addressing: " << (deviceProp.unifiedAddressing ? "Yes" : "No") << std::endl;

        if (!deviceProp.unifiedAddressing) {
            allSupportUVAS = false;
        }
    }

    if (allSupportUVAS) {
        std::cout << "All GPUs support Unified Virtual Address Space (UVAS)." << std::endl;
    } else {
        std::cout << "Not all GPUs support Unified Virtual Address Space (UVAS)." << std::endl;
    }

    std::cout << "Checking Peer-to-Peer (P2P) access between GPUs..." << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        for (int j = 0; j < deviceCount; ++j) {
            if (i != j) {
                int canAccessPeer = 0;
                cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
                if (canAccessPeer) {
                    std::cout << "GPU " << i << " can access GPU " << j << std::endl;
                    cudaSetDevice(i);
                    cudaDeviceEnablePeerAccess(j, 0);
                } else {
                    std::cout << "GPU " << i << " cannot access GPU " << j << std::endl;
                }
            }
        }
    }
}

int main() {
    checkP2PandUVAS();
    return 0;
}
