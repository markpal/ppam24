#include <iostream>
#include <algorithm>  // Include the algorithm header for the min function
#include <cuda_runtime.h>

__global__ void cudaKernel(int n, int** w, int** ck) {
    int w0 = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (w0 < n) {
        for (int h0 = -n + w0; h0 < 0; h0 += 1) {
            for (int i2 = -h0 + 1; i2 < w0 - h0; i2 += 1) {
                ck[(w0 - h0) * n + (-h0)] = std::min(ck[(w0 - h0) * n + (-h0)], (w[(w0 - h0) * n + (-h0)] + ck[(w0 - h0) * n + (-h0) - i2]) + ck[(w0 - h0) * n + (-h0) + i2]);
            }
        }
    }
}

// Rest of the code remains unchanged...


int main() {
    int n = 100;  // Replace with your desired value for n

    // Allocate and initialize data on the host
    int** h_w = new int*[n];
    int** h_ck = new int*[n];
    for (int i = 0; i < n; ++i) {
        h_w[i] = new int[n];
        h_ck[i] = new int[n];
        // Initialize h_w[i] and h_ck[i] as needed
    }

    // Allocate memory on the device
    int** d_w, **d_ck;
    cudaMalloc((void**)&d_w, n * sizeof(int*));
    cudaMalloc((void**)&d_ck, n * sizeof(int*));

    // Copy data from host to device
    for (int i = 0; i < n; ++i) {
        cudaMalloc((void**)&d_w[i], n * sizeof(int));
        cudaMalloc((void**)&d_ck[i], n * sizeof(int));
        cudaMemcpy(d_w[i], h_w[i], n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ck[i], h_ck[i], n * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Set up the grid and block dimensions
    int blockSize = 256;  // Adjust as needed
    int gridSize = (n - 2 + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    cudaKernel<<<gridSize, blockSize>>>(n, d_w, d_ck);

    // Copy the result back from device to host
    for (int i = 0; i < n; ++i) {
        cudaMemcpy(h_ck[i], d_ck[i], n * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Free allocated memory on the device
    for (int i = 0; i < n; ++i) {
        cudaFree(d_w[i]);
        cudaFree(d_ck[i]);
    }
    cudaFree(d_w);
    cudaFree(d_ck);

    // Free allocated memory on the host
    for (int i = 0; i < n; ++i) {
        delete[] h_w[i];
        delete[] h_ck[i];
    }
    delete[] h_w;
    delete[] h_ck;

    return 0;
}

