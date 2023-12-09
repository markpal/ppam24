#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <chrono>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CHUNK_SIZE 4

__global__ void computeCK(int n, int w0, int** d_ck, int** d_w) {
    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int h0 = -n + w0 + globalThreadIdx;

    for (int i2 = 1; i2 < w0 - h0; i2++) {
        d_ck[-h0][w0 - h0] = MIN(d_ck[-h0][w0 - h0], (d_w[-h0][w0 - h0] + d_ck[-h0][i2]) + d_ck[i2][w0 - h0]);
    }
}




int main() {
    int n = 3000;  // Example size
    int **h_w, **d_w, **h_ck, **d_ck, **h_ck_array;
    int *d_w_data, *d_ck_data;

    // Allocate and initialize host memory
    h_w = (int**)malloc(n * sizeof(int*));
    h_ck = (int**)malloc(n * sizeof(int*));

    for (int i = 0; i < n; i++) {
        h_w[i] = (int*)malloc(n * sizeof(int));
        h_ck[i] = (int*)malloc(n * sizeof(int));

        for (int j = 0; j < n; j++) {
            h_w[i][j] = rand() % 100;  // Example initialization
            h_ck[i][j] = rand() % 100; // Example initialization
        }
    }

    // Allocate device memory
    cudaMalloc(&d_w_data, n * n * sizeof(int));
    cudaMalloc(&d_ck_data, n * n * sizeof(int));
    cudaMalloc(&d_w, n * sizeof(int*));
    cudaMalloc(&d_ck, n * sizeof(int*));

    h_ck_array = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        h_ck_array[i] = d_ck_data + i * n;
    }
    cudaMemcpy(d_w, h_w, n * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ck, h_ck_array, n * sizeof(int*), cudaMemcpyHostToDevice);

    // Copy data to device
    for (int i = 0; i < n; i++) {
        cudaMemcpy(h_w[i], d_w[i], n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(h_ck_array[i], d_ck[i], n * sizeof(int), cudaMemcpyHostToDevice);
    }

    // GPU computation
    int threadsPerBlock = 256;
    int numBlocks = (n / threadsPerBlock) + 1;

    auto gpu_start = std::chrono::high_resolution_clock::now();

    computeCK<<<numBlocks, threadsPerBlock>>>(n, d_w, d_ck);
    cudaDeviceSynchronize();

    auto gpu_end = std::chrono::high_resolution_clock::now();

    // Copy results back to host
    for (int i = 0; i < n; i++) {
        cudaMemcpy(h_ck[i], h_ck_array[i], n * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Validate results
    // Add validation code here if needed

    // Print timings
    std::chrono::duration<double, std::milli> gpu_duration = gpu_end - gpu_start;
    printf("GPU calculation took: %f ms\n", gpu_duration.count());

    // Free device memory
    cudaFree(d_w_data);
    cudaFree(d_ck_data);
    cudaFree(d_w);
    cudaFree(d_ck);

    // Free host memory
    for (int i = 0; i < n; i++) {
        free(h_w[i]);
        free(h_ck[i]);
    }
    free(h_w);
    free(h_ck);
    free(h_ck_array);

    return 0;
}
