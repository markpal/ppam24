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
    int **h_ck, **d_ck, **cpu_ck, **h_w, **d_w;

    // Allocate and initialize host memory
    h_ck = (int**)malloc(n * sizeof(int*));
    cpu_ck = (int**)malloc(n * sizeof(int*));
    h_w = (int**)malloc(n * sizeof(int*));

    for (int i = 0; i < n; i++) {
        h_ck[i] = (int*)malloc(n * sizeof(int));
        cpu_ck[i] = (int*)malloc(n * sizeof(int));
        h_w[i] = (int*)malloc(n * sizeof(int));

        for (int j = 0; j < n; j++) {
            h_ck[i][j] = rand() % 100;  // Example initialization
            cpu_ck[i][j] = h_ck[i][j];
            h_w[i][j] = rand() % 100;  // Example initialization
        }
    }

    // Allocate device memory for ck
    cudaMalloc(&d_ck, n * sizeof(int*));
    int **h_ck_array = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        h_ck_array[i] = d_ck + i * n;
    }
    cudaMemcpy(d_ck, h_ck_array, n * sizeof(int *), cudaMemcpyHostToDevice);

    // Copy data to device for ck
    for (int i = 0; i < n; i++) {
        cudaMemcpy(h_ck_array[i], h_ck[i], n * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Allocate device memory for w
    cudaMalloc(&d_w, n * sizeof(int*));
    int **h_w_array = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        h_w_array[i] = d_w + i * n;
    }
    cudaMemcpy(d_w, h_w_array, n * sizeof(int *), cudaMemcpyHostToDevice);

    // Copy data to device for w
    for (int i = 0; i < n; i++) {
        cudaMemcpy(h_w_array[i], h_w[i], n * sizeof(int), cudaMemcpyHostToDevice);
    }

    // GPU computation
    int threadsPerBlock = 256;
    int numBlocks = (n / threadsPerBlock) + 1;

    auto gpu_start = std::chrono::high_resolution_clock::now();

    for (int w0 = 2; w0 < n; w0 += 1) {
        computeCK<<<numBlocks, threadsPerBlock>>>(n, w0, d_ck, d_w);
        cudaDeviceSynchronize();
    }

    auto gpu_end = std::chrono::high_resolution_clock::now();

    // Copy results back to host for ck
    for (int i = 0; i < n; i++) {
        cudaMemcpy(h_ck[i], h_ck_array[i], n * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // CPU computation for ck
    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (int w0 = 2; w0 < n; w0 += 1) {
        for (int h0 = -n + w0; h0 < 0; h0 += 1) {
            for (int i2 = 1; i2 < w0 - h0; i2 += 1) {
                cpu_ck[-h0][w0 - h0] = MIN(cpu_ck[-h0][w0 - h0], (h_w[-h0][w0 - h0] + cpu_ck[-h0][i2]) + cpu_ck[i2][w0 - h0]);
            }
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();

    // Validate results for ck
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            assert(h_ck[i][j] == cpu_ck[i][j]);
        }
    }

    // Print timings for ck
    std::chrono::duration<double, std::milli> gpu_duration = gpu_end - gpu_start;
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    printf("GPU calculation took: %f ms\n", gpu_duration.count());
    printf("CPU calculation took: %f ms\n", cpu_duration.count());

    // Free device memory for ck
    cudaFree(d_ck);

    // Free host memory for ck
    for (int i = 0; i < n; i++) {
        free(h_ck[i]);
        free(cpu_ck[i]);
    }
    free(h_ck);
    free(cpu_ck);
    free(h_ck_array);

    // Free device memory for w
    cudaFree(d_w);

    // Free host memory for w
    for (int i = 0; i < n; i++) {
        free(h_w[i]);
    }
    free(h_w);
    free(h_w_array);

    return 0;
}
