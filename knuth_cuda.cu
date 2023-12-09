#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <chrono>

#define MIN(a,b) ((a) < (b) ? (a) : (b))

__global__ void computeCK(int n, int** d_ck, int** d_w) {
    int h0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (h0 < 0) {
        for (int i2 = -h0 + 1; i2 < n - h0; i2++) {
            d_ck[-h0][n - h0] = MIN(d_ck[-h0][n - h0], (d_w[-h0][n - h0] + d_ck[-h0][i2]) + d_ck[i2][n - h0]);
        }
    }
}

// Host implementation for CK computation
void computeCKHost(int n, int** h_ck, int** h_w) {
    for (int w0 = 2; w0 < n; w0++) {
        for (int h0 = -n + w0; h0 < 0; h0++) {
            for (int i2 = -h0 + 1; i2 < w0 - h0; i2++) {
                h_ck[-h0][w0 - h0] = MIN(h_ck[-h0][w0 - h0], (h_w[-h0][w0 - h0] + h_ck[-h0][i2]) + h_ck[i2][w0 - h0]);
            }
        }
    }
}

int main() {
    int n = 1000;  // Example size
    int **h_ck, **d_ck, **h_w, **d_w;
    int *d_ck_data, *d_w_data;

    // Allocate and initialize host memory
    h_ck = (int**)malloc(n * sizeof(int*));
    h_w = (int**)malloc(n * sizeof(int*));

    for (int i = 0; i < n; i++) {
        h_ck[i] = (int*)malloc(n * sizeof(int));
        h_w[i] = (int*)malloc(n * sizeof(int));

        for (int j = 0; j < n; j++) {
            h_ck[i][j] = INT_MAX;  // Initialize to a large value
            h_w[i][j] = rand() % 100;  // Example initialization
        }
    }

    // Allocate device memory
    cudaMalloc(&d_ck_data, n * n * sizeof(int));
    cudaMalloc(&d_ck, n * sizeof(int*));
    cudaMalloc(&d_w_data, n * n * sizeof(int));
    cudaMalloc(&d_w, n * sizeof(int*));

    int **h_ck_array = (int **)malloc(n * sizeof(int *));
    int **h_w_array = (int **)malloc(n * sizeof(int *));

    for (int i = 0; i < n; i++) {
        h_ck_array[i] = d_ck_data + i * n;
        h_w_array[i] = d_w_data + i * n;
    }

    cudaMemcpy(d_ck, h_ck_array, n * sizeof(int *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w_array, n * sizeof(int *), cudaMemcpyHostToDevice);

    // Copy data to device
    for (int i = 0; i < n; i++) {
        cudaMemcpy(h_ck_array[i], h_ck[i], n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(h_w_array[i], h_w[i], n * sizeof(int), cudaMemcpyHostToDevice);
    }

    // GPU computation
    int threadsPerBlock = 256;
    int numBlocks = (n / threadsPerBlock) + 1;

    auto gpu_start = std::chrono::high_resolution_clock::now();

    for (int w0 = 2; w0 < n; w0++) {
        computeCK<<<numBlocks, threadsPerBlock>>>(n, d_ck, d_w);
        cudaDeviceSynchronize();
    }

    auto gpu_end = std::chrono::high_resolution_clock::now();

    // Copy results back to host
    for (int i = 0; i < n; i++) {
        cudaMemcpy(h_ck[i], h_ck_array[i], n * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Host computation
    auto cpu_start = std::chrono::high_resolution_clock::now();

    computeCKHost(n, h_ck, h_w);

    auto cpu_end = std::chrono::high_resolution_clock::now();


// Validate results
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        assert(h_ck[i][j] == h_ck_array[i][j]);
    }
}

    // Print timings
    std::chrono::duration<double, std::milli> gpu_duration = gpu_end - gpu_start;
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    printf("GPU calculation took: %f ms\n", gpu_duration.count());
    printf("CPU calculation took: %f ms\n", cpu_duration.count());

    // Free device memory
    cudaFree(d_ck_data);
    cudaFree(d_ck);
    cudaFree(d_w_data);
    cudaFree(d_w);

    // Free host memory
    for (int i = 0; i < n; i++) {
        free(h_ck[i]);
        free(h_w[i]);
    }
    free(h_ck);
    free(h_w);
    free(h_ck_array);
    free(h_w_array);

    return 0;
}
