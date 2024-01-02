#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <chrono>
#include <iostream>

using namespace std;

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CHUNK_SIZE 2

// Device sigma function
__device__ int sigma_device(int a, int b) {
    return a + b;
}

// Host sigma function
int sigma_host(int a, int b) {
    return a + b;
}

__global__ void computeCK(int n, int c1, int** d_CK, int** d_W) {
    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int c3_start = globalThreadIdx * CHUNK_SIZE;
    int c3_end = c3_start + CHUNK_SIZE;

    for (int c3 = c3_start; c3 < c3_end && c3 <= c1 / 2; c3++) {
        if (c3 >= MAX(1, -n + c1 + 2)) {
            for (int c5 = 0; c5 < c3; c5++) {
                d_CK[n-c1+c3-1][n-c1+2*c3] = MIN(d_CK[n-c1+c3-1][n-c1+2*c3], d_W[n-c1+c3-1][n-c1+2*c3] + d_CK[n-c1+c3-1][n-c1+c3+c5] + d_CK[n-c1+c3+c5][n-c1+2*c3]);
            }
        }
    }
}


__global__ void computeCKD(int n, int w0, int** d_CK, int** d_W) {
    int h0, i2;

    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate bounds for the iteration space
    int lb = -n + w0;
    int ub = 0;

    for (h0 = lb + globalThreadIdx; h0 < ub; h0 += blockDim.x * gridDim.x) {
        for (i2 = -h0 + 1; i2 < w0 - h0; i2++) {
            d_CK[-h0][w0 - h0] = MIN(d_CK[-h0][w0 - h0], (d_W[-h0][w0 - h0] + d_CK[-h0][i2]) + d_CK[i2][w0 - h0]);
            cudaDeviceSynchronize();

        }
    }
}


int main() {
    int N = 2000;
    int n = N + 2;
    int **h_CK, **d_CK, **cpu_CK, **h_W, **d_W, **cpu_W;
    int *d_CK_data, *d_W_data;



    // Allocate and initialize host memory
    h_CK = (int**)malloc(n * sizeof(int*));
    cpu_CK = (int**)malloc(n * sizeof(int*));
    h_W = (int**)malloc(n * sizeof(int*));
    cpu_W = (int**)malloc(n * sizeof(int*));

    for (int i = 0; i < n; i++) {
        h_CK[i] = (int*)malloc(n * sizeof(int));
        cpu_CK[i] = (int*)malloc(n * sizeof(int));
        h_W[i] = (int*)malloc(n * sizeof(int));
        cpu_W[i] = (int*)malloc(n * sizeof(int));

        for (int j = 0; j < n; j++) {
            h_CK[i][j] = rand() % 100;  // Example initialization for CK
            h_W[i][j] = rand() % 100;  // Example initialization for W
            cpu_CK[i][j] = h_CK[i][j];
            cpu_W[i][j] = h_W[i][j];
        }
    }

    // Allocate device memory
    cudaMalloc(&d_CK_data, n * n * sizeof(int));
    cudaMalloc(&d_CK, n * sizeof(int*));

    cudaMalloc(&d_W_data, n * n * sizeof(int));
    cudaMalloc(&d_W, n * sizeof(int*));

    int **h_CK_array = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        h_CK_array[i] = d_CK_data + i * n;
    }
    cudaMemcpy(d_CK, h_CK_array, n * sizeof(int *), cudaMemcpyHostToDevice);

    int **h_W_array = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        h_W_array[i] = d_W_data + i * n;
    }
    cudaMemcpy(d_W, h_W_array, n * sizeof(int *), cudaMemcpyHostToDevice);

    // Copy data to device
    for (int i = 0; i < n; i++) {
        cudaMemcpy(h_CK_array[i], h_CK[i], n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(h_W_array[i], h_W[i], n * sizeof(int), cudaMemcpyHostToDevice);
    }



    // GPU computation
    int threadsPerBlock = 64;
    int numBlocks = (n / (threadsPerBlock * CHUNK_SIZE)) + 1;

    auto gpu_start = std::chrono::high_resolution_clock::now();

// traco
//    for (int c1 = 2; c1 < 2 * n - 3; c1 += 1) {
//        computeCK<<<numBlocks, threadsPerBlock>>>(N, c1, d_CK, d_W);
//        cudaDeviceSynchronize();
 //   }
	// dapt
	for (int w0 = 2; w0 < n; w0 += 1) {
	//cout << w0 << endl;
	    computeCKD<<<numBlocks, threadsPerBlock>>>(N, w0, d_CK, d_W);
	}

    auto gpu_end = std::chrono::high_resolution_clock::now();

    // Copy results back to host
    for (int i = 0; i < n; i++) {
        cudaMemcpy(h_CK[i], h_CK_array[i], n * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // CPU computation
    auto cpu_start = std::chrono::high_resolution_clock::now();

	//if(1==0)
    for (int c1 = 2; c1 < 2 * N - 3; c1 += 1) {
        for (int c3 = max(1, -N + c1 + 2); c3 <= c1 / 2; c3++) {
            for (int c5 = 0; c5 < c3; c5++) {
                cpu_CK[N-c1+c3-1][N-c1+2*c3] = MIN(cpu_CK[N-c1+c3-1][N-c1+2*c3], cpu_W[N-c1+c3-1][N-c1+2*c3] + cpu_CK[N-c1+c3-1][N-c1+c3+c5] + cpu_CK[N-c1+c3+c5][N-c1+2*c3]);
            }
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();

    // Validate results
	//if(1==0)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            
            if(h_CK[i][j] != cpu_CK[i][j]) {
                cout << i << " " <<  j << " ";
            
            cout << h_CK[i][j] << "  " <<  cpu_CK[i][j];
            cout << endl;
			break;
           }
        }
    }

    // Print timings
    std::chrono::duration<double> gpu_duration = gpu_end - gpu_start;
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    printf("GPU calculation took: %f s\n", gpu_duration.count());
    printf("CPU calculation took: %f s\n", cpu_duration.count());

    // Free device memory
    cudaFree(d_CK_data);
    cudaFree(d_CK);
    cudaFree(d_W_data);
    cudaFree(d_W);

    // Free host memory
    for (int i = 0; i < n; i++) {
        free(h_CK[i]);
        free(cpu_CK[i]);
        free(h_W[i]);
        free(cpu_W[i]);
    }
    free(h_CK);
    free(cpu_CK);
    free(h_CK_array);
    free(h_W);
    free(cpu_W);
    free(h_W_array);

    return 0;
}
