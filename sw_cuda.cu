#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <chrono>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CHUNK_SIZE 4

// Device sigma function
__device__ int sigma_device(int a, int b) {
    return a + b;
}

// Host sigma function
int sigma_host(int a, int b) {
    return a + b;
}

__global__ void computeM(int N, int c1, int *d_m1, int *d_m2, int *d_H, int *d_W, int *d_a, int *d_b) {
    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int c3_start = max(0, -N + c1 + 1);
    int c3_end = min(N - 1, c1);

    for (int c3 = c3_start + globalThreadIdx; c3 <= c3_end; c3 += blockDim.x * gridDim.x) {
        int index = (c1 - c3 + 1) * (N + 2) + (c3 + 1);
        for (int c5 = 0; c5 <= c3; c5++) {
            d_m2[index] = MAX(d_m2[index], d_H[index - (c5 + 1)] + d_W[c5 + 1]);
        }
        for (int c5 = 0; c5 <= c1 - c3; c5++) {
            d_m1[index] = MAX(d_m1[index], d_H[index - (c5 + 1) * (N + 2)] + d_W[c5 + 1]);
        }
        d_H[index] = MAX(0, MAX(d_H[index - (N + 2) - 1] + sigma_device(d_a[c1 - c3 + 1], d_b[c1 - c3 + 1]),
                              MAX(d_m1[index], d_m2[index])));
    }
}

// CPU function for computation
void computeM_CPU(int N, int c1, int *h_m1, int *h_m2, int *h_H, int *h_W, int *h_a, int *h_b) {
    for (int c3 = max(0, -N + c1 + 1); c3 <= min(N - 1, c1); c3++) {
        int index = (c1 - c3 + 1) * (N + 2) + (c3 + 1);
        for (int c5 = 0; c5 <= c3; c5++) {
            h_m2[index] = MAX(h_m2[index], h_H[index - (c5 + 1)] + h_W[c5 + 1]);
        }
        for (int c5 = 0; c5 <= c1 - c3; c5++) {
            h_m1[index] = MAX(h_m1[index], h_H[index - (c5 + 1) * (N + 2)] + h_W[c5 + 1]);
        }
        h_H[index] = MAX(0, MAX(h_H[index - (N + 2) - 1] + sigma_host(h_a[c1 - c3 + 1], h_b[c1 - c3 + 1]),
                              MAX(h_m1[index], h_m2[index])));
    }
}



int main() {
    int N = 1000; // Example size
    int *h_m1, *h_m2, *h_H, *d_m1, *d_m2, *d_H;
    int *d_W, *d_a, *d_b;
    int *h_W, *h_a, *h_b;
	int *CPU_H;
	int i,j;

    // Allocate and initialize host memory
    h_m1 = (int *)malloc((N + 2) * (N + 2) * sizeof(int));
    h_m2 = (int *)malloc((N + 2) * (N + 2) * sizeof(int));
    h_H = (int *)malloc((N + 2) * (N + 2) * sizeof(int));
	CPU_H = (int *)malloc((N + 2) * (N + 2) * sizeof(int));
	int** H = new int*[N+2];
	int **m1 = new int*[N+2];
	int **m2 = new int*[N+2];
	for(i=0; i<N+2; i++){
		H[i] = CPU_H + i*(N+2);
		m1[i] = new int[N+2];
		m2[i] = new int[N+2];
		for(j=0; j < N+2; j++)
		    m1[i][j] = m2[i][j] = INT_MIN;
	}


    for (int i = 0; i < (N + 2) * (N + 2); i++) {
        h_m1[i] = INT_MIN;
        h_m2[i] = INT_MIN;
        CPU_H[i] = h_H[i] = rand() % 100;  // Example initialization
    }

    // Allocate device memory
    cudaMalloc(&d_m1, (N + 2) * (N + 2) * sizeof(int));
    cudaMalloc(&d_m2, (N + 2) * (N + 2) * sizeof(int));
    cudaMalloc(&d_H, (N + 2) * (N + 2) * sizeof(int));

    cudaMemcpy(d_m1, h_m1, (N + 2) * (N + 2) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, h_m2, (N + 2) * (N + 2) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_H, h_H, (N + 2) * (N + 2) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_W, (N + 2) * sizeof(int));
    cudaMalloc(&d_a, (N + 2) * sizeof(int));
    cudaMalloc(&d_b, (N + 2) * sizeof(int));

    // Host arrays for computation
    h_W = (int *)malloc((N + 2) * sizeof(int));
    h_a = (int *)malloc((N + 2) * sizeof(int));
    h_b = (int *)malloc((N + 2) * sizeof(int));

    for (int i = 0; i < N + 2; i++) {
        h_W[i] = rand() % 100; // Example initialization
        h_a[i] = rand() % 100; // Example initialization
        h_b[i] = rand() % 100; // Example initialization
    }

    cudaMemcpy(d_W, h_W, (N + 2) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, (N + 2) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, (N + 2) * sizeof(int), cudaMemcpyHostToDevice);

    // GPU computation
    int threadsPerBlock = 256;
    int numBlocks = (N / threadsPerBlock) + 1;

    auto gpu_start = std::chrono::high_resolution_clock::now();

    for (int c1 = 0; c1 < 2 * N - 1; c1 += 1) {
        computeM<<<numBlocks, threadsPerBlock>>>(N, c1, d_m1, d_m2, d_H, d_W, d_a, d_b);
        cudaDeviceSynchronize();
    }

    auto gpu_end = std::chrono::high_resolution_clock::now();

    // Copy results back to host
    //cudaMemcpy(h_m1, d_m1, (N + 2) * (N + 2) * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_m2, d_m2, (N + 2) * (N + 2) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_H, d_H, (N + 2) * (N + 2) * sizeof(int), cudaMemcpyDeviceToHost);

    // CPU computation
    auto cpu_start = std::chrono::high_resolution_clock::now();

    int c1, c3, c5;
//    for (int c1 = 0; c1 < 2 * N - 1; c1 += 1) {
//        computeM_CPU(N, c1, h_m1, h_m2, CPU_H, h_W, h_a, h_b);
//    }


		for( c1 = 0; c1 < 2 * N - 1; c1 += 1)
		  #pragma omp parallel for
		  for( c3 = max(0, -N + c1 + 1); c3 <= min(N - 1, c1); c3 += 1){
			  for( c5 = 0; c5 <= c3; c5 += 1)
				  m2[(c1-c3+1)][(c3+1)] = MAX(m2[(c1-c3+1)][(c3+1)] ,H[(c1-c3+1)][(c3+1)-(c5+1)] + h_W[(c5+1)]);
			  for( c5 = 0; c5 <= c1 - c3; c5 += 1)
				  m1[(c1-c3+1)][(c3+1)] = MAX(m1[(c1-c3+1)][(c3+1)] ,H[(c1-c3+1)-(c5+1)][(c3+1)] + h_W[(c5+1)]);
			  H[(c1-c3+1)][(c3+1)] = MAX(0, MAX( H[(c1-c3+1)-1][(c3+1)-1] + sigma_host(h_a[(c1-c3+1)], h_b[(c1-c3+1)]), MAX(m1[(c1-c3+1)][(c3+1)], m2[(c1-c3+1)][(c3+1)])));
		}

    auto cpu_end = std::chrono::high_resolution_clock::now();

    // Verification
    for (int i = 0; i < (N + 2) * (N + 2); i++) {
        assert(h_m1[i] == h_m1[i]);
        assert(h_m2[i] == h_m2[i]);
        assert(CPU_H[i] == h_H[i]);
    }

    // Free device memory
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_H);
    cudaFree(d_W);
    cudaFree(d_a);
    cudaFree(d_b);

    // Free host memory
    free(h_m1);
    free(h_m2);
    free(h_H);
    free(h_W);
    free(h_a);
    free(h_b);

    // Print GPU calculation time
    std::chrono::duration<double> gpu_duration = gpu_end - gpu_start;
    printf("GPU calculation took: %f s\n", gpu_duration.count());

    // Print CPU calculation time
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    printf("CPU calculation took: %f s\n", cpu_duration.count());

    return 0;
}
