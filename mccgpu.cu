#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <chrono>

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CHUNK_SIZE 2

// Device paired function
__device__ int paired_device(int a, int b) {
    // Replace this with your implementation of the paired function
    return a * b;  // Example implementation, replace with the actual logic
}

int ERT = 1;

// Host paired function
int paired_host(int a, int b) {
    // Replace this with your implementation of the paired function
    return a * b;  // Example implementation, replace with the actual logic
	
}

// Device max function
__device__ int max_device(int a, int b) {
    return MAX(a, b);
}


__global__ void computeQ(int N, int c1, int** d_Q1, int** d_Qbp1) {
    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int c3_start = globalThreadIdx * CHUNK_SIZE;
    int c3_end = c3_start + CHUNK_SIZE;
	
	int ERT=1;

    for (int c3 = c3_start; c3 < c3_end && c3 < (c1 + 1) / 2; c3++) {
        if (c3 >= max_device(0, -N + c1 + 1)) {
            d_Q1[N-c1+c3-1][N-c1+2*c3] = d_Q1[N-c1+c3-1][N-c1+2*c3-1];
            for (int c5 = 0; c5 <= c3; c5++) {
                d_Qbp1[c5+(N-c1+c3-1)][N-c1+2*c3] = d_Q1[c5+(N-c1+c3-1)+1][N-c1+2*c3-1] * ERT * paired_device(c5+(N-c1+c3-1),N-c1+2*c3-1);
                d_Q1[N-c1+c3-1][N-c1+2*c3] += d_Q1[N-c1+c3-1][c5+(N-c1+c3-1)] * d_Qbp1[c5+(N-c1+c3-1)][N-c1+2*c3];
            }
        }
    }
}


__global__ void computeQD(int N, int w0, int** d_Q1, int** d_Qbp1) {
    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
		int ERT=1;

    // Calculate bounds for the iteration space
    int lb = -N + w0 + 1;
    int ub = 0;


    if (w0 < N) {
       // for (int c3 = c3_start; c3 < c3_end && c3 < (c1 + 1) / 2; c3++) {
        for (int h0 = lb + globalThreadIdx; h0 <= ub && h0 <= 0; h0 += blockDim.x * gridDim.x) {
            d_Q1[-h0][w0 - h0] = d_Q1[-h0][w0 - h0 - 1];
            for (int i3 = 0; i3 < w0; i3 += 1) {
                d_Qbp1[-h0 + i3][w0 - h0] = d_Q1[-h0 + i3 + 1][w0 - h0 - 1] * ERT * paired_device(-h0 + i3, w0 - h0 - 1);
                d_Q1[-h0][w0 - h0] += d_Q1[-h0][-h0 + i3] * d_Qbp1[-h0 + i3][w0 - h0];
            }
        }
    }
}



int main() {
    int N = 30000;  // Example size
    int **h_Q1, **d_Q1, **h_Qbp1, **d_Qbp1, **cpu_Q1, **cpu_Qbp1;
    int *d_Q1_data, *d_Qbp1_data;

    // Allocate and initialize host memory
    h_Q1 = (int**)malloc(N * sizeof(int*));
    h_Qbp1 = (int**)malloc(N * sizeof(int*));
    cpu_Q1 = (int**)malloc(N * sizeof(int*));
    cpu_Qbp1 = (int**)malloc(N * sizeof(int*));

    for (int i = 0; i < N; i++) {
        h_Q1[i] = (int*)malloc(N * sizeof(int));
        h_Qbp1[i] = (int*)malloc(N * sizeof(int));
        cpu_Q1[i] = (int*)malloc(N * sizeof(int));
        cpu_Qbp1[i] = (int*)malloc(N * sizeof(int));

        for (int j = 0; j < N; j++) {
            h_Q1[i][j] = 1; rand() % 10;  // Example initialization
            h_Qbp1[i][j] = 1; rand() % 10;  // Example initialization
            cpu_Q1[i][j] = h_Q1[i][j];
            cpu_Qbp1[i][j] = h_Qbp1[i][j];
        }
    }

    // Allocate device memory
    cudaMalloc(&d_Q1_data, N * N * sizeof(int));
    cudaMalloc(&d_Qbp1_data, N * N * sizeof(int));
    cudaMalloc(&d_Q1, N * sizeof(int*));
    cudaMalloc(&d_Qbp1, N * sizeof(int*));

    int **h_Q1_array = (int **)malloc(N * sizeof(int *));
    int **h_Qbp1_array = (int **)malloc(N * sizeof(int *));

    for (int i = 0; i < N; i++) {
        h_Q1_array[i] = d_Q1_data + i * N;
        h_Qbp1_array[i] = d_Qbp1_data + i * N;
    }

    cudaMemcpy(d_Q1, h_Q1_array, N * sizeof(int *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Qbp1, h_Qbp1_array, N * sizeof(int *), cudaMemcpyHostToDevice);

    // Copy data to device
    for (int i = 0; i < N; i++) {
        cudaMemcpy(h_Q1_array[i], h_Q1[i], N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(h_Qbp1_array[i], h_Qbp1[i], N * sizeof(int), cudaMemcpyHostToDevice);
    }

    // GPU computation
    int threadsPerBlock = 8;
    int numBlocks = (N / (threadsPerBlock * CHUNK_SIZE)) + 1;

    auto gpu_start = std::chrono::high_resolution_clock::now();

    if(1==0)
	for (int c1 = 1; c1 < 2 * N - 2; c1 += 1) {
        computeQ<<<numBlocks, threadsPerBlock>>>(N, c1, d_Q1, d_Qbp1);
        cudaDeviceSynchronize();
    }
	
	// dapt
	
for (int w0 = 1; w0 < N; w0 += 1) {        
        computeQD<<<numBlocks, threadsPerBlock>>>(N, w0, d_Q1, d_Qbp1);
        cudaDeviceSynchronize();
	}
	
	/*
  for (int w0 = 1; w0 < N; w0 += 1) {
  //  #pragma omp parallel for
    for (int h0 = -N + w0 + 1; h0 <= 0; h0 += 1) {
      h_Q1[-h0][w0 - h0] = h_Q1[-h0][w0 - h0 - 1];
      for (int i3 = 0; i3 <  + w0; i3 += 1) {
        h_Qbp1[-h0 + i3][w0 - h0] = ((h_Q1[-h0 + i3 + 1][w0 - h0 - 1] * (ERT)) * paired_host((-h0 + i3), (w0 - h0 - 1)));
        h_Q1[-h0][w0 - h0] += (h_Q1[-h0][-h0 + i3] * h_Qbp1[-h0 + i3][w0 - h0]);
      }
    }
  }
*/


    auto gpu_end = std::chrono::high_resolution_clock::now();

    // Copy results back to host
    for (int i = 0; i < N; i++) {
     cudaMemcpy(h_Q1[i], h_Q1_array[i], N * sizeof(int), cudaMemcpyDeviceToHost);
       cudaMemcpy(h_Qbp1[i], h_Qbp1_array[i], N * sizeof(int), cudaMemcpyDeviceToHost);
   }

    // CPU computation
    auto cpu_start = std::chrono::high_resolution_clock::now();

if(1==0)
    for (int c1 = 1; c1 < 2 * N - 2; c1 += 1) {
       // #pragma omp parallel for
        for (int c3 = max(0, -N + c1 + 1); c3 < (c1 + 1) / 2; c3++) {
            cpu_Q1[N-c1+c3-1][N-c1+2*c3] = cpu_Q1[N-c1+c3-1][N-c1+2*c3-1];
            for (int c5 = 0; c5 <= c3; c5++) {
                cpu_Qbp1[c5+(N-c1+c3-1)][N-c1+2*c3] = cpu_Q1[c5+(N-c1+c3-1)+1][N-c1+2*c3-1] * ERT * paired_host(c5+(N-c1+c3-1),N-c1+2*c3-1);
                cpu_Q1[N-c1+c3-1][N-c1+2*c3] += cpu_Q1[N-c1+c3-1][c5+(N-c1+c3-1)] * cpu_Qbp1[c5+(N-c1+c3-1)][N-c1+2*c3];
            }
        }
    }
    


    auto cpu_end = std::chrono::high_resolution_clock::now();

    // Validate results
/*
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert(h_Q1[i][j] == cpu_Q1[i][j]);
           // assert(h_Qbp1[i][j] == cpu_Qbp1[i][j]);
        }
    }    
  */  
  if(1==0)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            
            if(h_Q1[i][j] != cpu_Q1[i][j]) {
                printf("%i %i \n", i, j);
            
            printf("%i %i \n", h_Q1[i][j], cpu_Q1[i][j]);

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
    cudaFree(d_Q1_data);
    cudaFree(d_Qbp1_data);
    cudaFree(d_Q1);
    cudaFree(d_Qbp1);

    // Free host memory
    for (int i = 0; i < N; i++) {
        free(h_Q1[i]);
        free(h_Qbp1[i]);
        free(cpu_Q1[i]);
        free(cpu_Qbp1[i]);
    }
    free(h_Q1);
    free(h_Qbp1);
    free(cpu_Q1);
    free(cpu_Qbp1);
    free(h_Q1_array);
    free(h_Qbp1_array);

    return 0;
}

