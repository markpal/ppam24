#include <cuda_runtime.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

__global__ void cudaKernel(int n, int *ck, int *w) {
    int w0 = blockIdx.x * blockDim.x + threadIdx.x + 2;

    if (w0 < n) {
        #pragma unroll
        for (int h0 = -n + w0; h0 < 0; h0 += 1) {
            for (int i2 = -h0 + 1; i2 < w0 - h0; i2 += 1) {
                ck[-h0 * n + w0 - h0] = MIN(ck[-h0 * n + w0 - h0], (w[-h0 * n + w0 - h0] + ck[-h0 * n + i2]) + ck[i2 * n + w0 - h0]);
            }
        }
    }
}

int main() {
    // Assuming you have allocated and initialized your arrays on the host
    int n = /* your value for n */;
    int *ck, *w;  // Assuming these arrays are properly declared and initialized

    // Allocate device memory
    int *d_ck, *d_w;
    cudaMalloc((void**)&d_ck, sizeof(int) * n * n);
    cudaMalloc((void**)&d_w, sizeof(int) * n * n);

    // Copy data from host to device
    cudaMemcpy(d_ck, ck, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, sizeof(int) * n * n, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(256);  // You may need to adjust this based on your specific GPU architecture
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

    // Launch the CUDA kernel
    cudaKernel<<<gridDim, blockDim>>>(n, d_ck, d_w);

    // Copy the result back to host if needed
    cudaMemcpy(ck, d_ck, sizeof(int) * n * n, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_ck);
    cudaFree(d_w);

    return 0;
}
