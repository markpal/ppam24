
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <chrono>
#include <iostream>



using namespace std;


#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CHUNK_SIZE 1

int sigma_host(int a, int b) {
    return a + b;
}



int main() {
    int n = 7500;
    int* h_S, *cpu_S;

    FILE* file = fopen("computeS.cl", "r");
    if (!file) {
        fprintf(stderr, "Error opening file.\n");
        return 1;
    }

    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    rewind(file);

    char* kernelSource = (char*)malloc(fileSize + 1);
    if (!kernelSource) {
        fclose(file);
        fprintf(stderr, "Memory allocation error.\n");
        return 1;
    }

    const char* sources[] = { kernelSource };

    fread(kernelSource, 1, fileSize, file);
    fclose(file);
    kernelSource[fileSize] = '\0';

    h_S = (int*)malloc(n * n * sizeof(int));
    cpu_S = (int*)malloc(n * n * sizeof(int));
    for (int i = 0; i < n * n; i++) {
        h_S[i] = rand() % 100;
        cpu_S[i] = h_S[i];
    }

    cl_int err;
    cl_platform_id cpPlatform[10];
	cl_uint platf_num;
	cl_device_id device;
    err = clGetPlatformIDs(10, cpPlatform, &platf_num);
	
	for(int i = 0; i < platf_num; i++)
	    if(0 == (err = clGetDeviceIDs(cpPlatform[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL))) {
		    printf("Platform #%d\n", i);
		    break;
            }
	printf("GetID=%d\n", err);
	
	cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    printf("CreateCommandQueue=%d\n", err);

    cl_program program = clCreateProgramWithSource(context, 1, sources, NULL, &err);

    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

        cl_int build_status;
clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_int), &build_status, NULL);

if (build_status != CL_SUCCESS) {
    size_t log_size;
    // Print compilation errors
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *program_log = (char *)malloc(log_size + 1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
    program_log[log_size] = '\0';
    printf("Compilation Log:\n%s\n", program_log);
    free(program_log);

    // Handle the compilation error as needed
    return 1; // or some other error code
}

    cl_kernel kernel = clCreateKernel(program, "computeS", &err);

    cl_mem d_S = clCreateBuffer(context, CL_MEM_READ_WRITE, n * n * sizeof(int), NULL, &err);

    clEnqueueWriteBuffer(queue, d_S, CL_TRUE, 0, n * n * sizeof(int), h_S, 0, NULL, NULL);

	int chunk = CHUNK_SIZE;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_S);
    clSetKernelArg(kernel, 1, sizeof(int), &n);
	clSetKernelArg(kernel, 2, sizeof(int), &chunk);

    size_t globalSize = (n / CHUNK_SIZE) + 1;

    auto gpu_start = std::chrono::high_resolution_clock::now();

    for (int c1 = 1; c1 < 2 * n - 2; c1 += 1) {
		if(c1 % 100 == 0)
			cout << c1 << "/" << 2 * n - 2 << endl;
        clSetKernelArg(kernel, 3, sizeof(int), &c1);
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
        clFinish(queue);
    }

    auto gpu_end = std::chrono::high_resolution_clock::now();

    clEnqueueReadBuffer(queue, d_S, CL_TRUE, 0, n * n * sizeof(int), h_S, 0, NULL, NULL);

    auto cpu_start = std::chrono::high_resolution_clock::now();

	if(1==0)
    for (int c1 = 1; c1 < 2 * n - 2; c1 += 1) {
		#pragma omp parallel
        for (int c3 = std::max(0, -n + c1 + 1); c3 < (c1 + 1) / 2; c3++) {
            for (int c5 = 0; c5 <= c3; c5++) {
                cpu_S[(n - c1 + c3 - 1) * n + (n - c1 + 2 * c3)] = MAX(cpu_S[(n - c1 + c3 - 1) * n + (n - c1 + c3 + c5 - 1)] + cpu_S[(n - c1 + c3 + c5) * n + (n - c1 + 2 * c3)], cpu_S[(n - c1 + c3 - 1) * n + (n - c1 + 2 * c3)]);
            }
            cpu_S[(n - c1 + c3 - 1) * n + (n - c1 + 2 * c3)] = MAX(cpu_S[(n - c1 + c3 - 1) * n + (n - c1 + 2 * c3)], cpu_S[(n - c1 + c3) * n + (n - c1 + 2 * c3 - 1)] + sigma_host(n - c1 + c3 - 1, n - c1 + 2 * c3));
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n * n; i++) {
     //   assert(h_S[i] == cpu_S[i]);
        //cout << h_S[i] << " ";
        //cout << cpu_S[i] << endl;
    }

    std::chrono::duration<double> gpu_duration = gpu_end - gpu_start;
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    printf("GPU calculation took: %f s\n", gpu_duration.count());
    printf("CPU calculation took: %f s\n", cpu_duration.count());

    clReleaseMemObject(d_S);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_S);
    free(cpu_S);

    free(kernelSource);

    return 0;
}
