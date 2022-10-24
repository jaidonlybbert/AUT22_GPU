
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

cudaError_t printfCuda();
cudaError_t saxpyCuda(int size);
cudaError_t matrixAddCuda(int M, int N);
cudaError_t gridAddCuda(int M, int N, int P);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void printfKernel()
{
    int gutID = blockIdx.x * blockDim.x + threadIdx.x;
    printf("ThreadID, blockID, GUTID: %d, %d, %d \n", threadIdx.x, blockIdx.x, gutID);
}

__global__ void saxpyKernel(float* c, const float* x, const float* y, const float a, int size)
{
    int gutID = blockIdx.x * blockDim.x + threadIdx.x;

    if (gutID < size) {
        c[gutID] = a * x[gutID] + y[gutID];
    }
}

__global__ void matrixAddKernel(float* c, const float* a, const float* b, int M, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < M) {
        c[row * M + col] = a[row * M + col] + b[row * M + col];
    }
}

__global__ void gridAddKernel(float* c, const float* a, const float* b, int M, int N, int P)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int layer = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < N && col < M && layer < P) {
        c[row * N * P + col * P + layer] = a[row * N * P + col * P + layer] + b[row * N * P + col * P + layer];
    }
}

int main()
{
    // Seed randomizer to generate input data
    srand((unsigned int)time(NULL));

    checkCuda(printfCuda());
    checkCuda(saxpyCuda(1E7));
    checkCuda(matrixAddCuda(3024, 4032));
    checkCuda(gridAddCuda(100, 100, 100));


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t printfCuda()
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // Launch a kernel on the GPU with one thread for each element.
    printfKernel<<<4, 4 >>>();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    return cudaStatus;
}

cudaError_t saxpyCuda(int size)
{
    float a = 7.8;
    
    // allocate host memory for input and output arrays
    float* h_x = (float*)malloc(size * sizeof(float));
    float* h_y = (float*)malloc(size * sizeof(float));
    float* h_c = (float*)malloc(size * sizeof(float));

    // initialize input arrays
    for (int i = 0; i < size; i++) {
        h_x[i] = ((float)rand()/(float)(RAND_MAX)) * 1.0;
        h_y[i] = ((float)rand() / (float)(RAND_MAX)) * 1.0;
    }

    // compute CPU verification result
    float* cpu_result = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        cpu_result[i] = a * h_x[i] + h_y[i];
    }

    float* dev_x = 0;
    float* dev_y = 0;
    float* dev_c = 0;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_x, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_y, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_x, h_x, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_y, h_y, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    saxpyKernel<<<ceil(size/1024.0), 1024>>> (dev_c, dev_x, dev_y, a, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    float tolerance = 1e-7;

    bool correct = true;
    float sum_of_squared_diff = 0.0;
    for (int i = 0; i < size; i++) {
        sum_of_squared_diff += pow(abs((float)(cpu_result[i] - h_c[i])), 2);
        if (abs((float)(cpu_result[i] - h_c[i])) >= tolerance) {
            correct = false;
        }
    }

    if (correct) {
        printf("SAXPY verification passed. ");
    }
    else {
        printf("SAXPY verification failed. ");
    }

    float variance = sum_of_squared_diff / (size);

    printf("Variance: %f\n", variance);
    

Error:
    cudaFree(dev_c);
    cudaFree(dev_x);
    cudaFree(dev_y);

    return cudaStatus;
}

cudaError_t matrixAddCuda(int M, int N)
{
    // allocate host memory for input and output arrays
    float* h_a = (float*)malloc(M * N * sizeof(float));
    float* h_b = (float*)malloc(M * N * sizeof(float));
    float* h_c = (float*)malloc(M * N * sizeof(float));

    // initialize input arrays
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_a[i * N + j] = ((float)rand() / (float)(RAND_MAX)) * 1.0;
            h_b[i * N + j] = ((float)rand() / (float)(RAND_MAX)) * 1.0;
        }
    }

    // compute CPU verification result
    float* cpu_result = (float*)malloc(M * N * sizeof(float));

    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            cpu_result[i * N + j] = h_a[i * N + j] + h_b[i * N + j];
        }
    }

    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, M * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, M * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, M * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, h_a, M * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, h_b, M * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    dim3 dimGrid(ceil(M / 16.0), ceil(N / 16.0), 1);
    dim3 dimBlock(16, 16, 1);

    matrixAddKernel<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b, M, N);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    float tolerance = 1e-7;

    bool correct = true;
    float sum_of_squared_diff = 0.0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            sum_of_squared_diff += pow(abs(cpu_result[i * N + j] - h_c[i * N + j]), 2);

            if (abs(cpu_result[i * N + j] - h_c[i * N + j]) >= tolerance) {
                correct = false;
            }
        }
    }

    float variance = sum_of_squared_diff / (M * N);

    if (correct) {
        printf("Matrix Add verification passed. ");
    }
    else {
        printf("Matrix Add verification failed. ");
    }

    printf("Variance: %f\n", variance);

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

cudaError_t gridAddCuda(int M, int N, int P)
{
    // allocate host memory for input and output arrays
    float* h_a = (float*)malloc(M * N * P * sizeof(float));
    float* h_b = (float*)malloc(M * N * P * sizeof(float));
    float* h_c = (float*)malloc(M * N * P * sizeof(float));

    // initialize input arrays
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < P; k++) {
                h_a[i * N * P + j * P + k] = ((float)rand() / (float)(RAND_MAX)) * 1.0;
                h_b[i * N * P + j * P + k] = ((float)rand() / (float)(RAND_MAX)) * 1.0;
            }
        }
    }

    // compute CPU verification result
    float* cpu_result = (float*)malloc(M * N * P * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < P; k++) {
                cpu_result[i * N * P + j * P + k] = h_a[i * N * P + j * P + k] + h_b[i * N * P + j * P + k];
            }
        }
    }

    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, M * N * P * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, M * N * P * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, M * N * P * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, h_a, M * N * P * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, h_b, M * N * P * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    dim3 dimGrid(ceil(M / 8.0), ceil(N / 8.0), ceil(P / 8.0));
    dim3 dimBlock(8, 8, 8);

    gridAddKernel<<<dimGrid, dimBlock>>> (dev_c, dev_a, dev_b, M, N, P);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_c, dev_c, M * N * P * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    float tolerance = 1e-7;

    bool correct = true;
    float sum_of_squared_diff = 0.0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < P; k++) {
                sum_of_squared_diff += pow(abs(cpu_result[i * N * P + j * P + k] - h_c[i * N * P + j * P + k]), 2);

                if (abs(cpu_result[i * N * P + j * P + k] - h_c[i * N * P + j * P + k]) >= tolerance) {
                    correct = false;
                }
            }
        }
    }

    float variance = sum_of_squared_diff / (M * N * P);

    if (correct) {
        printf("Grid Add verification passed. ");
    }
    else {
        printf("Grid Add verification failed. ");
    }

    printf("Variance: %f\n", variance);

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}