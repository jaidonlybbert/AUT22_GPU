
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Util/cuda_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TILE_WIDTH 64
#define BLOCK_HEIGHT 32
#define BLOCK_WIDTH 8

float max_error(float* dev_result, float* host_result, int size);

/*
    3 MMULT VARIANTS
*/
void host_mmult(float* c_mtx, float* a_mtx, float* b_mtx, int a_width, int a_height, int b_width)
/*
* Computes result c matrix from the matrix multiplication C = AB with the CPU
* 
* Assumed a_width == b_height
* 
* Dimensions of C are a_height x b_width
*/
{
    // For each position in the C matrix
    for (int i = 0; i < a_height; i++) {
        for (int j = 0; j < b_width; j++) {
            // Calculate the inner product of the row of A and column of B
            float innerProduct = 0;
            for (int k = 0; k < a_width; k++) {
                innerProduct += a_mtx[a_width * i + k] * b_mtx[b_width * k + j];
            }
            c_mtx[b_width * i + j] = innerProduct;
        }
    }
}

__global__ void global_mem_mmult(float* c_mtx, float* a_mtx, float* b_mtx, int a_width, int a_height, int b_width)
/*
* Computes result c matrix from the matrix multiplication C = AB using global memory with CUDA
*
* Assumed a_width == b_height
*
* Dimensions of C are a_height x b_width
*/
{
    // row and column of the C result
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < a_height && col < b_width) {
        // Calculate the inner product of the row of A and column of B
        float innerProduct = 0;
        for (int i = 0; i < a_width; i++) {
            innerProduct += a_mtx[a_width * row + i] * b_mtx[b_width * i + col];
        }

        c_mtx[b_width * row + col] = innerProduct;
    }
}


__global__ void shared_mem_mmult(float* c_mtx, float* a_mtx, float* b_mtx, int a_width, int a_height, int b_width)
/*
* Computes result c matrix from the matrix multiplication C = AB using shared memory with CUDA
*
* Assumed a_width == b_height
*
* Dimensions of C are a_height x b_width
*/
{
    // row and column of C result
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float bds[TILE_WIDTH][TILE_WIDTH];

    int ty = threadIdx.y, tx = threadIdx.x;

    int phases = ceil(a_width / (float)TILE_WIDTH);

    float pval = 0.0;
    for (int i = 0; i < phases; i++) {
        if ((i * TILE_WIDTH + tx < a_width) && (row < a_height)) {
            ads[ty][tx] = a_mtx[row * a_width + i * TILE_WIDTH + tx];
        }

        if ((i * TILE_WIDTH + ty < a_width) && (col < b_width)) {
            bds[ty][tx] = b_mtx[(i * TILE_WIDTH + ty) * b_width + col];
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            if ((i * TILE_WIDTH + k) < a_width)
                pval += ads[ty][k] * bds[k][tx];
        }
        __syncthreads();
    }

    if (col < b_width && row < a_height) {
        c_mtx[row * b_width + col] = pval;
    }
}


int main()
{
    const int A_HEIGHT = 2000, A_WIDTH = 2500, B_HEIGHT = 2500, B_WIDTH = 3000;

    // Host variables
    float* h_c = (float*)malloc(A_HEIGHT * B_WIDTH * sizeof(float));
    float* h_a = (float*)malloc(A_WIDTH * A_HEIGHT * sizeof(float));
    float* h_b = (float*)malloc(B_WIDTH * B_HEIGHT * sizeof(float));
    float* h_verify_res = (float*)malloc(A_HEIGHT * B_WIDTH * sizeof(float));

    // Seed randomizer to generate input data
    srand((unsigned int)time(NULL));

    // Initialize input arrays
    for (int i = 0; i < A_WIDTH * A_HEIGHT; i++) { h_a[i] = ((float)rand() / (float)(RAND_MAX)); }
    for (int i = 0; i < B_WIDTH * B_HEIGHT; i++) { h_b[i] = ((float)rand() / (float)(RAND_MAX)); }

    // Device variables
    checkCuda(cudaSetDevice(0));

    float* dev_c = 0, * dev_a = 0, * dev_b = 0;

    checkCuda(cudaMalloc((void**)&dev_c, A_HEIGHT * B_WIDTH * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_a, A_HEIGHT * A_WIDTH * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_b, B_HEIGHT * B_WIDTH * sizeof(float)));
    checkCuda(cudaMemcpy(dev_c, h_c, A_HEIGHT * B_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_a, h_a, A_HEIGHT * A_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_b, h_b, B_HEIGHT * B_WIDTH * sizeof(float), cudaMemcpyHostToDevice));

    // Setup grid and launch kernel
    dim3 DimGrid(ceil(B_WIDTH / (float)BLOCK_WIDTH), ceil(A_HEIGHT / (float)BLOCK_HEIGHT), 1); dim3 DimBlock(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    global_mem_mmult<<<DimGrid, DimBlock>>>(dev_c, dev_a, dev_b, A_WIDTH, A_HEIGHT, B_WIDTH);
    checkCuda(cudaGetLastError()); // Check for any errors launching the kernel
    checkCuda(cudaDeviceSynchronize());

    //shared_mem_mmult<<<DimGrid, DimBlock>>>(dev_c, dev_a, dev_b, A_WIDTH, A_HEIGHT, B_WIDTH);
    //checkCuda(cudaGetLastError()); // Check for any errors launching the kernel
    //checkCuda(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(h_c, dev_c, A_HEIGHT * B_WIDTH * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute host-side verification result
    host_mmult(h_verify_res, h_a, h_b, A_WIDTH, A_HEIGHT, B_WIDTH);

    // Compare results
    float maxError = max_error(h_verify_res, h_c, A_HEIGHT * B_WIDTH);

    if (maxError < 1E-6) { 
        printf("Host verification passed.\n");
    } else {
        printf("Host verification failed.\n");
    }

    printf("Max error: %.3E\n", maxError);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCuda(cudaDeviceReset());

Error:
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}


float max_error(float* dev_result, float* host_result, int size) {
    float maxError = 0.0;
    for (int i = 0; i < size; i++) {
        if (fabs(host_result[i] - dev_result[i]) / host_result[i] > maxError) {
            maxError = fabs((host_result[i] - dev_result[i])) / host_result[i];
        }
    }
    return maxError;
}