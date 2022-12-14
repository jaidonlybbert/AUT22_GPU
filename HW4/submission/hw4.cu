
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Util/cuda_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_HEIGHT 32
#define BLOCK_WIDTH 32
#define MATRX_SIZE 128

float max_error(float* dev_result, float* host_result, int size);
void print_results(float globalError, float sharedError, float cornerTurningError);

/*
    4 MMULT VARIANTS
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

    __shared__ float ads[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ float bds[BLOCK_HEIGHT][BLOCK_WIDTH];

    int ty = threadIdx.y, tx = threadIdx.x;

    int phases = ceil(a_width / (float)BLOCK_WIDTH);

    float pval = 0.0;
    for (int i = 0; i < phases; i++) {
        if ((i * BLOCK_WIDTH + tx < a_width) && (row < a_height)) {
            ads[ty][tx] = a_mtx[row * a_width + i * BLOCK_WIDTH + tx];
        }

        if ((i * BLOCK_WIDTH + ty < a_width) && (col < b_width)) {
            bds[ty][tx] = b_mtx[(i * BLOCK_WIDTH + ty) * b_width + col];
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_WIDTH; k++) {
            if ((i * BLOCK_WIDTH + k) < a_width)
                pval += ads[ty][k] * bds[k][tx];
        }
        __syncthreads();
    }

    if (col < b_width && row < a_height) {
        c_mtx[row * b_width + col] = pval;
    }
}

__global__ void corner_turning_mmult(float* c_mtx, float* a_mtx, float* b_mtx, int a_width, int a_height, int b_t_height)
/*
* Computes result c matrix from the matrix multiplication C = AB using shared memory and "corner turning"
*
* Assumed matrix B has been transposed in memory, so columns are now rows.
*
* Dimensions of C are a_height x b_t_height
*/
{
    // row and column of C result
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float ads[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ float bds[BLOCK_HEIGHT][BLOCK_WIDTH];

    int ty = threadIdx.y, tx = threadIdx.x;

    int phases = ceil(a_width / (float)BLOCK_WIDTH);

    float pval = 0.0;
    for (int i = 0; i < phases; i++) {
        if ((i * BLOCK_WIDTH + tx < a_width) && (row < a_height)) {
            ads[ty][tx] = a_mtx[row * a_width + i * BLOCK_WIDTH + tx];
        }

        if ((i * BLOCK_WIDTH + tx < a_width) && (blockIdx.x * blockDim.x + ty < b_t_height)) {
            bds[ty][tx] = b_mtx[(blockIdx.x * blockDim.x + ty) * a_width + i * BLOCK_WIDTH + tx];
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_WIDTH; k++) {
            if ((i * BLOCK_WIDTH + k) < a_width)
                pval += ads[ty][k] * bds[tx][k];
        }
        __syncthreads();
    }

    if (col < b_t_height && row < a_height) {
        c_mtx[row * b_t_height + col] = pval;
    }
}


__global__ void transpose(float* n_transpose, float* n_mtx, int n_width, int n_height)
/*
* Transposes matrix N in global memory
*
* E.g. column-major becomes row-major, and vice versa
*
*/
{
    // row and column of N transpose
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < n_width) && (col < n_height)) {
        n_transpose[row * MATRX_SIZE + col] = n_mtx[col * MATRX_SIZE + row];
    }
}


int main()
{
    const int A_HEIGHT = MATRX_SIZE, A_WIDTH = MATRX_SIZE, B_HEIGHT = MATRX_SIZE, B_WIDTH = MATRX_SIZE;

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

    host_mmult(h_verify_res, h_a, h_b, A_WIDTH, A_HEIGHT, B_WIDTH);

    // Device variables
    checkCuda(cudaSetDevice(0));

    float* dev_c = 0, * dev_a = 0, * dev_b = 0, * dev_b_transpose = 0;

    checkCuda(cudaMalloc((void**)&dev_c, A_HEIGHT * B_WIDTH * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_a, A_HEIGHT * A_WIDTH * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_b, B_HEIGHT * B_WIDTH * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_b_transpose, B_HEIGHT * B_WIDTH * sizeof(float)));
    checkCuda(cudaMemcpy(dev_c, h_c, A_HEIGHT * B_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_a, h_a, A_HEIGHT * A_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_b, h_b, B_HEIGHT * B_WIDTH * sizeof(float), cudaMemcpyHostToDevice));

    // Setup grid and launch kernel
    dim3 DimGrid(ceil(B_WIDTH / (float)BLOCK_WIDTH), ceil(A_HEIGHT / (float)BLOCK_HEIGHT), 1); dim3 DimBlock(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    global_mem_mmult << <DimGrid, DimBlock >> > (dev_c, dev_a, dev_b, A_WIDTH, A_HEIGHT, B_WIDTH);
    checkCuda(cudaGetLastError()); // Check for any errors launching the kernel
    checkCuda(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(h_c, dev_c, A_HEIGHT * B_WIDTH * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results to host calcs
    float maxErrorGlobal = max_error(h_verify_res, h_c, A_HEIGHT * B_WIDTH);

    shared_mem_mmult<<<DimGrid, DimBlock>>>(dev_c, dev_a, dev_b, A_WIDTH, A_HEIGHT, B_WIDTH);
    checkCuda(cudaGetLastError()); // Check for any errors launching the kernel
    checkCuda(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(h_c, dev_c, A_HEIGHT * B_WIDTH * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results to host calcs
    float maxErrorShared = max_error(h_verify_res, h_c, A_HEIGHT * B_WIDTH);

    transpose << <DimGrid, DimBlock >> > (dev_b_transpose, dev_b, B_WIDTH, B_HEIGHT);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    corner_turning_mmult << <DimGrid, DimBlock >> > (dev_c, dev_a, dev_b_transpose, A_WIDTH, A_HEIGHT, B_WIDTH);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(h_c, dev_c, A_HEIGHT * B_WIDTH * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results to host calcs
    float maxErrorCornerTurning = max_error(h_verify_res, h_c, A_HEIGHT * B_WIDTH);

    print_results(maxErrorGlobal, maxErrorShared, maxErrorCornerTurning);

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

void print_results(float globalError, float sharedError, float cornerTurningError) {
    if (globalError < 1E-6) {
        printf("Global Mem. host verification passed.\n");
    }
    else {
        printf("Global Mem. host verification failed.\n");
    }

    if (sharedError < 1E-6) {
        printf("Shared Mem. host verification passed.\n");
    }
    else {
        printf("Shared Mem. host verification failed.\n");
    }

    if (cornerTurningError < 1E-6) {
        printf("Corner turning host verification passed.\n");
    }
    else {
        printf("Corner turning host verification failed.\n");
    }

    printf("Max error global memory: %.3E\n", globalError);
    printf("Max error shared memory: %.3E\n", sharedError);
    printf("Max error corner turning: %.3E\n", cornerTurningError);
}