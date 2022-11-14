
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Util/cuda_helpers.h"
#include <stdio.h>
#include <stdlib.h>

cudaError_t executeCuda(int width);

/*
    3 KERNELS FOR TESTING EFFECTS OF DIVERGENCE
*/

__global__ void no_divergence(int* res, int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int temp = col;

    if (width % 2 == 0) {
        temp += row * 10;
    }

    __syncthreads();
    
    res[row * width + col] = temp;

    return;
}

__global__ void single_branch_divergence(int *res, int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int temp = col;

    if (temp % 2 == 0) {
        temp += row * 10;
    }
    else {
        temp += row * 1000;
    }

    __syncthreads();

    res[row * width + col] = temp;
    
    return;
}

__global__ void nested_branch_divergence(int* res, int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int temp = col;

    if (temp % 2 == 0) { // do for every other column
        temp += row * 10;
        if (temp % 3 == 0) { // do for every 6th column
            temp += row*1000;
        }
    }

    __syncthreads();

    res[row * width + col] = temp;

    return;
}

int main()
{
    int width = 36;

    // Run kernel
    checkCuda(executeCuda(width));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCuda(cudaDeviceReset());

    return 0;
}

cudaError_t executeCuda(int width)
{
    int *res = (int *)malloc(width*width*sizeof(int));
    int *dev_res = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCuda(cudaSetDevice(0));
    // Allocate GPU memory for variable
    checkCuda(cudaMalloc((void**)&dev_res, width*width*sizeof(int)));
    // Copy result from host memory to GPU buffers.
    checkCuda(cudaMemcpy(dev_res, res, width*width*sizeof(int), cudaMemcpyHostToDevice));

    // Launch a kernel on the GPU with one thread for each element.
    dim3 DimGrid(23, 36, 1); dim3 DimBlock(16, 16, 1);
    no_divergence<<<DimGrid, DimBlock >>>(dev_res, width);
    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());

    single_branch_divergence << <DimGrid, DimBlock >> > (dev_res, width);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    nested_branch_divergence << <DimGrid, DimBlock >> > (dev_res, width);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(res, dev_res, width*width*sizeof(int), cudaMemcpyDeviceToHost));


Error:
    cudaFree(dev_res);
    
    return cudaStatus;
}
