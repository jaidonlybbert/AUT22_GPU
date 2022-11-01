
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION // this is needed
#include "../util/stb_image.h"  // download from class website files
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../util/stb_image_write.h"  // download from class website files

// #include your error-check macro header file here
#include "../util/cuda_helpers.h"

//#ifndef __CUDACC__  
//#define __CUDACC__
//#endif
//#include <device_functions.h>

// global gaussian blur filter coefficients array here
#define BLUR_FILTER_WIDTH 9  // 9x9 (square) Gaussian blur filter
const float BLUR_FILT[81] = { 0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084,0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,0.3292,0.5353,0.7575,0.9329,1.0000,0.9329,0.7575,0.5353,0.3292,0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084};

// DEFINE your CUDA blur kernel function(s) here
// blur kernel #1 - global memory only
__global__ void blurKernelGlobalMemory(unsigned char* imgData, unsigned char* imgOut, float* blurFilt, int imgWidth, int imgHeight)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int filtPadding = (BLUR_FILTER_WIDTH - 1) / 2;

    if (col < imgWidth && row < imgHeight) {
        float pixFloatVal = 0.0;
        float pixNormalizeFactor = 0.0;
        int pixVal = 0;
        int pixels = 0;

        // Get the weighted average of the surrounding pixels using the gaussian blur filter
        for (int blurRow = -filtPadding; blurRow < filtPadding + 1; ++blurRow) {
            for (int blurCol = -filtPadding; blurCol < filtPadding + 1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                // Verify we have a valid image pixel
                if (curRow > -1 && curRow < imgHeight && curCol > -1 && curCol < imgWidth) {
                    pixFloatVal += (float)(imgData[curRow * imgWidth + curCol] * blurFilt[(blurRow+filtPadding) * BLUR_FILTER_WIDTH + blurCol+filtPadding]);
                    pixNormalizeFactor += blurFilt[(blurRow+filtPadding) * BLUR_FILTER_WIDTH + blurCol+filtPadding]; // Accumulate a factor to normalize by
                }
            }
        }
        // Write our new pixel value out
        imgOut[row * imgWidth + col] = (unsigned char)(int)(pixFloatVal / pixNormalizeFactor);
    }
}

// blur kernel #2 - device shared memory (static alloc)
__global__ void blurKernelStaticMemory(unsigned char* imgData, unsigned char* imgOut, float* blurFilt, int imgWidth, int imgHeight)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int filtPadding = (BLUR_FILTER_WIDTH - 1) / 2;

    // Copy filter coefficients from global -> shared memory using first 81 threads of the block
    __shared__ float ds_blurFilt[BLUR_FILTER_WIDTH][BLUR_FILTER_WIDTH];
    if (threadIdx.x < 9 && threadIdx.y < 9) {
        ds_blurFilt[threadIdx.y][threadIdx.x] = blurFilt[threadIdx.y * BLUR_FILTER_WIDTH + threadIdx.x];
    }
    __syncthreads();

    // Apply the filter to the image
    if (col < imgWidth && row < imgHeight) {
        float pixFloatVal = 0.0;
        float pixNormalizeFactor = 0.0;
        int pixVal = 0;
        int pixels = 0;

        // Get the weighted average of the surrounding pixels using the gaussian blur filter
        for (int blurRow = -filtPadding; blurRow < filtPadding + 1; ++blurRow) {
            for (int blurCol = -filtPadding; blurCol < filtPadding + 1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                // Verify we have a valid image pixel
                if (curRow > -1 && curRow < imgHeight && curCol > -1 && curCol < imgWidth) {
                    pixFloatVal += (float)(imgData[curRow * imgWidth + curCol] * ds_blurFilt[blurRow+filtPadding][blurCol+filtPadding]);
                    pixNormalizeFactor += ds_blurFilt[blurRow+filtPadding][blurCol+filtPadding]; // Accumulate a factor to normalize by
                }
            }
        }
        // Write our new pixel value out
        imgOut[row * imgWidth + col] = (unsigned char)(int)(pixFloatVal / pixNormalizeFactor);
    }
}


// blur kernel #2 - device shared memory (dynamic alloc)
extern __shared__ float s_blurFilt[];

__global__ void blurKernelDynamicMemory(unsigned char* imgData, unsigned char* imgOut, float* blurFilt, int imgWidth, int imgHeight)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int filtPadding = (BLUR_FILTER_WIDTH - 1) / 2;

    // Copy filter coefficients from global -> shared memory using first 81 threads of the block
    if (threadIdx.x < 9 && threadIdx.y < 9) {
        s_blurFilt[threadIdx.y * BLUR_FILTER_WIDTH + threadIdx.x] = blurFilt[threadIdx.y * BLUR_FILTER_WIDTH + threadIdx.x];
    }
    __syncthreads();

    // Apply the filter to the image
    if (col < imgWidth && row < imgHeight) {
        float pixFloatVal = 0.0;
        float pixNormalizeFactor = 0.0;
        int pixVal = 0;
        int pixels = 0;

        // Get the weighted average of the surrounding pixels using the gaussian blur filter
        int curRow = 0;
        int curCol = 0;
        for (int blurRow = -filtPadding; blurRow < filtPadding + 1; ++blurRow) {
            for (int blurCol = -filtPadding; blurCol < filtPadding + 1; ++blurCol) {
                curRow = row + blurRow;
                curCol = col + blurCol;
                // Verify we have a valid image pixel
                if (curRow > -1 && curRow < imgHeight && curCol > -1 && curCol < imgWidth) {
                    pixFloatVal += (float)(imgData[curRow * imgWidth + curCol] * s_blurFilt[(blurRow+filtPadding) * BLUR_FILTER_WIDTH + blurCol+filtPadding]);
                    pixNormalizeFactor += s_blurFilt[(blurRow+filtPadding) * BLUR_FILTER_WIDTH + blurCol+filtPadding]; // Accumulate a factor to normalize by
                }
            }
        }
        // Write our new pixel value out
        imgOut[row * imgWidth + col] = (unsigned char)(int)(pixFloatVal / pixNormalizeFactor);
    }
}


// EXTRA CREDIT
// define host sequential blur-kernel routine


int main()
{
    // read input image from file - be aware of image pixel bit-depth and resolution (horiz x vertical)
    const char filename[] = "../util/hw2_testimage1.png";
    int x_cols = 0;
    int y_rows = 0;
    int n_pixdepth = 0;
    unsigned char* h_imgData = stbi_load(filename, &x_cols, &y_rows, &n_pixdepth, 1);
    int imgSize = x_cols * y_rows * (int)sizeof(unsigned char);
    int imgWidth = x_cols;
    int imgHeight = y_rows;

    // setup additional host variables, allocate host memory as needed
    cudaError_t cudaStatus;
    unsigned char* h_imgOut = (unsigned char*)malloc(imgSize);

    // setup timers
    cudaEvent_t timer1_start, timer1_end, timer2_start, timer2_end;
    float timer1_elapsed, timer2_elapsed;
    cudaStatus = cudaEventCreate(&timer1_start);
    cudaStatus = cudaEventCreate(&timer2_start);
    cudaStatus = cudaEventCreate(&timer1_end);
    cudaStatus = cudaEventCreate(&timer2_end);

    // START timer #1
    cudaStatus = cudaEventRecord(timer1_start);

    // allocate device memory
    unsigned char* dev_imageData = 0;
    unsigned char* dev_imageOut = 0;
    float* dev_blurFilt = 0;
    
    cudaStatus = cudaMalloc((void**)&dev_imageData, imgSize);
    cudaStatus = cudaMalloc((void**)&dev_imageOut, imgSize);
    cudaStatus = cudaMalloc((void**)&dev_blurFilt, 81 * sizeof(float));

    // copy host data to device
    cudaStatus = cudaMemcpy(dev_imageData, h_imgData, imgSize, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_imageOut, h_imgOut, imgSize, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_blurFilt, &BLUR_FILT[0], 81 * sizeof(float), cudaMemcpyHostToDevice);

    // START timer #2
    cudaStatus = cudaEventRecord(timer2_start);

    // launch kernel --- use appropriate heuristics to determine #threads/block and #blocks/grid to ensure coverage of your 2D data range
    dim3 DimGrid(imgWidth / 16 + 1, imgHeight / 16 + 1, 1);
    dim3 DimBlock(16, 16, 1);
    
    blurKernelStaticMemory<<<DimGrid, DimBlock>>>(dev_imageData, dev_imageOut, dev_blurFilt, imgWidth, imgHeight);

    // Check for any errors launching the kernel
    cudaStatus = checkCuda(cudaGetLastError());
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // call cudaDeviceSynchronize() to wait for the kernel to finish, and return
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        goto Error;
    }
    
    // STOP timer #2
    cudaStatus = cudaEventRecord(timer2_end);
    // retrieve result data from device back to host
    cudaStatus = cudaMemcpy(h_imgOut, dev_imageOut, imgSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // STOP timer #1
    cudaStatus = cudaEventRecord(timer1_end);

    // cudaDeviceReset( ) must be called in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    // save result output image data to file

    const char imgFileOut[] = "../util/hw2_outimage1.png";
    stbi_write_png(imgFileOut, x_cols, y_rows, 1, h_imgOut, x_cols * n_pixdepth);


    // EXTRA CREDIT:
    // start timer #3
    // run host sequential blur routine
    // stop timer #3

    // retrieve and save timer results (write to console or file)
    cudaStatus = cudaEventSynchronize(timer1_end);
    cudaStatus = cudaEventSynchronize(timer2_end);
    cudaStatus = cudaEventElapsedTime(&timer1_elapsed, timer1_start, timer1_end);
    cudaStatus = cudaEventElapsedTime(&timer2_elapsed, timer2_start, timer2_end);
    cudaStatus = cudaEventDestroy(timer1_start);
    cudaStatus = cudaEventDestroy(timer2_start);
    cudaStatus = cudaEventDestroy(timer1_end);
    cudaStatus = cudaEventDestroy(timer2_end);
    fprintf(stderr, "Timer 1 elapsed: %2f\n", timer1_elapsed);
    fprintf(stderr, "Timer 2 elapsed: %2f\n", timer2_elapsed);
 
Error:  // assumes error macro has a goto Error statement

    // free host and device memory
    cudaFree(dev_blurFilt);
    cudaFree(dev_imageData);
    cudaFree(dev_imageOut);

    return 0;
}


