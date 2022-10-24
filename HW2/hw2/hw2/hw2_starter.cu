
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

// global gaussian blur filter coefficients array here
#define BLUR_FILTER_WIDTH 9  // 9x9 (square) Gaussian blur filter
const float BLUR_FILT[81] = { 0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084,0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,0.3292,0.5353,0.7575,0.9329,1.0000,0.9329,0.7575,0.5353,0.3292,0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084};

// DEFINE your CUDA blur kernel function(s) here
// blur kernel #1 - global memory only
// blur kernel #2 - device shared memory (static alloc)
// blur kernel #2 - device shared memory (dynamic alloc)


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

    // setup additional host variables, allocate host memory as needed
    cudaError_t cudaStatus;
    unsigned char* h_imgOut = (unsigned char*)malloc(imgSize);

    // START timer #1

    // allocate device memory
    unsigned char* dev_imageData = 0;
    float* dev_blurFilt = 0;
    
    cudaStatus = cudaMalloc((void**)&dev_imageData, imgSize);
    cudaStatus = cudaMalloc((void**)&dev_blurFilt, 81 * sizeof(float));

    // copy host data to device
    cudaStatus = cudaMemcpy(dev_imageData, h_imgData, imgSize, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_blurFilt, &BLUR_FILT[0], 81 * sizeof(float), cudaMemcpyHostToDevice);

    // START timer #2
    // launch kernel --- use appropriate heuristics to determine #threads/block and #blocks/grid to ensure coverage of your 2D data range

    // Check for any errors launching the kernel
    cudaStatus = checkCuda(cudaGetLastError());
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // call cudaDeviceSynchronize() to wait for the kernel to finish, and return
    // any errors encountered during the launch.
    
    // STOP timer #2
    // 
    // retrieve result data from device back to host

    // STOP timer #1

    // cudaDeviceReset( ) must be called in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    // save result output image data to file
    const char imgFileOut[] = "c:/Temp/AUT22_GPUCompute/dat/hw2_outimage1.png";
    stbi_write_png(imgFileOut, x_cols, y_rows, 1, h_imgOut, x_cols * n_pixdepth);


    // EXTRA CREDIT:
    // start timer #3
    // run host sequential blur routine
    // stop timer #3

    // retrieve and save timer results (write to console or file)
 
Error:  // assumes error macro has a goto Error statement

    // free host and device memory

    return 0;
}


