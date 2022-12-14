/*
* Author: Jaidon Lybbert
* Date:   12/16/2022
*
* CUDA implementation of the SIFT object detection algorithm
*
* Based on the original paper by David G. Lowe
*	D. G. Lowe, "Object recognition from local scale-invariant features,"
*	Proceedings of the Seventh IEEE International Conference on Computer Vision, 1999,
*	pp. 1150-1157 vol.2, doi: 10.1109/ICCV.1999.790410.
*
* Freely available: https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf
*
* Directory structure assumed:
* ..
*	\
*	|- util
*		\
*		| stb_image.h
*		| stb_image_write.h
*		| test_image.png
*	|- Layers
*		\
*		| empty
*	|- SIFT
*		\
*		| sift_host.cpp
*		| sift_host.h
*		| sift_device.cu
*       | host_main.cpp (this file)
*/

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#ifndef __CUDACC__  
//#define __CUDACC__
//#endif
//#include <device_functions.h>

// Std IO
#include <stdio.h>

// Image library
#define STB_IMAGE_IMPLEMENTATION 
#include "../util/stb_image.h"  
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../util/stb_image_write.h"

// Host functions
#include "sift_host.h"

// Device function declarations
#include "sift_device.cuh"
#include "../util/cuda_helpers.h"

// Execution parameters
#define BILINEAR_INTERPOLATE_BLOCK_WIDTH 32
#define BLUR_BLOCK_WIDTH 32

int main()
{
    cudaError_t cudaStatus;

    // 0) Load input image into an array -> 'inputArray'
    // read input image from file
    const char filename[] = "../util/test_image.png";
    int x_cols = 0;
    int y_rows = 0;
    int n_pixdepth = 0;
    unsigned char* h_rawImage = stbi_load(filename, &x_cols, &y_rows, &n_pixdepth, 1);
    int imgSize = x_cols * y_rows * (int)sizeof(unsigned char);
    int imgWidth = x_cols;
    int imgHeight = y_rows;

    // Copy image to device
    unsigned char* dev_rawImage = 0;
    unsigned char* dev_InputImage = 0;
    checkCuda(cudaMalloc((void**)&dev_rawImage, imgSize));
    checkCuda(cudaMemcpy(dev_rawImage, h_rawImage, imgSize, cudaMemcpyHostToDevice));

    // 1) Expand the raw image via bilinear interpolation -> 'dev_InputImage'
    //      This expanded image is the 'input' top layer of the image pyramid
    //      The image is expanded first so that high-resolution details aren't missed
    
    // Create the expanded image space
    int resultWidth = 0;
    int resultHeight = 0;

    // Call parallel bilinear interpolate on device
    dev_bilinear_interpolate(dev_rawImage, &dev_InputImage, imgWidth, imgHeight, 2.0, &resultWidth, &resultHeight);

    // Create the array of pyramid layers
    imagePyramidLayer pyramid[LAYERS];
    // This template is used to iteratively export layers as images, by replacing the placeholder 'X' with the layer index
    const char directoryTemplate[] = "../Layers/AX.png";

    // For each layer in pyramid, construct two images with increasing levels of blur, and compute the difference of gaussian (DoG)
    // Apply bilinear interpolation to the second of the two blurred images, and use that in the next iteration as the input image
    // Keep these images in DEVICE memory, and reference them via the pointers in the 'imagePyramidLayer' structs
    for (int i = 0; i < 1; i++) {
        // Copy directory name to layer
        for (int j = 0; j < 17; j++) {
            pyramid[i].layerOutputDir[j] = directoryTemplate[j];
        }
        pyramid[i].layerOutputDir[10] = (char)'A';
        pyramid[i].layerOutputDir[11] = (char)('0' + (char)i);
        pyramid[i].height = resultHeight;
        pyramid[i].width = resultWidth;

        // 2) Apply gaussian blur to the input image -> 'A'
        // Allocate device memory for blurred image 'A'
        checkCuda(cudaMalloc(&pyramid[i].imageA, resultWidth * resultHeight * (int)sizeof(unsigned char)));
        // Parameters for generating gaussian kernel
        float sigma = sqrt(2);
        int kernelWidth = 9;
        // Call device kernel for guassian blur
        dev_gaussian_blur(dev_InputImage, pyramid[i].imageA, resultWidth, resultHeight, sigma, kernelWidth);
    }

        // Allocate memory for blurred image A, B and DoG on device

        // execute gaussian blur kernel -> image A

        // execute gaussian blur kernel -> image B

        // execute matrix subtract kernel -> DoG

        // Allocate memory for expanded image for next layer

        // execute bilinear interpolate kernel

    // Allocate memory for gradient maps on device

    // Execute gradient map kernel

    // Allocate memory for pyramid on host

    // Copy pyramid to host

    // Execute host-side extrema indexing function to build keypoint list

    // Execute host-side keypoint characterize function

    // Execute host-side annotation function

    // Load host-verification annotated image

    // Compare device and host annotated image pix by pix for verification

    // return status

    return 0;
}

void dev_gaussian_blur(unsigned char* img, unsigned char* outputArray, int inputWidth, int inputHeight, float sigma, int kernelWidth) {
    /*
    * Convolves inputArray with a Gaussian kernel defined by g(x) = 1 / (sqrt(2*pi) * sigma) * exp(-(x**2) / (2 * sigma**2))
    *
    * kernelWidth is assumed to be odd.
    */

    assert(kernelWidth % 2 == 1);

    int kernelRadius = floor(kernelWidth / 2);

    // Generate gaussian kernel
    float* kernel = NULL;
    float* dev_kernel = NULL;
    get_gaussian_kernel(&kernel, sigma, kernelWidth);

    // Move gaussian kernel to device
    checkCuda(cudaMalloc(&dev_kernel, kernelWidth * kernelWidth * (int)sizeof(float)));
    checkCuda(cudaMemcpy(dev_kernel, kernel, kernelWidth * kernelWidth * (int)sizeof(float), cudaMemcpyHostToDevice));

    // Setup device grid
    dim3 DimGrid(inputHeight / BLUR_BLOCK_WIDTH + 1, inputWidth / BLUR_BLOCK_WIDTH + 1, 1);
    dim3 DimBlock(BLUR_BLOCK_WIDTH, BLUR_BLOCK_WIDTH, 1);

    gaussian_blur_kernel << <DimGrid, DimBlock, kernelWidth* kernelWidth * (int)sizeof(float) >> > (img, outputArray, dev_kernel, inputHeight, inputWidth, kernelWidth);
    checkCuda(cudaDeviceSynchronize());

    // DEBUG OUTPUT
    unsigned char* blurred = (unsigned char*)malloc(inputHeight * inputWidth * (int)sizeof(unsigned char));
    checkCuda(cudaMemcpy(blurred, outputArray, inputHeight * inputWidth * (int)sizeof(unsigned char), cudaMemcpyDeviceToHost));
    const char directoryTemplate[] = "../Layers/BlurredA.png";
    stbi_write_png(directoryTemplate, inputWidth, inputHeight, 1, blurred, inputWidth * 1);
    
    free(kernel);
}

void dev_bilinear_interpolate(unsigned char* inputArray, unsigned char** dev_ExpandedImage, int inputWidth, int inputHeight,
    float spacing, int* resultWidth, int* resultHeight) {
    /*
    * Applies bilinear interpolation to inputArray with spacing 'spacing', and stores result in 'inputArrayExpanded'.
    *
    * Pixels of the output image are calculated according to the equation: P(x,y) = R1 * (y2-y)/(y2-y1) + R2 * (y-y1)/(y2-y1)
    */
    assert(inputArray != NULL);
    assert(dev_ExpandedImage != NULL);

    // Create the expanded image space
    int expandedHeight = ceil(inputHeight * spacing);
    int expandedWidth = ceil(inputWidth * spacing);
    int expandedLength = expandedHeight * expandedWidth;
    checkCuda(cudaMalloc((void **)dev_ExpandedImage, expandedLength * (int)sizeof(unsigned char)));

    // Report the new width and height back to the main loop, for future use
    *resultWidth = expandedWidth;
    *resultHeight = expandedHeight;

    // Launch device kernel
    dim3 DimGrid(expandedHeight / BILINEAR_INTERPOLATE_BLOCK_WIDTH + 1, expandedWidth / BILINEAR_INTERPOLATE_BLOCK_WIDTH + 1, 1);
    dim3 DimBlock(BILINEAR_INTERPOLATE_BLOCK_WIDTH, BILINEAR_INTERPOLATE_BLOCK_WIDTH, 1);
    int tileWidth = ceil(BILINEAR_INTERPOLATE_BLOCK_WIDTH / spacing) + 1;

    bilinear_interpolate_kernel<<<DimGrid, DimBlock, tileWidth * tileWidth * (int)sizeof(unsigned char) >> >(inputArray, *dev_ExpandedImage, spacing, inputWidth, inputHeight, expandedWidth, expandedHeight, tileWidth);
    checkCuda(cudaDeviceSynchronize());

    // DEBUG OUTPUT
    unsigned char* expanded = (unsigned char*)malloc(expandedLength * sizeof(unsigned char));
    checkCuda(cudaMemcpy(expanded, *dev_ExpandedImage, expandedLength, cudaMemcpyDeviceToHost));
    const char directoryTemplate[] = "../Layers/interpolated.png";
    stbi_write_png(directoryTemplate, expandedWidth, expandedHeight, 1, expanded, expandedWidth * 1);
}


__global__ void bilinear_interpolate_kernel(unsigned char *input, unsigned char *output, float spacing, int inputWidth, int inputHeight, int resultWidth, int resultHeight, int inputTileWidth)
{
    // Dynamically allocate shared memory for image tile
    extern __shared__ unsigned char imageTile[];

    // Calculate row & col relative to output image
    int OutputRow = blockDim.y * blockIdx.y + threadIdx.y;
    int OutputCol = blockDim.x * blockIdx.x + threadIdx.x;

    // Collaboratively load section of input image into shared memory
    int topLeftX = floor((double)(blockDim.x / spacing * blockIdx.x));
    int topLeftY = floor((double)(blockDim.y / spacing * blockIdx.y));

    if (threadIdx.x < inputTileWidth && OutputRow < resultHeight && threadIdx.y < inputTileWidth && OutputCol < resultWidth) {
        imageTile[threadIdx.y * inputTileWidth + threadIdx.x] = input[(topLeftY + threadIdx.y) * inputWidth + topLeftX + threadIdx.x];
    }

    __syncthreads();

    // Coordinates of output pixel relative to the image tile
    float InputCol = OutputCol / spacing - topLeftX;
    float InputRow = OutputRow / spacing - topLeftY;

    // Coordinates of surrounding input image pixels relative to image tile
    float x1 = floor((double)InputCol);
    float x2 = (ceil((double)InputCol) < inputTileWidth) ? ceil((double)InputCol) : x1;

    __syncthreads();

    float y1 = floor((double)InputRow);
    float y2 = (ceil((double)InputRow) < inputTileWidth) ? ceil((double)InputRow) : y1;

    __syncthreads();

    // Pixel values of surrounding input image pixels
    unsigned char Q11, Q21, Q12, Q22;
    float R1, R2;
    if (OutputRow < resultHeight && OutputCol < resultWidth) {
        unsigned char Q11 = imageTile[(int)y1 * inputTileWidth + (int)x1];
        unsigned char Q21 = imageTile[(int)y1 * inputTileWidth + (int)x2];
        unsigned char Q12 = imageTile[(int)y2 * inputTileWidth + (int)x1];
        unsigned char Q22 = imageTile[(int)y2 * inputTileWidth + (int)x2];

        // Row-wise interpolated values
        // Skip row-wise interpolation if the coordinate lands on a column instead of between columns
        if (InputCol != x1) {
            R1 = Q11 * (x2 - InputCol) / (x2 - x1) + Q21 * (InputCol - x1) / (x2 - x1);
            R2 = Q12 * (x2 - InputCol) / (x2 - x1) + Q22 * (InputCol - x1) / (x2 - x1);
        }
        else {
            R1 = Q11;
            R2 = Q12;
        }

        // Final interpolated value
        // Skip column-wise interpolation if coordinate lands on a row instead of between rows
        if (InputRow != y1) {
            output[OutputRow * resultWidth + OutputCol] = (unsigned char)(R1 * (y2 - InputRow) / (y2 - y1) + R2 * (InputRow - y1) / (y2 - y1));
        }
        else {
            output[OutputRow * resultWidth + OutputCol] = (unsigned char)R1;
        }
    }
}

__global__ void gaussian_blur_kernel(unsigned char* input, unsigned char* output, float* gaussianKernel, int height, int width, int kernelDim) {
    // Dynamically allocate memory for blur kernel
    extern __shared__ float s_blurFilt[];
    
    // Compute thread coordinates
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int filtPadding = (kernelDim - 1) / 2;

    // Copy filter coefficients from global -> shared memory using first threads of the block
    if (threadIdx.x < kernelDim && threadIdx.y < kernelDim) {
        s_blurFilt[threadIdx.y * kernelDim + threadIdx.x] = gaussianKernel[threadIdx.y * kernelDim + threadIdx.x];
    }
    __syncthreads();

    // Apply the filter to the image
    if (col < width && row < height) {
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
                if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    pixFloatVal += (float)(input[curRow * width + curCol] * s_blurFilt[(blurRow + filtPadding) * kernelDim + blurCol + filtPadding]);
                    pixNormalizeFactor += s_blurFilt[(blurRow + filtPadding) * kernelDim + blurCol + filtPadding]; // Accumulate a factor to normalize by
                }
            }
        }
        // Write our new pixel value out
        output[row * width + col] = (unsigned char)(int)(pixFloatVal / pixNormalizeFactor);
    }
}

__global__ void matrix_subtract_kernel(unsigned char* A, unsigned char* B, unsigned char* C, int height, int width) {
    // Subtract corresponding values in A and B and store in C
}

__global__ void gradient_map_kernel(unsigned char* input, float* magnitudeMap, float* orientationMap, int height, int width) {
    // collaboratively load input into shared memory
    // store right pixval in register
    // store below pixval in register
    // check boundaries
        // compute gradient magnitude and store in magnitudemap
        // compute gradient orientation and store in orientationmap
}