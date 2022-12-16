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
#include <stdio.h>
#include <cmath>

// Host functions
#include "sift_host.h"

// Device function declarations
#include "sift_device.cuh"
#include "../util/cuda_helpers.h"

// Execution parameters
#define BILINEAR_INTERPOLATE_BLOCK_WIDTH 32
#define BLUR_BLOCK_WIDTH 32
#define SUBTRACT_BLOCK_WIDTH 32
#define MAP_BLOCK_WIDTH 32
#define HISTOGRAM_KERNEL_WIDTH 5

extern __device__ __constant__ unsigned char dev_KeypointGraphic[289];

void dev_generate_key_masks(imagePyramidLayer pyramid[LAYERS], 
                            unsigned char* dev_OutputImage, int outputWidth, int outputHeight) {
    /*
    * Searches pyramid structure for local extrema or 'keypoints'. Each pixel is evaluated against its nearest 
    * neighbors in scale space.
    *
    * The key mask is initialized as all zeroes and matches the dimensions of the image layer.
    * Locations of keypoints in each layer are marked with a value of 255 in the keypoint mask.
    */
    imagePyramidLayer* dev_Pyramid = NULL;

    checkCuda(cudaMalloc(&dev_Pyramid, LAYERS * (int)sizeof(imagePyramidLayer)));
    checkCuda(cudaMemcpy(dev_Pyramid, pyramid, LAYERS * (int)sizeof(imagePyramidLayer), cudaMemcpyHostToDevice));

    for (int i = 1; i < LAYERS - 1; i++) {
        //printf("\nGenerating key masks for layer %d\n", i);
        // Generate gaussian weight kernel for layer, used in histogram to determine keypoint orientation
        // A gaussian kernel is used for the weighted histogram
        // The 'sigma' parameter is based on the layers level of blur
        float sigma = (3 * sqrt(2) * i * 2);
        int kernelWidth = HISTOGRAM_KERNEL_WIDTH;
        int kernelRadius = floor(kernelWidth / 2);
        float* gaussian_kernel = NULL;
        float* dev_gaussian_kernel = NULL;
        // Get the 2d gaussian kernel
        get_gaussian_kernel(&gaussian_kernel, sigma, kernelWidth);
        // Move kernel to device
        checkCuda(cudaMalloc(&dev_gaussian_kernel, kernelWidth * kernelWidth * (int)sizeof(float)));
        checkCuda(cudaMemcpy(dev_gaussian_kernel, gaussian_kernel, kernelWidth * kernelWidth * (int)(sizeof(float)), 
            cudaMemcpyHostToDevice));
        //printf("- Histogram kernel generated and moved to device.\n");

        dim3 DimGrid(pyramid[i].height / MAP_BLOCK_WIDTH + 1, pyramid[i].width / MAP_BLOCK_WIDTH + 1, 1);
        dim3 DimBlock(MAP_BLOCK_WIDTH, MAP_BLOCK_WIDTH, 1);
        generate_key_mask_kernel<<<DimGrid, DimBlock>>>(dev_Pyramid, dev_gaussian_kernel, i, 
                                                        dev_OutputImage, outputWidth, outputHeight);
        checkCuda(cudaDeviceSynchronize());
        //printf("- Key mask generated.\n");

        //-----------DEBUG OUTPUT------------
        //int* h_keymask = (int*)malloc(pyramid[i].width * pyramid[i].height * sizeof(int));
        //checkCuda(cudaMemcpy(h_keymask, pyramid[i].keyMask, pyramid[i].height * pyramid[i].width * (int)sizeof(int), 
        //                     cudaMemcpyDeviceToHost));
        //int keypointCount = 0;
        //for (int j = 0; j < pyramid[i].height * pyramid[i].width; j++) {
        //    if (h_keymask[j] != -1) {
        //        keypointCount++;
        //    }
        //}
        //free(h_keymask);
        //printf("Layer %d keypoints: %d\n", i, keypointCount);
    }
}

__global__ void generate_key_mask_kernel(imagePyramidLayer pyramid[LAYERS], float* dev_gaussian_kernel,
    int layer, unsigned char* dev_OutputImage, int outputWidth, int outputHeight) {

    unsigned char* dev_DoG = pyramid[layer].DoG;
    unsigned char* dev_DoG_below = pyramid[layer + 1].DoG;
    unsigned char* dev_DoG_above = pyramid[layer - 1].DoG;
    int* dev_orientation_map = pyramid[layer].gradientOrientationMap;
    int* dev_keymask = pyramid[layer].keyMask;
    int width = pyramid[layer].width;
    int height = pyramid[layer].height;

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = row * width + col;

    unsigned char minVal = 255;
    unsigned char maxVal = 0;

    unsigned char neighborPixVal = 0;
    unsigned char pixVal = dev_DoG[idx];

    bool notOnEdge = (row > 0 && row < height - 1 && col > 0 && col < height - 1);

    //------DEBUG OUTPUT--
    //if (row == 0 && col == 0) {
    //    printf("- Comparing against neighbor pixels.\n");
    //}

    // Check pixel value against neighboring pixel values in the same layer
    // To determine if it is an extrema
    if (notOnEdge) {
        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                neighborPixVal = dev_DoG[(row + y) * width + (col + x)];

                if (x != 0 && y != 0) {
                    if (neighborPixVal < minVal) {
                        minVal = neighborPixVal;
                    }
                    else if (neighborPixVal > maxVal) {
                        maxVal = neighborPixVal;
                    }
                }
            }
        }
    }

    __syncthreads();

    // If the pixel value is an extrema of the current layer, check against pixels in layer below
    if (notOnEdge && (pixVal < minVal || pixVal > maxVal)) {

        int xcoordBelow = col * BILINEAR_INTERPOLATION_SPACING;
        int ycoordBelow = row * BILINEAR_INTERPOLATION_SPACING;
        int widthBelow = ceil((double)width * BILINEAR_INTERPOLATION_SPACING);
        int heightBelow = ceil((double)height * BILINEAR_INTERPOLATION_SPACING);

        for (int r = -1; r <= 1; r++) {
            for (int s = -1; s <= 1; s++) {
                if (ycoordBelow + r > 0 && ycoordBelow + r < heightBelow && xcoordBelow + s > 0 &&
                    xcoordBelow + s < widthBelow) {
                    neighborPixVal = dev_DoG_below[(ycoordBelow + r) * widthBelow + (xcoordBelow + s)];
                }

                if (neighborPixVal < minVal) {
                    minVal = neighborPixVal;
                }
                else if (neighborPixVal > maxVal) {
                    maxVal = neighborPixVal;
                }
            }
        }
    }

    __syncthreads();

    // If the pixel value is still an extrema, check against pixels in layer above
    if (notOnEdge && (pixVal < minVal || pixVal > maxVal)) {
        int xcoordAbove = col / BILINEAR_INTERPOLATION_SPACING;
        int ycoordAbove = row / BILINEAR_INTERPOLATION_SPACING;
        int widthAbove = floor((double)width / BILINEAR_INTERPOLATION_SPACING);
        int heightAbove = floor((double)height / BILINEAR_INTERPOLATION_SPACING);

        for (int r = -1; r <= 1; r++) {
            for (int s = -1; s <= 1; s++) {
                if (ycoordAbove + r > 0 && ycoordAbove + r < heightAbove && xcoordAbove + s > 0 &&
                    xcoordAbove + s < widthAbove) {
                    neighborPixVal = dev_DoG_above[(ycoordAbove + r) * widthAbove + (xcoordAbove + s)];
                }

                if (neighborPixVal < minVal) {
                    minVal = neighborPixVal;
                }
                else if (neighborPixVal > maxVal) {
                    maxVal = neighborPixVal;
                }
            }
        }
    }

    //---------DEBUG OUTPUT----
    //if (row == 0 && col == 0) {
    //    for (int m = 0; m < 17 * 17; m++) {
    //        unsigned char val = dev_KeypointGraphic[m];
    //        printf("Pixval: %d\n", (int)val);
    //    }
    //}

    __syncthreads();


    // If the pixel is still an extrema, it is a keypoint.
    // Call child kernel to determine orientation of keypoint.
    if (notOnEdge) {
        if (pixVal < minVal || pixVal > maxVal) {
            dim3 GridDim(1, 1, 1);
            dim3 BlockDim(HISTOGRAM_KERNEL_WIDTH, HISTOGRAM_KERNEL_WIDTH, 1);
            orientation_histogram_kernel << <GridDim, BlockDim >> > (dev_keymask, dev_orientation_map, dev_gaussian_kernel,
                width, height, col, row, idx);
            cudaDeviceSynchronize();
            dim3 BlockDimDraw(17, 17, 1);
            if (layer == 1) {
                draw_keypoint_kernel << <GridDim, BlockDimDraw >> > (dev_KeypointGraphic,
                    dev_OutputImage, dev_keymask, idx, row, col, layer, outputWidth, outputHeight);
            }
        }
        else {
            dev_keymask[idx] = -1; // Negative indicates no keypoint
        }
    }
    else if ((row == 0 || row == height - 1) && (col == 0 || col == height - 1)) {
        dev_keymask[idx] = -1;
    }
}

__global__ void draw_keypoint_kernel(unsigned char* dev_keypoint_template, unsigned char* dev_OutputImage, 
     int* dev_keymask, int idx, int row, int col, int layer, int outputWidth, int outputHeight) {
    // Orients and draws keypoint on output image

    // Scale row and column down to match base layer
    row = row * pow((double)BILINEAR_INTERPOLATION_SPACING, -layer);
    col = col * pow((double)BILINEAR_INTERPOLATION_SPACING, -layer);

    // Pixel value
    int orientation = dev_keymask[idx];
    unsigned char pixVal = dev_keypoint_template[threadIdx.y * 17 + threadIdx.x];
    
    // Coordinates of thread centered at zero for rotation
    float x_in = threadIdx.x - blockDim.x / 2.0;
    float y_in = threadIdx.y - blockDim.y / 2.0;

    // Transform matrix
    __shared__ double rotationMatrix[2][2];

    // Convert orientation to radians
    double orientationRad = (double)(orientation) * (double)PI / 180;

    // Compute rotation matrix coefficients collaboratively
    if (threadIdx.x == 0) {
        if (threadIdx.y == 0) {
            rotationMatrix[0][0] = cos(orientationRad);
        }
        else if (threadIdx.y == 1) {
            rotationMatrix[0][1] = -sin(orientationRad);
        }
        else if (threadIdx.y == 2) {
            rotationMatrix[1][0] = sin(orientationRad);
        }
        else if (threadIdx.y == 3) {
            rotationMatrix[1][1] = cos(orientationRad);
        }
    }
    
    __syncthreads();

    // Output coordinates
    float x_out = (rotationMatrix[0][0] * x_in + rotationMatrix[0][1] * y_in);
    float y_out = (rotationMatrix[1][0] * x_in + rotationMatrix[1][1] * y_in);

    // roatated coordinates of thread (pixel) relative to output image grid
    int outputRow = (int)(row + y_out - (blockDim.y / 2.0));
    int outputCol = (int)(col + x_out - (blockDim.x / 2.0));

    //-------DEBUG OUTPUT----------
    //if (threadIdx.x == 0 && threadIdx.y == 0) {
    //    printf("X: %d, Y: %d, Radians: %d\n", col, row, orientation);
    //}

    // Write to output image
    if (pixVal == 255 && outputRow < outputHeight && outputRow > 0 && outputCol < outputWidth && outputCol > 0) {
        dev_OutputImage[outputRow * outputWidth + outputCol] = pixVal;
    }
}

__global__ void orientation_histogram_kernel(int* keyMask, int* orientationMap, float* gaussianKernel, int width, 
    int height, int x, int y, int idx) {
    // For each keypoint, accumulate a histogram of local image gradient orientations using a Gaussian-weighted window
    //		with 'sigma' of 3 times that of the current smoothing scale
    //		Each histogram consists of 36 bins covering the 360 degree range of rotations. The peak of the histogram
    //      is stored for each keypoint as its canonical orientation.

    // Create histogram shared between threads
    __shared__ float orientation_histogram[36];

    // Assumes CUDA kernel dimensions are equal to gaussian weight kernel dimensions.
    int kernelRadius = floor((double)blockDim.x / 2.0);

    // Coordinates of thread relative to the orientation map dimensions
    int row = threadIdx.y + y - kernelRadius;
    int col = threadIdx.x + x - kernelRadius;
    int map_idx = row * width + col;
    int gaussianKernelIdx = threadIdx.y * blockDim.x + threadIdx.x;
    float weight = gaussianKernel[gaussianKernelIdx];

    if (row < height && row > 0 && col > 0 && col < width) {
        float orientation = orientationMap[map_idx]; 
        int bin = floor(orientation / 10);
        atomicAdd((orientation_histogram+bin), weight);
    }

    __syncthreads();

    // Find the peak of the gradient orientation histogram (naive)
    float maxVal = 0;
    int maxBin = 0;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < 36; i++) {
            if (orientation_histogram[i] > maxVal) {
                maxVal = orientation_histogram[i];
                maxBin = i;
            }
        }

        keyMask[idx] = maxBin * 10; // Orientation in degrees with resolution of 10 degrees
        /*printf("Orientation: %d\n", maxBin * 10);*/
    }
}

void dev_calculate_gradient_maps(unsigned char* baseImage, float* gradientMagnitudeMap, int* gradientOrientationMap, 
    int width, int height) {
    /*
    * Calculates gradient magnitude and gradient orientation maps for each layer of the pyramid on device.
    *
    * See D. Lowe, section 3.1
    */

    dim3 DimGrid(height / MAP_BLOCK_WIDTH + 1, width / MAP_BLOCK_WIDTH + 1, 1);
    dim3 DimBlock(MAP_BLOCK_WIDTH, MAP_BLOCK_WIDTH, 1);
    gradient_map_kernel<<<DimGrid, DimBlock>>>(baseImage, gradientMagnitudeMap, gradientOrientationMap, height, width);
    checkCuda(cudaDeviceSynchronize());
}

void dev_matrix_subtract(unsigned char* dev_imgA, unsigned char* dev_imgB, unsigned char* dev_imgC, int width, 
    int height) {
    /*
    * Computes matrix subraction 'A' - 'B' = 'C' with CUDA
    */

    dim3 DimGrid(height / SUBTRACT_BLOCK_WIDTH + 1, width / SUBTRACT_BLOCK_WIDTH + 1, 1);
    dim3 DimBlock(SUBTRACT_BLOCK_WIDTH, SUBTRACT_BLOCK_WIDTH, 1);
    matrix_subtract_kernel << <DimGrid, DimBlock >> > (dev_imgA, dev_imgB, dev_imgC, height, width);
    checkCuda(cudaDeviceSynchronize());
}

void dev_gaussian_blur(unsigned char* img, unsigned char* outputArray, int inputWidth, int inputHeight, float sigma, 
    int kernelWidth) {
    /*
    * Convolves inputArray with a Gaussian kernel defined by 
    * g(x) = 1 / (sqrt(2*pi) * sigma) * exp(-(x**2) / (2 * sigma**2))
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

    gaussian_blur_kernel << <DimGrid, DimBlock, kernelWidth* kernelWidth * (int)sizeof(float) >> > (img, outputArray, 
        dev_kernel, inputHeight, inputWidth, kernelWidth);
    checkCuda(cudaDeviceSynchronize());
    
    free(kernel);
}

void dev_bilinear_interpolate(unsigned char* inputArray, unsigned char** dev_ExpandedImage, int inputWidth, 
    int inputHeight, float spacing, int* resultWidth, int* resultHeight) {
    /*
    * Applies bilinear interpolation to inputArray with spacing 'spacing', and stores result in 'inputArrayExpanded'.
    *
    * Pixels of the output image are calculated according to the equation: 
    * P(x,y) = R1 * (y2-y)/(y2-y1) + R2 * (y-y1)/(y2-y1)
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
    dim3 DimGrid(expandedHeight / BILINEAR_INTERPOLATE_BLOCK_WIDTH + 1, 
        expandedWidth / BILINEAR_INTERPOLATE_BLOCK_WIDTH + 1, 1);
    dim3 DimBlock(BILINEAR_INTERPOLATE_BLOCK_WIDTH, BILINEAR_INTERPOLATE_BLOCK_WIDTH, 1);
    int tileWidth = ceil(BILINEAR_INTERPOLATE_BLOCK_WIDTH / spacing) + 1;

    bilinear_interpolate_kernel<<<DimGrid, DimBlock, tileWidth * tileWidth * (int)sizeof(unsigned char)>>>(inputArray,
        *dev_ExpandedImage, spacing, inputWidth, inputHeight, expandedWidth, expandedHeight, tileWidth);
    checkCuda(cudaDeviceSynchronize());

    // DEBUG OUTPUT
    //unsigned char* expanded = (unsigned char*)malloc(expandedLength * sizeof(unsigned char));
    //checkCuda(cudaMemcpy(expanded, *dev_ExpandedImage, expandedLength, cudaMemcpyDeviceToHost));
    //const char directoryTemplate[] = "../Layers/interpolated.png";
    //stbi_write_png(directoryTemplate, expandedWidth, expandedHeight, 1, expanded, expandedWidth * 1);
}


__global__ void bilinear_interpolate_kernel(unsigned char *input, unsigned char *output, float spacing, int inputWidth,
                                            int inputHeight, int resultWidth, int resultHeight, int inputTileWidth)
{
    // Dynamically allocate shared memory for image tile
    extern __shared__ unsigned char imageTile[];

    // Calculate row & col relative to output image
    int OutputRow = blockDim.y * blockIdx.y + threadIdx.y;
    int OutputCol = blockDim.x * blockIdx.x + threadIdx.x;

    // Collaboratively load section of input image into shared memory
    int topLeftX = floor((double)(blockDim.x / spacing * blockIdx.x));
    int topLeftY = floor((double)(blockDim.y / spacing * blockIdx.y));

    if (threadIdx.x < inputTileWidth && OutputRow < resultHeight && threadIdx.y < inputTileWidth 
                                                                 && OutputCol < resultWidth) {
        imageTile[threadIdx.y * inputTileWidth + threadIdx.x] = input[(topLeftY + threadIdx.y) * 
            inputWidth + topLeftX + threadIdx.x];
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
        if (InputCol != x1 && x1 != x2) {
            R1 = Q11 * (x2 - InputCol) / (x2 - x1) + Q21 * (InputCol - x1) / (x2 - x1);
            R2 = Q12 * (x2 - InputCol) / (x2 - x1) + Q22 * (InputCol - x1) / (x2 - x1);
        }
        else {
            R1 = Q11;
            R2 = Q12;
        }

        // Final interpolated value
        // Skip column-wise interpolation if coordinate lands on a row instead of between rows
        if (InputRow != y1 && y1 != y2) {
            output[OutputRow * resultWidth + OutputCol] = 
                (unsigned char)(R1 * (y2 - InputRow) / (y2 - y1) + R2 * (InputRow - y1) / (y2 - y1));
        }
        else {
            output[OutputRow * resultWidth + OutputCol] = (unsigned char)R1;
        }
    }
}

__global__ void gaussian_blur_kernel(unsigned char* input, unsigned char* output, float* gaussianKernel, int height, 
                                     int width, int kernelDim) {
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
                    pixFloatVal += (float)(input[curRow * width + curCol] * s_blurFilt[(blurRow + filtPadding) * 
                                                                                kernelDim + blurCol + filtPadding]);
                    // Accumulate a factor to normalize by
                    pixNormalizeFactor += s_blurFilt[(blurRow + filtPadding) * kernelDim + blurCol + filtPadding]; 
                }
            }
        }
        // Write our new pixel value out
        output[row * width + col] = (unsigned char)(int)(pixFloatVal / pixNormalizeFactor);
    }
}

__global__ void matrix_subtract_kernel(unsigned char* A, unsigned char* B, unsigned char* C, int height, int width) {
    // Subtract corresponding values in A and B and store in C
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = row * width + col;

    if (row < width && col < height) {
        int pixVal = A[idx] - B[idx];
        if (pixVal > 0) {
            C[idx] = (unsigned char)pixVal;
        }
        else {
            C[idx] = 0;
        }
    }
}

__global__ void gradient_map_kernel(unsigned char* dev_input, float* magnitudeMap, int* orientationMap, int height, 
    int width) {
    
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = row * width + col;

    if (row < height - 1 && col < width - 1) {
        unsigned char pixVal = dev_input[idx];
        unsigned char belowPixVal = dev_input[idx + width];
        unsigned char rightPixVal = dev_input[idx + 1];
        
        double temp1 = pixVal - belowPixVal;
        double temp2 = pixVal - rightPixVal;
        magnitudeMap[idx] = sqrt(temp1 * temp1 + temp2 * temp2);
        orientationMap[idx] = floor((atan2(temp1, (double)(rightPixVal - pixVal)) + PI) * 180 / PI);
    }
}