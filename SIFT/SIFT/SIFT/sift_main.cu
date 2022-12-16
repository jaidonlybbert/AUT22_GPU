/*
* Author: Jaidon Lybbert
* Date:   12/1/2022
* 
* CPU and CUDA implementations of the SIFT object detection algorithm
* 
* CUDA Toolkit Version 11.8
*
* Based on the paper by David G. Lowe
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


// Struct definitions, and forward declarations
#include "sift_host.h"

// CUDA
#include <nvtx3/nvToolsExt.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../util/cuda_helpers.h"
#include "sift_device.cuh"
__device__ __constant__ unsigned char dev_KeypointGraphic[289];

// Image library
#define STB_IMAGE_IMPLEMENTATION 
#include "../util/stb_image.h"  
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../util/stb_image_write.h"

// Debug flag for additional output
#define DEBUG false

void host_sift(unsigned char* h_rawImage, unsigned char** h_outputImage, int imgSize, int imgWidth, int imgHeight,
    int* outputWidth, int* outputHeight);

void compare_images();

int main() {
    /*
    * Comparison between host and device execution of the SIFT algorithm
    * for benchmarking with NVIDIA Nsight Systems
    */

    printf("Comparing");
    compare_images();
    printf("Compare Done.");

	nvtxMark("Program start!\n");
	// 0) Load input image into an array -> 'inputArray'
	// read input image from file
	const char filename[] = "../util/test_image.png";
	int x_cols = 0;
	int y_rows = 0;
	int n_pixdepth = 0;

	nvtxMark("Load image from disk.\n");
	unsigned char* h_rawImage = stbi_load(filename, &x_cols, &y_rows, &n_pixdepth, 1);
	int imgSize = x_cols * y_rows * (int)sizeof(unsigned char);
	int imgWidth = x_cols;
	int imgHeight = y_rows;
    int outputWidth = 0;
    int outputHeight = 0;
    unsigned char* h_outputImage = NULL;

	// Copy image to device
	nvtxMark("Copy image to device.\n");
	unsigned char* dev_rawImage = 0;
	unsigned char* dev_outputImage = 0;
	checkCuda(cudaMalloc((void**)&dev_rawImage, imgSize));
	checkCuda(cudaMemcpy(dev_rawImage, h_rawImage, imgSize, cudaMemcpyHostToDevice));

	//// Execute CUDA implementation of SIFT
	nvtxRangePush("CUDA SIFT");
	dev_sift(dev_rawImage, &dev_outputImage, imgSize, imgWidth, imgHeight, &outputWidth, &outputHeight);
	nvtxRangePop();

	// Execute CPU implementation of SIFT
	nvtxRangePush("Host SIFT");
    host_sift(h_rawImage, &h_outputImage, imgSize, imgWidth, imgHeight, &outputWidth, &outputHeight);
	nvtxRangePop();

	// Write output from device
	nvtxMark("Allocate host memory for output");
	unsigned char* h_fromDevOutput = (unsigned char*)malloc(outputWidth * outputHeight * (int)sizeof(unsigned char));
	const char AnnotatedDir[27] = "../Layers/DeviceOutput.png";
	nvtxMark("Copy output from device");
	checkCuda(cudaMemcpy(h_fromDevOutput, dev_outputImage, outputWidth * outputHeight * (int)sizeof(unsigned char),
		cudaMemcpyDeviceToHost));
	nvtxMark("Write device output to disk");
	stbi_write_png(AnnotatedDir, outputWidth, outputHeight, 1, h_fromDevOutput, outputWidth * 1);
	free(h_fromDevOutput);
	nvtxMark("Done.");

    // Write output from host
    nvtxMark("Write host output to disk");
    const char hostAnnotatedDir[25] = "../Layers/HostOutput.png";
    stbi_write_png(hostAnnotatedDir, outputWidth, outputHeight, 1, h_outputImage, outputWidth * 1);
    free(h_outputImage);
    nvtxMark("Program done.");

    return 0;
}

void compare_images() {
    const char filenameA[] = "../Layers/deviceInterp.png";
    const char filenameB[] = "../Layers/hostinterp.png";
    int x_cols = 0;
    int y_rows = 0;
    int n_pixdepth = 0;
    unsigned char* h_img = stbi_load(filenameA, &x_cols, &y_rows, &n_pixdepth, 1);
    unsigned char* d_img = stbi_load(filenameB, &x_cols, &y_rows, &n_pixdepth, 1);
    
    for (int i = 0; i < y_rows; i++) {
        for (int j = 0; j < x_cols; j++) {
            unsigned char a = h_img[i * x_cols + j];
            unsigned char b = d_img[i * x_cols + j];

            if (a != b) {
                printf("Mismatch: %d, Row: %d, Col: %d\n", (int)b - a, i, j);
            }
        }
    }
}

void dev_sift(unsigned char* dev_rawImage, unsigned char** dev_outputImage, int imgSize, int imgWidth, int imgHeight,
    int* outputWidth, int* outputHeight)
{
    cudaError_t cudaStatus;
    unsigned char* dev_PyramidInput = 0;
    unsigned char* dev_LayerInput = 0;

    // 1) Expand the raw image via bilinear interpolation -> 'dev_InputImage'
    //      This expanded image is the 'input' top layer of the image pyramid
    //      The image is expanded first so that high-resolution details aren't missed

    // Create the expanded image space
    int resultWidth = 0;
    int resultHeight = 0;

    // Call parallel bilinear interpolate on device
    nvtxMark("Bilinear interpolate kernel start.");
    dev_bilinear_interpolate(dev_rawImage, &dev_PyramidInput, imgWidth, imgHeight, 2.0, &resultWidth, &resultHeight);
    nvtxMark("Copy bilinear interpolate result.");
    nvtxMark("Allocate");
    checkCuda(cudaMalloc(&dev_LayerInput, resultWidth * resultHeight * (int)sizeof(unsigned char)));
    nvtxMark("Cuda copy");
    checkCuda(cudaMemcpy(dev_LayerInput, dev_PyramidInput, resultWidth * resultHeight * (int)sizeof(unsigned char),
        cudaMemcpyDeviceToDevice));
    nvtxMark("Cuda free");
    checkCuda(cudaFree(dev_rawImage));

    int pyramidInputWidth = resultWidth;
    int pyramidInputHeight = resultHeight;
    *outputWidth = resultWidth;
    *outputHeight = resultHeight;
    *dev_outputImage = dev_PyramidInput;

    // Create the array of pyramid layers
    imagePyramidLayer pyramid[LAYERS];
    // This template is used to iteratively export layers as images, by replacing the placeholder 'X' 
    // with the layer index
    const char directoryTemplate[] = "../Layers/AX.png";

    // Pointer to host-side output image for debugging
    unsigned char* h_debug_img = NULL;
    int* h_debug_orientations = NULL;

    // DEBUG OUTPUT IMAGE
    const char dev_interpimg[] = "../Layers/deviceInterp.png";
    h_debug_img = (unsigned char*)malloc(resultWidth * resultHeight * (int)sizeof(unsigned char));
    checkCuda(cudaMemcpy(h_debug_img, dev_PyramidInput, resultWidth * resultHeight * (int)sizeof(unsigned char),
        cudaMemcpyDeviceToHost));
    stbi_write_png(dev_interpimg, resultWidth, resultHeight, 1, h_debug_img, resultWidth * 1);
    free(h_debug_img);

    // For each layer in pyramid, construct two images with increasing levels of blur, and compute the difference of 
    // gaussian (DoG)
    // Apply bilinear interpolation to the second of the two blurred images, and use that in the next iteration as the 
    // input image
    // Keep these images in device memory, and reference them via the pointers in the 'imagePyramidLayer' structs
    for (int i = 0; i < LAYERS; i++) {
        nvtxMark("Start building next pyramid layer");
        // Copy directory name to layer
        for (int j = 0; j < 17; j++) {
            pyramid[i].layerOutputDir[j] = directoryTemplate[j];
        }
        pyramid[i].layerOutputDir[10] = (char)'E';
        pyramid[i].layerOutputDir[11] = (char)('0' + (char)i);
        pyramid[i].height = resultHeight;
        pyramid[i].width = resultWidth;

        // Allocate memory for blurred image A, B and DoG on device
        nvtxMark("Allocate memory for layer on device.");
        checkCuda(cudaMalloc(&pyramid[i].imageA, resultWidth * resultHeight * (int)sizeof(unsigned char)));
        checkCuda(cudaMalloc(&pyramid[i].imageB, resultWidth * resultHeight * (int)sizeof(unsigned char)));
        checkCuda(cudaMalloc(&pyramid[i].DoG, resultWidth * resultHeight * (int)sizeof(unsigned char)));
        checkCuda(cudaMalloc(&pyramid[i].gradientMagnitudeMap, resultWidth * resultHeight * (int)sizeof(float)));
        checkCuda(cudaMalloc(&pyramid[i].gradientOrientationMap, resultWidth * resultHeight * (int)sizeof(int)));
        // Allocate memory for keymasks for inner layers
        if (i > 0 && i < LAYERS - 1) {
            checkCuda(cudaMalloc(&pyramid[i].keyMask, resultWidth * resultHeight * (int)sizeof(int)));
        }

        // 2) Apply gaussian blur to the input image -> 'A'
        // Parameters for generating gaussian kernel
        float sigma = sqrt(2);
        int kernelWidth = 9;
        // Call device kernel for guassian blur
        nvtxMark("Gaussian blur A");
        dev_gaussian_blur(dev_LayerInput, pyramid[i].imageA, resultWidth, resultHeight, sigma, kernelWidth);
        // DEBUG OUTPUT IMAGE
        h_debug_img = (unsigned char*)malloc(resultWidth * resultHeight * (int)sizeof(unsigned char));
        checkCuda(cudaMemcpy(h_debug_img, pyramid[i].imageA, resultWidth * resultHeight * (int)sizeof(unsigned char), 
                             cudaMemcpyDeviceToHost));
        stbi_write_png(pyramid[i].layerOutputDir, resultWidth, resultHeight, 1, h_debug_img, resultWidth * 1);
        free(h_debug_img);

        // 3) Apply gaussian blur to the image A -> 'B'
        nvtxMark("Gaussian blur B");
        dev_gaussian_blur(pyramid[i].imageA, pyramid[i].imageB, resultWidth, resultHeight, sigma, kernelWidth);
        // DEBUG OUTPUT IMAGE
        h_debug_img = (unsigned char*)malloc(resultWidth * resultHeight * (int)sizeof(unsigned char));
        pyramid[i].layerOutputDir[10] = (char)'F';
        checkCuda(cudaMemcpy(h_debug_img, pyramid[i].imageB, resultWidth * resultHeight * (int)sizeof(unsigned char), 
                             cudaMemcpyDeviceToHost));
        stbi_write_png(pyramid[i].layerOutputDir, resultWidth, resultHeight, 1, h_debug_img, resultWidth * 1);
        free(h_debug_img);

        // 4) Subtract 'A' - 'B' -> 'DoG'
        nvtxMark("Matrix subtract");
        dev_matrix_subtract(pyramid[i].imageA, pyramid[i].imageB, pyramid[i].DoG, resultWidth, resultHeight);
        // DEBUG OUTPUT IMAGE
        //h_debug_img = (unsigned char*)malloc(resultWidth * resultHeight * (int)sizeof(unsigned char));
        //pyramid[i].layerOutputDir[10] = (char)'C';
        //checkCuda(cudaMemcpy(h_debug_img, pyramid[i].DoG, resultWidth * resultHeight * (int)sizeof(unsigned char), 
        //                     cudaMemcpyDeviceToHost));
        //stbi_write_png(pyramid[i].layerOutputDir, resultWidth, resultHeight, 1, h_debug_img, resultWidth * 1);
        //free(h_debug_img);

        // 5) Compute gradient maps
        nvtxMark("Generate gradient maps");
        dev_calculate_gradient_maps(pyramid[i].imageA, pyramid[i].gradientMagnitudeMap,
            pyramid[i].gradientOrientationMap, resultWidth, resultHeight);
        // DEBUG OUTPUT IMAGE
        h_debug_img = (unsigned char*)malloc(resultWidth * resultHeight * (int)sizeof(unsigned char));
        pyramid[i].layerOutputDir[10] = (char)'G';
        checkCuda(cudaMemcpy(h_debug_img, pyramid[i].DoG, resultWidth * resultHeight * (int)sizeof(unsigned char), 
                             cudaMemcpyDeviceToHost));
        stbi_write_png(pyramid[i].layerOutputDir, resultWidth, resultHeight, 1, h_debug_img, resultWidth * 1);
        free(h_debug_img);

        //------DEBUG CHECK VALID ORIENTATIONS-----
        //h_debug_orientations = (int*)malloc(resultWidth * resultHeight * (int)sizeof(int));
        //checkCuda(cudaMemcpy(h_debug_orientations, pyramid[i].gradientOrientationMap, resultWidth * resultHeight * 
        // (int)sizeof(int), cudaMemcpyDeviceToHost));
        //int countInvalidOrientations = 0;
        //for (int k = 0; k < resultHeight; k++) {
        //    for (int r = 0; r < resultWidth; r++) {
        //        int tmp = h_debug_orientations[k * resultWidth + r];
        //        if (tmp < 0 || tmp > 360) {
        //            printf("Orientation: %d Row: %d Col: %d\n", tmp, k, r);
        //        }
        //    }
        //}
        //free(h_debug_orientations);

        // 6) Apply bilinear interpolation to 'B' -> Image input for next layer
        //    Skip for last iteration
        nvtxMark("Free layer input");
        checkCuda(cudaFree(dev_LayerInput));
        if (i < LAYERS - 1) {
            nvtxMark("Bilinear interpolate for input to next layer");
            dev_bilinear_interpolate(pyramid[i].imageB, &dev_LayerInput, resultWidth, resultHeight,
                BILINEAR_INTERPOLATION_SPACING, &resultWidth, &resultHeight);
        }
    }

    //printf("Pyramid generated successfully.\n");

    nvtxMark("Creating keypoint template");
    // Draw keypoint graphic template. This will be copied and rotated to match each keypoint orientation.
    unsigned char keypointGraphic[17][17] = { 0 };
    for (int i = 0; i < 17; i++) {
        // Top line
        keypointGraphic[0][i] = 255;
        // Bottom line
        keypointGraphic[16][i] = 255;
        // Left line
        keypointGraphic[i][0] = 255;
        // Right line
        keypointGraphic[i][16] = 255;
        // Center line segment
        if (i > 8) {
            keypointGraphic[8][i] = 255;
        }
    }

    nvtxMark("Copy template to device");
    checkCuda(cudaMemcpyToSymbol(dev_KeypointGraphic, keypointGraphic, 17 * 17 * (int)sizeof(unsigned char)));

    // 7) Collect local maxima and minima coordinates in scale space, and mark them in the 'keyMask' of each layer
    //    Use CUDA dynamic parallelism to draw keypoints on the output image
    nvtxMark("Find keypoints and draw them on output image.");
    dev_generate_key_masks(pyramid, dev_PyramidInput, pyramidInputWidth, pyramidInputHeight);

    //printf("Keys generated, and written to output image.");
}

void host_sift(unsigned char* h_rawImage, unsigned char** h_outputImage, int imgSize, int imgWidth, int imgHeight, 
    int* outputWidth, int* outputHeight) {

	// 1) Expand the input image via bilinear interpolation -> 'inputArrayExpanded'
	//      This expanded image is the 'input' top layer of the image pyramid
	//      The image is expanded first so that high-resolution details aren't missed
	unsigned char* inputArrayExpanded = 0;
	int resultWidth = 0;
	int resultHeight = 0;

	bilinear_interpolate(h_rawImage, &inputArrayExpanded, imgWidth, imgHeight, 2, &resultWidth, &resultHeight);
    const char host_interp[] = "../Layers/hostinterp.png";
    stbi_write_png(host_interp, resultWidth, resultHeight, 1, inputArrayExpanded, resultWidth * 1);

	unsigned char* interpolatedInput = inputArrayExpanded;
	int interpolatedWidth = resultWidth;
	int interpolatedHeight = resultHeight;
    *outputWidth = resultWidth;
    *outputHeight = resultHeight;

	// Create the array of pyramid layers
	imagePyramidLayer pyramid[LAYERS];
	// This template is used to iteratively export layers as images, by replacing the placeholder 'X' with the layer index
	const char directoryTemplate[] = "../Layers/AX.png";

	// For each layer in pyramid, construct two images with increasing levels of blur, and compute the difference of gaussian (DoG)
	// Apply bilinear interpolation to the second of the two blurred images, and use that in the next iteration as the input image
	// Keep these images in memory, and reference them via the pointers in the 'imagePyramidLayer' structs
	for (int i = 0; i < LAYERS; i++) {

		// Copy directory name to layer
		for (int j = 0; j < 17; j++) {
			pyramid[i].layerOutputDir[j] = directoryTemplate[j];
		}
		pyramid[i].layerOutputDir[10] = (char)'A';
		pyramid[i].layerOutputDir[11] = (char)('0' + (char)i);
		pyramid[i].height = resultHeight;
		pyramid[i].width = resultWidth;

		// 2) Apply gaussian blur to the input image -> 'A'
		pyramid[i].imageA = (unsigned char*)malloc(resultWidth * resultHeight * (int)sizeof(unsigned char));
		float sigma = sqrt(2);
		int kernelWidth = 9;

		gaussian_blur(inputArrayExpanded, pyramid[i].imageA, resultWidth, resultHeight, sigma, kernelWidth);

        //----DEBUG OUTPUT
		stbi_write_png(pyramid[i].layerOutputDir, resultWidth, resultHeight, 1, pyramid[i].imageA, resultWidth * 1);

		// 3) Apply gaussian blur to the image A -> 'B'
		pyramid[i].imageB = (unsigned char*)malloc(resultWidth * resultHeight * (int)sizeof(unsigned char));

		gaussian_blur(pyramid[i].imageA, pyramid[i].imageB, resultWidth, resultHeight, sigma, kernelWidth);

        //----DEBUG OUTPUT
		pyramid[i].layerOutputDir[10] = (char)'B';
		stbi_write_png(pyramid[i].layerOutputDir, resultWidth, resultHeight, 1, pyramid[i].imageB, resultWidth * 1);

		// 4) Subtract 'A' - 'B' -> 'C'
		pyramid[i].DoG = (unsigned char*)malloc(resultWidth * resultHeight * (int)sizeof(unsigned char));

		matrix_subtract(pyramid[i].imageA, pyramid[i].imageB, pyramid[i].DoG, resultWidth, resultHeight);

        //----DEBUG OUTPUT
		pyramid[i].layerOutputDir[10] = (char)'C';
		stbi_write_png(pyramid[i].layerOutputDir, resultWidth, resultHeight, 1, pyramid[i].DoG, resultWidth * 1);

		// 5) Apply bilinear interpolation to 'B' -> Image input for next layer
		inputArrayExpanded = 0;

		bilinear_interpolate(pyramid[i].imageB, &inputArrayExpanded, resultWidth, resultHeight, 
			BILINEAR_INTERPOLATION_SPACING, &resultWidth, &resultHeight);
	}
	

	// 6) Collect local maxima and minima coordinates in scale space (x, y, layer) -> 'keypoints'
	// Start the linked list of keypoints
	keypoint* keypoints = NULL;
	// Populate the linked list with all coordinates of extrema in the pyramid
	
	find_extrema(pyramid, &keypoints);
	 
	// 7) Calculate image gradient magnitude 'M' and orientations 'R' at each pixel for each smoothed image 
	//    'A' at each layer of the pyramid. The result are maps with the same dimensions as their corresponding layers,
	//    referenced in the imagePyramidLayer structs, where each entry corresponds to a matching pixel.
	calculate_gradient(pyramid);

	// 8) For each keypoint, accumulate a histogram of local image gradient orientations using a Gaussian-weighted window 
	//		with 'sigma' of 3 times that of the current smoothing scale
	//		Each histogram consists of 36 bins covering the 360 degree range of rotations. The peak of the histogram
	//      is stored for each keypoint as its canonical orientation.
	characterize_keypoint(keypoints, pyramid);

	// 9) Draw keypoint annotations. Annotations are done by multiplying a rotation matrix with coordinates of a template annotation 
	//      to orient the annotation in the direction of the keypoint orientation. The translated annotation is written over top of
	//      the original input image
	unsigned char* annotatedImage = (unsigned char*)malloc(interpolatedHeight * interpolatedWidth * (int)sizeof(unsigned char));
	draw_keypoints(keypoints, interpolatedInput, annotatedImage, interpolatedWidth, interpolatedWidth);

    *h_outputImage = annotatedImage;

	int countKeypointsLayer1 = 0;
	int countKeypointsLayer2 = 0;
	while (keypoints->next != NULL) {
		printf("Layer: %d, X: %d, Y: %d, Rotation: %d\n", keypoints->layer, keypoints->x, keypoints->y, keypoints->orientation);
		keypoints = keypoints->next;
	}
}
