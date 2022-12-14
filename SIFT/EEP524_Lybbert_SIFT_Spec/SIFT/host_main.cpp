/*
* Author: Jaidon Lybbert
* Date:   12/1/2022
* 
* Sequential implementation of the SIFT object detection algorithm
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


// Struct definitions, and forward declarations (useful info on the structs in here)
#include "sift_host.h"

// Image library
#define STB_IMAGE_IMPLEMENTATION 
#include "../util/stb_image.h"  
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../util/stb_image_write.h"

// Windows timer API
#include <windows.h>
#include <profileapi.h>

int main() {
	LARGE_INTEGER counterA, counterB, elapsed_us;
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&counterA);

	// 0) Load input image into an array -> 'inputArray'
	// read input image from file
	const char filename[] = "../util/test_image.png";
	int x_cols = 0;
	int y_rows = 0;
	int n_pixdepth = 0;
	unsigned char* inputArray = stbi_load(filename, &x_cols, &y_rows, &n_pixdepth, 1);
	int imgSize = x_cols * y_rows * sizeof(unsigned char);
	int imgWidth = x_cols;
	int imgHeight = y_rows;

	QueryPerformanceCounter(&counterB);
	elapsed_us.QuadPart = (counterB.QuadPart - counterA.QuadPart) * 1000000 / frequency.QuadPart;
	printf("Loading input image: %d us\n", ((int)elapsed_us.QuadPart));

	// 1) Expand the input image via bilinear interpolation -> 'inputArrayExpanded'
	//      This expanded image is the 'input' top layer of the image pyramid
	//      The image is expanded first so that high-resolution details aren't missed
	unsigned char* inputArrayExpanded = 0;
	int resultWidth = 0;
	int resultHeight = 0;

	QueryPerformanceCounter(&counterA);
	bilinear_interpolate(inputArray, &inputArrayExpanded, imgWidth, imgHeight, 2, &resultWidth, &resultHeight);
	QueryPerformanceCounter(&counterB);
	elapsed_us.QuadPart = (counterB.QuadPart - counterA.QuadPart) * 1000000 / frequency.QuadPart;
	printf("Bilinear Interpolation 0: %d us\n", ((int)elapsed_us.QuadPart));

	unsigned char* interpolatedInput = inputArrayExpanded;
	int interpolatedWidth = resultWidth;
	int interpolatedHeight = resultHeight;
	free(inputArray);

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
		pyramid[i].imageA = (unsigned char*)malloc(resultWidth * resultHeight * sizeof(unsigned char));
		float sigma = sqrt(2);
		int kernelWidth = 9;

		QueryPerformanceCounter(&counterA);
		gaussian_blur(inputArrayExpanded, pyramid[i].imageA, resultWidth, resultHeight, sigma, kernelWidth);
		QueryPerformanceCounter(&counterB);
		elapsed_us.QuadPart = (counterB.QuadPart - counterA.QuadPart) * 1000000 / frequency.QuadPart;
		printf("Blur layer %d, A: %d us\n", i, ((int)elapsed_us.QuadPart));

		stbi_write_png(pyramid[i].layerOutputDir, resultWidth, resultHeight, 1, pyramid[i].imageA, resultWidth * n_pixdepth);

		// 3) Apply gaussian blur to the image A -> 'B'
		pyramid[i].imageB = (unsigned char*)malloc(resultWidth * resultHeight * sizeof(unsigned char));

		QueryPerformanceCounter(&counterA);
		gaussian_blur(pyramid[i].imageA, pyramid[i].imageB, resultWidth, resultHeight, sigma, kernelWidth);
		QueryPerformanceCounter(&counterB);
		elapsed_us.QuadPart = (counterB.QuadPart - counterA.QuadPart) * 1000000 / frequency.QuadPart;
		printf("Blur layer %d, B: %d us\n", i, ((int)elapsed_us.QuadPart));

		pyramid[i].layerOutputDir[10] = (char)'B';
		stbi_write_png(pyramid[i].layerOutputDir, resultWidth, resultHeight, 1, pyramid[i].imageB, resultWidth * n_pixdepth);

		// 4) Subtract 'A' - 'B' -> 'C'
		pyramid[i].DoG = (unsigned char*)malloc(resultWidth * resultHeight * sizeof(unsigned char));

		QueryPerformanceCounter(&counterA);
		matrix_subtract(pyramid[i].imageA, pyramid[i].imageB, pyramid[i].DoG, resultWidth, resultHeight);
		QueryPerformanceCounter(&counterB);
		elapsed_us.QuadPart = (counterB.QuadPart - counterA.QuadPart) * 1000000 / frequency.QuadPart;
		printf("Matrx Subtract %d: %d us\n", i, ((int)elapsed_us.QuadPart));

		pyramid[i].layerOutputDir[10] = (char)'C';
		stbi_write_png(pyramid[i].layerOutputDir, resultWidth, resultHeight, 1, pyramid[i].DoG, resultWidth * n_pixdepth);

		// 5) Apply bilinear interpolation to 'B' -> Image input for next layer
		inputArrayExpanded = 0;

		QueryPerformanceCounter(&counterA);
		bilinear_interpolate(pyramid[i].imageB, &inputArrayExpanded, resultWidth, resultHeight, 
			BILINEAR_INTERPOLATION_SPACING, &resultWidth, &resultHeight);
		QueryPerformanceCounter(&counterB);
		elapsed_us.QuadPart = (counterB.QuadPart - counterA.QuadPart) * 1000000 / frequency.QuadPart;
		printf("Bilinear Interpolate %d: %d us\n", i + 1, ((int)elapsed_us.QuadPart));
	}
	

	// 6) Collect local maxima and minima coordinates in scale space (x, y, layer) -> 'keypoints'
	// Start the linked list of keypoints
	keypoint* keypoints = NULL;
	// Populate the linked list with all coordinates of extrema in the pyramid
	
	QueryPerformanceCounter(&counterA);
	find_extrema(pyramid, &keypoints);
	QueryPerformanceCounter(&counterB);
	elapsed_us.QuadPart = (counterB.QuadPart - counterA.QuadPart) * 1000000 / frequency.QuadPart;
	printf("Find extrema: %d us\n", ((int)elapsed_us.QuadPart));
	 
	// 7) Calculate image gradient magnitude 'M' and orientations 'R' at each pixel for each smoothed image 
	//    'A' at each layer of the pyramid. The result are maps with the same dimensions as their corresponding layers,
	//    referenced in the imagePyramidLayer structs, where each entry corresponds to a matching pixel.
	QueryPerformanceCounter(&counterA);
	calculate_gradient(pyramid);
	QueryPerformanceCounter(&counterB);
	elapsed_us.QuadPart = (counterB.QuadPart - counterA.QuadPart) * 1000000 / frequency.QuadPart;
	printf("Gradient maps: %d us\n", ((int)elapsed_us.QuadPart));

	// 8) For each keypoint, accumulate a histogram of local image gradient orientations using a Gaussian-weighted window 
	//		with 'sigma' of 3 times that of the current smoothing scale
	//		Each histogram consists of 36 bins covering the 360 degree range of rotations. The peak of the histogram
	//      is stored for each keypoint as its canonical orientation.
	QueryPerformanceCounter(&counterA);
	characterize_keypoint(keypoints, pyramid);
	QueryPerformanceCounter(&counterB);
	elapsed_us.QuadPart = (counterB.QuadPart - counterA.QuadPart) * 1000000 / frequency.QuadPart;
	printf("Characterize keypoints: %d us\n", ((int)elapsed_us.QuadPart));

	// 9) Draw keypoint annotations. Annotations are done by multiplying a rotation matrix with coordinates of a template annotation 
	//      to orient the annotation in the direction of the keypoint orientation. The translated annotation is written over top of
	//      the original input image
	unsigned char* annotatedImage = (unsigned char*)malloc(interpolatedHeight * interpolatedWidth * sizeof(unsigned char));
	QueryPerformanceCounter(&counterA);
	draw_keypoints(keypoints, interpolatedInput, annotatedImage, interpolatedWidth, interpolatedWidth);
	QueryPerformanceCounter(&counterB);
	elapsed_us.QuadPart = (counterB.QuadPart - counterA.QuadPart) * 1000000 / frequency.QuadPart;
	printf("Draw keypoints: %d us\n", ((int)elapsed_us.QuadPart));

	// 10) Write image result with keypoint annotations
	const char imgFileInterpolated[] = "../Util/InterpolatedInput.png";
	stbi_write_png(imgFileInterpolated, interpolatedWidth, interpolatedHeight, 1, interpolatedInput, interpolatedWidth * n_pixdepth);

	const char imgFileAnnotated[] = "../Util/Annotated.png";
	stbi_write_png(imgFileAnnotated, interpolatedWidth, interpolatedHeight, 1, annotatedImage, interpolatedWidth * n_pixdepth);

	return 0;
}
