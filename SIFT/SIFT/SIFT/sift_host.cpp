#include "sift_host.h"
#include <cstdlib>
#include <cassert>
#include <stdio.h>

void bilinear_interpolate(unsigned char* inputArray, unsigned char** inputArrayExpanded, int inputWidth, int inputHeight,
	float spacing, int* resultWidth, int* resultHeight) {
	/*
	* Applies bilinear interpolation to inputArray with spacing 'spacing', and stores result in 'inputArrayExpanded'.
	*
	* Pixels of the output image are calculated according to the equation: P(x,y) = R1 * (y2-y)/(y2-y1) + R2 * (y-y1)/(y2-y1)
	*/
	assert(inputArray != NULL);
	assert(inputArrayExpanded != NULL);

	// Create the expanded image space
	int expandedHeight = ceil(inputHeight * spacing);
	int expandedWidth = ceil(inputWidth * spacing);
	int expandedLength = expandedHeight * expandedWidth;
	*inputArrayExpanded = (unsigned char*)malloc(expandedLength * sizeof(unsigned char));

	// Report the new width and height back to the main loop, for future use
	*resultWidth = expandedWidth;
	*resultHeight = expandedHeight;

	// Iterate over the expanded image space, and calculate the bilinearly interpolated value for each
	// pixel
	for (int i = 0; i < expandedHeight; i++) {
		for (int j = 0; j < expandedWidth; j++) {
			// Ignore edges
			if (i == 0 || j == 0) {
				(*inputArrayExpanded)[i * expandedWidth + j] = 0;
				continue;
			}
			// Coordinates of output pixel relative to the original image grid
			float x = j / spacing;
			float y = i / spacing;
			// Coordinates of surrounding input image pixels
			float x1 = floor(x);
			float x2 = (ceil(x) < inputWidth) ? ceil(x) : x1;
			float y1 = floor(y);
			float y2 = (ceil(y) < inputHeight) ? ceil(y) : y1;
			// Pixel values of surrounding input image pixels
			unsigned char Q11 = inputArray[(int)y1 * inputWidth + (int)x1];
			unsigned char Q21 = inputArray[(int)y1 * inputWidth + (int)x2];
			unsigned char Q12 = inputArray[(int)y2 * inputWidth + (int)x1];
			unsigned char Q22 = inputArray[(int)y2 * inputWidth + (int)x2];
			// Row-wise interpolated values
			float R1, R2;
			// Skip row-wise interpolation if the coordinate lands on a column instead of between columns
			if (x != x1) {
				R1 = Q11 * (x2 - x) / (x2 - x1) + Q21 * (x - x1) / (x2 - x1);
				R2 = Q12 * (x2 - x) / (x2 - x1) + Q22 * (x - x1) / (x2 - x1);
			}
			else {
				R1 = Q11;
				R2 = Q12;
			}

			// Final interpolated value
			// Skip column-wise interpolation if coordinate lands on a row instead of between rows
			if (y != y1) {
				(*inputArrayExpanded)[i * expandedWidth + j] = R1 * (y2 - y) / (y2 - y1) + R2 * (y - y1) / (y2 - y1);
			}
			else {
				(*inputArrayExpanded)[i * expandedWidth + j] = R1;
			}
		}
	}
}

void gaussian_blur(unsigned char* img, unsigned char* outputArray, int inputWidth, int inputHeight, float sigma, int kernelWidth) {
	/*
	* Convolves inputArray with a Gaussian kernel defined by g(x) = 1 / (sqrt(2*pi) * sigma) * exp(-(x**2) / (2 * sigma**2))
	*
	* kernelWidth is assumed to be odd.
	*/

	assert(kernelWidth % 2 == 1);

	int kernelRadius = floor(kernelWidth / 2);

	// Generate gaussian kernel
	float* kernel = NULL;
	get_gaussian_kernel(&kernel, sigma, kernelWidth);

	// Compute kernel sum for normalization
	float kernelCoefficientSum = 0;
	for (int i = 0; i < kernelWidth * kernelWidth; i++) {
		kernelCoefficientSum += kernel[i];
	}

	// Convolve image with 2D blur kernel
	for (int i = 0; i < inputHeight; i++) {
		for (int j = 0; j < inputWidth; j++) {
			float pixelConvolutionSum = 0;
			for (int k = 0; k < kernelWidth; k++) {
				for (int z = 0; z < kernelWidth; z++) {
					int vShift = k - kernelRadius;
					int hShift = z - kernelRadius;
					if ((i + vShift > 0) && (i + vShift < inputHeight) && (j + hShift > 0) && (j + hShift < inputWidth)) {
						pixelConvolutionSum += img[(i + vShift) * inputWidth + (j + hShift)] * kernel[k * kernelWidth + z];
					}
				}
			}
			outputArray[i * inputWidth + j] = pixelConvolutionSum / kernelCoefficientSum;
		}
	}

	free(kernel);

	return;
}

void matrix_subtract(unsigned char* A, unsigned char* B, unsigned char* C, int width, int height) {
	/*
	* Performs the matrix subtraction 'C' = 'A' - 'B'
	*/

	for (int i = 0; i < (width * height); i++) {
		if (A[i] - B[i] > 0) {
			C[i] = A[i] - B[i];
		}
		else {
			C[i] = 0;
		}
	}

	return;
}

void find_extrema(imagePyramidLayer pyramid[LAYERS], keypoint** keypoints) {
	/*
	* Searches pyramid structure for local extrema. Each pixel is evaluated against its nearest neighbors in scale space.
	*
	* Extrema are stored in 'keypoints' linked list as (x, y, layer) coordinates,
	* where x, y are relative to the input image (highest & smallest layer of the pyramid)
	*/

	int numKeypoints = 0;

	// Create first keypoint in linked list
	keypoint* currentKeypoint = new keypoint;
	*keypoints = currentKeypoint;
	currentKeypoint->previous = NULL;

	// Nearest neighbor approach to finding minima and maxima values in the Difference-of-Gaussian scale space
	// Each pixel of each inner layer is evaluated against nearest surrounding pixels
	// The coordinates of minima and maxima are stored as keypoints, and appended to the linked list
	for (int i = 1; i < LAYERS - 1; i++) {
		// Iterate over rows of the layer
		for (int j = 1; j < pyramid[i].height - 1; j++) {
			// Iterate over columns of the layer
			for (int k = 1; k < pyramid[i].width - 1; k++) {
				// Pixel to be evaluated
				unsigned char pixVal = pyramid[i].DoG[j * pyramid[i].width + k];

				unsigned char minVal = 255;
				unsigned char maxVal = 0;
				unsigned char neighborPixVal = 0;

				// Check pixel value against neighboring pixel values in the same layer
				for (int x = -1; x <= 1; x++) {
					for (int y = -1; y <= 1; y++) {
						neighborPixVal = pyramid[i].DoG[(j + y) * pyramid[i].width + (k + x)];

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

				// If the pixel value is an extrema of the current layer, check agains pixels in layer below
				if (pixVal < minVal || pixVal > maxVal) {
					unsigned char* layerBelow = pyramid[i + 1].DoG;
					int xcoordBelow = k * BILINEAR_INTERPOLATION_SPACING;
					int ycoordBelow = j * BILINEAR_INTERPOLATION_SPACING;
					int widthBelow = pyramid[i + 1].width;
					int heightBelow = pyramid[i + 1].height;

					for (int r = -1; r <= 1; r++) {
						for (int s = -1; s <= 1; s++) {
							if (ycoordBelow + r > 0 && ycoordBelow + r < heightBelow && xcoordBelow + s > 0 && xcoordBelow + s < widthBelow) {
								neighborPixVal = layerBelow[(ycoordBelow + r) * widthBelow + (xcoordBelow + s)];
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

				// If the pixel value is still an extrema, check against pixels in layer above
				if (pixVal < minVal || pixVal > maxVal) {
					unsigned char* layerAbove = pyramid[i - 1].DoG;
					int xcoordAbove = k / BILINEAR_INTERPOLATION_SPACING;
					int ycoordAbove = j / BILINEAR_INTERPOLATION_SPACING;
					int widthAbove = pyramid[i - 1].width;
					int heightAbove = pyramid[i - 1].height;

					for (int r = -1; r <= 1; r++) {
						for (int s = -1; s <= 1; s++) {
							if (ycoordAbove + r > 0 && ycoordAbove + r < heightAbove && xcoordAbove + s > 0 && xcoordAbove + s < widthAbove) {
								neighborPixVal = layerAbove[(ycoordAbove + r) * widthAbove + (xcoordAbove + s)];
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

				// If the pixel is still an extrema, append it's coordinates (x, y, layer) to the keypoints linked list
				// x, y coordinates are relative to the input image bounds (top smallest layer of the pyramid)
				if (pixVal < minVal || pixVal > maxVal) {
					currentKeypoint->layer = i;
					// Bilinear interpolation spacing is constant between layers, so the coordinates
					// relative to the input image can be computed by division of the spacing to the layer'th power
					currentKeypoint->x = k / pow(BILINEAR_INTERPOLATION_SPACING, i);
					currentKeypoint->y = j / pow(BILINEAR_INTERPOLATION_SPACING, i);

					currentKeypoint->next = new keypoint;
					keypoint* temp_previous = currentKeypoint;
					currentKeypoint = currentKeypoint->next;
					currentKeypoint->previous = temp_previous;

					numKeypoints++;
				}
			}
		}
	}

	currentKeypoint->next = NULL;

	// Debug check, keypoints should be on the order of 1000s
	// printf("%d", numKeypoints);
}

void calculate_gradient(imagePyramidLayer pyramid[LAYERS]) {
	/*
	* Calculates gradient magnitude and gradient orientation maps for each layer of the pyramid.
	*
	* See D. Lowe, section 3.1
	*/

	// Iterate over each layer of the pyramid and generate the gradient maps
	// Gradient maps have a 1-1 relationship with the layer's pixel grid
	// Each entry in the map has the same coordinates as the pixel it represents, and holds a value
	// corresponding to the image gradient at that pixel.
	unsigned char belowPixVal, rightPixVal;
	for (int i = 0; i < LAYERS; i++) {
		// Create image maps and reference them with pointers in the imagePyramidLayer structs
		pyramid[i].gradientMagnitudeMap = (float*)malloc(pyramid[i].height * pyramid[i].width * sizeof(float));
		pyramid[i].gradientOrientationMap = (int*)malloc(pyramid[i].height * pyramid[i].width * sizeof(int));
		int width = pyramid[i].width;
		int height = pyramid[i].height;

		// Iterate over the newly created maps and compute the gradient magnitude and orientation at each pixel
		// location
		for (int j = 0; j < height - 1; j++) {
			for (int k = 0; k < width - 1; k++) {
				unsigned char pixVal = pyramid[i].imageA[j * width + k];
				belowPixVal = pyramid[i].imageA[(j + 1) * width + k];
				rightPixVal = pyramid[i].imageA[j * width + k + 1];
				pyramid[i].gradientMagnitudeMap[j * width + k] = sqrt((pixVal - belowPixVal) * (pixVal - belowPixVal) +
					(pixVal - rightPixVal) * (pixVal - rightPixVal));
				pyramid[i].gradientOrientationMap[j * width + k] = floor(atan2((pixVal - belowPixVal),
					(rightPixVal, pixVal)) * 360 / (2 * PI));
			}
		}

		// Fill in bottom row of map with values from row above it (boundary condition)
		for (int k = 0; k < width; k++) {
			pyramid[i].gradientMagnitudeMap[(height - 1) * width + k] = pyramid[i].gradientMagnitudeMap[(height - 2) * width + k];
			pyramid[i].gradientOrientationMap[(height - 1) * width + k] = pyramid[i].gradientOrientationMap[(height - 2) * width + k];
		}
		// Fill in right-most column of map with values from the adjacent column (boundary condition)
		for (int j = 0; j < height; j++) {
			pyramid[i].gradientMagnitudeMap[j * width + (width - 1)] = pyramid[i].gradientMagnitudeMap[j * width + (width - 2)];
			pyramid[i].gradientOrientationMap[j * width + (width - 1)] = pyramid[i].gradientOrientationMap[j * width + (width - 2)];
		}
	}
}

void characterize_keypoint(keypoint* keypoints, imagePyramidLayer pyramid[LAYERS]) {
	/*
	* Uses a histogram of local gradient orientations to determine the keypoint gradient orientation.
	* The histogram is built with a Gaussian-weighted window with 'sigma' of 3 times that of the current smoothing scale
	* The histogram has 36 bins covering the 360 degrees of possible rotation.
	*/

	// Histogram to collect local orientations per keypoint
	float orientationHistogram[36] = { 0 };

	// Iterate over each keypoint in the linked list, and compute the gradient values
	while (keypoints->next != NULL) {
		// grab pixel coordinates and image dimensions from the keypoint
		int layer = keypoints->layer;
		int xcoord = keypoints->x * pow(BILINEAR_INTERPOLATION_SPACING, layer);
		int ycoord = keypoints->y * pow(BILINEAR_INTERPOLATION_SPACING, layer);
		int width = pyramid[layer].width;
		int height = pyramid[layer].height;

		// A gaussian kernel is used for the weighted histogram
		// The 'sigma' parameter is based on the layers level of blur
		float sigma = (layer == 0) ? (3 * sqrt(2)) : (3 * sqrt(2) * layer * 2);
		int kernelWidth = 5;
		int kernelRadius = floor(kernelWidth / 2);
		float* gaussian_kernel = NULL;
		// Get the 2d gaussian kernel
		get_gaussian_kernel(&gaussian_kernel, sigma, kernelWidth);

		// Sort the surrounding gradient orientations in the histogram
		for (int i = -kernelRadius; i <= kernelRadius; i++) {
			for (int j = -kernelRadius; j <= kernelRadius; j++) {
				float weight = gaussian_kernel[(i + kernelRadius) * kernelWidth + (j + kernelRadius)];

				if (ycoord + i < width && ycoord + i > 0 && xcoord + j < width && xcoord + j > 0) {
					float orientation = pyramid[layer].gradientOrientationMap[(ycoord + i) * width + (xcoord + j)];
					int bin = floor(orientation / 10);
					orientationHistogram[bin] += weight;
				}
			}
		}

		// Find the peak of the gradient orientation histogram
		float maxVal = 0;
		int maxBin = 0;
		for (int i = 0; i < 36; i++) {
			if (orientationHistogram[i] > maxVal) {
				maxVal = orientationHistogram[i];
				maxBin = i;
			}
		}

		// Assign the gradient values to the keypoint
		keypoints->orientation = maxBin * 10;
		keypoints->magnitude = pyramid[layer].gradientMagnitudeMap[ycoord * width + xcoord];

		// Move to next keypoint in the linked list
		keypoints = keypoints->next;

		free(gaussian_kernel);
	}


}

void draw_keypoints(keypoint* keypoints, unsigned char* inputImage, unsigned char* outputImage, int width, int height) {
	/*
	* Draws keypoint annotations for keypoint orientation and location on input array.
	*/

	// Draw keypoint graphic template. This will be copied and rotated to match each keypoint orientation.
	unsigned char keypointGraphicWidth = 17;
	unsigned char* keypointGraphic = (unsigned char*)malloc(keypointGraphicWidth * keypointGraphicWidth * sizeof(unsigned char));
	for (int i = 0; i < keypointGraphicWidth; i++) {
		// Top line
		keypointGraphic[i] = 255;
		// Bottom line
		keypointGraphic[keypointGraphicWidth * (keypointGraphicWidth - 1) + i] = 255;
		// Left line
		keypointGraphic[keypointGraphicWidth * i] = 255;
		// Right line
		keypointGraphic[keypointGraphicWidth * i + (keypointGraphicWidth - 1)] = 255;
		// Center line segment
		if (i > (int)floor(keypointGraphicWidth / 2)) {
			keypointGraphic[keypointGraphicWidth * (int)floor(keypointGraphicWidth / 2) + i] = 255;
		}
	}

	// Create a mask for translated keypoint graphics
	// Width has to accomodate a keypoint graphic rotated 45 degrees
	unsigned char keypointGraphicTranlatedWidth = sqrt(2 * (keypointGraphicWidth * keypointGraphicWidth)) + 1;
	unsigned char* keypointGraphicTranslated = NULL;
	// Shift in coordinates between keypoint template, and translated keypoint graphic
	float coord_shift = (keypointGraphicTranlatedWidth - keypointGraphicWidth) / 2.0;

	// Declare rotation matrix, well known matrix for 2D rotations
	float rotationMatrix[2][2];

	// Store pointer to first keypoint to restore after iterating through linked list
	keypoint* firstKeypoint = keypoints;

	// Iterate through all keypoints and compute the rotated annotation image, reference final annotation map as pointer in keypoint
	while (keypoints->next != NULL) {

		// Create annotation map for keypoint
		keypointGraphicTranslated = (unsigned char*)malloc(keypointGraphicTranlatedWidth * keypointGraphicTranlatedWidth * sizeof(unsigned char));
		// Store pointer to map in keypoint
		keypoints->graphic = keypointGraphicTranslated;

		// Convert orientation to radians
		double orientation = (double)(keypoints->orientation) * (double)PI / 180;

		// Compute rotation matrix coefficients
		rotationMatrix[0][0] = cos(orientation);
		rotationMatrix[0][1] = -sin(orientation);
		rotationMatrix[1][0] = sin(orientation);
		rotationMatrix[1][1] = cos(orientation);

		// Load keypoint mask coordinates into Nx2 matrix (x, y)
		// Matrix math requires map to be in coordinate form for transformation
		// Y coordinates are relected to match a right-hand coordinate system
		int numCoordinates = keypointGraphicWidth * 2 + (keypointGraphicWidth - 2) * 2 + (int)floor(keypointGraphicWidth / 2) - 1;
		float* keypointCoordinates = (float*)malloc(2 * numCoordinates * sizeof(float));
		float* keypointCoordinatesTranslated = (float*)malloc(2 * numCoordinates * sizeof(float));
		int loadedCoordinates = 0;
		for (int i = 0; i < keypointGraphicWidth; i++) {
			for (int j = 0; j < keypointGraphicWidth; j++) {

				if (keypointGraphic[keypointGraphicWidth * i + j] == 255 && (loadedCoordinates < numCoordinates)) {
					keypointCoordinates[loadedCoordinates * 2 + 1] = -(i + coord_shift);
					keypointCoordinates[loadedCoordinates * 2] = j + coord_shift;
					loadedCoordinates++;
				}
			}
		}

		// Set origin to center of graphic
		// Without this translation the template image will be rotated about the top left corner
		for (int i = 0; i < numCoordinates; i++) {
			keypointCoordinatesTranslated[i * 2] = keypointCoordinates[i * 2] - (keypointGraphicTranlatedWidth / 2.0);
			keypointCoordinatesTranslated[i * 2 + 1] = keypointCoordinates[i * 2 + 1] + (keypointGraphicTranlatedWidth / 2.0);
		}

		// Rotate graphic
		// Matrix multiplication of the rotation matrix with the coordinates of the template image
		for (int i = 0; i < numCoordinates; i++) {
			float x_temp = keypointCoordinatesTranslated[i * 2];
			float y_temp = keypointCoordinatesTranslated[i * 2 + 1];
			keypointCoordinatesTranslated[i * 2] = rotationMatrix[0][0] * x_temp + rotationMatrix[0][1] * y_temp;
			keypointCoordinatesTranslated[i * 2 + 1] = rotationMatrix[1][0] * x_temp + rotationMatrix[1][1] * y_temp;
		}

		// Shift origin back to top left
		// This converts the coordinates back to pixel coordinates
		for (int i = 0; i < numCoordinates; i++) {
			keypointCoordinatesTranslated[i * 2] = keypointCoordinatesTranslated[i * 2] + (keypointGraphicTranlatedWidth / 2.0);
			keypointCoordinatesTranslated[i * 2 + 1] = -(keypointCoordinatesTranslated[i * 2 + 1] - (keypointGraphicTranlatedWidth / 2.0));
		}

		// Zero out the annotation map
		for (int i = 0; i < keypointGraphicTranlatedWidth; i++) {
			for (int j = 0; j < keypointGraphicTranlatedWidth; j++) {
				keypointGraphicTranslated[i * keypointGraphicTranlatedWidth + j] = 0.0;
			}
		}

		// Write the rotated annotation coordinates to the map
		for (int i = 0; i < numCoordinates; i++) {
			int xcoord = (int)round(keypointCoordinatesTranslated[i * 2]);
			int ycoord = (int)round(keypointCoordinatesTranslated[i * 2 + 1]);
			if (xcoord < keypointGraphicTranlatedWidth && ycoord < keypointGraphicTranlatedWidth) {
				keypointGraphicTranslated[ycoord * keypointGraphicTranlatedWidth + xcoord] = 255;
			}
		}

		// Free coordinate arrays
		free(keypointCoordinates);
		free(keypointCoordinatesTranslated);

		// Continue to next keypoint in linked list
		keypoints = keypoints->next;
	}

	// Reset linked list
	keypoints = firstKeypoint;

	// DEBUG ANNOTATION IMAGE EXPORTS
	// Export translated annotation mask
	// const char imgFileTranslated[] = "../util/KeypointTranslated.png";
	// stbi_write_png(imgFileTranslated, keypointGraphicTranlatedWidth, keypointGraphicTranlatedWidth, 1, 
	//    keypointGraphicTranslated, keypointGraphicTranlatedWidth * 1);

	// Export annotation mask template
	// const char imgFileOut[] = "../util/KeypointTemplate.png";
	// stbi_write_png(imgFileOut, keypointGraphicWidth, keypointGraphicWidth, 1, keypointGraphic, keypointGraphicWidth * 1);


	// Copy input image to output image
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			outputImage[y * width + x] = inputImage[y * width + x];
		}
	}

	// Copy annotations to image
	int annotationRadius = (keypointGraphicTranlatedWidth / 2);
	int annotationWidth = keypointGraphicTranlatedWidth;

	// For each keypoint, draw the annotation map on top of the output image
	int keypointCount = 0;
	while (keypoints->next != NULL) {
		// Don't draw keypoints that don't meet the gradient magnitude threshold
		//if (keypoints->magnitude < GRADIENT_MAGNITUDE_THRESHOLD) {
		//	keypoints = keypoints->next;
		//	continue;
		//}

		int x = keypoints->x - annotationRadius;
		int y = keypoints->y - annotationRadius;
		unsigned char* annotation = keypoints->graphic;
		for (int i = 0; i < annotationWidth; i++) {
			for (int j = 0; j < annotationWidth; j++) {
				if (annotation[i * annotationWidth + j] == 255 && (y + i < height) && (y+i > 0) && (x + j < width) && (x+j > 0)) {
					outputImage[(y + i) * width + (x + j)] = 128;
				}
			}
		}
		keypoints = keypoints->next;
		keypointCount++;
	}
}

void get_gaussian_kernel(float** kernel, float sigma, int kernelWidth) {
	/*
	* Generates a 2D gaussian kernel given a width and a 'sigma' parameter
	*/
	assert(kernelWidth % 2 == 1);
	assert(kernelWidth > 0);

	float* blurKernel1D = (float*)malloc(kernelWidth * sizeof(float));
	*kernel = (float*)malloc(kernelWidth * kernelWidth * sizeof(float));

	// Build 1D blur kernel
	int kernelRadius = floor(kernelWidth / 2);
	for (int i = 0; i < kernelWidth; i++) {
		int x = i - kernelRadius; // Center range of x around 0
		blurKernel1D[i] = (float)(1 / (sqrt(2 * PI) * sigma) * exp(-(x * x) / (2 * sigma * sigma)));
	}

	// Build 2D blur kernel from 1D kernel coefficients
	for (int k = 0; k < kernelWidth; k++) {
		for (int z = 0; z < kernelWidth; z++) {
			// The 2D kernel is radially symmetric, so the distance from center is used to
			// determine the 1D coefficient to use.
			int maxDistanceFromKernelCenter;
			if (abs(z - kernelRadius) > abs(k - kernelRadius)) {
				maxDistanceFromKernelCenter = abs(z - kernelRadius);
			}
			else {
				maxDistanceFromKernelCenter = abs(k - kernelRadius);
			}
			float val = blurKernel1D[kernelRadius - maxDistanceFromKernelCenter];
			(*kernel)[k * kernelWidth + z] = val;
		}
	}

	free(blurKernel1D);
}