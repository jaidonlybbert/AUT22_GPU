
#ifndef SIFT_HOST_H
#define SIFT_HOST_H

// Pi approximate for generating gaussian distribution, and image rotations
#define PI 3.14159265
// Layers of the image 'pyramid' used for approximating Laplacian of Gaussian for keypoint detection
#define LAYERS 4
// Spacing used for bilinear interpolation between pyramid layers (i.e. factor by which to scale the image)
#define BILINEAR_INTERPOLATION_SPACING 1.5
// Keypoints with a gradient magnitude below this threshold are not used
#define GRADIENT_MAGNITUDE_THRESHOLD 36

// Container for pyramid layer data
// Holds pointers to associated generated images and maps in memory
// All images and maps have the same dimensions per layer
struct imagePyramidLayer {
	unsigned char* imageA;      // Base image for the layer
	unsigned char* imageB;      // Blurred image for the layer
	unsigned char* DoG;         // Difference of Gaussian (A-B)
	float* gradientMagnitudeMap; // Map of same dimensions as layer
	int* gradientOrientationMap; // Map of same dimension as layer
	int height;                 // Hieght of images in layer
	int width;                  // Width of images in layer
	char layerOutputDir[17];    // Directory to save images for layer
};

// Linked list of identified keypoints
// Contains coordinates of keypoints in scale space (x, y, layer)
// Keypoints are characterized by the gradient magnitude and orientation
// A small graphical image is used to represent the orientation
struct keypoint {
	keypoint* next;         // Pointer to next keypoint in list
	keypoint* previous;     // Pointer to previous keypoint in list
	int x;		            // x coordinate relative to original image
	int y;                  // y coordinate realative to original image
	int layer;              // Pyramid layer index
	int orientation;        // Gradient orientation (0-350 with a 10 degree resolution)
	float magnitude;        // Gradient magnitude
	unsigned char* graphic; // Graphic representation of keypoint
};

// Forward declarations
void calculate_gradient(imagePyramidLayer pyramid[LAYERS]);
void find_extrema(imagePyramidLayer pyramid[LAYERS], keypoint** keypoints);
void characterize_keypoint(keypoint* keypoints, imagePyramidLayer pyramid[LAYERS]);
void get_gaussian_kernel(float** kernel, float sigma, int kernelWidth);
void draw_keypoints(keypoint* keypoints, unsigned char* inputImage, unsigned char* outputImage, int width, int height);
void bilinear_interpolate(unsigned char* inputArray, unsigned char** inputArrayExpanded, int inputWidth, int inputHeight, float spacing, int* resultWidth, int* resultHeight);
void gaussian_blur(unsigned char* img, unsigned char* outputArray, int inputWidth, int inputHeight, float sigma, int kernelWidth);
void matrix_subtract(unsigned char* A, unsigned char* B, unsigned char* C, int width, int height);

#endif