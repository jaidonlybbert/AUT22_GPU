#ifndef SIFT_DEVICE_CUH
#define SIFT_DEVICE_CUH
void dev_bilinear_interpolate(unsigned char* inputArray, unsigned char** inputArrayExpanded, int inputWidth, int inputHeight,
    float spacing, int* resultWidth, int* resultHeight);
void dev_gaussian_blur(unsigned char* img, unsigned char* outputArray, int inputWidth, int inputHeight, float sigma, int kernelWidth);
__global__ void bilinear_interpolate_kernel(unsigned char* input, unsigned char* output, float spacing, int inputWidth, int inputHeight, int resultWidth, int resultHeight);
__global__ void gaussian_blur_kernel(unsigned char* input, unsigned char* output, float* gaussianKernel, int height, int width, int kernelDim, float kernelSum);
__global__ void matrix_subtract_kernel(unsigned char* A, unsigned char* B, unsigned char* C, int height, int width);
__global__ void gradient_map_kernel(unsigned char* input, float* magnitudeMap, float* orientationMap, int height, int width);
#endif