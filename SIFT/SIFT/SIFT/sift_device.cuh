#ifndef SIFT_DEVICE_CUH
#define SIFT_DEVICE_CUH
void dev_bilinear_interpolate(unsigned char* inputArray, unsigned char** inputArrayExpanded, int inputWidth, int inputHeight,
    float spacing, int* resultWidth, int* resultHeight);
void dev_gaussian_blur(unsigned char* img, unsigned char* outputArray, int inputWidth, int inputHeight, float sigma, int kernelWidth);
void dev_matrix_subtract(unsigned char* dev_imgA, unsigned char* dev_imgB, unsigned char* dev_imgC, int width, int height);
void dev_calculate_gradient_maps(unsigned char* baseImage, float* gradientMagnitudeMap, int* gradientOrientationMap, int width, int height);
void dev_generate_key_masks(imagePyramidLayer pyramid[LAYERS],
    unsigned char* dev_OutputImage, int outputWidth, int outputHeight);
void dev_sift(unsigned char* dev_rawImage, unsigned char** dev_outputImage, int imgSize, int imgWidth, int imgHeight,
    int* outputWidth, int* outputHeight);
__global__ void bilinear_interpolate_kernel(unsigned char* input, unsigned char* output, float spacing, int inputWidth, int inputHeight, int resultWidth, int resultHeight, int inputTileWidth);
__global__ void gaussian_blur_kernel(unsigned char* input, unsigned char* output, float* gaussianKernel, int height, int width, int kernelDim);
__global__ void matrix_subtract_kernel(unsigned char* A, unsigned char* B, unsigned char* C, int height, int width);
__global__ void gradient_map_kernel(unsigned char* input, float* magnitudeMap, int* orientationMap, int height, int width);
__global__ void generate_key_mask_kernel(imagePyramidLayer pyramid[LAYERS], float* dev_gaussian_kernel,
    int layer, unsigned char* dev_OutputImage, int outputWidth, int outputHeight);
__global__ void orientation_histogram_kernel(int* keyMask, int* orientationMap, float* gaussianKernel, int width, int height, int x, int y, int idx);
__global__ void draw_keypoint_kernel(unsigned char* dev_keypoint_template, unsigned char* dev_OutputImage, int* dev_keymask, int idx,
    int row, int col, int layer, int outputWidth, int outputHeight);
#endif