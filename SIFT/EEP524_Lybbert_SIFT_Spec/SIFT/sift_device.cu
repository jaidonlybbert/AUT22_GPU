
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*
* DEVICE PSEUDOCODE
*/

__global__ void dev_bilinear_interpolate(unsigned char *input, unsigned char *output, int spacing, int inputWidth, int inputHeight)
{
    // Collaboratively load input image into shared memory

    // Compute coordinates relative to input image

    // Calculate adjacent input-image pixel indices

    // Compute row-wise linear interpolation values

    // Compute final column-wise linear interpolation
    // and store in output image
}

__global__ void dev_gaussian_blur(unsigned char* input, unsigned char* output, float* gaussianKernel, int height, int width, int kernelDim, float kernelSum) {
    // Collaboratively load input image to shared memory

    // Iterate over area determined by kernelDim
       // check boundaries
        // Accumulate convolution sum

    // Calculate normalized convolution result
    // and store in output
}

__global__ void dev_matrix_subtract(unsigned char* A, unsigned char* B, unsigned char* C, int height, int width) {
    // Subtract corresponding values in A and B and store in C
}

__global__ void dev_gradient_map(unsigned char* input, float* magnitudeMap, float* orientationMap, int height, int width) {
    // collaboratively load input into shared memory
    // store right pixval in register
    // store below pixval in register
    // check boundaries
        // compute gradient magnitude and store in magnitudemap
        // compute gradient orientation and store in orientationmap
}



int dev_main()
{
    // Load image

    // Copy image to device

    // Allocate memory for expanded image on device

    // Execute bilinear interpolate kernel

    // For each layer in the pyramid

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
