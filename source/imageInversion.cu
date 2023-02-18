#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define TPB 32

/** @brief
* image inversion on gpu
*  each value of given image is substracted from highest possible ppixel value in rgb format
*
@param input
    array with image values
@param output
    array for result values
@param size of processed image (rows * columns * channels amount)

*/

__global__ void imageInversionCuda(unsigned char* input, unsigned char* output, int imageSize) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < imageSize) {
        output[index] = 255 - input[index];
    }
}

/** @brief 
* this function alocates memory and calls image inversion on gpu
* 
@param input
    object for openCV image format, contains data of processed image
@param output
    object for openCV image format where processed image array will be saved
    
*/
void imageInversion(const cv::Mat& input, cv::Mat& output) {

    const int inputBytes = input.step * input.rows;
    const int outputBytes = output.step * output.rows;

    unsigned char* deviceInput, * deviceOutput;

    cudaMalloc(&deviceOutput, outputBytes);
    cudaMalloc(&deviceInput, inputBytes);

    int imageSize = input.cols * input.rows * input.channels();
    int blocks = (imageSize + TPB - 1) / TPB;

    cudaMemcpy(deviceInput, input.ptr(), inputBytes, cudaMemcpyHostToDevice);

    imageInversionCuda <<< blocks, TPB >>> (deviceInput, deviceOutput, imageSize);

    cudaMemcpy(output.ptr(), deviceOutput, outputBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);



}