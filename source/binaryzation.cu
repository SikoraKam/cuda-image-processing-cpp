#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"


using namespace std;

#define TPB 32

/** @brief
* this function convert image values to 0 or 255, depending on given threshold
*
@param input
    object for openCV image format, contains data of processed image. Input should be in grayscale, channels amount should equals 1
@param output
    object for openCV image format where processed image array will be saved
@param imageSize
    size of image data (rows * columns)
 @param threshold
    number which will be threshold for dividing image values

*/
__global__ void binaryzationCuda(unsigned char* input, unsigned char* output, int imageSize, int threshold) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < imageSize) {

        if (input[index] < threshold) {
            output[index] = 0;
        }
        else {
            output[index] = 255;
        }
    }
}

/** @brief
* this function alocates memory and calls image binaryzation on gpu
*
@param input
    object for openCV image format, contains data of processed image. Input should be in grayscale, channels amount should equals 1
@param output
    object for openCV image format where processed image array will be saved
@param thresholdString
    number in string representation which will be threshold for dividing images values

*/
void binaryzation(const cv::Mat& input, cv::Mat& output, string thresholdString) {

    int threshold = stoi(thresholdString);

    const int inputBytes = input.step * input.rows;
    const int outputBytes = output.step * output.rows;


    unsigned char* deviceInput, * deviceOutput;

    cudaMalloc(&deviceOutput, outputBytes);
    cudaMalloc(&deviceInput, inputBytes);

    //without channel becasue Mat should be in grayscale
    int imageSize = input.cols * input.rows;
    int blocks = (imageSize + TPB - 1) / TPB;

    cudaMemcpy(deviceInput, input.ptr(), inputBytes, cudaMemcpyHostToDevice);

    binaryzationCuda << < blocks, TPB >> > (deviceInput, deviceOutput, imageSize, threshold);

    cudaMemcpy(output.ptr(), deviceOutput, outputBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);



}