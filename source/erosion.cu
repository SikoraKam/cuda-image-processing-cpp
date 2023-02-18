#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"

using namespace std;


/** @brief
* erosion on gpu. 
*  The lowest possible value from processed area is taken. The result in most cases should represent slimmer object
*
@param input
    array with image values
@param output
    array for result values
@param width
    width of image
@param height
    height of image
@param structSize
    indicates how many values should be processed together. The higher value the more visible effect
*/
__global__ void erosionCuda(unsigned char* input, unsigned char* output, int width, int height, int structSize) {
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (yIndex < height && xIndex < width) {
      
        int top = max(yIndex - structSize, 0);
        int bottom = min(height, yIndex + structSize);
        int left = max(xIndex - structSize, 0);
        int right = min(width, xIndex + structSize);
        int value = 255;
        for (int i = top; i <= bottom; i++) {
            for (int j = left; j <= right; j++) {
                value = min((float)value, (float)input[i * width + j]);
            }
        }
        output[yIndex * width + xIndex] = value;
    }
}


/** @brief
* erosion on gpu
*  the highest possible value from processed area is taken. The result in most cases should represent thicker object
*
@param input
    array with image values
@param output
    array for result values
@param width
    width of image
@param height
    height of image
@param structSize
    indicates how many values should be processed together. The higher value the more visible effect
*/
__global__ void dilatationCuda(unsigned char* input, unsigned char* output, int width, int height, int structSize) {
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (yIndex < height && xIndex < width) {

        int top = max(yIndex - structSize, 0);
        int bottom = min(height, yIndex + structSize);
        int left = max(xIndex - structSize, 0);
        int right = min(width, xIndex + structSize);
        int value = 0;
        for (int i = top; i <= bottom; i++) {
            for (int j = left; j <= right; j++) {
                value = max((float)value, (float)input[i * width + j]);
            }
        }
        output[yIndex * width + xIndex] = value;
    }
}


/** @brief
* this function alocates memory and calls image erosion on gpu
*
@param input
    object for openCV image format, contains data of processed image
@param output
    object for openCV image format where processed image array will be saved

@param structSizeS
    is number represented in string which indicates how many values should be processed together. The higher value the more visible effect

*/
void erosion(const cv::Mat& input, cv::Mat& output, string structSizeS) {

    int structSize = stoi(structSizeS);

    const int inputBytes = input.step * input.rows;
    const int outputBytes = output.step * output.rows;

    //Instantiate device pointers
    unsigned char* deviceInput, * deviceOutput;

    cudaMalloc(&deviceOutput, outputBytes);
    cudaMalloc(&deviceInput, inputBytes);

    //Specify a reasonable blockSize size
    const dim3 blockSize(32, 32);

    //Calculate gridSize size to cover the whole image
    const dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x, (input.rows + blockSize.y - 1) / blockSize.y);

    cudaMemcpy(deviceInput, input.ptr(), inputBytes, cudaMemcpyHostToDevice);

    erosionCuda << < gridSize, blockSize >> > (deviceInput, deviceOutput, input.cols, input.rows, structSize);

    cudaMemcpy(output.ptr(), deviceOutput, outputBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

}


/** @brief
* this function alocates memory and calls image dilatation on gpu
*
@param input
    object for openCV image format, contains data of processed image
@param output
    object for openCV image format where processed image array will be saved

@param structSizeS
    is number represented in string which indicates how many values should be processed together. The higher value the more visible effect

*/

void dilatation(const cv::Mat& input, cv::Mat& output, string structSizeS) {

    int structSize = stoi(structSizeS);


    const int inputBytes = input.step * input.rows;
    const int outputBytes = output.step * output.rows;

    unsigned char* deviceInput, * deviceOutput;

    cudaMalloc(&deviceOutput, outputBytes);
    cudaMalloc(&deviceInput, inputBytes);

    const dim3 blockSize(32, 32);

    const dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x, (input.rows + blockSize.y - 1) / blockSize.y);

    cudaMemcpy(deviceInput, input.ptr(), inputBytes, cudaMemcpyHostToDevice);

    dilatationCuda <<< gridSize, blockSize >>> (deviceInput, deviceOutput, input.cols, input.rows, structSize);

    cudaMemcpy(output.ptr(), deviceOutput, outputBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

}