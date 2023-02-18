#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "filtersKernels.h"
using namespace std;


/** @brief this function assign values to given parameters
* 
* @param number
*   number of case which user choosed to convert image
* 
* @param radius
*   kernel edge value distance from central element
* 
* @param weight
*   values from image will be divided by weight. This is workaround for having fractions in kernel
* 
* @param dynamicArrayLength
*   length of kernel array it will be useful in image processing (we cannot read length of dynamic array so we save it here)
* 
* @retval int pointer
*   returns pointer to array from seperated file. This is workaround, because we cannot asign array to other array

*/

int* getFilterKernel(int number, int &radius, int &weight, int &dynamicArrayLength) {
    switch (number) {
    case 1:
        radius = radius1;
        weight = weight1;
        dynamicArrayLength = sizeof(kernel1) / sizeof(kernel1[0]);
        return kernel1;
        break;
    case 2:
        radius = radius2;
        weight = weight2;
        dynamicArrayLength = sizeof(kernel2) / sizeof(kernel2[0]);

        return kernel2;
        break;
    case 3:
        radius = radius3;
        weight = weight3;
        dynamicArrayLength = sizeof(kernel3) / sizeof(kernel3[0]);

        return kernel3;
        break;
    case 4:
        radius = radius4;
        weight = weight4;
        dynamicArrayLength = sizeof(kernel4) / sizeof(kernel4[0]);

        return kernel4;
        break;
    case 5:
    
        radius = radius5;
        weight = weight5;
        dynamicArrayLength = sizeof(kernel5) / sizeof(kernel5[0]);

        return kernel5;
        break;
    case 6:

        radius = radius6;
        weight = weight6;
        dynamicArrayLength = sizeof(kernel6) / sizeof(kernel6[0]);

        return kernel6;
        break;
    default: 
       
        radius = radius1;
        weight = weight1;
        dynamicArrayLength = sizeof(kernel1) / sizeof(kernel1[0]);

        return kernel1;
    }

}

/** @brief
* filters image on gpu. 
* Based on chossen kernel, suitable values are taken from image and multiplied by values from kernel.
* Function includes movement in caluculating indexes because of 3 channels in image
*
@param input
    object for openCV image format, contains data of processed image. Input should be in grayscale, channels amount should equals 1
@param output
    object for openCV image format where processed image array will be saved
@param width
    width of image
@param height
    height of image
*
* @param radius
*   kernel edge value distance from central element
*
* @param weight
*   values from image will be divided by weight. This is workaround for having fractions in kernel
* 
* @rowLength
*   bytes for one images row
* 
* @param filterKernel
*   poniter to array of filtering kernel
*

*/
__global__ void filteringCudaRGB(unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int radius, int weight, int rowLength, int * filterKernel) {

    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if ((xIndex < width) && (yIndex < height)) {
       
        int accChannel1 = 0;
        int accChannel2 = 0;
        int accChannel3 = 0;

        int outputId = (3 * xIndex) + yIndex * rowLength;

        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {

                // assign value from filter kernel
                int temp = filterKernel[i + radius + (j + radius)];

                //get pixel from input (3 for 3 channels)
                int currInputId =  (3 * (xIndex + i)) + (yIndex + j) * rowLength;

                //get values for each channel
                const unsigned char channel1 = input[currInputId];
                const unsigned char channel2 = input[currInputId + 1];
                const unsigned char channel3 = input[currInputId + 2];

                //sum piixel multiplied by filter value
                accChannel1 += int(channel1) * temp;
                accChannel2 += int(channel2) * temp;
                accChannel3 += int(channel3) * temp;
            }
        }

        accChannel1 = accChannel1 / weight;
        accChannel2 = accChannel2 / weight;
        accChannel3 = accChannel3 / weight;

        // convert value to unsigned char expected by Mat type
        output[outputId] = (unsigned char)(accChannel1);
        output[outputId + 1] = (unsigned char)(accChannel2);
        output[outputId + 2] = (unsigned char)(accChannel3);

    }

}

/** @brief
* filters image on gpu.
* Based on chossen kernel, suitable values are taken from image and multiplied by values from kernel
*
@param input
    object for openCV image format, contains data of processed image. Input should be in grayscale, channels amount should equals 1
@param output
    object for openCV image format where processed image array will be saved
@param width
    width of image
@param height
    height of image
*
* @param radius
*   kernel edge value distance from central element
*
* @param weight
*   values from image will be divided by weight. This is workaround for having fractions in kernel
*
* @rowLength
*   bytes for one images row
*
* @param filterKernel
*   poniter to array of filtering kernel
*

*/
__global__ void filteringCudaGray(unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int radius, int weight, int rowLength, int* filterKernel) {

    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if ((xIndex < width) && (yIndex < height)) {

        int accChannel1 = 0;
        

        int outputId = yIndex * rowLength + xIndex;

        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {

                // assign value from filter kernel
                int temp = filterKernel[i + radius + (j + radius)];

                //get pixel from input (3 for 3 channels)
                int currInputId = ((xIndex + i)) + (yIndex + j) * rowLength;

                //get values for each channel
                const unsigned char channel1 = input[currInputId];
    
                //sum piixel multipled by filter value
                accChannel1 += int(channel1) * temp;
     
            }
        }
        accChannel1 = accChannel1 / weight;

        // convert value to unsigned char expected by Mat type
        output[outputId] = (unsigned char)(accChannel1);

    }

}

/** @brief
* this function alocates memory and calls image filtering on gpu.
* For grayscale images filteringCudaGray is called.
* For rgb images filteringCudaRGB is called
*
@param input
    object for openCV image format, contains data of processed image. Input should be in grayscale, channels amount should equals 1
@param output
    object for openCV image format where processed image array will be saved
@param filterKernelNumber
    number in string representation which will determine choosen kernel for processing
@param isGray
    true for grascale images, false for rgb (3 channel images)

*/

void filtering(const cv::Mat& input, cv::Mat& output, string filterKernelNumber, bool isGray ) {

    const int inputBytes = input.step * input.rows;
    const int outputBytes = output.step * output.rows;

    //Instantiate device pointers
    unsigned char* deviceInput, * deviceOutput;

    cudaMalloc(&deviceOutput, outputBytes);
    cudaMalloc(&deviceInput, inputBytes);

    const dim3 blockSize(32, 32);

    const dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x, (input.rows + blockSize.y - 1) / blockSize.y);

    int radius;
    int weight;
    int dynamicArrayLength;

    int* kernelArray = getFilterKernel(stoi(filterKernelNumber), radius, weight, dynamicArrayLength);

    int* deviceKernelArray;
    cudaMalloc(&deviceKernelArray, dynamicArrayLength* sizeof(int));
    cudaMemcpy(deviceKernelArray, kernelArray, dynamicArrayLength * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(deviceInput, input.ptr(), inputBytes, cudaMemcpyHostToDevice);

    if (isGray) {
        filteringCudaGray << < gridSize, blockSize >> > (deviceInput, deviceOutput, input.cols, input.rows, radius, weight, input.cols * input.channels() * sizeof(unsigned char), deviceKernelArray);
    }
    else {
        filteringCudaRGB << < gridSize, blockSize >> > (deviceInput, deviceOutput, input.cols, input.rows, radius, weight, input.cols * input.channels() * sizeof(unsigned char), deviceKernelArray);

    }

    cudaMemcpy(output.ptr(), deviceOutput, outputBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);



}