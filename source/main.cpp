#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda_runtime.h>
#include "functions.h"

using namespace std;


/** @brief
* main function.
* Takes commadline arguments.
* First argument is image file name. Image should be located in project folder, in subfolder "images".
* Second argument is name of function to use in processing. Available names: filtering, erosion, dilatation, binaryzation, inversion.
* Third argument meaning is different basd on which functoin user chose.
* For filtering its case/kernel number. Kernels are stored in filterKernels.h.
 *  For erosion and dilatation it's size of struct element .
 * For binaryzation it's threshold.
 * 
 * Fourth argument available only for filtering determines if image should be converted to gray. If yes it should be true, else false.
 * 
 * example: lena.png filtering 2 false.
*/

int main(int argc, char** argv) {

    string imageName = argv[1];
    string functionName = argv[2];
    string option = argv[3]; // kernel number when user choosed filtering or struct element size when user choosed dilatation/erosion or threshold for binaryzation
    string isGrayScaleForFiltering = argv[4];

    string outputFileName = "./images/result.jpg";
    cv::Mat srcImage = cv::imread("./images/" + imageName);
    if (srcImage.empty())
    {
        cout << "Image not found: " << endl;
        return -1;
    }


    cout << "\ninput image size: " << srcImage.cols << " " << srcImage.rows << endl;
    cout << "\ninput image channels amount: " << srcImage.channels() << endl;
        
    // grayscale image declaration for some of proccesing functions
    cv::Mat grayMat;
    cv::cvtColor(srcImage, grayMat, cv::COLOR_RGB2GRAY);


    if (functionName == "filtering") {
        if (isGrayScaleForFiltering == "true") {
            cout << argv[4] << endl;

            cv::Mat outImage(grayMat.size(), grayMat.type());
            filtering(grayMat, outImage, option, true);
            imwrite(outputFileName, outImage);
        }
        else {
            // output image declaration
            cv::Mat outImage(srcImage.size(), srcImage.type());
            filtering(srcImage, outImage, option, false);
            imwrite(outputFileName, outImage);
        }

    }
    else if (functionName == "erosion") {
        // output image declaration
        cv::Mat outImage(grayMat.size(), grayMat.type());
        erosion(grayMat, outImage, option);
        imwrite(outputFileName, outImage);
    }
    else if (functionName == "dilatation") {
        // output image declaration
        cv::Mat outImage(grayMat.size(), grayMat.type());
        dilatation(grayMat, outImage, option);
        imwrite(outputFileName, outImage);
    }
    else if (functionName == "binaryzation") {
        // output image declaration
        cv::Mat outImage(grayMat.size(), grayMat.type());
        binaryzation(grayMat, outImage, option);
        imwrite(outputFileName, outImage);
    }
    else if (functionName == "inversion") {
        // output image declaration
        cv::Mat outImage(srcImage.size(), srcImage.type());
        imageInversion(srcImage, outImage);
        imwrite(outputFileName, outImage);
    }
    else {
        cout << "PROCESSING COMMAND NOT FOUND" << endl;
        return -404;
    }


    return 0;
}
