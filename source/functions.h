#pragma once
using namespace std;
void imageInversion(const cv::Mat& input, cv::Mat& output);
void binaryzation(const cv::Mat& input, cv::Mat& output, string threshold);
void filtering(const cv::Mat& input, cv::Mat& output, string filterKernelNumber, bool isGray);
void erosion(const cv::Mat& input, cv::Mat& output, string structSize);
void dilatation(const cv::Mat& input, cv::Mat& output, string structSize);