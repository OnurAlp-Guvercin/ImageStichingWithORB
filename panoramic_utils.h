#ifndef LAB5__PANORAMIC__UTILS__H
#define LAB5__PANORAMIC__UTILS__H

#include <opencv2/core.hpp>

cv::Mat cylindricalProj(const cv::Mat& image, const double angle);
std::vector<cv::Mat> load_Project_Images(cv::String directoryPath, cv::String pattern, bool showProjected);
#endif // LAB5__PANORAMIC__UTILS__H
