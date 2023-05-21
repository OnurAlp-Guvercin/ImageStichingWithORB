#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <vector>

#include "panoramic_utils_2.h"
#include "panoramic_utils.h"
#include "ORB.h"

int main()
{
    cv::String directoryPath = "/Users/onuralpguvercin/Desktop/computer_vision_unipd/labs/Lab_7_Image stitching-20230513/Lab_7-Image_stitching_resources/Images/dolomites";
    cv::String pattern = "*.png";
    std::vector<cv::Mat> projectedImages;

    // Load image paths and get the projected images
    projectedImages = load_Project_Images(directoryPath, pattern, false);

    std::vector<cv::Point2f> translations = computeTranslations(projectedImages, false, 3.0);

    cv::Mat outputImage = createGlobalLandscape(projectedImages, translations);

    // Display the final panorama
    cv::imshow("Panorama", outputImage);
    cv::waitKey(0);

    return 0;
}







