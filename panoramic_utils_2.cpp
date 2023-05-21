//
//  panoramic_utils_2.cpp
//  lab 7
//
//  Created by onuralp güvercin on 21.05.2023.
//

#include "panoramic_utils_2.h"
#include "ORB.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/core/utils/filesystem.hpp>

std::vector<cv::Point2f> computeTranslations(std::vector<cv::Mat> projectedImages, bool show_matches, float threshold)
{
    // Calculate and show feature detection and matching
    std::vector<cv::Mat> transformedImages;
    std::vector<cv::Point2f> translations;

    for (size_t i = 0; i < projectedImages.size() - 1; i++) {
        cv::Mat mat1 = projectedImages[i];
        cv::Mat mat2 = projectedImages[i + 1];

        ORB_FeatureDetector ORB(mat1, mat2);

        std::vector<cv::DMatch> matches = ORB.compareImages(threshold, show_matches);
        std::vector<cv::Point2f> keypoints1 = ORB.getKeypoints1();
        std::vector<cv::Point2f> keypoints2 = ORB.getKeypoints2();

        std::vector<cv::Point2f> inlierPoints1, inlierPoints2;
        std::vector<uchar> inlierMask;

        // Filter out keypoints that don't have matches
        for (const cv::DMatch& match : matches) {
            inlierPoints1.push_back(keypoints1[match.queryIdx]);
            inlierPoints2.push_back(keypoints2[match.trainIdx]);
        }

        std::cout << "Number of keypoints in image 1: " << inlierPoints1.size() << std::endl;
        std::cout << "Number of keypoints in image 2: " << inlierPoints2.size() << std::endl;
        std::cout << "" << std::endl;

        // Find homography using RANSAC and obtain inlier points
        cv::findHomography(inlierPoints1, inlierPoints2, cv::RANSAC, 3.0, inlierMask);

        // Compute average translation using inlier points
        cv::Point2f translation(0, 0);
        int numInliers = 0;
        for (size_t j = 0; j < inlierMask.size(); j++) {
            if (inlierMask[j]) {
                translation += (inlierPoints2[j] - inlierPoints1[j]);
                numInliers++;
            }
        }

        if (numInliers > 0) {
            translation /= numInliers;
        }

        translations.push_back(translation);
    }
    
    // Display the estimated translations
        for (size_t i = 0; i < translations.size(); i++) {
            std::cout << "Translation between image " << i+1 << " and " << i+2 << ": (" << translations[i].x << ", " << translations[i].y << ")" << std::endl;
        }
    return translations;
}


std::vector<cv::Point2f> computeTranslations_alternative(std::vector<cv::Mat> projectedImages)
{
    std::vector<cv::Point2f> translations;

    for (size_t i = 0; i < projectedImages.size() - 1; i++) {
        cv::Mat mat1 = projectedImages[i];
        cv::Mat mat2 = projectedImages[i + 1];

        // Perform feature detection and matching
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
        orb->detectAndCompute(mat1, cv::noArray(), keypoints1, descriptors1);
        orb->detectAndCompute(mat2, cv::noArray(), keypoints2, descriptors2);

        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);

        std::vector<cv::Point2f> inlierPoints1, inlierPoints2;
        std::vector<uchar> inlierMask;

        // Filter out keypoints that don't have matches
        for (const cv::DMatch& match : matches) {
            inlierPoints1.push_back(keypoints1[match.queryIdx].pt);
            inlierPoints2.push_back(keypoints2[match.trainIdx].pt);
        }

        std::cout << "Number of keypoints in image 1: " << inlierPoints1.size() << std::endl;
        std::cout << "Number of keypoints in image 2: " << inlierPoints2.size() << std::endl;
        std::cout << "" << std::endl;

        // Find homography using RANSAC and obtain inlier points
        cv::Mat homography = cv::findHomography(inlierPoints1, inlierPoints2, cv::RANSAC, 3.0, inlierMask);

        // Compute average translation using inlier points
        cv::Point2f translation(0, 0);
        int numInliers = 0;
        for (size_t j = 0; j < inlierMask.size(); j++) {
            if (inlierMask[j]) {
                translation += (inlierPoints2[j] - inlierPoints1[j]);
                numInliers++;
            }
        }

        if (numInliers > 0) {
            translation /= numInliers;
        }

        translations.push_back(translation);
    }
    
    // Display the estimated translations
    for (size_t i = 0; i < translations.size(); i++) {
        std::cout << "Translation between image " << i+1 << " and " << i+2 << ": (" << translations[i].x << ", " << translations[i].y << ")" << std::endl;
    }
    
    return translations;
}



cv::Mat createGlobalLandscape(const std::vector<cv::Mat>& projectedImages, const std::vector<cv::Point2f>& translations)
{
    // Prepare a large empty image to merge the input images
    int outputWidth = projectedImages[0].cols;
    int outputHeight = projectedImages[0].rows;

    // Calculate the total width of the output image by summing the widths of all input images
    for (size_t i = 0; i < projectedImages.size(); i++) {
        cv::Point2f translation = translations[i];
        // Compute translation for the current image along the x-axis
        int tx = static_cast<int>(translation.x);
        // Take the absolute value of the translation along the x-axis
        tx = std::abs(tx);
        outputWidth += tx;
    }

    // Create an output image with the computed width and height
    cv::Mat outputImage(outputHeight, outputWidth, projectedImages[0].type(), cv::Scalar(0, 0, 0));

    // Merge the input images using the translations
    int x = 0; // Initialize the x-coordinate to track the current position
    int y = 0; // Initialize the y-coordinate to track the current position

    for (size_t i = 0; i < projectedImages.size(); i++) {
        cv::Mat subImage = projectedImages[i]; // Get the current input image
        cv::Point2f translation = translations[i]; // Get the translation for the current image
        int width = subImage.cols;
        int height = subImage.rows;

        // Compute translation for the current image along the x-axis
        int tx = static_cast<int>(translation.x);

        // Take the absolute value of the translation along the x-axis
        tx = std::abs(tx);

        // Compute the region of interest in the output image
        int roiX = std::max(x, 0); // The starting x-coordinate of the region of interest
        int roiY = std::max(y, 0); // The starting y-coordinate of the region of interest

        // Calculate the width and height of the region of interest based on the translation and available space
        int roiWidth = std::min(width - tx, outputWidth - roiX);
        int roiHeight = std::min(height, outputHeight - roiY);

        // Define the region of interest rectangles
        cv::Range rowRange(roiY, roiY + roiHeight);
        cv::Range colRange(roiX, roiX + roiWidth);

        // Copy the translated subimage to the output image
        cv::Mat roiOutputImage = outputImage(rowRange, colRange);
        subImage(cv::Range(0, roiHeight), cv::Range(std::max(-tx, 0), std::max(-tx, 0) + roiWidth)).copyTo(roiOutputImage);

        // Update the current x-coordinate for the next image
        x += tx;
    }

    // Resize the output image to a larger size
    cv::resize(outputImage, outputImage, cv::Size(outputWidth * 2, outputHeight * 2));

    return outputImage;
}
