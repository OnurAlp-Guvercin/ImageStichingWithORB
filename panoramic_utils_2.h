//
//  panoramic_utils_2.h
//  lab 7
//
//  Created by onuralp güvercin on 21.05.2023.
//

#ifndef panoramic_utils_2_h
#define panoramic_utils_2_h
#include <opencv2/core.hpp>

std::vector<cv::Point2f> computeTranslations(std::vector<cv::Mat> projectedImages, bool show_matches=false, float threshold=3.0);
std::vector<cv::Point2f> computeTranslations2(std::vector<cv::Mat> projectedImages);
cv::Mat createGlobalLandscape(const std::vector<cv::Mat>& projectedImages, const std::vector<cv::Point2f>& translations);

#endif /* panoramic_utils_2_h */
