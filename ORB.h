//
//  Header.h
//  lab6
//
//  Created by onuralp güvercin on 12.05.2023.
//

#ifndef Header_h
#define Header_h
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

class ORB_FeatureDetector {
public:
    ORB_FeatureDetector(cv::Mat img1_, cv::Mat img2_);
    std::vector<cv::DMatch> compareImages(float k = 1.2, bool show_matches = false);
    std::vector<cv::Point2f> getKeypoints1() const;
    std::vector<cv::Point2f> getKeypoints2() const;
    
private:
    cv::Mat img1;
    cv::Mat img2;
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> descriptor;
    cv::BFMatcher matcher;
    cv::Mat descriptors1;
    cv::Mat descriptors2;
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> topMatches;
    float avgDistance;
    float similarityThreshold;
    
    
    void detectKeypoints();
    void computeDescriptors();
    void matchDescriptors();
    void selectTopMatches(float k);
    void drawFilteredMatches();
        
    
};

#endif /* Header_h */
