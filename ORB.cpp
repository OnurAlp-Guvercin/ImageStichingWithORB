//
//  ORB_BruteForce.cpp
//  lab6
//
//  Created by onuralp güvercin on 12.05.2023.
//


#include "ORB.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

//-------------------------- public --------------------------
ORB_FeatureDetector::ORB_FeatureDetector(cv::Mat img1_, cv::Mat img2_)
    :
      img1(std::move(img1_)),
      img2(std::move(img2_)),
      detector(cv::ORB::create()),
      descriptor(cv::ORB::create()),
      matcher(cv::BFMatcher(cv::NORM_HAMMING)),
      similarityThreshold(50.0f)
{}
std::vector<cv::DMatch> ORB_FeatureDetector::compareImages(float k, bool show_matches) {
    
    detectKeypoints();
    computeDescriptors();
    matchDescriptors();
    selectTopMatches(k);
    if(show_matches == true){
        drawFilteredMatches();
    }
    return topMatches;
}

std::vector<cv::Point2f> ORB_FeatureDetector::getKeypoints1() const {
    std::vector<cv::Point2f> keypoints;
    for (const cv::KeyPoint& keypoint : keypoints1) {
        keypoints.push_back(keypoint.pt);
    }
    return keypoints;
}
std::vector<cv::Point2f> ORB_FeatureDetector::getKeypoints2() const {
    std::vector<cv::Point2f> keypoints;
    for (const cv::KeyPoint& keypoint : keypoints2) {
        keypoints.push_back(keypoint.pt);
    }
    return keypoints;
}
//-------------------------- private --------------------------
void ORB_FeatureDetector::detectKeypoints(){
    detector -> detect(img1, keypoints1);
    detector -> detect(img2, keypoints2);
    
}
void ORB_FeatureDetector::computeDescriptors(){
    descriptor->compute(img1, keypoints1, descriptors1);
    descriptor->compute(img2, keypoints2, descriptors2);
}
void ORB_FeatureDetector::matchDescriptors(){
    matcher.match(descriptors1, descriptors2, matches);
}
void ORB_FeatureDetector::selectTopMatches(float k) {
    //find minimum distance
    float minDistance = std::numeric_limits<float>::max();
    for(const cv::DMatch& match : matches){
        if(match.distance <= minDistance){
            minDistance = match.distance;
        }
    }
    std::cout << minDistance;
    //refine matches based on threshold
    for(const cv::DMatch& match : matches){
        if(match.distance <= k * minDistance){
            topMatches.push_back(match);
        }
    }
}
void ORB_FeatureDetector::drawFilteredMatches(){

    cv::Mat imgMatches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, topMatches, imgMatches);
    cv::namedWindow("Matches", cv::WINDOW_NORMAL);
    cv::imshow("Matches", imgMatches);
    cv::waitKey(0);
}
