#include "panoramic_utils.h"// Include the header file for any additional utility functions
#include "ORB.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/core/utils/filesystem.hpp>


std::vector<cv::Mat> load_Project_Images(cv::String directoryPath, cv::String pattern, bool showProjected){
    // Read and print the fies
    std::vector<cv::String> filenames;
    cv::utils::fs::glob(directoryPath, pattern, filenames, false, false);
    
    // Load & project all the images
    std::vector<cv::Mat> projectedImages;
    double angle = 33.0; //default angle was FoV 66
    for (const auto& filename : filenames){
            
        //load the image
        cv::Mat image = cv::imread(filename);
        if(image.empty()){
            std::cout << "Failed to load image: " << filename << std::endl;
            continue;
        }
        //perform cylindrical projection
        cv::Mat projectedImage = cylindricalProj(image, angle);
        //store projected images
        projectedImages.push_back(projectedImage);
    }

    // Show the projected the images
    if(showProjected == true){
        for (size_t i = 0; i < projectedImages.size(); ++i)
        {
            // Display the current projected image
            cv::imshow("Projected Image", projectedImages[i]);
            // Wait for a key press to proceed to the next image or exit the loop
            int key = cv::waitKey(0);
            if (key == 27) // Check if the ESC key is pressed (ASCII code 27)
                break;
            // Close the current image window
            cv::destroyWindow("Projected Image");
        }
    }
    return projectedImages;
};



// Function to perform cylindrical projection on an image
cv::Mat cylindricalProj(const cv::Mat& image, const double angle)
{
  cv::Mat tmp, result;
  cv::cvtColor(image, tmp, cv::COLOR_BGR2GRAY);  // Convert the input image to grayscale
  result = tmp.clone();  // Create a clone of the grayscale image

  double alpha(angle / 180 * CV_PI);  // Convert the input angle from degrees to radians
  double d((image.cols / 2.0) / tan(alpha));  // Calculate the distance from the image center to the projection plane
  double r(d / cos(alpha));  // Calculate the radius of the projection
  double d_by_r(d / r);  // Calculate a ratio used for efficiency
  int half_height_image(image.rows / 2);  // Calculate the half-height of the input image
  int half_width_image(image.cols / 2);  // Calculate the half-width of the input image

  // Loop over each pixel in the output image
  for (int x = -half_width_image + 1, x_end = half_width_image; x < x_end; ++x)
  {
    for (int y = -half_height_image + 1, y_end = half_height_image; y < y_end; ++y)
    {
      double x1(d * tan(x / r));  // Calculate the x-coordinate in the input image
      double y1(y * d_by_r / cos(x / r));  // Calculate the y-coordinate in the input image

      // Check if the calculated coordinates are within the bounds of the input image
      if (x1 < half_width_image &&
          x1 > -half_width_image + 1 &&
          y1 < half_height_image &&
          y1 > -half_height_image + 1)
      {
        // Assign the pixel value from the input image to the corresponding location in the output image
        result.at<uchar>(y + half_height_image, x + half_width_image) = tmp.at<uchar>(round(y1 + half_height_image), round(x1 + half_width_image));
      }
    }
  }

  return result;  // Return the resulting projected image
}
