#include <numeric>
#include "testMatching2D.hpp"

// Version control for the SIFT detector and extractor
// https://github.com/opencv/opencv/wiki/ChangeLog#version440
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR > 4)
    #define OPENCV_VERSION_GE_4_4
#endif

using namespace std;

// Find the best matches for the keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // Configure the matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    //cv::Ptr<cv::DescriptorMatcher> flann_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    //cv::FlannBasedMatcher flann_matcher(new cv::flann::LshIndexParams(20, 10, 2));
    //cv::FlannBasedMatcher flann_matcher(new cv::flann::KDTreeIndexParams());

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_HOG") == 0 ? cv::NORM_L2 : cv::NORM_HAMMING;

        if(descriptorType.compare("DES_HOG") == 0)
        {
            /*
            if (descSource.type() != CV_32F)
                descSource.convertTo(descSource, CV_32F);
            if (descRef.type() != CV_32F)
                descRef.convertTo(descRef, CV_32F);
            */
        }

        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {   
        // Comment out for the SIFT detector - BRIEF extractor
        if(descriptorType.compare("DES_HOG") == 0)
        {
            // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in the current OpenCV implementation
            //https://stackoverflow.com/questions/29694490/flann-error-in-opencv-3
            if (descSource.type() != CV_32F)
                descSource.convertTo(descSource, CV_32F);
            if (descRef.type() != CV_32F)
                descRef.convertTo(descRef, CV_32F);

            matcher = cv::FlannBasedMatcher::create();
        }

        else if(descriptorType.compare("DES_BINARY") == 0)
        {

            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);

            matcher = cv::FlannBasedMatcher::create();
        }

        cout << "FLANN matching" << endl;
        
    }

    // Perform the matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        const int k_size = 2;
        std::vector<std::vector<cv::DMatch>> knn_matches;
        //if(descriptorType.compare("DES_BINARY") == 0 && matcherType.compare("MAT_FLANN") == 0)
        if(matcherType.compare("MAT_FLANN") == 0)
        {
            //flann_matcher.knnMatch(descSource, descRef, knn_matches, k_size);
            //flann_matcher->knnMatch(descSource, descRef, knn_matches, k_size);
            matcher->knnMatch(descSource, descRef, knn_matches, k_size);
        }
        else
            matcher->knnMatch(descSource, descRef, knn_matches, k_size);
        // Filter matches by using descriptor distance ratio test
        const float rat_thr = 0.8f;
        int rem_pts = 0;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < rat_thr * knn_matches[i][1].distance)
                matches.push_back(knn_matches[i][0]);
            else
                rem_pts += 1;
        }
    }
}

// Use one of the several types of state-of-art descriptors to uniquely identify the keypoints
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType, double* t_desc)
{
    // Select an appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    //cv::Ptr<cv::AKAZE> akaze_extractor;
    double t = (double)cv::getTickCount();

    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else
    {
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT
        if(descriptorType.compare("BRIEF") == 0)
        {
            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        }
        else if(descriptorType.compare("ORB") == 0)
        {
            extractor = cv::ORB::create();
        }
        else if(descriptorType.compare("FREAK") == 0)
        {
            extractor = cv::xfeatures2d::FREAK::create();
        }
        else if(descriptorType.compare("AKAZE") == 0)
        {
            //t = (double)cv::getTickCount();
            //akaze_extractor->compute(img, keypoints, descriptors);
            //t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            extractor = cv::AKAZE::create();
            t = (double)cv::getTickCount();
            extractor->compute(img, keypoints, descriptors);
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        }
        else if(descriptorType.compare("SIFT") == 0)
        {
#ifdef OPENCV_VERSION_GE_4_4
            extractor = cv::SIFT::create();
#else
            extractor = cv::xfeatures2d::SIFT::create();
#endif
        }
    }

    // Perform the feature description
    if(descriptorType.compare("AKAZE") != 0)
    {
        t = (double)cv::getTickCount();
        extractor->compute(img, keypoints, descriptors);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    }
    
//cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    *t_desc = 1000 * t / 1.0;
}

// Detect keypoints in the image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double* t_det, bool bVis)
{
    // Compute detector parameters based on the image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply the corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // Add corners to the result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    
    //cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // Visualize the results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    *t_det = 1000 * t / 1.0;

}


// Detect keypoints in the image using the Harris detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double* t_det, bool bVis)
{
    double t = (double)cv::getTickCount();
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)
    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    // Perform the non-maximum suppression (NMS)
    double maxOverlap = 0.0;
    for(size_t j = 0; j < dst_norm.rows; j++)
    {
        for(size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if(response > minResponse)
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                bool bOverlap = false;
                for(auto it = keypoints.begin(); it != keypoints.end(); it++)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if(kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if(newKeyPoint.response > (*it).response)
                        {
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }
                if(!bOverlap)
                {
                    keypoints.push_back(newKeyPoint);
                }
            }
        }
    }
    
    //cout << "HARRIS detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    
    // visualize keypoints
    if(bVis)
    {
        string windowName = "Harris Corner Detection Results";
        cv::namedWindow(windowName, 5);
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
    *t_det = 1000 * t / 1.0;
}

// Detect keypoints in the image using modern detectors
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, double* t_det, bool bVis)
{
    double t = (double)cv::getTickCount();
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::AKAZE> akaze_detector;

    if(detectorType.compare("FAST")==0)
    {
        int fast_threshold = 30;
        bool use_nms = true;
        
        detector = cv::FastFeatureDetector::create(fast_threshold, use_nms);
    }
    else if(detectorType.compare("BRISK")==0)
        detector = cv::BRISK::create();

    else if(detectorType.compare("ORB")==0)
        detector = cv::ORB::create();

    else if(detectorType.compare("AKAZE")==0)
    {
        akaze_detector = cv::AKAZE::create();
    }

    else if(detectorType.compare("SIFT")==0)
    {
#ifdef OPENCV_VERSION_GE_4_4
        detector = cv::SIFT::create();
#else
        detector = cv::xfeatures2d::SIFT::create();
#endif
    }

    if(detectorType.compare("AKAZE")!=0)
    {
        t = (double)cv::getTickCount();
        detector->detect(img, keypoints);       //detect(imgGray, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    }
    else
    {
        t = (double)cv::getTickCount();
        akaze_detector->detect(img, keypoints); //detect(imgGray, keypoints)
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    }

    //cout << detectorType << " detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if(bVis)
    {
        // Visualize the results
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Results";
        cv::namedWindow(windowName, 1);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    *t_det = 1000 * t / 1.0;
}