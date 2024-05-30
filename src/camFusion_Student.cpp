#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Generate groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, 
                         float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // Loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // Assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // Project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // Pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // Shrink the current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // Check whether the point is within the current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // Check whether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // Add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, 
                   cv::Size imageSize, bool bWait)
{
    // Generate a topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // Generate a randomized color for the current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // Plot Lidar points into the top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // Find the enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // Draw the individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // Draw the enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // Augment the object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // Plot the distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // Visualize the image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// ToDo - Pass the zScore into the function
// Associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, 
                              std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    double meanDist = 0.0;
    std::vector<std::pair<double, cv::DMatch>> distMatch;
    
    for(auto& it: kptMatches)
    {
        auto currKptId = it.trainIdx;
        auto prevKptId = it.queryIdx;

        cv::KeyPoint currKpt = kptsCurr.at(currKptId);
        cv::KeyPoint prevKpt = kptsPrev.at(prevKptId);
        
        if(boundingBox.roi.contains(currKpt.pt) && 
           boundingBox.roi.contains(prevKpt.pt))
        {
            cv::Point trainPt = cv::Point(currKpt.pt.x, currKpt.pt.y);
            cv::Point queryPt = cv::Point(prevKpt.pt.x, prevKpt.pt.y);

            double _dist = std::sqrt( std::pow( (trainPt.x - queryPt.x), 2 ) + 
                                      std::pow( (trainPt.y - queryPt.y), 2 ) );
            meanDist += _dist;
            distMatch.push_back(std::make_pair(_dist, it));
        }
    }

    // https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
    // According to the 68–95–99.7 rule, 
    // Pr(μ - σ ≤ X ≤ μ + σ) ≈ 68.27%
    // Pr(μ - 1.44σ ≤ X ≤ μ + 1.44σ) ≈ 85.00%
    // Pr(μ - 1.5σ ≤ X ≤ μ + 1.5σ) ≈ 86.64%
    // Pr(μ - 1.96σ ≤ X ≤ μ + 1.96σ) ≈ 95.00%
    // Pr(μ - 2σ ≤ X ≤ μ + 2σ) ≈ 95.45%
    // Pr(μ - 2.5σ ≤ X ≤ μ + 2.5σ) ≈ 98.76%
    // Pr(μ - 3σ ≤ X ≤ μ + 3σ) ≈ 99.73%

    // https://towardsdatascience.com/why-1-5-in-iqr-method-of-outlier-detection-5d07fdc82097
    // IQR Scale = 1    -> Pr(μ - 2.025σ ≤ X ≤ μ + 2.025σ) ≈ 95.71%
    // IQR Scale = 1.5  -> Pr(μ - 2.7σ ≤ X ≤ μ + 2.7σ) ≈ 99.31%
    // IQR Scale = 1.7  -> Pr(μ - 3σ ≤ X ≤ μ + 3σ) ≈ 99.73%
    // IQR Scale = 2    -> Pr(μ - 3.375σ ≤ X ≤ μ + 3.375σ) ≈ 99.93%

    double stdDev = 0.0;
    const double zScore = 1.44; // 1.0, 1.44, 1.5, 1.96, 2.0, 2.025 (1 IQR), 2.5, 2.7 (1.5 IQR), 3.0

    if(distMatch.size() == 0)
    {
        return;
    }
    else
    {
        // Sort the calculated distances from smallest to largest
        std::sort(distMatch.begin(), distMatch.end());
        // Calculate the mean
        meanDist = meanDist / distMatch.size();
        // Calculate the standard deviation
        double _sqSum = 0.0;
        for(auto& it: distMatch)
        {
            _sqSum += (it.first - meanDist) * (it.first - meanDist);
        }
        stdDev = std::sqrt(_sqSum / (double)distMatch.size());
    }

    // Determine the lower and upper bound
    double lowUpLimit[] = {meanDist - zScore * stdDev, meanDist + zScore * stdDev}; // {lower, upper}

    for(auto& it: distMatch)
    {
        if(lowUpLimit[0] < it.first && it.first < lowUpLimit[1])
        {
            boundingBox.kptMatches.push_back(it.second);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in the successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    vector<double> distRatios;
    for (auto it = kptMatches.begin(); it != kptMatches.end() - 1; it++)
    {
        cv::KeyPoint kpOutCurr = kptsCurr.at(it->trainIdx);
        cv::KeyPoint kpOutPrev = kptsPrev.at(it->queryIdx);
        for (auto __it = kptMatches.begin() + 1; __it != kptMatches.end(); ++__it)
        {
            double minDist = 100.0; // min. required distance

            cv::KeyPoint kpInCurr = kptsCurr.at(__it->trainIdx);
            cv::KeyPoint kpInPrev = kptsPrev.at(__it->queryIdx); 

            double distCurr = cv::norm(kpOutCurr.pt - kpInCurr.pt);
            double distPrev = cv::norm(kpOutPrev.pt - kpInPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRat = distCurr / distPrev;
                distRatios.push_back(distRat);
            }
        }
    }

    size_t n = distRatios.size();
    
    if (n == 0)
    {
        TTC = NAN;
        return;
    }

    // Using the median for time to collision (TTC) is a more convenient method than using the mean
    // https://stackoverflow.com/questions/1719070/what-is-the-right-approach-when-using-stl-container-for-median-calculation
    double medDistRatio = 0.0;
    std::sort(distRatios.begin(), distRatios.end(), greater<double>());
    auto medId = n / 2;
    if(n % 2 == 1)
    {
        medDistRatio = distRatios[medId];
    }
    else 
    {
        medDistRatio = (distRatios[medId - 1] + distRatios[medId]) / 2.0;
    }

    double dT = 1.0 / frameRate;
    TTC = -dT / (1.0 - medDistRatio);

    std::cout << "TTC Camera: " << TTC << "\n";
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1.0 / frameRate;
    double laneWidth = 4.0;

    std::vector<std::vector<double> > PreCurX = {{}, {}}; // {Pre, Cur}

    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); it++)
    {
        if (abs(it->y) <= laneWidth / 2.0)
        {
            PreCurX[0].push_back(it->x);
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); it++)
    {
        if (abs(it->y) <= laneWidth / 2.0)
        {
            PreCurX[1].push_back(it->x);
        }
    }

    // https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
    // According to the 68–95–99.7 rule, 
    // Pr(μ - σ ≤ X ≤ μ + σ) ≈ 68.27%
    // Pr(μ - 1.44σ ≤ X ≤ μ + 1.44σ) ≈ 85.00%
    // Pr(μ - 1.5σ ≤ X ≤ μ + 1.5σ) ≈ 86.64%
    // Pr(μ - 1.96σ ≤ X ≤ μ + 1.96σ) ≈ 95.00%
    // Pr(μ - 2σ ≤ X ≤ μ + 2σ) ≈ 95.45%
    // Pr(μ - 2.5σ ≤ X ≤ μ + 2.5σ) ≈ 98.76%
    // Pr(μ - 3σ ≤ X ≤ μ + 3σ) ≈ 99.73%

    // https://towardsdatascience.com/why-1-5-in-iqr-method-of-outlier-detection-5d07fdc82097
    // IQR Scale = 1    -> Pr(μ - 2.025σ ≤ X ≤ μ + 2.025σ) ≈ 95.71%
    // IQR Scale = 1.5  -> Pr(μ - 2.7σ ≤ X ≤ μ + 2.7σ) ≈ 99.31%
    // IQR Scale = 1.7  -> Pr(μ - 3σ ≤ X ≤ μ + 3σ) ≈ 99.73%
    // IQR Scale = 2    -> Pr(μ - 3.375σ ≤ X ≤ μ + 3.375σ) ≈ 99.93%

    double stdDev[] = {0.0, 0.0};
    const double zScore = 1.44; // 1.0, 1.44, 1.5, 1.96, 2.0, 2.025 (1 IQR), 2.5, 2.7 (1.5 IQR), 3.0
    double meanX[] = {0.0, 0.0}; // {Pre, Cur}
    std::vector<std::vector<double> > lowUpLimit = {{}, {}}; // {lower, upper} // {meanDist - zScore * stdDev, meanDist + zScore * stdDev}; 

    if( PreCurX[0].size() > 0 && PreCurX[1].size() > 0)
    {
        double _size = 0.0;
        double _sqSum = 0.0;
        double _mean;
        for(auto it = 0; it < PreCurX.size(); it++)
        {
            // Calculate the mean
            _mean = 0.0;
            for(auto& __it: PreCurX[it])
            {
                _mean += __it;
            }
            _mean = _mean / PreCurX[it].size();

            // Calculate the standard deviation
            for(auto& __it: PreCurX[it])
            {
                _sqSum += (__it - _mean) * (__it - _mean);
                _size += 1.0;
            }
            
            stdDev[it] = std::sqrt(_sqSum / _size);

            // Determine the lower and upper bound
            lowUpLimit[it].push_back(_mean - zScore * stdDev[it]);
            lowUpLimit[it].push_back(_mean + zScore * stdDev[it]);
        }
    }
    else
    {
        TTC = NAN;
        return;
    }   

    // Calculate the means according to the filtered data
    for(auto it = 0; it < PreCurX.size(); it++)
    {
        double _size = 0.0;
        for(auto& __it: PreCurX[it])
        {
            if(lowUpLimit[it][0] < __it && __it < lowUpLimit[it][1])
            {
                meanX[it] += __it;
                _size += 1.0;
            }
        }
        meanX[it] = meanX[it] / _size;
    }

    TTC = meanX[1] * dT / (meanX[0] - meanX[1]);

    std::cout << "TTC Lidar: " << TTC << "\n";

}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, 
                        DataFrame &prevFrame, DataFrame &currFrame)
{
    unsigned long int Pts[prevFrame.boundingBoxes.size()][currFrame.boundingBoxes.size()] = {0};
    
    for(auto it = matches.begin(); it != matches.end(); it++)
    {
        std::pair<cv::KeyPoint, cv::KeyPoint> prevCurrKpt = {prevFrame.keypoints[it->queryIdx], currFrame.keypoints[it->trainIdx]};

        for(int i = 0; i < prevFrame.boundingBoxes.size(); i++)
        {
            if(prevFrame.boundingBoxes[i].roi.contains(prevCurrKpt.first.pt))
            {
                for(int j = 0; j < currFrame.boundingBoxes.size(); j++)
                {
                    if(currFrame.boundingBoxes[j].roi.contains(prevCurrKpt.second.pt))
                    {
                        Pts[i][j] += 1;
                    }
                }
            }
        }
    }

    for(int i = 0; i < prevFrame.boundingBoxes.size(); i++)
    {
        std::pair<int, int> countId = {0, -1};

        for(int j = 0; j < currFrame.boundingBoxes.size(); j++)
        {
            if(Pts[i][j] > countId.first)
            {
                countId.first = Pts[i][j];
                countId.second = j;
            }
        }
        bbBestMatches[i] = countId.second;
    }

}