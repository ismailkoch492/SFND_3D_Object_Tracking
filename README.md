# SFND 3D Object Tracking

The camera course project has reached its conclusion. By completing all lessons, a comprehensive understanding has been attained regarding keypoint detectors, descriptors, and the techniques for matching them between successive images. Furthermore, proficiency has been acquired in object detection within images employing the YOLO deep-learning framework. Additionally, the capability to correlate regions within a camera image with LiDAR sensor points in 3D space has been developed. The program schematic has been examined to assess the achievements and identify any remaining components yet to be addressed.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, the missing parts in the schematic have been successfully implemented, involving the completion of four major tasks:
1. A method to match 3D objects over time by utilizing keypoint correspondences has been developed.
2. The Time-to-Collision (TTC) has been computed based on Lidar measurements
3. The TTC has also been computed using the camera. This required first associating keypoint matches to regions of interest and then calculating the TTC based on those matches.
4. Various tests have been conducted within the framework. The most suitable detector/descriptor combination for TTC estimation was identified, and potential issues leading to faulty measurements by the camera or LiDAR sensor were detected.

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
  * Install Git LFS before cloning this Repo.
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. -DCMAKE_BUILD_TYPE="Release" -DOPENCV_ENABLE_NONFREE=ON && make`

## Run It
1. Running the project: `./3D_object_tracking <DetType> <DescType>`. Replace the <DetType> <DescType> with detector and descriptor types, respectively.
2. Running the test: `./test_3D_object_tracking`

## Rubrics

### FP.1 Match 3D Objects

#### __Criteria__
- Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.

#### __Solution__
- The best matches are determined by whether they are in the bounding boxes. The number of points was first logged into a two-dimensional array, and the best matches were found by comparing the number of points.

```cpp
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
```

### FP.2 Compute Lidar-based TTC

#### __Criteria__
- Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.

#### __Solution__
- Time to collision is calculated according to the constant speed model. The lane size determines the leading car's points, and then the normal distribution with 1.44 z-score eliminates outliers.

``` cpp
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
```

### FP.3 Associate Keypoint Correspondences with Bounding Boxes

#### __Criteria__
- Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

#### __Solution__
- After the necessary keypoints are checked and determined with the bounding boxes, the distances between the determined key points are calculated. Outliers are eliminated according to the normal distribution with a z score of 1.44.

```cpp
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
```

### FP.4 Compute Camera-based TTC

#### __Criteria__
- Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

#### __Solution__
- The collision time is calculated according to the constant velocity model. The distances between the determined points are calculated according to the mapping of the keypoints. Find the median of the calculated distances. The median of the calculated distances determines the time to collision.

```cpp
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
```

### FP.5 Performance Evaluation 1

#### __Criteria__
- Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

#### __Solution__

![]( /plot/TTC-LiDAR.png "TTC-LiDAR")

- According to frames 3, 4, 5, 15, 16, and 17, time to collision increases but the distance of the preceding car decreases. Some of the points need to be filtered. These points belong to shiny surfaces like mirrors.


### FP.6 Performance Evaluation 2

#### __Criteria__
- Run several detector/descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

#### __Solution__

![]( /plot/TTC-Camera-AKAZE-FREAK.png "TTC Camera AKAZE-FREAK")

According to the graph, time to collision decreases more significantly. This is more robust than the LiDAR. FP6_Performance_Evaluation.csv file was obtained by running `test_3D_object_tracking`. It includes combinations of descriptors and detectors to calculate time to collision. The Harris-Freak description/detection combination produced some TTC-Camera values as `nan`. Furthermore, the HARRIS-SIFT, ORB-ORB, ORB-FREAK, and ORB-SIFT desc./det. combinations resulted in some TTC-Camera values as `inf`. Also, some TTC-Camera values were obtained as negative values due to the constant velocity model.

- Detectors: `SHITOMASI`, `HARRIS`, `FAST`, `BRISK`, `ORB`, `AKAZE`, `SIFT`
- Descriptors: `BRIEF`, `ORB`, `FREAK`, `AKAZE`, `SIFT`

Note: The SIFT-ORB det./desc combination doesn't run due to the `inv_scale > 0` error. To test this combination, all keypoint sizes detected by the SIFT detector were changed to `1.7`.

Top 3 Detector-Descriptor combinations according to minimum TTC value differences
- `SHITOMASI-ORB`
- `SIFT-SIFT`
- `AKAZE-FREAK`