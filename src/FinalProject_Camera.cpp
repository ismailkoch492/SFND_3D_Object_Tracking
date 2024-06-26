/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* OBTAIN THE DETECTOR AND DESCRIPTOR TYPES */
    string detector[] = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    string descriptor[] = {"BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};
    string det_type = "";     // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    string desc_type = "";        // BRIEF, ORB, FREAK, AKAZE, SIFT
    
    // Change the value to "true" if the given arguments will be handled
    bool err_handle = false;
    
    // Choose a detector type according the passed argument(s)
    if(argc > 1)
    {
        for(int i = 0; i < sizeof(detector) / sizeof(string); i++)
        {
            if(detector[i].compare(argv[1]) == 0)
            {
                det_type = argv[1];
                break;
            }
        }
        if(det_type.compare("") == 0)
        {
            cout << "The given detector argument does not exist. The default detector method will be used" << endl;
            det_type = "ORB";
        }      
    }
    else if(argc == 1)
    {
        cout << "No detector and descriptor arguments are passed. The default detector and descriptor methods will be used" << endl;
        det_type = "ORB";
        desc_type = "FREAK";
    }

    // Choose a descriptor type according the passed argument(s)
    if(argc > 2)
    {
        // The SIFT detector is incompatible with the ORB and the AKAZE descriptors
        bool err_sift = det_type.compare("SIFT") == 0 && (descriptor[1].compare(argv[2]) == 0 || descriptor[3].compare(argv[2]) == 0);
        // The AKAZE detector is only compatible with the AKAZE detector
        bool err_akaze = det_type.compare("AKAZE") == 0 && descriptor[3].compare(argv[2]) != 0;
        for(int i = 0; i < sizeof(descriptor) / sizeof(string); i++)
        {
            if(descriptor[i].compare(argv[2]) == 0)
            {
                if(err_sift && err_handle)
                {
                    cout << "The given descriptor argument is incompatible with the given detector type. The default descriptor method will be used" << endl;
                    desc_type = descriptor[4]; // SIFT
                    break;
                }
                if(err_akaze && err_handle)
                {
                    cout << "The given descriptor argument is incompatible with the given detector type. The default descriptor method will be used" << endl;
                    desc_type = descriptor[3]; // AKAZE
                    break;
                }
                desc_type = argv[2];
                break;
            }
        }
        if(desc_type.compare("") == 0)
        {
            cout << "The given descriptor argument does not exist. The default descriptor method will be used" << endl;
            desc_type = "FREAK";
        }
    }
    else if(argc == 2)
    {
        cout << "No detector argument is passed. The default descriptor method will be used." << endl;
        // The SIFT detector is incompatible with the ORB and the AKAZE descriptors
        if(det_type.compare("SIFT") == 0)
            desc_type = descriptor[4]; // BRIEF, FREAK, [SIFT]
        // The AKAZE detector is only compatible with the AKAZE detector
        else if(det_type.compare("AKAZE"))
            desc_type = descriptor[3];
        else
            desc_type = "FREAK";
    }

    /* INIT VARIABLES AND DATA STRUCTURES */
    // The path of the dataset
    string dataPath = "../";

    // The path of the camera images
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes LiDAR and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // The path of the object detection model
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // The path of the LiDAR points
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // The calibration matrices for the sensor fusion
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for LiDAR and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {
        bVis = true;
        /* LOAD IMAGE INTO BUFFER */

        // Assemble the filenames for the current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // Load the image from the file and convert it to grayscale
        std::cout << imgFullFilename << "\n" ;
        cv::Mat img = cv::imread(imgFullFilename);

        // Pass the image into the data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;        
        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

        cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


        /* CROP LIDAR POINTS */

        // Load 3D LiDAR points from the file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // Remove the LiDAR points based on the distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
    
        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        cout << "#3 : CROP LIDAR POINTS done" << endl;


        /* CLUSTER LIDAR POINT CLOUD */

        // Associate the LiDAR points with the camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // Visualize the 3D objects
        bVis = true;
        if(bVis)
        {
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
        }
        bVis = false;

        cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
        

        /* DETECT IMAGE KEYPOINTS */

        // Convert the current image into grayscale format
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // Extract 2D keypoints from the current image
        vector<cv::KeyPoint> keypoints; // generate an empty feature list for the current image
        string detectorType = det_type;

        bVis = false;

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, bVis);
        }
        else
        {
            if(detectorType.compare("HARRIS") == 0)
                detKeypointsHarris(keypoints, imgGray, bVis);
            else
                detKeypointsModern(keypoints, imgGray, detectorType, bVis);
        }

        bVis = false;

        // Keep only keypoints on the preceding vehicle
        bool bFocusOnVehicle = false;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            vector<cv::KeyPoint> rect_kpts;
            for(auto &itr : keypoints)
            {
                if(vehicleRect.contains(itr.pt)) 
                    rect_kpts.push_back(itr);
            }
            keypoints.clear();
            keypoints = rect_kpts;
        }

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // A solution for the inv_scale > 0 error
        if(det_type.compare("SIFT") == 0 && desc_type.compare("ORB") == 0)
        {
            double _small = 10000.00;
            std::vector<cv::KeyPoint> _kpt_vect;
            for(auto& a: keypoints)
            {
                cv::KeyPoint _kpt;
                _kpt.pt = cv::Point2f(a.pt);
                _kpt.size = 1.75;
                _kpt_vect.push_back(_kpt);
                if(_small > a.size)
                {
                    _small = a.size;
                }
            }
            keypoints = _kpt_vect;
            std::cout << std::endl << "The smallest keypoint: " << _small << std::endl;
        }

        // Push the keypoints of the current frame into the end of the data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        cout << "#5 : DETECT KEYPOINTS done" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */
        // Binary-string descriptors: ORB, BRIEF, BRISK, FREAK, AKAZE, etc.
        // Floating-point descriptors: SIFT, SURF, GLOH, etc.
        cv::Mat descriptors;
        string descriptorType = desc_type;
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

        // Push the descriptors of the current frame into the end of the data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;
        cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descType = "DES_BINARY"; // DES_BINARY, DES_HOG
            // https://answers.opencv.org/question/10046/feature-2d-feature-matching-fails-with-assert-statcpp/
            descType = (detectorType.compare("SIFT") == 0) || (descriptorType.compare("SIFT")) == 0 ? "DES_HOG" : "DES_BINARY";
            std::cout << "DetectorType: " << detectorType << " DescriptorType: " << descriptorType << std::endl;
            string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN
            if(matcherType.compare("MAT_FLANN") == 0)
                selectorType = "SEL_KNN";

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descType, matcherType, selectorType);

            // Store the keypoint matches for the current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            
            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            map<int, int> bbBestMatches;
            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
            //// EOF STUDENT ASSIGNMENT

            // Store the keypoint matches in the current data frame
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;

            cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


            /* COMPUTE TTC ON OBJECT IN FRONT */

            // Loop over all the BB match pairs
            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
            {
                // Find the bounding boxes associates with the current match
                BoundingBox *prevBB, *currBB;
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                {
                    if (it1->second == it2->boxID) // check whether current match partner corresponds to this BB
                    {
                        currBB = &(*it2);
                    }
                }

                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                {
                    if (it1->first == it2->boxID) // check whether current match partner corresponds to this BB
                    {
                        prevBB = &(*it2);
                    }
                }

                // Compute TTC for the current match
                if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have LiDAR points
                {
                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> compute time-to-collision based on LiDAR data (implement -> computeTTCLidar)
                    double ttcLidar; 
                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                    //// EOF STUDENT ASSIGNMENT

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                    double ttcCamera;
                    clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);                    
                    computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
                    //// EOF STUDENT ASSIGNMENT

                    bVis = true;
                    if (bVis)
                    {
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                        
                        char str[200];
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                        string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        cout << "Press key to continue to next frame" << endl;
                        cv::waitKey(0);
                    }
                    bVis = false;

                } // eof TTC computation
            } // eof loop over all BB matches            

        }

    } // eof loop over all images

    return 0;
}
