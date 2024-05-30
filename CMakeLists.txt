cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(camera_fusion)

#SET(CMAKE_INCLUDE_PATH "/opt/local/include;${CMAKE_INCLUDE_PATH}")
#SET(CMAKE_LIBRARY_PATH "/opt/local/lib;${CMAKE_LIBRARY_PATH}")
set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV 4.1 REQUIRED)
# https://www.appsloveworld.com/cplus/100/1017/cmake-cannot-find-opencvconfig-cmake-even-after-specifying-opencv-dir
if (NOT "${OpenCV_FOUND}")
    unset(OpenCV_DIR)
    find_package(OpenCV 4.1 REQUIRED)
    # OpenCV_DIR = /lib/x86_64-linux-gnu/cmake/opencv4 -> OpenCVConfig.cmake
    message(STATUS "Checkout the solution: https://github.com/opencv/opencv/issues/21778")
endif ()

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Executable for create matrix exercise
add_executable (3D_object_tracking src/camFusion_Student.cpp src/FinalProject_Camera.cpp src/lidarData.cpp src/matching2D_Student.cpp src/objectDetection2D.cpp)
target_link_libraries (3D_object_tracking ${OpenCV_LIBRARIES})

add_executable (test_3D_object_tracking src/camFusion_Student.cpp src/test/testFinalProject_Camera.cpp src/lidarData.cpp src/test/testMatching2D_Student.cpp src/objectDetection2D.cpp)
target_link_libraries (test_3D_object_tracking ${OpenCV_LIBRARIES})
