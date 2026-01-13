cmake_minimum_required(VERSION 3.10)
project(VisionX-SLAM)

set(CMAKE_CXX_STANDARD 17)

find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
message(STATUS "Pangolin Found: ${Pangolin_FOUND}")
message(STATUS "Pangolin Libraries: ${Pangolin_LIBRARIES}")

add_subdirectory(apps)
