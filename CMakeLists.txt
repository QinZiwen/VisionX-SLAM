cmake_minimum_required(VERSION 3.10)
project(VisionX-SLAM)

set(CMAKE_CXX_STANDARD 17)

find_package(Sophus REQUIRED)

add_subdirectory(apps)
