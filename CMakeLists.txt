cmake_minimum_required(VERSION 3.10)
project(VisionX_SLAM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# find_program(CLANG_TIDY_EXE NAMES "clang-tidy" PATHS "/opt/homebrew/opt/llvm/bin")
# if(CLANG_TIDY_EXE)
#     message(STATUS "Clang-Tidy found: ${CLANG_TIDY_EXE}")
#     set(CMAKE_CXX_CLANG_TIDY 
#         ${CLANG_TIDY_EXE};
#         -checks=*,-modernize-use-trailing-return-type,-abseil-*,-llvm-header-guard,-llvmlibc-*;
#         -header-filter=${CMAKE_SOURCE_DIR}/core/.*;
#         -warnings-as-errors=*;
#     )
# endif()

find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(Pangolin CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
message(STATUS "OpenCV Found: ${OpenCV_FOUND}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Pangolin Found: ${Pangolin_FOUND}")
message(STATUS "    Pangolin Libraries: ${Pangolin_LIBRARIES}")

include_directories(core)

add_subdirectory(core)
add_subdirectory(apps)
