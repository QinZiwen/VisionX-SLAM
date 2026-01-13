add_library(visionx_slam
    common/logger.cpp
)

target_include_directories(visionx_slam
    PUBLIC
        ${PROJECT_SOURCE_DIR}/core
)

target_link_libraries(visionx_slam
    PUBLIC
        Eigen3::Eigen
        Sophus::Sophus
        glog::glog
        gflags
        fmt::fmt
)
