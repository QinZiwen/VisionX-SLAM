add_library(visionx_core)

target_compile_options(visionx_core PRIVATE 
    -Wall             # 开启几乎所有常用警告
    -Wextra           # 开启一些额外的进阶警告
    -Werror=reorder   # 强制要求初始化顺序
    -Werror=return-type # 如果函数没写 return 直接报错
)

file(GLOB_RECURSE ALL_SOURCE_FILES 
    "./*.cpp"
)
target_sources(visionx_core PRIVATE ${ALL_SOURCE_FILES})

target_include_directories(visionx_core PUBLIC ${PROJECT_SOURCE_DIR}/core)
target_link_libraries(visionx_core
    PUBLIC
        Eigen3::Eigen
        Sophus::Sophus
        glog::glog
        gflags
        fmt::fmt
        ${OpenCV_LIBS}
)
