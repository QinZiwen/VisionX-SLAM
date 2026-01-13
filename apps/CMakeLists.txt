add_executable(test_sophus test_sophus.cpp)
target_link_libraries(test_sophus PRIVATE Sophus::Sophus)

add_executable(test_pangolin test_pangolin.cpp)
target_link_libraries(test_pangolin PRIVATE
    pango_display  # 核心显示模块
    pango_core     # 核心基础模块
    pango_opengl   # OpenGL 支持
)

add_executable(mono_demo mono_demo.cpp)

target_link_libraries(mono_demo PRIVATE visionx_slam)
