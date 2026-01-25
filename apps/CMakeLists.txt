add_executable(test_sophus test_sophus.cpp)
target_link_libraries(test_sophus PRIVATE visionx_core)

add_executable(test_pangolin test_pangolin.cpp)
target_link_libraries(test_pangolin PRIVATE visionx_core
)

add_executable(mono_demo mono_demo.cpp)
target_link_libraries(mono_demo PRIVATE visionx_core)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE visionx_core)
