cmake -S . -B build \
    -DCMAKE_TOOLCHAIN_FILE=/Users/qinziwen/Tools/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j8