if [ "$1" == "remove" ]; then
    echo "remove build environment ..."
    rm -rf build/*
else
    echo "build ..."
fi

export VCPKG_LIBRARY_LINKAGE=dynamic
export SDKROOT=$(xcrun --show-sdk-path)
cmake -S . -B build \
    -DCMAKE_TOOLCHAIN_FILE=/Users/qinziwen/Tools/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DVCPKG_TARGET_TRIPLET=arm64-osx-dynamic \
    -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j8