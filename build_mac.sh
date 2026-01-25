#!/bin/bash

BUILD_DIR="build"

if [ "$1" == "RemoveALL" ]; then
    echo "Full reset: removing build directory..."
    rm -rf "$BUILD_DIR"
elif [ "$1" == "Clean" ]; then
    echo "Cleaning build artifacts..."
    if [ -d "$BUILD_DIR" ]; then
        cd "$BUILD_DIR"
        rm -rf Makefile apps cmake_install.cmake compile_commands.json core third_party CMakeCache.txt CMakeFiles
        cd ..
    fi
else
    echo "Standard build..."
fi

mkdir -p "$BUILD_DIR"

export VCPKG_LIBRARY_LINKAGE=dynamic
export SDKROOT=$(xcrun --show-sdk-path)

cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE=/Users/qinziwen/Tools/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DVCPKG_TARGET_TRIPLET=arm64-osx-dynamic \
    -DCMAKE_BUILD_TYPE=Debug

cmake --build "$BUILD_DIR" -j8
