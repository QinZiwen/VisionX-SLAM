#!/bin/bash

CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)
INSTALLED_DIR=${CURRENT_DIR}/third_party/installed

# pangolin
cd ${CURRENT_DIR}/third_party/pangolin
rm -rf build/*
cmake -B build -S . \
    -DCMAKE_INSTALL_PREFIX=${INSTALLED_DIR} \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_TESTS=OFF \
    -DBUILD_PANGOLIN_VIDEO=OFF \
    -DBUILD_PANGOLIN_PYTHON=OFF \
    -DBUILD_PANGOLIN_TESTS=OFF

cmake --build build -j8
cmake --install build