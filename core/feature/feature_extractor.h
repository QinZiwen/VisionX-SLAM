#pragma once

#include <vector>
#include <opencv2/core.hpp>

#include "frame/frame.h"

namespace visionx {

class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;

    virtual void Extract(
        Frame& frame,
        cv::Mat* descriptors = nullptr) = 0;
};

} // namespace visionx
