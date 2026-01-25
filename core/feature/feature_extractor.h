#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "frame/frame.h"

namespace visionx {

class FeatureExtractor {
public:
    using Ptr = std::shared_ptr<FeatureExtractor>;

    virtual ~FeatureExtractor() = default;
    virtual void Extract(Frame& frame) = 0;
};

}  // namespace visionx
