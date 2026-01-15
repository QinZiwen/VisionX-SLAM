#pragma once

#include "frame/frame.h"

namespace visionx {

class FeatureMatcher{
public:
    virtual ~FeatureMatcher() = default;
    virtual int Match(const Frame::Ptr& last,
                      const Frame::Ptr& curr,
                      std::vector<cv::DMatch>& matches) = 0;
};

} // namespace visionx
