#pragma once

#include <vector>
#include <opencv2/core.hpp>

#include "frame/frame.h"

namespace visionx {

class ORBMatcher {
public:
    struct Options {
        float nn_ratio = 0.8f;
        int min_matches = 50;
    };

    explicit ORBMatcher(const Options& options = Options());

    int Match(const Frame::Ptr& last,
              const Frame::Ptr& curr,
              std::vector<cv::DMatch>& matches);

private:
    Options options_;
};

} // namespace visionx
