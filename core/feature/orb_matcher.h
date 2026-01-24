#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "feature/feature_matcher.h"
#include "frame/frame.h"

namespace visionx {

class ORBMatcher : public FeatureMatcher {
public:
    struct Options {
        float nn_ratio = 0.8f;
        int min_matches = 50;
    };

    ORBMatcher() : ORBMatcher(Options()) {}
    explicit ORBMatcher(const Options& options);

    int Match(const Frame::Ptr& last, const Frame::Ptr& curr,
              std::vector<cv::DMatch>& matches) override;

private:
    Options options_;
};

}  // namespace visionx
