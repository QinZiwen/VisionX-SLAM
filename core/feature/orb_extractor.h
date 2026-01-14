#pragma once

#include "feature/feature_extractor.h"

#include <opencv2/features2d.hpp>

namespace visionx {

class ORBExtractor : public FeatureExtractor {
public:
    ORBExtractor(int n_features = 1000,
                 float scale_factor = 1.2f,
                 int n_levels = 8);

    virtual void Extract(
        Frame& frame,
        cv::Mat* descriptors = nullptr) override;

private:
    cv::Ptr<cv::ORB> orb_;
};

} // namespace visionx
