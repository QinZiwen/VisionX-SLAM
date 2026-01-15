#include "feature/orb_extractor.h"

namespace visionx {

ORBExtractor::ORBExtractor(int n_features,
                           float scale_factor,
                           int n_levels) {
    orb_ = cv::ORB::create(
        n_features,
        scale_factor,
        n_levels);
}

void ORBExtractor::Extract(Frame& frame) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat desc;

    orb_->detectAndCompute(
        frame.Image(),
        cv::noArray(),
        keypoints,
        desc);

    auto& features = frame.Features();
    features.clear();
    features.reserve(keypoints.size());

    for (const auto& kp : keypoints) {
        Feature f;
        f.position = Eigen::Vector2d(kp.pt.x, kp.pt.y);
        f.response = kp.response;
        features.emplace_back(f);
    }

    frame.Descriptors() = desc.clone();
}

} // namespace visionx
