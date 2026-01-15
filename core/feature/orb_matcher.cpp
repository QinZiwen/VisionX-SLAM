#include "feature/orb_matcher.h"

#include <opencv2/features2d.hpp>
#include <glog/logging.h>

namespace visionx {

ORBMatcher::ORBMatcher(const Options& options)
    : options_(options) {}

int ORBMatcher::Match(const Frame::Ptr& last,
                      const Frame::Ptr& curr,
                      std::vector<cv::DMatch>& matches) {
    matches.clear();

    const auto& desc1 = last->Descriptors();
    const auto& desc2 = curr->Descriptors();

    if (desc1.empty() || desc2.empty()) {
        return 0;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;

    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    for (const auto& knn : knn_matches) {
        if (knn.size() < 2) continue;

        const auto& m1 = knn[0];
        const auto& m2 = knn[1];

        if (m1.distance < options_.nn_ratio * m2.distance) {
            matches.push_back(m1);
        }
    }

    if (matches.size() < options_.min_matches) {
        LOG(WARNING) << "[ORBMatcher] Too few matches: "
                     << matches.size();
    }

    return static_cast<int>(matches.size());
}

} // namespace visionx
