#pragma once

#include <memory>

#include "frame/frame.h"
#include "feature/feature_extractor.h"
#include "feature/feature_matcher.h"

namespace visionx {

class Tracking {
public:
    enum class State {
        INIT = 0,
        TRACKING_GOOD,
        TRACKING_BAD,
        LOST
    };

    struct Options {
        int min_matches = 50;        // 最小匹配数
        int min_inliers = 30;        // 最小 PnP 内点
        double max_reproj_error = 3.0;
    };

    Tracking(const Options& options,
             std::shared_ptr<FeatureExtractor> extractor,
             std::shared_ptr<FeatureMatcher> matcher);

    // 前端主入口
    void ProcessFrame(Frame::Ptr frame);

    State GetState() const { return state_; }

private:
    // ===== Phase 1 核心逻辑 =====
    void InitFrame(const Frame::Ptr& frame);
    bool TrackLastFrame(const Frame::Ptr& frame, int& inliers);

    bool EstimatePoseByEssential(const Frame::Ptr& curr,
                                 const Frame::Ptr& last,
                                 const std::vector<cv::DMatch>& matches,
                                 int& inliers);


private:
    State state_ = State::INIT;
    Options options_;

    Frame::Ptr current_frame_;
    Frame::Ptr last_frame_;

    std::shared_ptr<FeatureExtractor> extractor_;
    std::shared_ptr<FeatureMatcher> matcher_;
};

} // namespace visionx
