#pragma once

#include <Eigen/Core>
#include <Sophus/se3.hpp>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

#include "feature/feature_extractor.h"
#include "feature/feature_matcher.h"
#include "frame/frame.h"

namespace visionx {

class Tracking {
public:
    using KeyFrame = Frame;

    enum class State { INIT = 0, TRACKING_GOOD, TRACKING_BAD, LOST };

    struct Options {
        int min_matches = 50;
        int min_inliers = 30;
        int min_keyframe_inliers = 50;
        double min_parallax = 10.0;  // 像素
    };

    Tracking(const Options& options,
             std::shared_ptr<FeatureExtractor> extractor,
             std::shared_ptr<FeatureMatcher> matcher);

    // 前端主入口
    void ProcessFrame(Frame::Ptr frame);

    State GetState() const { return state_; }

private:
    // ===== 主流程拆分 =====
    void InitFrame(const Frame::Ptr& frame);
    bool Track();
    bool TrackLastFrame();
    void UpdateTrackingState();
    void HandleTrackingFailure();

    // ===== 核心算法 =====
    bool EstimatePoseByEssential(const Frame::Ptr& curr, const Frame::Ptr& last,
                                 const std::vector<cv::DMatch>& matches,
                                 int& inliers);

    double ComputeParallax(const Frame::Ptr& ref, const Frame::Ptr& curr,
                           const std::vector<cv::DMatch>& matches);

    bool NeedNewKeyFrame() const;
    void CreateKeyFrame();

    // ===== 三角化（暂不入 Map）=====
    Eigen::Matrix<double, 3, 4> ProjectionMatrix(const Sophus::SE3d& T_cw,
                                                 const Camera& cam) const;

    void TriangulateWithLastKeyFrame();

    Eigen::Vector3d TriangulatePoint(const Eigen::Matrix<double, 3, 4>& P1,
                                     const Eigen::Matrix<double, 3, 4>& P2,
                                     const Eigen::Vector2d& x1,
                                     const Eigen::Vector2d& x2) const;

private:
    State state_ = State::INIT;
    Options options_;

    Frame::Ptr current_frame_;
    Frame::Ptr last_frame_;
    Frame::Ptr last_keyframe_;

    int last_inliers_ = 0;
    double last_parallax_ = 0.0;

    std::shared_ptr<FeatureExtractor> extractor_;
    std::shared_ptr<FeatureMatcher> matcher_;
};

}  // namespace visionx
