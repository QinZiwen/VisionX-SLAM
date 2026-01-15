#include "frontend/tracking.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <glog/logging.h>

namespace visionx {

Tracking::Tracking(const Options& options,
                   std::shared_ptr<FeatureExtractor> extractor,
                   std::shared_ptr<FeatureMatcher> matcher)
    : options_(options),
      extractor_(std::move(extractor)),
      matcher_(std::move(matcher)) {}

void Tracking::ProcessFrame(Frame::Ptr frame) {
    current_frame_ = frame;

    // 提取特征
    extractor_->Extract(*current_frame_);

    int inliers = 0;
    bool success = false;

    switch (state_) {
    case State::INIT:
        InitFrame(frame);
        return;

    case State::TRACKING_GOOD:
    case State::TRACKING_BAD:
        success = TrackLastFrame(frame, inliers);
        break;

    case State::LOST:
        LOG(WARNING) << "[Tracking] LOST. Frame ignored.";
        return;
    }

    // ===== 状态更新策略 =====
    if (!success) {
        if (state_ == State::TRACKING_GOOD) {
            state_ = State::TRACKING_BAD;
        } else {
            state_ = State::LOST;
        }
        return;
    }

    if (inliers >= options_.min_inliers * 2) {
        state_ = State::TRACKING_GOOD;
    } else {
        state_ = State::TRACKING_BAD;
    }

    last_frame_ = current_frame_;
}

void Tracking::InitFrame(const Frame::Ptr& frame) {
    // 第一帧：位姿设为单位阵
    frame->SetPose(Sophus::SE3d());
    last_frame_ = frame;
    state_ = State::TRACKING_GOOD;

    LOG(INFO) << "[Tracking] Initialized with first frame. "
              << "Features: " << frame->Features().size();
}

bool Tracking::TrackLastFrame(const Frame::Ptr& frame, int& inliers) {
    if (!last_frame_) {
        return false;
    }

    std::vector<cv::DMatch> matches;
    matcher_->Match(last_frame_, frame, matches);

    inliers = 0;
    bool success = EstimatePoseByEssential(
        frame,
        last_frame_,
        matches,
        inliers);

    return success && inliers >= options_.min_inliers;
}

bool Tracking::EstimatePoseByEssential(
    const Frame::Ptr& curr,
    const Frame::Ptr& last,
    const std::vector<cv::DMatch>& matches,
    int& inliers) {

    inliers = 0;

    if (matches.size() < static_cast<size_t>(options_.min_matches)) {
        return false;
    }

    std::vector<cv::Point2f> pts_last;
    std::vector<cv::Point2f> pts_curr;
    pts_last.reserve(matches.size());
    pts_curr.reserve(matches.size());

    const auto& feats_last = last->Features();
    const auto& feats_curr = curr->Features();

    for (const auto& m : matches) {
        const auto& fl = feats_last[m.queryIdx];
        const auto& fc = feats_curr[m.trainIdx];

        pts_last.emplace_back(
            static_cast<float>(fl.position.x()),
            static_cast<float>(fl.position.y()));

        pts_curr.emplace_back(
            static_cast<float>(fc.position.x()),
            static_cast<float>(fc.position.y()));
    }

    const auto& cam = curr->GetCamera();
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        cam->fx(), 0.0,       cam->cx(),
        0.0,       cam->fy(), cam->cy(),
        0.0,       0.0,       1.0);

    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(
        pts_last,
        pts_curr,
        K,
        cv::RANSAC,
        0.999,
        1.0,
        mask);

    if (E.empty()) {
        return false;
    }

    cv::Mat R, t;
    inliers = cv::recoverPose(
        E,
        pts_last,
        pts_curr,
        K,
        R,
        t,
        mask);

    if (inliers < options_.min_inliers) {
        return false;
    }

    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    cv::cv2eigen(R, R_eigen);
    cv::cv2eigen(t, t_eigen);

    curr->SetPose(Sophus::SE3d(R_eigen, t_eigen));
    return true;
}

} // namespace visionx
