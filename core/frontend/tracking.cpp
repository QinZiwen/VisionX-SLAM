#include "frontend/tracking.h"

#include <glog/logging.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace visionx {

// ================= 构造 =================

Tracking::Tracking(const Options& options, std::shared_ptr<FeatureExtractor> extractor,
                   std::shared_ptr<FeatureMatcher> matcher, std::shared_ptr<Map> map)
    : options_(options),
      extractor_(std::move(extractor)),
      matcher_(std::move(matcher)),
      map_(std::move(map)) {}

// ================= 主入口 =================

void Tracking::ProcessFrame(Frame::Ptr frame) {
    current_frame_ = frame;

    extractor_->Extract(*current_frame_);

    if (state_ == State::INIT) {
        if (!init_frame_) {
            InitWithFirstFrame();
        } else {
            if (InitWithSecondFrame()) {
                UpdateTrackingState();
                LOG(INFO) << "[Tracking] Initialization success.";
            }
        }
        return;
    }

    if (!Track()) {
        HandleTrackingFailure();
        return;
    }

    if (NeedNewKeyFrame()) {
        CreateKeyFrame();
    }

    UpdateTrackingState();
    last_frame_ = current_frame_;
}

// ================= 初始化 =================

bool Tracking::InitWithFirstFrame() {
    init_frame_ = current_frame_;
    init_frame_->SetPose(Sophus::SE3d());

    LOG(INFO) << "[Tracking] InitWithFirstFrame. Features: " << init_frame_->Features().size();
    return true;
}

bool Tracking::InitWithSecondFrame() {
    std::vector<cv::DMatch> matches;
    matcher_->Match(init_frame_, current_frame_, matches);

    if (matches.size() < static_cast<size_t>(options_.min_matches)) {
        LOG(WARNING) << "[InitWithSecondFrame] Not enough matches. Matches: " << matches.size()
                     << ", min_matches: " << options_.min_matches;
        return false;
    }
    LOG(INFO) << "[InitWithSecondFrame] Matches: " << matches.size();

    int inliers = 0;
    bool ok = EstimatePoseByEssential(current_frame_, init_frame_, matches, inliers);

    if (!ok || inliers < static_cast<int>(options_.min_inliers)) {
        LOG(WARNING) << "[EstimatePoseByEssential] Essential failed. ok: " << ok
                     << ", inliers: " << inliers;
        return false;
    }

    // === 三角化第一批地图点 ===
    TriangulateWithLastKeyFrame(init_frame_, current_frame_);

    // === 放进 Map ===
    map_->InsertKeyFrame(init_frame_);
    map_->InsertKeyFrame(current_frame_);

    last_parallax_ = ComputeParallax(init_frame_, current_frame_, matches);
    last_frame_ = current_frame_;
    last_inliers_ = inliers;
    LOG(INFO) << "[InitWithSecondFrame] Parallax: " << last_parallax_ << ", inliers: " << inliers;
    return true;
}

// ================= Tracking 主逻辑 =================

bool Tracking::Track() { return TrackWithPnP(); }

bool Tracking::TrackLastFrame() {
    if (!last_frame_) {
        return false;
    }

    std::vector<cv::DMatch> matches;
    matcher_->Match(last_frame_, current_frame_, matches);

    if (matches.size() < static_cast<size_t>(options_.min_matches)) {
        return false;
    }

    int inliers = 0;
    bool success = EstimatePoseByEssential(current_frame_, last_frame_, matches, inliers);

    last_inliers_ = inliers;
    last_parallax_ =
        ComputeParallax(last_keyframe_ ? last_keyframe_ : last_frame_, current_frame_, matches);

    return success && inliers >= options_.min_inliers;
}

bool Tracking::TrackWithPnP() {
    if (!last_frame_) return false;

    // === 1. 特征匹配（last frame ↔ current frame）===
    std::vector<cv::DMatch> matches;
    matcher_->Match(last_frame_, current_frame_, matches);

    if (matches.size() < static_cast<size_t>(options_.min_matches)) {
        LOG(WARNING) << "[TrackWithPnP] Not enough matches. Matches: " << matches.size()
                     << ", min_matches: " << options_.min_matches;
        return false;
    }
    LOG(INFO) << "[TrackWithPnP] Matches: " << matches.size();

    // === 2. 构建 3D–2D 对 ===
    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;

    const auto& feats_last = last_frame_->Features();
    const auto& feats_curr = current_frame_->Features();

    for (const auto& m : matches) {
        // 你需要从 last_frame_ 找到对应的 Landmark
        Landmark::Ptr lm = map_->GetLandmark(feats_last[m.queryIdx].landmark_id_);
        if (!lm) {
            continue;
        }

        const auto& p = lm->Position();
        pts_3d.emplace_back(p.x(), p.y(), p.z());

        const auto& px = feats_curr[m.trainIdx].position;
        pts_2d.emplace_back(px.x(), px.y());
    }

    if (pts_3d.size() < static_cast<size_t>(options_.min_inliers)) {
        LOG(WARNING) << "[PnP] Not enough 3D-2D correspondences. 3D-2D pairs: " << pts_3d.size()
                     << ", min_inliers: " << options_.min_inliers;
        return false;
    }

    // === 3. PnP RANSAC ===
    const auto cam = current_frame_->GetCamera();

    cv::Mat K =
        (cv::Mat_<double>(3, 3) << cam->fx(), 0, cam->cx(), 0, cam->fy(), cam->cy(), 0, 0, 1);

    cv::Mat rvec, tvec, inliers;
    bool ok = cv::solvePnPRansac(pts_3d, pts_2d, K, cv::Mat(), rvec, tvec, false, 100,
                                 options_.max_reproj_error, 0.99, inliers);

    if (!ok || inliers.rows < options_.min_inliers) {
        LOG(WARNING) << "[PnP] solvePnPRansac failed. Inliers: " << inliers.rows
                     << ", min_inliers: " << options_.min_inliers;
        return false;
    }

    // === 4. 转成 Sophus ===
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    cv::cv2eigen(R, R_eigen);
    cv::cv2eigen(tvec, t_eigen);

    Sophus::SE3d T_cw(R_eigen, t_eigen);
    current_frame_->SetPose(T_cw);

    LOG(INFO) << "[PnP] Inliers: " << inliers.rows;
    return true;
}

// ================= 状态管理 =================

void Tracking::UpdateTrackingState() {
    if (last_inliers_ >= options_.min_inliers * 2) {
        state_ = State::TRACKING_GOOD;
    } else {
        state_ = State::TRACKING_BAD;
    }
}

void Tracking::HandleTrackingFailure() {
    if (state_ == State::TRACKING_GOOD) {
        state_ = State::TRACKING_BAD;
    } else {
        state_ = State::LOST;
    }

    LOG(WARNING) << "[Tracking] Tracking failure, state = " << static_cast<int>(state_);
}

// ================= 位姿估计 =================

bool Tracking::EstimatePoseByEssential(const Frame::Ptr& curr, const Frame::Ptr& last,
                                       const std::vector<cv::DMatch>& matches, int& inliers) {
    std::vector<cv::Point2f> pts_last, pts_curr;
    pts_last.reserve(matches.size());
    pts_curr.reserve(matches.size());

    for (const auto& m : matches) {
        const auto& pl = last->Features()[m.queryIdx].position;
        const auto& pc = curr->Features()[m.trainIdx].position;
        pts_last.emplace_back(pl.x(), pl.y());
        pts_curr.emplace_back(pc.x(), pc.y());
    }

    auto cam = curr->GetCamera();
    cv::Mat K = (cv::Mat_<double>(3, 3) << cam->fx(), 0.0, cam->cx(), 0.0, cam->fy(), cam->cy(),
                 0.0, 0.0, 1.0);

    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(pts_last, pts_curr, K, cv::RANSAC, 0.999, 1.0, mask);

    if (E.empty()) {
        return false;
    }

    cv::Mat R, t;
    inliers = cv::recoverPose(E, pts_last, pts_curr, K, R, t, mask);

    if (inliers < options_.min_inliers) {
        return false;
    }

    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    cv::cv2eigen(R, R_eigen);
    cv::cv2eigen(t, t_eigen);

    Sophus::SE3d T_cl(R_eigen, t_eigen);
    Sophus::SE3d T_lw = last->Pose();
    curr->SetPose(T_cl * T_lw);

    return true;
}

// ================= KeyFrame 决策 =================

double Tracking::ComputeParallax(const Frame::Ptr& ref, const Frame::Ptr& curr,
                                 const std::vector<cv::DMatch>& matches) {
    double sum = 0.0;
    int cnt = 0;

    for (const auto& m : matches) {
        const auto& p1 = ref->Features()[m.queryIdx].position;
        const auto& p2 = curr->Features()[m.trainIdx].position;
        sum += (p1 - p2).norm();
        cnt++;
    }
    return cnt > 0 ? sum / cnt : 0.0;
}

bool Tracking::NeedNewKeyFrame() const {
    if (state_ != State::TRACKING_GOOD) return false;

    if (last_inliers_ < options_.min_keyframe_inliers) return false;

    if (last_parallax_ < options_.min_parallax) return false;

    return true;
}

void Tracking::CreateKeyFrame() {
    TriangulateWithLastKeyFrame(last_keyframe_, current_frame_);
    last_keyframe_ = current_frame_;
    map_->InsertKeyFrame(current_frame_);

    LOG(INFO) << "[Tracking] New keyframe created.";
}

// ================= 三角化 =================

Eigen::Matrix<double, 3, 4> Tracking::ProjectionMatrix(const Sophus::SE3d& T_cw,
                                                       const Camera& cam) const {
    Eigen::Matrix<double, 3, 4> P;
    P.leftCols<3>() = T_cw.rotationMatrix();
    P.rightCols<1>() = T_cw.translation();

    Eigen::Matrix3d K;
    K << cam.fx(), 0, cam.cx(), 0, cam.fy(), cam.cy(), 0, 0, 1;

    return K * P;
}

void Tracking::TriangulateWithLastKeyFrame(Frame::Ptr last_frame, Frame::Ptr curr_frame) {
    if (!last_frame || !curr_frame) {
        LOG(WARNING) << "[TriangulateWithLastKeyFrame] Invalid frames.";
        return;
    }

    std::vector<cv::DMatch> matches;
    matcher_->Match(last_frame, curr_frame, matches);

    auto cam = curr_frame->GetCamera();

    auto P1 = ProjectionMatrix(last_frame->Pose(), *cam);
    auto P2 = ProjectionMatrix(curr_frame->Pose(), *cam);

    for (const auto& m : matches) {
        const auto& px1 = last_frame->Features()[m.queryIdx].position;
        const auto& px2 = curr_frame->Features()[m.trainIdx].position;

        Eigen::Vector3d pw = TriangulatePoint(P1, P2, px1, px2);
        if (pw.z() <= 0) {
            continue;
        }

        auto lm = std::make_shared<Landmark>(landmark_id_++, pw);
        lm->AddObservation(last_frame->Id(), m.queryIdx);
        lm->AddObservation(curr_frame->Id(), m.trainIdx);
        map_->InsertLandmark(lm);

        last_frame->Features()[m.queryIdx].landmark_id_ = lm->Id();
        curr_frame->Features()[m.trainIdx].landmark_id_ = lm->Id();
    }

    LOG(INFO) << "[Tracking] Triangulated " << matches.size() << " landmarks.";
}

Eigen::Vector3d Tracking::TriangulatePoint(const Eigen::Matrix<double, 3, 4>& P1,
                                           const Eigen::Matrix<double, 3, 4>& P2,
                                           const Eigen::Vector2d& x1,
                                           const Eigen::Vector2d& x2) const {
    Eigen::Matrix4d A;
    A.row(0) = x1(0) * P1.row(2) - P1.row(0);
    A.row(1) = x1(1) * P1.row(2) - P1.row(1);
    A.row(2) = x2(0) * P2.row(2) - P2.row(0);
    A.row(3) = x2(1) * P2.row(2) - P2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X = svd.matrixV().col(3);

    return X.head<3>() / X(3);
}

}  // namespace visionx
