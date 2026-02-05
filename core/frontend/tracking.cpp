#include "frontend/tracking.h"

#include <glog/logging.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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
            if (!InitWithFirstFrame()) {
                LOG(INFO) << "[ProcessFrame] Waiting for a better initial frame...";
                return;  // 继续等待更好的第一帧
            }
        } else {
            if (InitWithSecondFrame()) {
                UpdateTrackingState();
                LOG(INFO) << "[Tracking] Initialization success.";
            }
        }
    } else if (state_ == State::TRACKING_GOOD) {
        if (!Track()) {
            HandleTrackingFailure();
            return;
        }
    } else if (state_ == State::TRACKING_BAD) {
        HandleTrackingBad();
        return;
    } else if (state_ == State::LOST) {
        HandleTrackingLost();
        return;
    }

    if (NeedNewKeyFrame()) {
        CreateKeyFrame();
    }

    UpdateTrackingState();
    last_frame_ = current_frame_;
}

// ================= 初始化 =================

bool Tracking::CheckFeatureDistribution(const std::vector<Feature>& features, int width,
                                        int height) const {
    // 划分网格，检查每个网格是否有特征点
    const int grid_cols = 5;
    const int grid_rows = 5;
    std::vector<std::vector<bool>> grid(grid_cols, std::vector<bool>(grid_rows, false));

    for (const auto& feat : features) {
        int col = static_cast<int>((feat.position.x() / width) * grid_cols);
        int row = static_cast<int>((feat.position.y() / height) * grid_rows);
        col = std::clamp(col, 0, grid_cols - 1);
        row = std::clamp(row, 0, grid_rows - 1);
        grid[col][row] = true;
    }

    // 计算有效网格数
    int valid_grids = 0;
    for (int i = 0; i < grid_cols; i++) {
        for (int j = 0; j < grid_rows; j++) {
            if (grid[i][j]) valid_grids++;
        }
    }

    // 要求至少70%的网格有特征点
    return valid_grids >= (grid_cols * grid_rows * 0.7);
}

bool Tracking::CheckImageQuality(const cv::Mat& image) const {
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // 计算图像亮度
    cv::Scalar mean, stddev;
    cv::meanStdDev(grayImage, mean, stddev);

    // 亮度范围检查（0-255）
    if (mean[0] < 30 || mean[0] > 225) {
        return false;
    }

    // 对比度检查
    if (stddev[0] < 20) {
        return false;
    }

    return true;
}

void Tracking::VisualizeFeatures(Frame::Ptr frame) const {
    if (!frame) {
        LOG(WARNING) << "[VisualizeFeatures] Empty frame.";
        return;
    }

    // 复制图像用于显示
    cv::Mat display = frame->Image().clone();
    const auto& features = frame->Features();

    // 绘制特征点
    for (const auto& feat : features) {
        cv::circle(display, cv::Point2f(feat.position.x(), feat.position.y()), 2,
                   cv::Scalar(0, 255, 0), -1);
    }

    // 创建窗口并显示
    std::string window_name = "Frame Features: " + std::to_string(frame->Id());
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 800, 600);
    cv::imshow(window_name, display);

    LOG(INFO) << "[VisualizeFeatures] Press 'q' to close the window and continue...";

    // 等待用户按q键
    while (true) {
        char key = cv::waitKey(10);
        if (key == 'q' || key == 'Q') {
            break;
        }
    }

    // 关闭窗口
    cv::destroyWindow(window_name);
}

bool Tracking::InitWithFirstFrame() {
    // 1. 检查特征点数量
    if (current_frame_->Features().size() < static_cast<size_t>(options_.min_matches)) {
        LOG(WARNING) << "[InitWithFirstFrame] Not enough features. Features: "
                     << current_frame_->Features().size()
                     << ", min_matches: " << options_.min_matches;
        return false;
    }

    // 2. 检查特征分布
    if (!CheckFeatureDistribution(current_frame_->Features(), current_frame_->Image().cols,
                                  current_frame_->Image().rows)) {
        LOG(WARNING) << "[InitWithFirstFrame] Poor feature distribution.";
        return false;
    }

    // 3. 检查图像质量
    if (!CheckImageQuality(current_frame_->Image())) {
        LOG(WARNING) << "[InitWithFirstFrame] Poor image quality (brightness/contrast).";
        return false;
    }

    init_frame_ = current_frame_;
    init_frame_->SetPose(Sophus::SE3d());

    LOG(INFO) << "[Tracking] InitWithFirstFrame. Features: " << init_frame_->Features().size();
    return true;
}

bool Tracking::InitWithSecondFrame() {
    std::vector<cv::DMatch> matches;
    matcher_->Match(init_frame_, current_frame_, matches);

    // 添加匹配质量过滤
    std::vector<cv::DMatch> good_matches;
    float max_dist = 0.0, min_dist = 100.0;
    for (const auto& match : matches) {
        if (match.distance < min_dist) min_dist = match.distance;
        if (match.distance > max_dist) max_dist = match.distance;
    }
    for (const auto& match : matches) {
        if (match.distance <= std::max(2 * min_dist, 30.0f)) {
            good_matches.push_back(match);
        }
    }
    matches.swap(good_matches);

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

    float parallax = ComputeParallax(init_frame_, current_frame_, matches);
    float min_parallax = 1.0f * M_PI / 180.0f;  // 1 度
    if (parallax < min_parallax) {
        LOG(WARNING) << "[InitWithSecondFrame] Parallax too small: " << parallax;
        return false;
    }

    // === 三角化第一批地图点 ===
    TriangulateWithLastKeyFrame(init_frame_, current_frame_);

    // === 放进 Map ===
    map_->InsertKeyFrame(init_frame_);
    map_->InsertKeyFrame(current_frame_);

    last_parallax_ = parallax;
    last_inliers_ = inliers;
    LOG(INFO) << "[InitWithSecondFrame] Parallax: " << last_parallax_ << ", inliers: " << inliers;
    return true;
}

// ================= Tracking 主逻辑 =================

bool Tracking::Track() {
    // 如果last_keyframe_存在，尝试使用PnP（更准确）
    if (last_keyframe_) {
        bool pnp_ok = TrackWithPnP();
        if (pnp_ok) {
            return true;
        }
        // PnP失败，回退到TrackLastFrame
        LOG(INFO) << "[Track] PnP failed, falling back to TrackLastFrame.";
    }
    // 使用TrackLastFrame（基于本质矩阵）
    return TrackLastFrame();
}

bool Tracking::TrackLastFrame() {
    if (!last_frame_) {
        LOG(WARNING) << "[TrackLastFrame] last_frame_ is null";
        return false;
    }

    std::vector<cv::DMatch> matches;
    matches.reserve(1000);  // 预分配内存
    matcher_->Match(last_frame_, current_frame_, matches);

    // 1. 匹配质量过滤
    std::vector<cv::DMatch> good_matches;
    if (!matches.empty()) {
        float min_dist = 100.0f;
        for (const auto& match : matches) {
            if (match.distance < min_dist) min_dist = match.distance;
        }
        for (const auto& match : matches) {
            if (match.distance <= std::max(2 * min_dist, 30.0f)) {
                good_matches.push_back(match);
            }
        }
        matches.swap(good_matches);
    }

    if (matches.size() < static_cast<size_t>(options_.min_matches)) {
        LOG(WARNING) << "[TrackLastFrame] Not enough matches. Matches: " << matches.size()
                     << ", min_matches: " << options_.min_matches;
        return false;
    }
    LOG(INFO) << "[TrackLastFrame] Matches: " << matches.size();

    // 2. 姿态估计
    int inliers = 0;
    bool success = EstimatePoseByEssential(current_frame_, last_frame_, matches, inliers);

    if (!success || inliers < static_cast<int>(options_.min_inliers)) {
        LOG(WARNING) << "[TrackLastFrame] Pose estimation failed. success: " << success
                     << ", inliers: " << inliers << ", min_inliers: " << options_.min_inliers;
        return false;
    }

    // 3. 计算视差
    last_inliers_ = inliers;
    last_parallax_ =
        ComputeParallax(last_keyframe_ ? last_keyframe_ : last_frame_, current_frame_, matches);
    LOG(INFO) << "[TrackLastFrame] Success. Inliers: " << inliers
              << ", Parallax: " << last_parallax_;

    return true;
}

bool Tracking::TrackWithPnP() {
    if (!last_keyframe_) {
        LOG(WARNING) << "[TrackWithPnP] last_keyframe_ is null";
        return false;
    }

    // === 1. 特征匹配（last keyframe ↔ current frame）===
    std::vector<cv::DMatch> matches;
    matcher_->Match(last_keyframe_, current_frame_, matches);

    // 匹配质量过滤
    std::vector<cv::DMatch> good_matches;
    if (!matches.empty()) {
        float min_dist = 100.0f;
        for (const auto& match : matches) {
            if (match.distance < min_dist) min_dist = match.distance;
        }
        for (const auto& match : matches) {
            if (match.distance <= std::max(2 * min_dist, 30.0f)) {
                good_matches.push_back(match);
            }
        }
        matches.swap(good_matches);
    }

    if (matches.size() < static_cast<size_t>(options_.min_matches)) {
        LOG(WARNING) << "[TrackWithPnP] Not enough matches. Matches: " << matches.size()
                     << ", min_matches: " << options_.min_matches;
        return false;
    }
    LOG(INFO) << "[TrackWithPnP] Matches: " << matches.size();

    // === 2. 构建 3D–2D 对 ===
    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    pts_3d.reserve(matches.size());  // 预分配内存
    pts_2d.reserve(matches.size());

    const auto& feats_last = last_keyframe_->Features();
    const auto& feats_curr = current_frame_->Features();

    for (const auto& m : matches) {
        // 从 last_keyframe_ 找到对应的 Landmark
        Landmark::Ptr lm = map_->GetLandmark(feats_last[m.queryIdx].landmark_id_);
        if (!lm) {
            continue;
        }

        const auto& p = lm->Position();
        // 检查 3D 点是否有效（避免无穷远点或异常值）
        if (std::isnan(p.x()) || std::isnan(p.y()) || std::isnan(p.z())) {
            continue;
        }
        if (std::abs(p.x()) > 1000 || std::abs(p.y()) > 1000 || std::abs(p.z()) > 1000) {
            continue;
        }

        pts_3d.emplace_back(p.x(), p.y(), p.z());

        const auto& px = feats_curr[m.trainIdx].position;
        pts_2d.emplace_back(px.x(), px.y());
    }

    if (pts_3d.size() < static_cast<size_t>(options_.min_inliers)) {
        LOG(WARNING) << "[TrackWithPnP] Not enough 3D-2D correspondences. 3D-2D pairs: "
                     << pts_3d.size() << ", min_inliers: " << options_.min_inliers;
        return false;
    }
    LOG(INFO) << "[TrackWithPnP] 3D-2D pairs: " << pts_3d.size();

    // === 3. PnP RANSAC ===
    const auto cam = current_frame_->GetCamera();
    if (!cam) {
        LOG(ERROR) << "[TrackWithPnP] Camera is null";
        return false;
    }

    cv::Mat K =
        (cv::Mat_<double>(3, 3) << cam->fx(), 0, cam->cx(), 0, cam->fy(), cam->cy(), 0, 0, 1);

    cv::Mat rvec, tvec, inliers;
    // 动态调整 RANSAC 参数
    int max_iterations = std::min(100, static_cast<int>(pts_3d.size() * 2));
    bool ok = cv::solvePnPRansac(pts_3d, pts_2d, K, cv::Mat(), rvec, tvec, false, max_iterations,
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

    // 检查旋转矩阵是否有效
    if (std::isnan(R_eigen.norm()) || std::isinf(R_eigen.norm())) {
        LOG(WARNING) << "[TrackWithPnP] Invalid rotation matrix";
        return false;
    }

    Sophus::SE3d T_cw(R_eigen, t_eigen);
    current_frame_->SetPose(T_cw);

    last_parallax_ = ComputeParallax(last_keyframe_, current_frame_, matches);
    last_inliers_ = inliers.rows;

    LOG(INFO) << "[TrackWithPnP] Success. Inliers: " << inliers.rows
              << ", Parallax: " << last_parallax_;
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

void Tracking::HandleTrackingBad() {
    // TODO(qinziwen): 目前线尝试重新初始化
    state_ = State::INIT;
    map_->removeAll();
    LOG(INFO) << "[ProcessFrame] Tracking bad. Trying to re-initialize...";
}

void Tracking::HandleTrackingLost() {
    // TODO(qinziwen): 目前线尝试重新初始化
    state_ = State::INIT;
    map_->removeAll();
    LOG(INFO) << "[ProcessFrame] Tracking lost. Trying to re-initialize...";
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
