#include "frontend/tracking.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "common/projection.h"

namespace visionx {

// ================= 构造 =================

Tracking::Tracking(const Options& options, std::shared_ptr<FeatureExtractor> extractor,
                   std::shared_ptr<FeatureMatcher> matcher, std::shared_ptr<Map> map)
    : options_(options),
      extractor_(std::move(extractor)),
      matcher_(std::move(matcher)),
      map_(std::move(map)) {
    if (options_.enable_local_ba) {
        LocalBA::Options ba_options;
        ba_options.window_size = options_.ba_window_size;
        ba_options.max_iterations = options_.ba_iterations;
        ba_options.min_pose_observations = options_.ba_min_pose_observations;
        ba_options.min_point_observations = options_.ba_min_point_observations;
        ba_options.huber_delta = options_.ba_huber_delta;
        ba_options.max_reproj_error = options_.ba_max_reproj_error;
        local_ba_ = std::make_unique<LocalBA>(ba_options);
    }
}

// ================= 主入口 =================

void Tracking::ProcessFrame(Frame::Ptr frame) {
    current_frame_ = frame;

    extractor_->Extract(*current_frame_);
    bool just_initialized = false;

    if (state_ == State::INIT) {
        if (!init_frame_) {
            if (!InitWithFirstFrame()) {
                LOG(INFO) << "[ProcessFrame] Waiting for a better initial frame...";
                return;  // 继续等待更好的第一帧
            }
            // 还未完成初始化（需要第二帧）
            return;
        } else {
            if (!InitWithSecondFrame()) {
                LOG(INFO) << "[ProcessFrame] Waiting for a better second frame...";
                return;
            }
            UpdateTrackingState();
            LOG(INFO) << "[Tracking] Initialization success.";
            last_frame_ = current_frame_;
            just_initialized = true;
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

    if (!just_initialized && NeedNewKeyFrame()) {
        CreateKeyFrame();
        if (options_.enable_culling) {
            CullLandmarks();
            CullKeyFrames();
        }
        if (local_ba_) {
            local_ba_->Optimize(map_, last_keyframe_);
        }
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

    // 要求至少50%的网格有特征点
    return valid_grids >= (grid_cols * grid_rows * 0.5);
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

    // === 深度辅助创建地图点 ===
    CreateLandmarksFromDepth(init_frame_);
    CreateLandmarksFromDepth(current_frame_);

    // === 三角化第一批地图点 ===
    TriangulateWithLastKeyFrame(init_frame_, current_frame_);

    // === 放进 Map ===
    map_->InsertKeyFrame(init_frame_);
    map_->InsertKeyFrame(current_frame_);
    last_keyframe_ = current_frame_;

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
    last_parallax_ = ComputeParallax(last_frame_, current_frame_, matches);
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
        const auto& feat_last = feats_last[m.queryIdx];
        if (!feat_last.has_landmark || feat_last.is_outlier) {
            continue;
        }
        Landmark::Ptr lm = map_->GetLandmark(feat_last.landmark_id_);
        if (!lm) {
            continue;
        }
        if (lm->IsBad()) {
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
    if (last_inliers_ >= options_.min_inliers) {
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
    init_frame_.reset();
    last_frame_.reset();
    last_keyframe_.reset();
    last_inliers_ = 0;
    last_parallax_ = 0.0;
    LOG(INFO) << "[ProcessFrame] Tracking bad. Trying to re-initialize...";
}

void Tracking::HandleTrackingLost() {
    // TODO(qinziwen): 目前线尝试重新初始化
    state_ = State::INIT;
    map_->removeAll();
    init_frame_.reset();
    last_frame_.reset();
    last_keyframe_.reset();
    last_inliers_ = 0;
    last_parallax_ = 0.0;
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
    if (!current_frame_ || !last_keyframe_) return false;

    if (last_inliers_ < options_.min_keyframe_inliers) return false;

    if (last_parallax_ < options_.min_parallax) return false;

    if (current_frame_->Id() - last_keyframe_->Id() < static_cast<uint64_t>(options_.min_keyframe_gap)) {
        return false;
    }

    return true;
}

void Tracking::CreateKeyFrame() {
    CreateLandmarksFromDepth(current_frame_);
    TriangulateWithLastKeyFrame(last_keyframe_, current_frame_);
    last_keyframe_ = current_frame_;
    map_->InsertKeyFrame(current_frame_);

    LOG(INFO) << "[Tracking] New keyframe created.";
}

void Tracking::CreateLandmarksFromDepth(const Frame::Ptr& frame) {
    if (!map_ || !frame) {
        return;
    }

    const cv::Mat& depth = frame->Depth();
    if (depth.empty()) {
        return;
    }

    const auto cam = frame->GetCamera();
    if (!cam) {
        return;
    }

    const int rows = depth.rows;
    const int cols = depth.cols;
    const double kDepthScale = 5000.0;  // TUM RGB-D depth scale
    const double kMinDepth = 0.1;
    const double kMaxDepth = 10.0;

    auto& features = frame->Features();
    for (size_t i = 0; i < features.size(); ++i) {
        auto& feat = features[i];
        if (feat.has_landmark) {
            continue;
        }

        const int u = static_cast<int>(feat.position.x() + 0.5);
        const int v = static_cast<int>(feat.position.y() + 0.5);
        if (u < 0 || u >= cols || v < 0 || v >= rows) {
            continue;
        }

        double depth_m = 0.0;
        if (depth.type() == CV_16U) {
            uint16_t d = depth.at<uint16_t>(v, u);
            if (d == 0) {
                continue;
            }
            depth_m = static_cast<double>(d) / kDepthScale;
        } else if (depth.type() == CV_32F) {
            depth_m = static_cast<double>(depth.at<float>(v, u));
        } else if (depth.type() == CV_64F) {
            depth_m = depth.at<double>(v, u);
        } else {
            continue;
        }

        if (depth_m < kMinDepth || depth_m > kMaxDepth) {
            continue;
        }

        Eigen::Vector3d pc = cam->pixelToCamera(feat.position, depth_m);
        Eigen::Vector3d pw = frame->Pose().inverse() * pc;

        auto lm = std::make_shared<Landmark>(landmark_id_++, pw);
        lm->AddObservation(frame->Id(), i);
        map_->InsertLandmark(lm);

        feat.landmark_id_ = lm->Id();
        feat.has_landmark = true;
        feat.is_outlier = false;
    }
}

void Tracking::CullLandmarks() {
    if (!map_) {
        return;
    }
    if (map_->LandmarkSize() < static_cast<size_t>(options_.min_landmarks_for_culling)) {
        return;
    }

    std::vector<uint64_t> to_remove;
    for (const auto& kv : map_->Landmarks()) {
        const auto& lm = kv.second;
        if (!lm) {
            continue;
        }
        if (lm->IsBad()) {
            to_remove.push_back(lm->Id());
            continue;
        }
        if (lm->ObservationCount() < static_cast<size_t>(options_.min_landmark_observations)) {
            lm->SetBad();
            to_remove.push_back(lm->Id());
            continue;
        }

        double err_sum = 0.0;
        int cnt = 0;
        bool large_error = false;

        for (const auto& [kf_id, feat_idx] : lm->Observations()) {
            auto frame = map_->GetFrame(kf_id);
            if (!frame) {
                continue;
            }
            if (feat_idx >= frame->Features().size()) {
                continue;
            }
            auto& feat = frame->Features()[feat_idx];
            if (!feat.has_landmark || feat.landmark_id_ != lm->Id()) {
                continue;
            }

            const auto cam = frame->GetCamera();
            if (!cam) {
                continue;
            }

            Eigen::Vector2d proj;
            if (!ProjectToPixel(*cam, frame->Pose(), lm->Position(), proj)) {
                continue;
            }

            const double err = (feat.position - proj).norm();
            err_sum += err;
            cnt++;

            if (err > options_.landmark_max_reproj_error * 2.0) {
                large_error = true;
                break;
            }
        }

        if (cnt == 0) {
            lm->SetBad();
            to_remove.push_back(lm->Id());
            continue;
        }

        if (large_error || (err_sum / cnt) > options_.landmark_max_reproj_error) {
            lm->SetBad();
            to_remove.push_back(lm->Id());
        }
    }

    if (to_remove.empty()) {
        return;
    }

    for (auto id : to_remove) {
        auto lm = map_->GetLandmark(id);
        if (!lm) {
            continue;
        }
        for (const auto& [kf_id, feat_idx] : lm->Observations()) {
            auto frame = map_->GetFrame(kf_id);
            if (!frame || feat_idx >= frame->Features().size()) {
                continue;
            }
            auto& feat = frame->Features()[feat_idx];
            if (feat.landmark_id_ == id) {
                feat.landmark_id_ = 0;
                feat.has_landmark = false;
                feat.is_outlier = true;
            }
        }
        map_->RemoveLandmark(id);
    }

    LOG(INFO) << "[Tracking] Culled landmarks: " << to_remove.size();
}

void Tracking::RemoveKeyFrame(const Frame::Ptr& keyframe) {
    if (!map_ || !keyframe) {
        return;
    }

    const uint64_t kf_id = keyframe->Id();
    for (auto& feat : keyframe->Features()) {
        if (!feat.has_landmark) {
            continue;
        }
        auto lm = map_->GetLandmark(feat.landmark_id_);
        if (!lm) {
            continue;
        }
        lm->RemoveObservation(kf_id);
        feat.landmark_id_ = 0;
        feat.has_landmark = false;
        feat.is_outlier = true;
    }

    map_->RemoveKeyFrame(kf_id);
}

void Tracking::CullKeyFrames() {
    if (!map_) {
        return;
    }

    const auto& keyframes = map_->KeyFrames();
    if (keyframes.size() <= static_cast<size_t>(options_.min_keyframes_for_culling)) {
        return;
    }

    const bool exceeded =
        options_.max_keyframes > 0 &&
        keyframes.size() > static_cast<size_t>(options_.max_keyframes);

    Frame::Ptr to_remove = nullptr;
    double removed_ratio = 0.0;

    for (const auto& kv : keyframes) {
        const auto& kf = kv.second;
        if (!kf) {
            continue;
        }
        if (kf == last_keyframe_ || kf == init_frame_) {
            continue;
        }
        if (current_frame_ && kf->Id() == current_frame_->Id()) {
            continue;
        }

        int total = 0;
        int redundant = 0;

        for (const auto& feat : kf->Features()) {
            if (!feat.has_landmark) {
                continue;
            }
            total++;
            auto lm = map_->GetLandmark(feat.landmark_id_);
            if (!lm || lm->IsBad()) {
                continue;
            }
            if (lm->ObservationCount() >=
                static_cast<size_t>(options_.kf_min_shared_observations)) {
                redundant++;
            }
        }

        if (total == 0) {
            continue;
        }

        const double ratio = static_cast<double>(redundant) / static_cast<double>(total);
        if (ratio > options_.kf_redundant_ratio && (exceeded || ratio > 0.95)) {
            to_remove = kf;
            removed_ratio = ratio;
            break;
        }
    }

    if (to_remove) {
        RemoveKeyFrame(to_remove);
        LOG(INFO) << "[Tracking] Culled keyframe " << to_remove->Id()
                  << ", redundant_ratio=" << removed_ratio;
        CullLandmarks();
    }
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

    const double min_angle_rad =
        options_.triangulation_min_angle_deg * M_PI / 180.0;

    for (const auto& m : matches) {
        const auto& px1 = last_frame->Features()[m.queryIdx].position;
        const auto& px2 = curr_frame->Features()[m.trainIdx].position;
        if (last_frame->Features()[m.queryIdx].has_landmark ||
            curr_frame->Features()[m.trainIdx].has_landmark) {
            continue;
        }

        // 视差角检查（避免几何退化）
        Eigen::Vector3d f1 = last_frame->GetCamera()->pixelToCamera(px1, 1.0).normalized();
        Eigen::Vector3d f2 = curr_frame->GetCamera()->pixelToCamera(px2, 1.0).normalized();
        Eigen::Matrix3d R1 = last_frame->Pose().inverse().rotationMatrix();
        Eigen::Matrix3d R2 = curr_frame->Pose().inverse().rotationMatrix();
        Eigen::Vector3d f1w = R1 * f1;
        Eigen::Vector3d f2w = R2 * f2;
        double cos_angle = f1w.dot(f2w) / (f1w.norm() * f2w.norm());
        cos_angle = std::clamp(cos_angle, -1.0, 1.0);
        const double angle = std::acos(cos_angle);
        if (angle < min_angle_rad) {
            continue;
        }

        Eigen::Vector3d pw = TriangulatePoint(P1, P2, px1, px2);
        if (!pw.allFinite()) {
            continue;
        }

        Eigen::Vector2d reproj1, reproj2;
        if (!ProjectToPixel(*last_frame->GetCamera(), last_frame->Pose(), pw, reproj1)) {
            continue;
        }
        if (!ProjectToPixel(*cam, curr_frame->Pose(), pw, reproj2)) {
            continue;
        }

        const double err1 = (reproj1 - px1).norm();
        const double err2 = (reproj2 - px2).norm();
        if (err1 > options_.triangulation_max_reproj_error ||
            err2 > options_.triangulation_max_reproj_error) {
            continue;
        }

        auto lm = std::make_shared<Landmark>(landmark_id_++, pw);
        lm->AddObservation(last_frame->Id(), m.queryIdx);
        lm->AddObservation(curr_frame->Id(), m.trainIdx);
        map_->InsertLandmark(lm);

        last_frame->Features()[m.queryIdx].landmark_id_ = lm->Id();
        last_frame->Features()[m.queryIdx].has_landmark = true;
        last_frame->Features()[m.queryIdx].is_outlier = false;
        curr_frame->Features()[m.trainIdx].landmark_id_ = lm->Id();
        curr_frame->Features()[m.trainIdx].has_landmark = true;
        curr_frame->Features()[m.trainIdx].is_outlier = false;
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
