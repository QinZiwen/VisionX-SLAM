#pragma once

#include <memory>
#include <vector>
#include <mutex>

#include <Eigen/Core>
#include <Sophus/se3.hpp>
#include <opencv2/core.hpp>

namespace visionx {

class Camera;

struct Feature {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector2d position;   // 像素坐标
    float response = 0.0f;      // 特征强度（ORB response 等）
};

class Frame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Frame>;

    Frame(uint64_t id,
          double timestamp,
          std::shared_ptr<Camera> camera,
          const cv::Mat& image);

    // ===== pose =====
    Sophus::SE3d Pose() const;
    void SetPose(const Sophus::SE3d& T_cw);

    // ===== basic info =====
    uint64_t Id() const { return id_; }
    double Timestamp() const { return timestamp_; }
    const cv::Mat& Image() const { return image_; }

    // ===== feature =====
    std::vector<Feature>& Features() { return features_; }
    const std::vector<Feature>& Features() const { return features_; }

    std::shared_ptr<Camera> GetCamera() const { return camera_; }

private:
    uint64_t id_ = 0;
    double timestamp_ = 0.0;

    // world -> camera
    Sophus::SE3d T_cw_;
    mutable std::mutex pose_mutex_;

    std::shared_ptr<Camera> camera_;
    cv::Mat image_;

    std::vector<Feature> features_;
};

} // namespace visionx
