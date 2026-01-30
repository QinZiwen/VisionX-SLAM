#pragma once

#include <Eigen/Core>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <vector>

#include "camera/camera.h"

namespace visionx {

class Landmark;

struct Feature {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector2d position;   // 像素坐标
    float response = 0.0f;      // 特征强度（ORB response 等）
    uint64_t landmark_id_ = 0;  // 关联的路标点 ID
};

class Frame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Frame>;

    Frame(uint64_t id, double timestamp, std::shared_ptr<Camera> camera, const cv::Mat& image);

    Sophus::SE3d Pose() const;
    void SetPose(const Sophus::SE3d& T_cw);

    uint64_t Id() const { return id_; }
    double Timestamp() const { return timestamp_; }
    const cv::Mat& Image() const { return image_; }

    std::vector<Feature>& Features() { return features_; }
    const std::vector<Feature>& Features() const { return features_; }
    cv::Mat& Descriptors() { return descriptors_; }
    const cv::Mat& Descriptors() const { return descriptors_; }

    std::shared_ptr<Camera> GetCamera() const { return camera_; }

    Frame::Ptr Clone(bool need_feature = true) const;

private:
    uint64_t id_ = 0;
    double timestamp_ = 0.0;

    // world -> camera
    Sophus::SE3d T_cw_;
    mutable std::mutex pose_mutex_;

    std::shared_ptr<Camera> camera_;
    cv::Mat image_;

    std::vector<Feature> features_;
    cv::Mat descriptors_;
};

}  // namespace visionx
