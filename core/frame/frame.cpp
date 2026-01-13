#include "frame/frame.h"
#include "camera/camera.h"

namespace visionx {

Frame::Frame(uint64_t id,
             double timestamp,
             std::shared_ptr<Camera> camera,
             const cv::Mat& image)
    : id_(id),
      timestamp_(timestamp),
      T_cw_(Sophus::SE3d()),
      camera_(std::move(camera)),
      image_(image.clone()) {}

Sophus::SE3d Frame::Pose() const {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    return T_cw_;
}

void Frame::SetPose(const Sophus::SE3d& T_cw) {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    T_cw_ = T_cw;
}

} // namespace visionx
