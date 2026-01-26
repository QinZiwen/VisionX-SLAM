#include "frame/frame.h"

#include "camera/camera.h"

namespace visionx {

Frame::Frame(uint64_t id, double timestamp, std::shared_ptr<Camera> camera, const cv::Mat& image)
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

Frame::Ptr Frame::Clone(bool need_feature) const {
    Sophus::SE3d current_pose;
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        current_pose = T_cw_;
    }

    auto new_frame = std::make_shared<Frame>(id_, timestamp_, camera_, image_.clone());
    new_frame->SetPose(current_pose);

    if (need_feature) {
        new_frame->features_ = this->features_;
        if (!this->descriptors_.empty()) {
            new_frame->descriptors_ = this->descriptors_.clone();
        }
    }

    return new_frame;
}

}  // namespace visionx
