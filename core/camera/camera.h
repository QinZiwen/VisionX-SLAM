#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace visionx {

class Camera {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Camera(double fx, double fy, double cx, double cy, double k1 = 0.0,
           double k2 = 0.0, double p1 = 0.0, double p2 = 0.0);

    // ===== 坐标变换 =====
    Eigen::Vector3d worldToCamera(const Eigen::Vector3d& pw,
                                  const Sophus::SE3d& T_cw) const;

    Eigen::Vector3d cameraToWorld(const Eigen::Vector3d& pc,
                                  const Sophus::SE3d& T_cw) const;

    Eigen::Vector2d cameraToPixel(const Eigen::Vector3d& pc) const;

    Eigen::Vector3d pixelToCamera(const Eigen::Vector2d& px,
                                  double depth = 1.0) const;

    // ===== getter =====
    double fx() const { return fx_; }
    double fy() const { return fy_; }
    double cx() const { return cx_; }
    double cy() const { return cy_; }

private:
    double fx_, fy_, cx_, cy_;
    double k1_, k2_, p1_, p2_;
};

}  // namespace visionx
