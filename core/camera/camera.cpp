#include "camera/camera.h"

namespace visionx {

Camera::Camera(double fx, double fy,
               double cx, double cy,
               double k1, double k2,
               double p1, double p2)
    : fx_(fx), fy_(fy),
      cx_(cx), cy_(cy),
      k1_(k1), k2_(k2),
      p1_(p1), p2_(p2) {}

Eigen::Vector3d Camera::worldToCamera(
    const Eigen::Vector3d& pw,
    const Sophus::SE3d& T_cw) const {
    return T_cw * pw;
}

Eigen::Vector3d Camera::cameraToWorld(
    const Eigen::Vector3d& pc,
    const Sophus::SE3d& T_cw) const {
    return T_cw.inverse() * pc;
}

Eigen::Vector2d Camera::cameraToPixel(
    const Eigen::Vector3d& pc) const {
    const double x = pc[0] / pc[2];
    const double y = pc[1] / pc[2];

    const double r2 = x * x + y * y;
    const double radial = 1.0 + k1_ * r2 + k2_ * r2 * r2;

    const double x_distorted =
        x * radial + 2.0 * p1_ * x * y + p2_ * (r2 + 2.0 * x * x);
    const double y_distorted =
        y * radial + p1_ * (r2 + 2.0 * y * y) + 2.0 * p2_ * x * y;

    return Eigen::Vector2d(
        fx_ * x_distorted + cx_,
        fy_ * y_distorted + cy_);
}

Eigen::Vector3d Camera::pixelToCamera(
    const Eigen::Vector2d& px,
    double depth) const {
    const double x = (px[0] - cx_) / fx_;
    const double y = (px[1] - cy_) / fy_;
    return Eigen::Vector3d(x * depth, y * depth, depth);
}

} // namespace visionx
