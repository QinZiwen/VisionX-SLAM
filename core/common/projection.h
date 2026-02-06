#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "camera/camera.h"

namespace visionx {

// NOTE: This uses a pinhole projection without distortion for stability and simplicity.
inline bool ProjectToPixel(const Camera& cam,
                           const Sophus::SE3d& T_cw,
                           const Eigen::Vector3d& pw,
                           Eigen::Vector2d& uv,
                           Eigen::Vector3d* pc_out = nullptr) {
    Eigen::Vector3d pc = T_cw * pw;
    if (pc.z() <= 1e-6) {
        return false;
    }

    const double inv_z = 1.0 / pc.z();
    const double x = pc.x() * inv_z;
    const double y = pc.y() * inv_z;

    uv = Eigen::Vector2d(cam.fx() * x + cam.cx(), cam.fy() * y + cam.cy());

    if (pc_out) {
        *pc_out = pc;
    }
    return true;
}

}  // namespace visionx
