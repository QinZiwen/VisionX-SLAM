#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>

namespace visionx {

struct ImageEntry {
    double timestamp;
    std::string rgb_path;
    std::string depth_path;
    Eigen::Vector3d t;     // Ground Truth Translation
    Eigen::Quaterniond q;  // Ground Truth Rotation
};

struct CameraIntrinsics {
    double fx;
    double fy;
    double cx;
    double cy;
    double k1 = 0.0;
    double k2 = 0.0;
    double k3 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;

    // 重载流操作符
    friend std::ostream& operator<<(std::ostream& os, const CameraIntrinsics& intrin) {
        os << "CameraIntrinsics:\n"
           << "  - Focal Length:  fx = " << intrin.fx << ", fy = " << intrin.fy << "\n"
           << "  - Principal Pt:  cx = " << intrin.cx << ", cy = " << intrin.cy << "\n"
           << "  - Radial Dist:   k1 = " << intrin.k1 << ", k2 = " << intrin.k2
           << ", k3 = " << intrin.k3 << "\n"
           << "  - Tangent Dist:  p1 = " << intrin.p1 << ", p2 = " << intrin.p2;
        return os;
    }
};

class Dataset {
public:
    using Ptr = std::shared_ptr<Dataset>;

    virtual ~Dataset() = default;

    virtual bool Load() = 0;
    const std::vector<ImageEntry>& ImageEntries() const { return entries_; }
    const CameraIntrinsics& Intrinsics() const { return intrinsics_; }

protected:
    std::vector<ImageEntry> entries_;
    CameraIntrinsics intrinsics_;
};

}  // namespace visionx