#include "backend/local_ba.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

#include "common/projection.h"

namespace visionx {
namespace {

Eigen::Matrix<double, 2, 3> ProjectionJacobian(const Camera& cam, const Eigen::Vector3d& pc) {
    const double x = pc.x();
    const double y = pc.y();
    const double z = pc.z();
    const double z2 = z * z;

    Eigen::Matrix<double, 2, 3> J;
    J << cam.fx() / z, 0.0, -cam.fx() * x / z2, 0.0, cam.fy() / z, -cam.fy() * y / z2;
    return J;
}

Eigen::Matrix<double, 2, 6> PoseJacobian(const Camera& cam, const Eigen::Vector3d& pc) {
    Eigen::Matrix<double, 3, 6> J_se3;
    J_se3.setZero();
    J_se3.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    J_se3.block<3, 3>(0, 3) = -Sophus::SO3d::hat(pc);

    return ProjectionJacobian(cam, pc) * J_se3;
}

double HuberWeight(double error, double delta) {
    if (error <= delta) {
        return 1.0;
    }
    return delta / error;
}

std::vector<Frame::Ptr> SelectKeyFrames(const Map::Ptr& map, const Frame::Ptr& ref_kf,
                                        int window_size) {
    std::vector<Frame::Ptr> keyframes;
    const auto& all = map->KeyFrames();
    if (all.empty()) {
        return keyframes;
    }

    uint64_t max_id = ref_kf ? ref_kf->Id() : all.rbegin()->first;
    int count = 0;
    for (auto it = all.rbegin(); it != all.rend() && count < window_size; ++it) {
        if (it->first > max_id) {
            continue;
        }
        keyframes.push_back(it->second);
        count++;
    }

    std::reverse(keyframes.begin(), keyframes.end());
    return keyframes;
}

}  // namespace

void LocalBA::Optimize(const Map::Ptr& map, const Frame::Ptr& ref_kf) {
    if (!map) {
        return;
    }

    const int window_size = std::max(1, options_.window_size);
    auto keyframes = SelectKeyFrames(map, ref_kf, window_size);
    if (keyframes.size() < 2) {
        return;
    }

    std::unordered_set<uint64_t> local_kf_ids;
    local_kf_ids.reserve(keyframes.size());
    for (const auto& kf : keyframes) {
        local_kf_ids.insert(kf->Id());
    }

    std::unordered_set<uint64_t> landmark_ids;
    for (const auto& kf : keyframes) {
        for (const auto& feat : kf->Features()) {
            if (!feat.has_landmark) {
                continue;
            }
            landmark_ids.insert(feat.landmark_id_);
        }
    }

    std::vector<Landmark::Ptr> landmarks;
    landmarks.reserve(landmark_ids.size());
    for (auto id : landmark_ids) {
        auto lm = map->GetLandmark(id);
        if (!lm || lm->IsBad()) {
            continue;
        }
        if (lm->ObservationCount() < static_cast<size_t>(options_.min_point_observations)) {
            continue;
        }
        landmarks.push_back(lm);
    }

    if (landmarks.empty()) {
        return;
    }

    double last_cost = std::numeric_limits<double>::max();

    for (int iter = 0; iter < options_.max_iterations; ++iter) {
        double total_cost = 0.0;
        int total_obs = 0;

        // === Pose optimization (fix landmarks) ===
        for (const auto& kf : keyframes) {
            if (!kf) {
                continue;
            }

            const auto cam = kf->GetCamera();
            if (!cam) {
                continue;
            }

            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
            int obs = 0;

            for (const auto& feat : kf->Features()) {
                if (!feat.has_landmark || feat.is_outlier) {
                    continue;
                }
                auto lm = map->GetLandmark(feat.landmark_id_);
                if (!lm || lm->IsBad()) {
                    continue;
                }

                Eigen::Vector2d proj;
                Eigen::Vector3d pc;
                if (!ProjectToPixel(*cam, kf->Pose(), lm->Position(), proj, &pc)) {
                    continue;
                }

                const Eigen::Vector2d err = feat.position - proj;
                const double err_norm = err.norm();
                if (err_norm > options_.max_reproj_error) {
                    continue;
                }

                const double w = HuberWeight(err_norm, options_.huber_delta);
                const Eigen::Matrix<double, 2, 6> J = PoseJacobian(*cam, pc);

                H.noalias() += w * J.transpose() * J;
                b.noalias() += w * (-J.transpose() * err);

                total_cost += w * err.squaredNorm();
                total_obs++;
                obs++;
            }

            if (obs < options_.min_pose_observations) {
                continue;
            }

            H += 1e-6 * Eigen::Matrix<double, 6, 6>::Identity();
            Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(b);
            if (!dx.allFinite()) {
                continue;
            }

            kf->SetPose(Sophus::SE3d::exp(dx) * kf->Pose());
        }

        // === Landmark optimization (fix poses) ===
        for (const auto& lm : landmarks) {
            if (!lm || lm->IsBad()) {
                continue;
            }

            Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
            Eigen::Vector3d b = Eigen::Vector3d::Zero();
            int obs = 0;

            for (const auto& [kf_id, feat_idx] : lm->Observations()) {
                if (local_kf_ids.find(kf_id) == local_kf_ids.end()) {
                    continue;
                }
                auto kf = map->GetFrame(kf_id);
                if (!kf) {
                    continue;
                }
                const auto cam = kf->GetCamera();
                if (!cam) {
                    continue;
                }
                if (feat_idx >= kf->Features().size()) {
                    continue;
                }
                const auto& feat = kf->Features()[feat_idx];
                if (!feat.has_landmark || feat.is_outlier || feat.landmark_id_ != lm->Id()) {
                    continue;
                }

                Eigen::Vector2d proj;
                Eigen::Vector3d pc;
                if (!ProjectToPixel(*cam, kf->Pose(), lm->Position(), proj, &pc)) {
                    continue;
                }

                const Eigen::Vector2d err = feat.position - proj;
                const double err_norm = err.norm();
                if (err_norm > options_.max_reproj_error) {
                    continue;
                }

                const double w = HuberWeight(err_norm, options_.huber_delta);
                const Eigen::Matrix<double, 2, 3> J_proj = ProjectionJacobian(*cam, pc);
                const Eigen::Matrix3d R = kf->Pose().rotationMatrix();
                const Eigen::Matrix<double, 2, 3> J = J_proj * R;

                H.noalias() += w * J.transpose() * J;
                b.noalias() += w * (-J.transpose() * err);
                obs++;
            }

            if (obs < options_.min_point_observations) {
                continue;
            }

            H += 1e-6 * Eigen::Matrix3d::Identity();
            Eigen::Vector3d dp = H.ldlt().solve(b);
            if (!dp.allFinite()) {
                continue;
            }
            lm->SetPosition(lm->Position() + dp);
        }

        if (total_obs == 0) {
            break;
        }

        if (std::abs(last_cost - total_cost) < 1e-6 * last_cost) {
            break;
        }
        last_cost = total_cost;
    }
}

}  // namespace visionx
