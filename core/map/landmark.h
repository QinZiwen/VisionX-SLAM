#pragma once

#include <Eigen/Core>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace visionx {

class Frame;

class Landmark {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Landmark>;

    Landmark(uint64_t id, const Eigen::Vector3d& pos) : id_(id), pos_(pos) {}

    uint64_t Id() const { return id_; }

    Eigen::Vector3d Position() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pos_;
    }

    void SetPosition(const Eigen::Vector3d& pos) {
        std::lock_guard<std::mutex> lock(mutex_);
        pos_ = pos;
    }

    // keyframe_id -> feature index
    void AddObservation(uint64_t keyframe_id, size_t feature_idx) {
        std::lock_guard<std::mutex> lock(mutex_);
        observations_[keyframe_id] = feature_idx;
    }

    void RemoveObservation(uint64_t keyframe_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        observations_.erase(keyframe_id);
    }

    size_t ObservationCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return observations_.size();
    }

    const std::unordered_map<uint64_t, size_t>& Observations() const {
        return observations_;
    }

    bool IsBad() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return is_bad_;
    }

    void SetBad(bool bad = true) {
        std::lock_guard<std::mutex> lock(mutex_);
        is_bad_ = bad;
    }

private:
    uint64_t id_;
    Eigen::Vector3d pos_;

    std::unordered_map<uint64_t, size_t> observations_;
    mutable std::mutex mutex_;
    bool is_bad_ = false;
};

}  // namespace visionx
