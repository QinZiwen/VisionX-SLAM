#include "map/map.h"

namespace visionx {

void Map::InsertKeyFrame(Frame::Ptr frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    keyframes_[frame->Id()] = frame;
}

void Map::InsertLandmark(Landmark::Ptr landmark) {
    std::lock_guard<std::mutex> lock(mutex_);
    landmarks_[landmark->Id()] = landmark;
}

void Map::RemoveKeyFrame(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    keyframes_.erase(id);
}

void Map::RemoveLandmark(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    landmarks_.erase(id);
}

void Map::removeAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    keyframes_.clear();
    landmarks_.clear();
}

Frame::Ptr Map::GetFrame(uint64_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = keyframes_.find(id);
    if (it == keyframes_.end()) {
        return nullptr;
    }
    return it->second;
}

Landmark::Ptr Map::GetLandmark(uint64_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = landmarks_.find(id);
    if (it == landmarks_.end()) {
        return nullptr;
    }
    return it->second;
}

size_t Map::LandmarkSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return landmarks_.size();
}

}  // namespace visionx
