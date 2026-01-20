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

}  // namespace visionx
