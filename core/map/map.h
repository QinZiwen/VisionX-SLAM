#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "frame/frame.h"
#include "map/landmark.h"

namespace visionx {

class Map {
public:
    using Ptr = std::shared_ptr<Map>;

    void InsertKeyFrame(Frame::Ptr frame);
    void InsertLandmark(Landmark::Ptr landmark);

    const std::unordered_map<uint64_t, Frame::Ptr>& KeyFrames() const {
        return keyframes_;
    }

    const std::unordered_map<uint64_t, Landmark::Ptr>& Landmarks() const {
        return landmarks_;
    }

private:
    std::unordered_map<uint64_t, Frame::Ptr> keyframes_;
    std::unordered_map<uint64_t, Landmark::Ptr> landmarks_;
    mutable std::mutex mutex_;
};

}  // namespace visionx
