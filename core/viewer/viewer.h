#pragma once

#include <pangolin/pangolin.h>

#include <Eigen/Core>
#include <atomic>
#include <memory>
#include <thread>

#include "frame/frame.h"
#include "map/map.h"

namespace visionx {

class Viewer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Viewer>;

    explicit Viewer(Map::Ptr map);

    void Start();
    void Stop();

    // Tracking 每帧调用
    void UpdateCurrentFrame(Frame::Ptr frame);

private:
    void Run();

    void DrawLandmarks();
    void DrawKeyFrames();
    void DrawCurrentCamera();

private:
    Map::Ptr map_;
    Frame::Ptr current_frame_ = nullptr;

    std::thread viewer_thread_;
    std::atomic<bool> running_{false};
};

}  // namespace visionx
