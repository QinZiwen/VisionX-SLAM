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

    explicit Viewer(bool use_thread = true);

    void SetMap(Map::Ptr map) { map_ = map; }

    void Start();
    void Stop();

    // Tracking 每帧调用
    void UpdateCurrentFrame(Frame::Ptr frame);

    void RunOnce();

private:
    void Run();
    void DrawWindow();

    void DrawLandmarks();
    void DrawKeyFrames();
    void DrawCurrentCamera();

private:
    Map::Ptr map_;
    Frame::Ptr current_frame_ = nullptr;

    bool use_thread_ = true;
    std::thread viewer_thread_;
    std::atomic<bool> running_{false};
};

}  // namespace visionx
