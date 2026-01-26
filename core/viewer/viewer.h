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
    void Run();

private:
    void InitPangolin();
    void DrawFrame();

    void DrawLandmarks();
    void DrawKeyFrames();
    void DrawCurrentCamera();
    void DrawCamera(const Sophus::SE3d& T_wc, float r, float g, float b);

private:
    bool initialized_ = false;
    std::unique_ptr<pangolin::OpenGlRenderState> s_cam_;
    pangolin::View* d_cam_ = nullptr;

    std::unique_ptr<pangolin::Var<std::string>> ui_lm_;
    std::unique_ptr<pangolin::Var<std::string>> ui_kf_;
    std::unique_ptr<pangolin::Var<std::string>> ui_fps_;

    std::chrono::steady_clock::time_point last_fps_time_;
    int frame_cnt_ = 0;

    Map::Ptr map_;
    Frame::Ptr current_frame_ = nullptr;

    bool use_thread_ = true;
    std::thread viewer_thread_;
    std::atomic<bool> running_{false};
};

}  // namespace visionx
