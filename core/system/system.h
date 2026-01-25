#pragma once

#include <memory>

#include "camera/camera.h"
#include "common/dataset.h"
#include "frontend/tracking.h"
#include "map/map.h"
#include "viewer/viewer.h"

namespace visionx {

class System {
public:
    using Ptr = std::shared_ptr<System>;

    System(const Tracking::Options& tracking_options, Camera::Ptr camera, Viewer::Ptr viewer);
    ~System();

    void run(Dataset::Ptr dataset);
    // 单帧接口（供测试 / dataset loop 调用）
    void ProcessFrame(uint64_t id, double timestamp, const cv::Mat& image);

private:
    Camera::Ptr camera_;
    Map::Ptr map_;
    FeatureExtractor::Ptr extractor_;
    FeatureMatcher::Ptr matcher_;
    Tracking::Ptr tracking_;
    Viewer::Ptr viewer_;

    std::thread tracking_thread_;
};

}  // namespace visionx
