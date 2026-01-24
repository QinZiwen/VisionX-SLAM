#pragma once

#include <memory>

#include "camera/camera.h"
#include "frontend/tracking.h"
#include "map/map.h"
#include "viewer/viewer.h"

namespace visionx {

class System {
public:
    using Ptr = std::shared_ptr<System>;

    System(const Tracking::Options& tracking_options, Camera::Ptr camera,
           Viewer::Ptr viewer = nullptr);

    void run();
    // 单帧接口（供测试 / dataset loop 调用）
    void ProcessFrame(uint64_t id, double timestamp, const cv::Mat& image);

private:
    std::shared_ptr<Camera> camera_;
    std::shared_ptr<Map> map_;

    std::shared_ptr<FeatureExtractor> extractor_;
    std::shared_ptr<FeatureMatcher> matcher_;
    std::shared_ptr<Tracking> tracking_;

    std::shared_ptr<Viewer> viewer_;
};

}  // namespace visionx
