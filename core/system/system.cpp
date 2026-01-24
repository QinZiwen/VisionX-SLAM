#include "system/system.h"

#include <thread>

#include "common/logger.h"
#include "feature/orb_extractor.h"
#include "feature/orb_matcher.h"

namespace visionx {

System::System(const Tracking::Options& tracking_options,
               std::shared_ptr<Camera> camera, Viewer::Ptr viewer)
    : camera_(std::move(camera)) {
    map_ = std::make_shared<Map>();

    extractor_ = std::make_shared<ORBExtractor>();
    matcher_ = std::make_shared<ORBMatcher>();

    tracking_ = std::make_shared<Tracking>(tracking_options, extractor_,
                                           matcher_, map_);

    if (viewer) {
        viewer_ = viewer;
    } else {
        viewer_ = std::make_shared<Viewer>(true);
    }
    viewer_->SetMap(map_);
}

void System::run() {
    std::thread t = std::thread([this]() {
        const int num_frames = 50;

        LOG(INFO) << "System::run ...";
        for (int i = 0; i < num_frames; ++i) {
            // 生成一张假的灰度图
            cv::Mat image(480, 640, CV_8UC1, cv::Scalar(0));
            cv::randu(image, 0, 255);

            ProcessFrame(static_cast<uint64_t>(i),
                         i * 0.1,  // timestamp
                         image);

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    if (t.joinable()) {
        t.join();
    }
}

void System::ProcessFrame(uint64_t id, double timestamp, const cv::Mat& image) {
    auto frame = std::make_shared<Frame>(id, timestamp, camera_, image);

    LOG(INFO) << "Processing frame " << id << ", timestamp " << timestamp << "";
    tracking_->ProcessFrame(frame);

    if (viewer_) {
        viewer_->UpdateCurrentFrame(frame);
    }
}

}  // namespace visionx
