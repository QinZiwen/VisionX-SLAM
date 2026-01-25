#include "system/system.h"

#include <opencv2/opencv.hpp>
#include <thread>

#include "common/logger.h"
#include "feature/orb_extractor.h"
#include "feature/orb_matcher.h"

namespace visionx {

System::System(const Tracking::Options& tracking_options, Camera::Ptr camera, Viewer::Ptr viewer)
    : camera_(std::move(camera)) {
    map_ = std::make_shared<Map>();
    extractor_ = std::make_shared<ORBExtractor>();
    matcher_ = std::make_shared<ORBMatcher>();

    tracking_ = std::make_shared<Tracking>(tracking_options, extractor_, matcher_, map_);

    if (viewer) {
        viewer_ = viewer;
    } else {
        viewer_ = std::make_shared<Viewer>(true);
    }
    viewer_->SetMap(map_);
}

System::~System() {
    if (viewer_) {
        viewer_->Stop();
    }

    if (tracking_thread_.joinable()) {
        tracking_thread_.join();
    }
}

void System::run(Dataset::Ptr dataset) {
    tracking_thread_ = std::thread(
        [this](Dataset::Ptr dataset) {
            const std::vector<ImageEntry>& images = dataset->ImageEntries();

            LOG(INFO) << "System::run ...";
            for (size_t i = 0; i < images.size(); ++i) {
                LOG(INFO) << "Processing image " << images[i].rgb_path;
                ProcessFrame(static_cast<uint64_t>(i), images[i].timestamp,
                             cv::imread(images[i].rgb_path));
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        },
        dataset);
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
