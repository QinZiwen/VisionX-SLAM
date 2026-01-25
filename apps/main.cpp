#include <opencv2/opencv.hpp>
#include <thread>

#include "common/dataset_tum_rgbd.h"
#include "common/logger.h"
#include "system/system.h"

using namespace visionx;

int main() {
    google::InstallFailureSignalHandler();

    std::string dataset_dir = "/Users/qinziwen/Projects/VisionX-SLAM/dataset/tum_rgbd";
    std::string sequence_name = "rgbd_dataset_freiburg1_desk";
    Dataset::Ptr dataset = std::make_shared<DatasetTUMRGBD>(dataset_dir, sequence_name);

    if (!dataset->Load()) {
        LOG(ERROR) << "Failed to load dataset: " << dataset_dir << "/" << sequence_name;
        return -1;
    }

    const CameraIntrinsics& d = dataset->Intrinsics();
    auto camera = std::make_shared<Camera>(d.fx, d.fy, d.cx, d.cy, d.k1, d.k2, d.p1, d.p2);

    Tracking::Options options;
    options.min_matches = 20;
    options.min_inliers = 15;
    options.min_keyframe_inliers = 20;
    options.min_parallax = 5.0;

    Viewer::Ptr viewer = std::make_shared<Viewer>(false);
    viewer->Start();

    System system(options, camera, viewer);
    LOG(INFO) << "System Initialized";
    system.run(dataset);

    LOG(INFO) << "Viewer run ...";
    while (true) {
        viewer->RunOnce();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    viewer->Stop();

    return 0;
}
