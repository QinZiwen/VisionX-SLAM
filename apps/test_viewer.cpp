#include <opencv2/opencv.hpp>
#include <thread>

#include "common/logger.h"
#include "system/system.h"

using namespace visionx;

int main() {
    google::InstallFailureSignalHandler();

    auto camera = std::make_shared<Camera>(520.0, 520.0,  // fx, fy
                                           325.0, 249.0   // cx, cy
    );

    Tracking::Options options;
    options.min_matches = 20;
    options.min_inliers = 15;
    options.min_keyframe_inliers = 20;
    options.min_parallax = 5.0;

    Viewer::Ptr viewer = std::make_shared<Viewer>(false);
    viewer->Start();

    System system(options, camera, viewer);
    LOG(INFO) << "System Initialized";
    system.run();

    while (true) {
        viewer->RunOnce();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    viewer->Stop();

    return 0;
}
