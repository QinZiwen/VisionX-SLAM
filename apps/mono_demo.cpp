#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <glog/logging.h>

#include "common/logger.h"
#include "camera/camera.h"
#include "frame/frame.h"
#include "feature/orb_extractor.h"

int main(int argc, char** argv) {
    visionx::InitLogger(argv[0]);

    if (argc < 2) {
        LOG(ERROR) << "Usage: mono_demo image_path";
        return -1;
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        LOG(ERROR) << "Failed to load image.";
        return -1;
    }

    auto camera = std::make_shared<visionx::Camera>(
        520.9, 521.0, 325.1, 249.7);

    visionx::Frame frame(
        0, 0.0, camera, image);

    visionx::ORBExtractor extractor;
    extractor.Extract(frame);

    LOG(INFO) << "Extracted features: "
              << frame.Features().size();

    // ===== 可视化 =====
    cv::Mat vis;
    cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);

    for (const auto& feat : frame.Features()) {
        cv::Point2f pt(
            static_cast<float>(feat.position.x()),
            static_cast<float>(feat.position.y()));

        // LOG(INFO) << "response: " << feat.response;
        int radius = std::min(5, static_cast<int>(feat.response * 1e4));
        cv::circle(vis, pt, radius, cv::Scalar(0, 255, 0), 1);
    }

    cv::imshow("ORB Features", vis);
    cv::waitKey(0);

    return 0;
}
