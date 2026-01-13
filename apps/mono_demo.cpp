#include "common/logger.h"
#include <glog/logging.h>

int main(int argc, char** argv) {
    visionx::InitLogger(argv[0]);

    LOG(INFO) << "VisionX-SLAM system starting...";
    return 0;
}
