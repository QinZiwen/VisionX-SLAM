#include "common/logger.h"
#include <glog/logging.h>

namespace visionx {

void InitLogger(const char* argv) {
    google::InitGoogleLogging(argv);
    google::SetStderrLogging(google::INFO);
    FLAGS_colorlogtostderr = true;
}

}
