#pragma once

#include <memory>

#include "frame/frame.h"
#include "map/map.h"

namespace visionx {

class LocalBA {
public:
    struct Options {
        int window_size = 5;
        int max_iterations = 5;
        int min_pose_observations = 20;
        int min_point_observations = 2;
        double huber_delta = 5.0;
        double max_reproj_error = 5.0;
    };

    explicit LocalBA(const Options& options) : options_(options) {}

    void Optimize(const Map::Ptr& map, const Frame::Ptr& ref_kf);

private:
    Options options_;
};

}  // namespace visionx
