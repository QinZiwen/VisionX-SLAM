#include <gflags/gflags.h>

#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>

#include "common/dataset_tum_rgbd.h"
#include "common/logger.h"
#include "system/system.h"

using namespace visionx;

DEFINE_string(config, "", "Path to config file (key=value, same names as flags)");
DEFINE_string(dataset_dir, "../dataset/tum_rgbd", "Path to TUM RGB-D root directory");
DEFINE_string(sequence, "rgbd_dataset_freiburg1_desk", "TUM RGB-D sequence name");
DEFINE_bool(viewer_thread, false, "Run viewer in a background thread");
DEFINE_int32(viewer_loop_ms, 10, "Sleep time in viewer loop (ms)");

DEFINE_int32(min_matches, 20, "Minimum feature matches");
DEFINE_int32(min_inliers, 15, "Minimum inliers");
DEFINE_int32(min_keyframe_inliers, 20, "Minimum inliers to create a keyframe");
DEFINE_double(min_parallax, 5.0, "Minimum parallax to create a keyframe (pixels)");
DEFINE_double(max_reproj_error, 2.0, "Maximum reprojection error (pixels)");
DEFINE_int32(min_keyframe_gap, 3, "Minimum frame gap between keyframes");
DEFINE_bool(enable_culling, false, "Enable landmark/keyframe culling");

DEFINE_int32(min_landmark_observations, 2, "Minimum landmark observations before culling");
DEFINE_int32(min_landmarks_for_culling, 200, "Minimum landmarks before running culling");
DEFINE_int32(min_keyframes_for_culling, 3, "Minimum keyframes before culling");
DEFINE_int32(max_keyframes, 30, "Maximum keyframes kept in the local map");
DEFINE_int32(kf_min_shared_observations, 3,
             "Minimum shared observations to consider a landmark redundant");
DEFINE_double(kf_redundant_ratio, 0.9, "Redundant ratio threshold for keyframe culling");
DEFINE_double(landmark_max_reproj_error, 5.0, "Max reprojection error for landmark culling");

DEFINE_double(triangulation_max_reproj_error, 5.0, "Max reprojection error for triangulation");
DEFINE_double(triangulation_min_angle_deg, 1.0, "Min triangulation angle (deg)");

DEFINE_bool(enable_local_ba, true, "Enable local bundle adjustment");
DEFINE_int32(ba_window_size, 5, "Local BA window size");
DEFINE_int32(ba_iterations, 5, "Local BA iterations");
DEFINE_int32(ba_min_pose_observations, 20, "Minimum pose observations for BA");
DEFINE_int32(ba_min_point_observations, 2, "Minimum point observations for BA");
DEFINE_double(ba_huber_delta, 5.0, "Huber delta for BA");
DEFINE_double(ba_max_reproj_error, 5.0, "Max reprojection error for BA (pixels)");

namespace {

std::string Trim(const std::string& s) {
    const char* whitespace = " \t\r\n";
    const auto start = s.find_first_not_of(whitespace);
    if (start == std::string::npos) {
        return "";
    }
    const auto end = s.find_last_not_of(whitespace);
    return s.substr(start, end - start + 1);
}

std::unordered_map<std::string, std::string> LoadConfig(const std::string& path) {
    std::unordered_map<std::string, std::string> kv;
    std::ifstream fin(path);
    if (!fin.is_open()) {
        LOG(WARNING) << "Failed to open config file: " << path;
        return kv;
    }

    std::string line;
    while (std::getline(fin, line)) {
        auto hash_pos = line.find('#');
        if (hash_pos != std::string::npos) {
            line = line.substr(0, hash_pos);
        }
        line = Trim(line);
        if (line.empty()) {
            continue;
        }
        auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            continue;
        }
        std::string key = Trim(line.substr(0, eq_pos));
        std::string value = Trim(line.substr(eq_pos + 1));
        if (!key.empty()) {
            kv[key] = value;
        }
    }
    return kv;
}

void ApplyConfigIfDefault(const std::unordered_map<std::string, std::string>& kv) {
    for (const auto& [key, value] : kv) {
        gflags::CommandLineFlagInfo info;
        if (!gflags::GetCommandLineFlagInfo(key.c_str(), &info)) {
            LOG(WARNING) << "Unknown config key: " << key;
            continue;
        }
        if (info.is_default) {
            gflags::SetCommandLineOption(key.c_str(), value.c_str());
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    visionx::InitLogger(argv[0]);
    google::InstallFailureSignalHandler();

    gflags::SetUsageMessage("VisionX-SLAM runner");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (!FLAGS_config.empty()) {
        auto kv = LoadConfig(FLAGS_config);
        ApplyConfigIfDefault(kv);
    }

    Dataset::Ptr dataset = std::make_shared<DatasetTUMRGBD>(FLAGS_dataset_dir, FLAGS_sequence);

    if (!dataset->Load()) {
        LOG(ERROR) << "Failed to load dataset: " << FLAGS_dataset_dir << "/" << FLAGS_sequence;
        return -1;
    }

    const CameraIntrinsics& d = dataset->Intrinsics();
    auto camera = std::make_shared<Camera>(d.fx, d.fy, d.cx, d.cy, d.k1, d.k2, d.p1, d.p2);

    Tracking::Options options;
    options.min_matches = FLAGS_min_matches;
    options.min_inliers = FLAGS_min_inliers;
    options.min_keyframe_inliers = FLAGS_min_keyframe_inliers;
    options.min_parallax = FLAGS_min_parallax;
    options.max_reproj_error = FLAGS_max_reproj_error;
    options.min_keyframe_gap = FLAGS_min_keyframe_gap;
    options.enable_culling = FLAGS_enable_culling;
    options.min_landmark_observations = FLAGS_min_landmark_observations;
    options.min_landmarks_for_culling = FLAGS_min_landmarks_for_culling;
    options.min_keyframes_for_culling = FLAGS_min_keyframes_for_culling;
    options.max_keyframes = FLAGS_max_keyframes;
    options.kf_min_shared_observations = FLAGS_kf_min_shared_observations;
    options.kf_redundant_ratio = FLAGS_kf_redundant_ratio;
    options.landmark_max_reproj_error = FLAGS_landmark_max_reproj_error;
    options.triangulation_max_reproj_error = FLAGS_triangulation_max_reproj_error;
    options.triangulation_min_angle_deg = FLAGS_triangulation_min_angle_deg;
    options.enable_local_ba = FLAGS_enable_local_ba;
    options.ba_window_size = FLAGS_ba_window_size;
    options.ba_iterations = FLAGS_ba_iterations;
    options.ba_min_pose_observations = FLAGS_ba_min_pose_observations;
    options.ba_min_point_observations = FLAGS_ba_min_point_observations;
    options.ba_huber_delta = FLAGS_ba_huber_delta;
    options.ba_max_reproj_error = FLAGS_ba_max_reproj_error;

    Viewer::Ptr viewer = std::make_shared<Viewer>(FLAGS_viewer_thread);
    viewer->Start();

    System system(options, camera, viewer);
    LOG(INFO) << "System Initialized";
    system.run(dataset);

    LOG(INFO) << "Viewer run ...";
    while (true) {
        if (!FLAGS_viewer_thread) {
            viewer->RunOnce();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(FLAGS_viewer_loop_ms));
    }

    viewer->Stop();
    return 0;
}
