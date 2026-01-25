#include "dataset_tum_rgbd.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "common/logger.h"

namespace visionx {

bool DatasetTUMRGBD::Load() {
    if (!LoadIntrinsics()) {
        LOG(ERROR) << "Failed to load intrinsics for " << sequence_name_ << std::endl;
        return false;
    }

    std::string dataset_full_dir = dataset_dir_ + "/" + sequence_name_;
    LOG(INFO) << "Loading TUM RGB-D sequence from: " << dataset_full_dir;

    // 1. 读取 RGB 索引
    auto rgb_map = ReadList(dataset_full_dir + "/rgb.txt");
    // 2. 读取 Depth 索引
    auto depth_map = ReadList(dataset_full_dir + "/depth.txt");
    // 3. 读取 Ground Truth
    auto gt_map = ReadGT(dataset_full_dir + "/groundtruth.txt");

    // 4. 对齐数据
    Associate(rgb_map, depth_map, gt_map);

    LOG(INFO) << "Successfully associated " << entries_.size() << " frames.";
    return !entries_.empty();
}

std::map<double, std::string> DatasetTUMRGBD::ReadList(const std::string& filename) {
    LOG(INFO) << "Reading list from: " << filename;
    std::map<double, std::string> res;
    std::ifstream f(filename);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        double ts;
        std::string path;
        ss >> ts >> path;
        res[ts] = path;
    }
    return res;
}

std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> DatasetTUMRGBD::ReadGT(
    const std::string& filename) {
    LOG(INFO) << "Reading GT from: " << filename;
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> gt_map;
    std::ifstream f_gt(filename);
    std::string line;
    while (std::getline(f_gt, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        double ts, tx, ty, tz, qx, qy, qz, qw;
        ss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        gt_map[ts] = {Eigen::Vector3d(tx, ty, tz), Eigen::Quaterniond(qw, qx, qy, qz)};
    }
    return gt_map;
}

void DatasetTUMRGBD::Associate(
    const std::map<double, std::string>& rgb_map, const std::map<double, std::string>& depth_map,
    const std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>>& gt_map) {
    entries_.clear();

    for (auto const& [ts_rgb, rgb_path] : rgb_map) {
        // --- 1. 查找最接近的深度图 ---
        auto it_d = depth_map.lower_bound(ts_rgb);
        auto best_d = it_d;

        // 如果 lower_bound 不是第一个元素，比较它和前一个元素哪个更近
        if (it_d != depth_map.begin()) {
            auto it_prev = std::prev(it_d);
            if (it_d == depth_map.end() ||
                std::abs(it_prev->first - ts_rgb) < std::abs(it_d->first - ts_rgb)) {
                best_d = it_prev;
            }
        }

        // 检查误差是否在阈值内
        double diff = std::abs(best_d->first - ts_rgb);
        if (best_d == depth_map.end() || diff > associate_max_diff_) {
            LOG(WARNING) << "Cannot find a corresponding depth image for timestamp: " << ts_rgb
                         << ", diff: " << diff << ", threshold: " << associate_max_diff_;
            continue;
        }

        // --- 2. 查找最接近的真值 (GT) ---
        auto it_g = gt_map.lower_bound(ts_rgb);
        auto best_g = it_g;

        if (it_g != gt_map.begin()) {
            auto it_prev = std::prev(it_g);
            if (it_g == gt_map.end() ||
                std::abs(it_prev->first - ts_rgb) < std::abs(it_g->first - ts_rgb)) {
                best_g = it_prev;
            }
        }

        diff = std::abs(best_g->first - ts_rgb);
        if (best_g == gt_map.end() || diff > associate_max_diff_) {
            LOG(WARNING) << "Cannot find a corresponding GT for timestamp: " << ts_rgb
                         << ", diff: " << diff << ", threshold: " << associate_max_diff_;
            continue;
        }

        // --- 3. 填充数据 ---
        ImageEntry entry;
        entry.timestamp = ts_rgb;
        entry.rgb_path = dataset_dir_ + "/" + sequence_name_ + "/" + rgb_path;
        entry.depth_path = dataset_dir_ + "/" + sequence_name_ + "/" + best_d->second;
        entry.t = best_g->second.first;
        entry.q = best_g->second.second;
        entries_.push_back(entry);
    }
}

bool DatasetTUMRGBD::LoadIntrinsics() {
    std::string version;
    if (sequence_name_.find("freiburg1") != std::string::npos)
        version = "1";
    else if (sequence_name_.find("freiburg2") != std::string::npos)
        version = "2";
    else if (sequence_name_.find("freiburg3") != std::string::npos)
        version = "3";
    else {
        LOG(ERROR) << "Unknown sequence version for: " << sequence_name_;
        return false;
    }

    std::string intrin_file = dataset_dir_ + "/color_camera_freiburg" + version + ".txt";
    std::ifstream fin(intrin_file);
    if (!fin.is_open()) {
        LOG(ERROR) << "Cannot open intrinsics file: " << intrin_file;
        return false;
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        // 按顺序读取 9 个参数
        // TUM 格式: fx fy cx cy d0(k1) d1(k2) d2(p1) d3(p2) d4(k3)
        if (ss >> intrinsics_.fx >> intrinsics_.fy >> intrinsics_.cx >> intrinsics_.cy >>
            intrinsics_.k1 >> intrinsics_.k2 >> intrinsics_.p1 >> intrinsics_.p2 >>
            intrinsics_.k3) {
            LOG(INFO) << "Successfully loaded intrinsics from " << intrin_file;
            LOG(INFO) << "\n" << intrinsics_;  // 使用你之前定义的 operator<<
            return true;                       // 成功读取到第一行有效参数即可退出
        } else {
            LOG(WARNING) << "Found data line but failed to parse 9 parameters: " << line;
            return false;
        }
    }

    LOG(ERROR) << "No valid intrinsics found in file: " << intrin_file;
    return false;
}

}  // namespace visionx