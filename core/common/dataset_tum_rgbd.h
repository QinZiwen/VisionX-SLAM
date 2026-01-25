#pragma once
#include <map>

#include "dataset.h"

namespace visionx {

class DatasetTUMRGBD : public Dataset {
public:
    DatasetTUMRGBD(const std::string& dataset_dir, const std::string& sequence_name)
        : dataset_dir_(dataset_dir), sequence_name_(sequence_name) {}
    virtual bool Load() override;

private:
    // TUM RGB-D 核心助手函数：将两个不同步的时间戳列表关联起来
    void Associate(const std::map<double, std::string>& rgb_map,
                   const std::map<double, std::string>& depth_map,
                   const std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>>& gt_map);
    bool LoadIntrinsics();
    std::map<double, std::string> ReadList(const std::string& filename);
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> ReadGT(
        const std::string& dataset_full_dir);

    std::string dataset_dir_;
    std::string sequence_name_;
    const double associate_max_diff_ = 0.02;
};

}  // namespace visionx