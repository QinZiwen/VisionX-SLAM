#include <iostream>
#include <sophus/se3.hpp>
#include <Eigen/Core>

int main() {
    // 创建一个绕 Z 轴旋转 90 度的旋转向量
    Eigen::Vector3d rotation_vector(0, 0, M_PI / 2.0);
    
    // 使用 Sophus 将其转换为旋转矩阵 (SO3)
    Sophus::SO3d so3 = Sophus::SO3d::exp(rotation_vector);
    
    std::cout << "Sophus 旋转矩阵 (SO3) 验证成功：\n" << so3.matrix() << std::endl;
    
    // 创建一个平移向量
    Eigen::Vector3d t(1, 2, 3);
    
    // 构造位姿 (SE3)
    Sophus::SE3d pose(so3, t);
    
    std::cout << "\nSE3 位姿矩阵：\n" << pose.matrix() << std::endl;
    std::cout << "\n[Success] VisionX-SLAM 环境测试通过！" << std::endl;

    return 0;
}