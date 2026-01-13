#include <pangolin/pangolin.h>

int main() {
    // 创建窗口
    pangolin::CreateWindowAndBind("VisionX-SLAM Viewer", 640, 480);
    glEnable(GL_DEPTH_TEST);

    // 设置相机投影
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,100),
        pangolin::ModelViewLookAt(-2,2,-2, 0,0,0, pangolin::AxisY)
    );

    // 在窗口中创建交互视图
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
        .SetHandler(&handler);

    while( !pangolin::ShouldQuit() ) {
        // 清除屏幕
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        // 画一个简单的彩色立方体
        pangolin::glDrawColouredCube();

        // 推进窗口事件
        pangolin::FinishFrame();
    }
    
    return 0;
}