#include "viewer/viewer.h"

#include <unistd.h>  // usleep

namespace visionx {

Viewer::Viewer(Map::Ptr map) : map_(std::move(map)) {}

void Viewer::Start() {
    running_ = true;
    viewer_thread_ = std::thread(&Viewer::Run, this);
}

void Viewer::Stop() {
    running_ = false;
    if (viewer_thread_.joinable()) viewer_thread_.join();
}

void Viewer::UpdateCurrentFrame(Frame::Ptr frame) {
    current_frame_ = std::move(frame);
}

void Viewer::Run() {
    pangolin::CreateWindowAndBind("VisionX Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0, -1, 0));

    pangolin::View& d_cam = pangolin::CreateDisplay()
                                .SetBounds(0, 1, 0, 1, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    while (running_ && !pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        DrawLandmarks();
        DrawKeyFrames();
        DrawCurrentCamera();

        pangolin::FinishFrame();
        usleep(5000);  // ~200 FPS
    }
}

void Viewer::DrawLandmarks() {
    const auto& landmarks = map_->Landmarks();
    if (landmarks.empty()) return;

    glPointSize(2.0f);
    glBegin(GL_POINTS);
    glColor3f(0.0f, 0.0f, 1.0f);  // 蓝色地图点

    for (const auto& kv : landmarks) {
        const auto& lm = kv.second;
        const auto& p = lm->Position();
        glVertex3d(p.x(), p.y(), p.z());
    }
    glEnd();
}

void Viewer::DrawKeyFrames() {
    const auto& keyframes = map_->KeyFrames();
    if (keyframes.size() < 2) return;

    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glColor3f(0.0f, 0.0f, 1.0f);  // 蓝色历史轨迹

    Frame::Ptr last = nullptr;
    for (const auto& kv : keyframes) {
        if (!last) {
            last = kv.second;
            continue;
        }

        Eigen::Vector3d p1 = last->Pose().inverse().translation();
        Eigen::Vector3d p2 = kv.second->Pose().inverse().translation();

        glVertex3d(p1.x(), p1.y(), p1.z());
        glVertex3d(p2.x(), p2.y(), p2.z());

        last = kv.second;
    }
    glEnd();
}

void Viewer::DrawCurrentCamera() {
    if (!current_frame_) return;

    Sophus::SE3d T_wc = current_frame_->Pose().inverse();
    Eigen::Matrix4d Twc = T_wc.matrix();

    glPushMatrix();
    glMultMatrixd(Twc.data());

    const float w = 0.3f;
    const float h = 0.2f;
    const float z = 0.6f;

    glLineWidth(3.0f);
    glColor3f(1.0f, 0.0f, 0.0f);  // 红色当前相机

    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();

    glPopMatrix();
}

}  // namespace visionx
