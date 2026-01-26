#include "viewer/viewer.h"

#include <fmt/core.h>
#include <fmt/format.h>
#include <pangolin/pangolin.h>
#include <unistd.h>

#include <chrono>

#include "common/logger.h"

namespace visionx {

Viewer::Viewer(bool use_thread) : use_thread_(use_thread) {}

void Viewer::Start() {
    running_ = true;
    if (use_thread_) {
        viewer_thread_ = std::thread(&Viewer::Run, this);
    }
}

void Viewer::Stop() {
    running_ = false;
    if (viewer_thread_.joinable()) {
        viewer_thread_.join();
    }
}

void Viewer::UpdateCurrentFrame(Frame::Ptr frame) {
    if (!frame) return;
    current_frame_ = frame->Clone(false);
}

/* =========================
 *  Pangolin Init
 * ========================= */
void Viewer::InitPangolin() {
    if (initialized_) return;

    pangolin::CreateWindowAndBind("VisionX Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.f, 0.f, 0.f, 1.f);  // 黑色背景

    s_cam_ = std::make_unique<pangolin::OpenGlRenderState>(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0, -1, 0));

    d_cam_ = &pangolin::CreateDisplay()
                  .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                  .SetHandler(new pangolin::Handler3D(*s_cam_));

    // ===== UI (右上角，小尺寸) =====
    pangolin::CreatePanel("ui").SetBounds(0.88, 1.0, 0.85, 1.0);
    ui_fps_ = std::make_unique<pangolin::Var<std::string>>("ui.FPS", "0.0");
    ui_kf_ = std::make_unique<pangolin::Var<std::string>>("ui.KeyFrames", "0");
    ui_lm_ = std::make_unique<pangolin::Var<std::string>>("ui.Landmarks", "0");

    last_fps_time_ = std::chrono::steady_clock::now();
    frame_cnt_ = 0;

    initialized_ = true;
}

/* =========================
 *  Render One Frame
 * ========================= */
void Viewer::DrawFrame() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam_->Activate(*s_cam_);

    DrawLandmarks();
    DrawKeyFrames();
    DrawCurrentCamera();

    pangolin::FinishFrame();

    // ===== FPS =====
    frame_cnt_++;
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fps_time_).count();

    if (dt > 1000.0) {
        double fps = frame_cnt_ * 1000.0 / dt;

        *ui_fps_ = fmt::format("{:.1f}", fps);
        *ui_kf_ = std::to_string(map_->KeyFrames().size());
        *ui_lm_ = std::to_string(map_->Landmarks().size());

        frame_cnt_ = 0;
        last_fps_time_ = now;
    }
}

/* =========================
 *  Thread Loop
 * ========================= */
void Viewer::Run() {
    LOG(INFO) << "[Viewer] started";
    InitPangolin();

    while (running_ && !pangolin::ShouldQuit()) {
        DrawFrame();
        usleep(5000);
    }

    LOG(INFO) << "[Viewer] exited";
}

void Viewer::RunOnce() {
    InitPangolin();
    if (pangolin::ShouldQuit()) return;
    DrawFrame();
}

/* =========================
 *  Draw Elements
 * ========================= */
void Viewer::DrawLandmarks() {
    const auto& landmarks = map_->Landmarks();
    if (landmarks.empty()) return;

    constexpr int kStride = 5;  // 降采样
    int cnt = 0;

    glPointSize(2.0f);
    glBegin(GL_POINTS);
    glColor3f(1.f, 1.f, 1.f);  // 白色

    for (const auto& kv : landmarks) {
        if (++cnt % kStride != 0) continue;
        const auto& p = kv.second->Position();
        glVertex3d(p.x(), p.y(), p.z());
    }
    glEnd();
}

void Viewer::DrawKeyFrames() {
    const auto& keyframes = map_->KeyFrames();
    if (keyframes.size() < 2) return;

    // ===== 轨迹 =====
    glLineWidth(2.0f);
    glColor3f(0.3f, 1.0f, 0.3f);  // 浅绿色
    glBegin(GL_LINE_STRIP);

    for (const auto& kv : keyframes) {
        const auto& kf = kv.second;
        Eigen::Vector3d t = kf->Pose().inverse().translation();
        glVertex3d(t.x(), t.y(), t.z());
    }
    glEnd();

    // ===== 相机模型 =====
    for (const auto& kv : keyframes) {
        DrawCamera(kv.second->Pose().inverse(), 0.3f, 1.0f, 0.3f);
    }
}

void Viewer::DrawCurrentCamera() {
    if (!current_frame_) return;

    Sophus::SE3d T_wc = current_frame_->Pose().inverse();
    Eigen::Vector3d curr_pos = T_wc.translation();

    // 1. 画当前相机模型（红色）
    DrawCamera(T_wc, 1.0f, 0.0f, 0.0f);

    // 2. 从最后一个 KeyFrame 连一条红线
    const auto& keyframes = map_->KeyFrames();
    if (keyframes.empty()) return;

    // 取“最后一个”关键帧
    auto last_it = keyframes.end();
    --last_it;

    Sophus::SE3d T_last_wc = last_it->second->Pose().inverse();
    Eigen::Vector3d last_pos = T_last_wc.translation();

    glLineWidth(2.5f);
    glColor3f(1.0f, 0.0f, 0.0f);  // 红色

    glBegin(GL_LINES);
    glVertex3d(last_pos.x(), last_pos.y(), last_pos.z());
    glVertex3d(curr_pos.x(), curr_pos.y(), curr_pos.z());
    glEnd();
}

void Viewer::DrawCamera(const Sophus::SE3d& T_wc, float r, float g, float b) {
    Eigen::Matrix4d Twc = T_wc.matrix();

    glPushMatrix();
    glMultMatrixd(Twc.data());

    const float w = 0.3f;
    const float h = 0.2f;
    const float z = 0.6f;

    glLineWidth(2.0f);
    glColor3f(r, g, b);

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
