
#include "mj_env.hpp"
#include "ppo.hpp"
#include "rl_config.hpp"

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

// MuJoCo visualization state
static mjvCamera cam;
static mjvOption opt;
static mjvScene scn;
static mjrContext con;

// mouse state
static bool button_left = false, button_middle = false, button_right = false;
static double lastx = 0.0, lasty = 0.0;

// 사용자 컨텍스트
struct ViewerUser {
  MjEnv *env = nullptr;
  ActorCritic net = nullptr; // 빈 상태로 초기화 (ModuleHolder)
  torch::Tensor obs;
  bool paused = false;
  bool show_contacts = false;
  bool stochastic = false;   // 기본: 결정적(평균)
  bool sync_realtime = true; // T toggle
  double vx_cmd = 0.0;
  double last_reward = 0.0;
  double fps = 0.0;
  int screenshot_id = 0;
};

// 관측 정규화(옵션)
struct NormStats {
  torch::Tensor mean;
  torch::Tensor std;
  bool valid = false;
};

static NormStats load_norm_stats_if_any(const std::string &path, int obs_dim) {
  NormStats ns;
  try {
    YAML::Node y = YAML::LoadFile(path);
    if (!y["mean"] || !y["std"])
      return ns;
    auto vmean = y["mean"].as<std::vector<double>>();
    auto vstd = y["std"].as<std::vector<double>>();
    if ((int)vmean.size() != obs_dim || (int)vstd.size() != obs_dim) {
      std::cerr << "[viewer] obs_norm.yaml dimension mismatch: file("
                << vmean.size() << ") vs obs_dim(" << obs_dim << ")\n";
      return ns;
    }
    ns.mean = torch::from_blob((void *)vmean.data(), {obs_dim}, torch::kDouble)
                  .clone()
                  .to(torch::kFloat32);
    ns.std = torch::from_blob((void *)vstd.data(), {obs_dim}, torch::kDouble)
                 .clone()
                 .to(torch::kFloat32);
    ns.std = torch::clamp(ns.std, 1e-6, 1e6);
    ns.valid = true;
    std::cout << "[viewer] Loaded normalization: " << path << std::endl;
  } catch (...) {
  }
  return ns;
}

static inline torch::Tensor normalize_obs(const torch::Tensor &obs,
                                          const NormStats &ns) {
  if (!ns.valid)
    return obs;
  auto x = obs.to(torch::kFloat32);
  x = (x - ns.mean) / ns.std;
  x = torch::clamp(x, -10.0, 10.0);
  return x;
}

// 유틸: 바디 프레임 X 속도
static double body_vx_in_body(const mjModel *m, const mjData *d) {
  const double *q = &d->qpos[3];
  double R[9];
  mju_quat2Mat(R, q);
  double v_world[3] = {d->qvel[0], d->qvel[1], d->qvel[2]};
  double v_body_x = R[0] * v_world[0] + R[3] * v_world[1] + R[6] * v_world[2];
  return v_body_x;
}

// 키 콜백
static void keyboard(GLFWwindow *window, int key, int scancode, int act,
                     int mods) {
  if (act != GLFW_PRESS)
    return;
  auto *u = static_cast<ViewerUser *>(glfwGetWindowUserPointer(window));
  if (!u || !u->env)
    return;

  switch (key) {
  case GLFW_KEY_ESCAPE:
    glfwSetWindowShouldClose(window, GLFW_TRUE);
    break;
  case GLFW_KEY_SPACE:
    u->paused = !u->paused;
    break;
  case GLFW_KEY_R: {
    u->obs = u->env->reset();
    u->last_reward = 0.0;
  } break;
  case GLFW_KEY_LEFT_BRACKET: {
    u->vx_cmd = std::max(0.0, u->vx_cmd - 0.05);
    u->env->set_target_vx(u->vx_cmd);
  } break;
  case GLFW_KEY_RIGHT_BRACKET: {
    u->vx_cmd = std::min(1.5, u->vx_cmd + 0.05);
    u->env->set_target_vx(u->vx_cmd);
  } break;
  case GLFW_KEY_C: {
    u->show_contacts = !u->show_contacts;
    opt.flags[mjVIS_CONTACTPOINT] = u->show_contacts ? 1 : 0;
    opt.flags[mjVIS_CONTACTFORCE] = u->show_contacts ? 1 : 0;
  } break;
  case GLFW_KEY_S:
    u->stochastic = true;
    break;
  case GLFW_KEY_D:
    u->stochastic = false;
    break;
  case GLFW_KEY_T:
    u->sync_realtime = !u->sync_realtime;
    break;
  case GLFW_KEY_F12: {
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    std::vector<unsigned char> rgb(3 * w * h);
    mjr_readPixels(rgb.data(), nullptr, mjrRect{0, 0, w, h}, &con);
    char fname[128];
    snprintf(fname, sizeof(fname), "screenshot_%04d.ppm", u->screenshot_id++);
    std::ofstream f(fname, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    for (int row = h - 1; row >= 0; --row)
      f.write(reinterpret_cast<char *>(rgb.data() + row * w * 3), w * 3);
    f.close();
    std::cout << "[viewer] Saved " << fname << std::endl;
  } break;
  default:
    break;
  }
}

// 마우스 콜백
static void mouse_button(GLFWwindow *window, int button, int act, int mods) {
  button_left =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
  button_middle =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
  button_right =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
  glfwGetCursorPos(window, &lastx, &lasty);
}

static void mouse_move(GLFWwindow *window, double xpos, double ypos) {
  if (!button_left && !button_middle && !button_right)
    return;
  auto *u = static_cast<ViewerUser *>(glfwGetWindowUserPointer(window));
  if (!u)
    return;
  mjModel *m = u->env->model();

  double dx = xpos - lastx;
  double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ||
                   (glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  int action = 0;
  if (button_right)
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_ROTATE_H;
  else if (button_left)
    action = mod_shift ? mjMOUSE_MOVE_V : mjMOUSE_ROTATE_V;
  else if (button_middle)
    action = mjMOUSE_ZOOM;

  mjv_moveCamera(m, action, dx / 5.0, dy / 5.0, &scn, &cam);
}

static void scroll(GLFWwindow *window, double xoffset, double yoffset) {
  auto *u = static_cast<ViewerUser *>(glfwGetWindowUserPointer(window));
  if (!u)
    return;
  mjModel *m = u->env->model();
  mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

// 창 리사이즈 콜백 (빈 처리)
static void framebuffer_size_callback(GLFWwindow *window, int width,
                                      int height) {
  (void)window;
  (void)height;
  (void)width;
  // MuJoCo 2.3.x 온스크린은 별도 리사이즈 호출 필요 없음.
}

int main(int argc, char **argv) {
  // 인자 파싱
  bool force_cpu = false, force_cuda = false;
  std::string ckpt_path, cfg_path, xml_override;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--cpu")
      force_cpu = true;
    else if (a == "--cuda")
      force_cuda = true;
    else if (a.rfind("--xml=", 0) == 0)
      xml_override = a.substr(6);
    else if (a.size() > 5 && a.substr(a.size() - 5) == ".yaml")
      cfg_path = a;
    else if (a.size() > 3 && a.substr(a.size() - 3) == ".pt")
      ckpt_path = a;
    else if (a.size() > 4 && a.substr(a.size() - 4) == ".xml")
      xml_override = a;
  }

  // 최소 인자 체크
  if (ckpt_path.empty()) {
    std::cerr
        << "Usage: " << argv[0]
        << " <ckpt.pt> [config.yaml] [--xml=/path/model.xml] [--cpu|--cuda]\n";
    return 1;
  }

  // config 자동 탐색 (없으면 ckpt 디렉터리에서 찾기)
  if (cfg_path.empty()) {
    try {
      fs::path p(ckpt_path);
      fs::path cand = p.parent_path() / "config_dump.yaml";
      if (fs::exists(cand)) {
        cfg_path = cand.string();
        std::cout << "[viewer] Using auto config: " << cfg_path << "\n";
      }
    } catch (...) {
    }
  }

  // config 로드
  AppConfig app;
  if (!cfg_path.empty()) {
    if (!load_config_yaml(cfg_path, app)) {
      std::cerr
          << "[viewer] WARN: failed to load config YAML, using defaults.\n";
    }
  }

  // xml 오버라이드 적용
  if (!xml_override.empty())
    app.env.xml_path = xml_override;

  // MuJoCo 모델 파일 존재 확인
  if (!fs::exists(app.env.xml_path)) {
    std::cerr << "[viewer] ERROR: MJCF not found: " << app.env.xml_path << "\n";
    std::cerr
        << "  Provide a valid config.yaml or use --xml=/abs/path/model.xml\n";
    return 1;
  }

  // device 결정
  bool use_cuda = (force_cuda || (!force_cpu && torch::cuda::is_available()));
  torch::Device device = use_cuda ? torch::kCUDA : torch::kCPU;
  std::cout << "[viewer] Device: " << (use_cuda ? "CUDA" : "CPU") << "\n";

  MjEnv env(app.env);
  ActorCritic net(env.obs_dim(), env.act_dim(), 128);
  try {
    torch::load(net, ckpt_path);
    net->to(device); // 수정된 부분
    net->eval();
  } catch (const std::exception &e) {
    std::cerr << "[viewer] Error loading model: " << e.what() << "\n";
    return 1;
  }

  NormStats norm;
  try {
    fs::path p(ckpt_path);
    fs::path normp = p.parent_path() / "obs_norm.yaml";
    norm = load_norm_stats_if_any(normp.string(), env.obs_dim());
  } catch (...) {
  }

  if (!glfwInit())
    return -1;
  GLFWwindow *window =
      glfwCreateWindow(1280, 900, "MuJoCo Viewer (RL)", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  mjv_defaultCamera(&cam);
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);
  opt.flags[mjVIS_CONTACTPOINT] = 0;
  opt.flags[mjVIS_CONTACTFORCE] = 0;

  mjModel *m = env.model();
  mjv_makeScene(m, &scn, 2000);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  int base_id = mj_name2id(m, mjOBJ_BODY, "base");
  if (base_id >= 0) {
    cam.type = mjCAMERA_TRACKING;
    cam.trackbodyid = base_id;
    cam.distance = 1.2;
    cam.azimuth = 120.0;
    cam.elevation = -10.0;
  }

  ViewerUser user;
  user.env = &env;
  user.net = net; // ModuleHolder 대입
  user.vx_cmd = app.env.target_vx;
  user.obs = env.reset();

  glfwSetWindowUserPointer(window, &user);
  glfwSetKeyCallback(window, keyboard);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetScrollCallback(window, scroll);

  torch::NoGradGuard no_grad;
  auto t_prev = std::chrono::steady_clock::now();
  const double sim_dt = m->opt.timestep * std::max(1, app.env.frame_skip);

  while (!glfwWindowShouldClose(window)) {
    if (!user.paused) {
      auto x = normalize_obs(user.obs, norm).to(device);
      torch::Tensor action;
      if (user.stochastic) {
        auto out = user.net->act(x.unsqueeze(0));
        action = std::get<0>(out).squeeze(0).to(torch::kCPU);
      } else {
        auto fwd = user.net->forward(x.unsqueeze(0));
        auto mean = std::get<0>(fwd).squeeze(0);
        action = mean.to(torch::kCPU);
      }
      auto sr = env.step(action);
      user.obs = sr.obs;
      user.last_reward = sr.reward;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    mjrRect viewport = {0, 0, width, height};

    mjv_updateScene(m, env.data(), &opt, nullptr, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    auto t_now = std::chrono::steady_clock::now();
    double dt_wall = std::chrono::duration<double>(t_now - t_prev).count();
    t_prev = t_now;
    double alpha = 0.1;
    user.fps =
        (1.0 - alpha) * user.fps + alpha * (1.0 / std::max(1e-6, dt_wall));

    char overlayL[512], overlayR[512];
    double vx = body_vx_in_body(m, env.data());
    snprintf(overlayL, sizeof(overlayL),
             "Cmd vx: %.2f  |  Body vx: %.2f\nReward: %.3f\nMode: %s  "
             "Contacts: %s  Sync: %s\nFPS: %.1f",
             user.vx_cmd, vx, user.last_reward,
             user.stochastic ? "Stochastic(S)" : "Deterministic(D)",
             user.show_contacts ? "On(C)" : "Off(C)",
             user.sync_realtime ? "On(T)" : "Off(T)", user.fps);
    snprintf(overlayR, sizeof(overlayR),
             "[Space] Pause/Resume\n[R] Reset   [ [ / ] ] vx_cmd +/-\n[C] "
             "Contacts  [S]/[D] Action mode\n[T] Realtime sync  [F12] "
             "Screenshot\n[Mouse] Rotate/Move/Zoom, [Esc] Quit");
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, overlayL, nullptr,
                &con);
    mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMRIGHT, viewport, overlayR, nullptr,
                &con);

    glfwSwapBuffers(window);
    glfwPollEvents();

    if (user.sync_realtime && !user.paused) {
      int sleep_ms = (int)std::round(sim_dt * 1000.0);
      if (sleep_ms > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    } else if (user.paused) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  mjr_freeContext(&con);
  mjv_freeScene(&scn);
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
