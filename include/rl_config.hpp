#pragma once
#include <string>
#include <vector>

struct EnvConfig {
  std::string xml_path = "models/quad.xml";
  std::vector<std::string> foot_geom_names = {"foot_fl", "foot_fr", "foot_rl",
                                              "foot_rr"};

  // control / timing
  double action_scale = 0.30;
  int frame_skip = 10;
  double init_height = 0.28;
  double target_vx = 0.4;
  double max_episode_sec = 10.0;

  // command conditioning
  bool include_cmd = true;           // 관측에 vx_cmd 포함
  bool sample_cmd_per_reset = false; // reset마다 vx_cmd 샘플링
  double cmd_vx_min = 0.4;
  double cmd_vx_max = 0.4;

  // domain randomization
  double mass_noise = 0.05;
  double friction_min = 0.8, friction_max = 1.0;
  double joint_damping_scale_min = 0.9, joint_damping_scale_max = 1.1;
  double sensor_gyro_noise_std = 0.01;
  double sensor_linvel_noise_std = 0.02;

  bool use_contact_slip_penalty = true;
  bool use_clip_actions = true;
  unsigned int seed = 1234;

  // reward weights
  double w_speed = 2.0;
  double w_posture = 0.2;    // roll^2 + pitch^2 penalty
  double w_yaw = 0.05;       // |yaw| penalty
  double w_action = 0.001;   // action^2 penalty
  double w_slip = 0.02;      // foot slip penalty
  double w_alive = 0.05;     // alive bonus
  double w_angvel = 0.01;    // roll/pitch angular velocity penalty
  double w_act_rate = 0.005; // (Δaction)^2 penalty

  // Locomotion ref + residual
  bool use_bezier_ref = true;
  bool use_residual = true;
  double residual_scale = 0.25;
  double leg_L1 = 0.10;
  double leg_L2 = 0.10;
  double step_L_span = 0.12;
  double step_base_h = 0.18;
  double step_clearance = 0.03;
};

struct PPOConfig {
  int obs_dim = 0, act_dim = 0, hidden = 128;
  double lr = 2e-4, gamma = 0.99, gae_lambda = 0.95, clip_range = 0.10,
         vf_coef = 0.5, ent_coef = 0.0003;
  int rollout_horizon = 256, num_epochs = 5, minibatch_size = 1024;
  bool use_cuda = true;

  // 추가된 필드
  std::string init_ckpt = "";    // 시작 시 로드할 ckpt 경로 (비우면 무시)
  bool resume_optimizer = false; // 옵티마이저 상태도 재개할지
  double target_kl = 0.03;       // KL 타깃, 초과 시 epoch 조기 중단
  double bc_coef = 0.05;         // BC 정규화 계수 (0이면 비활성)
  int bc_batch = 512;            // BC 배치 크기
};

struct TrainConfig {
  int num_envs = 8;
  int total_updates = 2000;
  std::string algo = "PPO"; // or "ARS"
  std::string tag = "mj_const";
  bool curriculum = false;
  int curriculum_every = 100;
  double curriculum_dvx = 0.1;
  double curriculum_vx_max = 1.5;
};

struct Paths {
  std::string run_dir;
  std::string log_csv;
  std::string config_yaml_dump;
};

struct AppConfig {
  EnvConfig env;
  PPOConfig ppo;
  TrainConfig train;
  Paths paths;
};

// API
bool load_config_yaml(const std::string &yaml_path, AppConfig &out);
std::string make_run_dir(const std::string &root = "runs",
                         const std::string &tag = "mj_const");
void dump_config_yaml(const AppConfig &cfg, const std::string &path);
