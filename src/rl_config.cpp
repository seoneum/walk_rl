#include "rl_config.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <yaml-cpp/yaml.h>
namespace fs = std::filesystem;

static inline void clamp_friction(EnvConfig &e) {
  if (e.friction_min > e.friction_max)
    std::swap(e.friction_min, e.friction_max);
}

bool load_config_yaml(const std::string &yaml_path, AppConfig &out) {
  try {
    YAML::Node y = YAML::LoadFile(yaml_path);

    // env
    if (y["env"]) {
      auto e = y["env"];
      if (e["xml_path"])
        out.env.xml_path = e["xml_path"].as<std::string>();
      if (e["foot_geoms"])
        out.env.foot_geom_names =
            e["foot_geoms"].as<std::vector<std::string>>();
      if (e["action_scale"])
        out.env.action_scale = e["action_scale"].as<double>();
      if (e["frame_skip"])
        out.env.frame_skip = e["frame_skip"].as<int>();
      if (e["init_height"])
        out.env.init_height = e["init_height"].as<double>();
      if (e["target_vx"])
        out.env.target_vx = e["target_vx"].as<double>();
      if (e["max_episode_sec"])
        out.env.max_episode_sec = e["max_episode_sec"].as<double>();

      if (e["use_bezier_ref"])
        out.env.use_bezier_ref = e["use_bezier_ref"].as<bool>();
      if (e["use_residual"])
        out.env.use_residual = e["use_residual"].as<bool>();
      if (e["residual_scale"])
        out.env.residual_scale = e["residual_scale"].as<double>();
      if (e["leg_L1"])
        out.env.leg_L1 = e["leg_L1"].as<double>();
      if (e["leg_L2"])
        out.env.leg_L2 = e["leg_L2"].as<double>();
      if (e["step_L_span"])
        out.env.step_L_span = e["step_L_span"].as<double>();
      if (e["step_base_h"])
        out.env.step_base_h = e["step_base_h"].as<double>();
      if (e["step_clearance"])
        out.env.step_clearance = e["step_clearance"].as<double>();

      // command conditioning
      if (e["include_cmd"])
        out.env.include_cmd = e["include_cmd"].as<bool>();
      if (e["sample_cmd_per_reset"])
        out.env.sample_cmd_per_reset = e["sample_cmd_per_reset"].as<bool>();
      if (e["cmd_vx_min"])
        out.env.cmd_vx_min = e["cmd_vx_min"].as<double>();
      if (e["cmd_vx_max"])
        out.env.cmd_vx_max = e["cmd_vx_max"].as<double>();

      // domain randomization
      if (e["mass_noise"])
        out.env.mass_noise = e["mass_noise"].as<double>();
      if (e["friction_min"])
        out.env.friction_min = e["friction_min"].as<double>();
      if (e["friction_max"])
        out.env.friction_max = e["friction_max"].as<double>();
      if (e["joint_damping_scale_min"])
        out.env.joint_damping_scale_min =
            e["joint_damping_scale_min"].as<double>();
      if (e["joint_damping_scale_max"])
        out.env.joint_damping_scale_max =
            e["joint_damping_scale_max"].as<double>();
      if (e["sensor_gyro_noise_std"])
        out.env.sensor_gyro_noise_std = e["sensor_gyro_noise_std"].as<double>();
      if (e["sensor_linvel_noise_std"])
        out.env.sensor_linvel_noise_std =
            e["sensor_linvel_noise_std"].as<double>();
      if (e["use_contact_slip_penalty"])
        out.env.use_contact_slip_penalty =
            e["use_contact_slip_penalty"].as<bool>();
      if (e["use_clip_actions"])
        out.env.use_clip_actions = e["use_clip_actions"].as<bool>();
      if (e["seed"])
        out.env.seed = e["seed"].as<unsigned int>();

      // reward weights
      auto rw = e;
      if (rw["w_speed"])
        out.env.w_speed = rw["w_speed"].as<double>();
      if (rw["w_posture"])
        out.env.w_posture = rw["w_posture"].as<double>();
      if (rw["w_yaw"])
        out.env.w_yaw = rw["w_yaw"].as<double>();
      if (rw["w_action"])
        out.env.w_action = rw["w_action"].as<double>();
      if (rw["w_slip"])
        out.env.w_slip = rw["w_slip"].as<double>();
      if (rw["w_alive"])
        out.env.w_alive = rw["w_alive"].as<double>();
      if (rw["w_angvel"])
        out.env.w_angvel = rw["w_angvel"].as<double>();
      if (rw["w_act_rate"])
        out.env.w_act_rate = rw["w_act_rate"].as<double>();

      clamp_friction(out.env);
    }

    // train
    if (y["train"]) {
      auto t = y["train"];
      if (t["num_envs"])
        out.train.num_envs = t["num_envs"].as<int>();
      if (t["total_updates"])
        out.train.total_updates = t["total_updates"].as<int>();
      if (t["algo"])
        out.train.algo = t["algo"].as<std::string>();
      if (t["tag"])
        out.train.tag = t["tag"].as<std::string>();
      if (t["curriculum"])
        out.train.curriculum = t["curriculum"].as<bool>();
      if (t["curriculum_every"])
        out.train.curriculum_every = t["curriculum_every"].as<int>();
      if (t["curriculum_dvx"])
        out.train.curriculum_dvx = t["curriculum_dvx"].as<double>();
      if (t["curriculum_vx_max"])
        out.train.curriculum_vx_max = t["curriculum_vx_max"].as<double>();
    }

    // ppo (옵션)
    if (y["ppo"]) {
      auto p = y["ppo"];
      if (p["lr"])
        out.ppo.lr = p["lr"].as<double>();
      if (p["gamma"])
        out.ppo.gamma = p["gamma"].as<double>();
      if (p["gae_lambda"])
        out.ppo.gae_lambda = p["gae_lambda"].as<double>();
      if (p["clip_range"])
        out.ppo.clip_range = p["clip_range"].as<double>();
      if (p["vf_coef"])
        out.ppo.vf_coef = p["vf_coef"].as<double>();
      if (p["ent_coef"])
        out.ppo.ent_coef = p["ent_coef"].as<double>();
      if (p["hidden"])
        out.ppo.hidden = p["hidden"].as<int>();
      if (p["rollout_horizon"])
        out.ppo.rollout_horizon = p["rollout_horizon"].as<int>();
      if (p["num_epochs"])
        out.ppo.num_epochs = p["num_epochs"].as<int>();
      if (p["minibatch_size"])
        out.ppo.minibatch_size = p["minibatch_size"].as<int>();
      if (p["use_cuda"])
        out.ppo.use_cuda = p["use_cuda"].as<bool>();
      if (p["init_ckpt"])
        out.ppo.init_ckpt = p["init_ckpt"].as<std::string>();
      if (p["resume_optimizer"])
        out.ppo.resume_optimizer = p["resume_optimizer"].as<bool>();
      if (p["target_kl"])
        out.ppo.target_kl = p["target_kl"].as<double>();
      if (p["bc_coef"])
        out.ppo.bc_coef = p["bc_coef"].as<double>();
      if (p["bc_batch"])
        out.ppo.bc_batch = p["bc_batch"].as<int>();
    }

    // xml_path를 config.yaml 기준으로 절대경로화
    fs::path base = fs::absolute(fs::path(yaml_path)).parent_path();
    fs::path xmlp(out.env.xml_path);
    if (!xmlp.is_absolute()) {
      xmlp = (base / xmlp).lexically_normal();
      out.env.xml_path = fs::weakly_canonical(xmlp).string();
    }
  } catch (...) {
    return false;
  }
  return true;
}

std::string make_run_dir(const std::string &root, const std::string &tag) {
  fs::create_directories(root);
  int idx = 1;
  for (auto &p : fs::directory_iterator(root)) {
    if (!p.is_directory())
      continue;
    auto name = p.path().filename().string();
    try {
      int n = std::stoi(name.substr(0, 4));
      if (n >= idx)
        idx = n + 1;
    } catch (...) {
    }
  }
  char buf[64];
  snprintf(buf, sizeof(buf), "%04d_%s", idx, tag.c_str());
  fs::path d = fs::path(root) / buf;
  fs::create_directories(d);
  return d.string();
}

void dump_config_yaml(const AppConfig &cfg, const std::string &path) {
  std::ofstream f(path);
  f << "env:\n";
  f << "  xml_path: " << cfg.env.xml_path << "\n";
  f << "  foot_geoms: [";
  for (size_t i = 0; i < cfg.env.foot_geom_names.size(); ++i) {
    f << cfg.env.foot_geom_names[i];
    if (i + 1 < cfg.env.foot_geom_names.size())
      f << ", ";
  }
  f << "]\n";
  f << "  action_scale: " << cfg.env.action_scale << "\n";
  f << "  frame_skip: " << cfg.env.frame_skip << "\n";
  f << "  init_height: " << cfg.env.init_height << "\n";
  f << "  target_vx: " << cfg.env.target_vx << "\n";
  f << "  max_episode_sec: " << cfg.env.max_episode_sec << "\n";

  f << "  # command conditioning\n";
  f << "  include_cmd: " << (cfg.env.include_cmd ? "true" : "false") << "\n";
  f << "  sample_cmd_per_reset: "
    << (cfg.env.sample_cmd_per_reset ? "true" : "false") << "\n";
  f << "  cmd_vx_min: " << cfg.env.cmd_vx_min << "\n";
  f << "  cmd_vx_max: " << cfg.env.cmd_vx_max << "\n";

  f << "  # domain randomization\n";
  f << "  mass_noise: " << cfg.env.mass_noise << "\n";
  f << "  friction_min: " << cfg.env.friction_min << "\n";
  f << "  friction_max: " << cfg.env.friction_max << "\n";
  f << "  joint_damping_scale_min: " << cfg.env.joint_damping_scale_min << "\n";
  f << "  joint_damping_scale_max: " << cfg.env.joint_damping_scale_max << "\n";
  f << "  sensor_gyro_noise_std: " << cfg.env.sensor_gyro_noise_std << "\n";
  f << "  sensor_linvel_noise_std: " << cfg.env.sensor_linvel_noise_std << "\n";
  f << "  use_contact_slip_penalty: "
    << (cfg.env.use_contact_slip_penalty ? "true" : "false") << "\n";
  f << "  use_clip_actions: " << (cfg.env.use_clip_actions ? "true" : "false")
    << "\n";
  f << "  seed: " << cfg.env.seed << "\n";

  f << "  # reward weights\n";
  f << "  w_speed: " << cfg.env.w_speed << "\n";
  f << "  w_posture: " << cfg.env.w_posture << "\n";
  f << "  w_yaw: " << cfg.env.w_yaw << "\n";
  f << "  w_action: " << cfg.env.w_action << "\n";
  f << "  w_slip: " << cfg.env.w_slip << "\n";
  f << "  w_alive: " << cfg.env.w_alive << "\n";
  f << "  w_angvel: " << cfg.env.w_angvel << "\n";
  f << "  w_act_rate: " << cfg.env.w_act_rate << "\n";

  f << "train:\n";
  f << "  num_envs: " << cfg.train.num_envs << "\n";
  f << "  total_updates: " << cfg.train.total_updates << "\n";
  f << "  algo: " << cfg.train.algo << "\n";
  f << "  tag: " << cfg.train.tag << "\n";
  f << "  curriculum: " << (cfg.train.curriculum ? "true" : "false") << "\n";
  f << "  curriculum_every: " << cfg.train.curriculum_every << "\n";
  f << "  curriculum_dvx: " << cfg.train.curriculum_dvx << "\n";
  f << "  curriculum_vx_max: " << cfg.train.curriculum_vx_max << "\n";

  f << "ppo:\n";
  f << "  lr: " << cfg.ppo.lr << "\n";
  f << "  gamma: " << cfg.ppo.gamma << "\n";
  f << "  gae_lambda: " << cfg.ppo.gae_lambda << "\n";
  f << "  clip_range: " << cfg.ppo.clip_range << "\n";
  f << "  vf_coef: " << cfg.ppo.vf_coef << "\n";
  f << "  ent_coef: " << cfg.ppo.ent_coef << "\n";
  f << "  hidden: " << cfg.ppo.hidden << "\n";
  f << "  rollout_horizon: " << cfg.ppo.rollout_horizon << "\n";
  f << "  num_epochs: " << cfg.ppo.num_epochs << "\n";
  f << "  minibatch_size: " << cfg.ppo.minibatch_size << "\n";
  f << "  use_cuda: " << (cfg.ppo.use_cuda ? "true" : "false") << "\n";

  f.close();
}
