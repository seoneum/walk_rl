
#include "mj_env.hpp"
#include "gait_ref.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static inline double clamp(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

// 전진 방향: 월드 x (앞이면 +1, 반대면 -1로 바꾸세요)
static constexpr int FWD_SIGN = 1;

static inline std::string lower_str(const char *cstr) {
  if (!cstr)
    return std::string();
  std::string s(cstr);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  return s;
}

MjEnv::MjEnv(const EnvConfig &cfg) : cfg_(cfg), rng_(cfg.seed) {
  char error[1024] = {0};
  m_ = mj_loadXML(cfg_.xml_path.c_str(), nullptr, error, sizeof(error));
  if (!m_) {
    std::cerr << "MuJoCo load error: " << error << std::endl;
    throw std::runtime_error("loadXML");
  }
  d_ = mj_makeData(m_);
  if (!d_)
    throw std::runtime_error("makeData");

  build_mappings_();
  find_feet_();

  // 강제 중력 오버라이드(디버그용)
  m_->opt.gravity[0] = 0.0;
  m_->opt.gravity[1] = 0.0;
  m_->opt.gravity[2] = -9.81;
  std::cerr << "[phys] gravity = " << m_->opt.gravity[0] << " "
            << m_->opt.gravity[1] << " " << m_->opt.gravity[2] << std::endl;

  act_dim_ = (int)act_jnt_ids_.size();
  obs_dim_ = 2 + 1 + 2 + (cfg_.include_cmd ? 1 : 0) + act_dim_ + act_dim_ +
             act_dim_ + (int)foot_geom_ids_.size();

  prev_action_ = torch::zeros({act_dim_}, torch::kFloat32);
  max_steps_per_ep_ = (int)std::round(cfg_.max_episode_sec /
                                      (m_->opt.timestep * cfg_.frame_skip));

  mj_resetData(m_, d_);
  mj_forward(m_, d_);
  randomize_physics_();

  // 참조 보행 초기화
  ref_.ik = IK2Link{cfg_.leg_L1, cfg_.leg_L2};
  ref_.L_span = cfg_.step_L_span;
  ref_.base_h = cfg_.step_base_h;
  ref_.clearance = cfg_.step_clearance;
  ref_.set_speed(cfg_.target_vx);

  auto find_act_for_joint = [&](int joint_id) -> int {
    if (joint_id < 0)
      return -1;
    for (int a = 0; a < m_->nu; ++a)
      if (m_->actuator_trnid[2 * a] == joint_id)
        return a;
    return -1;
  };
  auto find_joint_by_tokens = [&](const std::vector<std::string> &toks) -> int {
    for (int j = 0; j < m_->njnt; ++j) {
      std::string name = lower_str(mj_id2name(m_, mjOBJ_JOINT, j));
      if (name.empty())
        continue;
      bool ok = true;
      for (auto &t : toks)
        if (name.find(t) == std::string::npos) {
          ok = false;
          break;
        }
      if (ok)
        return j;
    }
    return -1;
  };
  auto find_act_by_tokens = [&](const std::vector<std::string> &toks) -> int {
    int jid = find_joint_by_tokens(toks);
    return find_act_for_joint(jid);
  };
  auto pick_first_act =
      [&](const std::vector<std::vector<std::string>> &sets) -> int {
    for (auto &s : sets) {
      int a = find_act_by_tokens(s);
      if (a >= 0)
        return a;
    }
    return -1;
  };

  // 이름 토큰 매핑
  ref_.fl.hip =
      pick_first_act({{"front", "left", "hip"}, {"front", "left", "shoulder"}});
  ref_.fl.knee = pick_first_act({{"front", "left", "knee"}});
  ref_.fr.hip = pick_first_act(
      {{"front", "right", "hip"}, {"front", "right", "shoulder"}});
  ref_.fr.knee = pick_first_act({{"front", "right", "knee"}});
  ref_.rl.hip =
      pick_first_act({{"back", "left", "hip"}, {"rear", "left", "hip"}});
  ref_.rl.knee =
      pick_first_act({{"back", "left", "knee"}, {"rear", "left", "knee"}});
  ref_.rr.hip =
      pick_first_act({{"back", "right", "hip"}, {"rear", "right", "hip"}});
  ref_.rr.knee =
      pick_first_act({{"back", "right", "knee"}, {"rear", "right", "knee"}});

  // 정확명 폴백
  auto map_exact = [&](const char *jname) -> int {
    int jid = mj_name2id(m_, mjOBJ_JOINT, jname);
    return (jid >= 0) ? find_act_for_joint(jid) : -1;
  };
  if (ref_.fl.hip < 0)
    ref_.fl.hip = map_exact("hip_front_left");
  if (ref_.fl.knee < 0)
    ref_.fl.knee = map_exact("knee_front_left");
  if (ref_.fr.hip < 0)
    ref_.fr.hip = map_exact("hip_front_right");
  if (ref_.fr.knee < 0)
    ref_.fr.knee = map_exact("knee_front_right");
  if (ref_.rl.hip < 0)
    ref_.rl.hip = map_exact("hip_back_left");
  if (ref_.rl.knee < 0)
    ref_.rl.knee = map_exact("knee_back_left");
  if (ref_.rr.hip < 0)
    ref_.rr.hip = map_exact("hip_back_right");
  if (ref_.rr.knee < 0)
    ref_.rr.knee = map_exact("knee_back_right");

  auto log_leg = [&](const char *nm, const LegIndex &L) {
    std::cerr << "[ref] " << nm << " hip=" << L.hip << " knee=" << L.knee
              << "\n";
  };
  log_leg("FL", ref_.fl);
  log_leg("FR", ref_.fr);
  log_leg("RL", ref_.rl);
  log_leg("RR", ref_.rr);
}

MjEnv::~MjEnv() {
  if (d_)
    mj_deleteData(d_);
  if (m_)
    mj_deleteModel(m_);
}

void MjEnv::build_mappings_() {
  int nu = m_->nu;
  act_jnt_ids_.resize(nu);
  act_jnt_qposadr_.resize(nu);
  act_jnt_qveladr_.resize(nu);
  for (int i = 0; i < nu; ++i) {
    int j = m_->actuator_trnid[2 * i];
    act_jnt_ids_[i] = j;
    act_jnt_qposadr_[i] = m_->jnt_qposadr[j];
    act_jnt_qveladr_[i] = m_->jnt_dofadr[j];
  }
}

void MjEnv::find_feet_() {
  foot_geom_ids_.clear();
  for (auto &n : cfg_.foot_geom_names) {
    int id = mj_name2id(m_, mjOBJ_GEOM, n.c_str());
    if (id >= 0)
      foot_geom_ids_.push_back(id);
  }
  if (foot_geom_ids_.empty())
    std::cerr << "[WARN] No foot geoms found.\n";
}

void MjEnv::randomize_physics_() {
  double mass_scale = 1.0 + cfg_.mass_noise * (2.0 * uni01_(rng_) - 1.0);
  for (int i = 0; i < m_->nbody; ++i) {
    if (i == 0)
      continue;
    m_->body_mass[i] *= mass_scale;
    m_->body_inertia[3 * i + 0] *= mass_scale;
    m_->body_inertia[3 * i + 1] *= mass_scale;
    m_->body_inertia[3 * i + 2] *= mass_scale;
  }
  double damp_scale =
      cfg_.joint_damping_scale_min +
      (cfg_.joint_damping_scale_max - cfg_.joint_damping_scale_min) *
          uni01_(rng_);
  for (int i = 0; i < m_->nv; ++i)
    m_->dof_damping[i] *= damp_scale;

  for (int i = 0; i < m_->ngeom; ++i) {
    double mu = cfg_.friction_min +
                (cfg_.friction_max - cfg_.friction_min) * uni01_(rng_);
    m_->geom_friction[3 * i + 0] = mu;
    m_->geom_friction[3 * i + 1] = 0.005;
    m_->geom_friction[3 * i + 2] = 0.0001;
  }
}

// 발끝이 바닥 아래면 몸체 z를 들어올려 여유 확보
void MjEnv::lift_body_clearance_(double margin) {
  for (int it = 0; it < 5; ++it) {
    mj_forward(m_, d_);
    double min_z = 1e9;
    for (int gid : foot_geom_ids_) {
      const double *p = &d_->geom_xpos[3 * gid];
      if (p[2] < min_z)
        min_z = p[2];
    }
    if (min_z >= margin)
      break;
    d_->qpos[2] += (margin - min_z); // root z 올리기
  }
  mj_forward(m_, d_);
}

// 초기 정착(서보로 현재각 고정해 몇 스텝)
void MjEnv::settle_(int steps) {
  for (int i = 0; i < act_dim_; ++i)
    d_->ctrl[i] = d_->qpos[act_jnt_qposadr_[i]];
  for (int k = 0; k < steps; ++k)
    mj_step(m_, d_);
}

void MjEnv::randomize_state_() {
  mj_resetData(m_, d_);

  d_->qpos[0] = 0.0;
  d_->qpos[1] = 0.0;
  d_->qpos[2] = cfg_.init_height;
  d_->qpos[3] = 1.0;
  d_->qpos[4] = 0.0;
  d_->qpos[5] = 0.0;
  d_->qpos[6] = 0.0;

  for (int a = 0; a < act_dim_; ++a) {
    int qp = act_jnt_qposadr_[a];
    int j = act_jnt_ids_[a];
    const char *jname = mj_id2name(m_, mjOBJ_JOINT, j);
    double off = 0.0;
    if (jname) {
      std::string s = lower_str(jname);
      if (s.find("hip") != std::string::npos)
        off = 0.05;
      if (s.find("knee") != std::string::npos)
        off = -0.9; // 더 쭈그린 자세
    }
    d_->qpos[qp] = off;
  }

  mj_forward(m_, d_);
  // 발끝이 바닥 아래면 들어올리기(1cm 여유)
  lift_body_clearance_(0.01);
  // 초기 정착
  settle_(20);

  prev_action_.zero_();
  last_action_rate2_ = 0.0;
  sim_time_ = 0.0;
  step_count_ = 0;
  ref_.t = 0.0;
}

torch::Tensor MjEnv::reset() {
  randomize_state_();
  return build_obs_();
}

void MjEnv::quat_to_rpy_(const double *q, double &roll, double &pitch,
                         double &yaw) const {
  double w = q[0], x = q[1], y = q[2], z = q[3];
  double sinr = 2.0 * (w * x + y * z), cosr = 1.0 - 2.0 * (x * x + y * y);
  roll = std::atan2(sinr, cosr);
  double sinp = 2.0 * (w * y - z * x);
  pitch =
      std::abs(sinp) >= 1 ? std::copysign(M_PI / 2.0, sinp) : std::asin(sinp);
  double siny = 2.0 * (w * z + x * y), cosy = 1.0 - 2.0 * (y * y + z * z);
  yaw = std::atan2(siny, cosy);
}

torch::Tensor MjEnv::build_obs_() {
  const double *q = &d_->qpos[3];
  double roll, pitch, yaw;
  quat_to_rpy_(q, roll, pitch, yaw);
  double Rm[9];
  mju_quat2Mat(Rm, q);
  double v_world[3] = {d_->qvel[0], d_->qvel[1], d_->qvel[2]};
  double v_body[3] = {
      Rm[0] * v_world[0] + Rm[3] * v_world[1] + Rm[6] * v_world[2],
      Rm[1] * v_world[0] + Rm[4] * v_world[1] + Rm[7] * v_world[2],
      Rm[2] * v_world[0] + Rm[5] * v_world[1] + Rm[8] * v_world[2]};
  double yaw_rate = d_->qvel[5];

  std::vector<float> o;
  o.reserve(obs_dim_);
  o.push_back((float)roll);
  o.push_back((float)pitch);
  o.push_back((float)(yaw_rate + cfg_.sensor_gyro_noise_std * n01_(rng_)));
  o.push_back((float)(v_body[0] + cfg_.sensor_linvel_noise_std * n01_(rng_)));
  o.push_back((float)(v_body[1] + cfg_.sensor_linvel_noise_std * n01_(rng_)));
  if (cfg_.include_cmd)
    o.push_back((float)cfg_.target_vx);
  for (int i = 0; i < act_dim_; ++i)
    o.push_back((float)d_->qpos[act_jnt_qposadr_[i]]);
  for (int i = 0; i < act_dim_; ++i)
    o.push_back((float)d_->qvel[act_jnt_qveladr_[i]]);
  for (int i = 0; i < act_dim_; ++i)
    o.push_back(prev_action_[i].item<float>());

  std::vector<char> footc(foot_geom_ids_.size(), 0);
  for (int c = 0; c < d_->ncon; ++c) {
    auto &con = d_->contact[c];
    for (size_t k = 0; k < foot_geom_ids_.size(); ++k)
      if (con.geom1 == foot_geom_ids_[k] || con.geom2 == foot_geom_ids_[k]) {
        footc[k] = 1;
        break;
      }
  }
  for (auto fc : footc)
    o.push_back((float)fc);
  return torch::from_blob(o.data(), {(int)o.size()}, torch::kFloat32).clone();
}

void MjEnv::apply_action_position_servo_(const torch::Tensor &act) {
  std::vector<double> qref(act_dim_, 0.0);
  bool any_missing = (ref_.fl.hip < 0 || ref_.fl.knee < 0 || ref_.fr.hip < 0 ||
                      ref_.fr.knee < 0 || ref_.rl.hip < 0 || ref_.rl.knee < 0 ||
                      ref_.rr.hip < 0 || ref_.rr.knee < 0);
  if (cfg_.use_bezier_ref && !any_missing) {
    ref_.set_speed(cfg_.target_vx);
    ref_.build_qref(qref, act_jnt_qposadr_, act_jnt_ids_, m_, d_);
  }

  auto a = act.to(torch::kCPU).contiguous();
  const float *p = a.data_ptr<float>();
  const double alpha = 0.2;

  for (int i = 0; i < act_dim_; ++i) {
    int j = act_jnt_ids_[i];
    double sign = 1.0;
    if (const char *jname = mj_id2name(m_, mjOBJ_JOINT, j)) {
      std::string s = lower_str(jname);
      if (s.find("knee") != std::string::npos)
        sign = -1.0; // 무릎 부호 반전
    }

    double base = (cfg_.use_bezier_ref && !any_missing)
                      ? sign * qref[i]
                      : d_->qpos[act_jnt_qposadr_[i]];
    double target =
        base + (cfg_.use_residual ? cfg_.residual_scale * (double)p[i] : 0.0);

    if (m_->jnt_limited[j]) {
      double lo = m_->jnt_range[2 * j], hi = m_->jnt_range[2 * j + 1];
      target = clamp(target, lo, hi);
    }

    double q_now = d_->qpos[act_jnt_qposadr_[i]];
    double soft_target = q_now + alpha * (target - q_now);
    d_->ctrl[i] = soft_target;
  }
}

bool MjEnv::is_fallen_() const {
  double r, p, y;
  quat_to_rpy_(&d_->qpos[3], r, p, y);
  if (std::abs(r) > 0.8 || std::abs(p) > 0.8)
    return true;
  if (d_->qpos[2] < 0.12)
    return true;
  return false;
}

double MjEnv::compute_reward_and_check_done_(bool &done_flag) {
  done_flag = false;

  // 전진 보상: 월드 x속도 (부호는 FWD_SIGN 한 줄로 조정)
  double vwx = d_->qvel[0];
  double fwd = FWD_SIGN * vwx;
  double vel_err = (fwd - cfg_.target_vx);
  double r_speed = cfg_.w_speed * std::exp(-(vel_err * vel_err) / 0.04);

  double roll, pitch, yaw;
  quat_to_rpy_(&d_->qpos[3], roll, pitch, yaw);
  double r_posture = -cfg_.w_posture * (roll * roll + pitch * pitch);
  double r_yaw = -cfg_.w_yaw * std::abs(yaw);

  double a2 = prev_action_.pow(2).mean().item<double>();
  double r_action = -cfg_.w_action * a2;

  double r_slip = 0.0;
  if (cfg_.use_contact_slip_penalty && !foot_geom_ids_.empty()) {
    for (int gid : foot_geom_ids_) {
      bool in_contact = false;
      for (int c = 0; c < d_->ncon; ++c) {
        const mjContact &con = d_->contact[c];
        if (con.geom1 == gid || con.geom2 == gid) {
          in_contact = true;
          break;
        }
      }
      if (in_contact) {
        double res[6] = {0};
        mj_objectVelocity(m_, d_, mjOBJ_GEOM, gid, res, 0);
        double vxy = std::sqrt(res[0] * res[0] + res[1] * res[1]);
        r_slip -= cfg_.w_slip * vxy;
      }
    }
  }

  double r_angvel =
      -cfg_.w_angvel * (d_->qvel[3] * d_->qvel[3] + d_->qvel[4] * d_->qvel[4]);
  double r_actrate = -cfg_.w_act_rate * last_action_rate2_;
  double r_alive = cfg_.w_alive;

  double reward = r_speed + r_posture + r_yaw + r_action + r_slip + r_angvel +
                  r_actrate + r_alive;
  if (is_fallen_() || step_count_ >= max_steps_per_ep_)
    done_flag = true;
  return reward;
}

StepResult MjEnv::step(const torch::Tensor &action) {
  torch::Tensor a =
      cfg_.use_clip_actions ? torch::clamp(action, -1.0, 1.0) : action;

  auto delta = (a - prev_action_).to(torch::kFloat32);
  last_action_rate2_ = delta.pow(2).mean().item<double>();

  apply_action_position_servo_(a);
  prev_action_ = a.detach().clone();

  int fs = std::max(1, cfg_.frame_skip);
  for (int i = 0; i < fs; ++i)
    mj_step(m_, d_);
  sim_time_ += m_->opt.timestep * fs;
  step_count_++;

  ref_.step(m_->opt.timestep * fs);

  bool done = false;
  double r = compute_reward_and_check_done_(done);
  auto o = build_obs_();
  if (done)
    o = reset();
  return {o, r, done};
}
