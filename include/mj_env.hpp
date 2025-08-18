
#pragma once
#include "gait_ref.hpp"
#include "rl_config.hpp"
#include <mujoco/mujoco.h>
#include <random> // ← 반드시 포함
#include <string>
#include <torch/torch.h>
#include <vector>

struct StepResult {
  torch::Tensor obs;
  double reward;
  bool done;
};

class MjEnv {
public:
  explicit MjEnv(const EnvConfig &cfg);
  ~MjEnv();

  torch::Tensor reset();
  StepResult step(const torch::Tensor &action);

  int obs_dim() const { return obs_dim_; }
  int act_dim() const { return act_dim_; }
  double sim_time() const { return sim_time_; }
  int step_count() const { return step_count_; }

  // 뷰어/컨트롤용
  mjModel *model() const { return m_; }
  mjData *data() const { return d_; }
  void set_target_vx(double vx) {
    cfg_.target_vx = vx;
    ref_.set_speed(vx);
  }

private:
  void build_mappings_();
  void find_feet_();
  void randomize_physics_();
  void randomize_state_();
  torch::Tensor build_obs_();
  void apply_action_position_servo_(const torch::Tensor &act);
  bool is_fallen_() const;
  double compute_reward_and_check_done_(bool &done_flag);
  void quat_to_rpy_(const double *q, double &roll, double &pitch,
                    double &yaw) const;

  // 추가한 헬퍼(선언 필수)
  void lift_body_clearance_(double margin);
  void settle_(int steps);

private:
  EnvConfig cfg_;
  mjModel *m_ = nullptr;
  mjData *d_ = nullptr;

  // RNG (오타 수정: mt1937 → mt19937)
  std::mt19937 rng_;
  std::normal_distribution<double> n01_{0.0, 1.0};
  std::uniform_real_distribution<double> uni01_{0.0, 1.0};

  std::vector<int> foot_geom_ids_;
  std::vector<int> act_jnt_ids_;
  std::vector<int> act_jnt_qposadr_;
  std::vector<int> act_jnt_qveladr_;

  int obs_dim_ = 0;
  int act_dim_ = 0;

  torch::Tensor prev_action_;
  double sim_time_ = 0.0;
  int step_count_ = 0;
  int max_steps_per_ep_ = 0;
  double last_action_rate2_ = 0.0;

  LocomotionRef ref_;
};
