#pragma once
#include "mj_env.hpp"
#include <random>
#include <torch/torch.h>
#include <vector>

struct ARSConfig {
  int obs_dim = 0, act_dim = 0;
  int population = 32;
  int topk = 16;
  double sigma = 0.02;
  double alpha = 0.02;
  int horizon = 1000; // steps per rollout
  int iters = 300;
  unsigned seed = 123;
};

struct ARSPolicy {
  torch::Tensor W; // [act_dim, obs_dim]
};

double rollout_return(MjEnv &env, const ARSPolicy &pol, int horizon);
void ars_train(std::vector<std::unique_ptr<MjEnv>> &envs, ARSPolicy &pol,
               const ARSConfig &cfg, const std::string &log_csv,
               const std::string &ckpt_path);
