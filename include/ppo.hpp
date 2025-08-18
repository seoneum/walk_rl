#pragma once
#include "elite_buffer.hpp"
#include "rl_config.hpp"
#include <torch/torch.h>

struct ActorCriticImpl : torch::nn::Module {
  torch::nn::Sequential pi{nullptr};
  torch::nn::Linear mu{nullptr};
  torch::Tensor log_std;
  torch::nn::Sequential vf{nullptr};
  int obs_dim = 0, act_dim = 0;
  ActorCriticImpl(int obs_dim, int act_dim, int hidden);
  std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor &x);
  std::tuple<torch::Tensor, torch::Tensor> act(const torch::Tensor &x);
};
TORCH_MODULE(ActorCritic);

struct Rollout {
  torch::Tensor obs, actions, logprobs, rewards, dones, values;
  void reserve(int T, int N, int obs_dim, int act_dim, torch::Device dev);
};

class PPOTrainer {
public:
  PPOTrainer(const PPOConfig &cfg, ActorCritic net);
  void save(const std::string &path);
  void load(const std::string &path);
  void update(Rollout &ro, EliteBuffer *elite = nullptr,
              double *out_approx_kl = nullptr);
  ActorCritic net() { return net_; }
  torch::Device device() const { return device_; }
  torch::optim::Adam &optimizer() { return optimizer_; }

private:
  PPOConfig cfg_;
  ActorCritic net_;
  torch::Device device_;
  torch::optim::Adam optimizer_;
  std::tuple<torch::Tensor, torch::Tensor> compute_gae_(Rollout &ro);
};
