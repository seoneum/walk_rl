// ppo.cpp

#include "ppo.hpp"
#include <cmath>
#include <numeric>
#include <random>

ActorCriticImpl::ActorCriticImpl(int obs_dim_, int act_dim_, int hidden)
    : obs_dim(obs_dim_), act_dim(act_dim_) {
  pi = torch::nn::Sequential(
      torch::nn::Linear(obs_dim, hidden), torch::nn::Tanh(),
      torch::nn::Linear(hidden, hidden), torch::nn::Tanh());
  mu = torch::nn::Linear(hidden, act_dim);
  vf = torch::nn::Sequential(torch::nn::Linear(obs_dim, hidden),
                             torch::nn::Tanh(),
                             torch::nn::Linear(hidden, hidden),
                             torch::nn::Tanh(), torch::nn::Linear(hidden, 1));
  log_std = torch::zeros({act_dim}, torch::kFloat32);
  register_module("pi", pi);
  register_module("mu", mu);
  register_parameter("log_std", log_std);
  register_module("vf", vf);
}

std::tuple<torch::Tensor, torch::Tensor>
ActorCriticImpl::forward(const torch::Tensor &x) {
  auto h = pi->forward(x);
  auto mean = mu->forward(h);
  auto value = vf->forward(x).squeeze(-1);
  return {mean, value};
}

std::tuple<torch::Tensor, torch::Tensor>
ActorCriticImpl::act(const torch::Tensor &x) {
  auto [mean, value] = forward(x);
  auto std = torch::exp(log_std);
  auto noise = torch::randn_like(mean);
  auto action = mean + noise * std;
  auto var = std * std;
  constexpr double LOG_2PI = 1.8378770664093453;
  auto logp =
      -0.5 * (((action - mean).pow(2) / var) + 2.0 * torch::log(std) + LOG_2PI)
                 .sum(-1);
  return std::make_tuple(action, logp);
}

void Rollout::reserve(int T, int N, int obs_dim, int act_dim,
                      torch::Device dev) {
  obs = torch::zeros({T, N, obs_dim}).to(dev);
  actions = torch::zeros({T, N, act_dim}).to(dev);
  logprobs = torch::zeros({T, N}).to(dev);
  rewards = torch::zeros({T, N}).to(dev);
  dones = torch::zeros({T, N}).to(dev);
  values = torch::zeros({T, N}).to(dev);
}

PPOTrainer::PPOTrainer(const PPOConfig &cfg, ActorCritic net)
    : cfg_(cfg), net_(net),
      device_(cfg.use_cuda && torch::cuda::is_available() ? torch::kCUDA
                                                          : torch::kCPU),
      optimizer_(net_->parameters(), torch::optim::AdamOptions(cfg.lr)) {
  net_->to(device_);
}

void PPOTrainer::save(const std::string &path) { torch::save(net_, path); }
void PPOTrainer::load(const std::string &path) {
  torch::load(net_, path, device_);
}

std::tuple<torch::Tensor, torch::Tensor> PPOTrainer::compute_gae_(Rollout &ro) {
  // ro.rewards: [T, N], ro.values: [T+1, N], ro.dones: [T+1, N]
  const int T = ro.rewards.size(0);
  const int N = ro.rewards.size(1);

  TORCH_CHECK(ro.values.size(0) == T + 1, "ro.values must be [T+1, N]");
  TORCH_CHECK(ro.dones.size(0) == T + 1, "ro.dones must be [T+1, N]");
  TORCH_CHECK(ro.values.size(1) == N && ro.dones.size(1) == N,
              "values/dones N mismatch");

  auto opts = ro.rewards.options().device(device_);
  auto adv = torch::zeros({T, N}, opts);
  auto ret = torch::zeros({T, N}, opts);
  auto gae = torch::zeros({N}, opts);

  for (int t = T - 1; t >= 0; --t) {
    auto nonterm = 1.0 - ro.dones[t + 1];
    auto delta =
        ro.rewards[t] + cfg_.gamma * ro.values[t + 1] * nonterm - ro.values[t];
    gae = delta + cfg_.gamma * cfg_.gae_lambda * nonterm * gae;
    adv[t] = gae;
    ret[t] = gae + ro.values[t];
  }

  auto mean = adv.mean();
  auto std = adv.std().clamp_min(1e-6);
  adv = (adv - mean) / std;

  return {ret.detach(), adv.detach()};
}
// in src/ppo.cpp

void PPOTrainer::update(Rollout &ro, EliteBuffer *elite,
                        double *out_approx_kl) {
  auto [returns, advantages] = compute_gae_(ro);
  const int T = ro.rewards.size(0), N = ro.rewards.size(1), B = T * N;

  auto obs = ro.obs.view({B, -1}).to(device_);
  auto actions = ro.actions.view({B, -1}).to(device_);
  auto old_logp = ro.logprobs.view({B}).to(device_).detach();
  auto adv = advantages.view({B}).to(device_);
  auto ret = returns.view({B}).to(device_);

  constexpr double LOG_2PI = 1.8378770664093453;
  int mb = std::min(cfg_.minibatch_size, B);
  double approx_kl_running = 0.0;
  int approx_kl_count = 0;

  for (int epoch = 0; epoch < cfg_.num_epochs; ++epoch) {
    auto perm = torch::randperm(
        B, torch::TensorOptions().dtype(torch::kLong).device(device_));
    bool stop_epoch = false;

    for (int start = 0; start < B; start += mb) {
      int end = std::min(B, start + mb);
      auto inds = perm.slice(0, start, end).contiguous();

      auto b_obs = obs.index_select(0, inds);
      auto b_actions = actions.index_select(0, inds);
      auto b_oldlogp = old_logp.index_select(0, inds);
      auto b_adv = adv.index_select(0, inds);
      auto b_ret = ret.index_select(0, inds);

      auto [mean, value] = net_->forward(b_obs);
      auto std = torch::exp(net_->log_std);
      auto var = std * std;

      auto logp = -0.5 * (((b_actions - mean).pow(2) / var) +
                          2.0 * torch::log(std) + LOG_2PI)
                             .sum(-1);
      auto ratio = (logp - b_oldlogp).exp();

      auto approx_kl = (b_oldlogp - logp).mean();
      approx_kl_running += approx_kl.item<double>();
      approx_kl_count++;

      auto surr1 = ratio * b_adv;
      auto surr2 =
          torch::clamp(ratio, 1.0 - cfg_.clip_range, 1.0 + cfg_.clip_range) *
          b_adv;
      auto policy_loss = -torch::min(surr1, surr2).mean();
      auto value_loss = 0.5 * (b_ret - value).pow(2).mean();
      auto entropy =
          0.5 * (2.0 * torch::log(std) + LOG_2PI + 1.0).sum(-1).mean();

      torch::Tensor bc_loss = torch::tensor(0.0, obs.options());
      if (elite && cfg_.bc_coef > 0.0 && elite->size() > 0) {
        auto sample = elite->sample(cfg_.bc_batch);
        if (sample.first.defined()) {
          auto o = sample.first;
          auto a = sample.second;
          auto mean_bc = std::get<0>(net_->forward(o));
          bc_loss = 0.5 * (mean_bc - a).pow(2).mean();
        }
      }

      auto loss = policy_loss + cfg_.vf_coef * value_loss -
                  cfg_.ent_coef * entropy + cfg_.bc_coef * bc_loss;

      optimizer_.zero_grad();
      loss.backward();
      torch::nn::utils::clip_grad_norm_(net_->parameters(), 1.0);
      optimizer_.step();

      if (cfg_.target_kl > 0.0 && approx_kl.item<double>() > cfg_.target_kl) {
        stop_epoch = true;
        break;
      }
    }
    if (stop_epoch)
      break;
  }
  net_->log_std.data().clamp_(-2.0, 0.5);

  if (out_approx_kl && approx_kl_count > 0)
    *out_approx_kl = approx_kl_running / approx_kl_count;
}
