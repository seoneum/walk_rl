#include "ars.hpp"
#include <algorithm>
#include <fstream>
#include <omp.h>

static torch::Tensor act_from_linear(const ARSPolicy &pol,
                                     const torch::Tensor &obs) {
  auto a = torch::matmul(pol.W, obs); // [act_dim]
  return torch::tanh(a);              // clamp to [-1,1]
}

double rollout_return(MjEnv &env, const ARSPolicy &pol, int horizon) {
  auto obs = env.reset();
  double ret = 0.0;
  for (int t = 0; t < horizon; ++t) {
    auto a = act_from_linear(pol, obs);
    auto sr = env.step(a);
    ret += sr.reward;
    obs = sr.obs;
  }
  return ret;
}

void ars_train(std::vector<std::unique_ptr<MjEnv>> &envs, ARSPolicy &pol,
               const ARSConfig &cfg, const std::string &log_csv,
               const std::string &ckpt_path) {
  std::mt19937 rng(cfg.seed);
  std::normal_distribution<double> n01(0.0, 1.0);
  std::ofstream logf(log_csv);
  logf << "iter,ret_mean\n";

  int P = cfg.population;
  std::vector<torch::Tensor> Deltas;
  Deltas.reserve(P);
  for (int i = 0; i < P; ++i)
    Deltas.push_back(torch::randn_like(pol.W));

  for (int it = 1; it <= cfg.iters; ++it) {
    // sample deltas
    for (int i = 0; i < P; ++i)
      Deltas[i].normal_(0.0, 1.0);

    std::vector<double> Rpos(P, 0.0), Rneg(P, 0.0);

// evaluate in parallel across envs
#pragma omp parallel for schedule(static) if (envs.size() > 1)
    for (int i = 0; i < P; ++i) {
      int eidx = i % envs.size();
      ARSPolicy ppos{pol.W + cfg.sigma * Deltas[i]};
      ARSPolicy pneg{pol.W - cfg.sigma * Deltas[i]};
      Rpos[i] = rollout_return(*envs[eidx], ppos, cfg.horizon);
      Rneg[i] = rollout_return(*envs[eidx], pneg, cfg.horizon);
    }

    // select topk by max(Rpos, Rneg)
    std::vector<int> idx(P);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(
        idx.begin(), idx.begin() + cfg.topk, idx.end(), [&](int a, int b) {
          return std::max(Rpos[a], Rneg[a]) > std::max(Rpos[b], Rneg[b]);
        });

    double stddev = 1e-8;
    {
      std::vector<double> all;
      all.reserve(2 * cfg.topk);
      for (int k = 0; k < cfg.topk; ++k) {
        all.push_back(Rpos[idx[k]]);
        all.push_back(Rneg[idx[k]]);
      }
      double mean = std::accumulate(all.begin(), all.end(), 0.0) / all.size();
      double var = 0.0;
      for (auto x : all)
        var += (x - mean) * (x - mean);
      var /= all.size();
      stddev = std::sqrt(var) + 1e-8;
    }

    // update
    torch::Tensor step = torch::zeros_like(pol.W);
    for (int k = 0; k < cfg.topk; ++k) {
      int i = idx[k];
      step += (Rpos[i] - Rneg[i]) * Deltas[i];
    }
    step = (cfg.alpha / (cfg.topk * stddev)) * step;
    pol.W += step;

    double mean_ret = 0.0;
    for (int i = 0; i < P; ++i)
      mean_ret += 0.5 * (Rpos[i] + Rneg[i]);
    mean_ret /= P;
    std::cout << "ARS iter " << it << " | meanR " << mean_ret << std::endl;
    logf << it << "," << mean_ret << "\n";
    logf.flush();

    if (it % 20 == 0) {
      // save as Torch tensor
      torch::save(pol.W, ckpt_path);
    }
  }
  torch::save(pol.W, ckpt_path);
}
