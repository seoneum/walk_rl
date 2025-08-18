#include "ars.hpp"
#include "elite_buffer.hpp"
#include "mj_env.hpp"
#include "ppo.hpp"
#include "rl_config.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <tuple>
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  try {
    AppConfig cfg;
    std::string yaml_path = (argc > 1) ? argv[1] : "";
    if (!yaml_path.empty()) {
      if (!load_config_yaml(yaml_path, cfg)) {
        std::cerr << "Failed to load config YAML. Using defaults.\n";
      }
    }

    // Envs
    std::vector<std::unique_ptr<MjEnv>> envs;
    envs.reserve(cfg.train.num_envs);
    for (int i = 0; i < cfg.train.num_envs; ++i) {
      EnvConfig e = cfg.env;
      e.seed += i * 101;
      envs.emplace_back(std::make_unique<MjEnv>(e));
    }
    int obs_dim = envs[0]->obs_dim();
    int act_dim = envs[0]->act_dim();

    // Run dir
    auto run_dir = make_run_dir("runs", cfg.train.tag);
    cfg.paths.run_dir = run_dir;
    cfg.paths.log_csv = run_dir + "/train.csv";
    cfg.paths.config_yaml_dump = run_dir + "/config_dump.yaml";
    dump_config_yaml(cfg, cfg.paths.config_yaml_dump);

    std::ofstream logf(cfg.paths.log_csv);
    logf << "update,avgR,doneEps,fps,KL,elite\n";

    // ARS 옵션
    if (cfg.train.algo == "ARS") {
      ARSConfig ac;
      ac.obs_dim = obs_dim;
      ac.act_dim = act_dim;
      ac.population = 32;
      ac.topk = 16;
      ac.horizon = 600;
      ac.iters = 300;
      ARSPolicy pol{torch::zeros({act_dim, obs_dim}, torch::kFloat32)};
      std::string ckpt = run_dir + "/ars_W.pt";
      ars_train(envs, pol, ac, cfg.paths.log_csv, ckpt);
      std::cout << "Saved ARS policy to " << ckpt << std::endl;
      return 0;
    }

    // PPO 설정: cfg.ppo를 그대로 사용
    PPOConfig pc = cfg.ppo;
    pc.obs_dim = obs_dim;
    pc.act_dim = act_dim;
    auto net = ActorCritic(obs_dim, act_dim, pc.hidden);
    PPOTrainer ppo(pc, net);

    if (!cfg.ppo.init_ckpt.empty()) {
      try {
        ppo.load(cfg.ppo.init_ckpt);
        std::cout << "Loaded init_ckpt: " << cfg.ppo.init_ckpt << std::endl;
      } catch (const std::exception &e) {
        std::cerr << "WARN: failed to load init_ckpt: " << e.what()
                  << std::endl;
      }
    }

    // 초기 reset
    std::vector<torch::Tensor> obs_list(envs.size());
    for (size_t i = 0; i < envs.size(); ++i)
      obs_list[i] = envs[i]->reset();

    auto device = ppo.device();
    const int T = pc.rollout_horizon;

    // 엘리트 버퍼 (스텝 상위 20%)
    EliteBuffer elite(100000, obs_dim, act_dim, device);

    double bestR = -1e9;

    for (int update = 1; update <= cfg.train.total_updates; ++update) {
      Rollout ro;
      ro.reserve(T, (int)envs.size(), obs_dim, act_dim, device);

      double ep_rew_sum = 0.0;
      int ep_done_count = 0;
      auto t0 = std::chrono::steady_clock::now();

      // 1) Rollout T steps
      for (int t = 0; t < T; ++t) {
        auto obs_batch = torch::stack(obs_list).to(device);
        torch::Tensor action, logp, values;
        {
          torch::NoGradGuard no_grad;
          std::tie(action, logp) = ppo.net()->act(obs_batch);
          values = std::get<1>(ppo.net()->forward(obs_batch));
        }

        // env step
        std::vector<torch::Tensor> next_obs_list(envs.size());
        std::vector<float> rewards(envs.size());
        std::vector<float> dones(envs.size());
        auto act_cpu = action.to(torch::kCPU);
#pragma omp parallel for schedule(static) if (envs.size() > 1)
        for (int i = 0; i < (int)envs.size(); ++i) {
          auto sr = envs[i]->step(act_cpu[i]);
          next_obs_list[i] = sr.obs;
          rewards[i] = (float)sr.reward;
          dones[i] = sr.done ? 1.0f : 0.0f;
        }
        obs_list.swap(next_obs_list);

        // 통계
        for (size_t i = 0; i < envs.size(); ++i) {
          ep_rew_sum += rewards[i];
          if (dones[i] > 0.5f)
            ep_done_count++;
        }

        // rollout 버퍼 (detach)
        ro.obs[t] = obs_batch;
        ro.actions[t] = action.detach();
        ro.logprobs[t] = logp.detach();
        ro.rewards[t] = torch::from_blob(rewards.data(), {(long)envs.size()})
                            .to(device)
                            .clone();
        ro.dones[t] = torch::from_blob(dones.data(), {(long)envs.size()})
                          .to(device)
                          .clone();
        ro.values[t] = values.detach();
      }

      // 2) 엘리트 버퍼(상위 20% 스텝) 축적
      {
        int B = T * (int)envs.size();
        auto rew_flat_cpu = ro.rewards.view({B}).to(torch::kCPU);
        int topK = std::max(1, (int)(0.2 * B));
        auto sorted_idx = std::get<1>(rew_flat_cpu.sort()); // ascending
        int thr_index = B - topK;
        float thr = rew_flat_cpu[sorted_idx[thr_index]].item<float>();

        auto obs_flat = ro.obs.view({B, obs_dim}).to(device);
        auto act_flat = ro.actions.view({B, act_dim}).to(device);
        auto rew_dev = ro.rewards.view({B}).to(device);
        auto mask = (rew_dev >= thr);
        auto idx = mask.nonzero().squeeze(-1);
        if (idx.numel() > 0) {
          elite.add_batch(obs_flat.index_select(0, idx),
                          act_flat.index_select(0, idx),
                          rew_dev.index_select(0, idx));
        }
      }

      // 3) Bootstrap 1 step (values/dones만 T+1로)
      {
        torch::NoGradGuard no_grad;
        auto obs_batch = torch::stack(obs_list).to(device);
        auto v_last = std::get<1>(ppo.net()->forward(obs_batch)).detach();
        ro.values = torch::cat({ro.values, v_last.unsqueeze(0)}, 0);
        ro.dones = torch::cat(
            {ro.dones, torch::zeros_like(ro.dones[0]).unsqueeze(0)}, 0);
      }

      // 4) 학습률 디케이(선택)
      {
        double frac = 1.0 - (double)update / (double)cfg.train.total_updates;
        double new_lr = pc.lr * std::max(0.0, frac);
        for (auto &pg : ppo.optimizer().param_groups())
          static_cast<torch::optim::AdamOptions &>(pg.options()).lr(new_lr);
      }

      // 5) PPO update (KL/BC/Elite 통합)
      double approx_kl = 0.0;
      ppo.update(ro, (pc.bc_coef > 0.0 ? &elite : nullptr), &approx_kl);

      // 6) 로그/체크포인트/커리큘럼
      auto t1 = std::chrono::steady_clock::now();
      double dt = std::chrono::duration<double>(t1 - t0).count();
      double fps = (double)(T * envs.size()) / dt;
      double avgR = ep_rew_sum / (T * envs.size());

      std::cout << "Update " << update << " | avgR " << avgR << " | done "
                << ep_done_count << " | fps " << fps << " | KL " << approx_kl
                << " | elite " << elite.size() << std::endl;
      logf << update << "," << avgR << "," << ep_done_count << "," << fps << ","
           << approx_kl << "," << elite.size() << "\n";
      logf.flush();

      if (avgR > bestR) {
        bestR = avgR;
        std::string best = run_dir + "/best.pt";
        ppo.save(best);
      }
      if (update % 50 == 0) {
        std::string ckpt = run_dir + "/ckpt_" + std::to_string(update) + ".pt";
        ppo.save(ckpt);
        std::cout << "Saved " << ckpt << std::endl;
      }
      if (cfg.train.curriculum && update % cfg.train.curriculum_every == 0) {
        cfg.env.target_vx =
            std::min(cfg.train.curriculum_vx_max,
                     cfg.env.target_vx + cfg.train.curriculum_dvx);
        for (auto &e : envs)
          e->set_target_vx(cfg.env.target_vx);
      }
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }
}
